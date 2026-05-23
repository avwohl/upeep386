"""Peephole optimizer for uc386 NASM assembly output.

Applies pattern-based and structural rewrites on the generated asm text
just before it's written. Runs to fixed point.

This is the inline phase of the optimizer. Once the pattern set has
crystallized, this module will be extracted to a separate `upeep386`
package mirroring the upeepz80/uc80 split.

Currently implements:
  - dead_after_terminator: drop instrs between an unconditional `jmp`/
    `ret` and the next label/directive/data line.
  - jmp_to_next_label: drop `jmp X` immediately followed by `X:`.
  - binop_collapse: replace the 4-line stack-machine right-operand
    transfer (`push eax; mov eax, src; mov ecx, eax; pop eax`) with
    `mov ecx, src` when src is a single-instruction load.
  - store_collapse: drop the push/pop pair around a store (`push eax;
    mov eax, src; pop ecx; mov [ecx], eax`) when src is a
    single-instruction load that doesn't read ECX.
  - leave_collapse: replace the function epilogue's `mov esp, ebp;
    pop ebp` with the equivalent single-byte `leave`. Saves 1 instr
    + 2 bytes per function epilogue.
  - imm_store_collapse: replace `mov eax, IMM; mov [addr], eax;
    mov eax, X` with `mov dword [addr], IMM; mov eax, X`. The
    immediate-store form is one NASM instruction; the EAX overwrite
    in the third line confirms EAX was dead.
  - setcc_jcc_collapse: replace the `setCC al; movzx eax, al; test
    eax, eax; jz/jnz LBL` boolean-materialize-then-branch sequence
    with a single conditional jump using the (possibly inverted) CC.
    Saves 3 instructions per `if`/`while`/`for` condition.
  - push_immediate: replace `mov eax, IMM_OR_LABEL; push eax` with
    `push IMM_OR_LABEL` when the next instruction overwrites EAX
    (the mov was just for the push). Common at every cdecl arg site.
  - ecx_binop_collapse: replace `mov ecx, <src>; OP eax, ecx` with
    `OP eax, <src>` for OP ∈ {add, sub, and, or, xor, cmp, test,
    imul, adc, sbb}. Saves the bytes of the prior `mov ecx, <src>`
    (5 for imm32, 3-7 for memory, 2 for register). Source can be
    any addressing mode the OP supports as a memory or immediate
    source operand. Witness: next instruction overwrites ECX (full
    mov / pop / xor / lea / call) or doesn't reference ECX/CX/CL/CH.
  - mov_zero_to_xor: replace `mov eax, 0` with `xor eax, eax`. Saves
    3 bytes (5-byte mov-imm32 → 2-byte xor reg, reg). Conservative:
    only fires when a forward scan finds a flag-clobbering instruction
    or a `ret` before any flag-reading instruction. Same pattern for
    other gp registers (ECX, EDX, EBX, ESI, EDI, EBP).
  - store_load_collapse: drop the redundant load after a store of
    the same register to the same address: `mov [X], R; mov R, [X]`
    → just the store. The register still holds its stored value, so
    the load is unnecessary. Saves the bytes of the load instruction
    (3 for [ebp-N], 6 for [_glob]).
  - right_operand_retarget: collapse the RHS-chain-then-copy pattern
    `push eax; <chain of EAX-writing ops>; mov ecx, eax; pop eax`
    by retargeting every instruction in the chain to write ECX
    instead. The `mov ecx, eax` becomes redundant and is dropped.
    Saves 2 bytes per binop with a non-trivial right-hand side
    (e.g. `a[i] cmp b[i]` where each side dereferences an array).
  - cmp_zero_to_test: replace `cmp reg, 0` with `test reg, reg`.
    Saves 1 byte per match (3-byte cmp-imm8-sign-ext → 2-byte test).
    Identical flag effects for ZF/SF/OF/CF/PF.
  - dead_mov_to_reg: drop `mov <reg>, <src>` when forward scan finds
    a full reg overwrite before any read. Common in postfix `i++`
    in for-loop steps where the value is discarded. Crosses labels
    and unconditional jumps under the codegen invariant (regs are
    always overwritten at the start of each labeled block).
  - prologue_to_enter: `push ebp; mov ebp, esp; sub esp, IMM`
    becomes `enter IMM, 0`. Saves 2-5 bytes per function (depending
    on whether IMM fits in imm8). Identical semantics — enter
    pushes EBP, sets EBP=ESP, allocates IMM stack bytes.
  - redundant_eax_load: drop `mov eax, M` when EAX already provably
    holds M's value (we just loaded it; intervening instructions
    don't write EAX or alias-modify M). Common in `int v = x; v * v`
    where the second `mov eax, [ebp+8]` is redundant.
  - label_offset_fold: collapse `mov reg, LABEL; add reg, IMM` (or
    `sub reg, IMM`) into `mov reg, LABEL ± IMM`. NASM resolves the
    `LABEL ± IMM` at assemble time, producing a single 5-byte
    `mov reg, imm32` and dropping the 3-byte add/sub. Saves 3 bytes
    per match. Common in pointer arithmetic on globals
    (`mov eax, _g; add eax, 8` for `g[2]`).
  - cmp_load_collapse: collapse `mov reg, [mem]; cmp reg, X` into
    `cmp dword [mem], X` when reg is dead after. Saves 2-3 bytes
    per match. Common in `if (var == X)` and `if (var)` (the
    latter via `cmp_zero_to_test`'s output).
  - rmw_collapse: collapse `mov reg, [mem]; OP reg, IMM; mov [mem],
    reg` into `OP dword [mem], IMM` when reg is dead after. OP is
    one of add/sub/and/or/xor. Saves 5 bytes per match. Common in
    compound assignments to memory operands (`x += 5`).
  - fst_fstp_collapse: collapse `fst <addr>; fstp st0` into a single
    `fstp <addr>`. Saves 2 bytes per match. Common in FPU-heavy
    code — uc386 codegen lowers `*p = expr` for float/double via
    fst+fstp-st0, which equals fstp-to-mem.
  - fpu_op_collapse: collapse `fld <addr>; foppc st1, st0` into a
    single memory-form FPU op `fop <addr>`. Same FPU stack/memory
    state, 2 bytes saved per match. Applies to faddp / fsubp /
    fmulp / fdivp (the fsubrp/fdivrp swapped variants get their own
    rewrites).
  - add_one_to_inc: `add reg, 1` → `inc reg` (and `sub reg, 1` →
    `dec reg`). Saves 2 bytes per register match, 1 byte per memory
    match (`add dword [mem], 1` → `inc dword [mem]`, etc., for byte/
    word/dword forms). inc/dec leave CF unchanged while add/sub set
    CF; the rewrite is safe only when CF is dead after (no
    `jc/jnc/ja/jb/...` reads it before being overwritten).
  - redundant_test_collapse: drop `test reg, reg` immediately after
    a flag-setting arithmetic op on the same reg. Common after
    `and eax, MASK; test eax, eax; jcc` — the AND already set ZF/SF
    based on its result, so the test is redundant. Saves 2 bytes.
  - push_memory: collapse `mov reg, [mem]; push reg` into a single
    `push dword [mem]`. NASM's `push r/m32` form is 3 bytes for ebp-
    relative addressing; the original mov+push is 4 bytes. Saves 1
    byte per match. Common in cdecl call-site arg setup, where the
    intermediate register is dead after the push.
  - disp_load_collapse: collapse `add REG, DISP; mov DST, [REG]`
    into `mov DST, [REG + DISP]` using x86's disp32 addressing.
    Saves 1-2 bytes per match. Common in `p->member` struct member
    access for non-zero offsets, where the codegen emits explicit
    `add reg, offset` before dereference.
  - index_load_collapse: collapse the array-index address-
    computation chain `shl IDX, N; add BASE, IDX; mov DST, [BASE]`
    into a single `mov DST, [BASE + IDX*SCALE]` using x86's SIB
    byte. Saves 4 bytes per match (shl + add gone, mov gains 1
    byte for the SIB byte). N ∈ {1, 2, 3} for scale 2/4/8. Common
    in `arr[i]` array indexing for int arrays (scale 4) and
    pointer arrays (scale 4).
  - compound_assign_collapse: collapse the codegen's compound-
    assignment frame `push dword [m]; <chain>; mov ecx, eax;
    pop eax; OP eax, ecx; mov [m], eax` into a single in-place
    `OP [m], eax`. The chain must not modify [m] or perform stack
    manipulation, and ECX must be dead after the store. Saves 8
    bytes per match (drops the push/pop framing + transfer + store).
    Common in `x += rhs` / `x -= rhs` etc. for slot-typed lvalues
    where the rhs computation can't be reduced via simpler passes.
  - redundant_xor_zero: drop `xor reg, reg` when reg is already
    zero from a recent `xor reg, reg` and nothing modified it
    since. Saves 2 bytes per match. Common after zero_init_collapse
    fires twice in a function prologue, leaving back-to-back
    `xor eax, eax` instructions.
  - zero_init_collapse: replace 1+ adjacent `mov <size> [m], 0`
    stores with `xor eax, eax` + per-store `mov [m], <eax/ax/al>`.
    Saves 2 bytes per dword-zero store (7 bytes → 3 bytes per store
    plus a single 2-byte xor) and 1 byte per byte-zero store. Common
    in function prologues with multiple `int x = 0;` initializers.
  - jcc_jmp_inversion: collapse `jcc L1; jmp L2; L1:` into
    `j!cc L2; L1:` (with the cc inverted). Saves 5 bytes per match —
    drops the unconditional jmp. Common in `if-else` and ternary
    lowering where one branch is just a fallthrough to L2 after
    redundant_eax_load eliminates the only instruction.
  - narrowing_load_test_collapse: collapse
    `movsx/movzx eax, byte [SRC]; test eax, eax` into
    `cmp byte [SRC], 0` (and likewise for word). The narrowing load
    only sets EAX; the test then checks for zero. Direct `cmp <size>
    [SRC], 0` produces the same flags (a zero-extended/sign-extended
    byte/word is zero iff the byte/word itself is zero). Saves 2
    bytes per match. Common in string/byte loops:
    `while (*p) p++;` → `... cmp byte [eax], 0; jz end ...`.

Long-tail patterns not yet implemented: tail calls, additional
jump threading, multi-instruction right-operand retargeting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Instruction line: leading whitespace, mnemonic, operands.
# Examples:
#   "        jmp     .epilogue"
#   "        mov     eax, [ebp - 4]"
#   "        ret"
_INSTR_RE = re.compile(r"^(\s+)([a-zA-Z][a-zA-Z0-9]*)(?:\s+(.*?))?\s*$")

# Label line: identifier optionally prefixed by `.`, ending in `:`.
# Examples: `_main:`, `.L1_for_top:`, `__start:`
_LABEL_RE = re.compile(r"^(\.?[A-Za-z_][A-Za-z0-9_]*)\s*:\s*$")

# `[ebp + N]` / `[ebp - N]` reference (literal-offset form).
_EBP_OFFSET_RE = re.compile(r"\[\s*ebp\s*([+-])\s*(\d+)\s*\]")

# Top-level directives we treat as fences (end any dead-zone).
# `section .text` / `section .data` / `section .bss` / `bits 32` /
# `global _name` / `extern _name`.
_FENCE_PREFIXES = ("section", "bits", "global", "extern")


@dataclass
class Line:
    """One line of the asm output, classified."""
    raw: str
    kind: str  # "instr" | "label" | "directive" | "data" | "comment" | "blank"
    op: str = ""        # for instr: mnemonic (lowercased)
    operands: str = ""  # for instr: everything after mnemonic, raw
    label: str = ""     # for label: the name


def _classify(raw: str) -> Line:
    stripped = raw.strip()
    if not stripped:
        return Line(raw=raw, kind="blank")
    if stripped.startswith(";"):
        return Line(raw=raw, kind="comment")
    # Label
    m = _LABEL_RE.match(raw.lstrip())
    if m and not raw.lstrip().startswith(("section", "bits", "global", "extern")):
        # Make sure the colon is the WHOLE thing — not e.g. `mov eax, [foo]`
        # which doesn't match _LABEL_RE anyway.
        return Line(raw=raw, kind="label", label=m.group(1))
    # Indented mnemonic line
    m = _INSTR_RE.match(raw)
    if m:
        op = m.group(2).lower()
        operands = (m.group(3) or "").strip()
        # Strip trailing `; comment` from operands. The instr regex
        # captures everything after the mnemonic into group 3, so a
        # line like `push eax  ; save size` ends up with operands =
        # `eax  ; save size`. Downstream passes that test
        # `operands.strip().lower() == "eax"` (or call
        # `_is_general_register(operands)`) miss when the comment is
        # there — leading e.g. `_pass_push_pop_to_mov` to treat
        # `push eax` as a non-register push and rewrite it to
        # `mov eax, eax`, dropping the round-trip across an int.
        # Comments don't appear inside any of our codegen operand
        # strings (we never emit `mov eax, [foo;bar]`), so the
        # split is safe.
        semi_idx = operands.find(";")
        if semi_idx >= 0:
            operands = operands[:semi_idx].rstrip()
        # NASM directives like `db`, `dw`, `dd`, `dq`, `times`, `resb`,
        # `resw`, `resd`, `resq`, `equ` — these live in .data/.bss and
        # aren't real instructions. Keep them as "data" so dead-after-jmp
        # leaves them alone.
        if op in {"db", "dw", "dd", "dq", "dt", "times",
                  "resb", "resw", "resd", "resq", "equ"}:
            return Line(raw=raw, kind="data", op=op, operands=operands)
        # Section/bits/global/extern at indented positions
        if op in _FENCE_PREFIXES:
            return Line(raw=raw, kind="directive", op=op, operands=operands)
        return Line(raw=raw, kind="instr", op=op, operands=operands)
    # Unrecognized — keep verbatim. Common case: `_label equ value`,
    # or weird user-supplied lines.
    return Line(raw=raw, kind="directive")


_NORETURN_CALL_TARGETS: frozenset[str] = frozenset({
    # C standard library — guaranteed noreturn.
    "_abort", "_exit", "__Exit",
    # C standard setjmp/longjmp — noreturn.
    "_longjmp", "_siglongjmp",
    # GCC builtins — wrap the standard funcs.
    "___builtin_abort", "___builtin_exit",
    "___builtin_unreachable", "___builtin_trap",
    "___builtin_longjmp",
    # POSIX assertion failure handlers — call abort.
    "___assert_fail", "___assert", "___assert_perror_fail",
})


def _is_unconditional_terminator(line: Line) -> bool:
    """An instruction whose execution never falls through to the next line."""
    if line.kind != "instr":
        return False
    # `jmp <anything>` (direct or indirect)
    if line.op == "jmp":
        return True
    # `ret` alone (no condition codes on x86 ret).
    if line.op == "ret":
        return True
    # `iret`, `retf`, `retn` — uncommon in our codegen but be safe.
    if line.op in {"iret", "iretd", "retf", "retn"}:
        return True
    # `call <noreturn fn>` — control never returns.
    if line.op == "call":
        target = (line.operands or "").strip()
        if target in _NORETURN_CALL_TARGETS:
            return True
    return False


def _operands_split(operands: str) -> tuple[str, str] | None:
    """Split a two-operand `mov` into (dest, src). Returns None for
    anything that isn't a clean `dst, src` pair."""
    if "," not in operands:
        return None
    # NASM allows commas inside `[...]` (`[ebp - 4]` doesn't have one,
    # but `[ebp + ecx*4]` doesn't either; `[base + index, scale]`
    # syntax doesn't exist in NASM). Effective addresses use `+` / `-`
    # / `*`, never bare commas, so a top-level split on `,` is safe.
    dest, _, src = operands.partition(",")
    return dest.strip(), src.strip()


def _references_register(text: str, reg: str) -> bool:
    """Does `text` mention register `reg` (case-insensitive)? Looks for
    word-boundary matches so `ecx` doesn't match `cx` or `recx`."""
    return bool(re.search(rf"\b{reg}\b", text, re.IGNORECASE))


def _is_simple_eax_load(line: Line) -> bool:
    """Is this instruction a single `mov eax, <src>` where src is a
    valid mov source operand (immediate, memory ref, register, label)?
    The pattern needs this to know it can rewrite EAX→ECX safely."""
    if line.kind != "instr" or line.op != "mov":
        return False
    parts = _operands_split(line.operands)
    if parts is None:
        return False
    dest, _ = parts
    return dest.lower() == "eax"


def _is_imm_or_label_to_eax(line: Line) -> bool:
    """Is this `mov eax, <constant-expr>` — immediate or label-address,
    NOT a memory deref or register source? Used by both
    imm_store_collapse and push_immediate."""
    if line.kind != "instr" or line.op != "mov":
        return False
    parts = _operands_split(line.operands)
    if parts is None:
        return False
    dest, src = parts
    if dest.lower() != "eax":
        return False
    # Reject memory derefs.
    if "[" in src:
        return False
    # Reject bare register names.
    regs = {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
            "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
            "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh"}
    if src.lower() in regs:
        return False
    return True


def _retarget_eax_to_ecx(line: Line) -> Line:
    """Rewrite `mov eax, src` to `mov ecx, src`. Caller verified the
    line shape via `_is_simple_eax_load`."""
    parts = _operands_split(line.operands)
    assert parts is not None
    _, src = parts
    new_raw = re.sub(r"\beax\b", "ecx", line.raw, count=1)
    new_operands = f"ecx, {src}"
    return Line(raw=new_raw, kind="instr", op="mov", operands=new_operands)


def _jmp_target(line: Line) -> str | None:
    """Return the target label of a direct `jmp <label>`, or None for
    indirect / unrecognized."""
    if line.kind != "instr" or line.op != "jmp":
        return None
    return _branch_target(line)


def _branch_target(line: Line) -> str | None:
    """Return the target label of any direct branch (jmp/jcc) — i.e.
    the operand is a literal label name. Returns None for indirect
    branches (`jmp eax`, `jmp [_tbl]`) or unparseable operands.
    """
    if line.kind != "instr":
        return None
    operand = line.operands.strip()
    if not operand or operand[0] in "[":
        return None
    if operand.lower() in {"eax", "ebx", "ecx", "edx", "esi", "edi",
                           "ebp", "esp"}:
        return None
    if re.fullmatch(r"\.?[A-Za-z_][A-Za-z0-9_]*", operand):
        return operand
    return None


class PeepholeOptimizer:
    """Multi-pass peephole optimizer.

    Usage:
        opt = PeepholeOptimizer()
        new_asm = opt.optimize(asm_text)
        for name, count in opt.stats.items(): ...
    """

    MAX_PASSES = 10

    def __init__(self) -> None:
        self.stats: dict[str, int] = {}

    def optimize(self, asm_text: str) -> str:
        # Preserve trailing newline behavior.
        trailing_nl = asm_text.endswith("\n")
        raw_lines = asm_text.splitlines()
        lines = [_classify(r) for r in raw_lines]

        for _ in range(self.MAX_PASSES):
            before = len(lines)
            lines = self._pass_dead_after_terminator(lines)
            lines = self._pass_jmp_to_next_label(lines)
            lines = self._pass_binop_collapse(lines)
            lines = self._pass_store_collapse(lines)
            lines = self._pass_leave_collapse(lines)
            lines = self._pass_imm_store_collapse(lines)
            lines = self._pass_imm_store_then_imm_load_collapse(lines)
            lines = self._pass_setcc_jcc_collapse(lines)
            lines = self._pass_push_immediate(lines)
            lines = self._pass_imm_binop_collapse(lines)
            lines = self._pass_mov_zero_to_xor(lines)
            lines = self._pass_mov_neg_one_to_or(lines)
            lines = self._pass_store_load_collapse(lines)
            lines = self._pass_right_operand_retarget(lines)
            lines = self._pass_store_chain_retarget(lines)
            # push_memory runs AFTER right_operand_retarget — that pass
            # consumes inner `push eax / chain / mov ecx,eax / pop eax`
            # save-restore frames, eliminating the inner mov+push
            # entirely (saves more than push_memory's 1-byte rewrite).
            # Run push_memory afterwards so it picks up only the
            # remaining mov+push pairs (outer save-pushes that don't
            # have a matching mov-ecx-eax pop pattern).
            lines = self._pass_push_memory(lines)
            lines = self._pass_push_memory_to_push_reg(lines)
            lines = self._pass_cmp_zero_to_test(lines)
            lines = self._pass_dead_mov_to_reg(lines)
            lines = self._pass_prologue_to_enter(lines)
            lines = self._pass_label_offset_fold(lines)
            # cmp_load_collapse runs BEFORE redundant_eax_load: the
            # smarter redundant_eax_load (with jcc-target tracking)
            # can drop a mov that cmp_load_collapse would've fused
            # with its cmp. Running cmp_load_collapse first lets both
            # passes fire on independent patterns.
            lines = self._pass_cmp_load_collapse(lines)
            lines = self._pass_redundant_eax_load(lines)
            lines = self._pass_redundant_ecx_load(lines)
            lines = self._pass_rmw_collapse(lines)
            lines = self._pass_rmw_intermediate_collapse(lines)
            lines = self._pass_rmw_mem_src_collapse(lines)
            lines = self._pass_div_mem_form(lines)
            lines = self._pass_mov_test_setcc_movzx_collapse(lines)
            lines = self._pass_shift_const_imm(lines)
            lines = self._pass_same_memory_operand_reuse(lines)
            lines = self._pass_chain_binop_collapse(lines)
            lines = self._pass_fst_fstp_collapse(lines)
            lines = self._pass_fpu_op_collapse(lines)
            lines = self._pass_add_one_to_inc(lines)
            lines = self._pass_redundant_test_collapse(lines)
            lines = self._pass_narrowing_load_test_collapse(lines)
            lines = self._pass_jcc_jmp_inversion(lines)
            lines = self._pass_bool_materialize_collapse(lines)
            lines = self._pass_bool_materialize_collapse_form_b(lines)
            lines = self._pass_zero_init_collapse(lines)
            lines = self._pass_same_imm_store_share_reg(lines)
            lines = self._pass_redundant_xor_zero(lines)
            lines = self._pass_dual_zero_init_consolidate(lines)
            lines = self._pass_narrow_store_reload_collapse(lines)
            lines = self._pass_add_esp_to_pop(lines)
            lines = self._pass_compound_assign_collapse(lines)
            lines = self._pass_index_load_collapse(lines)
            lines = self._pass_index_load_collapse_label(lines)
            lines = self._pass_mov_label_shl_add_load_to_sib(lines)
            # Runs AFTER mov_label_shl_add_load_to_sib so that pass
            # can consume the 4-line `mov BASE, LABEL; shl IDX, N;
            # add IDX, BASE; load` pattern entirely. My pass picks
            # up the remaining cases where the shift between mov
            # and add is non-power-of-2 (imul) or otherwise can't
            # fold into SIB.
            lines = self._pass_chain_label_to_add_operand(lines)
            # Runs AFTER index_load_collapse_label so that pass can
            # consume the `shl + add LABEL + load` pattern entirely
            # (saves more bytes). My pass picks up the remaining cases
            # where the shl + add LABEL isn't followed by a load.
            lines = self._pass_shl_add_label_to_lea(lines)
            lines = self._pass_disp_load_collapse(lines)
            lines = self._pass_disp_store_collapse(lines)
            lines = self._pass_push_disp_collapse(lines)
            lines = self._pass_index_store_xfer_collapse(lines)
            lines = self._pass_push_index_collapse(lines)
            lines = self._pass_self_mov_elimination(lines)
            lines = self._pass_indirect_call_collapse(lines)
            lines = self._pass_cmp_load_promote(lines)
            lines = self._pass_dead_cleanup_before_leave(lines)
            lines = self._pass_transfer_pop_collapse(lines)
            lines = self._pass_transfer_pop_cmp_collapse(lines)
            lines = self._pass_dup_push_pop_self_op(lines)
            lines = self._pass_push_pop_op_to_memop(lines)
            lines = self._pass_push_pop_register_op(lines)
            lines = self._pass_zero_load_after_zero_store(lines)
            lines = self._pass_xor_then_and_collapse(lines)
            lines = self._pass_xor_or_store_collapse(lines)
            lines = self._pass_dead_xor_or_chain_drop(lines)
            lines = self._pass_const_fold_chain(lines)
            lines = self._pass_or_with_zero_reg_drop(lines)
            lines = self._pass_label_load_collapse(lines)
            lines = self._pass_label_push_collapse(lines)
            lines = self._pass_label_store_collapse(lines)
            lines = self._pass_lea_load_collapse(lines)
            lines = self._pass_lea_sib_load_collapse(lines)
            lines = self._pass_lea_sib_label_load_collapse(lines)
            lines = self._pass_lea_self_sib_into_load(lines)
            lines = self._pass_lea_offset_fold(lines)
            lines = self._pass_lea_forward_to_reg(lines)
            lines = self._pass_lea_store_collapse(lines)
            lines = self._pass_dead_stack_store(lines)
            lines = self._pass_dead_unused_slot_stores(lines)
            lines = self._pass_slot_constant_propagation(lines)
            lines = self._pass_reg_copy_addr_forward(lines)
            lines = self._pass_value_forward_to_reg(lines)
            lines = self._pass_xfer_store_collapse(lines)
            lines = self._pass_dead_store_before_push(lines)
            lines = self._pass_dup_load_to_copy(lines)
            lines = self._pass_dup_load_chain_to_copy(lines)
            lines = self._pass_copy_save_deref_collapse(lines)
            lines = self._pass_dup_load_with_intermediate(lines)
            lines = self._pass_dup_addr_compute_collapse(lines)
            lines = self._pass_dead_addr_recompute(lines)
            lines = self._pass_uncollapse_cmp_when_reload(lines)
            lines = self._pass_redundant_cmp_at_label(lines)
            lines = self._pass_op_mem_to_reg_collapse(lines)
            lines = self._pass_push_pop_to_free_reg(lines)
            lines = self._pass_redundant_recompute_after_cmp(lines)
            lines = self._pass_redundant_mem_load_via_xfer(lines)
            lines = self._pass_load_add_xfer_forward(lines)
            # Runs AFTER load_add_xfer_forward so that pass gets first
            # crack on the `mov R1, SRC; add R1, IMM; mov R2, R1` shape.
            # My pass picks up the residual `mov XFER, BASE; mov BASE,
            # SRC; mov [XFER], BASE` triple that survives — common in
            # struct-array stores where the address arithmetic uses
            # imul, not add+disp.
            lines = self._pass_xfer_load_store_collapse(lines)
            lines = self._pass_byte_stores_to_dword(lines)
            lines = self._pass_pop_index_push_collapse(lines)
            lines = self._pass_pop_index_load_collapse(lines)
            # Runs AFTER all index/push/load consumers above so they
            # get first crack on the full shl+add+consumer chain.
            # When the chain has no load/push consumer (just produces
            # an address, e.g. for dec/push later), this pass folds
            # the shl+add pair into a single LEA.
            lines = self._pass_shl_add_reg_to_lea(lines)
            # Runs AFTER pop_index_*_collapse so those passes can
            # consume the SIB-form patterns first.
            lines = self._pass_pop_op_chain_retarget(lines)
            lines = self._pass_pop_cmp_chain_retarget(lines)
            lines = self._pass_push_pop_to_mov(lines)
            lines = self._pass_dead_push_pop_pair(lines)
            lines = self._pass_sib_const_index_fold(lines)
            lines = self._pass_push_const_index_fold(lines)
            lines = self._pass_trampoline_elimination(lines)
            lines = self._pass_pure_leaf_drop_frame(lines)
            lines = self._pass_drop_dead_frame_alloc(lines)
            # Drop labels that aren't referenced anywhere. Other
            # passes may have replaced jcc/jmp targets, leaving some
            # labels orphaned. After dropping, dead_after_terminator
            # in the next pass-iteration may extend the dead zone
            # past the dropped label, dropping more code.
            lines = self._pass_unreferenced_label_removal(lines)
            after = len(lines)
            if after == before:
                # All passes only delete or replace-with-fewer; if the
                # length didn't change, no rewrites fired this round.
                break

        out = "\n".join(line.raw for line in lines)
        if trailing_nl:
            out += "\n"
        return out

    # ── Patterns ──────────────────────────────────────────────────

    def _pass_dead_after_terminator(self, lines: list[Line]) -> list[Line]:
        """P1: Drop instruction lines (and comments/blanks adjacent to
        them) between an unconditional terminator and the next label /
        directive / data line.

        Labels, directives, data lines, and the terminator itself are
        preserved — only instructions in the dead zone get dropped, plus
        comments and blanks that have no live neighbor on either side.
        Conservative: we keep comments/blanks if any live instr exists
        before the next fence in the dead zone (none can, by definition,
        but the rule is: "comment/blank lines between dead code stay
        with the code" — so they're dropped too since the surrounding
        code is dead).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            out.append(line)
            if not _is_unconditional_terminator(line):
                i += 1
                continue
            # Found a terminator. Skip ahead past dead instructions.
            j = i + 1
            killed = 0
            while j < len(lines):
                nxt = lines[j]
                if nxt.kind in ("label", "directive", "data"):
                    break
                if nxt.kind == "instr":
                    killed += 1
                    j += 1
                    continue
                # Blank / comment in the dead zone — drop too, since
                # they were attached to the dead instructions.
                j += 1
            if killed:
                self.stats["dead_after_terminator"] = (
                    self.stats.get("dead_after_terminator", 0) + killed
                )
            i = j
        return out

    def _pass_trampoline_elimination(
        self, lines: list[Line]
    ) -> list[Line]:
        """Redirect jumps to a label that's just a `jmp` to another
        label.

        Pattern:
            ; (somewhere)
            jcc/jmp L1
            ; ...
            L1:
                jmp L3   ; first instruction at L1 is unconditional jmp
        →
            ; (somewhere)
            jcc/jmp L3   ; redirected
            ; ...
            (L1: jmp L3 — becomes unreferenced if all jumps redirected;
             cleaned up by unreferenced_label_removal in the next
             iteration)

        Conditions:
        - L1 is a NASM-local label (`.LX`)
        - L1's first instruction (after blanks/comments) is an
          unconditional `jmp` to some other label L3
        - L3 != L1 (no infinite-loop trampoline)
        - The jcc/jmp instruction's target is L1
        - L1 must not be the implicit fall-through from any other
          instruction (i.e., previous line is a terminator) — otherwise
          the original code may fall through into L1, and dropping
          the trampoline can change semantics.

        Why this works: the only way to reach the trampoline is via
        a jump targeting L1. Each such jump can be redirected to L3
        directly, since the trampoline just forwards.

        Saves 5 bytes per redirected jump (the `jmp imm32` form is
        5 bytes; `jmp imm8` is 2 bytes — saves 1 byte minimum). After
        all references are redirected, L1's def becomes unreferenced
        and `unreferenced_label_removal` cleans it up.

        Common shape: arises after `jcc_jmp_inversion` and other
        passes that simplify control flow, leaving short trampoline
        labels.
        """
        # Build map: (scope_owner, label) -> target_label for trampoline
        # labels (L1: jmp L3 — first instr after L1: is jmp).
        # NASM scopes local labels (.LX) per the most recent global
        # label. Two functions can each have their own .L12_case: ; only
        # one may be a trampoline. Track scope-aware to avoid redirecting
        # references in the wrong function's scope.
        trampolines: dict[tuple[str, str], str] = {}
        # current_scope tracks the most recent global label as we scan.
        current_scope = ""
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.kind != "label":
                i += 1
                continue
            stripped = ln.raw.strip()
            if not stripped.endswith(":"):
                i += 1
                continue
            label_name = stripped[:-1].strip()
            if not label_name.startswith("."):
                # Global label — update scope and skip (don't consider
                # globals for elimination; may be exported).
                current_scope = label_name
                i += 1
                continue
            # Verify previous line is a terminator (no fallthrough
            # into L1). If we don't, dropping the trampoline can
            # change behavior on the fallthrough path.
            prev_term = False
            for k in range(i - 1, -1, -1):
                pl = lines[k]
                if pl.kind in ("blank", "comment"):
                    continue
                if pl.kind == "label":
                    # Multiple labels in a row — fallthrough from
                    # the prior label's predecessor. Bail (safer).
                    prev_term = False
                    break
                if pl.kind == "instr":
                    if pl.op in (
                        "jmp", "ret", "retn", "retf",
                        "iret", "iretd",
                    ):
                        prev_term = True
                    # Also noreturn calls — skip for now (rare).
                    break
                # directive/data: also a boundary; treat as terminator
                if pl.kind in ("directive", "data"):
                    prev_term = True
                    break
            if not prev_term:
                i += 1
                continue
            # Look for the first instruction after the label.
            j = i + 1
            while j < len(lines) and lines[j].kind in (
                "blank", "comment", "label",
            ):
                # Multiple labels at same position: each is a separate
                # alias for the next instruction. We're looking for
                # what L1 maps to. If another label sits between L1
                # and the first instr, our target is the same.
                if lines[j].kind == "label":
                    # Don't redirect through multiple labels
                    # conservatively. Bail.
                    j = len(lines)
                    break
                j += 1
            if j >= len(lines):
                i += 1
                continue
            first_instr = lines[j]
            if first_instr.kind != "instr" or first_instr.op != "jmp":
                i += 1
                continue
            target = first_instr.operands.strip()
            # Don't redirect to self (infinite loop).
            if target == label_name:
                i += 1
                continue
            # Only same-scope redirects (NASM-local labels don't
            # cross scopes).  We require both the trampoline label
            # and its target to be local labels (`.X`) — emitting
            # `je <target>` from this scope only resolves to the
            # target if it's defined in this scope. Skip if target
            # is not local (a `jmp _global_other` trampoline target
            # is a different beast — punt for now).
            if not target.startswith("."):
                i += 1
                continue
            trampolines[(current_scope, label_name)] = target
            i += 1

        if not trampolines:
            return list(lines)

        # Now redirect every jcc/jmp targeting any L1 in trampolines.
        # Each redirect saves nothing in bytes here (jmp/jcc to L1 =
        # jmp/jcc to L3 — same instruction), but allows L1: jmp L3
        # to be cleaned up.
        out: list[Line] = []
        fired = False
        # Track the current scope as we walk the redirect pass too.
        cur_scope_redir = ""
        for ln in lines:
            if ln.kind == "label":
                stripped = ln.raw.strip()
                if stripped.endswith(":"):
                    name = stripped[:-1].strip()
                    if not name.startswith("."):
                        cur_scope_redir = name
                out.append(ln)
                continue
            if ln.kind != "instr":
                out.append(ln)
                continue
            # Match jmp or jcc with operand referencing a trampoline.
            if not (
                ln.op == "jmp"
                or (ln.op.startswith("j") and ln.op != "jmp")
                or ln.op.startswith("set")
                or ln.op.startswith("cmov")
            ):
                out.append(ln)
                continue
            tgt = ln.operands.strip()
            # Only consider local-label targets (qualified
            # `_other.LX` references are cross-scope and we don't
            # touch those).
            if not tgt.startswith("."):
                out.append(ln)
                continue
            key = (cur_scope_redir, tgt)
            if key not in trampolines:
                out.append(ln)
                continue
            # Build redirected instruction.
            new_target = trampolines[key]
            # Compute new raw line preserving indentation and op-text.
            indent = self._extract_indent(ln.raw)
            # Match how the original line was formatted: op,
            # whitespace, target.
            # Use the same op spelling (case) and a default
            # tab-aligned form.
            # The original's whitespace pattern is hard to replicate
            # exactly; emit with the standard 8-char alignment used
            # by the codegen.
            new_raw = f"{indent}{ln.op:<8}{new_target}"
            new_line = Line(
                raw=new_raw, kind="instr", op=ln.op,
                operands=new_target,
            )
            out.append(new_line)
            fired = True
            self.stats["trampoline_elimination"] = (
                self.stats.get("trampoline_elimination", 0) + 1
            )
        # If nothing changed, return original; otherwise return out.
        return out if fired else list(lines)

    def _pass_unreferenced_label_removal(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop NASM-local label definitions (`.LX:`) that have zero
        references within their function scope (and aren't referenced
        cross-scope as `<global>.LX`) AND are preceded by an
        unconditional terminator (jmp/ret/call-to-noreturn).

        Reference counting is **scope-aware** with **cross-scope
        qualified references**: NASM scopes local labels per the most
        recent global label, so a `.epilogue:` in `_main` and a
        `.epilogue:` in `_xb` are distinct. We count references only
        within the function's scope (the range from one global label
        to the next), PLUS any `_<global>.LX` qualified references
        from anywhere in the file (e.g. `dd _bar.L1` in `.data`).

        A "reference" is any textual occurrence of:
        - `.<name>` in an instruction operand or data directive
          within the same scope.
        - `<global_label>.<name>` (qualified) anywhere in the file.

        The label's own definition (`.<name>:`) does NOT count as a
        reference.

        After dropping, `dead_after_terminator` (next iteration) may
        extend its dead zone past the (now-dropped) label and drop
        any trailing dead code (e.g. a `leave; ret` epilogue that was
        unreachable after a noreturn `call _exit` / `call _abort`).

        Limitations:
        - Global labels (those without a leading `.`) are not
          considered for removal — they may be exported.
        """
        label_pattern = re.compile(r"\.[A-Za-z_]\w*")
        # `_global.local` qualified reference pattern.
        qualified_pattern = re.compile(
            r"\b([A-Za-z_]\w*)(\.[A-Za-z_]\w*)"
        )

        # Build per-scope ranges. Each scope is [start, end) of lines,
        # bounded by global labels or file boundaries.
        boundaries: list[int] = [0]
        scope_globals: list[str] = []
        # Walk to identify global labels and boundary positions.
        first_global_idx: int | None = None
        for k, line in enumerate(lines):
            if line.kind == "label":
                stripped = line.raw.strip()
                if stripped.endswith(":"):
                    n = stripped[:-1].strip()
                    if not n.startswith("."):
                        boundaries.append(k)
                        if first_global_idx is None:
                            first_global_idx = k
        boundaries.append(len(lines))
        # Dedup.
        boundaries = sorted(set(boundaries))
        # Determine the global label owning each scope.
        # The scope before the first global label has no owner (treat
        # as None).
        for s in range(len(boundaries) - 1):
            scope_start = boundaries[s]
            owner: str | None = None
            for k in range(scope_start, boundaries[s + 1]):
                line = lines[k]
                if line.kind == "label":
                    stripped = line.raw.strip()
                    if stripped.endswith(":"):
                        n = stripped[:-1].strip()
                        if not n.startswith("."):
                            owner = n
                            break
                # Stop at first instruction (no global label in this
                # scope past it).
                if line.kind == "instr":
                    break
            scope_globals.append(owner if owner else "")

        # Count cross-scope qualified references anywhere in the file.
        # cross_refs maps (global_owner, ".local") -> count.
        cross_refs: dict[tuple[str, str], int] = {}
        for line in lines:
            for m in qualified_pattern.finditer(line.raw):
                global_part, local_part = m.group(1), m.group(2)
                key = (global_part, local_part)
                cross_refs[key] = cross_refs.get(key, 0) + 1

        # For each scope, count IN-SCOPE references (= textual
        # occurrences of `.<name>` minus the definition's own match).
        # We exclude qualified references `<global>.<local>` from this
        # in-scope count to avoid double-counting (they're tracked
        # separately in cross_refs).
        scope_ref_counts: list[dict[str, int]] = []
        for s in range(len(boundaries) - 1):
            scope_start = boundaries[s]
            scope_end = boundaries[s + 1]
            ref_counts: dict[str, int] = {}
            for k in range(scope_start, scope_end):
                line = lines[k]
                # Identify this line's own label-definition name (so
                # we don't count it as a reference).
                own_name: str | None = None
                if line.kind == "label":
                    stripped = line.raw.strip()
                    if stripped.endswith(":"):
                        n = stripped[:-1].strip()
                        if n.startswith("."):
                            own_name = n
                # Build a set of "covered" character spans for
                # qualified `_g.local` matches — those are tracked
                # separately, so don't double-count their `.local`
                # parts in the in-scope counting.
                qualified_spans: list[tuple[int, int]] = []
                for qm in qualified_pattern.finditer(line.raw):
                    qualified_spans.append(qm.span())
                # Count `.<name>` matches that are NOT inside a
                # qualified match.
                for match in label_pattern.finditer(line.raw):
                    name = match.group()
                    span_start = match.start()
                    inside_qualified = any(
                        qs[0] <= span_start < qs[1]
                        for qs in qualified_spans
                    )
                    if inside_qualified:
                        continue
                    if name == own_name:
                        # Own definition match — don't count.
                        own_name = None  # only skip first match
                        continue
                    ref_counts[name] = ref_counts.get(name, 0) + 1
            scope_ref_counts.append(ref_counts)

        # Map each line idx to its scope idx.
        def scope_idx_for(idx: int) -> int:
            # Find largest j where boundaries[j] <= idx.
            for j in range(len(boundaries) - 2, -1, -1):
                if boundaries[j] <= idx:
                    return j
            return 0

        # Walk and drop unreferenced label definitions.
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.kind == "label":
                stripped = line.raw.strip()
                if stripped.endswith(":"):
                    name = stripped[:-1].strip()
                    if name.startswith("."):
                        s_idx = scope_idx_for(i)
                        in_scope = scope_ref_counts[s_idx].get(
                            name, 0
                        )
                        owner = scope_globals[s_idx]
                        cross = cross_refs.get((owner, name), 0)
                        if (
                            in_scope == 0
                            and cross == 0
                            and self._preceded_by_terminator(
                                lines, i
                            )
                        ):
                            self.stats["unreferenced_label_removal"] = (
                                self.stats.get(
                                    "unreferenced_label_removal", 0
                                ) + 1
                            )
                            i += 1
                            continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _preceded_by_terminator(
        lines: list[Line], idx: int
    ) -> bool:
        """Look BACKWARD from `idx` for the previous instruction
        (skipping blanks/comments/labels). Return True if it's an
        unconditional terminator. If no instruction is found before
        a directive or the file start, return False (conservative)."""
        for k in range(idx - 1, -1, -1):
            ln = lines[k]
            if ln.kind in ("blank", "comment"):
                continue
            if ln.kind == "label":
                continue  # consecutive labels — keep looking back
            if ln.kind == "directive":
                return False
            if ln.kind == "instr":
                return _is_unconditional_terminator(ln)
            return False
        return False

    @staticmethod
    def _followed_by_function_epilogue(
        lines: list[Line], start: int
    ) -> bool:
        """Look at the next instruction (skipping blanks/comments).
        Return True if it's `leave`, `ret`, or another label (which
        means the label we're considering is part of a chain leading
        to an epilogue). Return False otherwise."""
        for k in range(start, len(lines)):
            ln = lines[k]
            if ln.kind == "label":
                # Chained label — if THAT is followed by leave/ret,
                # then ours is too. Recurse.
                return PeepholeOptimizer._followed_by_function_epilogue(
                    lines, k + 1
                )
            if ln.kind in ("blank", "comment"):
                continue
            if ln.kind == "instr":
                return ln.op in {"leave", "ret",
                                 "iret", "iretd", "retf", "retn"}
            return False
        return False

    def _pass_binop_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse the stack-machine right-operand transfer pattern.

        Match (4 consecutive instr lines, blanks/comments tolerated):
            push    eax
            mov     eax, <src>           ; single-instruction right
            mov     ecx, eax
            pop     eax
        Replace with:
            mov     ecx, <src>

        The push saves the left operand; the right is loaded into EAX,
        transferred to ECX as the binop's second operand, and EAX is
        restored. After collapse, EAX is unchanged (no clobber → no
        save needed), and the right lands directly in ECX.

        Safe because `mov ecx, src` reads `src` with the same EAX/ECX
        values as the original middle instruction did (push doesn't
        change EAX, and the original middle hadn't touched ECX yet).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for `push eax`.
            if (line.kind == "instr" and line.op == "push"
                    and line.operands.strip().lower() == "eax"):
                # Need three more instr lines below: mov eax, src;
                # mov ecx, eax; pop eax.
                instrs = self._next_n_instrs(lines, i + 1, 3)
                if instrs is not None and self._matches_binop_collapse(instrs):
                    [(_, c), (_, d), (_, e)] = instrs
                    # Keep original blanks/comments between push and the
                    # first instr (none in our codegen, but be safe).
                    # Build the replacement: a single retargeted line.
                    new_line = _retarget_eax_to_ecx(c)
                    # Skip from `push eax` (i) through `pop eax` (e_idx).
                    e_idx = instrs[2][0]
                    out.append(new_line)
                    self.stats["binop_collapse"] = (
                        self.stats.get("binop_collapse", 0) + 1
                    )
                    i = e_idx + 1
                    continue
            out.append(line)
            i += 1
        return out

    def _pass_store_collapse(self, lines: list[Line]) -> list[Line]:
        """Drop the push/pop pair around a store-through-pointer.

        Match (4 consecutive instr lines):
            push    eax
            mov     eax, <src>           ; single-instruction value
            pop     ecx
            mov     [ecx], eax
        Replace with:
            mov     ecx, eax
            mov     eax, <src>
            mov     [ecx], eax

        Net: drop push + pop, add `mov ecx, eax`. Saves 1 instruction
        plus 2 stack ops. Conservative compared to the full retarget
        variant; that one rewrites the previous `mov eax, addr` line
        too. Defer.

        Safe because `<src>` is verified not to read ECX (would
        observe the new save value rather than the old caller-state).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr" and line.op == "push"
                    and line.operands.strip().lower() == "eax"):
                instrs = self._next_n_instrs(lines, i + 1, 3)
                if instrs is not None and self._matches_store_collapse(instrs):
                    [(c_idx, c), (d_idx, _d), (e_idx, e)] = instrs
                    # Replace `push eax` with `mov ecx, eax`.
                    push_line = line
                    new_push = Line(
                        raw=push_line.raw.replace(
                            "push    eax", "mov     ecx, eax", 1
                        ).replace("push eax", "mov ecx, eax", 1),
                        kind="instr", op="mov", operands="ecx, eax",
                    )
                    out.append(new_push)
                    out.append(c)
                    out.append(e)
                    self.stats["store_collapse"] = (
                        self.stats.get("store_collapse", 0) + 1
                    )
                    i = e_idx + 1
                    continue
            out.append(line)
            i += 1
        return out

    # ── Helpers for multi-line patterns ───────────────────────────

    def _next_n_instrs(self, lines: list[Line], start: int,
                       n: int) -> list[tuple[int, Line]] | None:
        """Return the next N instruction lines starting at index
        `start`, as `[(idx, line), ...]`. Skip blanks/comments. Return
        None if we hit a label/directive/data before finding N instrs."""
        out: list[tuple[int, Line]] = []
        j = start
        while j < len(lines) and len(out) < n:
            ln = lines[j]
            if ln.kind == "instr":
                out.append((j, ln))
            elif ln.kind in ("label", "directive", "data"):
                return None
            j += 1
        if len(out) < n:
            return None
        return out

    @staticmethod
    def _matches_binop_collapse(
        instrs: list[tuple[int, Line]],
    ) -> bool:
        """Verify the 3-instruction tail of the binop pattern:
        `mov eax, <src>; mov ecx, eax; pop eax`."""
        c_line = instrs[0][1]
        d_line = instrs[1][1]
        e_line = instrs[2][1]
        if not _is_simple_eax_load(c_line):
            return False
        # `push eax` decreases ESP by 4. If src is ESP-relative, the
        # original `mov eax, [esp + N]` reads from a different location
        # than the post-collapse `mov ecx, [esp + N]` would. Skip.
        parts = _operands_split(c_line.operands)
        assert parts is not None
        _, src = parts
        if _references_register(src, "esp"):
            return False
        if not (d_line.op == "mov"
                and d_line.operands.replace(" ", "").lower() == "ecx,eax"):
            return False
        if not (e_line.op == "pop"
                and e_line.operands.strip().lower() == "eax"):
            return False
        return True

    @staticmethod
    def _matches_store_collapse(
        instrs: list[tuple[int, Line]],
    ) -> bool:
        """Verify the 3-instruction tail of the store pattern:
        `mov eax, <src>; pop ecx; mov [ecx], eax` where src doesn't
        read ECX or ESP."""
        c_line = instrs[0][1]
        d_line = instrs[1][1]
        e_line = instrs[2][1]
        if not _is_simple_eax_load(c_line):
            return False
        parts = _operands_split(c_line.operands)
        assert parts is not None
        _, src = parts
        # src must not read ECX (would alias the saved address) or ESP
        # (the push shifted ESP, so an `[esp + N]` operand reads a
        # different byte than it would post-collapse).
        if _references_register(src, "ecx"):
            return False
        if _references_register(src, "esp"):
            return False
        if not (d_line.op == "pop"
                and d_line.operands.strip().lower() == "ecx"):
            return False
        if not (e_line.op == "mov"
                and e_line.operands.replace(" ", "").lower() == "[ecx],eax"):
            return False
        return True

    # Inverse table for `j<cc>` mnemonics. Mirrors `set<cc>`'s opposite
    # condition. Used by setcc_jcc_collapse.
    _CC_INVERSE: dict[str, str] = {
        "e": "ne", "ne": "e",
        "z": "nz", "nz": "z",   # z is alias of e
        "l": "ge", "ge": "l",
        "le": "g", "g": "le",
        "b": "ae", "ae": "b",
        "be": "a", "a": "be",
        "c": "nc", "nc": "c",   # c is alias of b
        "s": "ns", "ns": "s",
        "o": "no", "no": "o",
        "p": "np", "np": "p",
        "pe": "po", "po": "pe",
    }

    def _pass_setcc_jcc_collapse(self, lines: list[Line]) -> list[Line]:
        """Replace the setCC + movzx + test + jz/jnz boolean-materialize
        chain with a single conditional jump.

        Match (4 consecutive instr lines):
            setCC   al                ; or any sub-byte register
            movzx   eax, al           ; widen
            test    eax, eax
            j[n]z   LBL

        Replace with:
            j<CC-or-inverse>  LBL

        Mapping: `jz LBL` after `setCC al` jumps when AL was 0 — i.e.,
        when the inverted CC was true. So emit `j<inverse-of-CC> LBL`.
        `jnz LBL` jumps when AL was 1 — emit `j<CC> LBL` directly.

        The setCC's destination must be the low byte of EAX (AL),
        the movzx must widen AL into EAX, and the test must be on
        EAX. We don't try to handle other registers (BL, CL, etc.) —
        the codegen always uses AL → EAX in this pattern.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr"
                    and line.op.startswith("set")
                    and line.op[3:] in self._CC_INVERSE):
                # setCC. Check operand is `al`.
                if line.operands.strip().lower() != "al":
                    out.append(line)
                    i += 1
                    continue
                instrs = self._next_n_instrs(lines, i + 1, 3)
                if instrs is None:
                    out.append(line)
                    i += 1
                    continue
                b_line = instrs[0][1]
                c_line = instrs[1][1]
                d_line = instrs[2][1]
                # b: `movzx eax, al`
                if not (b_line.op == "movzx"
                        and b_line.operands.replace(" ", "").lower()
                            == "eax,al"):
                    out.append(line)
                    i += 1
                    continue
                # c: `test eax, eax`
                if not (c_line.op == "test"
                        and c_line.operands.replace(" ", "").lower()
                            == "eax,eax"):
                    out.append(line)
                    i += 1
                    continue
                # d: `jz LBL` or `jnz LBL`. Resolve target label.
                if d_line.op not in ("jz", "je", "jnz", "jne"):
                    out.append(line)
                    i += 1
                    continue
                target = d_line.operands.strip()
                if not target:
                    out.append(line)
                    i += 1
                    continue
                # Map: setCC + jz → j<inverted CC>; setCC + jnz → j<CC>
                cc = line.op[3:]
                jz_like = d_line.op in ("jz", "je")
                final_cc = self._CC_INVERSE[cc] if jz_like else cc
                leading_ws = re.match(r"^\s*", line.raw).group(0)
                new_raw = f"{leading_ws}j{final_cc:<7}{target}"
                new_line = Line(raw=new_raw, kind="instr",
                                op=f"j{final_cc}", operands=target)
                out.append(new_line)
                self.stats["setcc_jcc_collapse"] = (
                    self.stats.get("setcc_jcc_collapse", 0) + 1
                )
                # Skip past lines i (setCC), b, c, d.
                d_idx = instrs[2][0]
                i = d_idx + 1
                continue
            out.append(line)
            i += 1
        return out

    def _pass_push_immediate(self, lines: list[Line]) -> list[Line]:
        """Replace `mov <reg>, IMM_OR_LABEL; ... ; push <reg>;
        <reg-write>` with `... ; push IMM_OR_LABEL; <reg-write>`.

        The codegen emits `mov reg, X; push reg` for every cdecl arg
        push. NASM's `push imm8` / `push imm32` instructions skip the
        register round-trip entirely. The instruction after `push reg`
        must overwrite the register for the rewrite to be safe.

        Up to 4 "reg-neutral" instructions may sit between the mov
        and the push — they must not read OR write the register's
        family (EAX/AX/AL/AH for EAX, etc.). This catches the LL-push
        pattern `mov eax, lo; mov edx, hi; push edx; push eax` which
        becomes `mov edx, hi; push edx; push lo` (and recurses on
        the EDX form to give `push hi; push lo`).

        Works for EAX, EBX, ECX, EDX, ESI, EDI, EBP. ESP excluded
        (used as stack pointer; rewrites would break alignment).
        """
        SUPPORTED_REGS = {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp"}

        def overwrites_reg(line: Line, reg32: str) -> bool:
            return PeepholeOptimizer._is_pure_reg_write(line, reg32)

        def is_reg_neutral(line: Line, reg32: str) -> bool:
            """Doesn't read or write the register family. May touch
            other state."""
            if line.kind != "instr":
                return False
            if line.op == "call":
                # Direct calls clobber EAX/ECX/EDX. For other regs
                # they're preserved. But indirect calls or any call
                # might invalidate analysis; conservative: block.
                return False
            if line.op in {"ret", "iret", "iretd", "retf", "retn",
                            "jmp"} or line.op.startswith("j"):
                return False
            if not PeepholeOptimizer._has_explicit_operands_only(line):
                return False
            if PeepholeOptimizer._references_reg_family(
                    line.operands, reg32):
                return False
            return True

        def is_imm_or_label_to_reg(line: Line, reg32: str) -> bool:
            if line.kind != "instr" or line.op != "mov":
                return False
            parts = _operands_split(line.operands)
            if parts is None:
                return False
            dest, src = parts
            if dest.lower() != reg32:
                return False
            if "[" in src:
                return False
            regs = {"eax", "ebx", "ecx", "edx", "esi", "edi",
                    "ebp", "esp",
                    "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
                    "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh"}
            if src.lower() in regs:
                return False
            return True

        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            if (line.kind == "instr" and line.op == "mov"):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    reg32 = dest.lower()
                    if (reg32 in SUPPORTED_REGS
                            and is_imm_or_label_to_reg(line, reg32)):
                        # Find the matching `push <reg>` within the
                        # next 5 instructions, where each intermediate
                        # is reg-neutral.
                        push_idx: int | None = None
                        witness_idx: int | None = None
                        neutrals = 0
                        for j_offset, ln in enumerate(
                                out[i + 1:], start=i + 1):
                            if ln.kind in ("blank", "comment"):
                                continue
                            if ln.kind != "instr":
                                break
                            if (ln.op == "push"
                                    and ln.operands.strip().lower()
                                    == reg32):
                                push_idx = j_offset
                                # Find the witness — scan past
                                # reg-neutral instrs (up to 10).
                                w_neutrals = 0
                                for k_offset, kln in enumerate(
                                        out[j_offset + 1:],
                                        start=j_offset + 1):
                                    if kln.kind in ("blank", "comment"):
                                        continue
                                    if kln.kind != "instr":
                                        break
                                    if overwrites_reg(kln, reg32):
                                        witness_idx = k_offset
                                        break
                                    if not is_reg_neutral(kln, reg32):
                                        break
                                    w_neutrals += 1
                                    if w_neutrals > 10:
                                        break
                                break
                            if not is_reg_neutral(ln, reg32):
                                break
                            neutrals += 1
                            if neutrals > 10:
                                break
                        if (push_idx is not None
                                and witness_idx is not None):
                            push_line = out[push_idx]
                            leading_ws = re.match(
                                r"^\s*", push_line.raw,
                            ).group(0)
                            new_raw = f"{leading_ws}push    {src}"
                            out[push_idx] = Line(
                                raw=new_raw, kind="instr",
                                op="push", operands=src,
                            )
                            out.pop(i)
                            self.stats["push_immediate"] = (
                                self.stats.get("push_immediate", 0) + 1
                            )
                            continue
            i += 1
        return out

    def _pass_push_memory(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, [mem]; push reg`` into a single
        ``push dword [mem]``.

        NASM's ``push r/m32`` form is 3 bytes for ebp-relative
        addressing (``FF /6 modrm``); the original mov+push is 4
        bytes (``mov reg, [mem]`` is 3 bytes + ``push reg`` is 1
        byte). Saves 1 byte per match.

        Common in cdecl call-site arg setup, where the intermediate
        register is dead after the push (the caller's call instruction
        doesn't read its arg-setup registers).

        Conditions:
        - Line A: ``mov reg, [mem]`` (memory source, register dest).
        - Line B: ``push reg`` (same reg).
        - reg is dead after the push (CFG-aware via `_reg_dead_after`).

        Restricted to EAX/EBX/ECX/EDX/ESI/EDI/EBP — avoids ESP
        (which would corrupt the stack) and segment registers.

        ESP-relative memory operands (``[esp + N]``) are skipped:
        ``push`` decrements ESP, so the memory operand's address
        would shift between mov and push (well, it's stable in
        mov-then-push-with-direct-mem because the memory access
        happens before ESP changes — but we play conservative).
        """
        SUPPORTED_REGS = {
            "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp",
        }
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if (
                        dest_low in SUPPORTED_REGS
                        and src_norm.startswith("[")
                        and src_norm.endswith("]")
                        and "esp" not in src_norm.lower()
                    ):
                        nxt = lines[i + 1]
                        if (
                            nxt.kind == "instr"
                            and nxt.op == "push"
                            and nxt.operands.strip().lower()
                            == dest_low
                            and self._reg_dead_after(
                                lines, i + 2, dest_low
                            )
                        ):
                            indent = self._extract_indent(nxt.raw)
                            new_operands = f"dword {src_norm}"
                            new_raw = (
                                f"{indent}push    {new_operands}"
                            )
                            new_line = Line(
                                raw=new_raw,
                                kind="instr",
                                op="push",
                                operands=new_operands,
                            )
                            out.append(new_line)
                            self.stats["push_memory"] = (
                                self.stats.get("push_memory", 0) + 1
                            )
                            i += 2
                            continue
            out.append(line)
            i += 1
        return out

    def _pass_push_memory_to_push_reg(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace ``push <size?> [m]`` with ``push REG`` when REG is
        known to already hold ``[m]``'s value from a recent
        ``mov REG, [m]`` that hasn't been invalidated.

        Saves 2 bytes per match for ebp-relative ``[m]`` (3-byte
        push-mem → 1-byte push-reg), or 4-5 bytes for label or SIB
        memory (push-mem with disp32 / SIB → 1-byte push-reg).

        Common shape: a for-loop body that loads ``i`` for the
        condition cmp into EAX, then later pushes ``i`` as a call
        argument. The push currently re-reads memory; my pass reuses
        the value already in EAX.

        Differs from ``push_memory`` (collapses adjacent
        ``mov reg, [m]; push reg`` into ``push dword [m]`` — saving
        bytes when the reg is dead, since the dropped mov is bigger
        than the gained `dword` prefix). My pass goes the OTHER
        direction: when the load and push are NON-adjacent (separated
        by a chain that uses REG's value), keeping REG and using
        ``push REG`` is shorter than the explicit memory push.

        Tracking model: per-register ``reg_mem`` map updated as the
        pass walks lines. Set on ``mov REG, [m]``, invalidated by
        any write to REG or to overlapping memory, by labels (control-
        flow boundaries — REG might enter from elsewhere), by jumps
        / calls (caller-saved-reg clobber, control flow), by
        directives / data sections (function boundary).

        Conditions:
        - REG is a 32-bit GP register.
        - [m] is ebp-relative literal `[ebp ± N]` or label literal
          `[_name]` (we can compare textually). Other forms (SIB,
          register-base) bail on safety grounds.
        - When push appears, find a REG whose tracked mem equals
          push's memory operand (after stripping size prefix and
          normalizing whitespace).
        """
        out: list[Line] = []
        # reg_mem maps reg32 → memory operand (string).
        reg_mem: dict[str, str] = {}
        # Track whether previous instruction was an unconditional
        # terminator (ret/jmp/leave): if so, the next label has no
        # text fallthrough predecessor.
        prev_unconditional = False

        def normalize_mem(text: str) -> str | None:
            """Strip size prefix and normalize. Return None if not
            a recognized ebp-relative or label form."""
            t = text.strip()
            for prefix in (
                "dword ", "word ", "byte ", "qword ",
            ):
                if t.lower().startswith(prefix):
                    t = t[len(prefix):].lstrip()
                    break
            if not (t.startswith("[") and t.endswith("]")):
                return None
            inner = t[1:-1].strip()
            # Reject if contains *, [, ] — SIB form.
            if "*" in inner:
                return None
            # Normalize whitespace.
            return "[" + " ".join(inner.split()) + "]"

        for i, ln in enumerate(lines):
            if ln.kind == "blank" or ln.kind == "comment":
                out.append(ln)
                continue
            if ln.kind == "label":
                # Control-flow boundary — invalidate.
                reg_mem = {}
                out.append(ln)
                prev_unconditional = False
                continue
            if ln.kind in ("directive", "data"):
                reg_mem = {}
                out.append(ln)
                prev_unconditional = False
                continue
            if ln.kind != "instr":
                out.append(ln)
                continue
            # Process the instruction.
            op = ln.op
            ops = ln.operands

            # Case: push <size?> [m] — try to substitute.
            if op == "push" and reg_mem:
                src = ops.strip()
                m = normalize_mem(src)
                if m is not None:
                    # Find a reg whose tracked mem matches.
                    matched_reg: str | None = None
                    for r, rm in reg_mem.items():
                        if rm == m:
                            matched_reg = r
                            break
                    if matched_reg is not None:
                        indent = self._extract_indent(ln.raw)
                        new_raw = (
                            f"{indent}push    {matched_reg}"
                        )
                        new_line = Line(
                            raw=new_raw,
                            kind="instr",
                            op="push",
                            operands=matched_reg,
                        )
                        out.append(new_line)
                        self.stats[
                            "push_memory_to_push_reg"
                        ] = self.stats.get(
                            "push_memory_to_push_reg", 0
                        ) + 1
                        prev_unconditional = False
                        # push doesn't write any reg; tracking
                        # state unchanged.
                        continue

            # Update tracking for `mov REG, [m]`.
            if op == "mov":
                parts = _operands_split(ops)
                if parts is not None:
                    dst = parts[0].strip().lower()
                    src = parts[1].strip()
                    if (
                        self._is_general_register(dst)
                        and src.startswith("[")
                        and src.endswith("]")
                    ):
                        m = normalize_mem(src)
                        if m is not None:
                            # Invalidate any other reg holding the
                            # same memory (since dst now owns it).
                            reg_mem = {
                                r: rm for r, rm in reg_mem.items()
                                if rm != m
                            }
                            # Don't track if the memory expression
                            # references the destination register
                            # itself: the value of `dst` just
                            # changed, so the cached `[dst+N]`
                            # textual key now points to a different
                            # location than it did before this
                            # instruction. A later
                            # `push [dst+N]` reads the NEW location;
                            # collapsing it to `push dst` would push
                            # the value just loaded (the old location's
                            # contents), not the new location's
                            # contents — corrupts a chained pointer
                            # dereference like
                            # `self->context->module.globals`.
                            inner = m[1:-1]  # strip "[" "]"
                            if not _references_register(inner, dst):
                                reg_mem[dst] = m
                            else:
                                # Drop any prior tracker for dst —
                                # the register was overwritten and
                                # we have nothing safe to track.
                                reg_mem.pop(dst, None)
                            out.append(ln)
                            prev_unconditional = False
                            continue
                    # mov dst, X (any other shape): invalidate dst.
                    if self._is_general_register(dst):
                        reg_mem.pop(dst, None)
                    # If dst is a memory operand, invalidate any
                    # reg whose tracked mem matches it (or
                    # conservatively, invalidate all).
                    if dst.startswith("[") or dst.startswith(
                        "dword "
                    ) or dst.startswith("word "
                    ) or dst.startswith("byte "):
                        # Conservative: clear all tracking on any
                        # memory write (don't try alias analysis).
                        reg_mem = {}
                    out.append(ln)
                    prev_unconditional = False
                    continue

            # Calls clobber caller-saved regs and the called
            # function may modify memory.
            if op == "call":
                reg_mem = {}
                out.append(ln)
                prev_unconditional = False
                continue

            # Jumps end the basic block.
            if op == "jmp":
                reg_mem = {}
                out.append(ln)
                prev_unconditional = True
                continue
            if op.startswith("j"):
                # Conditional — fall through with same tracking
                # but the taken path resets it (we don't track
                # across taken paths).
                out.append(ln)
                prev_unconditional = False
                continue
            if op in {
                "ret", "iret", "iretd", "retf", "retn",
                "leave",
            }:
                reg_mem = {}
                out.append(ln)
                prev_unconditional = True
                continue

            # Other instructions: invalidate the destination
            # register if it's a write.
            parts = _operands_split(ops)
            if parts is not None:
                dst = parts[0].strip().lower()
                if self._is_general_register(dst):
                    # 2-operand op writes dst (cmp/test are reads;
                    # all others write dst).
                    if op not in {"cmp", "test"}:
                        reg_mem.pop(dst, None)
                # Memory write — invalidate all tracking.
                if dst.startswith("["):
                    reg_mem = {}
                # Implicit-reg-using ops (cdq writes EDX from EAX,
                # etc.): conservative, clear all.
                if not PeepholeOptimizer._has_explicit_operands_only(
                    ln
                ):
                    reg_mem = {}
            else:
                # 1-operand or zero-operand op. inc/dec/neg/not on
                # a reg modifies it; pop writes its operand.
                # Conservative: invalidate.
                if op in {
                    "inc", "dec", "neg", "not", "pop", "mul",
                    "imul", "div", "idiv", "cdq", "stosb",
                    "stosw", "stosd", "lodsb", "lodsw", "lodsd",
                }:
                    # These touch eax/edx/ecx/etc. implicitly.
                    reg_mem = {}
            out.append(ln)
            prev_unconditional = False
        return out

    def _pass_imm_store_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse `mov eax, IMM; mov [addr], eax` to `mov dword
        [addr], IMM`, when EAX is provably dead after the store.

        The fast path uses an immediate witness (the very next instr
        is a fresh EAX overwrite); a CFG-aware fallback uses
        ``_reg_dead_after`` for cases where the witness is past a
        label boundary (basic block boundary).

        NASM accepts `mov dword [addr], IMM` directly.

        Conditions:
        - Line A: `mov eax, src` where src is a constant expression
          (no `[` — i.e., not a memory load, not a register-to-EAX
          where the register might be live).
        - Line B: `mov [addr], eax` — a 4-byte store from EAX.
        - EAX must be dead after Line B (immediate witness OR CFG-
          aware liveness).
        """
        out_list: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr" and line.op == "mov"
                    and self._is_imm_to_eax(line)):
                # Try the fast path first: immediate witness in the
                # next 2 instructions (B then C-as-witness).
                fast_instrs = self._next_n_instrs(lines, i + 1, 2)
                fired_fast = False
                b_idx_fast = None
                if fast_instrs is not None and self._matches_imm_store(
                    fast_instrs, line
                ):
                    [(b_idx_fast, _b), (_c_idx, _c)] = fast_instrs
                    fired_fast = True
                if fired_fast:
                    [(b_idx, b_line), (_c_idx, _c)] = fast_instrs
                    parts_a = _operands_split(line.operands)
                    assert parts_a is not None
                    _, src_imm = parts_a
                    parts_b = _operands_split(b_line.operands)
                    assert parts_b is not None
                    addr, _ = parts_b
                    leading_ws = re.match(r"^\s*", line.raw).group(0)
                    new_raw = f"{leading_ws}mov     dword {addr}, {src_imm}"
                    new_line = Line(
                        raw=new_raw, kind="instr", op="mov",
                        operands=f"dword {addr}, {src_imm}",
                    )
                    out_list.append(new_line)
                    self.stats["imm_store_collapse"] = (
                        self.stats.get("imm_store_collapse", 0) + 1
                    )
                    i = b_idx + 1
                    continue
                # Slow path: CFG-aware liveness — check Line B (next
                # instr is `mov [m], eax`) and EAX dead after.
                next_instrs = self._next_n_instrs(lines, i + 1, 1)
                if next_instrs is not None:
                    [(b_idx, b_line)] = next_instrs
                    if (
                        b_line.op == "mov"
                        and self._is_eax_to_mem_store(b_line)
                        and self._reg_dead_after(
                            lines, b_idx + 1, "eax"
                        )
                    ):
                        parts_a = _operands_split(line.operands)
                        assert parts_a is not None
                        _, src_imm = parts_a
                        addr = self._mem_dest(b_line)
                        if addr is not None:
                            leading_ws = re.match(
                                r"^\s*", line.raw
                            ).group(0)
                            new_raw = (
                                f"{leading_ws}mov     dword "
                                f"{addr}, {src_imm}"
                            )
                            new_line = Line(
                                raw=new_raw, kind="instr", op="mov",
                                operands=(
                                    f"dword {addr}, {src_imm}"
                                ),
                            )
                            out_list.append(new_line)
                            self.stats[
                                "imm_store_collapse"
                            ] = (
                                self.stats.get(
                                    "imm_store_collapse", 0
                                ) + 1
                            )
                            i = b_idx + 1
                            continue
            out_list.append(line)
            i += 1
        return out_list

    def _pass_imm_store_then_imm_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Reorder `mov <size> [m], IMM; mov reg32, IMM` into
        `mov reg32, IMM; mov [m], <reg/sub>`. Saves 4 bytes per match.

        Pattern (2 adjacent instructions):
            A: mov <size> [m], IMM_A
            B: mov reg32, IMM_B   (IMM_A == IMM_B numerically)
        Rewrite:
            A': mov reg32, IMM_A
            B': mov [m], <reg or sub-alias matching size>

        Conditions:
        - A's IMM is a numeric literal (not label arithmetic).
        - B's IMM is a numeric literal equal to A's.
        - A's memory operand m doesn't reference reg32 (or its
          sub-aliases). Else writing reg32 first changes the
          effective address.
        - For byte size, reg32 must be eax/ebx/ecx/edx (have 8-bit
          aliases). esi/edi rejected.

        Both directions are semantically equivalent; the rewrite is
        4 bytes shorter because:
            - Original: `mov dword [m], IMM` (~7 bytes for ebp-rel
              + imm32) + `mov reg32, IMM` (5 bytes for B8 + imm32)
              = 12 bytes
            - Rewrite: `mov reg32, IMM` (5 bytes) + `mov [m], reg32`
              (3 bytes for ebp-rel + reg) = 8 bytes
        """
        sub_aliases = {
            "eax": ("al", "ax"),
            "ebx": ("bl", "bx"),
            "ecx": ("cl", "cx"),
            "edx": ("dl", "dx"),
            "esi": (None, "si"),
            "edi": (None, "di"),
        }
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr" and line.op == "mov"
                    and i + 1 < len(lines)):
                a_parts = _operands_split(line.operands)
                if a_parts is not None:
                    a_dst, a_src = a_parts
                    a_dst_t = a_dst.strip()
                    size = None
                    addr = None
                    if a_dst_t.startswith("dword "):
                        size = "dword"
                        addr = a_dst_t[6:].strip()
                    elif a_dst_t.startswith("word "):
                        size = "word"
                        addr = a_dst_t[5:].strip()
                    elif a_dst_t.startswith("byte "):
                        size = "byte"
                        addr = a_dst_t[5:].strip()
                    if size is not None and addr is not None:
                        a_imm_val = self._parse_imm_value(a_src.strip())
                        if a_imm_val is not None:
                            b_line = lines[i + 1]
                            if (b_line.kind == "instr"
                                    and b_line.op == "mov"):
                                b_parts = _operands_split(b_line.operands)
                                if b_parts is not None:
                                    b_dst, b_src = b_parts
                                    b_dst_low = b_dst.strip().lower()
                                    if b_dst_low in sub_aliases:
                                        b_imm_val = self._parse_imm_value(
                                            b_src.strip()
                                        )
                                        if b_imm_val == a_imm_val:
                                            reg32 = b_dst_low
                                            sub8, sub16 = sub_aliases[
                                                reg32
                                            ]
                                            ok = True
                                            if size == "byte":
                                                if sub8 is None:
                                                    ok = False
                                            if ok and self._addr_refs_reg(
                                                addr, reg32
                                            ):
                                                ok = False
                                            if ok:
                                                if size == "dword":
                                                    src_reg = reg32
                                                elif size == "word":
                                                    src_reg = sub16
                                                else:
                                                    src_reg = sub8
                                                a_lead = re.match(
                                                    r"^\s*", line.raw
                                                ).group(0)
                                                b_lead = re.match(
                                                    r"^\s*", b_line.raw
                                                ).group(0)
                                                new_a_raw = (
                                                    f"{a_lead}mov     "
                                                    f"{reg32}, "
                                                    f"{a_src.strip()}"
                                                )
                                                new_a = Line(
                                                    raw=new_a_raw,
                                                    kind="instr",
                                                    op="mov",
                                                    operands=(
                                                        f"{reg32}, "
                                                        f"{a_src.strip()}"
                                                    ),
                                                )
                                                new_b_raw = (
                                                    f"{b_lead}mov     "
                                                    f"{addr}, {src_reg}"
                                                )
                                                new_b = Line(
                                                    raw=new_b_raw,
                                                    kind="instr",
                                                    op="mov",
                                                    operands=(
                                                        f"{addr}, "
                                                        f"{src_reg}"
                                                    ),
                                                )
                                                out.append(new_a)
                                                out.append(new_b)
                                                self.stats[
                                                    "imm_store_then_imm_load_collapse"
                                                ] = (
                                                    self.stats.get(
                                                        "imm_store_then_imm_load_collapse",
                                                        0,
                                                    ) + 1
                                                )
                                                i += 2
                                                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _parse_imm_value(text: str) -> int | None:
        """Parse a numeric immediate (decimal, hex). Returns None for
        non-literals (labels, expressions, register names)."""
        s = text.strip()
        if not s:
            return None
        sign = 1
        if s.startswith("-"):
            sign = -1
            s = s[1:].strip()
        elif s.startswith("+"):
            s = s[1:].strip()
        if not s:
            return None
        try:
            if s.lower().startswith("0x"):
                return sign * int(s, 16)
            if s.isdigit():
                return sign * int(s, 10)
        except ValueError:
            return None
        return None

    def _addr_refs_reg(self, addr: str, reg32: str) -> bool:
        """Does the memory operand `addr` (without size prefix)
        reference reg32 or its sub-aliases?"""
        return self._references_reg_family(addr, reg32)

    @staticmethod
    def _is_imm_to_eax(line: Line) -> bool:
        """Is this a `mov eax, <constant-expr>` (immediate or label,
        but not a memory deref or register source)?"""
        return _is_imm_or_label_to_eax(line)

    @staticmethod
    def _matches_imm_store(
        instrs: list[tuple[int, Line]],
        a_line: Line,
    ) -> bool:
        """Verify line B (`mov [addr], eax`) and line C (`mov eax, X`).

        Line C must be a *fresh* EAX overwrite — its source operand
        must not reference EAX/AX/AL/AH. Otherwise the instruction is
        a self-RMW (e.g. ``mov eax, [eax]``) and depends on EAX's
        prior value. Dropping line A (``mov eax, IMM``) in that case
        would change line C's effective behavior because EAX would no
        longer hold IMM.
        """
        b_line = instrs[0][1]
        c_line = instrs[1][1]
        if b_line.op != "mov":
            return False
        parts_b = _operands_split(b_line.operands)
        if parts_b is None:
            return False
        b_dest, b_src = parts_b
        # Destination must be a memory ref.
        if not b_dest.startswith("["):
            return False
        # Source must be EAX (not a sub-register — those need narrower
        # store widths and a different rewrite).
        if b_src.strip().lower() != "eax":
            return False
        # Line C must be a fresh EAX overwrite (any of mov/lea/movsx/
        # movzx/pop/xor-self) whose source is EAX-independent.
        if not PeepholeOptimizer._is_fresh_eax_write(c_line):
            return False
        return True

    def _pass_leave_collapse(self, lines: list[Line]) -> list[Line]:
        """Replace the function epilogue's `mov esp, ebp; pop ebp` with
        the single-byte `leave` instruction. NASM emits both encodings
        equivalently; `leave` is 1 byte (0xC9) vs the 3-byte combination
        (0x89 0xEC 0x5D)."""
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr"
                    and line.op == "mov"
                    and line.operands.replace(" ", "").lower() == "esp,ebp"):
                # Look at the next non-blank/non-comment instr.
                j = i + 1
                while j < len(lines) and lines[j].kind in ("blank", "comment"):
                    j += 1
                if (j < len(lines)
                        and lines[j].kind == "instr"
                        and lines[j].op == "pop"
                        and lines[j].operands.strip().lower() == "ebp"):
                    # Build a `leave` line, preserving the original
                    # leading whitespace from the `mov esp, ebp` line.
                    leading_ws = re.match(r"^\s*", line.raw).group(0)
                    leave_raw = f"{leading_ws}leave"
                    out.append(Line(raw=leave_raw, kind="instr",
                                    op="leave", operands=""))
                    self.stats["leave_collapse"] = (
                        self.stats.get("leave_collapse", 0) + 1
                    )
                    i = j + 1
                    continue
            out.append(line)
            i += 1
        return out

    # Binops that accept `OP eax, imm32` directly (no ecx round-trip).
    # `imul eax, ecx` becomes `imul eax, eax, IMM` — NASM accepts both
    # `imul eax, IMM` (which encodes as `imul eax, eax, IMM`) and the
    # explicit three-operand form. We use the two-operand form below
    # so the rewrite is uniform across all ops in this set.
    _IMM_BINOP_OPS: frozenset[str] = frozenset({
        "add", "sub", "and", "or", "xor", "cmp", "test",
        "imul", "adc", "sbb",
    })

    def _pass_imm_binop_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse `mov ecx, <src>; OP eax, ecx` to `OP eax, <src>`.

        Match (2 consecutive instr lines):
            mov     ecx, <src>           ; immediate, label, mem, or reg
            <OP>    eax, ecx

        Replace with:
            <OP>    eax, <src>

        Witness: the instruction after the OP must overwrite ECX
        (full mov / pop / xor reg, reg / lea / call) OR not reference
        ECX (or CX/CL/CH aliases). The codegen always reloads ECX
        before each use, so this fires for every `eax OP <something>`
        shape with a transferred ECX.

        Sources we support:
        - Immediate / label: `mov ecx, 50` / `mov ecx, _glob`.
        - Memory: `mov ecx, [ebp - 4]` / `mov ecx, [_glob]`.
        - Register: `mov ecx, ebx`.

        We don't support sources that contain ECX/CX/CL/CH (would
        self-reference after collapse) or ESP-relative memory if the
        OP itself reads ESP (none of the binops here do, but be
        defensive — only forbid ESP-relative if the OP also reads
        flags, which isn't a real case here).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr"
                    and line.op == "mov"
                    and self._is_simple_to_ecx(line)):
                # Find the next instruction line (the binop).
                instrs = self._next_n_instrs(lines, i + 1, 1)
                if instrs is not None:
                    [(b_idx, b_line)] = instrs
                    if (b_line.op in self._IMM_BINOP_OPS
                            and self._is_eax_ecx_binop(b_line)
                            and self._reg_dead_after(
                                lines, b_idx + 1, "ecx"
                            )):
                        # Build the rewritten op line.
                        parts_a = _operands_split(line.operands)
                        assert parts_a is not None
                        _, src_imm = parts_a
                        leading_ws = re.match(r"^\s*", b_line.raw).group(0)
                        new_raw = (
                            f"{leading_ws}{b_line.op:<8}eax, {src_imm}"
                        )
                        new_line = Line(
                            raw=new_raw, kind="instr", op=b_line.op,
                            operands=f"eax, {src_imm}",
                        )
                        out.append(new_line)
                        self.stats["imm_binop_collapse"] = (
                            self.stats.get("imm_binop_collapse", 0) + 1
                        )
                        i = b_idx + 1
                        continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_simple_to_ecx(line: Line) -> bool:
        """`mov ecx, <src>` where the source is an immediate, label,
        memory operand, or register that doesn't self-reference ECX.

        Excludes:
        - Sources containing ECX / CX / CL / CH (post-rewrite would
          read the OP's destination operand, changing semantics).
        """
        if line.kind != "instr" or line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        if dest.lower() != "ecx":
            return False
        # No ECX self-reference in the source.
        for alias in ("ecx", "cx", "cl", "ch"):
            if _references_register(src, alias):
                return False
        return True

    @staticmethod
    def _is_eax_ecx_binop(line: Line) -> bool:
        """`<OP> eax, ecx` shape — the only one we can fold a preceding
        `mov ecx, IMM` into."""
        if line.kind != "instr":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        return dest.lower() == "eax" and src.lower() == "ecx"

    @staticmethod
    def _ecx_dead_after(line: Line) -> bool:
        """Witness that ECX is dead at this point. Conservative: either
        the instruction overwrites ECX entirely, or it doesn't reference
        ECX at all. Anything that reads ECX (e.g. `cmp ebx, ecx`,
        `mov [ecx], eax`, `add ebx, ecx`) defeats the rewrite."""
        if line.kind != "instr":
            # A label / directive / data line means we can't see what
            # comes after — be conservative and assume ECX may be live
            # (some other path could reach the label).
            return False
        # Full ECX overwrite: mov/pop/lea ecx, ...; xor ecx, ecx.
        if line.op == "mov":
            parts = _operands_split(line.operands)
            if parts and parts[0].lower() == "ecx":
                return True
        if line.op == "pop" and line.operands.strip().lower() == "ecx":
            return True
        if line.op == "lea":
            parts = _operands_split(line.operands)
            if parts and parts[0].lower() == "ecx":
                return True
        if line.op == "xor":
            parts = _operands_split(line.operands)
            if (parts and parts[0].lower() == "ecx"
                    and parts[1].strip().lower() == "ecx"):
                return True
        # `call X` clobbers caller-saved (EAX/ECX/EDX) per cdecl.
        if line.op == "call":
            return True
        # `ret` ends the function — ECX no longer matters.
        if line.op == "ret":
            return True
        # Otherwise: instruction must not reference ECX or any of its
        # sub-registers (CX, CL, CH). The original sequence wrote the
        # IMM into all bits of ECX before the binop, so a downstream
        # read of CL etc. saw IMM-low-bits — but post-collapse, ECX
        # is whatever the prior code left it. Different value → unsafe.
        for alias in ("ecx", "cx", "cl", "ch"):
            if _references_register(line.operands, alias):
                return False
        return True

    # Instructions that clobber EFLAGS (any of CF/PF/AF/ZF/SF/OF).
    # Source: Intel SDM Volume 2 — instructions whose "Flags Affected"
    # section lists at least one flag as written. We use this to know
    # when a prior `xor eax, eax`'s flag write becomes irrelevant.
    _FLAG_CLOBBERING_OPS: frozenset[str] = frozenset({
        "add", "sub", "cmp", "inc", "dec", "and", "or", "xor", "neg",
        "test", "shl", "shr", "sar", "rol", "ror", "rcl", "rcr",
        "shld", "shrd", "imul", "mul", "idiv", "div", "adc", "sbb",
        "bsr", "bsf", "bt", "btc", "btr", "bts", "popf", "popfd",
        "sahf", "stc", "clc", "cmc", "std", "cld", "sti", "cli",
    })

    # Instructions that READ EFLAGS — would observe our `xor`'s
    # spurious flag write. Source: Intel SDM Vol 2 "Flags Tested".
    _FLAG_READING_OPS: frozenset[str] = frozenset({
        # Conditional branches: jcc family
        "ja", "jae", "jb", "jbe", "jc", "jcxz", "je", "jecxz", "jg",
        "jge", "jl", "jle", "jna", "jnae", "jnb", "jnbe", "jnc",
        "jne", "jng", "jnge", "jnl", "jnle", "jno", "jnp", "jns",
        "jnz", "jo", "jp", "jpe", "jpo", "js", "jz",
        # Conditional sets
        "seta", "setae", "setb", "setbe", "setc", "sete", "setg",
        "setge", "setl", "setle", "setna", "setnae", "setnb",
        "setnbe", "setnc", "setne", "setng", "setnge", "setnl",
        "setnle", "setno", "setnp", "setns", "setnz", "seto", "setp",
        "setpe", "setpo", "sets", "setz",
        # Conditional moves
        "cmova", "cmovae", "cmovb", "cmovbe", "cmovc", "cmove",
        "cmovg", "cmovge", "cmovl", "cmovle", "cmovna", "cmovnae",
        "cmovnb", "cmovnbe", "cmovnc", "cmovne", "cmovng", "cmovnge",
        "cmovnl", "cmovnle", "cmovno", "cmovnp", "cmovns", "cmovnz",
        "cmovo", "cmovp", "cmovpe", "cmovpo", "cmovs", "cmovz",
        # Loop-with-condition
        "loope", "loopne", "loopz", "loopnz",
        # Read CF
        "adc", "sbb", "rcl", "rcr",
        # Save flags
        "pushf", "pushfd", "lahf", "into",
    })

    def _pass_mov_zero_to_xor(self, lines: list[Line]) -> list[Line]:
        """Replace `mov reg, 0` with `xor reg, reg` for any 32-bit
        general-purpose register. Saves 3 bytes per match (5-byte
        mov-imm32 → 2-byte xor reg, reg).

        Conservative flag analysis: scan forward up to ~10 instructions
        from the mov. If a flag-CLOBBERING op appears before any
        flag-READING op, the rewrite is safe. If a flag-reading op
        appears first, OR we hit a label / unconditional jump before
        either, skip — the new flags from xor might be observed.

        Special case: `ret` is treated as safe (function return doesn't
        propagate flags semantically — flags are caller-saved by
        convention; the System V i386 ABI doesn't preserve them
        across calls anyway).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            reg = self._mov_reg_zero(line)
            if reg is not None and self._flags_safe_after(lines, i + 1):
                leading_ws = re.match(r"^\s*", line.raw).group(0)
                new_raw = f"{leading_ws}xor     {reg}, {reg}"
                new_line = Line(
                    raw=new_raw, kind="instr", op="xor",
                    operands=f"{reg}, {reg}",
                )
                out.append(new_line)
                self.stats["mov_zero_to_xor"] = (
                    self.stats.get("mov_zero_to_xor", 0) + 1
                )
                i += 1
                continue
            out.append(line)
            i += 1
        return out

    def _pass_mov_neg_one_to_or(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace `mov reg, -1` (or its NASM-equivalent `mov reg,
        4294967295`) with `or reg, -1` for any 32-bit GP register.
        Saves 2 bytes per match (5-byte `mov reg, imm32` → 3-byte
        `or reg, imm8` since -1 fits in a sign-extended imm8).

        `or reg, -1` produces `reg | 0xFFFFFFFF = 0xFFFFFFFF` for ALL
        prior reg values, including undefined / poisoned. So the
        rewrite is value-equivalent regardless of reg's prior state.

        Flag effects DIFFER: `mov` preserves flags; `or` clears OF
        and CF, sets SF=1 (since result is negative), ZF=0, PF
        based on low byte. So we require flags to be dead after
        the instruction (same as mov_zero_to_xor's check).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            reg = self._mov_reg_neg_one(line)
            if (
                reg is not None
                and self._flags_safe_after(lines, i + 1)
            ):
                leading_ws = re.match(r"^\s*", line.raw).group(0)
                new_raw = f"{leading_ws}or      {reg}, -1"
                new_line = Line(
                    raw=new_raw, kind="instr", op="or",
                    operands=f"{reg}, -1",
                )
                out.append(new_line)
                self.stats["mov_neg_one_to_or"] = (
                    self.stats.get("mov_neg_one_to_or", 0) + 1
                )
                i += 1
                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _mov_reg_neg_one(line: Line) -> str | None:
        """Return the register name if this is `mov <gp32-reg>, -1`
        (or `mov reg, 4294967295`), else None.

        NASM accepts both forms; the codegen emits both depending on
        whether the literal was explicitly negative or unsigned-cast.
        """
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        src_text = src.strip()
        # Accept literal -1 or 4294967295 (= 0xFFFFFFFF).
        if src_text not in ("-1", "4294967295", "0xFFFFFFFF",
                            "0xffffffff", "0xFFFFffff"):
            # Also handle hex forms with full 32-bit width.
            if not (
                src_text.lower() == "0xffffffff"
            ):
                return None
        dest_lower = dest.lower()
        if dest_lower in {"eax", "ebx", "ecx", "edx",
                          "esi", "edi", "ebp"}:
            return dest_lower
        return None

    @staticmethod
    def _mov_reg_zero(line: Line) -> str | None:
        """Return the register name if this is `mov <gp32-reg>, 0`,
        else None.

        Recognizes both decimal `0` and hex `0x0` / `0x00000000` forms.
        The codegen sometimes emits the hex form (notably the long-long
        path's high-half clear `mov edx, 0x00000000`).
        """
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        src_text = src.strip()
        if src_text != "0":
            # Hex zero: 0x followed by all-zero digits.
            low = src_text.lower()
            if not (
                low.startswith("0x")
                and len(low) > 2
                and all(c == "0" for c in low[2:])
            ):
                return None
        dest_lower = dest.lower()
        if dest_lower in {"eax", "ebx", "ecx", "edx",
                          "esi", "edi", "ebp"}:
            return dest_lower
        return None

    def _flags_safe_after(self, lines: list[Line], start_idx: int) -> bool:
        """Scan forward up to ~20 instructions. Return True if a
        flag-CLOBBERING op or function exit appears before any
        flag-READING op.

        Cross-label scanning relies on a uc386 codegen invariant:
        every entry to a labeled block re-establishes flag state via
        an explicit `cmp` / `test` / arithmetic op before any
        conditional branch. The optimizer's input is always uc386
        output (libc is appended later), so this invariant holds.
        """
        scanned = 0
        j = start_idx
        while j < len(lines) and scanned < 20:
            ln = lines[j]
            if ln.kind in ("blank", "comment", "label"):
                # Label crossings are safe under the codegen invariant.
                j += 1
                continue
            if ln.kind in ("directive", "data"):
                # End of function (or section change). Treat as safe —
                # we're past any flag-reading code in this function.
                return True
            if ln.kind != "instr":
                return False
            scanned += 1
            if ln.op == "ret" or ln.op in {"iret", "iretd", "retf", "retn"}:
                return True
            if ln.op == "jmp":
                # Codegen jmps lead to labels whose first instruction
                # re-establishes flag state. Keep scanning past.
                j += 1
                continue
            if ln.op == "call":
                # Calls clobber flags via the callee. Safe.
                return True
            if ln.op in self._FLAG_READING_OPS:
                return False
            if ln.op in self._FLAG_CLOBBERING_OPS:
                return True
            # Otherwise: flag-neutral instr (mov / lea / push / pop /
            # nop / xchg / etc.) — keep scanning.
            j += 1
        return False

    def _pass_store_load_collapse(self, lines: list[Line]) -> list[Line]:
        """Drop the redundant load after a store-then-load of the
        same register to the same address.

        Match (2 consecutive instr lines):
            mov     [<addr>], <reg>
            mov     <reg>, [<addr>]

        Replace with just the store. The register still holds its
        stored value, so the load is a no-op.

        Address comparison is textual (after stripping whitespace),
        which is correct: NASM's `[ebp - 4]` always renders the same
        way for the same operand. The codegen always emits stores
        and loads with matching syntax.

        Skip when the address is ESP-relative — pushes/pops that
        might come between... wait, the pattern requires the two
        instructions to be CONSECUTIVE (next_n_instrs already skips
        blanks/comments only, not instrs), so there's nothing
        between them. ESP-relative is safe.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            store_info = self._mov_reg_to_mem(line)
            if store_info is not None:
                addr, reg = store_info
                instrs = self._next_n_instrs(lines, i + 1, 1)
                if instrs is not None:
                    [(b_idx, b_line)] = instrs
                    load_info = self._mov_mem_to_reg(b_line)
                    if (load_info is not None
                            and load_info == (reg, addr)):
                        out.append(line)
                        self.stats["store_load_collapse"] = (
                            self.stats.get("store_load_collapse", 0) + 1
                        )
                        i = b_idx + 1
                        continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _mov_reg_to_mem(line: Line) -> tuple[str, str] | None:
        """Return (addr, reg) if this is `mov [<addr>], <reg>`, else
        None. Reg must be a 32-bit gp register."""
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        if not dest.startswith("["):
            return None
        # Source must be a 32-bit gp register (not a sub-byte alias —
        # sub-byte stores have different widths).
        src_lower = src.lower()
        if src_lower not in {"eax", "ebx", "ecx", "edx",
                              "esi", "edi", "ebp", "esp"}:
            return None
        # Normalize address whitespace for matching.
        return (re.sub(r"\s+", "", dest), src_lower)

    @staticmethod
    def _mov_mem_to_reg(line: Line) -> tuple[str, str] | None:
        """Return (reg, addr) if this is `mov <reg>, [<addr>]`, else
        None."""
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        dest_lower = dest.lower()
        if dest_lower not in {"eax", "ebx", "ecx", "edx",
                               "esi", "edi", "ebp", "esp"}:
            return None
        if not src.startswith("["):
            return None
        return (dest_lower, re.sub(r"\s+", "", src))

    def _pass_pop_op_chain_retarget(
        self, lines: list[Line]
    ) -> list[Line]:
        """Sister of `right_operand_retarget` for the short-form
        commutative tail. Match (looking back from
        `pop ecx; OP_commutative eax, ecx`):

            push    eax                   ← LHS save
            <fresh EAX write>             ← chain instr 1
            <EAX read-or-RMW>*            ← chain instrs 2..N
            pop     ecx                   ← restore LHS to ECX
            <OP>    eax, ecx              ← commutative OP

        Replace with:

            <retargeted chain instr 1>    ← dest ECX
            <retargeted chain instrs 2..N>
            <OP>    eax, ecx              ← unchanged; LHS in EAX,
                                            RHS in ECX (commutative)

        Drops push + pop = 2 bytes per match. Common in compound
        assigns / additive chains where the codegen emits the
        short-form save/restore (no intervening `mov ecx, eax`)
        because the OP is commutative.

        Conditions are the same as `right_operand_retarget` except
        the terminal pattern is `pop ecx; OP eax, ecx` instead of
        `mov ecx, eax; pop eax`.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            # Match `pop ecx`.
            if not (
                line.kind == "instr"
                and line.op == "pop"
                and line.operands.strip().lower() == "ecx"
            ):
                i += 1
                continue
            # Next instr must be `OP_commutative eax, ecx`.
            instrs_after = self._next_n_instrs(out, i + 1, 1)
            if instrs_after is None:
                i += 1
                continue
            op_line = instrs_after[0][1]
            if op_line.op not in PeepholeOptimizer._COMMUTATIVE_BINOPS:
                i += 1
                continue
            opp = _operands_split(op_line.operands)
            if opp is None:
                i += 1
                continue
            if (
                opp[0].strip().lower() != "eax"
                or opp[1].strip().lower() != "ecx"
            ):
                i += 1
                continue
            # Backward scan to find the chain.
            chain_info = self._find_rhs_chain(out, i)
            if chain_info is None:
                i += 1
                continue
            push_idx, chain_indices = chain_info
            # Retarget every chain instruction.
            for k in chain_indices:
                out[k] = self._retarget_instr_eax_to_ecx(out[k])
            # Drop pop_idx (= i) and push_idx (back to front).
            out.pop(i)
            out.pop(push_idx)
            self.stats["pop_op_chain_retarget"] = (
                self.stats.get("pop_op_chain_retarget", 0) + 1
            )
            i = push_idx
            continue
        return out

    def _pass_pop_cmp_chain_retarget(
        self, lines: list[Line]
    ) -> list[Line]:
        """Sister of `pop_op_chain_retarget` for cmp followed by a
        ZF-only Jcc. Match (looking back from
        `pop ecx; cmp eax, ecx; <ZF jcc>`):

            push    eax                   ← LHS save
            <fresh EAX write>             ← chain instr 1
            <EAX read-or-RMW>*            ← chain instrs 2..N
            pop     ecx                   ← restore LHS to ECX
            cmp     eax, ecx              ← compare RHS vs LHS
            <ZF Jcc>                      ← je/jne/jz/jnz/...

        Replace with:

            <retargeted chain instr 1>    ← dest ECX
            <retargeted chain instrs 2..N>
            cmp     eax, ecx              ← unchanged; LHS still in EAX
                                            (we didn't push), RHS in ECX
            <ZF Jcc>                      ← unchanged

        Drops push + pop = 2 bytes per match.

        Why this works: the original computes `cmp computed,
        orig_eax`; the rewrite computes `cmp orig_eax, computed`.
        ZF (= operands equal) is symmetric, so subsequent
        je/jne/sete/setne/etc. produce the same branch. SF/CF/OF
        flip — the ZF-only Jcc constraint excludes any consumer
        that reads them.

        Common shape: equality comparisons in if/while conditions
        where the rhs is a multi-instruction expression (constant
        load + shift, label-arithmetic, etc.) that survives
        binop_collapse / right_operand_retarget unchanged because
        the trailing pattern is `pop ecx; cmp eax, ecx; je`
        (no `mov ecx, eax` and no commutative OP).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            # Match `pop ecx`.
            if not (
                line.kind == "instr"
                and line.op == "pop"
                and line.operands.strip().lower() == "ecx"
            ):
                i += 1
                continue
            # Next instr must be `cmp eax, ecx`.
            instrs_after = self._next_n_instrs(out, i + 1, 2)
            if instrs_after is None:
                i += 1
                continue
            cmp_idx, cmp_line = instrs_after[0]
            if cmp_line.op != "cmp":
                i += 1
                continue
            cmpp = _operands_split(cmp_line.operands)
            if cmpp is None:
                i += 1
                continue
            if (
                cmpp[0].strip().lower() != "eax"
                or cmpp[1].strip().lower() != "ecx"
            ):
                i += 1
                continue
            # Following instr (the consumer) must be a ZF-only Jcc
            # (or other ZF-only flag reader) — and there must be no
            # other flag reader before flags get re-set. We check
            # the immediate next instruction conservatively.
            jcc_idx, jcc_line = instrs_after[1]
            if jcc_line.op not in (
                PeepholeOptimizer._ZF_ONLY_FLAG_READERS
            ):
                i += 1
                continue
            # Also require flags safe past the jcc — i.e., no other
            # flag-reader (non-ZF) sees the cmp's flags. In our
            # codegen the next flag-setter usually follows quickly,
            # but be defensive: walk forward from jcc_idx + 1 until
            # the first flag-clobber or flag-reader; if any reader
            # is non-ZF-only, bail.
            if not self._flags_safe_zf_only_after(
                out, jcc_idx + 1
            ):
                i += 1
                continue
            # Backward scan to find the chain.
            chain_info = self._find_rhs_chain(out, i)
            if chain_info is None:
                i += 1
                continue
            push_idx, chain_indices = chain_info
            # Retarget every chain instruction.
            for k in chain_indices:
                out[k] = self._retarget_instr_eax_to_ecx(out[k])
            # Drop pop_idx (= i) and push_idx (back to front).
            out.pop(i)
            out.pop(push_idx)
            self.stats["pop_cmp_chain_retarget"] = (
                self.stats.get("pop_cmp_chain_retarget", 0) + 1
            )
            i = push_idx
            continue
        return out

    def _flags_safe_zf_only_after(
        self, lines: list[Line], start: int,
    ) -> bool:
        """Scan forward from `start`. Until the first flag-clobbering
        instruction (or fence — label/jmp/ret/call), every flag-reading
        instruction we encounter must be in `_ZF_ONLY_FLAG_READERS`.
        Returns True if so (or if no flag readers seen before a
        clobber/fence).

        Used to verify it's safe to swap operand order in a `cmp` —
        only ZF is invariant under operand swap; other flags differ.
        """
        bound = min(len(lines), start + 20)
        for j in range(start, bound):
            ln = lines[j]
            if ln.kind != "instr":
                # Label/directive/data — control-flow merge or section
                # boundary; assume safe (caller's code didn't make
                # assumptions across the boundary).
                return True
            op = ln.op
            # Fences: ret/leave/call (callee may set/use flags;
            # cdecl says caller-saved flags are clobbered) — boundary.
            if op in ("ret", "retn", "retf", "leave", "iret", "iretd"):
                return True
            if op == "call":
                # Call clobbers flags by ABI assumption.
                return True
            # Unconditional jmp — boundary.
            if op == "jmp":
                return True
            # Flag-readers: jcc, setcc, cmovcc, adc, sbb, rcl, rcr.
            if op in PeepholeOptimizer._FLAG_READING_OPS:
                if op not in PeepholeOptimizer._ZF_ONLY_FLAG_READERS:
                    return False
                # ZF-only reader: OK, keep scanning (we may have a
                # chain je/jne/...). Continue.
                continue
            # Flag-clobbers: any arithmetic / logical / cmp / test /
            # shift would re-set flags. If we hit one before a non-
            # ZF reader, we're safe.
            if op in PeepholeOptimizer._FLAG_CLOBBERING_OPS:
                return True
        return True

    def _pass_right_operand_retarget(self, lines: list[Line]) -> list[Line]:
        """Retarget the binop RHS chain from EAX to ECX, then drop
        the entire `push eax / pop eax / mov ecx, eax` save/restore
        scaffold — EAX is preserved naturally because the retargeted
        chain only writes ECX.

        Match (looking back from `mov ecx, eax; pop eax`):
            push    eax                   ← LHS save (chain start marker)
            <fresh EAX write>             ← chain instr 1: mov/lea/pop/xor
            <EAX read-or-RMW>*            ← chain instrs 2..N: any op
                                            with dest=eax that doesn't
                                            read ECX/CX/CL/CH.
            mov     ecx, eax              ← redundant copy
            pop     eax                   ← restore LHS (also redundant)

        Replace with:
            <retargeted chain instr 1>    ← dest changed to ECX
            <retargeted chain instrs 2..N> ← dest + [eax]→[ecx] in src

        Saves 4 bytes per binop with non-trivial RHS: 1 (push eax) +
        2 (mov ecx, eax) + 1 (pop eax) = 4. The chain's running value
        now lives in ECX from the start; EAX's LHS value is preserved
        across the chain because no chain instruction touches EAX.

        Conditions:
        - All instructions in the chain must write EAX as their dest.
        - Source operands must not reference ECX/CX/CL/CH (else the
          retargeted version would self-reference).
        - The chain's first instruction must be a "fresh write"
          (mov / lea / pop / xor reg+reg / call) — otherwise the
          chain depends on EAX's prior value (the LHS), which post-
          retarget wouldn't be in ECX.
        - Bound the backward scan at 12 instructions (any reasonable
          RHS chain).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            if not (line.kind == "instr"
                    and line.op == "mov"
                    and line.operands.replace(" ", "").lower()
                        == "ecx,eax"):
                i += 1
                continue
            # Next instr must be `pop eax`.
            instrs_after = self._next_n_instrs(out, i + 1, 1)
            if (instrs_after is None
                    or instrs_after[0][1].op != "pop"
                    or instrs_after[0][1].operands.strip().lower()
                        != "eax"):
                i += 1
                continue
            pop_idx = instrs_after[0][0]
            # Backward scan to find the chain.
            chain_info = self._find_rhs_chain(out, i)
            if chain_info is None:
                i += 1
                continue
            push_idx, chain_indices = chain_info
            # Retarget every chain instruction (dest + src EAX → ECX
            # for all aliases).
            for k in chain_indices:
                out[k] = self._retarget_instr_eax_to_ecx(out[k])
            # Drop in this order (back to front so indices stay valid):
            # pop_idx > i > push_idx (always true).
            out.pop(pop_idx)
            out.pop(i)
            out.pop(push_idx)
            self.stats["right_operand_retarget"] = (
                self.stats.get("right_operand_retarget", 0) + 1
            )
            # Next iteration: start scanning from where push was
            # (now occupied by the first chain instruction).
            i = push_idx
            continue
        return out

    @staticmethod
    def _find_rhs_chain(
        out: list[Line], mov_ecx_idx: int,
    ) -> tuple[int, list[int]] | None:
        """Backward-scan from mov_ecx_idx looking for a chain of
        EAX-writing instructions terminated by `push eax`. Return
        (push_eax_index, chain_indices_in_source_order), or None."""
        chain: list[int] = []
        j = mov_ecx_idx - 1
        bound = max(0, mov_ecx_idx - 12)
        while j >= bound:
            ln = out[j]
            if ln.kind in ("blank", "comment"):
                j -= 1
                continue
            if ln.kind != "instr":
                return None  # label/directive — basic block boundary
            # `push eax` is the chain-start marker.
            if (ln.op == "push"
                    and ln.operands.strip().lower() == "eax"):
                if not chain:
                    return None  # empty chain — nothing to retarget
                # Chain must start with a fresh EAX write (the first
                # chain instr in source order is at chain[-1] since
                # we built it backward).
                first_ln = out[chain[-1]]
                if not PeepholeOptimizer._is_fresh_eax_write(first_ln):
                    return None
                return (j, list(reversed(chain)))
            # Otherwise this should be a chain instruction.
            if not PeepholeOptimizer._is_chain_eax_writer(ln):
                return None
            chain.append(j)
            j -= 1
        return None

    @staticmethod
    def _is_chain_eax_writer(line: Line) -> bool:
        """Is this an instruction that writes EAX as its destination
        and doesn't read ECX/CX/CL/CH? Eligible for retargeting to
        ECX. Also rejects ESP-relative sources because the surrounding
        push/pop that bracket the chain affect ESP — dropping them
        changes [esp + N] semantics."""
        if line.kind != "instr":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            # Single-operand or no-operand — must inspect by op.
            # neg/not/imul (one-op form) have a single operand.
            if line.op in {"neg", "not", "inc", "dec",
                           "mul", "imul", "div", "idiv"}:
                operand = line.operands.strip().lower()
                if operand == "eax":
                    # Unary on EAX — writes EAX.
                    if PeepholeOptimizer._references_ecx(operand):
                        return False
                    return True
            return False
        dest, src = parts
        dest_lower = dest.lower()
        if dest_lower != "eax":
            return False
        # Source mustn't reference ECX. (Sources that reference [eax]
        # are fine — they'll be rewritten in retarget.)
        if PeepholeOptimizer._references_ecx(src):
            return False
        # Source mustn't be ESP-relative. The retarget drops the
        # push/pop scaffold that surrounds the chain; if the chain
        # reads `[esp + N]`, dropping the push (which decremented
        # ESP by 4) changes which byte is read. Conservative skip.
        if _references_register(src, "esp"):
            return False
        # cmp/test read EAX without writing — exclude.
        if line.op in {"cmp", "test"}:
            return False
        # `xchg eax, X` swaps — read+write both. Skip.
        if line.op == "xchg":
            return False
        # `cdq` writes EDX based on EAX's sign. EAX unchanged. Skip.
        if line.op == "cdq":
            return False
        return True

    @staticmethod
    def _is_fresh_eax_write(line: Line) -> bool:
        """Does this instruction write EAX without reading it first
        (a fresh write rather than a read-modify-write)? The source
        operand must NOT reference EAX/AX/AL/AH — otherwise the
        instruction depends on EAX's prior value and isn't fresh."""
        if line.kind != "instr":
            return False
        if line.op in {"mov", "lea", "movsx", "movzx"}:
            parts = _operands_split(line.operands)
            if parts and parts[0].lower() == "eax":
                src = parts[1]
                if PeepholeOptimizer._references_eax_low_high(src):
                    return False
                return True
            return False
        if line.op == "pop" and line.operands.strip().lower() == "eax":
            return True
        if line.op == "xor":
            parts = _operands_split(line.operands)
            if (parts and parts[0].lower() == "eax"
                    and parts[1].strip().lower() == "eax"):
                return True
            return False
        if line.op == "call":
            # Direct call clobbers EAX with the callee's return value.
            # No EAX read at the call site (cdecl args are on stack).
            # Indirect call `call eax` reads EAX — exclude.
            operand = line.operands.strip().lower()
            if operand in {"eax", "ax", "al", "ah"}:
                return False
            if "[" in operand and "eax" in operand:
                # `call [eax + N]` reads EAX as part of the address.
                return False
            return True
        return False

    @staticmethod
    def _references_ecx(text: str) -> bool:
        """Does `text` reference ECX or any of its aliases (CX/CL/CH)?"""
        for alias in ("ecx", "cx", "cl", "ch"):
            if _references_register(text, alias):
                return True
        return False

    @staticmethod
    def _references_eax_low_high(text: str) -> bool:
        """Does `text` reference EAX or any of its aliases (AX/AL/AH)?"""
        for alias in ("eax", "ax", "al", "ah"):
            if _references_register(text, alias):
                return True
        return False

    # Mapping from EAX-family register names to ECX-family equivalents.
    # Order matters for the substitution: `eax` first so we don't
    # accidentally match the `ax` inside `eax`. (Word-boundary `\b`
    # would also handle this, but explicit ordering is safer.)
    _EAX_TO_ECX_ALIASES: list[tuple[str, str]] = [
        ("eax", "ecx"),
        ("ax", "cx"),
        ("al", "cl"),
        ("ah", "ch"),
    ]

    def _pass_store_chain_retarget(
        self, lines: list[Line]
    ) -> list[Line]:
        """Multi-instruction store_collapse via chain retargeting.

        Pattern (forward from `push eax`):
            push    eax                ← save address
            <chain that writes EAX (fresh-write at start, no ECX-src refs)>
            pop     ecx                ← restore address into ECX
            mov     [ecx], eax         ← store

        Replace with:
            <chain' (rewritten so EAX-family → ECX-family throughout)>
            mov     [eax], ecx         ← store: dest=[address-in-eax], src=value-in-ecx

        Saves 2 bytes per match (drops push + pop, no addition since the
        retargeted chain has the same length).

        Conditions:
        - Chain length 1+ instructions, all writing EAX as dest.
        - Chain first instr is a fresh-write (mov/lea/movsx/movzx/pop/xor self).
        - Chain instrs don't reference ECX (ECX is dead before; we're
          using it as the new running-value register).
        - Chain instrs don't reference ESP (the push shifted ESP).
        - EAX and ECX both dead after the store. Original post-state was
          EAX = chain result, ECX = address. Retargeted post-state is
          EAX = address, ECX = chain result. The two are different — if
          subsequent code reads either, retarget breaks semantics.

        The codegen typically emits this pattern for `*p = expr` where
        the chain is `expr` and the address `p` is on the stack. After
        the store the next statement starts fresh (clobbers EAX/ECX).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            if not (
                line.kind == "instr" and line.op == "push"
                and line.operands.strip().lower() == "eax"
            ):
                i += 1
                continue
            # Walk forward through chain instrs.
            chain_indices: list[int] = []
            j = i + 1
            max_chain = 12
            pop_idx: int | None = None
            store_idx: int | None = None
            while j < len(out) and len(chain_indices) < max_chain:
                ln = out[j]
                if ln.kind in ("blank", "comment"):
                    j += 1
                    continue
                if ln.kind != "instr":
                    break
                if (
                    ln.op == "pop"
                    and ln.operands.strip().lower() == "ecx"
                ):
                    # Check next instr is mov [ecx], eax.
                    k = j + 1
                    while (
                        k < len(out)
                        and out[k].kind in ("blank", "comment")
                    ):
                        k += 1
                    if (
                        k < len(out)
                        and out[k].kind == "instr"
                        and out[k].op == "mov"
                        and out[k].operands.replace(" ", "").lower()
                            == "[ecx],eax"
                    ):
                        pop_idx = j
                        store_idx = k
                    break
                # Chain instr: must write EAX, no ECX refs, no ESP refs.
                if not PeepholeOptimizer._is_chain_eax_writer(ln):
                    break
                chain_indices.append(j)
                j += 1
            if (
                pop_idx is None or store_idx is None
                or not chain_indices
            ):
                i += 1
                continue
            # Verify chain starts with fresh-write.
            first_ln = out[chain_indices[0]]
            if not PeepholeOptimizer._is_fresh_eax_write(first_ln):
                i += 1
                continue
            # Verify EAX and ECX dead after the store.
            if not self._reg_dead_after(out, store_idx + 1, "eax"):
                i += 1
                continue
            if not self._reg_dead_after(out, store_idx + 1, "ecx"):
                i += 1
                continue
            # Retarget chain instrs (dest + any [eax] in src → ECX).
            for k in chain_indices:
                out[k] = self._retarget_instr_eax_to_ecx(out[k])
            # Replace the store: mov [ecx], eax → mov [eax], ecx.
            store_line = out[store_idx]
            new_raw = re.sub(
                r"\[\s*ecx\s*\]\s*,\s*eax",
                "[eax], ecx",
                store_line.raw,
                count=1,
                flags=re.IGNORECASE,
            )
            new_store = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands="[eax], ecx",
            )
            out[store_idx] = new_store
            # Drop push (i) and pop (pop_idx). Drop in reverse order
            # so indices stay valid.
            out.pop(pop_idx)
            out.pop(i)
            self.stats["store_chain_retarget"] = (
                self.stats.get("store_chain_retarget", 0) + 1
            )
            # Continue from the position of the dropped push (now
            # pointing at the first chain instr).
        return out

    @staticmethod
    def _retarget_instr_eax_to_ecx(line: Line) -> Line:
        """Rewrite every EAX-family register reference in operands and
        raw line to the corresponding ECX-family register (eax→ecx,
        ax→cx, al→cl, ah→ch). The chain's running value lives in
        ECX after retarget; any `[eax]` becomes `[ecx]` for the same
        reason."""
        new_operands = line.operands
        new_raw = line.raw
        for old, new in PeepholeOptimizer._EAX_TO_ECX_ALIASES:
            new_operands = re.sub(
                rf"\b{old}\b", new, new_operands, flags=re.IGNORECASE
            )
            new_raw = re.sub(
                rf"\b{old}\b", new, new_raw, flags=re.IGNORECASE
            )
        return Line(
            raw=new_raw, kind=line.kind, op=line.op,
            operands=new_operands,
        )

    def _pass_cmp_zero_to_test(self, lines: list[Line]) -> list[Line]:
        """Replace `cmp <reg>, 0` with `test <reg>, <reg>`.

        Both forms set flags identically for the comparison-with-zero
        case:
        - ZF: 1 iff reg == 0
        - SF: 1 iff reg's high bit is set (signed-negative)
        - OF: 0 (no overflow possible)
        - CF: 0 (no borrow possible)
        - PF: parity of reg's low byte

        Subsequent JCCs (je/jne/jl/jge/js/jns/etc.) all see the same
        flag state, so the rewrite preserves semantics.

        Saves 1 byte per match: `cmp eax, 0` is 3 bytes (with imm8
        sign-extension); `test eax, eax` is 2 bytes.
        """
        out: list[Line] = []
        for line in lines:
            if (line.kind == "instr" and line.op == "cmp"):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    if (src.strip() == "0"
                            and dest.lower() in {"eax", "ebx", "ecx",
                                                 "edx", "esi", "edi",
                                                 "ebp", "esp"}):
                        leading_ws = re.match(r"^\s*", line.raw).group(0)
                        new_raw = (
                            f"{leading_ws}test    {dest}, {dest}"
                        )
                        out.append(Line(
                            raw=new_raw, kind="instr", op="test",
                            operands=f"{dest}, {dest}",
                        ))
                        self.stats["cmp_zero_to_test"] = (
                            self.stats.get("cmp_zero_to_test", 0) + 1
                        )
                        continue
            out.append(line)
        return out

    # Mapping from each 32-bit gp reg to its sub-register aliases
    # (16-bit and 8-bit). Used by dead-mov / liveness analyses.
    _REG_FAMILY: dict[str, frozenset[str]] = {
        "eax": frozenset({"eax", "ax", "al", "ah"}),
        "ebx": frozenset({"ebx", "bx", "bl", "bh"}),
        "ecx": frozenset({"ecx", "cx", "cl", "ch"}),
        "edx": frozenset({"edx", "dx", "dl", "dh"}),
        "esi": frozenset({"esi", "si"}),
        "edi": frozenset({"edi", "di"}),
        "ebp": frozenset({"ebp", "bp"}),
        "esp": frozenset({"esp", "sp"}),
    }

    # Reverse map: each alias → its 32-bit family. Built once at class
    # definition; used by the bulk-extraction helpers below to map a
    # regex match back to the canonical reg.
    _ALIAS_TO_FAMILY: dict[str, str] = {
        a: fam for fam, aliases in _REG_FAMILY.items() for a in aliases
    }

    # One pre-compiled regex per family — the hot _references_reg_family
    # path was the dominant cost (>90%) of multi-TU codegen with
    # peephole, calling re.search for each alias on each line on each
    # register on each pass. With this pre-compile, each family check is
    # a single regex search.
    _FAMILY_RE: dict[str, "re.Pattern"] = {
        fam: re.compile(rf"\b(?:{'|'.join(sorted(aliases, key=len, reverse=True))})\b", re.IGNORECASE)
        for fam, aliases in _REG_FAMILY.items()
    }

    # One pre-compiled regex matching ANY alias of any family. Used by
    # the bulk-extraction helper to compute "which families does this
    # text touch" in a single pass over the text.
    _ALL_ALIASES_RE: "re.Pattern" = re.compile(
        rf"\b(?:{'|'.join(sorted(_ALIAS_TO_FAMILY, key=len, reverse=True))})\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _families_referenced(text: str) -> set[str]:
        """Return the set of 32-bit reg families that ``text`` mentions
        (via any alias). One regex pass per call — much cheaper than
        looping over each family and each alias separately."""
        if not text:
            return set()
        out: set[str] = set()
        a2f = PeepholeOptimizer._ALIAS_TO_FAMILY
        for m in PeepholeOptimizer._ALL_ALIASES_RE.finditer(text):
            out.add(a2f[m.group().lower()])
        return out

    # libc internal helpers that read EAX as a non-cdecl input
    # register (the value to print/format). Without this set, the
    # liveness analyzer treats `call _printf_emit_dec` like any
    # cdecl call — assumes EAX is dead — and the dead-mov pass
    # then drops the `mov eax, [edi]` that loads the actual value
    # to print. Result: random values printed via `_printf_legacy`.
    _CALLS_READ_EAX: frozenset[str] = frozenset({
        "_printf_emit_dec",
        "_printf_emit_udec",
        "_printf_emit_hex",
        "_printf_emit_oct",
        "_printf_emit_double",
    })

    @staticmethod
    def _references_reg_family(text: str, reg32: str) -> bool:
        """Does `text` reference reg32 or any of its sub-aliases?"""
        if not text:
            return False
        return PeepholeOptimizer._FAMILY_RE[reg32].search(text) is not None

    @staticmethod
    def _is_pure_reg_write(line: Line, reg32: str) -> bool:
        """Does this instruction write reg32 in full WITHOUT reading
        any of its sub-aliases first? Pure writes:
        - mov reg, src where src doesn't reference reg-family
        - lea reg, [...] where the address doesn't reference reg-family
        - pop reg
        - xor reg, reg
        - movsx reg, src / movzx reg, src where src doesn't reference reg-family
        - call <label> (clobbers EAX/ECX/EDX in cdecl, doesn't read reg)
          for reg ∈ {eax, ecx, edx}
        """
        if line.kind != "instr":
            return False
        family = PeepholeOptimizer._REG_FAMILY[reg32]
        if line.op in {"mov", "lea", "movsx", "movzx"}:
            parts = _operands_split(line.operands)
            if not parts:
                return False
            dest, src = parts
            if dest.lower() != reg32:
                return False
            return not PeepholeOptimizer._references_reg_family(src, reg32)
        if line.op == "pop":
            if line.operands.strip().lower() == reg32:
                return True
            return False
        if line.op == "xor":
            parts = _operands_split(line.operands)
            if (parts and parts[0].lower() == reg32
                    and parts[1].strip().lower() == reg32):
                return True
            return False
        if line.op == "call":
            # Direct call clobbers caller-saved EAX/ECX/EDX.
            # Indirect call via reg-family reads it — exclude.
            operand = line.operands.strip().lower()
            if operand in family:
                return False
            if "[" in operand:
                # `call [reg + N]` reads reg.
                if PeepholeOptimizer._references_reg_family(operand, reg32):
                    return False
            # uc386's libc has helpers that READ EAX as input
            # (non-cdecl, register-passing — see _CALLS_READ_EAX).
            # Those calls are NOT a pure-write of EAX since they
            # use the prior value first.
            if reg32 == "eax" and operand in (
                PeepholeOptimizer._CALLS_READ_EAX
            ):
                return False
            if reg32 in {"eax", "ecx", "edx"}:
                return True
        # `cdq` sign-extends EAX into EDX — pure write to EDX (doesn't
        # read EDX's prior value, only EAX's).
        if line.op == "cdq" and reg32 == "edx":
            return True
        return False

    # Instructions known to reference registers IMPLICITLY (beyond
    # what's in the operand string). Used to know when our reg-
    # tracking peepholes need to bail. Examples: cdq reads EAX, idiv
    # reads EDX:EAX, lodsd reads ESI, rep* reads ECX.
    _IMPLICIT_REG_USERS: frozenset[str] = frozenset({
        # Sign/zero extension family
        "cdq", "cwde", "cwd", "cbw", "cqo", "cdqe",
        # One-operand mul/div implicitly read/write EDX:EAX
        "mul", "div", "idiv",  # (one-operand form)
        # String operations: read/write ESI, EDI, ECX
        "movsb", "movsw", "movsd",  # NB: movsd here is the
                                       # string instr, not SSE.
                                       # NASM uses `movsd` for both
                                       # but in 32-bit mode without
                                       # operands it's the string op.
        "lodsb", "lodsw", "lodsd",
        "stosb", "stosw", "stosd",
        "scasb", "scasw", "scasd",
        "cmpsb", "cmpsw", "cmpsd",
        # Repeat prefixes (read ECX)
        "rep", "repe", "repne", "repnz", "repz",
        # Loop instructions
        "loop", "loope", "loopne", "loopnz", "loopz",
        # XLAT reads AL, EBX
        "xlat", "xlatb",
        # Flag ops
        "pushf", "pushfd", "popf", "popfd", "lahf", "sahf",
        # bswap reads+writes a reg (single operand, but the operand
        # form is fine — we'll detect via the operand string)
        # enter/leave touch ESP/EBP
        "enter",  # leave is in the explicit-fine list
        # FPU instructions (don't touch gp regs by themselves)
        # but `fnstsw ax` writes AX
        "fnstsw", "fstsw",
        # int / iret / sysenter / sysexit — system calls
        "int", "into", "int1", "int3",
    })

    @staticmethod
    def _has_explicit_operands_only(line: Line) -> bool:
        """Does this instruction's behavior depend ONLY on its
        explicit operands (and not on implicit register references)?
        Used to know when scanning past an instruction without
        finding the tracked reg in the operand string is safe."""
        if line.op in PeepholeOptimizer._IMPLICIT_REG_USERS:
            return False
        # `fistp` and friends? FPU instructions don't touch gp regs.
        # An instruction starting with `f` (FPU) is fine.
        if line.op.startswith("f"):
            # Excludes the few `f*` we listed in _IMPLICIT_REG_USERS
            # like fnstsw — those are caught above.
            return True
        return True

    def _reg_dead_after(
        self, lines: list[Line], start_idx: int, reg32: str,
        visited_labels: frozenset[str] = frozenset(),
        depth: int = 0,
        treat_as_scratch: bool = False,
    ) -> bool:
        """Forward scan from start_idx. Return True if `reg32` is
        provably dead — a pure-write happens before any read.

        CFG-aware: at an unconditional `jmp X`, jumps to X (does not
        continue text-fallthrough). At a label by text-fallthrough,
        crosses transparently (the label may have other predecessors
        but they all reach the same first instruction, so dead-ness
        is determined by that instruction).

        The codegen invariant — registers are not assumed to carry
        values across labels — means cross-label scanning is safe
        for the reg liveness question.

        ``treat_as_scratch=True`` treats EAX/EDX as dead at ret.
        Use only when the caller knows the register is being used
        purely as scratch (e.g., a base-address holder for SIB
        arithmetic) and not as a return-value setup. The codegen
        explicitly emits long-long return setup via cdq + EDX:EAX
        loads, so a stale `mov edx, _label` doesn't constitute a
        return value.
        """
        if depth > 5:
            return False
        family = self._REG_FAMILY[reg32]
        scanned = 0
        j = start_idx
        while j < len(lines) and scanned < 20:
            ln = lines[j]
            if ln.kind in ("blank", "comment", "label"):
                # Codegen invariant: regs are not assumed live at any
                # label by predecessor jumps — any user of reg loads it
                # fresh. So crossing labels by text-fallthrough is OK.
                j += 1
                continue
            if ln.kind in ("directive", "data"):
                # End of function (or section change). The function's
                # return path was reached without finding a use.
                if reg32 == "ecx":
                    return True
                return False
            if ln.kind != "instr":
                return False
            scanned += 1
            # Check pure-write FIRST (before checking reads), since
            # mov reg, src can be a pure write OR self-RMW.
            if PeepholeOptimizer._is_pure_reg_write(ln, reg32):
                return True
            # Function exit terminators.
            if ln.op in {"ret", "iret", "iretd", "retf", "retn"}:
                # EAX/EDX may be the return value (live).
                # EBX/ESI/EDI/EBP are callee-saved (caller expects
                # them preserved — dropping a write inside the function
                # leaves them at the caller's value, which is fine).
                # ECX is caller-saved scratch.
                if reg32 in ("eax", "edx"):
                    if treat_as_scratch:
                        return True
                    return False
                return True
            # `leave` is `mov esp, ebp; pop ebp` — it READS EBP and
            # WRITES ESP and EBP. For EBP/ESP, the read makes them
            # alive at this point; for other regs, leave doesn't
            # touch them, continue scanning.
            if ln.op == "leave":
                if reg32 in ("ebp", "esp"):
                    return False
                j += 1
                continue
            # If this instruction references the reg family in any
            # operand, treat as a read.
            if self._references_reg_family(ln.operands, reg32):
                return False
            # `jmp <label>` (direct, unconditional): chase the target.
            # DO NOT fall through to text-next — that's unreachable
            # from this CFG path.
            if ln.op == "jmp":
                target = _jmp_target(ln)
                if target is None:
                    return False  # indirect jmp — bail
                if target in visited_labels:
                    return False  # loop — bail
                target_idx = self._find_label_idx(lines, target, j)
                if target_idx is None:
                    return False
                # Tail-call into the scan at the label's first
                # post-label instruction. Tail-call style avoids
                # unbounded recursion when chains of jmps are short.
                return self._reg_dead_after(
                    lines, target_idx + 1, reg32,
                    visited_labels | frozenset({target}),
                    depth + 1,
                    treat_as_scratch=treat_as_scratch,
                )
            # Conditional jumps split control flow. Both successors
            # must reach a pure-write before any read. Recurse on the
            # target path; continue scanning the fallthrough.
            if ln.op.startswith("j") and ln.op != "jmp":
                target = _branch_target(ln)
                if target is None:
                    return False  # indirect/unparseable — bail
                if target in visited_labels:
                    return False  # loop — bail
                target_idx = self._find_label_idx(lines, target, j)
                if target_idx is None:
                    return False
                # Target path: must be dead.
                target_dead = self._reg_dead_after(
                    lines, target_idx + 1, reg32,
                    visited_labels | frozenset({target}),
                    depth + 1,
                    treat_as_scratch=treat_as_scratch,
                )
                if not target_dead:
                    return False
                # Fallthrough path: keep scanning.
                j += 1
                continue
            # `call <label>` clobbers caller-saved regs (EAX/ECX/EDX).
            if ln.op == "call":
                operand = ln.operands.strip().lower()
                if operand in family:
                    # Indirect via reg — reads it.
                    return False
                if "[" in operand and self._references_reg_family(
                        operand, reg32):
                    return False
                # uc386's libc has internal helpers that read EAX as
                # an INPUT register (a non-cdecl, register-passing
                # convention chosen because they're hot inner loops).
                # Treat the call as a read of EAX in that case.
                if reg32 == "eax" and operand in self._CALLS_READ_EAX:
                    return False
                if reg32 in {"eax", "ecx", "edx"}:
                    return True
                # Reg preserved across cdecl call. Continue.
                j += 1
                continue
            # If this instruction can have implicit register reads
            # (e.g. cdq, idiv, lodsd, rep movsd) — bail conservatively.
            if not PeepholeOptimizer._has_explicit_operands_only(ln):
                return False
            # Otherwise: explicit-operand instr that doesn't reference
            # reg, didn't write it purely. Just move on.
            j += 1
        return False

    @staticmethod
    def _find_label_idx(
        lines: list[Line], label: str, from_idx: int | None = None,
    ) -> int | None:
        """Find the index of label `label` in `lines`. Returns the
        index of the label line (the next instr is at idx+1).

        NASM local-label scoping: a label starting with `.` is
        scoped to the most recent non-local (`_name`) label before
        the reference. Two functions can have `.L2_endif:` and
        they're DIFFERENT labels.

        If `label` starts with `.` and `from_idx` is provided, the
        scope is determined by finding the most recent non-local
        label before `from_idx`. Search proceeds within that scope
        only (until the next non-local label).

        For non-local labels OR when `from_idx` is None (legacy
        callers), returns the first occurrence.
        """
        if label.startswith(".") and from_idx is not None:
            # Find scope owner: most recent non-local label before
            # (or at) `from_idx - 1`.
            scope_start = -1
            for k in range(min(from_idx - 1, len(lines) - 1), -1, -1):
                ln = lines[k]
                if (
                    ln.kind == "label"
                    and not ln.label.startswith(".")
                ):
                    scope_start = k
                    break
            if scope_start == -1:
                # No scope owner found; fall back to first match.
                for k, ln in enumerate(lines):
                    if ln.kind == "label" and ln.label == label:
                        return k
                return None
            # Search within scope: from scope_start to the next
            # non-local label (exclusive).
            for k in range(scope_start, len(lines)):
                ln = lines[k]
                if ln.kind == "label":
                    if ln.label == label:
                        return k
                    # Hit a different non-local label — past our scope.
                    if (
                        not ln.label.startswith(".")
                        and k > scope_start
                    ):
                        return None
            return None
        # Non-local label OR no from_idx context — use first match.
        for k, ln in enumerate(lines):
            if ln.kind == "label" and ln.label == label:
                return k
        return None

    def _pass_dead_mov_to_reg(self, lines: list[Line]) -> list[Line]:
        """Drop `mov eax, <src>` (or `lea eax, ...`) when the loaded
        value is provably dead — a pure overwrite of EAX follows
        before any read.

        Common in postfix `i++` in for-loop step where the value is
        discarded:
            mov eax, [ebp - 4]   ← this load
            inc dword [ebp - 4]
            jmp .L1_for_top      ← .L1_for_top: mov eax, [...] (overwrites)

        Restricted to EAX because uc386 has a static-link convention
        (nested-function trampolines) where ECX is an implicit input
        to certain calls — dropping a `mov ecx, X` before such a
        call would break the call. EBX/ESI/EDI/EBP are callee-saved
        and rarely have dead writes worth dropping.

        Conditions:
        - Source must not reference EAX/AX/AL/AH (would be self-RMW).
        - Forward scan finds a pure-write to EAX before any read.
        - At a `ret`, EAX is live (return value).
        """
        out: list[Line] = []
        for i, line in enumerate(lines):
            if line.kind == "instr" and line.op in {"mov", "lea",
                                                     "movsx", "movzx"}:
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_lower = dest.lower()
                    if dest_lower == "eax":
                        # When src references eax-family, this is a
                        # self-RMW. If src is a memory deref (e.g.,
                        # ``mov eax, [eax]``), the read could have
                        # observable effects (page faults, MMIO) —
                        # skip dropping. If src is a sub-register
                        # (e.g., ``movsx eax, al``), no memory access,
                        # safe to drop when EAX is dead.
                        src_is_self = self._references_reg_family(
                            src, "eax"
                        )
                        src_is_mem = "[" in src
                        if not (src_is_self and src_is_mem):
                            if self._reg_dead_after(lines, i + 1,
                                                     "eax"):
                                self.stats["dead_mov_to_reg"] = (
                                    self.stats.get("dead_mov_to_reg", 0)
                                    + 1
                                )
                                continue
            # `xor reg, reg` is a fresh write to reg (sets reg=0
            # plus flags). If reg is dead before any read AND flags
            # are dead before the next flag-clobber, the xor is
            # entirely dead.
            if line.kind == "instr" and line.op == "xor":
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_low = src.strip().lower()
                    if (
                        dest_low == "eax"
                        and dest_low == src_low
                        and self._reg_dead_after(lines, i + 1, "eax")
                        and self._flags_safe_after(lines, i + 1)
                    ):
                        self.stats["dead_xor_zero_drop"] = (
                            self.stats.get("dead_xor_zero_drop", 0)
                            + 1
                        )
                        continue
            out.append(line)
        return out

    def _pass_prologue_to_enter(self, lines: list[Line]) -> list[Line]:
        """P-prologue-to-enter: collapse the standard cdecl prologue
        ``push ebp; mov ebp, esp; sub esp, IMM`` to a single
        ``enter IMM, 0`` instruction. Saves 2 bytes when IMM fits in
        imm8, 5 bytes when IMM is imm32. Identical semantics — Intel
        defines ``enter IMM, 0`` to be exactly the three-instruction
        sequence (no extra register touches at level=0).

        Constraints:
        - Three consecutive ``instr`` lines (no intervening comments,
          blanks, or labels).
        - IMM must be a literal in (0, 65535] (the imm16 width of the
          first ``enter`` operand).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 2 < len(lines)
                and line.kind == "instr"
                and line.op == "push"
                and line.operands.strip().lower() == "ebp"
                and lines[i + 1].kind == "instr"
                and self._is_mov_ebp_esp(lines[i + 1])
                and lines[i + 2].kind == "instr"
                and lines[i + 2].op == "sub"
            ):
                imm = self._extract_sub_esp_imm(lines[i + 2])
                if imm is not None and 0 < imm <= 65535:
                    indent = self._extract_indent(line.raw)
                    new_raw = f"{indent}enter   {imm}, 0"
                    new_line = Line(
                        raw=new_raw,
                        kind="instr",
                        op="enter",
                        operands=f"{imm}, 0",
                    )
                    out.append(new_line)
                    self.stats["prologue_to_enter"] = (
                        self.stats.get("prologue_to_enter", 0) + 1
                    )
                    i += 3
                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_mov_ebp_esp(line: Line) -> bool:
        if line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        return dest.strip().lower() == "ebp" and src.strip().lower() == "esp"

    @staticmethod
    def _extract_sub_esp_imm(line: Line) -> int | None:
        if line.op != "sub":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        if dest.strip().lower() != "esp":
            return None
        s = src.strip()
        # Reject memory derefs and bare register names.
        if "[" in s:
            return None
        try:
            if s.lower().startswith("0x"):
                return int(s, 16)
            return int(s)
        except ValueError:
            return None

    @staticmethod
    def _extract_indent(raw: str) -> str:
        n = 0
        while n < len(raw) and raw[n] in " \t":
            n += 1
        return raw[:n]

    def _pass_redundant_eax_load(self, lines: list[Line]) -> list[Line]:
        """Drop ``mov eax, M`` when EAX already provably holds M.
        See ``_run_redundant_reg_load`` for the algorithm."""
        return self._run_redundant_reg_load(
            lines, "eax", {"al", "ah", "ax"}, "redundant_eax_load",
        )

    def _pass_redundant_ecx_load(self, lines: list[Line]) -> list[Line]:
        """Drop ``mov ecx, M`` when ECX already provably holds M.
        Sister of ``_pass_redundant_eax_load``. Common after
        ``index_load_collapse`` / ``push_index_collapse`` leaves
        a ``mov ecx, [m]`` whose value is still in ECX from the
        prior load."""
        return self._run_redundant_reg_load(
            lines, "ecx", {"cl", "ch", "cx"}, "redundant_ecx_load",
        )

    def _run_redundant_reg_load(
        self, lines: list[Line], reg32: str,
        sub_regs: set[str], stat_key: str,
    ) -> list[Line]:
        """Drop ``mov REG, M`` when REG already provably holds M's
        value — we loaded it earlier in the basic block and nothing
        between then and now wrote REG or touched M.

        Tracking is conservative: we follow REG's "current memory
        source" through a single straight-line block. Any of these
        invalidate the tracker:
        - Any write to REG (or its sub-regs) by any instruction
        - Any memory-write whose destination might alias M
        - Calls and unconditional control flow (jmp/ret/etc.)
        - Any instruction in ``_IMPLICIT_REG_USERS`` (over-
          conservative — some only touch other regs, but safe)
        - Any label that's the target of a BACKWARD jump (loop top —
          the back-edge means downstream stores to M may invalidate
          the cached value before re-entry)

        Conditional jumps (jcc) DO preserve the tracker on the
        fall-through path; we record the jcc's reg_mem at the time
        of the jump, keyed by target label. At each label, we merge
        the jcc-recorded states with the fall-through state (if any).

        For aliasing, we only trust two stack expressions to be
        disjoint when both are literal ``[ebp + N]`` / ``[ebp - N]``
        offsets (different N). Stores via a register address (e.g.
        ``[ecx]``), or to a label, conservatively invalidate.

        Tracked operands: literal-offset stack reads. Register
        sources, immediates, and labels are not tracked because they
        don't alias-conflict with anything codegen emits (immediates)
        or aren't worth tracking (registers — codegen reloads from
        memory, not register-to-register).
        """
        # Pre-compute line indices that are targets of backward jumps.
        # At those positions, we conservatively reset reg_mem because
        # the back-edge may have stores that invalidate cached state
        # before re-entry.
        #
        # Local NASM labels (starting with `.`) are scoped to the
        # preceding non-local label. The whole-file asm reuses the
        # same `.LN_do_top` name across many functions, so a flat
        # name → position dict would resolve every reference to the
        # LAST definition in the file — silently missing back-edges
        # and letting the redundant-load pass nuke a needed reload.
        # We key labels by (scope_line, label_name); the scope is
        # the line index of the enclosing non-local label.
        label_positions: dict[tuple[int, str], int] = {}
        scope_at: list[int] = []
        current_scope = 0
        for k, ln in enumerate(lines):
            if ln.kind == "label" and not ln.label.startswith("."):
                current_scope = k
            scope_at.append(current_scope)
            if ln.kind == "label":
                scope_key = 0 if not ln.label.startswith(".") else current_scope
                label_positions[(scope_key, ln.label)] = k
        loop_top_positions: set[int] = set()
        for k, ln in enumerate(lines):
            if ln.kind == "instr" and (
                ln.op == "jmp" or ln.op.startswith("j")
            ):
                target = _branch_target(ln)
                if target is None:
                    continue
                scope_key = 0 if not target.startswith(".") else scope_at[k]
                target_idx = label_positions.get((scope_key, target))
                if target_idx is not None and target_idx <= k:
                    loop_top_positions.add(target_idx)

        # Per-function address-taken set: a register-base write
        # (e.g. `mov [ecx], eax`) cannot alias `[ebp + N]` unless N is
        # in the function's address-taken set. Without this analysis,
        # the conservative `_mem_disjoint` would invalidate `reg_mem`
        # on every register-base write — losing all redundant-reload
        # opportunities through pointer-based stores.
        addr_taken_per_line = self._compute_addr_taken_per_line(lines)
        # Per-function label-address-taken set: enables tracking of
        # `[_label]` loads — a register-base write can only alias a
        # global if its address has been exposed somewhere in the
        # function (via `mov reg, _label`, `lea reg, [_label]`, etc.).
        label_addr_taken_per_line = (
            self._compute_label_addr_taken_per_line(lines)
        )

        out: list[Line] = []
        reg_mem: str | None = None  # The literal text of REG's known mem source
        # Independent tracker for "extension form": after `movsx reg, byte
        # [...]` or `movsx reg, low_byte_of_reg`, EAX bits 8-31 are sign-
        # extended from bit 7 of AL. A subsequent `movsx reg, al` is then
        # redundant. Same for movzx and word forms.
        # Values: None / "movsx_byte" / "movzx_byte" / "movsx_word" /
        # "movzx_word". Conservatively cleared on any write to reg32 or
        # any sub-register of reg32.
        ext_form: str | None = None
        jcc_states: dict[str, set[str | None]] = {}
        ext_jcc_states: dict[str, set[str | None]] = {}
        prev_unconditional = False
        for line_idx, line in enumerate(lines):
            if line.kind != "instr":
                if line.kind == "label":
                    label = line.label
                    # Non-local label = function boundary. Local
                    # labels (.X) are scoped to the enclosing
                    # function, so a leaked jcc_states / reg_mem
                    # entry from a previous function with the same
                    # local name would corrupt analysis here.
                    if not label.startswith("."):
                        jcc_states.clear()
                        ext_jcc_states.clear()
                        reg_mem = None
                        ext_form = None
                        prev_unconditional = False
                        out.append(line)
                        continue
                    saved = jcc_states.pop(label, None)
                    saved_ext = ext_jcc_states.pop(label, None)
                    if line_idx in loop_top_positions:
                        # Loop top: a back-edge may store to memory
                        # between iterations, invalidating any cached
                        # reg_mem value. Conservatively reset.
                        reg_mem = None
                        ext_form = None
                    elif prev_unconditional:
                        if saved is not None and len(saved) == 1:
                            (only,) = saved
                            reg_mem = only
                        else:
                            reg_mem = None
                        if saved_ext is not None and len(saved_ext) == 1:
                            (only,) = saved_ext
                            ext_form = only
                        else:
                            ext_form = None
                    else:
                        if saved is None:
                            pass
                        else:
                            states = saved | {reg_mem}
                            if len(states) == 1:
                                (only,) = states
                                reg_mem = only
                            else:
                                reg_mem = None
                        if saved_ext is None:
                            pass
                        else:
                            states_ext = saved_ext | {ext_form}
                            if len(states_ext) == 1:
                                (only,) = states_ext
                                ext_form = only
                            else:
                                ext_form = None
                    prev_unconditional = False
                out.append(line)
                continue
            op = line.op
            ops = line.operands
            if op in ("movsx", "movzx"):
                # Track sign/zero-extend loads as a distinct key
                # form: f"{op} {src_norm_with_size}". A subsequent
                # identical movsx/movzx load is redundant.
                # Different load forms (e.g. mov vs movsx, or
                # movsx byte vs movsx word) are textually distinct
                # and don't match each other — correct, since they
                # produce different register values.
                #
                # ALSO: `movsx <reg>, <low_byte_of_reg>` is redundant
                # when reg already holds a sign-extended byte (set
                # via `movsx <reg>, byte [...]` previously, or via
                # any other `movsx <reg>, byte X` regardless of X's
                # location). Same for movzx and word forms. Tracked
                # via the independent ``ext_form`` field.
                parts = _operands_split(ops)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if dest_low == reg32:
                        # Determine reg32's low byte and low word
                        # sub-registers (eax → al/ax, ecx → cl/cx, etc.)
                        low_byte = reg32[1] + "l"  # eax → al, ecx → cl
                        low_word = reg32[1:]  # eax → ax, ecx → cx
                        src_low = src_norm.lower()
                        # Self-extension form: movsx eax, al / movzx eax, ax
                        if src_low == low_byte:
                            # Redundant if ext_form matches
                            target_form = f"{op}_byte"
                            if ext_form == target_form:
                                self.stats[stat_key] = (
                                    self.stats.get(stat_key, 0) + 1
                                )
                                continue
                            # Otherwise the operation changes EAX
                            # extension semantic. Set new state.
                            ext_form = target_form
                            reg_mem = None  # value is now ext-of-AL, not memory
                            out.append(line)
                            continue
                        if src_low == low_word:
                            target_form = f"{op}_word"
                            if ext_form == target_form:
                                self.stats[stat_key] = (
                                    self.stats.get(stat_key, 0) + 1
                                )
                                continue
                            ext_form = target_form
                            reg_mem = None
                            out.append(line)
                            continue
                        # Source is a memory operand (or other reg).
                        full_key = f"{op} {src_norm}"
                        trackable = (
                            self._is_ebp_offset_mem(src_norm)
                            or self._is_label_mem(src_norm)
                        )
                        if (
                            reg_mem is not None
                            and trackable
                            and reg_mem == full_key
                        ):
                            self.stats[stat_key] = (
                                self.stats.get(stat_key, 0) + 1
                            )
                            continue
                        if trackable:
                            reg_mem = full_key
                        else:
                            reg_mem = None
                        # Set ext_form based on the size in src_norm.
                        # Strip optional size prefix and detect.
                        src_lower = src_norm.lower()
                        if (
                            src_lower.startswith("byte ")
                            or "byte [" in src_lower
                            or src_lower in {"al", "bl", "cl", "dl",
                                             "ah", "bh", "ch", "dh"}
                        ):
                            ext_form = f"{op}_byte"
                        elif (
                            src_lower.startswith("word ")
                            or "word [" in src_lower
                            or src_lower in {"ax", "bx", "cx", "dx",
                                             "si", "di", "bp", "sp"}
                        ):
                            ext_form = f"{op}_word"
                        else:
                            ext_form = None
                        out.append(line)
                        continue
                    # Fall through: unusual forms (sub-reg dest etc.)
                    # — invalidate. movsx/movzx writing a sub-reg of
                    # reg32 is rare (movsx writes to 32-bit dest in
                    # x86 only — there's no `movsx al, byte [...]`
                    # form), but be defensive.
                    if dest_low in sub_regs:
                        reg_mem = None
                        ext_form = None
                    out.append(line)
                    continue
            if op == "mov":
                parts = _operands_split(ops)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if dest_low == reg32:
                        # mov REG, X
                        trackable = (
                            self._is_ebp_offset_mem(src_norm)
                            or self._is_label_mem(src_norm)
                        )
                        if (
                            reg_mem is not None
                            and trackable
                            and src_norm == reg_mem
                        ):
                            self.stats[stat_key] = (
                                self.stats.get(stat_key, 0) + 1
                            )
                            continue
                        if trackable:
                            reg_mem = src_norm
                        else:
                            reg_mem = None
                        # Mov writes the full register; ext_form
                        # property no longer holds (we don't know
                        # if the source was sign/zero-extended).
                        ext_form = None
                        out.append(line)
                        continue
                    if "[" in dest:
                        # Aliasing check: if the store might alias
                        # the currently-tracked memory, invalidate.
                        # Use the per-function address-taken sets so
                        # register-base writes are recognized as
                        # disjoint from `[ebp + N]` and `[_label]`
                        # reads when the slot/label isn't address-
                        # taken.
                        addr_taken = addr_taken_per_line[line_idx]
                        label_taken = (
                            label_addr_taken_per_line[line_idx]
                        )
                        if reg_mem is not None and not (
                            self._mem_disjoint_with_taken(
                                reg_mem, dest.strip(), addr_taken,
                                label_taken,
                            )
                        ):
                            reg_mem = None
                        # Sub-reg dest: shouldn't happen with `[`,
                        # but be defensive.
                        if dest_low in sub_regs:
                            reg_mem = None
                            ext_form = None
                        # If the store is `mov [m], REG` where m is
                        # a tracked memory form, then after the
                        # store REG equals what's at m. Track that
                        # — a later `mov REG, m` is then redundant.
                        # Only set when reg_mem was None (we don't
                        # want to overwrite a still-valid prior
                        # tracking entry; full multi-location
                        # tracking would be ideal but reg_mem is
                        # single-valued).
                        if (
                            reg_mem is None
                            and src_norm.lower() == reg32
                            and (
                                self._is_ebp_offset_mem(dest.strip())
                                or self._is_label_mem(dest.strip())
                            )
                        ):
                            reg_mem = dest.strip()
                        # Memory store doesn't modify the register
                        # itself, so ext_form is preserved (unless
                        # alias-invalidation already handled).
                        out.append(line)
                        continue
                    if dest_low in sub_regs:
                        # Sub-register write (mov al, X / mov ax, X).
                        # Bits not written (e.g., 8-31 for `mov al, X`)
                        # are unchanged. The sign/zero-extension
                        # property is broken: even if reg32 was
                        # sign-extended from old AL, the new AL might
                        # not match bits 8-31. Reset both.
                        reg_mem = None
                        ext_form = None
                    out.append(line)
                    continue
            if op == "call":
                reg_mem = None
                ext_form = None
                prev_unconditional = False
                out.append(line)
                continue
            if op in {"jmp", "ret", "iret", "iretd", "retf",
                      "retn", "leave", "enter"}:
                if op == "jmp":
                    target = line.operands.strip()
                    jcc_states.setdefault(target, set()).add(reg_mem)
                    ext_jcc_states.setdefault(target, set()).add(
                        ext_form
                    )
                reg_mem = None
                ext_form = None
                prev_unconditional = op != "enter"
                out.append(line)
                continue
            if op.startswith("j"):
                target = line.operands.strip()
                jcc_states.setdefault(target, set()).add(reg_mem)
                ext_jcc_states.setdefault(target, set()).add(ext_form)
                prev_unconditional = False
                out.append(line)
                continue
            if op in {"cmp", "test", "push"}:
                out.append(line)
                continue
            if (
                self._references_reg_family(ops, reg32)
                or op in PeepholeOptimizer._IMPLICIT_REG_USERS
            ):
                reg_mem = None
                ext_form = None
                out.append(line)
                continue
            if "[" in ops:
                parts = _operands_split(ops)
                if parts is not None:
                    dest, _ = parts
                    if "[" in dest and reg_mem is not None and not (
                        self._mem_disjoint_with_taken(
                            reg_mem, dest.strip(),
                            addr_taken_per_line[line_idx],
                            label_addr_taken_per_line[line_idx],
                        )
                    ):
                        reg_mem = None
                else:
                    if reg_mem is not None:
                        reg_mem = None
            out.append(line)
        return out

    @staticmethod
    def _is_ebp_offset_mem(text: str) -> bool:
        """Match exactly `[ebp + N]` or `[ebp - N]` (or `dword [...]`,
        etc.) where N is a numeric literal."""
        # Strip optional size prefix (`dword`, `word`, `byte`).
        stripped = text.strip()
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if stripped.lower().startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
                break
        return bool(re.fullmatch(
            r"\[\s*ebp\s*[+-]\s*\d+\s*\]", stripped, re.IGNORECASE
        ))

    @staticmethod
    def _is_label_mem(text: str) -> bool:
        """Match exactly `[_label]` (or with optional size prefix
        `dword`/`word`/`byte`/`qword`). Matches global-label-only
        forms with no offset and no SIB index — `[_g + 4]` and
        `[_g + reg*4]` do NOT match (they aren't constant memory
        references over the lifetime of a basic block under
        plausible aliasing assumptions)."""
        stripped = text.strip()
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if stripped.lower().startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
                break
        return bool(re.fullmatch(
            r"\[\s*_[A-Za-z_][A-Za-z0-9_]*\s*\]", stripped
        ))

    @staticmethod
    def _label_in_mem(text: str) -> str | None:
        """Extract the label name from a `[_label]` operand, or None
        if not a label-only memory operand."""
        stripped = text.strip()
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if stripped.lower().startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
                break
        m = re.fullmatch(
            r"\[\s*(_[A-Za-z_][A-Za-z0-9_]*)\s*\]", stripped
        )
        return m.group(1) if m else None

    @staticmethod
    def _mem_disjoint(a: str, b: str) -> bool:
        """Conservative aliasing check: True if `a` and `b` are
        provably-disjoint memory references. Only handles literal
        ``[ebp + N]`` offsets (different N → disjoint).

        Returns False (= "may alias") for anything we can't reason
        about (register-base derefs, labels, etc.).
        """
        a_off = PeepholeOptimizer._ebp_offset(a)
        b_off = PeepholeOptimizer._ebp_offset(b)
        if a_off is None or b_off is None:
            return False
        return a_off != b_off

    @staticmethod
    def _mem_disjoint_with_taken(
        a: str, b: str, addr_taken: set[int] | None,
        label_addr_taken: set[str] | None = None,
    ) -> bool:
        """Like ``_mem_disjoint`` but augmented with address-taken
        sets:

        - When one operand is ``[ebp + N]`` and the other is a
          register-base deref, the two are provably disjoint iff N is
          NOT in ``addr_taken`` (and the register isn't ``ebp``).
        - When one operand is ``[_label]`` and the other is a
          register-base deref, disjoint iff ``_label`` is NOT in
          ``label_addr_taken``.
        - ``[ebp + N]`` and ``[_label]`` are always disjoint (frame
          and global storage never alias).
        - Two ``[_label]`` operands are disjoint iff label names
          differ.

        ``addr_taken`` is the set of ebp-offsets that the current
        function explicitly takes the address of (via
        ``lea X, [ebp ± N]``). Slots not in this set can never be
        aliased by a write through any general-purpose register
        because no register in the function ever holds an address
        inside this frame.

        ``label_addr_taken`` is the set of label names whose address
        is taken (loaded as immediate or via ``lea``) in the current
        function. Globals not in this set can never be aliased by a
        register-base write.

        Either (or both) ``addr_taken`` / ``label_addr_taken`` may be
        None; ``addr_taken=None`` falls back to the static
        ``_mem_disjoint``. ``label_addr_taken=None`` means label-vs-
        register-base aliasing is conservatively unknown.
        """
        if addr_taken is None and label_addr_taken is None:
            return PeepholeOptimizer._mem_disjoint(a, b)
        a_off = PeepholeOptimizer._ebp_offset(a)
        b_off = PeepholeOptimizer._ebp_offset(b)
        if a_off is not None and b_off is not None:
            return a_off != b_off
        a_lbl = PeepholeOptimizer._label_in_mem(a)
        b_lbl = PeepholeOptimizer._label_in_mem(b)
        # Both label-only.
        if a_lbl is not None and b_lbl is not None:
            return a_lbl != b_lbl
        # Frame vs label: always disjoint.
        if a_off is not None and b_lbl is not None:
            return True
        if b_off is not None and a_lbl is not None:
            return True
        # One is frame, other is something else.
        if a_off is not None or b_off is not None:
            if a_off is not None:
                other = b
                this_off = a_off
            else:
                other = a
                this_off = b_off
            other_stripped = other.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if other_stripped.lower().startswith(prefix):
                    other_stripped = other_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            if not (
                other_stripped.startswith("[")
                and other_stripped.endswith("]")
            ):
                return False
            inner = other_stripped[1:-1].strip().lower()
            if "ebp" in inner:
                return False
            if addr_taken is None:
                return False
            return this_off not in addr_taken
        # One is label-only, other is something else.
        if a_lbl is not None or b_lbl is not None:
            if a_lbl is not None:
                other = b
                this_lbl = a_lbl
            else:
                other = a
                this_lbl = b_lbl
            other_stripped = other.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if other_stripped.lower().startswith(prefix):
                    other_stripped = other_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            if not (
                other_stripped.startswith("[")
                and other_stripped.endswith("]")
            ):
                return False
            inner = other_stripped[1:-1].strip().lower()
            # `ebp` appearing in the other deref means it touches
            # the frame — frame and global storage are disjoint.
            if "ebp" in inner:
                return True
            if label_addr_taken is None:
                return False
            return this_lbl not in label_addr_taken
        # Neither side is a recognized literal form.
        return False

    @staticmethod
    def _function_ranges(lines: list[Line]) -> list[tuple[int, int]]:
        """Identify function boundaries via global labels.

        A "global label" (in NASM terms) is one whose name doesn't
        start with ``.`` (which would be a local-to-function label).
        Each function spans from its global label's line to (but not
        including) the next global label or end-of-lines.

        Returns a list of ``(start_idx, end_idx)`` half-open intervals
        covering the input — each line index falls in exactly one
        range. The first range covers the prelude (before any global
        label).
        """
        bounds: list[int] = []
        for i, line in enumerate(lines):
            if line.kind == "label" and not line.label.startswith("."):
                bounds.append(i)
        ranges: list[tuple[int, int]] = []
        if not bounds:
            return [(0, len(lines))]
        if bounds[0] > 0:
            ranges.append((0, bounds[0]))
        for k, start in enumerate(bounds):
            end = bounds[k + 1] if k + 1 < len(bounds) else len(lines)
            ranges.append((start, end))
        return ranges

    @staticmethod
    def _compute_unused_regs_per_line(
        lines: list[Line],
    ) -> list[set[str] | None]:
        """For each line index, return the set of GP registers that are
        UNUSED in the function containing this line.

        A register is "unused" if no instruction in the function reads
        or writes it (including sub-aliases). Such registers are safe
        to clobber by a peephole rewrite — no prior write would have a
        future read, and no future read of pre-clobber state matters.

        Returns a list of length ``len(lines)``. Lines outside any
        function (prelude before first global label) get ``None``.

        Note: implicit register users (e.g., ``cdq``, ``idiv``,
        ``mul``) are caught by their explicit operands when present;
        we additionally treat the bare-form opcodes as reads of EAX
        and EDX in those cases.
        """
        # Implicit register users: ops that reference EAX/EDX (or
        # other regs) via implicit semantics, not just operands.
        IMPLICIT_EAX_USERS = {
            "cdq", "cwd", "cwde", "lodsd", "lodsw", "lodsb",
            "stosd", "stosw", "stosb", "scasd", "scasw", "scasb",
            "mul", "div", "idiv",
            # `imul` only implicitly touches EAX/EDX in 1-operand
            # form. The 2- and 3-operand forms have explicit
            # destinations and don't touch EAX/EDX implicitly. We
            # detect via operand-count below.
        }
        IMPLICIT_EDX_USERS = {
            "cdq", "cwd", "mul", "div", "idiv",
        }
        IMPLICIT_ESI_USERS = {
            "lodsd", "lodsw", "lodsb", "movsd", "movsw", "movsb",
            "scasd", "scasw", "scasb",
        }
        IMPLICIT_EDI_USERS = {
            "stosd", "stosw", "stosb", "movsd", "movsw", "movsb",
            "scasd", "scasw", "scasb",
        }
        IMPLICIT_ECX_USERS = {
            "rep", "repe", "repne", "repnz", "loope", "loopne",
            "loopz", "loopnz", "loop",
        }
        ALL_GP = {"eax", "ecx", "edx", "ebx", "esi", "edi"}
        out: list[set[str] | None] = [None] * len(lines)
        ranges = PeepholeOptimizer._function_ranges(lines)
        families_referenced = PeepholeOptimizer._families_referenced
        for start, end in ranges:
            referenced: set[str] = set()
            for i in range(start, end):
                line = lines[i]
                if line.kind != "instr":
                    continue
                # Operand-level reference check — single regex pass
                # per line via _families_referenced.
                ops = line.operands
                if ops:
                    referenced |= families_referenced(ops) & ALL_GP
                if referenced >= ALL_GP:
                    # All 6 GP regs already referenced — nothing more
                    # to learn from later instructions in this function.
                    # Still need to walk to the end to set out[] so we
                    # can drop the work shortly.
                    break
                # Implicit-reference ops.
                op = line.op
                if op in IMPLICIT_EAX_USERS:
                    referenced.add("eax")
                if op in IMPLICIT_EDX_USERS:
                    referenced.add("edx")
                # Special case for `imul`: only the 1-operand form
                # implicitly touches EAX/EDX. 2-/3-operand forms
                # have explicit destinations and don't.
                if op == "imul" and "," not in (ops or ""):
                    referenced.add("eax")
                    referenced.add("edx")
                if op in IMPLICIT_ESI_USERS:
                    referenced.add("esi")
                if op in IMPLICIT_EDI_USERS:
                    referenced.add("edi")
                if op in IMPLICIT_ECX_USERS:
                    referenced.add("ecx")
            unused = ALL_GP - referenced
            for i in range(start, end):
                out[i] = unused
        return out

    @staticmethod
    def _compute_addr_taken_per_line(
        lines: list[Line],
    ) -> list[set[int] | None]:
        """For each line index, return the address-taken set of the
        function containing that line.

        The address-taken set for a function is the set of ebp-offsets
        N such that ``lea X, [ebp ± N]`` appears in that function.
        Slots in this set are potentially aliased by writes through
        any GP register holding their address.

        Returns a list of length ``len(lines)``. Lines outside any
        function (prelude before first global label) get ``None`` —
        no analysis available.
        """
        out: list[set[int] | None] = [None] * len(lines)
        ranges = PeepholeOptimizer._function_ranges(lines)
        for start, end in ranges:
            taken: set[int] = set()
            for i in range(start, end):
                line = lines[i]
                if line.kind != "instr" or line.op != "lea":
                    continue
                parts = _operands_split(line.operands)
                if parts is None:
                    continue
                _, src = parts
                off = PeepholeOptimizer._ebp_offset(src.strip())
                if off is not None:
                    taken.add(off)
            for i in range(start, end):
                out[i] = taken
        return out

    @staticmethod
    def _compute_label_addr_taken_per_line(
        lines: list[Line],
    ) -> list[set[str] | None]:
        """For each line index, return the set of label names whose
        address is taken (could end up in a register) in the function
        containing that line.

        A label ``_foo`` is address-taken if it appears anywhere in
        the function as:
        - An immediate operand (e.g., ``mov reg, _foo``,
          ``push _foo``, ``add reg, _foo``).
        - A bracketed operand of ``lea`` (``lea reg, [_foo]``,
          ``lea reg, [_foo + 4]``, etc.).

        Labels that only appear as memory operands of normal
        instructions (e.g., ``mov reg, [_foo]``) are NOT address-
        taken — those access memory but don't expose the address.

        Lines outside any function (prelude before first global
        label) get ``None`` — no analysis available.
        """
        out: list[set[str] | None] = [None] * len(lines)
        ranges = PeepholeOptimizer._function_ranges(lines)
        # A label ref starts with `_` followed by ident chars.
        # NASM-local labels (`.L1_for_top`) start with `.` and
        # contain underscores embedded — those won't match because
        # the leading `_` requirement enforces a word-start `_`.
        label_re = re.compile(r"\b_[A-Za-z_][A-Za-z0-9_]*\b")
        bracket_re = re.compile(r"\[[^\]]*\]")
        for start, end in ranges:
            taken: set[str] = set()
            for i in range(start, end):
                line = lines[i]
                if line.kind != "instr":
                    continue
                ops = line.operands or ""
                if line.op == "lea":
                    # Every label reference inside `lea`'s operands
                    # (including those inside brackets) is an
                    # address-take.
                    for m in label_re.finditer(ops):
                        taken.add(m.group(0))
                else:
                    # For any other op, label refs INSIDE brackets
                    # are memory operands — they read/write memory
                    # at the label's address but don't expose the
                    # address itself. Refs OUTSIDE brackets are
                    # immediate-form address-takes.
                    outside = bracket_re.sub("", ops)
                    for m in label_re.finditer(outside):
                        taken.add(m.group(0))
            for i in range(start, end):
                out[i] = taken
        return out

    @staticmethod
    def _ebp_offset(text: str) -> int | None:
        """Extract the offset from `[ebp + N]` / `[ebp - N]`. Returns
        None for non-matching forms.

        Also accepts the redundant_reg_load tracker's prefixed
        forms: ``movsx byte [ebp - 4]``, ``movzx word [ebp + 8]``,
        etc. The ``op_prefix size_prefix`` are stripped before the
        offset match.
        """
        stripped = text.strip()
        # Strip optional movsx/movzx op prefix (used by
        # redundant_reg_load's tracker for sub-word loads).
        for op_prefix in ("movsx ", "movzx "):
            if stripped.lower().startswith(op_prefix):
                stripped = stripped[len(op_prefix):].lstrip()
                break
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if stripped.lower().startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
                break
        m = re.fullmatch(
            r"\[\s*ebp\s*([+-])\s*(\d+)\s*\]", stripped, re.IGNORECASE
        )
        if not m:
            return None
        sign, num = m.group(1), m.group(2)
        return int(num) if sign == "+" else -int(num)

    def _pass_label_offset_fold(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, LABEL; add reg, IMM`` (or ``sub reg, IMM``)
        into ``mov reg, LABEL ± IMM``. NASM resolves the ``LABEL ± IMM``
        at assemble time, producing a single 5-byte ``mov reg, imm32``
        and eliminating the 3-byte add/sub. Saves 3 bytes per match.

        Common in pointer arithmetic on globals: ``mov eax, _g;
        add eax, 8`` to compute ``&g[2]`` or ``&s.field``.

        Conditions:
        - First line is ``mov R, X`` where X is non-numeric, non-memory,
          non-register (i.e., a label or label-relative expression).
        - Second line is ``add R, IMM`` or ``sub R, IMM`` (numeric
          literal immediate; not a memory or register).
        - Same R.
        - The flags-after-add are dead (no Jcc / setCC / cmov / adc /
          sbb reading them before they're overwritten). The folded
          ``mov`` doesn't set flags, so the original add/sub's flag
          effects must be unobserved.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if (
                        self._is_general_register(dest_low)
                        and self._is_label_like(src_norm)
                    ):
                        nxt = lines[i + 1]
                        if (
                            nxt.kind == "instr"
                            and nxt.op in ("add", "sub")
                        ):
                            nxt_parts = _operands_split(nxt.operands)
                            if nxt_parts is not None:
                                ndest, nsrc = nxt_parts
                                imm = self._parse_numeric_immediate(
                                    nsrc.strip()
                                )
                                if (
                                    ndest.strip().lower() == dest_low
                                    and imm is not None
                                    and self._flags_safe_after(lines, i + 2)
                                ):
                                    op_sign = "+" if nxt.op == "add" else "-"
                                    indent = self._extract_indent(line.raw)
                                    new_src = f"{src_norm} {op_sign} {imm}"
                                    new_raw = (
                                        f"{indent}mov     "
                                        f"{dest.strip()}, {new_src}"
                                    )
                                    new_line = Line(
                                        raw=new_raw,
                                        kind="instr",
                                        op="mov",
                                        operands=(
                                            f"{dest.strip()}, {new_src}"
                                        ),
                                    )
                                    out.append(new_line)
                                    self.stats["label_offset_fold"] = (
                                        self.stats.get(
                                            "label_offset_fold", 0
                                        ) + 1
                                    )
                                    i += 2
                                    continue
            out.append(line)
            i += 1
        return out

    def _pass_shl_add_label_to_lea(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``shl REG, N; add REG, LABEL`` (where N ∈ {1, 2, 3})
        into ``lea REG, [LABEL + REG*SCALE]`` (where SCALE = 2^N).

        x86 supports the SIB form `[disp32 + idx*scale]` (no base) for
        lea, encoded as `8D /r SIB disp32` = 7 bytes. The original
        sequence is `shl reg, imm8` (3 bytes) + `add reg, imm32` (5
        bytes for label) = 8 bytes. So the lea form saves 1 byte and
        1 instruction per match.

        Common in global-array address computation:
        ``mov eax, [ebp - 4]; shl eax, 2; add eax, _g`` → an indexed
        address into _g, scaled by 4 (int element size). After my pass:
        ``mov eax, [ebp - 4]; lea eax, [_g + eax*4]``.

        Conditions:
        - Line 1: ``shl REG, N`` where N ∈ {1, 2, 3}.
        - Line 2: ``add REG, LABEL`` where LABEL is non-numeric,
          non-memory (a label or label-arithmetic).
        - Same REG.
        - Flags after the add are dead (lea doesn't set flags; the
          original shl + add did).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "shl"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    reg, count = parts
                    reg_low = reg.strip().lower()
                    count_str = count.strip()
                    if (
                        self._is_general_register(reg_low)
                        and count_str in ("1", "2", "3")
                    ):
                        scale = 2 ** int(count_str)
                        nxt = lines[i + 1]
                        if (
                            nxt.kind == "instr"
                            and nxt.op == "add"
                        ):
                            nxt_parts = _operands_split(nxt.operands)
                            if nxt_parts is not None:
                                ndest, nsrc = nxt_parts
                                ndest_low = ndest.strip().lower()
                                nsrc_norm = nsrc.strip()
                                if (
                                    ndest_low == reg_low
                                    and self._is_label_like(nsrc_norm)
                                    and self._flags_safe_after(
                                        lines, i + 2
                                    )
                                ):
                                    indent = self._extract_indent(
                                        line.raw
                                    )
                                    new_src = (
                                        f"[{nsrc_norm} + "
                                        f"{reg_low}*{scale}]"
                                    )
                                    new_raw = (
                                        f"{indent}lea     "
                                        f"{reg.strip()}, {new_src}"
                                    )
                                    new_line = Line(
                                        raw=new_raw,
                                        kind="instr",
                                        op="lea",
                                        operands=(
                                            f"{reg.strip()}, {new_src}"
                                        ),
                                    )
                                    out.append(new_line)
                                    self.stats["shl_add_label_to_lea"] = (
                                        self.stats.get(
                                            "shl_add_label_to_lea", 0
                                        ) + 1
                                    )
                                    i += 2
                                    continue
            out.append(line)
            i += 1
        return out

    def _pass_shl_add_reg_to_lea(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``shl IDX, N; add DST, IDX`` (DST != IDX, N ∈
        {1, 2, 3}) into ``lea DST, [DST + IDX*scale]`` (scale = 2^N).

        Sister pass to ``shl_add_label_to_lea``. Handles the case where
        the second operand of `add` is a register (not a label).
        Common when both base and index live in registers — typically
        `mov BASE, [m]; mov IDX, [n]; shl IDX, N; add BASE, IDX` where
        the result is the address of `BASE + IDX*scale` for a non-load
        consumer (push, dec, etc.).

        Encoding: original `shl IDX, N` (3 bytes) + `add DST, IDX` (2
        bytes) = 5 bytes. Rewrite: `lea DST, [DST + IDX*scale]` = 3
        bytes (8D ModRM SIB). Saves 2 bytes per match.

        Conditions:
        - Line 1: ``shl IDX, N`` where N ∈ {1, 2, 3}.
        - Line 2: ``add DST, IDX`` (DST is a 32-bit GP register).
        - DST != IDX (else the rewrite would produce a different
          arithmetic result — `shl IDX, N; add IDX, IDX` = IDX*(2^N+1),
          while `lea IDX, [IDX + IDX*scale]` = IDX*(scale+1)).
        - Flags after the add are dead (lea doesn't set flags).
        - IDX dead after the rewrite OR IDX is preserved (the lea
          doesn't modify IDX, so this always holds — no liveness
          constraint on IDX needed).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "shl"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    idx_reg, count = parts
                    idx_low = idx_reg.strip().lower()
                    count_str = count.strip()
                    if (
                        self._is_general_register(idx_low)
                        and count_str in ("1", "2", "3")
                    ):
                        scale = 2 ** int(count_str)
                        nxt = lines[i + 1]
                        if (
                            nxt.kind == "instr"
                            and nxt.op == "add"
                        ):
                            nxt_parts = _operands_split(nxt.operands)
                            if nxt_parts is not None:
                                ndest, nsrc = nxt_parts
                                ndest_low = ndest.strip().lower()
                                nsrc_low = nsrc.strip().lower()
                                if (
                                    self._is_general_register(ndest_low)
                                    and nsrc_low == idx_low
                                    and ndest_low != idx_low
                                    and self._flags_safe_after(
                                        lines, i + 2
                                    )
                                ):
                                    indent = self._extract_indent(
                                        line.raw
                                    )
                                    new_src = (
                                        f"[{ndest_low} + "
                                        f"{idx_low}*{scale}]"
                                    )
                                    new_raw = (
                                        f"{indent}lea     "
                                        f"{ndest_low}, {new_src}"
                                    )
                                    new_line = Line(
                                        raw=new_raw,
                                        kind="instr",
                                        op="lea",
                                        operands=(
                                            f"{ndest_low}, {new_src}"
                                        ),
                                    )
                                    out.append(new_line)
                                    self.stats["shl_add_reg_to_lea"] = (
                                        self.stats.get(
                                            "shl_add_reg_to_lea", 0
                                        ) + 1
                                    )
                                    i += 2
                                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_general_register(text: str) -> bool:
        return text.strip().lower() in {
            "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
        }

    @staticmethod
    def _is_label_like(text: str) -> bool:
        """True if text looks like a label or label-arithmetic expr —
        not a memory deref, not a pure numeric literal, not a register
        name."""
        s = text.strip()
        if not s:
            return False
        if s.startswith("[") or s.endswith("]"):
            return False
        if PeepholeOptimizer._is_general_register(s):
            return False
        if s.lower() in {
            "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh",
            "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
        }:
            return False
        # Pure numeric literal (decimal, hex, octal, binary)?
        try:
            int(s, 0)
            return False
        except ValueError:
            pass
        # Looks like a label (or label expression). Could be `_foo`,
        # `_foo + 4`, `.L1_top`, `__heap`, etc.
        return True

    @staticmethod
    def _parse_numeric_immediate(text: str) -> int | None:
        """Parse `42`, `0x1A`, `0o17`, `0b101`. Returns None for
        anything else (registers, memory, labels, expressions)."""
        s = text.strip()
        if not s:
            return None
        try:
            return int(s, 0)
        except ValueError:
            return None

    def _pass_cmp_load_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, [mem]; ...chain...; cmp reg, X`` into
        ``...chain...; cmp dword [mem], X`` when ``reg`` is dead
        after the cmp AND the chain doesn't read/write reg or write
        to [mem]. Also handles ``mov reg, [mem]; ...chain...; test
        reg, reg`` → ``...chain...; cmp dword [mem], 0`` (test r,r
        and cmp r,0 set the same flags for standard Jcc).

        x86 supports ``cmp r/m32, imm`` and ``cmp r/m32, r32``
        directly, so the load can be elided. Saves 2-3 bytes per
        match (drops the mov; cmp gains a memory operand of equal
        or 1-byte-larger encoding).

        The fast path matches the immediate-adjacent shape (chain
        length 0). The slow path walks forward through a chain,
        bailing on:
        - any instruction that reads or writes EAX (we'd lose the
          loaded value or the chain would clobber it).
        - any memory store that may alias [mem].
        - any control flow (label, jmp, ret, call) — calls clobber
          EAX per cdecl.

        Restricted to EAX for now: most cmp+load chains funnel through
        EAX, and limiting scope keeps the safety analysis simple.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                line.kind == "instr"
                and line.op == "mov"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if (
                        dest_low == "eax"
                        and src_norm.startswith("[")
                        and src_norm.endswith("]")
                    ):
                        # Find the cmp/test at the end of the chain.
                        cmp_idx = self._find_cmp_after_load(
                            lines, i + 1, src_norm
                        )
                        if cmp_idx is not None:
                            new_cmp = self._collapse_cmp_with_load(
                                lines[cmp_idx], src_norm
                            )
                            if (
                                new_cmp is not None
                                and self._reg_dead_after(
                                    lines, cmp_idx + 1, "eax"
                                )
                            ):
                                indent = self._extract_indent(
                                    line.raw
                                )
                                new_raw = f"{indent}{new_cmp}"
                                new_line = Line(
                                    raw=new_raw,
                                    kind="instr",
                                    op="cmp",
                                    operands=new_cmp.split(
                                        None, 1
                                    )[1] if " " in new_cmp else "",
                                )
                                # Emit chain (between mov and cmp),
                                # then the new cmp. Drop mov + old
                                # cmp.
                                for k in range(i + 1, cmp_idx):
                                    out.append(lines[k])
                                out.append(new_line)
                                self.stats["cmp_load_collapse"] = (
                                    self.stats.get(
                                        "cmp_load_collapse", 0
                                    ) + 1
                                )
                                i = cmp_idx + 1
                                continue
            out.append(line)
            i += 1
        return out

    def _find_cmp_after_load(
        self, lines: list[Line], start: int, mem: str
    ) -> int | None:
        """Walk forward from `start` looking for `cmp eax, X` (X non-
        memory) or `test eax, eax`. Bail on any chain instruction
        that:
        - references EAX (read or write — dest or operand)
        - stores to memory that may alias `mem`
        - writes to any register used in `mem`'s addressing (e.g., a
          chain `pop ecx` would invalidate `[eax + ecx*4]` since the
          collapsed cmp re-evaluates the SIB with the new ECX)
        - is a label, control flow, or call

        Returns the cmp/test index on success, None on failure.
        """
        # Registers used in `mem`'s addressing (other than EAX, which
        # is already covered by the EAX-reference check).
        mem_regs = self._regs_in_mem_operand(mem) - {"eax"}
        for k in range(start, len(lines)):
            ln = lines[k]
            if ln.kind == "label":
                return None
            if ln.kind != "instr":
                continue
            op = ln.op
            # Found the cmp/test?
            if op in ("cmp", "test"):
                opp = _operands_split(ln.operands or "")
                if opp is None:
                    return None
                d, s = opp
                if d.strip().lower() == "eax":
                    return k
                # Some other cmp/test — bail conservatively.
                return None
            # Control flow / call — bail.
            if op in ("ret", "iret", "iretd", "retf", "retn",
                      "leave", "enter", "call"):
                return None
            if op == "jmp" or op.startswith("j"):
                return None
            # Any reference to EAX (or sub-regs) — bail.
            if (
                _references_register(ln.operands or "", "eax")
                or op in PeepholeOptimizer._IMPLICIT_REG_USERS
            ):
                return None
            # Writes to a register used in `mem`'s addressing — bail.
            # The collapsed cmp will re-evaluate `mem`'s SIB
            # expression at the cmp's location, so a chain write to
            # any of those regs changes which address is read.
            if mem_regs:
                dest_reg = self._instr_dest_reg(ln)
                if dest_reg is not None and dest_reg in mem_regs:
                    return None
            # Memory write that may alias `mem` — bail. The first
            # operand is the destination for most ALU/move ops; for
            # 1-operand RMW ops (inc/dec/not/neg/pop/setCC) the only
            # operand is the dest. Skip known read-only ops where
            # first operand is just a source.
            ops_text = ln.operands or ""
            if ops_text and op not in self._READ_ONLY_FIRST_MEM:
                first_op = (
                    ops_text.split(",", 1)[0].strip()
                    if "," in ops_text
                    else ops_text.strip()
                )
                if "[" in first_op:
                    if not self._mem_disjoint(
                        self._strip_size_prefix(first_op),
                        self._strip_size_prefix(mem.strip()),
                    ):
                        return None
        return None

    @staticmethod
    def _regs_in_mem_operand(mem: str) -> set[str]:
        """Extract register names referenced inside the brackets of a
        memory operand. E.g. `[eax + ecx*4]` → {'eax', 'ecx'}, plus
        their 16/8-bit aliases. Operand may include size prefix.
        """
        # Strip optional size prefix and outer brackets.
        m = mem.strip()
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if m.lower().startswith(prefix):
                m = m[len(prefix):].lstrip()
                break
        if m.startswith("[") and m.endswith("]"):
            m = m[1:-1]
        # Find all register-like words.
        ALL_REGS = {
            "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
            "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
            "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh",
        }
        # Promote to 32-bit canonical names.
        SUBREG_TO_32 = {
            "ax": "eax", "al": "eax", "ah": "eax",
            "bx": "ebx", "bl": "ebx", "bh": "ebx",
            "cx": "ecx", "cl": "ecx", "ch": "ecx",
            "dx": "edx", "dl": "edx", "dh": "edx",
            "si": "esi", "di": "edi",
            "bp": "ebp", "sp": "esp",
        }
        regs = set()
        for tok in re.findall(r"\b[a-zA-Z]+\b", m):
            t = tok.lower()
            if t in ALL_REGS:
                regs.add(SUBREG_TO_32.get(t, t))
        return regs

    @staticmethod
    def _instr_dest_reg(ln: Line) -> str | None:
        """Return the canonical 32-bit register name written by this
        instruction, or None if the dest is memory, the instruction
        has no dest, or the dest can't be cheaply determined.

        Recognized: pop reg, mov reg, X, lea reg, ..., add/sub/and/or
        /xor/imul reg, X, inc reg, dec reg, neg reg, not reg, shl/
        shr/sar reg, ..., setCC reg, movsx/movzx reg, ..., xor reg,
        reg (write — note this also reads, but it's a write).
        """
        if ln.kind != "instr":
            return None
        op = ln.op
        ops = ln.operands or ""
        SUBREG_TO_32 = {
            "ax": "eax", "al": "eax", "ah": "eax",
            "bx": "ebx", "bl": "ebx", "bh": "ebx",
            "cx": "ecx", "cl": "ecx", "ch": "ecx",
            "dx": "edx", "dl": "edx", "dh": "edx",
            "si": "esi", "di": "edi",
            "bp": "ebp", "sp": "esp",
        }
        ALL_REGS = {
            "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
        } | set(SUBREG_TO_32.keys())
        # 1-operand ops where the operand is the dest (and a register).
        if op in {"pop", "inc", "dec", "neg", "not"} and ops:
            tok = ops.strip().lower()
            if tok in ALL_REGS:
                return SUBREG_TO_32.get(tok, tok)
        # setCC reg: dest is the operand.
        if op.startswith("set") and len(op) <= 6 and ops:
            tok = ops.strip().lower()
            if tok in ALL_REGS:
                return SUBREG_TO_32.get(tok, tok)
        # 2-operand ops where dest is the first operand.
        if op in {"mov", "lea", "add", "sub", "and", "or", "xor",
                  "imul", "shl", "shr", "sar", "rol", "ror", "sal",
                  "movsx", "movzx", "adc", "sbb", "cmovz", "cmovnz",
                  "cmove", "cmovne", "cmovl", "cmovle", "cmovg",
                  "cmovge", "cmovs", "cmovns", "cmovo", "cmovno",
                  "cmovb", "cmovbe", "cmova", "cmovae", "cmovc",
                  "cmovnc", "cmovp", "cmovnp"} and ops:
            parts = _operands_split(ops)
            if parts is not None:
                dest = parts[0].strip().lower()
                if dest in ALL_REGS:
                    return SUBREG_TO_32.get(dest, dest)
        return None

    # Ops whose first memory operand is a SOURCE (read-only). All
    # other ops with a memory first operand write to it.
    _READ_ONLY_FIRST_MEM: frozenset[str] = frozenset({
        "cmp", "test",
        "lea",  # dest is reg; mem only as effective addr (read-only)
        "push",
        # Single-operand mul/div/idiv/imul: mem operand is source,
        # result lands in EDX:EAX (caught by IMPLICIT_REG_USERS).
        "mul", "div", "idiv", "imul",
        # Conditional jumps with mem target — control flow, but
        # they're already bailed on at the control-flow check above.
        # Listed here for completeness.
    })

    @staticmethod
    def _collapse_cmp_with_load(
        nxt: Line, src_mem: str
    ) -> str | None:
        """If `nxt` is `cmp eax, X` (X non-memory) or `test eax, eax`,
        return the replacement instruction text. Otherwise None."""
        if nxt.kind != "instr":
            return None
        if nxt.op == "cmp":
            np = _operands_split(nxt.operands)
            if np is None:
                return None
            cdest, csrc = np
            if cdest.strip().lower() != "eax":
                return None
            if "[" in csrc:
                return None
            return f"cmp     dword {src_mem}, {csrc.strip()}"
        if nxt.op == "test":
            np = _operands_split(nxt.operands)
            if np is None:
                return None
            d, s = np
            if (
                d.strip().lower() == "eax"
                and s.strip().lower() == "eax"
            ):
                return f"cmp     dword {src_mem}, 0"
        return None

    def _pass_rmw_intermediate_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Like rmw_collapse but allows intermediate non-touching
        instructions between the load and the OP+store pair.

        Pattern: ``mov eax, [m]; ...; OP eax, REG; mov [m], eax``
        → ``...; OP [m], REG``. Saves 6 bytes per match.

        Intermediate instructions must:
        - Not read or write EAX (or AX/AL/AH).
        - Not read or write [m] (alias check via _mem_overlaps).
        - Not read flags (the OP sets flags; dropping the load+op
          changes the flag state seen between).
        - Not be control flow (label/jump/call/ret).

        After the store:
        - EAX must be dead.
        - Flags must reflect the OP, not the original chain. Since
          the rewrite still emits OP, flags are produced. But the
          flags' VALUE is the same in both versions (`OP eax, REG`
          and `OP [m], REG` produce identical flag bits because
          eax was just loaded from [m], so they're computing the
          same arithmetic).

        OP ∈ {add, sub, and, or, xor}. SRC is a non-EAX register
        (numeric immediate already covered by basic rmw_collapse;
        keeping this pass simpler for register sources only).

        Common in `s += head->val` patterns where the rhs needs an
        intermediate load (e.g. dereferencing through a pointer)."""
        out = list(lines)
        i = 0
        while i < len(out):
            a = out[i]
            if not (
                a.kind == "instr" and a.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            adest, asrc = ap
            adest_low = adest.strip().lower()
            asrc_norm = asrc.strip()
            if adest_low != "eax":
                i += 1
                continue
            if not (
                asrc_norm.startswith("[") and asrc_norm.endswith("]")
            ):
                i += 1
                continue
            mem_addr = asrc_norm
            # Walk forward looking for `OP eax, REG; mov [m], eax`.
            scan_count = 0
            j = i + 1
            op_idx = None
            while j < len(out) and scan_count <= 8:
                s = out[j]
                if s.kind in ("blank", "comment"):
                    j += 1
                    continue
                if s.kind != "instr":
                    break
                # Check if this is the candidate OP.
                if s.op in {"add", "sub", "and", "or", "xor"}:
                    sp = _operands_split(s.operands)
                    if sp is not None:
                        sdst, ssrc = sp
                        ssrc_norm = ssrc.strip()
                        if (
                            sdst.strip().lower() == "eax"
                            and self._rmw_source_text(ssrc_norm)
                                is not None
                            and self._rmw_source_text(ssrc_norm)
                                != "eax"
                        ):
                            # Found candidate OP. Check next is
                            # the matching store.
                            k = j + 1
                            while k < len(out):
                                if out[k].kind in (
                                    "blank", "comment"
                                ):
                                    k += 1
                                    continue
                                break
                            if k < len(out):
                                t = out[k]
                                if t.kind == "instr" and t.op == "mov":
                                    tp = _operands_split(t.operands)
                                    if tp is not None:
                                        tdst, tsrc = tp
                                        if (
                                            tdst.strip() == mem_addr
                                            and tsrc.strip().lower()
                                                == "eax"
                                        ):
                                            op_idx = j
                                            store_idx = k
                                            op_line = s
                                            op_src = ssrc_norm
                                            break
                # Otherwise verify intermediate-safe.
                if self._references_reg_family(s.operands, "eax"):
                    break
                if s.op in self._FLAG_READING_OPS:
                    break
                # Stores or any mem ref overlapping mem_addr blocks
                # the rewrite.
                if (
                    s.op == "mov"
                    and self._operands_modify_mem(
                        s.operands, mem_addr
                    )
                ):
                    break
                if s.op in (
                    "ret", "iret", "retf", "retn", "leave", "enter"
                ):
                    break
                if s.op.startswith("j") or s.op == "call":
                    break
                # Intermediate that doesn't touch our state — ok.
                scan_count += 1
                j += 1
            if op_idx is None:
                i += 1
                continue
            # EAX must be dead after the store.
            if not self._reg_dead_after(out, store_idx + 1, "eax"):
                i += 1
                continue
            # Apply: drop line A and the OP line; rewrite the store
            # as `OP [m], REG`.
            indent = self._extract_indent(op_line.raw)
            spacer = " " * (8 - len(op_line.op))
            new_raw = (
                f"{indent}{op_line.op}{spacer}dword "
                f"{mem_addr}, {op_src}"
            )
            new_line = Line(
                raw=new_raw, kind="instr", op=op_line.op,
                operands=f"dword {mem_addr}, {op_src}",
            )
            # Delete in reverse to keep indices valid.
            del out[store_idx]
            del out[op_idx]
            out.insert(op_idx, new_line)
            del out[i]
            self.stats["rmw_intermediate_collapse"] = (
                self.stats.get("rmw_intermediate_collapse", 0) + 1
            )
            # Don't advance i.
        return out

    def _operands_modify_mem(
        self, operands: str, mem_addr: str
    ) -> bool:
        """Does this instruction's operands modify memory location
        `mem_addr`? Detects `mov [m], src` where m might overlap.
        Conservative: returns True for any memory-write where overlap
        is possible."""
        op_parts = _operands_split(operands)
        if op_parts is None:
            return False
        dest, _ = op_parts
        d = dest.strip()
        if d.startswith("[") or any(
            d.lower().startswith(p)
            for p in ("dword [", "word [", "byte [", "qword [")
        ):
            return self._mem_overlaps(d, mem_addr)
        return False

    def _pass_rmw_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, [mem]; OP reg, SRC; mov [mem], reg``
        into ``OP dword [mem], SRC`` when reg is dead after the store.

        OP ∈ {add, sub, and, or, xor}. SRC is a numeric immediate or
        a non-`reg` general-purpose register. x86 supports both
        ``OP r/m32, imm`` and ``OP r/m32, r32`` forms.

        Saves 5 bytes per match (3-byte load + 3-byte op + 3-byte
        store → single 3-4-byte memory-RMW instruction).

        Common in compound assignments to memory locations:
        ``int x; x += 5;`` (immediate form) and ``x += y;`` (register
        form, where y has been hoisted into a register).

        Conditions:
        - Line A: ``mov reg, [mem]`` (any 32-bit GP reg; plain mov;
          mem must be a memory deref).
        - Line B: ``OP reg, SRC`` where OP is add/sub/and/or/xor and
          SRC is a numeric literal OR a non-`reg` 32-bit register.
        - Line C: ``mov [mem], reg`` (same mem as line A).
        - reg is dead after line C (CFG-aware scan; uses
          treat_as_scratch=True since reg's value is being used purely
          for the RMW).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 2 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    adest, asrc = ap
                    adest_low = adest.strip().lower()
                    asrc_norm = asrc.strip()
                    if (
                        adest_low in {
                            "eax", "ebx", "ecx", "edx", "esi", "edi",
                        }
                        and asrc_norm.startswith("[")
                        and asrc_norm.endswith("]")
                    ):
                        b = lines[i + 1]
                        if (
                            b.kind == "instr"
                            and b.op in {"add", "sub", "and", "or", "xor"}
                        ):
                            bp = _operands_split(b.operands)
                            if bp is not None:
                                bdest, bsrc = bp
                                bsrc_norm = bsrc.strip()
                                src_for_rewrite = (
                                    self._rmw_source_text_for_reg(
                                        bsrc_norm, adest_low,
                                    )
                                )
                                if (
                                    bdest.strip().lower() == adest_low
                                    and src_for_rewrite is not None
                                    and not self._operand_references_reg(
                                        asrc_norm, adest_low
                                    )
                                ):
                                    c = lines[i + 2]
                                    if (
                                        c.kind == "instr"
                                        and c.op == "mov"
                                    ):
                                        cp = _operands_split(c.operands)
                                        if cp is not None:
                                            cdest, csrc = cp
                                            if (
                                                cdest.strip()
                                                == asrc_norm
                                                and csrc.strip().lower()
                                                == adest_low
                                                and self._reg_dead_after(
                                                    lines, i + 3,
                                                    adest_low,
                                                    treat_as_scratch=(
                                                        adest_low
                                                        != "eax"
                                                    ),
                                                )
                                            ):
                                                indent = (
                                                    self._extract_indent(
                                                        line.raw
                                                    )
                                                )
                                                spacer = " " * (
                                                    8 - len(b.op)
                                                )
                                                new_raw = (
                                                    f"{indent}{b.op}"
                                                    f"{spacer}dword "
                                                    f"{asrc_norm}, "
                                                    f"{src_for_rewrite}"
                                                )
                                                new_line = Line(
                                                    raw=new_raw,
                                                    kind="instr",
                                                    op=b.op,
                                                    operands=(
                                                        f"dword "
                                                        f"{asrc_norm}, "
                                                        f"{src_for_rewrite}"
                                                    ),
                                                )
                                                out.append(new_line)
                                                self.stats[
                                                    "rmw_collapse"
                                                ] = (
                                                    self.stats.get(
                                                        "rmw_collapse", 0
                                                    ) + 1
                                                )
                                                i += 3
                                                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _rmw_source_text(src: str) -> str | None:
        """For an `OP eax, src` where the rewrite goes to `OP [mem], src`,
        decide if `src` is acceptable.

        Acceptable:
        - numeric immediate (NASM accepts `OP r/m32, imm8` and imm32)
        - non-EAX 32-bit GP register

        Not acceptable: memory deref (would be mem-mem), EAX itself
        (the EAX value is stale after we drop the load), labels.
        """
        s = src.strip()
        # Numeric immediate.
        imm = PeepholeOptimizer._parse_numeric_immediate(s)
        if imm is not None:
            return str(imm)
        # Non-EAX register.
        sl = s.lower()
        if sl in {"ebx", "ecx", "edx", "esi", "edi"}:
            return sl
        return None

    @staticmethod
    def _rmw_source_text_for_reg(
        src: str, working_reg: str,
    ) -> str | None:
        """For `OP <working_reg>, src` where the rewrite goes to
        `OP [mem], src`, decide if `src` is acceptable.

        Like `_rmw_source_text` but parameterized over the working
        register: `src` may not be `working_reg` (its value is stale
        after we drop the load) but any other GP32 is fine.
        """
        s = src.strip()
        imm = PeepholeOptimizer._parse_numeric_immediate(s)
        if imm is not None:
            return str(imm)
        sl = s.lower()
        if sl in PeepholeOptimizer._GP32 and sl != working_reg.lower():
            return sl
        return None

    @staticmethod
    def _operand_references_reg(operand: str, reg32: str) -> bool:
        """Does `operand` mention `reg32` or any of its sub-register
        aliases? Word-boundary match."""
        sub_aliases = {
            "eax": ("eax", "ax", "al", "ah"),
            "ebx": ("ebx", "bx", "bl", "bh"),
            "ecx": ("ecx", "cx", "cl", "ch"),
            "edx": ("edx", "dx", "dl", "dh"),
            "esi": ("esi", "si"),
            "edi": ("edi", "di"),
            "ebp": ("ebp", "bp"),
            "esp": ("esp", "sp"),
        }
        for alias in sub_aliases.get(reg32.lower(), (reg32.lower(),)):
            if _references_register(operand, alias):
                return True
        return False

    def _pass_rmw_mem_src_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov reg, [m1]; OP reg, [m2]; mov [m1], reg``
        (compound assign with memory rhs) to
        ``mov reg, [m2]; OP [m1], reg``.

        Pattern (3 consecutive instr lines):
            mov     reg, [m1]    ; load destination
            OP      reg, [m2]    ; OP ∈ {add, sub, and, or, xor}
            mov     [m1], reg    ; store back

        Rewrite to:
            mov     reg, [m2]    ; load source instead
            OP      [m1], reg    ; mem-form RMW

        Saves 3 bytes per match: drops the load + store pair, adds
        the same-size load of [m2] and the same-size mem-form OP.

        Differs from `rmw_collapse`: that pass handles immediate or
        register source (no memory). Here both operands are memory,
        which x86's `OP r/m32, imm/r32` doesn't allow — but we can
        still benefit by rearranging which mem becomes the dest.

        Common in compound-assign with memory rhs:
            s += i;     // both s and i in memory
        Lowers to `mov eax, [s]; add eax, [i]; mov [s], eax`
        (after imm_binop_collapse merges the inner load).

        Conditions:
        - Line A: ``mov reg, [m1]`` where reg is GP32, [m1] is mem.
        - Line B: ``OP reg, [m2]`` where OP ∈ {add, sub, and, or, xor},
          same reg, [m2] is mem.
        - Line C: ``mov [m1], reg`` (same m1, same reg).
        - reg dead after C (we change reg's final value).
        - [m1] textually != [m2] (otherwise same_memory_operand_reuse
          already handled the chain differently).
        - [m1] and [m2] are both restricted to ``[ebp ± N]`` or
          ``[_label]`` forms (no register-base derefs to avoid
          aliasing concerns).
        - [m2] doesn't reference reg (would change after the RMW
          modifies dest, but we read m2 BEFORE the RMW, so this is
          actually fine — but be defensive).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 2 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    a_dst, a_src = ap
                    a_dst_low = a_dst.strip().lower()
                    a_src_norm = a_src.strip()
                    if (
                        a_dst_low in PeepholeOptimizer._GP32
                        and self._is_safe_mem_form(a_src_norm)
                    ):
                        b = lines[i + 1]
                        if (
                            b.kind == "instr"
                            and b.op in {"add", "sub", "and", "or", "xor"}
                        ):
                            bp = _operands_split(b.operands)
                            if bp is not None:
                                b_dst, b_src = bp
                                b_dst_low = b_dst.strip().lower()
                                b_src_norm = b_src.strip()
                                if (
                                    b_dst_low == a_dst_low
                                    and self._is_safe_mem_form(
                                        b_src_norm
                                    )
                                    and self._normalize_mem(b_src_norm)
                                    != self._normalize_mem(a_src_norm)
                                    and not _references_register(
                                        b_src_norm, a_dst_low
                                    )
                                ):
                                    c = lines[i + 2]
                                    if (
                                        c.kind == "instr"
                                        and c.op == "mov"
                                    ):
                                        cp = _operands_split(c.operands)
                                        if cp is not None:
                                            c_dst, c_src = cp
                                            c_src_low = (
                                                c_src.strip().lower()
                                            )
                                            if (
                                                self._normalize_mem(
                                                    c_dst.strip()
                                                )
                                                == self._normalize_mem(
                                                    a_src_norm
                                                )
                                                and c_src_low == a_dst_low
                                                and self._reg_dead_after(
                                                    lines, i + 3,
                                                    a_dst_low,
                                                )
                                            ):
                                                indent_a = (
                                                    self._extract_indent(
                                                        line.raw
                                                    )
                                                )
                                                indent_b = (
                                                    self._extract_indent(
                                                        b.raw
                                                    )
                                                )
                                                # New A: mov reg, [m2]
                                                spacer_mov = " " * (
                                                    8 - len("mov")
                                                )
                                                new_a_raw = (
                                                    f"{indent_a}mov"
                                                    f"{spacer_mov}"
                                                    f"{a_dst_low}, "
                                                    f"{b_src_norm}"
                                                )
                                                new_a = Line(
                                                    raw=new_a_raw,
                                                    kind="instr",
                                                    op="mov",
                                                    operands=(
                                                        f"{a_dst_low}, "
                                                        f"{b_src_norm}"
                                                    ),
                                                )
                                                # New B: OP [m1], reg
                                                spacer_b = " " * max(
                                                    1, 8 - len(b.op)
                                                )
                                                new_b_raw = (
                                                    f"{indent_b}{b.op}"
                                                    f"{spacer_b}"
                                                    f"{a_src_norm}, "
                                                    f"{a_dst_low}"
                                                )
                                                new_b = Line(
                                                    raw=new_b_raw,
                                                    kind="instr",
                                                    op=b.op,
                                                    operands=(
                                                        f"{a_src_norm}, "
                                                        f"{a_dst_low}"
                                                    ),
                                                )
                                                out.append(new_a)
                                                out.append(new_b)
                                                # Drop C
                                                self.stats[
                                                    "rmw_mem_src_collapse"
                                                ] = (
                                                    self.stats.get(
                                                        "rmw_mem_src_collapse",
                                                        0,
                                                    ) + 1
                                                )
                                                i += 3
                                                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_safe_mem_form(operand: str) -> bool:
        """Is `operand` a memory operand that's safe for our
        peephole rewrites — i.e., `[ebp ± N]` (frame slot) or
        `[_label]` (global)?

        Excludes register-base addressing (`[eax]`, `[ebx + 4]`,
        SIB forms like `[ebp + ecx*4]`) where aliasing with
        another memory operand could give different results post-
        rewrite.
        """
        s = operand.strip()
        if not (s.startswith("[") and s.endswith("]")):
            return False
        inner = s[1:-1].strip()
        # Form 1: ebp ± N
        if re.match(
            r"^\s*ebp\s*[+-]\s*(\d+|0x[0-9a-fA-F]+)\s*$",
            inner,
            re.IGNORECASE,
        ):
            return True
        # Form 2: bare label _name (or with disp)
        if re.match(
            r"^\s*_[A-Za-z_][\w]*(\s*[+-]\s*(\d+|0x[0-9a-fA-F]+))?\s*$",
            inner,
        ):
            return True
        return False

    def _pass_mov_test_setcc_movzx_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the 4-instruction boolean-materialize pattern:

            mov     reg, [m]      ; load value
            test    reg, reg      ; check for zero
            setCC   al            ; set bool based on flags
            movzx   eax, al       ; zero-extend to 32-bit

        into:

            cmp     dword [m], 0  ; equivalent flags
            setCC   al
            movzx   eax, al

        Saves 1 byte per match. ``mov reg, [ebp ± N]; test reg, reg``
        is 3 + 2 = 5 bytes; ``cmp dword [ebp ± N], 0`` is 4 bytes.

        Common in `x == 0`, `x != 0`, `x > 0`, `!x`, etc. when used
        in expression context (assigned to a variable, returned, used
        in arithmetic). When used in a control-flow context (`if`,
        `while`), `cmp_load_collapse` already handles `mov reg, [m];
        cmp reg, X; jcc` patterns.

        Why is this safe?
        - Original: `mov` loads [m] into reg (full width); `test`
          sets ZF/SF based on reg; `setCC al` writes AL based on
          flags; `movzx eax, al` zero-extends AL into EAX.
        - The high bits of EAX after `mov`/`test`/`setCC al` are
          [m]'s high bytes, but `movzx` discards them. So the
          intermediate full-EAX-load is unused.
        - Replacement: `cmp dword [m], 0` sets ZF/SF identically (both
          test for [m] == 0 and produce SF = high bit of [m]). EAX is
          unchanged before `setCC al`, but `movzx` writes EAX from AL
          alone — final result is the same boolean.

        Conditions:
        - Line A: `mov reg, [m]` where reg is a GP32, [m] is mem.
        - Line B: `test reg, reg` (same reg).
        - Line C: `setCC al` (any setCC variant: sete, setne, setl,
          setg, setle, setge, seta, setb, setae, setbe, sets, setns,
          seto, setno, setp, setnp).
        - Line D: `movzx eax, al`.
        - reg can be ANY GP32 (typically EAX, but could be others
          via earlier passes). The replacement uses memory addressing,
          so reg's value is no longer needed.
        - [m] must not reference EAX (we'll be reading [m] but EAX
          will be the final destination via setCC + movzx).
        """
        out: list[Line] = []
        i = 0
        setcc_ops = {
            "sete", "setne", "setl", "setg", "setle", "setge",
            "seta", "setb", "setae", "setbe",
            "sets", "setns", "seto", "setno",
            "setp", "setnp", "setz", "setnz", "setc", "setnc",
            "setpe", "setpo",
        }
        while i < len(lines):
            line = lines[i]
            if (
                i + 3 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    a_dst, a_src = ap
                    a_dst_low = a_dst.strip().lower()
                    a_src_norm = a_src.strip()
                    if (
                        a_dst_low in PeepholeOptimizer._GP32
                        and a_src_norm.startswith("[")
                        and a_src_norm.endswith("]")
                        and not _references_register(a_src_norm, "eax")
                    ):
                        b = lines[i + 1]
                        c = lines[i + 2]
                        d = lines[i + 3]
                        # Line B: `test reg, reg` (same reg)
                        is_test = (
                            b.kind == "instr"
                            and b.op == "test"
                            and self._is_test_reg_reg(b, a_dst_low)
                        )
                        # Line C: `setCC al`
                        is_setcc_al = (
                            c.kind == "instr"
                            and c.op in setcc_ops
                            and c.operands.strip().lower() == "al"
                        )
                        # Line D: `movzx eax, al`
                        is_movzx = (
                            d.kind == "instr"
                            and d.op == "movzx"
                            and d.operands.replace(" ", "").lower()
                                == "eax,al"
                        )
                        if is_test and is_setcc_al and is_movzx:
                            indent = self._extract_indent(line.raw)
                            spacer = " " * max(1, 8 - len("cmp"))
                            new_a_raw = (
                                f"{indent}cmp{spacer}"
                                f"dword {a_src_norm}, 0"
                            )
                            new_a = Line(
                                raw=new_a_raw, kind="instr",
                                op="cmp",
                                operands=f"dword {a_src_norm}, 0",
                            )
                            # Drop B (test); keep C and D.
                            out.append(new_a)
                            out.append(c)
                            out.append(d)
                            self.stats[
                                "mov_test_setcc_movzx_collapse"
                            ] = (
                                self.stats.get(
                                    "mov_test_setcc_movzx_collapse",
                                    0,
                                ) + 1
                            )
                            i += 4
                            continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_test_reg_reg(line: Line, reg: str) -> bool:
        """Is `line` a `test REG, REG` for the given register?"""
        if line.op != "test":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        a, b = parts
        return (
            a.strip().lower() == reg.lower()
            and b.strip().lower() == reg.lower()
        )

    def _pass_div_mem_form(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, [m]; <ext>; div/idiv reg`` to
        ``<ext>; div/idiv [m]``.

        Pattern (3 consecutive instr lines):
            mov     reg, [m]    ; reg ∈ {ebx, ecx, esi, edi}, [m] is mem
            <ext>               ; cdq (idiv) or xor edx, edx (div)
            div/idiv reg

        Rewrite to:
            <ext>
            div/idiv dword [m]

        Saves 2 bytes per match: drops the 3-byte ``mov reg, [m]``;
        gains 1 byte for the mem-form div/idiv (modrm/sib/disp).

        Common in `a / b` and `a % b` for non-constant divisor.

        Conditions:
        - Line A: ``mov REG, [m]`` where REG is a GP32 != EAX/EDX/ESP,
          [m] is a memory deref.
        - Line B: ``cdq`` (paired with idiv) or ``xor edx, edx``
          (paired with div for unsigned).
        - Line C: ``idiv REG`` or ``div REG`` (matching B).
        - REG dead after C (we drop its only def).
        - [m] doesn't reference EAX or EDX (those are involved in
          div/idiv as the dividend).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 2 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    a_dst, a_src = ap
                    a_dst_low = a_dst.strip().lower()
                    a_src_norm = a_src.strip()
                    if (
                        a_dst_low in {"ebx", "ecx", "esi", "edi"}
                        and a_src_norm.startswith("[")
                        and a_src_norm.endswith("]")
                        and not _references_register(a_src_norm, "eax")
                        and not _references_register(a_src_norm, "edx")
                    ):
                        b = lines[i + 1]
                        c = lines[i + 2]
                        # Match B + C as a paired (ext, div) op.
                        is_idiv = (
                            b.kind == "instr"
                            and b.op == "cdq"
                            and c.kind == "instr"
                            and c.op == "idiv"
                            and c.operands.strip().lower() == a_dst_low
                        )
                        is_div = (
                            b.kind == "instr"
                            and b.op == "xor"
                            and self._is_xor_zero_idiom(b, "edx")
                            and c.kind == "instr"
                            and c.op == "div"
                            and c.operands.strip().lower() == a_dst_low
                        )
                        if (
                            (is_idiv or is_div)
                            and self._reg_dead_after(
                                lines, i + 3, a_dst_low,
                            )
                        ):
                            indent_c = self._extract_indent(c.raw)
                            spacer_c = " " * max(1, 8 - len(c.op))
                            new_c_operands = f"dword {a_src_norm}"
                            new_c_raw = (
                                f"{indent_c}{c.op}{spacer_c}"
                                f"{new_c_operands}"
                            )
                            new_c = Line(
                                raw=new_c_raw, kind="instr",
                                op=c.op, operands=new_c_operands,
                            )
                            # Skip A (the mov), keep B (ext), replace C.
                            out.append(b)
                            out.append(new_c)
                            self.stats["div_mem_form"] = (
                                self.stats.get("div_mem_form", 0) + 1
                            )
                            i += 3
                            continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_xor_zero_idiom(line: Line, reg: str) -> bool:
        """Is `line` a `xor reg, reg` zero-idiom for the given reg?"""
        if line.op != "xor":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        a, b = parts
        return (
            a.strip().lower() == reg.lower()
            and b.strip().lower() == reg.lower()
        )

    def _pass_shift_const_imm(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov ecx, IMM; <shift> reg, cl`` into
        ``<shift> reg, IMM`` for compile-time-constant shift counts.

        Pattern (2 consecutive instr lines):
            mov     ecx, IMM            ; numeric literal in 0..31
            <op>    reg, cl             ; op ∈ {shl, shr, sar, sal,
                                                rol, ror, rcl, rcr,
                                                shld, shrd}

        Replace with:
            <op>    reg, IMM

        Saves 4-5 bytes per match. ``mov ecx, imm32`` is 5 bytes;
        ``shl reg, cl`` is 2 bytes; combined 7 bytes. ``shl reg, 1``
        is 2 bytes (special D1 form), ``shl reg, imm8`` (imm > 1) is
        3 bytes.

        The codegen lowers every constant-count shift through CL —
        ``mov ecx, N; shl eax, cl`` — even when N is a compile-time
        literal. x86 has direct ``shl r/m32, imm8`` so the CL detour
        is wasted bytes.

        Conditions:
        - Line A is ``mov ecx, IMM`` with IMM a numeric literal in
          [0, 31] (the 32-bit shift count is masked by 31 in
          hardware; NASM accepts the imm form for these values).
        - Line B is a shift/rotate op with destination = some 32-bit
          register != ECX, source = literal ``cl``. Sub-word
          destinations (al/ax) are excluded — they have their own
          encoding considerations.
        - ECX must be dead after Line B (we're dropping its only
          definition).
        - shld/shrd are 3-operand instructions (`shld dst, src, cl`);
          the count operand is the third one. Same rewrite pattern.
        """
        out: list[Line] = []
        i = 0
        shift_ops = {
            "shl", "shr", "sar", "sal",
            "rol", "ror", "rcl", "rcr",
            "shld", "shrd",
        }
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    a_dst, a_src = ap
                    if a_dst.strip().lower() == "ecx":
                        imm = self._parse_numeric_immediate(a_src)
                        if imm is not None and 0 <= imm <= 31:
                            b = lines[i + 1]
                            if (
                                b.kind == "instr"
                                and b.op in shift_ops
                                and self._shift_uses_cl(b)
                                and self._shift_dest_not_ecx(b)
                                and self._reg_dead_after(
                                    lines, i + 2, "ecx"
                                )
                            ):
                                indent = self._extract_indent(b.raw)
                                spacer = " " * max(1, 8 - len(b.op))
                                # Replace the trailing `cl` with imm.
                                new_operands = self._replace_cl_with_imm(
                                    b.operands, imm,
                                )
                                new_raw = (
                                    f"{indent}{b.op}{spacer}"
                                    f"{new_operands}"
                                )
                                new_line = Line(
                                    raw=new_raw, kind="instr",
                                    op=b.op,
                                    operands=new_operands,
                                )
                                out.append(new_line)
                                self.stats["shift_const_imm"] = (
                                    self.stats.get(
                                        "shift_const_imm", 0,
                                    ) + 1
                                )
                                i += 2
                                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _shift_uses_cl(line: Line) -> bool:
        """Does this shift instruction use CL as the count operand?
        Two-operand form: `shl reg, cl`. Three-operand (shld/shrd):
        `shld dst, src, cl`."""
        ops = line.operands
        # Look for a trailing `, cl` (case-insensitive).
        return bool(re.search(r",\s*cl\s*$", ops, re.IGNORECASE))

    @staticmethod
    def _shift_dest_not_ecx(line: Line) -> bool:
        """The shift destination (first operand) must not be ECX or
        any of its aliases (or a memory operand referencing ECX).
        We're dropping the `mov ecx, IMM` so the destination must
        not depend on ECX's value."""
        ops = line.operands
        # Split on commas; first operand is the destination.
        first = ops.split(",")[0].strip().lower()
        if first in {"ecx", "cx", "cl", "ch"}:
            return False
        # If destination is memory and it references ECX, skip.
        if "[" in first and PeepholeOptimizer._references_ecx(first):
            return False
        return True

    @staticmethod
    def _replace_cl_with_imm(operands: str, imm: int) -> str:
        """Replace the trailing `cl` in a shift's operands with `imm`.

        For `eax, cl` → `eax, <imm>`.
        For `eax, ebx, cl` → `eax, ebx, <imm>` (shld/shrd form).
        """
        return re.sub(
            r",(\s*)cl(\s*)$",
            f",\\g<1>{imm}\\g<2>",
            operands,
            count=1,
            flags=re.IGNORECASE,
        )

    def _pass_same_memory_operand_reuse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG, [m]; OP REG, [same m]`` into
        ``mov REG, [m]; OP REG, REG`` for commutative OPs.

        After the load, REG holds [m]. For commutative ops where the
        second operand is the SAME memory location, we can use REG
        twice instead of re-reading. The result is identical
        (reg op reg == reg op [m] when reg == [m]).

        Saves 1 byte per match: ``OP r32, [ebp ± imm8]`` is 3 bytes;
        ``OP r32, r32`` is 2 bytes.

        Common after `store_chain_retarget` rewrites the chain for
        ``arr[i] = i + i``: the chain becomes
        ``mov ecx, [i]; add ecx, [i]`` which we collapse here.

        Conditions:
        - Line A: ``mov REG, [m]`` where REG is a 32-bit GP register,
          [m] is a memory deref.
        - Line B: ``OP REG, [m]`` where OP ∈ {add, and, or, xor},
          same REG, same [m] (whitespace-normalized text match).
        - [m] doesn't reference REG itself (would self-modify after
          collapse — but actually safe since the value is the same;
          still defensive).
        - [m] is restricted to ``[ebp ± N]`` (frame-local, never
          volatile in our codegen) for safety against potential
          volatile MMIO loads through globals or pointers.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    a_dst, a_src = ap
                    a_dst_low = a_dst.strip().lower()
                    a_src_norm = a_src.strip()
                    if (
                        a_dst_low in PeepholeOptimizer._GP32
                        and self._is_ebp_relative_mem(a_src_norm)
                    ):
                        b = lines[i + 1]
                        if (
                            b.kind == "instr"
                            and b.op in PeepholeOptimizer._COMMUTATIVE_BINOPS
                        ):
                            bp = _operands_split(b.operands)
                            if bp is not None:
                                b_dst, b_src = bp
                                b_dst_low = b_dst.strip().lower()
                                b_src_norm = b_src.strip()
                                if (
                                    b_dst_low == a_dst_low
                                    and self._normalize_mem(b_src_norm)
                                    == self._normalize_mem(a_src_norm)
                                ):
                                    indent = self._extract_indent(b.raw)
                                    spacer = " " * max(1, 8 - len(b.op))
                                    new_raw = (
                                        f"{indent}{b.op}{spacer}"
                                        f"{a_dst_low}, {a_dst_low}"
                                    )
                                    new_line = Line(
                                        raw=new_raw, kind="instr",
                                        op=b.op,
                                        operands=(
                                            f"{a_dst_low}, {a_dst_low}"
                                        ),
                                    )
                                    out.append(line)
                                    out.append(new_line)
                                    self.stats[
                                        "same_memory_operand_reuse"
                                    ] = (
                                        self.stats.get(
                                            "same_memory_operand_reuse",
                                            0,
                                        ) + 1
                                    )
                                    i += 2
                                    continue
            out.append(line)
            i += 1
        return out

    def _pass_chain_binop_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the linear chain
            mov   ECX, X
            OP    ECX, Y
            OP    EAX, ECX
        to:
            OP    EAX, X
            OP    EAX, Y

        when both inner and outer OP match and OP is associative
        (add/and/or/xor/imul). Saves 1 instruction per match.

        The original computes ``eax = eax OP (X OP Y)``; the rewrite
        computes ``eax = (eax OP X) OP Y``. By associativity, both are
        equal for OP ∈ {+, &, |, ^, *}. (NOT for sub: `eax - (X - Y) =
        eax - X + Y ≠ (eax - X) - Y`.)

        Common shape: ``a + b + c`` chains the codegen evaluates as
        ``mov ecx, [b]; add ecx, [c]; add eax, ecx`` (after
        binop_collapse already merged the inner b+c push/pop). After
        this pass: ``add eax, [b]; add eax, [c]``.

        Conditions:
        - 3 consecutive instr lines.
        - A is `mov ecx, X` (any source).
        - B is `OP ecx, Y` where OP ∈ COMMUTATIVE_BINOPS.
        - C is `OP eax, ecx` (same OP, same ECX).
        - Y must NOT reference EAX (would be read with new EAX value
          after the first rewritten op modifies EAX).
        - Y must NOT reference ECX (would be read with pre-A ECX in
          the rewrite, vs A's ECX=X in the original).
        - X may reference EAX or ECX (both ops read with
          pre-modification values, same in original).
        - ECX dead after C (we drop the only definition of ECX from A).

        Saves 3 bytes per match (drops the `mov ecx, X` line; the
        rewritten ops have the same encoded length as the originals).
        """
        out: list[Line] = []
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            if (
                line.kind == "instr"
                and line.op == "mov"
                and i + 2 < n
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    a_dst, a_src = ap
                    a_dst_low = a_dst.strip().lower()
                    a_src_norm = a_src.strip()
                    if a_dst_low == "ecx":
                        b = lines[i + 1]
                        if (
                            b.kind == "instr"
                            and b.op in (
                                PeepholeOptimizer._COMMUTATIVE_BINOPS
                            )
                        ):
                            bp = _operands_split(b.operands)
                            if bp is not None:
                                b_dst, b_src = bp
                                b_dst_low = b_dst.strip().lower()
                                b_src_norm = b_src.strip()
                                if b_dst_low == "ecx":
                                    c = lines[i + 2]
                                    if (
                                        c.kind == "instr"
                                        and c.op == b.op
                                    ):
                                        cp = _operands_split(
                                            c.operands
                                        )
                                        if cp is not None:
                                            c_dst, c_src = cp
                                            c_dst_low = (
                                                c_dst.strip().lower()
                                            )
                                            c_src_low = (
                                                c_src.strip().lower()
                                            )
                                            if (
                                                c_dst_low == "eax"
                                                and c_src_low == "ecx"
                                                and not (
                                                    _references_register(
                                                        b_src_norm,
                                                        "eax",
                                                    )
                                                    or _references_register(
                                                        b_src_norm,
                                                        "ax",
                                                    )
                                                    or _references_register(
                                                        b_src_norm,
                                                        "al",
                                                    )
                                                    or _references_register(
                                                        b_src_norm,
                                                        "ah",
                                                    )
                                                )
                                                and not (
                                                    _references_register(
                                                        b_src_norm,
                                                        "ecx",
                                                    )
                                                    or _references_register(
                                                        b_src_norm,
                                                        "cx",
                                                    )
                                                    or _references_register(
                                                        b_src_norm,
                                                        "cl",
                                                    )
                                                    or _references_register(
                                                        b_src_norm,
                                                        "ch",
                                                    )
                                                )
                                                and self._reg_dead_after(
                                                    lines, i + 3,
                                                    "ecx",
                                                )
                                            ):
                                                # Build new lines.
                                                indent_a = (
                                                    self._extract_indent(
                                                        line.raw
                                                    )
                                                )
                                                indent_b = (
                                                    self._extract_indent(
                                                        b.raw
                                                    )
                                                )
                                                spacer = " " * max(
                                                    1, 8 - len(b.op)
                                                )
                                                new_a_raw = (
                                                    f"{indent_a}{b.op}"
                                                    f"{spacer}"
                                                    f"eax, {a_src_norm}"
                                                )
                                                new_a = Line(
                                                    raw=new_a_raw,
                                                    kind="instr",
                                                    op=b.op,
                                                    operands=(
                                                        f"eax, "
                                                        f"{a_src_norm}"
                                                    ),
                                                )
                                                new_b_raw = (
                                                    f"{indent_b}{b.op}"
                                                    f"{spacer}"
                                                    f"eax, {b_src_norm}"
                                                )
                                                new_b = Line(
                                                    raw=new_b_raw,
                                                    kind="instr",
                                                    op=b.op,
                                                    operands=(
                                                        f"eax, "
                                                        f"{b_src_norm}"
                                                    ),
                                                )
                                                out.append(new_a)
                                                out.append(new_b)
                                                self.stats[
                                                    "chain_binop_collapse"
                                                ] = (
                                                    self.stats.get(
                                                        "chain_binop_collapse",
                                                        0,
                                                    ) + 1
                                                )
                                                i += 3
                                                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_ebp_relative_mem(operand: str) -> bool:
        """Is `operand` of the form `[ebp + N]` or `[ebp - N]` with a
        numeric literal N? Whitespace-tolerant."""
        s = operand.strip()
        if not (s.startswith("[") and s.endswith("]")):
            return False
        inner = s[1:-1].strip()
        # Match `ebp [+|-] <numeric>`.
        m = re.match(
            r"^\s*ebp\s*([+-])\s*(\d+|0x[0-9a-fA-F]+)\s*$",
            inner,
            re.IGNORECASE,
        )
        return m is not None

    @staticmethod
    def _normalize_mem(operand: str) -> str:
        """Whitespace-normalized lowercase form of a memory operand
        for textual comparison."""
        return "".join(operand.lower().split())

    def _pass_fst_fstp_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse ``fst <addr>; fstp st0`` into a single
        ``fstp <addr>``. Saves 2 bytes per match.

        ``fst <addr>`` stores st0 to memory without popping; ``fstp
        st0`` then pops the FPU stack (writing st0 to st0 first,
        which is a no-op). The combined effect is equivalent to a
        single ``fstp <addr>`` (store-and-pop). Same final FPU
        stack state, same memory write, fewer bytes.

        uc386 lowers ``*p = float_expr`` and similar through this
        pattern — see ``_eval_float_to_st0`` callers that store via
        ``fst`` followed by an explicit ``fstp st0`` cleanup.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "fst"
            ):
                nxt = lines[i + 1]
                if (
                    nxt.kind == "instr"
                    and nxt.op == "fstp"
                    and nxt.operands.strip().lower()
                    in ("st0", "st(0)")
                ):
                    indent = self._extract_indent(line.raw)
                    new_raw = f"{indent}fstp    {line.operands}"
                    new_line = Line(
                        raw=new_raw,
                        kind="instr",
                        op="fstp",
                        operands=line.operands,
                    )
                    out.append(new_line)
                    self.stats["fst_fstp_collapse"] = (
                        self.stats.get("fst_fstp_collapse", 0) + 1
                    )
                    i += 2
                    continue
            out.append(line)
            i += 1
        return out

    # Mapping: pop-form FPU op → memory-form (same direction).
    _FPU_POP_TO_MEM: dict[str, str] = {
        "faddp": "fadd",
        "fmulp": "fmul",
        "fsubp": "fsub",
        "fdivp": "fdiv",
        "fsubrp": "fsubr",
        "fdivrp": "fdivr",
    }

    def _pass_fpu_op_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse ``fld <addr>; faddp st1, st0`` (and friends) into
        a single ``fadd <addr>``. Same FPU stack/memory state, 2
        bytes saved per match.

        Maps:
          fld <addr>; faddp [st1, st0]  → fadd <addr>
          fld <addr>; fmulp [st1, st0]  → fmul <addr>
          fld <addr>; fsubp [st1, st0]  → fsub <addr>
          fld <addr>; fdivp [st1, st0]  → fdiv <addr>
          fld <addr>; fsubrp [st1, st0] → fsubr <addr>
          fld <addr>; fdivrp [st1, st0] → fdivr <addr>

        The pop-form ``faddp st1, st0`` computes ``st1 = st1 + st0;
        pop`` — the new top is ``old_st1 + addr_value``. The memory
        form ``fadd <addr>`` computes ``st0 = st0 + <addr_value>``.
        With the surrounding context (st0 was loaded from <addr>,
        st1 was the previous top), both produce the same final
        ``old_st0 + <addr_value>`` on the new top.

        The pop-form variants with explicit ``st1, st0`` operands
        are accepted; bare ``faddp`` (which defaults to st1, st0)
        is also accepted for completeness, though our codegen
        currently always emits the explicit form.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "fld"
            ):
                nxt = lines[i + 1]
                mem_form = self._FPU_POP_TO_MEM.get(nxt.op or "")
                if mem_form is not None and self._is_pop_st1_st0(nxt):
                    indent = self._extract_indent(line.raw)
                    spacer = " " * (8 - len(mem_form))
                    new_raw = (
                        f"{indent}{mem_form}{spacer}{line.operands}"
                    )
                    new_line = Line(
                        raw=new_raw,
                        kind="instr",
                        op=mem_form,
                        operands=line.operands,
                    )
                    out.append(new_line)
                    self.stats["fpu_op_collapse"] = (
                        self.stats.get("fpu_op_collapse", 0) + 1
                    )
                    i += 2
                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_pop_st1_st0(line: Line) -> bool:
        """Match the ``st1, st0`` operand form (or the bare form,
        which defaults to ``st1, st0``)."""
        ops = line.operands.strip().lower()
        if ops == "":
            return True
        no_space = ops.replace(" ", "")
        return no_space in {"st1,st0", "st(1),st(0)"}

    def _pass_add_one_to_inc(self, lines: list[Line]) -> list[Line]:
        """``add reg, 1`` → ``inc reg`` (and ``sub reg, 1`` →
        ``dec reg``). Saves 2 bytes per register match (3-byte add-imm8
        → 1-byte inc-reg).

        Also handles ``add <size> [mem], 1`` → ``inc <size> [mem]``
        for byte/word/dword sizes — saves 1 byte per match in 32-bit
        mode (e.g., ``add dword [ebp + 8], 1`` is 4 bytes,
        ``inc dword [ebp + 8]`` is 3 bytes; same 1-byte delta for byte
        and word forms).

        ``inc``/``dec`` leave CF unchanged while ``add``/``sub`` set
        CF. The rewrite is safe only when CF is dead after — no
        ``jc/jnc/ja/jb/jae/jbe/setc/setnc/setb/setbe/etc.`` reads CF
        before flags are overwritten by another flag-clobberer or the
        function exits.

        Conservative: uses `_flags_safe_after` which bails on any
        flag-reader (not just CF readers). Most uc386 codegen patterns
        do `add reg, 1` followed by `test`/`cmp`/another arithmetic
        op (which clobbers flags), so the conservative check still
        fires often.

        Applies to all 8 32-bit GP registers and to memory operands
        with explicit `byte`/`word`/`dword` size.
        """
        out: list[Line] = []
        for i, line in enumerate(lines):
            if (
                line.kind == "instr"
                and line.op in ("add", "sub")
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_stripped = dest.strip()
                    dest_low = dest_stripped.lower()
                    is_reg = self._is_general_register(dest_low)
                    is_sized_mem = self._is_sized_memory_operand(
                        dest_stripped
                    )
                    if (
                        (is_reg or is_sized_mem)
                        and src.strip() == "1"
                        and self._flags_safe_after(lines, i + 1)
                    ):
                        new_op = "inc" if line.op == "add" else "dec"
                        indent = self._extract_indent(line.raw)
                        spacer = " " * (8 - len(new_op))
                        # Preserve original spacing/casing for memory
                        # operands; canonicalize register name.
                        new_dest = (
                            dest_low if is_reg else dest_stripped
                        )
                        new_raw = f"{indent}{new_op}{spacer}{new_dest}"
                        new_line = Line(
                            raw=new_raw,
                            kind="instr",
                            op=new_op,
                            operands=new_dest,
                        )
                        out.append(new_line)
                        self.stats["add_one_to_inc"] = (
                            self.stats.get("add_one_to_inc", 0) + 1
                        )
                        continue
            out.append(line)
        return out

    @staticmethod
    def _is_sized_memory_operand(text: str) -> bool:
        """Return True if `text` is a NASM sized memory operand like
        ``dword [ebp + 8]`` / ``byte [eax]`` / ``word [_glob]``.

        Matches the size keyword (byte/word/dword/qword) followed by
        whitespace and a bracketed memory expression. Not strict on
        the bracket contents (any `[...]`).
        """
        m = re.match(
            r"^(byte|word|dword|qword)\s+\[.*\]\s*$",
            text,
            re.IGNORECASE,
        )
        return m is not None

    # Instructions whose ZF/SF are set based on the result AND which
    # ALSO clear OF and CF — the same flag state as `test reg, reg`.
    # After any of these, a subsequent `test reg, reg` (with the same
    # reg as the dest) is fully redundant: ALL flag bits match what
    # the prior op set, so any subsequent Jcc reads the same state.
    #
    # Restricted to logical ops only. add/sub/inc/dec/neg/shl/shr/sar
    # set OF and CF based on overflow / shifted-out bits — those
    # diverge from test's "always 0" — so dropping the test changes
    # the flag state seen by jg/jl/ja/jb/jo/etc. A specific failure:
    # `sub eax, ecx; test eax, eax; jg L` with overflow gives SF != OF,
    # so jg is not-taken; dropping the test gives SF maybe == OF
    # depending on overflow direction. Caught by torture's signed-
    # comparison-after-subtraction tests (20000403-1, bf-sign-2).
    _ZF_SF_SETTING_RMW_OPS: frozenset[str] = frozenset({
        "and", "or", "xor",
    })

    def _pass_redundant_test_collapse(self, lines: list[Line]) -> list[Line]:
        """Drop ``test reg, reg`` when the immediately-preceding
        instruction was a flag-setting arithmetic on the same reg.
        The prior op already set ZF/SF based on its result, so a
        ``test reg, reg`` is redundant — same flag state.

        Saves 2 bytes per match. Common in:

            and  eax, MASK   ; sets flags
            test eax, eax    ; redundant
            jz   target

        becomes:

            and  eax, MASK
            jz   target
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                line.kind == "instr"
                and line.op == "test"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    a, b = parts
                    al = a.strip().lower()
                    bl = b.strip().lower()
                    # Self-test: `test reg, reg` (same reg both sides).
                    if al == bl and self._is_general_register(al):
                        # Look back for the preceding instruction that
                        # set this reg as its destination.
                        prev_idx = self._find_prev_instr(out)
                        if prev_idx is not None:
                            prev = out[prev_idx]
                            if prev.op in self._ZF_SF_SETTING_RMW_OPS:
                                # All current ops in the safe set are
                                # 2-operand (and/or/xor). The
                                # destination is the first operand.
                                pp = _operands_split(prev.operands)
                                if (
                                    pp is not None
                                    and pp[0].strip().lower() == al
                                ):
                                    self.stats[
                                        "redundant_test_collapse"
                                    ] = (
                                        self.stats.get(
                                            "redundant_test_collapse", 0
                                        ) + 1
                                    )
                                    i += 1
                                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _find_prev_instr(out: list[Line]) -> int | None:
        """Return the index of the last `instr` line in `out`, or
        None if there isn't one (or there's a label/directive in
        between, which is a basic-block boundary)."""
        for i in range(len(out) - 1, -1, -1):
            ln = out[i]
            if ln.kind == "instr":
                return i
            if ln.kind in ("label", "directive", "data"):
                return None  # block boundary
        return None

    # Match `byte [...]` or `word [...]` (with optional whitespace).
    _NARROWING_LOAD_SRC_RE = re.compile(
        r"^(byte|word)\s+(\[.*\])\s*$",
        re.IGNORECASE,
    )

    # Jcc's that read ONLY ZF — they're flag-state-invariant under
    # the byte-vs-dword cmp difference. Any other Jcc (jl/jg/ja/jb/
    # js/jno/etc.) reads SF, OF, or CF, whose values DO differ between
    # `test eax, eax` (after movzx of byte 0xFF: SF=0, since EAX bit
    # 31 = 0) and `cmp byte [...], 0` (8-bit subtraction of 0 from
    # 0xFF: SF=1). So the rewrite is unsafe for non-ZF Jcc.
    _ZF_ONLY_JCC: frozenset[str] = frozenset({
        "jz", "jnz", "je", "jne",
    })

    def _pass_narrowing_load_test_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``movsx/movzx eax, byte/word [SRC]; test eax, eax;
        j[n]z LBL`` into ``cmp byte/word [SRC], 0; j[n]z LBL``.

        The narrowing load fills EAX with a sign- or zero-extended
        byte/word from memory. ``test eax, eax`` then checks the result
        for zero. A sign- or zero-extended byte/word is zero iff the
        source byte/word is itself zero — so the same ZF state results
        from a direct ``cmp <size> [SRC], 0`` against the narrow memory
        operand.

        Restricted to ZF-only Jcc (jz/jnz/je/jne). For signed Jcc
        (jl/jg/jge/jle) or unsigned Jcc (jb/ja/jae/jbe), the byte
        comparison sets SF/OF/CF differently than the dword comparison
        does on a movzx-extended value: e.g., for byte 0xFF, byte cmp
        sets SF=1 (8-bit MSB) while ``test eax, eax`` after movzx sets
        SF=0 (32-bit MSB of 0x000000FF). Caught a regression in
        torture's ``doloop-1`` where ``unsigned char z; --z > 0`` used
        ``movzx; cmp; setg; ...; jnz`` — the inner setg-chain collapses
        to ``cmp byte; jg``, which gives the wrong answer for z=255.
        Restricting to ZF-only Jcc avoids the trap.

        Saves 2 bytes per match.

        Conditions:
        - First line is ``movsx eax, byte/word [SRC]`` or
          ``movzx eax, byte/word [SRC]``.
        - Second line is ``test eax, eax``.
        - Third line is a ZF-only Jcc (jz/jnz/je/jne).
        - EAX is dead after the test (CFG-aware scan via
          `_reg_dead_after`).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 2 < len(lines)
                and line.kind == "instr"
                and line.op in ("movsx", "movzx")
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    if dest.strip().lower() == "eax":
                        m = self._NARROWING_LOAD_SRC_RE.match(
                            src.strip()
                        )
                        if m is not None:
                            size = m.group(1).lower()
                            mem = m.group(2)
                            nxt = lines[i + 1]
                            jcc_line = self._next_instr_after(
                                lines, i + 2
                            )
                            if (
                                nxt.kind == "instr"
                                and nxt.op == "test"
                                and jcc_line is not None
                                and jcc_line.op in self._ZF_ONLY_JCC
                            ):
                                np = _operands_split(nxt.operands)
                                if (
                                    np is not None
                                    and np[0].strip().lower() == "eax"
                                    and np[1].strip().lower() == "eax"
                                    and self._reg_dead_after(
                                        lines, i + 2, "eax"
                                    )
                                ):
                                    indent = self._extract_indent(
                                        line.raw
                                    )
                                    new_raw = (
                                        f"{indent}cmp     "
                                        f"{size} {mem}, 0"
                                    )
                                    new_line = Line(
                                        raw=new_raw,
                                        kind="instr",
                                        op="cmp",
                                        operands=(
                                            f"{size} {mem}, 0"
                                        ),
                                    )
                                    out.append(new_line)
                                    self.stats[
                                        "narrowing_load_test_collapse"
                                    ] = (
                                        self.stats.get(
                                            "narrowing_load_test_collapse",
                                            0,
                                        ) + 1
                                    )
                                    i += 2
                                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _next_instr_after(
        lines: list[Line], start: int
    ) -> "Line | None":
        """Return the next ``instr`` line at or after `start`, skipping
        blanks/comments. Returns None at labels/directives/data (basic-
        block boundaries) or end-of-list."""
        for j in range(start, len(lines)):
            ln = lines[j]
            if ln.kind == "instr":
                return ln
            if ln.kind in ("label", "directive", "data"):
                return None
        return None

    def _pass_jcc_jmp_inversion(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``jcc L1; jmp L2; L1:`` into ``j!cc L2; L1:`` —
        invert the condition and drop the unconditional jmp. The
        ``L1:`` label can stay (harmless) or be dropped if no other
        code references it; we leave it in place for simplicity.

        Saves 5 bytes per match (the dropped `jmp imm32` is 5 bytes;
        a short `jmp imm8` is 2 bytes — either way, dropping the jmp
        is a strict win).

        Common in if-else / ternary lowering where one arm collapses
        to a fall-through after `redundant_eax_load`:
            jle .L1_false
            jmp .L_end          ; branch when fallthrough = "true case"
            .L1_false:
            mov eax, [b]        ; "false case"
            .L_end:
        becomes:
            jg .L_end             ; invert condition
            .L1_false:           ; harmless leftover
            mov eax, [b]
            .L_end:
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Match jcc.
            if (
                line.kind == "instr"
                and line.op.startswith("j")
                and line.op != "jmp"
                and line.op[1:] in self._CC_INVERSE
            ):
                cc = line.op[1:]
                jcc_target = line.operands.strip()
                # Look for the next instr line (skip blanks/comments).
                j_idx = i + 1
                while (
                    j_idx < len(lines)
                    and lines[j_idx].kind in ("blank", "comment")
                ):
                    j_idx += 1
                if (
                    j_idx < len(lines)
                    and lines[j_idx].kind == "instr"
                    and lines[j_idx].op == "jmp"
                ):
                    jmp_line = lines[j_idx]
                    jmp_target = jmp_line.operands.strip()
                    # Look for the label after the jmp.
                    k_idx = j_idx + 1
                    while (
                        k_idx < len(lines)
                        and lines[k_idx].kind in ("blank", "comment")
                    ):
                        k_idx += 1
                    if (
                        k_idx < len(lines)
                        and lines[k_idx].kind == "label"
                        and lines[k_idx].label == jcc_target
                    ):
                        # Match! Rewrite jcc with inverted CC and
                        # the jmp's target.
                        inv_cc = self._CC_INVERSE[cc]
                        indent = self._extract_indent(line.raw)
                        new_op = f"j{inv_cc}"
                        spacer = " " * max(8 - len(new_op), 1)
                        new_raw = (
                            f"{indent}{new_op}{spacer}{jmp_target}"
                        )
                        new_line = Line(
                            raw=new_raw,
                            kind="instr",
                            op=new_op,
                            operands=jmp_target,
                        )
                        out.append(new_line)
                        # Drop the jmp (between i and j_idx, also
                        # drop any blanks/comments between).
                        # Keep the label and continue from there.
                        self.stats["jcc_jmp_inversion"] = (
                            self.stats.get("jcc_jmp_inversion", 0) + 1
                        )
                        i = j_idx + 1
                        continue
            out.append(line)
            i += 1
        return out

    def _pass_bool_materialize_collapse_form_b(
        self, lines: list[Line]
    ) -> list[Line]:
        """Form B: the LL-cmp variant where labels are reversed.

        Pattern (with optional `xor edx, edx` for LL-cmp):

            prior_jccs:  jcc .L_false  (taken when condition fails)
            (.L_true:)             [optional]
            mov   eax, 1     ; (no jcc fired → all conditions met)
            jmp   .L_end
            .L_false:        (some jcc fired → result 0)
            xor   eax, eax
            .L_end:
            [xor edx, edx]    ; optional
            test  eax, eax
            jcc   .L_target

        Result: EAX = 1 if no jcc fired, 0 if any did. Inverse of
        Form A's "EAX = 0 default; 1 on hit".

        Consumer semantics:
        - `jnz .L_target`: target taken when result=1 = all conditions
          met. Use synthetic skip + jmp insertion (same as Form A jz).
        - `jz .L_target`: target taken when result=0 = some jcc fired.
          Rewrite each prior `jcc .L_false` → `jcc .L_target`.

        We anchor detection on `mov eax, 1` followed by `jmp .L_end;
        .L_false: xor eax, eax; .L_end:` instead of starting from xor.
        """
        out: list[Line] = list(lines)
        n = len(out)

        i = 0
        while i + 6 < n:
            # A: mov eax, 1
            a = out[i]
            if not (
                a.kind == "instr"
                and a.op == "mov"
                and a.operands.replace(" ", "").lower() == "eax,1"
            ):
                i += 1
                continue
            # B: jmp .L_end
            b = out[i + 1] if i + 1 < n else None
            if b is None or b.kind != "instr" or b.op != "jmp":
                i += 1
                continue
            l_end_target = b.operands.strip()
            if not l_end_target.startswith("."):
                i += 1
                continue
            # C: label .L_false
            c = out[i + 2] if i + 2 < n else None
            if c is None or c.kind != "label":
                i += 1
                continue
            c_label = c.label
            if not c_label.startswith("."):
                i += 1
                continue
            # D: xor eax, eax
            d = out[i + 3] if i + 3 < n else None
            if d is None or d.kind != "instr" or d.op != "xor":
                i += 1
                continue
            if d.operands.replace(" ", "").lower() != "eax,eax":
                i += 1
                continue
            # E: label .L_end (matches l_end_target)
            e = out[i + 4] if i + 4 < n else None
            if e is None or e.kind != "label":
                i += 1
                continue
            if e.label != l_end_target:
                i += 1
                continue
            # F: optional `xor edx, edx`, then `test eax, eax`.
            f_idx = i + 5
            edx_clear = False
            f = out[f_idx] if f_idx < n else None
            if (
                f is not None
                and f.kind == "instr"
                and f.op == "xor"
                and f.operands.replace(" ", "").lower() == "edx,edx"
            ):
                edx_clear = True
                f_idx += 1
                f = out[f_idx] if f_idx < n else None
            if f is None or f.kind != "instr" or f.op != "test":
                i += 1
                continue
            if f.operands.replace(" ", "").lower() != "eax,eax":
                i += 1
                continue
            # G: jnz/jz .L_target
            g_idx = f_idx + 1
            g = out[g_idx] if g_idx < n else None
            if g is None or g.kind != "instr" or g.op not in ("jnz", "jz"):
                i += 1
                continue
            consumer_op = g.op
            l_target = g.operands.strip()
            if not l_target.startswith("."):
                i += 1
                continue
            # EAX dead after consumer (we eliminate the EAX=0/1 result).
            if not self._reg_dead_after(out, g_idx + 1, "eax"):
                i += 1
                continue
            if edx_clear and not self._reg_dead_after(
                out, g_idx + 1, "edx"
            ):
                i += 1
                continue
            block_size = 7 + (1 if edx_clear else 0)
            # Find the function scope.
            scope = ""
            for k in range(i, -1, -1):
                pk = out[k]
                if pk.kind == "label":
                    name = pk.label
                    if not name.startswith("."):
                        scope = name
                        break
            scope_start = None
            scope_end = n
            cur_scope = ""
            for k in range(n):
                pk = out[k]
                if pk.kind == "label":
                    name = pk.label
                    if not name.startswith("."):
                        if cur_scope == scope:
                            scope_end = k
                            break
                        if name == scope:
                            scope_start = k
                            cur_scope = scope
                        else:
                            cur_scope = name
            if scope_start is None:
                i += 1
                continue
            # Collect all jcc references to c_label (=L_false) in scope.
            # ALSO: there may be a `.L_true:` label JUST before A
            # (mov eax, 1) — the codegen sometimes emits that. If
            # present, ensure it's only referenced from within our
            # block (or unreferenced).
            jcc_indices: list[int] = []
            other_ref = False
            for k in range(scope_start, scope_end):
                pk = out[k]
                if pk.kind != "instr":
                    continue
                if pk.kind == "instr" and pk.operands and (
                    c_label in pk.operands
                ):
                    if (
                        pk.op.startswith("j")
                        and pk.op != "jmp"
                        and pk.operands.strip() == c_label
                    ):
                        jcc_indices.append(k)
                    else:
                        other_ref = True
                        break
            if other_ref or not jcc_indices:
                i += 1
                continue
            # Check for an optional `.L_true:` label just before A.
            # If present, drop it as well (only if unreferenced).
            true_label_idx = None
            k = i - 1
            while k >= 0 and out[k].kind in ("blank", "comment"):
                k -= 1
            if k >= 0 and out[k].kind == "label":
                tlabel = out[k].label
                if tlabel.startswith("."):
                    # Check if anything within scope references tlabel.
                    referenced = False
                    for m in range(scope_start, scope_end):
                        pm = out[m]
                        if pm.kind != "instr":
                            continue
                        if pm.operands and tlabel in pm.operands:
                            if pm.operands.strip() == tlabel:
                                referenced = True
                                break
                    if not referenced:
                        true_label_idx = k
            if consumer_op == "jz":
                # Rewrite each jcc to point at l_target directly.
                for k in jcc_indices:
                    pk = out[k]
                    indent = self._extract_indent(pk.raw)
                    spacer = " " * max(1, 8 - len(pk.op))
                    new_raw = f"{indent}{pk.op}{spacer}{l_target}"
                    out[k] = Line(
                        raw=new_raw, kind="instr",
                        op=pk.op, operands=l_target,
                    )
                # Drop block. Also drop the optional .L_true label.
                if true_label_idx is not None:
                    del out[true_label_idx]
                    i_block_start = i - 1
                    n -= 1
                else:
                    i_block_start = i
                del out[i_block_start:i_block_start + block_size]
                n -= block_size
                self.stats["bool_materialize_collapse"] = (
                    self.stats.get(
                        "bool_materialize_collapse", 0
                    ) + 1
                )
            else:
                # consumer_op == "jnz": all-conditions-met goes to
                # target. Insert `jmp .L_target` + skip label;
                # retarget jccs to .L_skip.
                skip_label = self._unique_local_label(
                    out, base="bm_skip"
                )
                for k in jcc_indices:
                    pk = out[k]
                    indent = self._extract_indent(pk.raw)
                    spacer = " " * max(1, 8 - len(pk.op))
                    new_raw = f"{indent}{pk.op}{spacer}{skip_label}"
                    out[k] = Line(
                        raw=new_raw, kind="instr",
                        op=pk.op, operands=skip_label,
                    )
                indent_a = self._extract_indent(out[i].raw)
                jmp_raw = f"{indent_a}jmp     {l_target}"
                jmp_line = Line(
                    raw=jmp_raw, kind="instr",
                    op="jmp", operands=l_target,
                )
                lbl_raw = f"{skip_label}:"
                lbl_line = Line(
                    raw=lbl_raw, kind="label",
                    label=skip_label,
                )
                if true_label_idx is not None:
                    out[true_label_idx:i + block_size] = [
                        jmp_line, lbl_line,
                    ]
                    n -= (block_size + 1) - 2
                else:
                    out[i:i + block_size] = [jmp_line, lbl_line]
                    n -= (block_size - 2)
                self.stats["bool_materialize_collapse"] = (
                    self.stats.get(
                        "bool_materialize_collapse", 0
                    ) + 1
                )
        return out

    def _pass_bool_materialize_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the boolean-materialize-then-test pattern:

            xor    eax, eax
            jmp    .L_end
        .L_true:
            mov    eax, 1
        .L_end:
            test   eax, eax
            jnz    .L_target

        Drop everything from `xor` to the consumer `jnz` and rewrite
        every prior `jcc .L_true` reference to `jcc .L_target`. Save
        7+ bytes per match (the dropped 7-line block).

        For the `jz` consumer form (`jz .L_target` taken when result
        is 0 = no jcc fired), the rewrite generates a synthetic
        label `.bm_skip_N` after the dropped block, retargets each
        prior `jcc .L_true` to `.bm_skip_N`, and inserts `jmp
        .L_target` where the block was. Net savings: 7 lines dropped,
        2 lines added (jmp + label) = 5 lines saved per match. Each
        retargeted jcc keeps its operator; only the target name
        changes.

        Conditions:
        - The pattern is exactly: `xor eax, eax; jmp .L_end;
          .L_true: mov eax, 1; .L_end: test eax, eax; jnz .L_target`.
        - EAX is dead after the consumer's not-taken path (i.e., the
          consumer's fall-through code doesn't read EAX). After our
          rewrite, EAX no longer holds a meaningful value.
        - The .L_true label MUST only be referenced by jcc-to-true
          instructions (and the trampoline's own definition), so the
          materialize block was the only consumer.
        - Both labels are local (`.X` form, NASM-scoped per global).

        Real-world impact: 58 fires across 200 torture tests. Saves
        ~10-15 bytes per fire (7+ bytes from the dropped block, plus
        the savings from each rewritten `jcc` taking a more direct
        target).

        Common in `if (a || b)`-style conditions where the codegen
        materializes the boolean for `test eax, eax` use.
        """
        # Pre-pass: build list of all jcc target references for each
        # local label. Used to verify .L_true is only referenced by
        # jcc-to-true (and not also from elsewhere).
        out: list[Line] = list(lines)
        n = len(out)

        i = 0
        fires = 0
        while i + 6 < n:
            # Find the 7-instruction sequence (xor/jmp/label/mov/label/
            # test/jnz). Allow blanks/comments between in principle, but
            # practically they're contiguous in our codegen.
            a = out[i]
            if not (
                a.kind == "instr"
                and a.op == "xor"
                and a.operands.replace(" ", "").lower() == "eax,eax"
            ):
                i += 1
                continue
            # B: jmp .L_end
            b = out[i + 1] if i + 1 < n else None
            if b is None or b.kind != "instr" or b.op != "jmp":
                i += 1
                continue
            l_end_target = b.operands.strip()
            if not l_end_target.startswith("."):
                i += 1
                continue
            # C: label .L_true
            c = out[i + 2] if i + 2 < n else None
            if c is None or c.kind != "label":
                i += 1
                continue
            c_label = c.label
            if not c_label.startswith("."):
                i += 1
                continue
            # D: mov eax, 1
            d = out[i + 3] if i + 3 < n else None
            if d is None or d.kind != "instr" or d.op != "mov":
                i += 1
                continue
            if d.operands.replace(" ", "").lower() != "eax,1":
                i += 1
                continue
            # E: label .L_end (matches l_end_target)
            e = out[i + 4] if i + 4 < n else None
            if e is None or e.kind != "label":
                i += 1
                continue
            e_label = e.label
            if e_label != l_end_target:
                i += 1
                continue
            # F: test eax, eax. Allow an optional `xor edx, edx`
            # before the test (LL-cmp emits one to clear the high
            # half of the LL result; safe to retain since the
            # rewritten code has no LL semantics anymore).
            f_idx = i + 5
            edx_clear = False
            f = out[f_idx] if f_idx < n else None
            if (
                f is not None
                and f.kind == "instr"
                and f.op == "xor"
                and f.operands.replace(" ", "").lower() == "edx,edx"
            ):
                edx_clear = True
                f_idx += 1
                f = out[f_idx] if f_idx < n else None
            if f is None or f.kind != "instr" or f.op != "test":
                i += 1
                continue
            if f.operands.replace(" ", "").lower() != "eax,eax":
                i += 1
                continue
            # G: jnz .L_target  OR  jz .L_target
            g_idx = f_idx + 1
            g = out[g_idx] if g_idx < n else None
            if g is None or g.kind != "instr" or g.op not in ("jnz", "jz"):
                i += 1
                continue
            consumer_op = g.op
            l_target = g.operands.strip()
            if not l_target.startswith("."):
                i += 1
                continue
            # Verify EAX is dead after G (the not-taken path).
            if not self._reg_dead_after(out, g_idx + 1, "eax"):
                i += 1
                continue
            # If edx_clear is True, also verify EDX is dead — we'll
            # drop the xor edx, edx along with the rest.
            if edx_clear and not self._reg_dead_after(
                out, g_idx + 1, "edx"
            ):
                i += 1
                continue
            # Compute the size of the block to drop: 7 instrs base +
            # 1 if edx_clear.
            block_size = 7 + (1 if edx_clear else 0)
            # Find the function scope this lives in.
            scope = ""
            for k in range(i, -1, -1):
                pk = out[k]
                if pk.kind == "label":
                    name = pk.label
                    if not name.startswith("."):
                        scope = name
                        break
            # Find all references to c_label within this scope. They
            # must all be `jcc c_label` (any condition).
            scope_start = None
            scope_end = n
            cur_scope = ""
            for k in range(n):
                pk = out[k]
                if pk.kind == "label":
                    name = pk.label
                    if not name.startswith("."):
                        if cur_scope == scope:
                            scope_end = k
                            break
                        if name == scope:
                            scope_start = k
                            cur_scope = scope
                        else:
                            cur_scope = name
            if scope_start is None:
                i += 1
                continue
            # Collect all jcc references to c_label within the scope.
            jcc_indices: list[int] = []
            other_ref = False
            for k in range(scope_start, scope_end):
                pk = out[k]
                if pk.kind != "instr":
                    continue
                if k == i + 1:
                    continue  # the trampoline's own jmp
                if pk.kind == "instr" and pk.operands and (
                    c_label in pk.operands
                ):
                    # Must be a conditional jump.
                    if (
                        pk.op.startswith("j")
                        and pk.op != "jmp"
                        and pk.operands.strip() == c_label
                    ):
                        jcc_indices.append(k)
                    else:
                        # Unexpected reference (jmp, data, lea, etc.)
                        other_ref = True
                        break
            if other_ref or not jcc_indices:
                i += 1
                continue
            # All checks passed.
            if consumer_op == "jnz":
                # Rewrite each jcc to point at l_target.
                for k in jcc_indices:
                    pk = out[k]
                    indent = self._extract_indent(pk.raw)
                    spacer = " " * max(1, 8 - len(pk.op))
                    new_raw = f"{indent}{pk.op}{spacer}{l_target}"
                    out[k] = Line(
                        raw=new_raw, kind="instr",
                        op=pk.op, operands=l_target,
                    )
                # Drop the entire block.
                del out[i:i + block_size]
                n -= block_size
                fires += 1
                self.stats["bool_materialize_collapse"] = (
                    self.stats.get("bool_materialize_collapse", 0) + 1
                )
            else:
                # consumer_op == "jz": replace the block with
                # `jmp .L_target` + new synthetic skip label. Each
                # prior jcc retargets .L_or_true → .L_skip.
                # Generate a unique skip label name.
                skip_label = self._unique_local_label(
                    out, base="bm_skip"
                )
                # Rewrite each jcc's target.
                for k in jcc_indices:
                    pk = out[k]
                    indent = self._extract_indent(pk.raw)
                    spacer = " " * max(1, 8 - len(pk.op))
                    new_raw = f"{indent}{pk.op}{spacer}{skip_label}"
                    out[k] = Line(
                        raw=new_raw, kind="instr",
                        op=pk.op, operands=skip_label,
                    )
                # Replace block with `jmp .L_target` + label.
                indent_a = self._extract_indent(out[i].raw)
                jmp_raw = f"{indent_a}jmp     {l_target}"
                jmp_line = Line(
                    raw=jmp_raw, kind="instr",
                    op="jmp", operands=l_target,
                )
                lbl_raw = f"{skip_label}:"
                lbl_line = Line(
                    raw=lbl_raw, kind="label",
                    label=skip_label,
                )
                out[i:i + block_size] = [jmp_line, lbl_line]
                n -= (block_size - 2)
                fires += 1
                self.stats["bool_materialize_collapse"] = (
                    self.stats.get("bool_materialize_collapse", 0) + 1
                )
            # Don't advance i — the new state at i is whatever
            # followed G; let the loop reconsider.
        return out

    def _unique_local_label(
        self, lines: list[Line], base: str = "bm_skip",
    ) -> str:
        """Return a synthetic local label name (`.<base>_N`) that's
        not already defined or referenced anywhere in `lines`. Used
        when a peephole pass needs to insert a new label."""
        # Collect all label names defined or referenced.
        used: set[str] = set()
        for ln in lines:
            if ln.kind == "label":
                used.add(ln.label)
            if ln.kind == "instr" and ln.operands:
                for tok in ln.operands.split(","):
                    tok = tok.strip()
                    if tok.startswith("."):
                        used.add(tok)
        # Find first unused.
        n = 0
        while True:
            cand = f".{base}_{n}"
            if cand not in used:
                return cand
            n += 1

    # Map NASM size keyword to the EAX-family sub-register that
    # carries 0 when EAX = 0.
    _ZERO_INIT_SIZE_TO_REG: dict[str, str] = {
        "byte": "al",
        "word": "ax",
        "dword": "eax",
    }

    def _pass_zero_init_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace 1+ adjacent ``mov <size> [m], 0`` stores with
        ``xor eax, eax`` followed by per-store ``mov [m], <reg>``.

        Each ``mov dword [m], 0`` is 7 bytes; the rewrite produces
        ``mov [m], eax`` at 3 bytes plus one upfront ``xor eax, eax``
        at 2 bytes. Net savings:
        - 1 store: 7 → 5 (xor + mov), save 2 bytes.
        - N stores: 7N → 2 + 3N, save 4N - 2 bytes.

        Byte stores: 4 → 3 (1 byte saved + 2 bytes xor amortized).
        Word stores: 5 (66-prefixed) → 4, save 1 byte.

        Conditions:
        - 1+ adjacent ``mov <size> [m], 0`` instructions (size in
          {byte, word, dword}; mixed sizes within a chain are OK).
        - EAX is dead after the chain (we'll write 0 to it).
        - Flags are safe after the chain (xor sets ZF/SF/PF/CF/OF;
          subsequent flag-readers would see different state).

        Common in function prologues with multiple zero-initialized
        locals (e.g. ``int a = 0, b = 0, c = 0;``).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not (
                line.kind == "instr"
                and self._is_zero_imm_store(line)
            ):
                out.append(line)
                i += 1
                continue
            # Found a zero-imm-store. Collect adjacent ones.
            chain_idx = [i]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.kind in ("blank", "comment"):
                    j += 1
                    continue
                if (
                    nxt.kind == "instr"
                    and self._is_zero_imm_store(nxt)
                ):
                    chain_idx.append(j)
                    j += 1
                    continue
                break
            # Validity: EAX dead after, flags safe after.
            after_idx = j
            if not self._reg_dead_after(lines, after_idx, "eax"):
                # Conservative: skip the whole chain (don't fire).
                for k in chain_idx:
                    out.append(lines[k])
                i = j
                continue
            if not self._flags_safe_after(lines, after_idx):
                for k in chain_idx:
                    out.append(lines[k])
                i = j
                continue
            # Emit xor + per-store rewrites.
            indent = self._extract_indent(line.raw)
            out.append(Line(
                raw=f"{indent}xor     eax, eax",
                kind="instr",
                op="xor",
                operands="eax, eax",
            ))
            for k in chain_idx:
                ld = lines[k]
                parts = _operands_split(ld.operands)
                if parts is None:
                    # Shouldn't happen if _is_zero_imm_store
                    # accepted it; bail conservatively.
                    out.append(ld)
                    continue
                dest, _ = parts
                # Strip the size keyword and pick the matching reg.
                size_kw, mem = self._split_sized_mem(dest.strip())
                reg = self._ZERO_INIT_SIZE_TO_REG.get(
                    size_kw.lower(), "eax"
                )
                out.append(Line(
                    raw=f"{indent}mov     {mem}, {reg}",
                    kind="instr",
                    op="mov",
                    operands=f"{mem}, {reg}",
                ))
            self.stats["zero_init_collapse"] = (
                self.stats.get("zero_init_collapse", 0)
                + len(chain_idx)
            )
            i = j
        return out

    def _pass_same_imm_store_share_reg(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse 2+ adjacent ``mov <size> [m], imm`` stores with
        the SAME non-zero immediate into a register-share form.

        Pattern (N >= 2 consecutive instrs):
            mov <size> [m1], imm
            mov <size> [m2], imm
            ...
            mov <size> [mN], imm

        Rewrite (when EAX is dead after):
            mov eax, imm        ; imm is non-zero
            mov [m1], <reg>     ; reg = eax / ax / al per size
            mov [m2], <reg>
            ...
            mov [mN], <reg>

        Saves bytes for non-zero imms with 2+ stores. Each direct
        ``mov dword [m], imm32`` is 7 bytes (with disp8) or 10 bytes
        (with disp32). The rewrite uses 5 bytes for the upfront
        mov-imm32 plus 3 bytes per store. For dword:
        - 1 store: 7 → 8, +1 byte (skipped — not worth it)
        - 2 stores: 14 → 11, save 3 bytes
        - 3 stores: 21 → 14, save 7 bytes
        - N stores: 7N → 5 + 3N, save 4N - 5 bytes

        For word/byte sizes the savings differ but still positive
        for N >= 2.

        Conditions:
        - 2+ adjacent stores with same imm and same size
        - imm is non-zero (zero is handled by zero_init_collapse)
        - Each store's imm fits in the encoded form
        - EAX dead after the chain
        - Flags safe after (mov-imm32 doesn't set flags, but the
          chain might be followed by code that reads flags from a
          prior cmp; safe by virtue of mov not setting flags)

        Common in array/struct initializers like ``int a[5] = {7,7,7,7,7}``
        or ``struct s = {.a=42, .b=42, .c=42}``.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Find a non-zero same-imm store
            store_info = self._is_nonzero_imm_store(line)
            if store_info is None:
                out.append(line)
                i += 1
                continue
            # Collect adjacent stores with SAME imm and SAME size
            target_imm, target_size = store_info
            chain_idx = [i]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.kind in ("blank", "comment"):
                    j += 1
                    continue
                if nxt.kind == "instr":
                    nxt_info = self._is_nonzero_imm_store(nxt)
                    if (
                        nxt_info is not None
                        and nxt_info == (target_imm, target_size)
                    ):
                        chain_idx.append(j)
                        j += 1
                        continue
                break
            # Need 2+ for the rewrite to be a win
            if len(chain_idx) < 2:
                out.append(line)
                i += 1
                continue
            # Validity: EAX dead after, flags safe after
            after_idx = j
            if not self._reg_dead_after(lines, after_idx, "eax"):
                for k in chain_idx:
                    out.append(lines[k])
                i = j
                continue
            # Emit `mov eax, imm` + per-store `mov [m], <reg>`
            indent = self._extract_indent(line.raw)
            out.append(Line(
                raw=f"{indent}mov     eax, {target_imm}",
                kind="instr",
                op="mov",
                operands=f"eax, {target_imm}",
            ))
            for k in chain_idx:
                ld = lines[k]
                parts = _operands_split(ld.operands)
                if parts is None:
                    out.append(ld)
                    continue
                dest, _ = parts
                size_kw, mem = self._split_sized_mem(dest.strip())
                # Pick the matching sub-register for the store size
                size_to_reg = {
                    "dword": "eax",
                    "word": "ax",
                    "byte": "al",
                }
                reg = size_to_reg.get(size_kw.lower(), "eax")
                out.append(Line(
                    raw=f"{indent}mov     {mem}, {reg}",
                    kind="instr",
                    op="mov",
                    operands=f"{mem}, {reg}",
                ))
            self.stats["same_imm_store_share_reg"] = (
                self.stats.get("same_imm_store_share_reg", 0)
                + len(chain_idx)
            )
            i = j
        return out

    def _is_nonzero_imm_store(
        self, line: Line
    ) -> tuple[str, str] | None:
        """Return (imm_text, size_kw) for a ``mov <size> [m], imm``
        with non-zero imm. Otherwise None.

        The imm is returned as its source text (e.g., "7", "0x42",
        "-100"). The size_kw is "byte"/"word"/"dword".

        For zero values (which would conflict with zero_init_collapse),
        returns None.
        """
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        src_text = src.strip()
        # Reject zero (handled by zero_init_collapse).
        if src_text == "0":
            return None
        if (
            src_text.lower().startswith("0x")
            and all(c == "0" for c in src_text[2:])
        ):
            return None
        # Source must be a numeric immediate (not memory, register,
        # or label).
        if (
            src_text.startswith("[")
            or self._is_general_register(src_text.lower())
            or self._is_subregister(src_text.lower())
        ):
            return None
        # Try parsing as a number to confirm it's an integer literal.
        # Accept decimal, hex, octal, and negative.
        try:
            int(src_text, 0)
        except ValueError:
            return None
        # Dest must be a sized memory operand
        m = re.match(
            r"^(byte|word|dword)\s+(\[.*\])\s*$",
            dest.strip(),
            re.IGNORECASE,
        )
        if m is None:
            return None
        size_kw = m.group(1).lower()
        return (src_text, size_kw)

    @staticmethod
    def _is_subregister(text: str) -> bool:
        """Return True if text is a sub-register like ax, al, ah,
        bx, etc."""
        return text.lower() in {
            "ax", "al", "ah",
            "bx", "bl", "bh",
            "cx", "cl", "ch",
            "dx", "dl", "dh",
            "si", "di", "bp", "sp",
        }

    @staticmethod
    def _is_zero_imm_store(line: Line) -> bool:
        """Return True if this is ``mov <byte|word|dword> [m], 0``.
        Only sized memory destinations are recognized (NASM requires
        the size keyword for memory + immediate stores).

        Recognizes both decimal `0` and hex `0x0` / `0x00000000` forms
        for the source — the codegen sometimes emits hex zero (notably
        in long-long zero-init patterns).
        """
        if line.kind != "instr" or line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        src_text = src.strip()
        if src_text != "0":
            low = src_text.lower()
            if not (
                low.startswith("0x")
                and len(low) > 2
                and all(c == "0" for c in low[2:])
            ):
                return False
        m = re.match(
            r"^(byte|word|dword)\s+\[.*\]\s*$",
            dest.strip(),
            re.IGNORECASE,
        )
        return m is not None

    @staticmethod
    def _split_sized_mem(text: str) -> tuple[str, str]:
        """Split ``"<size> [m]"`` into ``("<size>", "[m]")``."""
        m = re.match(
            r"^(byte|word|dword|qword)\s+(\[.*\])\s*$",
            text,
            re.IGNORECASE,
        )
        if m is None:
            return ("", text)
        return (m.group(1), m.group(2))

    # Map sub-register name → enclosing 32-bit register. Writing to
    # any sub-register changes the 32-bit reg's value, so we must
    # invalidate the "is zero" state for the enclosing reg.
    _SUBREG_TO_GP32: dict[str, str] = {
        "ax": "eax", "al": "eax", "ah": "eax",
        "bx": "ebx", "bl": "ebx", "bh": "ebx",
        "cx": "ecx", "cl": "ecx", "ch": "ecx",
        "dx": "edx", "dl": "edx", "dh": "edx",
        "si": "esi",
        "di": "edi",
        "bp": "ebp",
        "sp": "esp",
    }

    def _pass_disp_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``add REG, DISP; mov DST, [REG]`` into
        ``mov DST, [REG + DISP]`` using x86's disp addressing form.

        Saves bytes:
        - DISP fits in imm8 (-128..127): save 2 bytes (add 3 bytes
          + mov 2 bytes → mov 3 bytes).
        - DISP needs imm32: save 1 byte (add 5/6 bytes + mov 2 bytes
          → mov 6 bytes).

        Common in `p->member` struct member access where the
        codegen emits an explicit `add reg, offset` before the
        dereference, even for small offsets.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``add REG, DISP`` (DISP is a numeric literal).
        - Line B: ``mov DST, [REG]`` (plain deref of REG, no
          existing offset/SIB).
        - REG either == DST (overwritten by the load) or dead after
          line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "add"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            reg = ap[0].strip().lower()
            try:
                disp = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if not self._is_general_register(reg):
                i += 1
                continue
            dst_reg = bp[0].strip().lower()
            src = bp[1].strip()
            if not self._is_general_register(dst_reg):
                i += 1
                continue
            # Source must be `[REG]` (no offset, no SIB).
            src_stripped = src
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if src_stripped.lower().startswith(prefix):
                    src_stripped = src_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            mem_re = re.match(
                r"^\[([a-zA-Z]+)\s*\]$", src_stripped
            )
            if mem_re is None:
                i += 1
                continue
            mem_reg = mem_re.group(1).lower()
            if mem_reg != reg:
                i += 1
                continue
            # Liveness: REG dead after line B (we drop the add).
            # If REG == DST, the mov overwrites REG anyway — safe.
            if reg != dst_reg and not self._reg_dead_after(
                out, i + 2, reg
            ):
                i += 1
                continue
            # Build the rewritten mov.
            indent = self._extract_indent(b.raw)
            sign = "+" if disp >= 0 else "-"
            new_src = f"[{reg} {sign} {abs(disp)}]"
            new_raw = f"{indent}mov     {dst_reg}, {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{dst_reg}, {new_src}",
            )
            new_out = out[:i] + [new_line] + out[i + 2:]
            out = new_out
            self.stats["disp_load_collapse"] = (
                self.stats.get(
                    "disp_load_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_disp_store_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``add REG, DISP; ...; mov [REG], SRC`` into
        ``...; mov [REG + DISP], SRC`` using x86's disp addressing
        form.

        Mirror of disp_load_collapse for stores. Saves bytes:
        - DISP fits in imm8: save 2 bytes (3-byte add + 2-byte store
          modrm → 3-byte store modrm with disp8).
        - DISP needs imm32: save 1 byte similarly.

        Tolerates up to 8 intermediate instructions between the add
        and the store, provided each one:
        - Doesn't read or write REG (we'd lose REG's pre-add value
          which the rewrite needs to use directly).
        - Doesn't read flags (the add sets flags; dropping the add
          changes the observable flag state).

        After the store:
        - REG must be dead (the rewrite leaves REG with the pre-add
          value).
        - Flags must be safe (the add's flag effects aren't observed
          via the rewrite).

        Common in struct member assignment after the codegen produces
        `add reg, offset; mov src, ...; mov [reg], src`. Pairs with
        `load_add_xfer_forward` which collapses the prior
        load+add+transfer into a direct load+add to the dest reg.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            a = out[i]
            if not (a.kind == "instr" and a.op == "add"):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            reg = ap[0].strip().lower()
            try:
                disp = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if not self._is_general_register(reg):
                i += 1
                continue
            # Walk forward up to 8 instr lines; each must:
            # - not reference REG
            # - not be a flag reader
            # - not be a control-flow op (label, jump, ret)
            # Stop when we find `mov [REG], SRC` (the store) or a
            # blocker.
            store_idx = None
            scan_count = 0
            j = i + 1
            while j < len(out) and scan_count <= 8:
                s = out[j]
                if s.kind in ("blank", "comment"):
                    j += 1
                    continue
                if s.kind != "instr":
                    # Label / directive / data — can't continue.
                    break
                # Check if this is the candidate store.
                if s.op == "mov":
                    sp = _operands_split(s.operands)
                    if sp is not None:
                        s_dst, s_src = sp
                        s_dst_stripped = s_dst.strip()
                        size_prefix = ""
                        for prefix in (
                            "dword ", "word ", "byte ", "qword "
                        ):
                            if s_dst_stripped.lower().startswith(
                                prefix
                            ):
                                size_prefix = s_dst_stripped[
                                    :len(prefix)
                                ]
                                s_dst_stripped = s_dst_stripped[
                                    len(prefix):
                                ].lstrip()
                                break
                        mem_re = re.match(
                            r"^\[([a-zA-Z]+)\s*\]$", s_dst_stripped
                        )
                        if (
                            mem_re is not None
                            and mem_re.group(1).lower() == reg
                            and not self._references_reg_family(
                                s_src.strip(), reg
                            )
                        ):
                            # Found the store. Verify the size
                            # prefix is preserved properly.
                            store_idx = j
                            store_size_prefix = size_prefix
                            store_src = s_src.strip()
                            break
                # Not the candidate store. Check that this
                # intermediate doesn't break the rewrite.
                if self._references_reg_family(s.operands, reg):
                    break
                if s.op in self._FLAG_READING_OPS:
                    break
                # Control-flow ops break the scan.
                if s.op in (
                    "ret", "iret", "retf", "retn", "leave", "enter"
                ):
                    break
                if s.op.startswith("j") or s.op == "call":
                    break
                scan_count += 1
                j += 1
            if store_idx is None:
                i += 1
                continue
            # REG must be dead after the store.
            if not self._reg_dead_after(out, store_idx + 1, reg):
                i += 1
                continue
            # Flags must be safe after the store (we drop the add's
            # flag-side effect).
            if not self._flags_safe_after(out, store_idx + 1):
                i += 1
                continue
            # Apply: drop line A, rewrite line at store_idx with
            # `mov [REG + DISP], SRC`.
            indent = self._extract_indent(out[store_idx].raw)
            sign = "+" if disp >= 0 else "-"
            new_dst = (
                f"{store_size_prefix}[{reg} {sign} {abs(disp)}]"
            )
            new_raw = f"{indent}mov     {new_dst}, {store_src}"
            new_line = Line(
                raw=new_raw, kind="instr", op="mov",
                operands=f"{new_dst}, {store_src}",
            )
            out[store_idx] = new_line
            del out[i]
            self.stats["disp_store_collapse"] = (
                self.stats.get("disp_store_collapse", 0) + 1
            )
            # Don't advance i — the deletion shifted everything
            # down by 1, and the new line at i may be another add.
        return out

    def _pass_index_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``shl IDX, N; add BASE, IDX; mov DST, [BASE]``
        (or ``mov DST, [BASE + DISP]``) into a SIB-form load.
        ``[BASE]`` → ``[BASE + IDX*SCALE]``;
        ``[BASE + DISP]`` → ``[BASE + IDX*SCALE + DISP]``.

        x86 supports the SIB byte form ``[base + index*scale]``
        (with optional 8/32-bit displacement) directly in mov memory
        operands, eliminating the need for explicit shl + add to
        compute the address. Saves 4 bytes per match (shl + add).

        Conditions:
        - Three consecutive instr lines (no labels/comments between).
        - Line A: ``shl IDX, N`` where N ∈ {1, 2, 3} (scale 2/4/8;
          scale 1 = no-op shl, irrelevant).
        - Line B: ``add BASE, IDX`` (different reg).
        - Line C: ``mov DST, [BASE]`` or ``mov DST, [BASE +/- DISP]``.
        - IDX must be dead after Line C (we drop the shl, so IDX
          retains its pre-shl value; subsequent code expecting the
          shifted value would be wrong).
        - BASE must be either the same reg as DST (overwritten by
          the load) OR dead after Line C.

        Restricted to GP 32-bit registers.
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        # mov, movsx, movzx — all support SIB-form memory operand.
        LOAD_OPS = {"mov", "movsx", "movzx"}
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "add"
                and c.kind == "instr" and c.op in LOAD_OPS
            ):
                i += 1
                continue
            # Parse line A.
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            # Parse line B.
            bp = _operands_split(b.operands)
            if bp is None:
                i += 1
                continue
            base_reg = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(base_reg)
                or b_src != idx_reg
                or base_reg == idx_reg
            ):
                i += 1
                continue
            # Parse line C.
            cp = _operands_split(c.operands)
            if cp is None:
                i += 1
                continue
            dst_reg = cp[0].strip().lower()
            c_src = cp[1].strip()
            if not self._is_general_register(dst_reg):
                i += 1
                continue
            # Source must be `[BASE]` or `[BASE +/- DISP]`.
            c_src_stripped = c_src
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if c_src_stripped.lower().startswith(prefix):
                    size_prefix = c_src_stripped[:len(prefix)]
                    c_src_stripped = c_src_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)"
                r"(?:\s*([+-])\s*(\d+))?"
                r"\s*\]$",
                c_src_stripped,
            )
            if mem_re is None:
                i += 1
                continue
            mem_reg = mem_re.group(1).lower()
            if mem_reg != base_reg:
                i += 1
                continue
            disp_sign = mem_re.group(2)
            disp_val = mem_re.group(3)
            # Liveness: IDX dead after line C; BASE either ==
            # DST (overwritten) or dead after.
            if not self._reg_dead_after(out, i + 3, idx_reg):
                i += 1
                continue
            if base_reg != dst_reg and not self._reg_dead_after(
                out, i + 3, base_reg
            ):
                i += 1
                continue
            # Rewrite: drop A and B; replace C with SIB-form load.
            # Preserve the original op (mov / movsx / movzx).
            indent = self._extract_indent(c.raw)
            scale = SCALE[n]
            if disp_sign is None:
                new_src = (
                    f"{size_prefix}[{base_reg} + {idx_reg}*{scale}]"
                )
            else:
                new_src = (
                    f"{size_prefix}[{base_reg} + {idx_reg}*{scale} "
                    f"{disp_sign} {disp_val}]"
                )
            opname = c.op
            spacer = " " * max(8 - len(opname), 1)
            new_raw = (
                f"{indent}{opname}{spacer}{dst_reg}, {new_src}"
            )
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op=opname,
                operands=f"{dst_reg}, {new_src}",
            )
            new_out = out[:i] + [new_line] + out[i + 3:]
            out = new_out
            self.stats["index_load_collapse"] = (
                self.stats.get(
                    "index_load_collapse", 0
                ) + 1
            )
            # Restart from the rewrite point.
            continue
        return out

    def _pass_index_load_collapse_label(
        self, lines: list[Line]
    ) -> list[Line]:
        """Sister of `index_load_collapse` for label-base global
        array indexing.

        Pattern:
            shl IDX, N
            add IDX, LABEL              ; LABEL is a non-register
                                        ; symbolic expression
            mov/movsx/movzx DST, [IDX]

        Rewrite to a SIB-form load with the label as displacement:
            mov/movsx/movzx DST, [LABEL + IDX*SCALE]

        x86's SIB byte plus disp32 form supports `[disp32 + idx*N]`,
        so the explicit shl + add are unneeded. Saves ~8 bytes per
        match (drops shl 3 bytes + add reg, label 5+ bytes; SIB form
        with disp32 is the same encoding cost as the original mov).

        Common in global short/long arrays (``unsigned short g[5]``)
        where the codegen emits the explicit address computation
        instead of going directly to SIB form.

        Conditions:
        - Three consecutive instr lines.
        - Line A: ``shl IDX, N`` where N ∈ {1, 2, 3} (scale 2/4/8).
        - Line B: ``add IDX, X`` where X is a non-register
          expression (label, label-arithmetic).
        - Line C: ``mov/movsx/movzx DST, [IDX]`` (no displacement —
          we'd lose track if the address already had a displacement
          since IDX would have it folded in).
        - DST == IDX, OR IDX dead after Line C (we drop the shl, so
          IDX retains its pre-shl original-index value).

        Restricted to GP 32-bit registers.
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        LOAD_OPS = {"mov", "movsx", "movzx"}
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "add"
                and c.kind == "instr" and c.op in LOAD_OPS
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            # Parse line B: `add IDX, LABEL`.
            bp = _operands_split(b.operands)
            if bp is None:
                i += 1
                continue
            b_dst = bp[0].strip().lower()
            b_src = bp[1].strip()
            if b_dst != idx_reg:
                i += 1
                continue
            # b_src must be a non-register, non-numeric expression
            # (a label or label-arithmetic). Reject regs and ints.
            if self._is_general_register(b_src.lower()):
                i += 1
                continue
            try:
                int(b_src)
                # It's a numeric literal — treat as plain
                # disp_load_collapse, not our pattern.
                i += 1
                continue
            except ValueError:
                pass
            # Reject if b_src contains `[` (memory ref).
            if "[" in b_src:
                i += 1
                continue
            # Parse line C.
            cp = _operands_split(c.operands)
            if cp is None:
                i += 1
                continue
            dst_reg = cp[0].strip().lower()
            c_src = cp[1].strip()
            if not self._is_general_register(dst_reg):
                i += 1
                continue
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if c_src.lower().startswith(prefix):
                    size_prefix = c_src[:len(prefix)]
                    c_src = c_src[len(prefix):].lstrip()
                    break
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$",
                c_src,
            )
            if mem_re is None:
                i += 1
                continue
            mem_reg = mem_re.group(1).lower()
            if mem_reg != idx_reg:
                i += 1
                continue
            # IDX must be dead after, unless DST == IDX.
            if dst_reg != idx_reg:
                if not self._reg_dead_after(out, i + 3, idx_reg):
                    i += 1
                    continue
            # Rewrite.
            indent = self._extract_indent(c.raw)
            scale = SCALE[n]
            new_src = (
                f"{size_prefix}[{b_src} + {idx_reg}*{scale}]"
            )
            opname = c.op
            spacer = " " * max(8 - len(opname), 1)
            new_raw = (
                f"{indent}{opname}{spacer}{dst_reg}, {new_src}"
            )
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op=opname,
                operands=f"{dst_reg}, {new_src}",
            )
            new_out = out[:i] + [new_line] + out[i + 3:]
            out = new_out
            self.stats["index_load_collapse_label"] = (
                self.stats.get(
                    "index_load_collapse_label", 0
                ) + 1
            )
            continue
        return out

    def _pass_mov_label_shl_add_load_to_sib(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the 4-line global-array-index pattern produced
        when the index is in EAX (so the codegen materializes the
        base label into a separate scratch register first).

        Pattern (4 consecutive instr lines):
            A: mov BASE, LABEL
            B: shl IDX, N            ; N in {1, 2, 3}
            C: add IDX, BASE
            D: mov/movsx/movzx DST, [IDX (+/- DISP)?]

        Rewrite:
            D': mov/movsx/movzx DST, [LABEL + IDX*scale (+/- DISP)?]

        Drops A, B, C entirely. Saves 3 instructions / ~10 bytes.

        Common shape: a for-loop's body where the loop counter sits
        in EAX (loaded for the cmp at the loop top), then the body
        derefs `g[i]` (zero disp) or `s_arr[i].member` (non-zero
        disp for member access). Codegen picks a fresh register
        (EDX) to hold the global base because EAX is busy with the
        index.

        Conditions:
        - BASE, IDX, DST are 32-bit GP registers; BASE != IDX.
        - LABEL is non-numeric, non-memory, non-register.
        - N in {1, 2, 3} (scale 2/4/8).
        - D's mem operand is `[IDX]` or `[IDX + DISP]` /
          `[IDX - DISP]` with literal DISP.
        - BASE dead after D (the original sequence ended with
          BASE = LABEL; after rewrite BASE keeps its pre-A value).
        - IDX dead after D OR DST == IDX (original ended with
          IDX = LABEL + idx*scale; after rewrite IDX keeps its
          pre-A unscaled value unless overwritten by D).
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        LOAD_OPS = {"mov", "movsx", "movzx"}
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            d = out[i + 3]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "shl"
                and c.kind == "instr" and c.op == "add"
                and d.kind == "instr" and d.op in LOAD_OPS
            ):
                i += 1
                continue
            # Parse A: `mov BASE, LABEL`.
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            base_reg = ap[0].strip().lower()
            label = ap[1].strip()
            if not self._is_general_register(base_reg):
                i += 1
                continue
            # LABEL must be non-register, non-numeric, non-memory.
            if self._is_general_register(label.lower()):
                i += 1
                continue
            try:
                int(label)
                i += 1
                continue
            except ValueError:
                pass
            try:
                int(label, 16)
                i += 1
                continue
            except ValueError:
                pass
            if "[" in label:
                i += 1
                continue
            # Parse B: `shl IDX, N`.
            bp = _operands_split(b.operands)
            if bp is None:
                i += 1
                continue
            idx_reg = bp[0].strip().lower()
            try:
                n = int(bp[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            if idx_reg == base_reg:
                i += 1
                continue
            # Parse C: `add IDX, BASE`.
            cp = _operands_split(c.operands)
            if cp is None:
                i += 1
                continue
            c_dst = cp[0].strip().lower()
            c_src = cp[1].strip().lower()
            if c_dst != idx_reg or c_src != base_reg:
                i += 1
                continue
            # Parse D: `mov/movsx/movzx DST, [IDX]` (plain deref).
            dp = _operands_split(d.operands)
            if dp is None:
                i += 1
                continue
            dst_reg = dp[0].strip().lower()
            d_src = dp[1].strip()
            if not self._is_general_register(dst_reg):
                i += 1
                continue
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if d_src.lower().startswith(prefix):
                    size_prefix = d_src[:len(prefix)]
                    d_src = d_src[len(prefix):].lstrip()
                    break
            # Match `[IDX]` or `[IDX +/- DISP]`.
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*"
                r"(?:([+-])\s*(\d+|0x[0-9a-fA-F]+))?\s*\]$",
                d_src,
            )
            if mem_re is None:
                i += 1
                continue
            mem_reg = mem_re.group(1).lower()
            if mem_reg != idx_reg:
                i += 1
                continue
            disp_sign = mem_re.group(2)
            disp_val = mem_re.group(3)
            # BASE must be dead after D (the rewrite leaves BASE at
            # its pre-A value; original ended with BASE = LABEL).
            # treat_as_scratch=True since BASE is a scratch holding a
            # label address, not a return-value setup.
            if not self._reg_dead_after(
                out, i + 4, base_reg, treat_as_scratch=True
            ):
                i += 1
                continue
            # IDX must be dead after D, unless DST == IDX (D
            # overwrites IDX, so its pre-D value doesn't escape).
            if dst_reg != idx_reg:
                if not self._reg_dead_after(
                    out, i + 4, idx_reg, treat_as_scratch=True
                ):
                    i += 1
                    continue
            # Rewrite.
            indent = self._extract_indent(d.raw)
            scale = SCALE[n]
            disp_text = ""
            if disp_sign and disp_val:
                disp_text = f" {disp_sign} {disp_val}"
            new_src = (
                f"{size_prefix}"
                f"[{label} + {idx_reg}*{scale}{disp_text}]"
            )
            opname = d.op
            spacer = " " * max(8 - len(opname), 1)
            new_raw = (
                f"{indent}{opname}{spacer}{dst_reg}, {new_src}"
            )
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op=opname,
                operands=f"{dst_reg}, {new_src}",
            )
            out = out[:i] + [new_line] + out[i + 4:]
            self.stats["mov_label_shl_add_load_to_sib"] = (
                self.stats.get(
                    "mov_label_shl_add_load_to_sib", 0
                ) + 1
            )
            continue
        return out

    def _pass_chain_label_to_add_operand(
        self, lines: list[Line]
    ) -> list[Line]:
        """Fold a `mov BASE, LABEL` setup into a later `add OTHER,
        BASE` (or sub/and/or/xor/cmp/test/adc/sbb), turning it into
        `add OTHER, LABEL`. Saves 1-2 bytes per match (the dropped
        `mov BASE, LABEL` is 5 bytes; the new `<OP> OTHER, LABEL`
        with imm32 source is 5 bytes for EAX, 6 for other regs —
        net savings 2 bytes for EAX dest, 1 byte otherwise; the
        original `<OP> OTHER, BASE` reg-reg form was 2 bytes).

        Common shape: codegen for non-power-of-2 struct arrays:
            mov     edx, _g_arr           ; A
            mov     eax, [ebp + 8]
            imul    eax, eax, 12          ; intermediate
            add     eax, edx              ; B
        After the rewrite:
            mov     eax, [ebp + 8]
            imul    eax, eax, 12
            add     eax, _g_arr

        Pattern:
        - Line A: `mov BASE, LABEL` where BASE is 32-bit GP reg,
          LABEL is non-numeric/non-memory/non-register.
        - Up to 8 intermediate instructions that don't read or
          write BASE, don't have calls (caller-saved BASE would
          clobber), don't have labels or control flow boundaries.
        - Line B: `<OP> OTHER, BASE` where OP ∈ {add, sub, and,
          or, xor, cmp, test, adc, sbb}, OTHER != BASE, OTHER is
          a 32-bit GP reg.
        - BASE dead after Line B (treat_as_scratch=True since
          BASE is a scratch holding a label address).

        imul is excluded because x86 has no `imul reg, imm32` form
        compatible with our 2-operand pattern (only the 3-operand
        form supports immediates, which is a different shape).
        """
        OPS = {
            "add", "sub", "and", "or", "xor",
            "cmp", "test", "adc", "sbb",
        }
        out = list(lines)
        i = 0
        while i < len(out):
            a = out[i]
            if not (
                a.kind == "instr" and a.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            base_reg = ap[0].strip().lower()
            label = ap[1].strip()
            if not self._is_general_register(base_reg):
                i += 1
                continue
            # LABEL must be non-register, non-numeric, non-memory.
            if self._is_general_register(label.lower()):
                i += 1
                continue
            try:
                int(label)
                i += 1
                continue
            except ValueError:
                pass
            try:
                int(label, 16)
                i += 1
                continue
            except ValueError:
                pass
            if "[" in label:
                i += 1
                continue
            # Walk forward up to 8 instructions looking for
            # `<OP> OTHER, BASE`.
            found = False
            j = i + 1
            scanned = 0
            while j < len(out) and scanned < 8:
                ln = out[j]
                if ln.kind in ("blank", "comment"):
                    j += 1
                    continue
                if ln.kind in ("label", "directive", "data"):
                    break  # control-flow boundary
                if ln.kind != "instr":
                    break
                # Calls clobber caller-saved regs.
                if ln.op == "call":
                    break
                # jmp/jcc/ret end the chain.
                if ln.op == "jmp" or (
                    ln.op.startswith("j") and ln.op != "jmp"
                ):
                    break
                if ln.op in {
                    "ret", "iret", "iretd", "retf", "retn",
                    "leave", "enter",
                }:
                    break
                # Check if this is line B.
                if ln.op in OPS:
                    bp = _operands_split(ln.operands)
                    if bp is not None:
                        b_dst = bp[0].strip().lower()
                        b_src = bp[1].strip().lower()
                        if (
                            b_src == base_reg
                            and b_dst != base_reg
                            and self._is_general_register(b_dst)
                        ):
                            # Make sure b_dst isn't BASE-itself
                            # (handled above) and that B reads
                            # only BASE in the source position.
                            # Now check intermediates don't touch
                            # BASE (we walked past them). Also
                            # check BASE dead after B.
                            if self._reg_dead_after(
                                out, j + 1, base_reg,
                                treat_as_scratch=True,
                            ):
                                found = True
                                b_idx = j
                                b_line = ln
                            break
                # If this instruction reads or writes BASE,
                # bail (chain is broken).
                if self._references_reg_family(
                    ln.operands, base_reg
                ):
                    break
                # If implicit-reg-using instr (cdq/idiv/mul/etc.)
                # touches our reg family, bail.
                base_to_implicit = {
                    "eax": True, "edx": True,
                }
                if base_reg in base_to_implicit:
                    if not (
                        PeepholeOptimizer
                        ._has_explicit_operands_only(ln)
                    ):
                        break
                scanned += 1
                j += 1
            if not found:
                i += 1
                continue
            # Rewrite: drop line A, replace line B with `OP b_dst,
            # LABEL`.
            indent = self._extract_indent(b_line.raw)
            opname = b_line.op
            spacer = " " * max(8 - len(opname), 1)
            new_raw = (
                f"{indent}{opname}{spacer}{b_dst}, {label}"
            )
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op=opname,
                operands=f"{b_dst}, {label}",
            )
            new_out = (
                out[:i] + out[i + 1:b_idx]
                + [new_line] + out[b_idx + 1:]
            )
            out = new_out
            self.stats["chain_label_to_add_operand"] = (
                self.stats.get(
                    "chain_label_to_add_operand", 0
                ) + 1
            )
            # Don't advance i — the rewrite may have shifted
            # things and the same i could match again with a
            # different mov.
            continue
        return out

    def _pass_push_disp_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``add REG, DISP; push dword [REG]`` into
        ``push dword [REG + DISP]`` using x86's disp addressing form.

        Sister of ``disp_load_collapse`` for the push case. Saves
        2 bytes per match when DISP fits in imm8, 1 byte for imm32.
        Common in `push p->member` arg-setup paths.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``add REG, DISP`` (DISP is a numeric literal).
        - Line B: ``push dword [REG]`` (plain deref, no offset/SIB).
        - REG dead after Line B (push doesn't write REG).
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "add"
                and b.kind == "instr" and b.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            reg = ap[0].strip().lower()
            try:
                disp = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if not self._is_general_register(reg):
                i += 1
                continue
            # Push source: `[REG]` only (with optional size prefix).
            b_src = b.operands.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_src.lower().startswith(prefix):
                    b_src = b_src[len(prefix):].lstrip()
                    break
            mem_re = re.match(r"^\[([a-zA-Z]+)\s*\]$", b_src)
            if mem_re is None:
                i += 1
                continue
            if mem_re.group(1).lower() != reg:
                i += 1
                continue
            # REG dead after the push (push reads but doesn't write).
            if not self._reg_dead_after(out, i + 2, reg):
                i += 1
                continue
            indent = self._extract_indent(b.raw)
            sign = "+" if disp >= 0 else "-"
            new_src = f"dword [{reg} {sign} {abs(disp)}]"
            new_raw = f"{indent}push    {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="push",
                operands=new_src,
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["push_disp_collapse"] = (
                self.stats.get("push_disp_collapse", 0) + 1
            )
            continue
        return out

    def _pass_index_store_xfer_collapse_pre_compute(self, lines: list[Line]):
        """Pre-compute unused-regs per line for use in the full SIB
        rewrite. Returns a per-line list (or empty list if no SIB
        opportunities are likely)."""
        return self._compute_unused_regs_per_line(lines)

    def _pass_index_store_xfer_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the codegen's array-index-store pattern by dropping
        the address-transfer step.

        Pattern (5 consecutive instrs):
            shl IDX, N            ; A: scale idx
            add BASE, IDX         ; B: BASE = address (base + idx*scale)
            mov XFER, BASE        ; C: XFER = address (transfer)
            mov BASE, SRC         ; D: BASE = src value (clobbers address)
            mov [XFER], BASE      ; E: store src at address

        Rewrite (drops C, swaps roles of XFER and BASE in D/E):
            shl IDX, N            ; KEEP
            add BASE, IDX         ; KEEP — BASE = address still
            mov XFER, SRC         ; D': XFER = src value (BASE preserved)
            mov [BASE], XFER      ; E': store via BASE (still address)

        Saves 1 instruction per match (drops the xfer mov).

        The codegen emits the original 5-line shape for every
        ``arr[i] = val`` statement (or any single-step assignment to a
        register-base lvalue). The xfer's only purpose is to free up
        BASE so the rhs eval can reuse EAX. By using XFER as the rhs
        target instead, BASE keeps the address and the xfer disappears.

        Conditions:
        - 5 consecutive instr lines (no labels/blanks/comments).
        - A: ``shl IDX, N`` (N ∈ {1, 2, 3}).
        - B: ``add BASE, IDX``, BASE != IDX, both GP regs.
        - C: ``mov XFER, BASE``, XFER != BASE.
        - D: ``mov/movsx/movzx BASE, SRC`` (BASE is dest, gets clobbered).
        - E: ``mov [XFER], BASE`` (with optional size prefix).
        - SRC must NOT reference XFER (or its sub-regs) — XFER's value
          differs at the SRC-read site after the rewrite.
        - All of BASE, IDX, XFER dead after E.
        - Flags must be safe after E (we keep A and B's flag-setting,
          so this is automatically OK — same flag state as before).
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        LOAD_OPS = {"mov", "movsx", "movzx"}
        out = list(lines)
        unused_per_line: list[set[str] | None] | None = None
        i = 0
        while i + 4 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            d = out[i + 3]
            e = out[i + 4]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "add"
                and c.kind == "instr" and c.op == "mov"
                and d.kind == "instr" and d.op in LOAD_OPS
                and e.kind == "instr" and e.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            cp = _operands_split(c.operands)
            dp = _operands_split(d.operands)
            ep = _operands_split(e.operands)
            if any(p is None for p in (ap, bp, cp, dp, ep)):
                i += 1
                continue
            # Line A: shl IDX, N
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if (
                n not in SCALE
                or not self._is_general_register(idx_reg)
            ):
                i += 1
                continue
            # Line B: add BASE, IDX
            base_reg = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(base_reg)
                or b_src != idx_reg
                or base_reg == idx_reg
            ):
                i += 1
                continue
            # Line C: mov XFER, BASE
            xfer_reg = cp[0].strip().lower()
            c_src = cp[1].strip().lower()
            if (
                not self._is_general_register(xfer_reg)
                or c_src != base_reg
                or xfer_reg == base_reg
            ):
                i += 1
                continue
            # Line D: mov BASE, SRC (D's dest must be BASE; gets clobbered)
            d_dst = dp[0].strip().lower()
            d_src = dp[1].strip()
            if d_dst != base_reg:
                i += 1
                continue
            # SRC must not reference XFER's family — XFER's value
            # differs at the SRC-read site after the rewrite.
            if self._references_reg_family(d_src, xfer_reg):
                i += 1
                continue
            # Line E: mov [XFER], BASE (BASE as src; XFER as memory base)
            e_dst = ep[0].strip()
            e_src = ep[1].strip().lower()
            if e_src != base_reg:
                i += 1
                continue
            # E_dst must be `[XFER]` exactly (no displacement / index).
            # Allow optional size prefix.
            e_stripped = e_dst
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if e_stripped.lower().startswith(prefix):
                    size_prefix = e_stripped[:len(prefix)]
                    e_stripped = e_stripped[len(prefix):].lstrip()
                    break
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", e_stripped
            )
            if mem_re is None or mem_re.group(1).lower() != xfer_reg:
                i += 1
                continue
            # Liveness: BASE, IDX, XFER all dead after E.
            if not self._reg_dead_after(out, i + 5, base_reg):
                i += 1
                continue
            if not self._reg_dead_after(out, i + 5, idx_reg):
                i += 1
                continue
            if (
                xfer_reg != idx_reg
                and not self._reg_dead_after(out, i + 5, xfer_reg)
            ):
                i += 1
                continue
            # Try the FULL SIB rewrite first: drop A, B, C, D, E and
            # emit `mov R, SRC; mov [BASE + IDX*scale], R` where R is
            # a free register distinct from BASE/IDX/XFER and not
            # referenced in SRC. Saves 3 instructions instead of 1.
            #
            # Conditions for full SIB:
            # - SRC must not reference BASE or IDX (their values
            #   differ post-rewrite — A/B are dropped, so BASE/IDX
            #   stay at pre-pattern values, not address/scaled).
            # - A free register R available (try EDX, ESI, EDI, EBX
            #   in order). R must not be BASE, IDX, XFER. R must be
            #   dead after E. R must not be referenced in SRC.
            # - Flags must be safe after E (we drop A's and B's
            #   flag-setting).
            full_sib_done = False
            if (
                not self._references_reg_family(d_src, base_reg)
                and not self._references_reg_family(d_src, idx_reg)
                and self._flags_safe_after(out, i + 5)
            ):
                # Find a free register. A register is "free" if it's
                # unused throughout the entire current function — no
                # instruction reads or writes it. That's strictly
                # safer than a forward `_reg_dead_after` scan, which
                # may return False for a register that's truly
                # unused but whose pure-write isn't found within the
                # 20-instruction scan limit.
                if unused_per_line is None:
                    unused_per_line = (
                        self._compute_unused_regs_per_line(out)
                    )
                func_unused = unused_per_line[i] or set()
                for free_reg in ("edx", "esi", "edi", "ebx"):
                    if free_reg in (base_reg, idx_reg, xfer_reg):
                        continue
                    if self._references_reg_family(d_src, free_reg):
                        continue
                    if free_reg not in func_unused:
                        continue
                    # Build full SIB rewrite.
                    scale = SCALE[n]
                    indent_d = self._extract_indent(d.raw)
                    indent_e = self._extract_indent(e.raw)
                    d_op = d.op
                    d_spacer = " " * max(8 - len(d_op), 1)
                    new_d_raw = (
                        f"{indent_d}{d_op}{d_spacer}{free_reg}, "
                        f"{d_src}"
                    )
                    new_d = Line(
                        raw=new_d_raw,
                        kind="instr",
                        op=d_op,
                        operands=f"{free_reg}, {d_src}",
                    )
                    new_e_dst = (
                        f"{size_prefix}[{base_reg} + "
                        f"{idx_reg}*{scale}]"
                    )
                    new_e_raw = (
                        f"{indent_e}mov     {new_e_dst}, {free_reg}"
                    )
                    new_e = Line(
                        raw=new_e_raw,
                        kind="instr",
                        op="mov",
                        operands=f"{new_e_dst}, {free_reg}",
                    )
                    out = out[:i] + [new_d, new_e] + out[i + 5:]
                    self.stats["index_store_sib_full"] = (
                        self.stats.get(
                            "index_store_sib_full", 0
                        ) + 1
                    )
                    full_sib_done = True
                    break
            if full_sib_done:
                continue
            # Fall back to simpler rewrite: drop C, replace D and E.
            indent_d = self._extract_indent(d.raw)
            indent_e = self._extract_indent(e.raw)
            # D': mov/movsx/movzx XFER, SRC (use XFER as dest)
            d_op = d.op
            d_spacer = " " * max(8 - len(d_op), 1)
            new_d_raw = (
                f"{indent_d}{d_op}{d_spacer}{xfer_reg}, {d_src}"
            )
            new_d = Line(
                raw=new_d_raw,
                kind="instr",
                op=d_op,
                operands=f"{xfer_reg}, {d_src}",
            )
            # E': mov [BASE], XFER (use BASE as memory, XFER as src)
            new_e_dst = f"{size_prefix}[{base_reg}]"
            new_e_raw = (
                f"{indent_e}mov     {new_e_dst}, {xfer_reg}"
            )
            new_e = Line(
                raw=new_e_raw,
                kind="instr",
                op="mov",
                operands=f"{new_e_dst}, {xfer_reg}",
            )
            out = (
                out[:i]
                + [a, b, new_d, new_e]
                + out[i + 5:]
            )
            self.stats["index_store_xfer_collapse"] = (
                self.stats.get(
                    "index_store_xfer_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_push_index_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``shl IDX, N; add BASE, IDX; push dword [BASE]``
        into ``push dword [BASE + IDX*SCALE]`` (SCALE = 2^N).

        Sister of ``index_load_collapse`` for the push case. Saves
        4 bytes per match. Common in cdecl arg-push of array elements
        like ``f(arr[i])``.

        Conditions:
        - Three consecutive instr lines.
        - Line A: ``shl IDX, N`` where N ∈ {1, 2, 3}.
        - Line B: ``add BASE, IDX`` (different reg).
        - Line C: ``push dword [BASE]`` (plain deref of BASE).
        - Both IDX and BASE dead after Line C (push doesn't write
          either; we drop the shl and add).
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "add"
                and c.kind == "instr" and c.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            base_reg = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(base_reg)
                or b_src != idx_reg
                or base_reg == idx_reg
            ):
                i += 1
                continue
            # Push source: `[BASE]` only.
            c_src = c.operands.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if c_src.lower().startswith(prefix):
                    c_src = c_src[len(prefix):].lstrip()
                    break
            mem_re = re.match(r"^\[([a-zA-Z]+)\s*\]$", c_src)
            if mem_re is None:
                i += 1
                continue
            if mem_re.group(1).lower() != base_reg:
                i += 1
                continue
            # IDX and BASE both dead after the push.
            if not self._reg_dead_after(out, i + 3, idx_reg):
                i += 1
                continue
            if not self._reg_dead_after(out, i + 3, base_reg):
                i += 1
                continue
            indent = self._extract_indent(c.raw)
            scale = SCALE[n]
            new_src = f"dword [{base_reg} + {idx_reg}*{scale}]"
            new_raw = f"{indent}push    {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="push",
                operands=new_src,
            )
            out = out[:i] + [new_line] + out[i + 3:]
            self.stats["push_index_collapse"] = (
                self.stats.get("push_index_collapse", 0) + 1
            )
            continue
        return out

    # Compound-assign binops (used by _pass_compound_assign_collapse).
    # imul is excluded — `imul r/m32` (one-operand) writes EDX:EAX,
    # not the obvious in-place form.
    _COMPOUND_BINOPS: frozenset[str] = frozenset({
        "add", "sub", "and", "or", "xor",
    })

    def _pass_compound_assign_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the compound-assignment frame into an in-place
        memory-RMW.

        Match (looking from a final ``mov [m], eax``):
            push    dword [m]              ; L_push
            ... chain ...                   ; produces value in EAX
            mov     ecx, eax                ; L_pop - 3
            pop     eax                     ; L_pop - 2
            <OP>    eax, ecx                ; L_pop - 1
            mov     [m], eax                ; L_pop

        Replace with:
            ... chain ...                   ; unchanged
            <OP>    [m], eax                ; in-place RMW

        Drops the entire framing (push, mov ecx eax, pop, OP eax ecx,
        mov [m] eax) — saves 8 bytes per match.

        Conditions:
        - L_push must be ``push dword [m]`` with same `m` as the final
          store.
        - The chain (between L_push and L_pop - 3) must not modify
          [m] (no stores to that address) and must be stack-balanced
          (no other push/pop or sub/add esp).
        - ECX must be dead after L_pop (typically true since ECX is
          a scratch register in our codegen).

        Common in compound assignments to slot-typed lvalues where
        the rhs computation isn't reducible via simpler passes (e.g.,
        `s += p[i]` where p[i] requires multi-instruction array
        indexing).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            if (
                line.kind == "instr"
                and line.op == "mov"
                and self._is_eax_to_mem_store(line)
            ):
                # The mov [m], eax is line i. Look back for the
                # frame pattern.
                store_dest = self._mem_dest(line)
                if store_dest is None:
                    i += 1
                    continue
                # Try the 4-line tail first (canonical form for
                # any OP). Then fall back to the 3-line short tail
                # (commutative-OP-only fast path the codegen emits).
                match = self._match_compound_assign_frame(
                    out, i, store_dest
                )
                tail_count = 4 if match is not None else None
                if match is None:
                    match = self._match_compound_assign_frame_short(
                        out, i, store_dest
                    )
                    if match is not None:
                        tail_count = 3
                if match is None:
                    i += 1
                    continue
                push_idx, op_str = match
                # Verify ECX dead after the store.
                if not self._reg_dead_after(out, i + 1, "ecx"):
                    i += 1
                    continue
                # Determine if EAX is dead after the store. If yes,
                # we can drop the framing entirely. If no, we drop
                # the framing AND insert ``mov eax, [m]`` to restore
                # EAX = new [m] value (the canonical tail leaves
                # EAX = lhs OP rhs = new [m]; the reload restores
                # this for downstream consumers).
                eax_dead = self._reg_dead_after(out, i + 1, "eax")
                # Rewrite: drop push, framing tail, mov [m] eax —
                # replace with ``OP [m], eax`` in place of the tail.
                indent = self._extract_indent(line.raw)
                spacer = " " * max(8 - len(op_str), 1)
                new_raw = (
                    f"{indent}{op_str}{spacer}{store_dest}, eax"
                )
                new_line = Line(
                    raw=new_raw,
                    kind="instr",
                    op=op_str,
                    operands=f"{store_dest}, eax",
                )
                # Chain runs from push_idx+1 to (i - tail_count)
                # inclusive — exclusive slice end = i - tail_count + 1.
                # For tail_count=4 (canonical) this is i - 3; for
                # tail_count=3 (commutative short form) it's i - 2.
                replace_lines = [new_line]
                # If EAX is live, append `mov eax, [m]` to reload.
                # Lets downstream code see EAX = new [m] value,
                # matching the canonical tail's post-state.
                if not eax_dead:
                    reload_raw = (
                        f"{indent}mov     eax, {store_dest}"
                    )
                    reload_line = Line(
                        raw=reload_raw,
                        kind="instr",
                        op="mov",
                        operands=f"eax, {store_dest}",
                    )
                    replace_lines.append(reload_line)
                new_out = []
                new_out.extend(out[:push_idx])
                new_out.extend(
                    out[push_idx + 1: i - tail_count + 1]
                )
                new_out.extend(replace_lines)
                new_out.extend(out[i + 1:])
                self.stats["compound_assign_collapse"] = (
                    self.stats.get(
                        "compound_assign_collapse", 0
                    ) + 1
                )
                out = new_out
                # Restart from the rewrite point (chain start).
                i = push_idx
                continue
            i += 1
        return out

    @staticmethod
    def _is_eax_to_mem_store(line: Line) -> bool:
        """Is this ``mov [...], eax``?"""
        if line.kind != "instr" or line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        if "[" not in dest:
            return False
        if src.strip().lower() != "eax":
            return False
        return True

    @staticmethod
    def _mem_dest(line: Line) -> str | None:
        """Extract the memory operand from ``mov [...], eax``."""
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, _ = parts
        return dest.strip()

    def _match_compound_assign_frame(
        self,
        lines: list[Line],
        store_idx: int,
        store_dest: str,
    ) -> tuple[int, str] | None:
        """Look back from `store_idx` for the compound-assign frame.

        Expects (going backward from store_idx):
            store_idx - 1: ``OP eax, ecx`` (OP in _COMPOUND_BINOPS).
            store_idx - 2: ``pop eax``.
            store_idx - 3: ``mov ecx, eax``.
            ...chain (no stack manipulation, no [m] modification)...
            push_idx: ``push dword [m]`` matching store_dest.

        Returns (push_idx, op_str) on match, None otherwise. The
        chain is identified as `push_idx+1 .. store_idx-4`.
        """
        # Walk back through the framing tail.
        ops_at = lambda j: (
            lines[j] if 0 <= j < len(lines) else None
        )
        # store_idx - 1: OP eax, ecx.
        op_line = ops_at(store_idx - 1)
        if op_line is None or op_line.kind != "instr":
            return None
        if op_line.op not in self._COMPOUND_BINOPS:
            return None
        op_parts = _operands_split(op_line.operands)
        if op_parts is None:
            return None
        if (
            op_parts[0].strip().lower() != "eax"
            or op_parts[1].strip().lower() != "ecx"
        ):
            return None
        op_str = op_line.op
        # store_idx - 2: pop eax.
        pop_line = ops_at(store_idx - 2)
        if (
            pop_line is None
            or pop_line.kind != "instr"
            or pop_line.op != "pop"
            or pop_line.operands.strip().lower() != "eax"
        ):
            return None
        # store_idx - 3: mov ecx, eax.
        xfer_line = ops_at(store_idx - 3)
        if (
            xfer_line is None
            or xfer_line.kind != "instr"
            or xfer_line.op != "mov"
        ):
            return None
        xfer_parts = _operands_split(xfer_line.operands)
        if xfer_parts is None:
            return None
        if (
            xfer_parts[0].strip().lower() != "ecx"
            or xfer_parts[1].strip().lower() != "eax"
        ):
            return None
        # Walk back from store_idx - 4 to find the matching push.
        # Track stack depth — balanced inner push/pop pairs are
        # OK (e.g. nested array-indexing pushes).
        push_target = f"dword {store_dest}"
        depth = 0
        for j in range(store_idx - 4, -1, -1):
            ln = lines[j]
            if ln.kind != "instr":
                if ln.kind == "label":
                    return None  # crossed a basic-block boundary
                continue
            if ln.op == "pop":
                # Walking backward past a pop — stack was deeper.
                depth += 1
                continue
            if ln.op == "push":
                if depth > 0:
                    # Matches a later (in source order) pop.
                    depth -= 1
                    continue
                # Outer push at chain start.
                if (
                    ln.operands.strip().lower()
                    == push_target.lower()
                ):
                    return (j, op_str)
                return None
            if (
                ln.op in ("add", "sub")
                and ln.operands.strip().lower().startswith("esp,")
            ):
                return None
            if ln.op == "call":
                return None  # call clobbers ECX/EDX, also stack
            # Memory write to store_dest — bail.
            mod_parts = _operands_split(ln.operands or "")
            if mod_parts is not None:
                m_dest, _ = mod_parts
                if (
                    "[" in m_dest
                    and self._mem_overlaps(
                        m_dest.strip(), store_dest
                    )
                ):
                    return None
        return None

    # Commutative subset of compound-assign ops. The short-tail form
    # `pop ecx; OP eax, ecx; mov [m], eax` is only correct when OP
    # is commutative (so `rhs OP lhs == lhs OP rhs`). sub is
    # excluded.
    _COMPOUND_BINOPS_COMMUTATIVE: frozenset[str] = frozenset({
        "add", "and", "or", "xor",
    })

    def _match_compound_assign_frame_short(
        self,
        lines: list[Line],
        store_idx: int,
        store_dest: str,
    ) -> tuple[int, str] | None:
        """Look back from `store_idx` for the 3-line short tail.
        The codegen emits this directly for commutative compound
        ops:
            store_idx - 1: ``OP eax, ecx`` (commutative OP).
            store_idx - 2: ``pop ecx``.
            ...chain (no stack manipulation, no [m] modification)...
            push_idx: ``push dword [m]`` matching store_dest.

        For commutative OPs, ``rhs OP lhs == lhs OP rhs``, so the
        codegen skips the canonical save-restore (``mov ecx, eax;
        pop eax``) and uses the simpler ``pop ecx`` directly.

        Returns (push_idx, op_str) on match, None otherwise.
        """
        ops_at = lambda j: (
            lines[j] if 0 <= j < len(lines) else None
        )
        # store_idx - 1: OP eax, ecx (commutative).
        op_line = ops_at(store_idx - 1)
        if op_line is None or op_line.kind != "instr":
            return None
        if op_line.op not in self._COMPOUND_BINOPS_COMMUTATIVE:
            return None
        op_parts = _operands_split(op_line.operands)
        if op_parts is None:
            return None
        if (
            op_parts[0].strip().lower() != "eax"
            or op_parts[1].strip().lower() != "ecx"
        ):
            return None
        op_str = op_line.op
        # store_idx - 2: pop ecx.
        pop_line = ops_at(store_idx - 2)
        if (
            pop_line is None
            or pop_line.kind != "instr"
            or pop_line.op != "pop"
            or pop_line.operands.strip().lower() != "ecx"
        ):
            return None
        # Walk back from store_idx - 3 to find the matching push.
        # Track stack depth — balanced inner push/pop pairs OK.
        push_target = f"dword {store_dest}"
        depth = 0
        for j in range(store_idx - 3, -1, -1):
            ln = lines[j]
            if ln.kind != "instr":
                if ln.kind == "label":
                    return None
                continue
            if ln.op == "pop":
                depth += 1
                continue
            if ln.op == "push":
                if depth > 0:
                    depth -= 1
                    continue
                if (
                    ln.operands.strip().lower()
                    == push_target.lower()
                ):
                    return (j, op_str)
                return None
            if (
                ln.op in ("add", "sub")
                and ln.operands.strip().lower().startswith("esp,")
            ):
                return None
            if ln.op == "call":
                return None
            mod_parts = _operands_split(ln.operands or "")
            if mod_parts is not None:
                m_dest, _ = mod_parts
                if (
                    "[" in m_dest
                    and self._mem_overlaps(
                        m_dest.strip(), store_dest
                    )
                ):
                    return None
        return None

    @staticmethod
    def _mem_overlaps(addr1: str, addr2: str) -> bool:
        """Conservative overlap check: True if two memory operands
        could refer to the same bytes. Disjoint ebp-relative memory
        operands return False; everything else returns True (be
        defensive).

        For our pattern (we want to detect modification of `addr2`
        within the chain), a False return means "definitely
        disjoint, safe to ignore"."""
        # Strip optional size keyword.
        def strip(text: str) -> str:
            t = text.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if t.lower().startswith(prefix):
                    t = t[len(prefix):].lstrip()
            return t.lower()

        a, b = strip(addr1), strip(addr2)
        if a == b:
            return True
        # Both ebp-rel literal? Disjoint if different offsets.
        ebp_re = re.compile(
            r"^\[ebp\s*([+-])\s*(\d+)\]$"
        )
        ma, mb = ebp_re.match(a), ebp_re.match(b)
        if ma and mb:
            sa = (1 if ma.group(1) == "+" else -1) * int(ma.group(2))
            sb = (1 if mb.group(1) == "+" else -1) * int(mb.group(2))
            return sa == sb
        # Conservative: assume overlap.
        return True

    def _pass_redundant_xor_zero(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop ``xor reg, reg`` when reg is already known to be zero
        from a recent prior ``xor reg, reg`` and nothing modified it
        since.

        Tracks a per-register "is zero" state forward through a
        single basic block. Invalidated by:
        - Any write to the reg (mov, add, sub, xor with another src,
          inc, dec, neg, shl, etc.).
        - A label (control-flow merge — multiple incoming paths
          might have different states).
        - Unconditional jmp / ret / leave / enter (next instr is
          unreachable as fall-through).
        - Calls (preserve EBX/ESI/EDI/EBP per cdecl, but clobber
          EAX/ECX/EDX).

        Conditional jumps (jcc) preserve the state on the fall-
        through path; the taken path's state is forgotten (we don't
        track per-target zero states like redundant_eax_load does).

        Saves 2 bytes per match (xor reg, reg is `33 modrm` for the
        gp 32-bit registers = 2 bytes).
        """
        out: list[Line] = []
        zero_regs: set[str] = set()
        # Caller-saved regs that get clobbered by `call`.
        CALLER_SAVED = {"eax", "ecx", "edx"}
        for line in lines:
            if line.kind != "instr":
                if line.kind == "label":
                    zero_regs = set()
                out.append(line)
                continue
            op = line.op
            ops = line.operands
            # Detect `xor reg, reg`: drop if reg is already 0.
            if op == "xor":
                parts = _operands_split(ops)
                if parts is not None:
                    a, b = parts
                    al = a.strip().lower()
                    bl = b.strip().lower()
                    if al == bl and self._is_general_register(al):
                        if al in zero_regs:
                            self.stats[
                                "redundant_xor_zero"
                            ] = (
                                self.stats.get(
                                    "redundant_xor_zero", 0
                                ) + 1
                            )
                            continue
                        zero_regs.add(al)
                        out.append(line)
                        continue
            # Other instructions: invalidate any reg they write.
            # `mov [...], reg` doesn't write reg (just reads).
            # `mov reg, X` writes reg.
            if op == "mov":
                parts = _operands_split(ops)
                if parts is not None:
                    dest, _ = parts
                    dest_low = dest.strip().lower()
                    # Direct 32-bit reg write.
                    if (
                        self._is_general_register(dest_low)
                        and dest_low in zero_regs
                    ):
                        zero_regs.discard(dest_low)
                    # Sub-register write (al/ah/ax/bl/bh/bx/etc.) —
                    # invalidates the enclosing 32-bit reg.
                    elif dest_low in self._SUBREG_TO_GP32:
                        gp32 = self._SUBREG_TO_GP32[dest_low]
                        zero_regs.discard(gp32)
                # `mov [...], reg` — no reg-dest write, preserve.
                out.append(line)
                continue
            # cmp/test/push: read-only on regs.
            if op in {"cmp", "test", "push"}:
                out.append(line)
                continue
            # Calls: clobber caller-saved regs.
            if op == "call":
                for r in CALLER_SAVED:
                    zero_regs.discard(r)
                out.append(line)
                continue
            # Unconditional jumps / returns: invalidate everything.
            if op in {"jmp", "ret", "iret", "iretd", "retf",
                      "retn", "leave", "enter"}:
                zero_regs = set()
                out.append(line)
                continue
            # Conditional jumps (jcc): preserve state on fallthrough.
            if op.startswith("j"):
                out.append(line)
                continue
            # Instructions that may write a tracked reg. Conservative:
            # if any tracked reg appears in operands as dest (or
            # appears at all for read+write ops), invalidate.
            # For 2-operand ops: dest is first operand.
            parts = _operands_split(ops)
            if parts is not None:
                dest, _ = parts
                dest_stripped = dest.strip().lower()
                if (
                    self._is_general_register(dest_stripped)
                    and dest_stripped in zero_regs
                ):
                    zero_regs.discard(dest_stripped)
                elif dest_stripped in self._SUBREG_TO_GP32:
                    gp32 = self._SUBREG_TO_GP32[dest_stripped]
                    zero_regs.discard(gp32)
            # Single-operand ops (inc/dec/neg/not/idiv/etc.): operand
            # is dest+source.
            else:
                op_stripped = ops.strip().lower()
                if self._is_general_register(op_stripped):
                    zero_regs.discard(op_stripped)
                elif op_stripped in self._SUBREG_TO_GP32:
                    gp32 = self._SUBREG_TO_GP32[op_stripped]
                    zero_regs.discard(gp32)
            # `cdq` writes EDX; `lodsd` writes EAX; etc. — implicit
            # writers. Conservative: clear all caller-saved regs.
            if op in PeepholeOptimizer._IMPLICIT_REG_USERS:
                for r in CALLER_SAVED:
                    zero_regs.discard(r)
            out.append(line)
        return out

    def _pass_dual_zero_init_consolidate(
        self, lines: list[Line]
    ) -> list[Line]:
        """Consolidate `xor REG_A, REG_A; xor REG_B, REG_B; <stores from
        REG_A or REG_B>` into a single xor + stores all using REG_A.

        Common after long-long zero init: codegen emits
            xor eax, eax
            xor edx, edx
            mov [m1], eax
            mov [m2], edx
            mov [m3], eax
            mov [m4], edx
            ...
        REG_B (= EDX in the LL case) holds 0 from the second xor, but
        any other zero reg can serve the same stores. After rewriting
        all `mov [m], REG_B` to `mov [m], REG_A`, the second xor is
        dead — drop it.

        Saves 2 bytes per fire (the dropped xor).

        Conditions for the rewrite chain (after the two xors):
        - Each line is `mov [m], REG_A` or `mov [m], REG_B` (no other
          ops, no other regs).
        - Chain ends at the next non-matching instruction OR at any
          control-flow boundary (label, jump, call, ret).
        - REG_B is dead after the chain (no forward uses in the basic
          block). The exact subsequent value of REG_B doesn't matter;
          if it gets read before being written, that read would have
          gotten 0 from the now-dropped xor — so we conservatively bail
          when there's any read.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            # Find an `xor REG_A, REG_A` line.
            line_a = out[i]
            reg_a = self._xor_reg_self(line_a)
            if reg_a is None:
                i += 1
                continue
            # Find the next instr line; must be `xor REG_B, REG_B`.
            j = i + 1
            while j < len(out) and out[j].kind in ("blank", "comment"):
                j += 1
            if j >= len(out):
                i += 1
                continue
            line_b = out[j]
            reg_b = self._xor_reg_self(line_b)
            if reg_b is None or reg_b == reg_a:
                i += 1
                continue
            # Walk forward collecting `mov [m], REG_A` or `mov [m], REG_B`
            # stores. Stop at any other instruction.
            k = j + 1
            store_indices = []
            uses_reg_b = False
            while k < len(out):
                ln = out[k]
                if ln.kind in ("blank", "comment"):
                    k += 1
                    continue
                if ln.kind != "instr":
                    break
                if not (ln.op == "mov"
                        and self._is_store_from_reg(ln, reg_a, reg_b)):
                    break
                # Track whether this store reads REG_B.
                parts = _operands_split(ln.operands)
                if parts is not None:
                    _dest, src = parts
                    if src.strip().lower() == reg_b:
                        uses_reg_b = True
                store_indices.append(k)
                k += 1
            # Need at least one `mov [m], REG_B` to make the rewrite
            # worthwhile (else there's nothing to consolidate).
            if not uses_reg_b:
                i += 1
                continue
            # Check REG_B is dead after the chain.
            if not self._reg_dead_after(out, k, reg_b):
                i += 1
                continue
            # Rewrite each store-from-REG_B to use REG_A.
            for idx in store_indices:
                ln = out[idx]
                parts = _operands_split(ln.operands)
                if parts is None:
                    continue
                dest, src = parts
                if src.strip().lower() != reg_b:
                    continue
                indent = self._extract_indent(ln.raw)
                new_raw = f"{indent}mov     {dest.strip()}, {reg_a}"
                out[idx] = Line(
                    raw=new_raw, kind="instr", op="mov",
                    operands=f"{dest.strip()}, {reg_a}",
                )
            # Drop the `xor REG_B, REG_B` at index j.
            del out[j]
            self.stats["dual_zero_init_consolidate"] = (
                self.stats.get("dual_zero_init_consolidate", 0) + 1
            )
            # Advance past the rewritten chain.
            i = j  # j now points to first store (or next line)
        return out

    @staticmethod
    def _xor_reg_self(line: Line) -> str | None:
        """Return reg name (lowercase) if this is `xor reg, reg` for a
        GP32 reg, else None."""
        if line.kind != "instr" or line.op != "xor":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        a, b = parts
        al = a.strip().lower()
        bl = b.strip().lower()
        if al == bl and al in {
            "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp",
        }:
            return al
        return None

    @staticmethod
    def _is_store_from_reg(line: Line, *regs: str) -> bool:
        """Return True if this is `mov [<addr>], <reg>` where reg is
        one of the given GP32 regs (case-insensitive)."""
        if line.kind != "instr" or line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        dest_text = dest.strip()
        if "[" not in dest_text or not dest_text.endswith("]"):
            return False
        # Strip optional size keyword.
        if dest_text.lower().startswith(("byte ", "word ", "dword ", "qword ")):
            return False  # sub-word stores excluded (different reg)
        src_low = src.strip().lower()
        return src_low in {r.lower() for r in regs}

    # Map low byte/word sub-registers to their containing 32-bit reg.
    # Note: AH/BH/CH/DH (high-byte) NOT included — they're not the
    # natural source for byte stores from ASM lowering's perspective,
    # and movsx/movzx from a high byte requires a different opcode.
    _LOW_BYTE_REGS: dict[str, str] = {
        "al": "eax", "bl": "ebx", "cl": "ecx", "dl": "edx",
    }
    _LOW_WORD_REGS: dict[str, str] = {
        "ax": "eax", "bx": "ebx", "cx": "ecx", "dx": "edx",
        "si": "esi", "di": "edi", "bp": "ebp",
    }

    def _pass_narrow_store_reload_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse `mov <size> [m], REG_LOW; movsx/movzx DST, <size>
        [m]` into `mov <size> [m], REG_LOW; movsx/movzx DST, REG_LOW`.
        Saves 1 byte per match (the memory operand on the extension
        becomes a register operand, dropping the disp/sib).

        Common when the codegen narrows a value to a sub-word slot via
        a store, then reads it back with sign/zero extension to a
        32-bit register. The reload is redundant — REG_LOW still holds
        the value that was stored.

        Conditions:
        - Line A: `mov <byte|word> [m], REG_LOW` where REG_LOW is a
          recognized low byte (al/bl/cl/dl) or low word (ax/bx/cx/dx/
          si/di/bp) register.
        - Line B: `movsx <DST>, <byte|word> [m]` or
          `movzx <DST>, <byte|word> [m]` — same size, same memory.
        - Rewrite: line B becomes
          `movsx <DST>, REG_LOW` / `movzx <DST>, REG_LOW`.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op in ("movsx", "movzx")
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            a_dest, a_src = ap
            b_dest, b_src = bp
            a_dest_text = a_dest.strip()
            a_src_low = a_src.strip().lower()
            b_dest_low = b_dest.strip().lower()
            b_src_text = b_src.strip()
            # Line A: extract size keyword and memory operand.
            a_size, a_mem = self._split_sized_mem(a_dest_text)
            if a_size.lower() not in ("byte", "word"):
                i += 1
                continue
            # Line A's source must be a recognized low sub-register
            # matching the size.
            if a_size.lower() == "byte":
                if a_src_low not in self._LOW_BYTE_REGS:
                    i += 1
                    continue
            else:  # word
                if a_src_low not in self._LOW_WORD_REGS:
                    i += 1
                    continue
            # Line B: source must be `<size> [m]` where <size> matches
            # and [m] equals A's memory operand.
            b_size, b_mem = self._split_sized_mem(b_src_text)
            if b_size.lower() != a_size.lower():
                i += 1
                continue
            if self._normalize_mem(b_mem) != self._normalize_mem(a_mem):
                i += 1
                continue
            # Dest of line B must be a 32-bit GP register.
            if not self._is_general_register(b_dest_low):
                i += 1
                continue
            # Rewrite line B: replace memory operand with the register.
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}{b.op:7s} {b_dest_low}, {a_src_low}"
            out[i + 1] = Line(
                raw=new_raw, kind="instr", op=b.op,
                operands=f"{b_dest_low}, {a_src_low}",
            )
            self.stats["narrow_store_reload_collapse"] = (
                self.stats.get(
                    "narrow_store_reload_collapse", 0
                ) + 1
            )
            i += 2
        return out

    @staticmethod
    def _normalize_mem(mem_text: str) -> str:
        """Normalize whitespace inside a memory operand for textual
        comparison. `[ebp - 4]` and `[ebp-4]` both compare equal."""
        return re.sub(r"\s+", "", mem_text.strip())

    def _pass_add_esp_to_pop(self, lines: list[Line]) -> list[Line]:
        """Replace `add esp, 4` with `pop ecx` (saves 2 bytes) when ECX
        is dead after, and `add esp, 8` with `pop ecx; pop ecx` (saves
        1 byte). Both cases require ECX to be dead after.

        x86 byte sizes:
        - `add esp, 4` = 3 bytes (`83 C4 04`).
        - `pop ecx` = 1 byte (`59`).
        - `add esp, 8` = 3 bytes.

        For larger N, `add esp, N` is at most 3 bytes (imm8) or 6 bytes
        (imm32). Since each `pop reg` is 1 byte, breakeven is at 3
        pops. So:
        - N=4: 1 pop = 1 byte (saves 2).
        - N=8: 2 pops = 2 bytes (saves 1).
        - N>=12: not worth it.

        Common after cdecl call cleanup: ECX is caller-saved scratch,
        so it's dead by convention after the call. We use a fast
        path (`prev is call`) to avoid the expensive CFG-aware
        liveness check on hot paths — pr23135's 720K-line asm has
        thousands of `add esp` sites and the slow check adds 50+
        seconds.

        Conditions:
        - `add esp, IMM` where IMM is exactly 4 or 8.
        - Either: previous instr is `call` (cdecl: ECX dead by
          convention), OR `_reg_dead_after` confirms ECX is dead.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            if line.kind != "instr" or line.op != "add":
                i += 1
                continue
            parts = _operands_split(line.operands)
            if parts is None:
                i += 1
                continue
            dest, src = parts
            if dest.strip().lower() != "esp":
                i += 1
                continue
            src_text = src.strip()
            if src_text not in ("4", "8"):
                i += 1
                continue
            # ECX must be dead after this instruction (after the
            # rewrite, ECX gets overwritten by the pop; subsequent
            # code must not depend on ECX's prior value).
            #
            # We restrict to the common case: `add esp, IMM`
            # immediately preceded by `call`. After a cdecl call,
            # ECX is caller-saved scratch and dead by convention.
            # The next instruction must NOT read ECX (rare but
            # possible in adversarial code; standard codegen never
            # does this).
            #
            # This restriction avoids the expensive `_reg_dead_after`
            # call entirely, which became a bottleneck for very large
            # files (pr23135 has 720K lines and thousands of `add esp`
            # sites; the slow check pushed compile time over the
            # runner timeout).
            # Eligible if either:
            # (a) previous is `call` AND next doesn't read ECX
            #     (cdecl arg cleanup — ECX dead by convention)
            # (b) next is `pop ecx` (overwrites ECX — `pop` reads
            #     from [esp], not from ECX)
            #
            # Case (b) handles patterns like `fild dword [esp]; add
            # esp, 4; pop ecx` where the `add` is x87-stack scratch
            # cleanup, not a call cleanup. Saves 2 bytes per match.
            if self._next_instr_is_pop_ecx(out, i + 1):
                # Always eligible — pop ecx overwrites ECX without
                # reading it.
                pass
            elif self._prev_instr_is_call(out, i):
                if not self._next_instr_doesnt_read_ecx(out, i + 1):
                    i += 1
                    continue
            else:
                i += 1
                continue
            indent = self._extract_indent(line.raw)
            if src_text == "4":
                out[i] = Line(
                    raw=f"{indent}pop     ecx", kind="instr",
                    op="pop", operands="ecx",
                )
                self.stats["add_esp_to_pop"] = (
                    self.stats.get("add_esp_to_pop", 0) + 1
                )
                i += 1
            else:  # 8
                out[i] = Line(
                    raw=f"{indent}pop     ecx", kind="instr",
                    op="pop", operands="ecx",
                )
                out.insert(i + 1, Line(
                    raw=f"{indent}pop     ecx", kind="instr",
                    op="pop", operands="ecx",
                ))
                self.stats["add_esp_to_pop"] = (
                    self.stats.get("add_esp_to_pop", 0) + 1
                )
                i += 2
        return out

    @staticmethod
    def _prev_instr_is_call(lines: list[Line], idx: int) -> bool:
        """Walk backward from idx to find the previous instruction.
        Return True if it's a `call` (direct or indirect)."""
        j = idx - 1
        while j >= 0:
            ln = lines[j]
            if ln.kind in ("blank", "comment"):
                j -= 1
                continue
            return ln.kind == "instr" and ln.op == "call"
        return False

    def _next_instr_doesnt_read_ecx(
        self, lines: list[Line], idx: int,
    ) -> bool:
        """Walk forward from idx to find the next instruction. Return
        True if it doesn't reference ECX (or its sub-regs)."""
        j = idx
        while j < len(lines):
            ln = lines[j]
            if ln.kind in ("blank", "comment"):
                j += 1
                continue
            if ln.kind != "instr":
                # End of function or section: ECX is dead.
                return True
            return not self._references_reg_family(ln.operands, "ecx")
        return True

    @staticmethod
    def _next_instr_is_pop_ecx(
        lines: list[Line], idx: int,
    ) -> bool:
        """Walk forward from idx to find the next instruction. Return
        True if it's `pop ecx` — which overwrites ECX with the new
        TOS value, making the prior ECX value irrelevant."""
        j = idx
        while j < len(lines):
            ln = lines[j]
            if ln.kind in ("blank", "comment"):
                j += 1
                continue
            if ln.kind != "instr":
                return False
            return (
                ln.op == "pop"
                and ln.operands.strip().lower() == "ecx"
            )
        return False

    # Commutative two-operand binops where `OP eax, ecx` and
    # `OP ecx, eax` produce identical EAX results. imul (two-operand
    # form) is included; sub/div/idiv etc. are NOT.
    _COMMUTATIVE_BINOPS: frozenset[str] = frozenset({
        "add", "and", "or", "xor", "imul",
    })

    def _pass_transfer_pop_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the binop tail `mov ecx, eax; pop eax; OP eax, ecx`
        into `pop ecx; OP eax, ecx` when OP is commutative.

        The original sequence saves the chain's EAX value into ECX,
        restores the LHS from stack to EAX, then computes EAX = LHS
        OP RHS. For commutative ops the operand order doesn't matter,
        so we can pop the saved LHS into ECX directly and compute
        EAX = RHS OP LHS in place.

        Saves 2 bytes per match (drops the `mov ecx, eax`).

        Conditions:
        - Three consecutive instr lines.
        - Line A: ``mov ecx, eax``.
        - Line B: ``pop eax``.
        - Line C: ``OP eax, ecx`` with OP ∈ {add, and, or, xor, imul}.
        - ECX is dead before the chain (the chain only uses ECX as
          scratch; we change which value lands there but the next
          read of ECX overwrites it). This is the codegen invariant
          for stack-machine binops.

        Note: ECX's post-collapse value is the LHS instead of the
        RHS. Any reader of ECX between `OP` and the next ECX-writer
        would see a different value. The codegen never generates
        such readers — the binop tail always either ends the
        expression or feeds into a `push eax` / next subexpression
        that re-loads ECX from scratch.
        """
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "mov"
                and a.operands.strip().lower() == "ecx, eax"
                and b.kind == "instr" and b.op == "pop"
                and b.operands.strip().lower() == "eax"
                and c.kind == "instr"
                and c.op in PeepholeOptimizer._COMMUTATIVE_BINOPS
            ):
                i += 1
                continue
            cp = _operands_split(c.operands)
            if cp is None:
                i += 1
                continue
            cdst, csrc = cp
            if (
                cdst.strip().lower() != "eax"
                or csrc.strip().lower() != "ecx"
            ):
                i += 1
                continue
            # Rewrite: drop A, replace B with `pop ecx`, keep C.
            indent = self._extract_indent(b.raw)
            new_pop = Line(
                raw=f"{indent}pop     ecx",
                kind="instr",
                op="pop",
                operands="ecx",
            )
            out = out[:i] + [new_pop, c] + out[i + 3:]
            self.stats["transfer_pop_collapse"] = (
                self.stats.get("transfer_pop_collapse", 0) + 1
            )
            continue
        return out

    _ZF_ONLY_FLAG_READERS: frozenset[str] = frozenset({
        "je", "jne", "jz", "jnz",
        "sete", "setne", "setz", "setnz",
        "cmove", "cmovne", "cmovz", "cmovnz",
    })

    _NON_ZF_FLAG_READERS: frozenset[str] = frozenset({
        "jl", "jle", "jg", "jge",
        "js", "jns", "jo", "jno",
        "jb", "jbe", "ja", "jae",
        "jc", "jnc", "jp", "jnp", "jpe", "jpo",
        "setl", "setle", "setg", "setge",
        "sets", "setns", "seto", "setno",
        "setb", "setbe", "seta", "setae",
        "setc", "setnc", "setp", "setnp",
        "cmovl", "cmovle", "cmovg", "cmovge",
        "cmovs", "cmovns", "cmovo", "cmovno",
        "cmovb", "cmovbe", "cmova", "cmovae",
        "cmovc", "cmovnc", "cmovp", "cmovnp",
        "adc", "sbb", "rcl", "rcr",
    })

    _FLAG_CLOBBERS: frozenset[str] = frozenset({
        "add", "sub", "and", "or", "xor", "cmp", "test",
        "inc", "dec", "neg", "not_",
        "shl", "shr", "sar", "rol", "ror", "sal",
        "imul", "mul", "idiv", "div",
        "bt", "bts", "btr", "btc",
        "bsf", "bsr",
    })

    def _flags_used_only_for_zf(
        self, lines: list[Line], start_idx: int
    ) -> bool:
        """Scan forward from start_idx; return True iff every
        flag-reader before the first flag-clobber/fence is ZF-only.

        Fences: labels, ret, leave, enter, call, unconditional jmp.
        After a fence, the cmp's flags are irrelevant.
        """
        n = len(lines)
        for j in range(start_idx, n):
            line = lines[j]
            if line.kind == "label":
                return True
            if line.kind != "instr":
                continue
            op = line.op
            if op in PeepholeOptimizer._ZF_ONLY_FLAG_READERS:
                continue
            if op in PeepholeOptimizer._NON_ZF_FLAG_READERS:
                return False
            if op in PeepholeOptimizer._FLAG_CLOBBERS:
                return True
            if op in {"ret", "retn", "retf", "iret",
                      "leave", "enter", "call", "jmp",
                      "int", "iretd", "sysenter"}:
                return True
        return True

    def _pass_transfer_pop_cmp_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse `mov ecx, eax; pop eax; cmp eax, ecx` into
        `pop ecx; cmp eax, ecx` when the cmp's flags are read only
        for equality (je/jne/jz/jnz/sete/setne/etc.).

        Operand order matters for cmp: SF/CF/OF flip when arguments
        swap. ZF is symmetric. So the rewrite is safe iff every
        flag-reader between the cmp and the next flag-clobber/fence
        reads only ZF.

        Saves 2 bytes per match (drops the `mov ecx, eax`).
        """
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "mov"
                and a.operands.strip().lower() == "ecx, eax"
                and b.kind == "instr" and b.op == "pop"
                and b.operands.strip().lower() == "eax"
                and c.kind == "instr" and c.op == "cmp"
            ):
                i += 1
                continue
            cp = _operands_split(c.operands)
            if cp is None:
                i += 1
                continue
            cdst, csrc = cp
            if (
                cdst.strip().lower() != "eax"
                or csrc.strip().lower() != "ecx"
            ):
                i += 1
                continue
            if not self._flags_used_only_for_zf(out, i + 3):
                i += 1
                continue
            indent = self._extract_indent(b.raw)
            new_pop = Line(
                raw=f"{indent}pop     ecx",
                kind="instr",
                op="pop",
                operands="ecx",
            )
            out = out[:i] + [new_pop, c] + out[i + 3:]
            self.stats["transfer_pop_cmp_collapse"] = (
                self.stats.get("transfer_pop_cmp_collapse", 0) + 1
            )
            continue
        return out

    def _pass_dup_push_pop_self_op(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the canonical "X OP X" pattern where the codegen
        pushes one copy of X and reloads X for the second operand.

        Pattern (4 consecutive instr lines):
            push    X
            mov     reg1, X
            pop     reg2
            OP      reg1, reg2

        Rewrite to:
            mov     reg1, X
            OP      reg1, reg1

        Saves 2 instructions per match. The push/pop pair is dead
        because both copies of X are equal (the mov reloaded it from
        the same memory) — the OP can use reg1 as both operands.

        Common in `arr[i] * arr[i]` style where the codegen evaluates
        the left to memory-push form (because evaluating the right
        will clobber EAX), then reloads from the same memory address.

        Conditions:
        - Line 1 op = "push", any operand X.
        - Line 2 op = "mov", parts = (reg1, X) where reg1 is a 32-bit
          GP register and X matches line 1's push operand text.
        - Line 3 op = "pop", operand = reg2 (32-bit GP register, ≠
          reg1).
        - Line 4 op ∈ {add, and, or, xor, imul} (commutative), parts
          = (reg1, reg2).
        - X must not reference esp (push/pop modify ESP between
          line 1 and line 3; if X depends on esp, the mov in line 2
          could read different bytes than the push wrote).
        - X must not reference reg1 if reg1 is non-eax — actually
          this isn't an issue because we keep line 2 (the mov) which
          loads X to reg1 in the original order. The OP comes after
          the load, so reg1 holds X. We're just changing the OP's
          second operand from reg2 (= X via pop) to reg1 (= X via
          mov). Both equal X.
        - reg2 must be dead after line 4. The original ends with
          reg2 = X; the rewrite leaves reg2 unchanged.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            instrs = self._next_n_instrs(out, i, 4)
            if instrs is None:
                i += 1
                continue
            (a_idx, a), (b_idx, b), (c_idx, c), (d_idx, d) = instrs
            if a.op != "push" or b.op != "mov" or c.op != "pop":
                i += 1
                continue
            if d.op not in PeepholeOptimizer._COMMUTATIVE_BINOPS:
                i += 1
                continue
            push_x = a.operands.strip()
            # Line 2: mov reg1, X
            bp = _operands_split(b.operands)
            if bp is None:
                i += 1
                continue
            b_dst, b_src = bp
            reg1 = b_dst.strip().lower()
            if reg1 not in PeepholeOptimizer._GP32:
                i += 1
                continue
            # Compare push X and mov src X. They must read identical
            # bytes; we compare normalized text. The push may have a
            # `dword` prefix that the mov source omits; strip it.
            push_x_norm = self._strip_size_prefix(push_x)
            mov_src_norm = self._strip_size_prefix(b_src.strip())
            if push_x_norm.lower() != mov_src_norm.lower():
                i += 1
                continue
            # X must not reference esp (push moves esp).
            if _references_register(push_x_norm, "esp"):
                i += 1
                continue
            # Line 3: pop reg2
            reg2 = c.operands.strip().lower()
            if reg2 not in PeepholeOptimizer._GP32:
                i += 1
                continue
            if reg2 == reg1:
                i += 1
                continue
            # Line 4: OP reg1, reg2
            dp = _operands_split(d.operands)
            if dp is None:
                i += 1
                continue
            d_dst, d_src = dp
            if d_dst.strip().lower() != reg1:
                i += 1
                continue
            if d_src.strip().lower() != reg2:
                i += 1
                continue
            # reg2 must be dead after line 4.
            if not self._reg_dead_after(out, d_idx + 1, reg2):
                i += 1
                continue
            # Rewrite: drop A (push) and C (pop); modify D's operand.
            indent = self._extract_indent(d.raw)
            spaces = max(1, 8 - len(d.op))
            new_d = Line(
                raw=f"{indent}{d.op}{' ' * spaces}{reg1}, {reg1}",
                kind="instr",
                op=d.op,
                operands=f"{reg1}, {reg1}",
            )
            out = (
                out[:a_idx]
                + [b]
                + [new_d]
                + out[d_idx + 1:]
            )
            self.stats["dup_push_pop_self_op"] = (
                self.stats.get("dup_push_pop_self_op", 0) + 1
            )
            # Don't advance i; new line at i may fire another pass.
            continue
        return out

    _GP32: frozenset[str] = frozenset({
        "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
    })

    def _pass_push_pop_op_to_memop(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the LHS-saved-across-call binop pattern:

            push    dword [ebp ± N]    ; save LHS
            ... chain (incl. calls) ... ; produces RHS in EAX
            pop     reg2                ; restore LHS into reg2
            OP      reg1, reg2          ; reg1 = chain result (EAX)

        Rewrite to:

            ... chain ...
            OP      reg1, [ebp ± N]    ; OP reg-mem (commutative)

        Saves 2 instructions per match (drops the push and the pop).
        Common in recursive functions where the LHS is saved across
        the recursive call (e.g. ``n * factorial(n-1)``).

        Conditions:
        - Line 1: ``push <size?> [ebp + N]`` or ``push <size?> [ebp - N]``
          where N is a numeric literal. Restricted to ebp-relative
          addressing because ebp is callee-saved (cdecl), so X's
          memory location is preserved across calls.
        - Lines 2..N: chain. Stack-balanced (every inner push has a
          matching inner pop or `add esp, K` cleanup). Must not store
          to X (no `mov [ebp ± N], ...` with the same offset).
          Calls allowed; we trust cdecl + ebp-callee-saved.
        - Line N+1: ``pop reg2`` (32-bit GP register, ≠ reg1).
        - Line N+2: ``OP reg1, reg2`` where OP ∈ commutative binops.
        - reg2 must be dead after line N+2.
        - Chain must not take the address of X via ``lea reg, [ebp ± N]``
          with the same offset (a pointer to X could be passed to a
          call that mutates it).
        - Excludes the narrow 4-line case (chain length 1 with same X)
          which is handled by `_pass_dup_push_pop_self_op` and saves
          1 byte more (reg-reg form vs reg-mem form).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            push_line = out[i]
            if (
                push_line.kind != "instr"
                or push_line.op != "push"
            ):
                i += 1
                continue
            push_x = push_line.operands.strip()
            x_clean = self._strip_size_prefix(push_x)
            x_off = self._ebp_offset(x_clean)
            if x_off is None:
                i += 1
                continue
            # Function-wide aliasing check: bail if any `lea` in the
            # function takes the address of X's slot. If `&X` is
            # captured anywhere, a call in the chain might mutate X
            # via that pointer, and the saved-value-vs-current-value
            # difference would be observable. (The codegen pushes
            # unconditionally — it doesn't know whether the call
            # could mutate; my pass assumes X is unchanged.)
            if self._function_takes_ebp_addr(out, i, x_off):
                i += 1
                continue
            # Walk forward to find the matching pop. Track stack
            # depth — balanced inner push/pop pairs and add/sub esp
            # adjustments are part of the chain.
            depth = 1
            pop_idx = None
            chain_safe = True
            j = i + 1
            while j < len(out):
                ln = out[j]
                if ln.kind == "label":
                    chain_safe = False
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                op = ln.op
                if op == "push":
                    depth += 1
                    j += 1
                    continue
                if op == "pop":
                    depth -= 1
                    if depth == 0:
                        pop_idx = j
                        break
                    j += 1
                    continue
                if op in ("ret", "iret", "iretd", "retf", "retn",
                          "leave", "jmp", "enter"):
                    chain_safe = False
                    break
                if (
                    op in ("add", "sub")
                    and ln.operands.lower().startswith("esp,")
                ):
                    # Adjust depth: `add esp, 4` pops 1, `sub esp, 4`
                    # pushes 1. We don't know the K dynamically — but
                    # after `call`, ``add esp, K`` is the standard
                    # cleanup. Treat as zero-net inner push/pop:
                    # if it's `add esp, K`, decrement depth by K/4;
                    # if `sub esp, K`, increment by K/4.
                    parts_esp = _operands_split(ln.operands)
                    if parts_esp is None:
                        chain_safe = False
                        break
                    _, esp_amt = parts_esp
                    try:
                        amt = int(esp_amt.strip(), 0)
                    except ValueError:
                        chain_safe = False
                        break
                    if amt % 4 != 0:
                        chain_safe = False
                        break
                    delta = amt // 4
                    if op == "add":
                        depth -= delta
                    else:
                        depth += delta
                    if depth <= 0:
                        # add esp, K can pop the saved X if K is too
                        # large. Bail.
                        chain_safe = False
                        break
                    j += 1
                    continue
                # Memory store check: bail if writes to X's slot.
                op_parts = _operands_split(ln.operands or "")
                if op_parts is not None:
                    m_dest, _ = op_parts
                    if "[" in m_dest:
                        m_off = self._ebp_offset(
                            self._strip_size_prefix(m_dest.strip())
                        )
                        if m_off is not None and m_off == x_off:
                            chain_safe = False
                            break
                # LEA of address-of-X: bail (pointer escape).
                if op == "lea":
                    if op_parts is not None:
                        _, lea_src = op_parts
                        lea_off = self._ebp_offset(
                            lea_src.strip()
                        )
                        if lea_off is not None and lea_off == x_off:
                            chain_safe = False
                            break
                j += 1
            if not chain_safe or pop_idx is None:
                i += 1
                continue
            # Chain must have at least 2 instructions (the narrow
            # 1-instr case is handled by _pass_dup_push_pop_self_op
            # and saves 1 byte more via reg-reg form).
            chain_instrs = sum(
                1 for k in range(i + 1, pop_idx)
                if out[k].kind == "instr"
            )
            if chain_instrs < 2:
                i += 1
                continue
            # Pop reg2 (32-bit GP).
            pop_reg = out[pop_idx].operands.strip().lower()
            if pop_reg not in PeepholeOptimizer._GP32:
                i += 1
                continue
            # Find the OP line right after pop (skip blanks/comments).
            op_idx = None
            for k in range(pop_idx + 1, len(out)):
                if out[k].kind == "label":
                    break
                if out[k].kind == "instr":
                    op_idx = k
                    break
            if op_idx is None:
                i += 1
                continue
            op_line = out[op_idx]
            if op_line.op not in PeepholeOptimizer._COMMUTATIVE_BINOPS:
                i += 1
                continue
            opp = _operands_split(op_line.operands)
            if opp is None:
                i += 1
                continue
            reg1 = opp[0].strip().lower()
            op_src = opp[1].strip().lower()
            if reg1 not in PeepholeOptimizer._GP32 or reg1 == pop_reg:
                i += 1
                continue
            if op_src != pop_reg:
                i += 1
                continue
            # reg2 (pop_reg) dead after op_idx.
            if not self._reg_dead_after(out, op_idx + 1, pop_reg):
                i += 1
                continue
            # Rewrite. New OP line uses memory operand X.
            indent = self._extract_indent(op_line.raw)
            spacer = " " * max(8 - len(op_line.op), 1)
            mem_text = f"dword {x_clean}"
            new_op_raw = (
                f"{indent}{op_line.op}{spacer}{reg1}, {mem_text}"
            )
            new_op_line = Line(
                raw=new_op_raw,
                kind="instr",
                op=op_line.op,
                operands=f"{reg1}, {mem_text}",
            )
            new_out = []
            new_out.extend(out[:i])  # before push
            new_out.extend(out[i + 1:pop_idx])  # chain (drop push)
            new_out.extend(out[pop_idx + 1:op_idx])  # between pop & op
            new_out.append(new_op_line)
            new_out.extend(out[op_idx + 1:])
            out = new_out
            self.stats["push_pop_op_to_memop"] = (
                self.stats.get("push_pop_op_to_memop", 0) + 1
            )
            # Don't advance i; new line may fire another pass.
            continue
        return out

    def _pass_push_pop_register_op(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop save/restore around register-via-different-reg op.

        Pattern:
            push    REG_PUSHED         ; save REG_PUSHED's value
            ... chain (no write to REG_PUSHED) ...
            pop     REG_POPPED         ; recover into different reg
            OP      TARGET, REG_POPPED ; commutative binop

        Rewrite (drop push and pop, source from REG_PUSHED directly):
            ... chain ...
            OP      TARGET, REG_PUSHED

        Saves 2 instructions (1 push + 1 pop = 2 bytes).
        REG_PUSHED still holds its original value at the OP site
        because the chain doesn't write it; using REG_PUSHED instead
        of REG_POPPED yields the same operand value.

        Common in bit-field write idioms where the codegen pushes the
        value to write across a sequence that builds the masked
        storage word, then ORs the popped value in. The push is
        unnecessary because the value's source register isn't
        clobbered by the masking computation.

        Conditions:
        - Line 1: ``push REG_PUSHED`` (32-bit GP register).
        - Lines 2..N (chain): stack-balanced, no labels, no
          control flow (calls allowed), no write to REG_PUSHED.
        - Line N+1: ``pop REG_POPPED`` (32-bit GP register).
        - Line N+2: ``OP TARGET, REG_POPPED`` for OP in commutative
          binops (add/and/or/xor/imul).
        - REG_PUSHED, REG_POPPED, TARGET all distinct.
        - REG_POPPED dead after line N+2 (else its post-state
          differs).
        - Chain length ≥ 1 (the 0-length case is handled by
          `transfer_pop_collapse` and similar).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            push_line = out[i]
            if (
                push_line.kind != "instr"
                or push_line.op != "push"
            ):
                i += 1
                continue
            push_reg = push_line.operands.strip().lower()
            if push_reg not in PeepholeOptimizer._GP32:
                i += 1
                continue
            # Walk forward to find matching pop. Track stack depth.
            depth = 1
            pop_idx = None
            chain_safe = True
            chain_writes_push_reg = False
            j = i + 1
            while j < len(out):
                ln = out[j]
                if ln.kind == "label":
                    chain_safe = False
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                op = ln.op
                if op == "push":
                    depth += 1
                    j += 1
                    continue
                if op == "pop":
                    depth -= 1
                    if depth == 0:
                        pop_idx = j
                        break
                    # Inner pop into something that might be push_reg
                    # — those are part of the chain. We still need to
                    # check if push_reg becomes the dest.
                    pop_dest = ln.operands.strip().lower()
                    if pop_dest == push_reg:
                        chain_writes_push_reg = True
                    j += 1
                    continue
                if op in ("ret", "iret", "iretd", "retf", "retn",
                          "leave", "jmp", "enter"):
                    chain_safe = False
                    break
                # Conditional jumps (control flow leaves the chain
                # unchecked). Bail.
                if op.startswith("j") and op != "jmp":
                    chain_safe = False
                    break
                # `add esp, K` / `sub esp, K` — depth tracking.
                if (
                    op in ("add", "sub")
                    and ln.operands.lower().startswith("esp,")
                ):
                    parts_esp = _operands_split(ln.operands)
                    if parts_esp is None:
                        chain_safe = False
                        break
                    _, esp_amt = parts_esp
                    try:
                        amt = int(esp_amt.strip(), 0)
                    except ValueError:
                        chain_safe = False
                        break
                    if amt % 4 != 0:
                        chain_safe = False
                        break
                    delta = amt // 4
                    if op == "add":
                        depth -= delta
                    else:
                        depth += delta
                    if depth <= 0:
                        chain_safe = False
                        break
                    j += 1
                    continue
                # Calls clobber EAX/ECX/EDX per cdecl. If push_reg
                # is one of those, the chain "writes" it.
                if op == "call":
                    if push_reg in ("eax", "ecx", "edx"):
                        chain_writes_push_reg = True
                    j += 1
                    continue
                # Generic: check if instr writes push_reg as dest.
                op_parts = _operands_split(ln.operands or "")
                if op_parts is not None:
                    dest = op_parts[0].strip().lower()
                    # Sub-register write to push_reg's family.
                    dest_canon = (
                        PeepholeOptimizer._SUBREG_TO_GP32.get(
                            dest, dest
                        )
                    )
                    if dest_canon == push_reg:
                        chain_writes_push_reg = True
                else:
                    # Single-operand op (inc/dec/neg/not/pop/etc.)
                    single = (ln.operands or "").strip().lower()
                    single_canon = (
                        PeepholeOptimizer._SUBREG_TO_GP32.get(
                            single, single
                        )
                    )
                    if (
                        single_canon == push_reg
                        and op in (
                            "inc", "dec", "neg", "not", "mul",
                            "div", "idiv", "imul",
                        )
                    ):
                        chain_writes_push_reg = True
                # Implicit reg writers (cdq writes EDX, mul/div
                # write EAX/EDX).
                if op == "cdq" and push_reg == "edx":
                    chain_writes_push_reg = True
                if op in ("mul", "div", "idiv"):
                    if push_reg in ("eax", "edx"):
                        chain_writes_push_reg = True
                if (
                    op in ("imul",)
                    and ln.operands
                    and "," not in ln.operands
                ):
                    # 1-operand imul: writes EDX:EAX
                    if push_reg in ("eax", "edx"):
                        chain_writes_push_reg = True
                j += 1
            if (
                not chain_safe
                or pop_idx is None
                or chain_writes_push_reg
            ):
                i += 1
                continue
            # Pop reg.
            pop_reg = out[pop_idx].operands.strip().lower()
            if pop_reg not in PeepholeOptimizer._GP32:
                i += 1
                continue
            if pop_reg == push_reg:
                # Same-reg restore: standard save/restore. Other
                # passes handle this case (e.g. push_pop_op_to_memop
                # with mem-target, or right_operand_retarget).
                i += 1
                continue
            # Find the OP line right after pop.
            op_idx = None
            for k in range(pop_idx + 1, len(out)):
                if out[k].kind == "label":
                    break
                if out[k].kind == "instr":
                    op_idx = k
                    break
            if op_idx is None:
                i += 1
                continue
            op_line = out[op_idx]
            if op_line.op not in PeepholeOptimizer._COMMUTATIVE_BINOPS:
                i += 1
                continue
            opp = _operands_split(op_line.operands)
            if opp is None:
                i += 1
                continue
            target = opp[0].strip().lower()
            op_src = opp[1].strip().lower()
            if target not in PeepholeOptimizer._GP32:
                i += 1
                continue
            # All three regs distinct.
            if target in (pop_reg, push_reg) or push_reg == pop_reg:
                i += 1
                continue
            if op_src != pop_reg:
                i += 1
                continue
            # pop_reg dead after op. Use treat_as_scratch=True
            # because the codegen never relies on EDX being the LL
            # high-half return value at this point — it would set
            # EDX explicitly before function exit, not via this
            # bit-field-write idiom's pop.
            if not self._reg_dead_after(
                out, op_idx + 1, pop_reg, treat_as_scratch=True
            ):
                i += 1
                continue
            # Rewrite. New OP uses push_reg as source.
            indent = self._extract_indent(op_line.raw)
            spacer = " " * max(8 - len(op_line.op), 1)
            new_op_raw = (
                f"{indent}{op_line.op}{spacer}{target}, {push_reg}"
            )
            new_op_line = Line(
                raw=new_op_raw,
                kind="instr",
                op=op_line.op,
                operands=f"{target}, {push_reg}",
            )
            new_out = []
            new_out.extend(out[:i])
            new_out.extend(out[i + 1:pop_idx])
            new_out.extend(out[pop_idx + 1:op_idx])
            new_out.append(new_op_line)
            new_out.extend(out[op_idx + 1:])
            out = new_out
            self.stats["push_pop_register_op"] = (
                self.stats.get("push_pop_register_op", 0) + 1
            )
            continue
        return out

    def _pass_or_with_zero_reg_drop(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop ``or REG_T, REG_S`` when REG_S is known to be 0.

        After ``xor REG_S, REG_S`` (and no subsequent write to REG_S),
        REG_S = 0. ``or REG_T, 0 = REG_T`` — the OR is a no-op for
        the value. Flags differ in general (or sets flags based on
        REG_T's value; pre-OR flags reflect whatever set them last),
        so we require the flags to be dead after the OR.

        Saves 2 bytes per match (drops the `or reg, reg`).

        Common after slice 30 + 31 cascade in bit-field init code:
        the codegen ORs zeroed registers together, but the ORs are
        all no-ops when both regs are 0.

        Conditions:
        - Track per-register `zero_regs` state in a basic block.
          Set on `xor reg, reg`; cleared on any write to reg or at
          control-flow boundaries.
        - At `or REG_T, REG_S` where REG_S ∈ zero_regs, drop the OR.
        - Flag state safe at the drop point (no flag-reader between
          the would-be OR and the next flag-clobber).
        """
        out = list(lines)
        i = 0
        zero_regs: set[str] = set()
        while i < len(out):
            ln = out[i]
            if ln.kind == "label":
                zero_regs.clear()
                i += 1
                continue
            if ln.kind != "instr":
                i += 1
                continue
            op = ln.op
            # Control flow boundaries: clear all tracking.
            if op in (
                "call", "ret", "iret", "iretd", "retf", "retn",
                "leave", "jmp", "enter",
            ) or (op.startswith("j") and op != "jmp"):
                zero_regs.clear()
                i += 1
                continue
            # `xor reg, reg`: reg is now 0.
            if op == "xor":
                op_parts = _operands_split(ln.operands)
                if op_parts is not None:
                    a_dst, a_src = op_parts
                    a_dst_low = a_dst.strip().lower()
                    a_src_low = a_src.strip().lower()
                    if (
                        a_dst_low in PeepholeOptimizer._GP32
                        and a_dst_low == a_src_low
                    ):
                        zero_regs.add(a_dst_low)
                        i += 1
                        continue
            # `or REG_T, REG_S` where REG_S ∈ zero_regs: drop.
            if op == "or":
                op_parts = _operands_split(ln.operands)
                if op_parts is not None:
                    t_dst, t_src = op_parts
                    t_dst_low = t_dst.strip().lower()
                    t_src_low = t_src.strip().lower()
                    if (
                        t_dst_low in PeepholeOptimizer._GP32
                        and t_src_low in zero_regs
                        and t_src_low != t_dst_low
                        and self._flags_safe_after(out, i + 1)
                    ):
                        # Drop the OR. zero_regs unchanged (REG_T's
                        # zero status is unaffected by the no-op OR;
                        # if REG_T was 0, it still is. If REG_T was
                        # not 0, it stays not 0).
                        del out[i]
                        self.stats["or_with_zero_reg_drop"] = (
                            self.stats.get(
                                "or_with_zero_reg_drop", 0
                            ) + 1
                        )
                        # Don't advance i; new content at i.
                        continue
            # `shl/shr/sar/sal/rol/ror REG, X` where REG ∈ zero_regs:
            # all bit shifts/rotates of 0 yield 0. Drop.
            if op in ("shl", "shr", "sar", "sal", "rol", "ror"):
                op_parts = _operands_split(ln.operands)
                if op_parts is not None:
                    t_dst, t_src = op_parts
                    t_dst_low = t_dst.strip().lower()
                    if (
                        t_dst_low in zero_regs
                        and self._flags_safe_after(out, i + 1)
                    ):
                        del out[i]
                        self.stats["shift_on_zero_reg_drop"] = (
                            self.stats.get(
                                "shift_on_zero_reg_drop", 0
                            ) + 1
                        )
                        continue
            # `and REG, X` where REG ∈ zero_regs: drop (REG | 0
            # would be REG, but 0 AND X = 0 — REG stays 0).
            if op == "and":
                op_parts = _operands_split(ln.operands)
                if op_parts is not None:
                    t_dst, t_src = op_parts
                    t_dst_low = t_dst.strip().lower()
                    t_src_low = t_src.strip().lower()
                    # Require dst is a known-zero reg, AND src is
                    # not the same reg (`and reg, reg` keeps reg
                    # unchanged but sets flags — different idiom).
                    if (
                        t_dst_low in zero_regs
                        and t_src_low != t_dst_low
                        and self._flags_safe_after(out, i + 1)
                    ):
                        del out[i]
                        self.stats["and_on_zero_reg_drop"] = (
                            self.stats.get(
                                "and_on_zero_reg_drop", 0
                            ) + 1
                        )
                        # zero_regs unchanged — REG was 0, still 0.
                        continue
            # Generic: any write to a reg invalidates its zero state.
            op_parts = _operands_split(ln.operands or "")
            if op_parts is not None:
                dest = op_parts[0].strip().lower()
                # Sub-register write invalidates the canonical 32-bit.
                dest_canon = (
                    PeepholeOptimizer._SUBREG_TO_GP32.get(
                        dest, dest
                    )
                )
                if dest_canon in zero_regs:
                    # Read-write ops like add/sub/and/or/xor/shl
                    # — they modify the register, but if they're
                    # `op reg, 0` they preserve. We're conservative:
                    # any write clears.
                    zero_regs.discard(dest_canon)
            else:
                # Single-operand RMW (inc/dec/neg/not/pop/etc.).
                operand = (ln.operands or "").strip().lower()
                operand_canon = (
                    PeepholeOptimizer._SUBREG_TO_GP32.get(
                        operand, operand
                    )
                )
                if operand_canon in zero_regs and op in (
                    "inc", "dec", "neg", "not", "pop",
                    "mul", "div", "idiv",
                ):
                    zero_regs.discard(operand_canon)
            # Implicit reg writers.
            if op == "cdq":
                zero_regs.discard("edx")
            if op in ("mul", "div", "idiv"):
                zero_regs.discard("eax")
                zero_regs.discard("edx")
            if op == "imul" and ln.operands and "," not in ln.operands:
                zero_regs.discard("eax")
                zero_regs.discard("edx")
            i += 1
        return out

    def _pass_dead_xor_or_chain_drop(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop a dead `xor REG_T, REG_T; <chain>; or REG_T, REG_X`
        triplet when REG_T is dead after the OR.

        Pattern (with no store requirement, unlike slice 34):
            xor   REG_T, REG_T
            ... chain (no labels, calls, jumps,
                       no read/write of REG_T) ...
            or    REG_T, REG_X

        If REG_T is dead after the OR, both the xor and the or can
        be dropped. Saves 4 bytes per match.

        Common in bit-field assemble idioms where the codegen emits
        the assemble triplet but the slot is never stored (the slot
        is unread elsewhere — `dead_unused_slot_stores` already
        dropped the store).

        Conditions:
        - REG_T and REG_X are distinct GP32 regs.
        - Chain doesn't read/write REG_T (REG_T stays 0).
        - Chain has no labels, calls, jumps.
        - REG_T is dead after the OR.
        - Flags safe after (xor + or set flags; we drop both).
        """
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            if a.kind != "instr" or a.op != "xor":
                i += 1
                continue
            a_parts = _operands_split(a.operands)
            if a_parts is None:
                i += 1
                continue
            a_dst, a_src = a_parts
            a_dst_low = a_dst.strip().lower()
            a_src_low = a_src.strip().lower()
            if (
                a_dst_low not in PeepholeOptimizer._GP32
                or a_dst_low != a_src_low
            ):
                i += 1
                continue
            t_reg = a_dst_low
            # Walk forward looking for `or REG_T, REG_X`.
            or_idx = None
            chain_safe = True
            j = i + 1
            while j < len(out) and j < i + 12:
                ln = out[j]
                if ln.kind == "label":
                    chain_safe = False
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                op = ln.op
                if op in (
                    "call", "ret", "iret", "iretd", "retf", "retn",
                    "leave", "jmp", "enter",
                ) or (op.startswith("j") and op != "jmp"):
                    chain_safe = False
                    break
                if op == "or":
                    o_parts = _operands_split(ln.operands)
                    if o_parts is not None:
                        o_dst, o_src = o_parts
                        if (
                            o_dst.strip().lower() == t_reg
                            and o_src.strip().lower()
                                in PeepholeOptimizer._GP32
                            and o_src.strip().lower() != t_reg
                        ):
                            or_idx = j
                            break
                if self._references_reg_family(ln.operands, t_reg):
                    chain_safe = False
                    break
                if op in ("cdq", "mul", "div", "idiv"):
                    if t_reg in ("eax", "edx"):
                        chain_safe = False
                        break
                if (
                    op == "imul" and ln.operands
                    and "," not in ln.operands
                ):
                    if t_reg in ("eax", "edx"):
                        chain_safe = False
                        break
                j += 1
            if not chain_safe or or_idx is None:
                i += 1
                continue
            # REG_T dead after the OR.
            if not self._reg_dead_after(
                out, or_idx + 1, t_reg, treat_as_scratch=True
            ):
                i += 1
                continue
            # Flags safe after.
            if not self._flags_safe_after(out, or_idx + 1):
                i += 1
                continue
            # Drop xor and or.
            new_out = list(out[:i])
            new_out.extend(out[i + 1:or_idx])  # chain (drop xor)
            new_out.extend(out[or_idx + 1:])  # drop or
            out = new_out
            self.stats["dead_xor_or_chain_drop"] = (
                self.stats.get("dead_xor_or_chain_drop", 0) + 1
            )
            continue
        return out

    def _pass_const_fold_chain(
        self, lines: list[Line]
    ) -> list[Line]:
        """Constant-fold ``mov reg, IMM_INIT; <imm-op chain on reg>``
        to a single ``mov reg, RESULT``.

        Pattern:
            mov   reg, IMM_INIT
            op1   reg, IMM1
            op2   reg, IMM2
            ... (chain of imm-source ops on reg)

        For ops in {and, or, xor, add, sub, shl, shr, sar} where each
        op's source is an immediate, simulate the chain at peephole
        time and emit a single ``mov reg, RESULT``. Saves bytes per
        chain instruction folded.

        Common in bit-field encoding idioms where the codegen emits
        ``mov eax, IMM; and eax, MASK; shl eax, BITPOS`` to position
        a value into bit-field bits — these are entirely constant
        and should fold.

        Conditions:
        - reg is a 32-bit GP register.
        - Initial `mov reg, IMM_INIT` with IMM_INIT a numeric literal.
        - Subsequent chain ops are immediate-source arithmetic on the
          same reg, with numeric literal source.
        - Chain length ≥ 1 (a single fold gains at least 2 bytes).
        - Flags safe after the chain (chain's last op sets flags;
          the folded mov doesn't).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            a = out[i]
            if a.kind != "instr" or a.op != "mov":
                i += 1
                continue
            a_parts = _operands_split(a.operands)
            if a_parts is None:
                i += 1
                continue
            a_dst, a_src = a_parts
            a_dst_low = a_dst.strip().lower()
            if a_dst_low not in PeepholeOptimizer._GP32:
                i += 1
                continue
            try:
                acc = int(a_src.strip(), 0)
            except ValueError:
                i += 1
                continue
            # Mask to 32 bits.
            acc = acc & 0xFFFFFFFF
            # Look past intervening non-reg-writing instructions to
            # find the chain start. Allow: instructions that don't
            # write reg, don't contain labels, don't make calls, and
            # aren't unconditional control flow. Disallow: any write
            # to reg, labels (incoming jumps), calls (callee clobber),
            # unconditional jumps (chain becomes unreachable; dead).
            chain_start = i + 1
            while chain_start < len(out):
                ln = out[chain_start]
                if ln.kind in ("blank", "comment"):
                    chain_start += 1
                    continue
                if ln.kind == "label":
                    break
                if ln.kind != "instr":
                    break
                op = ln.op
                # Calls clobber caller-saved regs (eax, ecx, edx).
                if op == "call" and a_dst_low in (
                    "eax", "ecx", "edx"
                ):
                    break
                # Unconditional flow ends fall-through to chain.
                if op in (
                    "ret", "iret", "iretd", "retf", "retn",
                    "leave", "jmp", "enter",
                ):
                    break
                # Foldable chain op on reg = chain start.
                if op in (
                    "and", "or", "xor", "add", "sub",
                    "shl", "shr", "sar", "sal",
                ):
                    op_parts = _operands_split(ln.operands)
                    if op_parts is None:
                        break
                    op_dst, op_src = op_parts
                    if op_dst.strip().lower() == a_dst_low:
                        try:
                            int(op_src.strip(), 0)
                            # This is the chain start.
                            break
                        except ValueError:
                            break
                # Single-operand RMW on reg (not/neg) = chain start.
                if op in ("not", "neg"):
                    op_dst = ln.operands.strip()
                    if op_dst.lower() == a_dst_low:
                        break
                # Otherwise: check the instruction doesn't write reg.
                if self._instr_writes_reg(ln, a_dst_low):
                    break
                chain_start += 1
            # Walk forward consuming chain ops on reg.
            chain_end = chain_start
            chain_count = 0
            chain_acc = acc
            while chain_end < len(out):
                ln = out[chain_end]
                if ln.kind in ("blank", "comment"):
                    chain_end += 1
                    continue
                if ln.kind != "instr":
                    break
                op = ln.op
                if op not in (
                    "and", "or", "xor", "add", "sub",
                    "shl", "shr", "sar", "sal",
                    "not", "neg",
                ):
                    break
                # Single-operand (not, neg) vs 2-operand (rest) vs
                # 3-operand (imul reg, reg, IMM).
                if op in ("not", "neg"):
                    op_dst = ln.operands.strip()
                    if op_dst.lower() != a_dst_low:
                        break
                    if op == "not":
                        chain_acc = (~chain_acc) & 0xFFFFFFFF
                    else:  # neg
                        chain_acc = (-chain_acc) & 0xFFFFFFFF
                    chain_count += 1
                    chain_end += 1
                    continue
                op_parts = _operands_split(ln.operands)
                if op_parts is None:
                    break
                op_dst, op_src = op_parts
                if op_dst.strip().lower() != a_dst_low:
                    break
                try:
                    imm = int(op_src.strip(), 0)
                except ValueError:
                    break
                # Apply op to chain_acc.
                if op == "and":
                    chain_acc = chain_acc & (imm & 0xFFFFFFFF)
                elif op == "or":
                    chain_acc = (chain_acc | imm) & 0xFFFFFFFF
                elif op == "xor":
                    chain_acc = (chain_acc ^ imm) & 0xFFFFFFFF
                elif op == "add":
                    chain_acc = (chain_acc + imm) & 0xFFFFFFFF
                elif op == "sub":
                    chain_acc = (chain_acc - imm) & 0xFFFFFFFF
                elif op in ("shl", "sal"):
                    chain_acc = (chain_acc << (imm & 0x1F)) & 0xFFFFFFFF
                elif op == "shr":
                    chain_acc = (chain_acc >> (imm & 0x1F)) & 0xFFFFFFFF
                elif op == "sar":
                    # Arithmetic shift right (sign-extend high bit).
                    if chain_acc & 0x80000000:
                        # Sign-extend
                        signed = chain_acc - 0x100000000
                        chain_acc = (signed >> (imm & 0x1F)) & 0xFFFFFFFF
                    else:
                        chain_acc = (chain_acc >> (imm & 0x1F)) & 0xFFFFFFFF
                chain_count += 1
                chain_end += 1
            if chain_count == 0:
                i += 1
                continue
            # Flags safe after the last chain op.
            if not self._flags_safe_after(out, chain_end):
                i += 1
                continue
            # Rewrite. The folded mov replaces the chain region.
            # When there's no intervening (chain_start == i + 1), we
            # replace A directly (drops the original `mov reg, IMM`
            # since the folded mov supersedes it). When there IS
            # intervening code (chain_start > i + 1), we KEEP A and
            # the intervening so the intervening code still sees the
            # original IMM in reg; the folded mov goes after the
            # intervening, replacing the chain.
            indent = self._extract_indent(a.raw)
            new_raw = f"{indent}mov     {a_dst_low}, {chain_acc}"
            new_mov = Line(
                raw=new_raw, kind="instr",
                op="mov", operands=f"{a_dst_low}, {chain_acc}",
            )
            if chain_start == i + 1:
                # No intervening — original behavior.
                new_out = list(out[:i])
                new_out.append(new_mov)
                new_out.extend(out[chain_end:])
            else:
                # Keep A, keep intervening, replace chain.
                new_out = list(out[:i + 1])
                new_out.extend(out[i + 1:chain_start])
                new_out.append(new_mov)
                new_out.extend(out[chain_end:])
            out = new_out
            self.stats["const_fold_chain"] = (
                self.stats.get("const_fold_chain", 0) + chain_count
            )
            # Don't advance i; the new mov could be folded with more.
            continue
        return out

    def _pass_xor_or_store_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the bit-field assemble idiom:

            xor   REG_T, REG_T
            ... chain (no labels, calls, jumps,
                       no read/write of REG_T) ...
            or    REG_T, REG_X      ; REG_T = 0|REG_X = REG_X
            mov   [m], REG_T        ; store REG_T = REG_X

        Rewrite:
            ... chain ...
            mov   [m], REG_X

        Saves 4 bytes per match (xor 2 bytes + or 2 bytes = 4 bytes;
        mov stays the same length).

        Common in bit-field write idioms where the codegen first
        zeros REG_T then ORs in the value to be stored.

        Conditions:
        - REG_T and REG_X are both 32-bit GP registers, distinct.
        - Chain has no labels, no calls, no jumps, no read/write
          of REG_T (REG_T is dead through the chain).
        - REG_T is dead after the mov (since we drop the OR which
          would have left REG_T = REG_X; downstream reads of REG_T
          would see different values).
        - Flags safe after the mov (xor + or set flags; we drop
          both — the rewrite preserves only the chain's last flag
          state, which differs in CF/OF from or's behavior).
        - The mov stores REG_T to memory `[m]`.
        """
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            if a.kind != "instr" or a.op != "xor":
                i += 1
                continue
            a_parts = _operands_split(a.operands)
            if a_parts is None:
                i += 1
                continue
            a_dst, a_src = a_parts
            a_dst_low = a_dst.strip().lower()
            a_src_low = a_src.strip().lower()
            if (
                a_dst_low not in PeepholeOptimizer._GP32
                or a_dst_low != a_src_low
            ):
                i += 1
                continue
            t_reg = a_dst_low
            # Walk forward looking for `or REG_T, REG_X` followed by
            # `mov [m], REG_T`. Chain must not read/write REG_T.
            or_idx = None
            chain_safe = True
            j = i + 1
            while j < len(out) and j < i + 12:
                ln = out[j]
                if ln.kind == "label":
                    chain_safe = False
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                op = ln.op
                # Control flow.
                if op in (
                    "call", "ret", "iret", "iretd", "retf", "retn",
                    "leave", "jmp", "enter",
                ) or (op.startswith("j") and op != "jmp"):
                    chain_safe = False
                    break
                # Match the OR target.
                if op == "or":
                    o_parts = _operands_split(ln.operands)
                    if o_parts is not None:
                        o_dst, o_src = o_parts
                        if (
                            o_dst.strip().lower() == t_reg
                            and o_src.strip().lower()
                                in PeepholeOptimizer._GP32
                            and o_src.strip().lower() != t_reg
                        ):
                            or_idx = j
                            x_reg = o_src.strip().lower()
                            break
                # Check the chain doesn't read or write REG_T.
                if self._references_reg_family(ln.operands, t_reg):
                    chain_safe = False
                    break
                # Implicit reg writers/readers.
                if op in ("cdq", "mul", "div", "idiv"):
                    if t_reg in ("eax", "edx"):
                        chain_safe = False
                        break
                if (
                    op == "imul" and ln.operands
                    and "," not in ln.operands
                ):
                    if t_reg in ("eax", "edx"):
                        chain_safe = False
                        break
                j += 1
            if not chain_safe or or_idx is None:
                i += 1
                continue
            # Find the mov [m], REG_T immediately after the OR.
            mov_idx = None
            for k in range(or_idx + 1, len(out)):
                if out[k].kind == "label":
                    break
                if out[k].kind != "instr":
                    continue
                mov_idx = k
                break
            if mov_idx is None:
                i += 1
                continue
            mov_ln = out[mov_idx]
            if mov_ln.op != "mov":
                i += 1
                continue
            m_parts = _operands_split(mov_ln.operands)
            if m_parts is None:
                i += 1
                continue
            m_dst, m_src = m_parts
            if (
                m_src.strip().lower() != t_reg
                or "[" not in m_dst
            ):
                i += 1
                continue
            # REG_T dead after the mov.
            if not self._reg_dead_after(out, mov_idx + 1, t_reg):
                i += 1
                continue
            # Flags safe after.
            if not self._flags_safe_after(out, mov_idx + 1):
                i += 1
                continue
            # Rewrite. Replace mov's source REG_T with REG_X. Drop
            # xor (at i) and or (at or_idx).
            indent = self._extract_indent(mov_ln.raw)
            new_mov_raw = (
                f"{indent}mov     {m_dst.strip()}, {x_reg}"
            )
            new_mov_line = Line(
                raw=new_mov_raw,
                kind="instr",
                op="mov",
                operands=f"{m_dst.strip()}, {x_reg}",
            )
            new_out = []
            new_out.extend(out[:i])  # before xor
            new_out.extend(out[i + 1:or_idx])  # chain (drop xor)
            new_out.extend(out[or_idx + 1:mov_idx])  # between or & mov (drop or)
            new_out.append(new_mov_line)
            new_out.extend(out[mov_idx + 1:])
            out = new_out
            self.stats["xor_or_store_collapse"] = (
                self.stats.get("xor_or_store_collapse", 0) + 1
            )
            # Don't advance i; new content at i.
            continue
        return out

    def _pass_xor_then_and_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop ``and REG, X`` immediately after ``xor REG, REG``.

        After `xor REG, REG`, REG is 0. ANDing 0 with anything
        leaves it 0. So the AND is dead.

        Flag state is preserved: both `xor REG, REG` and `and REG, X`
        (for REG=0) leave ZF=1, SF=0, PF=1, OF=0, CF=0 — identical.
        So no flag-deadness check needed.

        Saves 2-6 bytes per match (3-byte and-imm8-sign-ext or
        6-byte and-imm32 → gone).

        Common in bit-field init idioms after `zero_load_after_zero_store`
        replaces a load with xor: the surrounding `and REG, MASK`
        becomes a no-op.

        Conditions:
        - Adjacent instructions.
        - Line A is `xor REG, REG` (REG is 32-bit GP).
        - Line B is `and REG, X` (same REG; X is non-register
          immediate or memory — not REG itself).
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            if a.kind != "instr" or a.op != "xor":
                i += 1
                continue
            a_parts = _operands_split(a.operands)
            if a_parts is None:
                i += 1
                continue
            a_dst, a_src = a_parts
            a_dst_low = a_dst.strip().lower()
            a_src_low = a_src.strip().lower()
            if (
                a_dst_low not in PeepholeOptimizer._GP32
                or a_dst_low != a_src_low
            ):
                i += 1
                continue
            # Find next instr (skip blanks/comments).
            j = i + 1
            while j < len(out) and out[j].kind in ("blank", "comment"):
                j += 1
            if j >= len(out):
                i += 1
                continue
            b = out[j]
            if b.kind != "instr" or b.op != "and":
                i += 1
                continue
            b_parts = _operands_split(b.operands)
            if b_parts is None:
                i += 1
                continue
            b_dst, b_src = b_parts
            b_dst_low = b_dst.strip().lower()
            b_src_low = b_src.strip().lower()
            if b_dst_low != a_dst_low:
                i += 1
                continue
            # Skip if src is the same register (and reg, reg is a
            # standard idiom — different shape; my pass shouldn't
            # rewrite that).
            if b_src_low == a_dst_low:
                i += 1
                continue
            # Drop B.
            del out[j]
            self.stats["xor_then_and_collapse"] = (
                self.stats.get("xor_then_and_collapse", 0) + 1
            )
            # Don't advance i — there may be another and after this.
            continue
        return out

    def _pass_zero_load_after_zero_store(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace ``mov REG, [m]`` with ``xor REG, REG`` when the
        slot was just zero-stored and no intervening write/control
        flow has happened.

        Pattern:
            xor   REG_A, REG_A
            mov   [m], REG_A          ; slot = 0
            ... chain (no labels, no calls, no jumps,
                       no write to [m], no lea of &m) ...
            mov   REG_B, [m]          ; load — slot is still 0

        Rewrite the final ``mov REG_B, [m]`` to ``xor REG_B, REG_B``.
        Saves 1 byte per match (3-byte ebp-rel mov → 2-byte xor reg).

        Common in bit-field init idioms where the codegen first
        zeroes the slot and then re-reads it during the field-write
        RMW. The reload is redundant — we know the value is 0.

        Conditions:
        - Memory operand is `[ebp ± N]` literal-offset.
        - Slot's offset is NOT address-taken anywhere in the function
          (else indirect writes via captured pointers could mutate).
        - Chain has no labels, no calls, no jumps, no writes to [m].
        - Flags must be safe at the new xor's position (xor sets
          ZF/SF/PF/CF/OF; original mov didn't touch them — so any
          flag-reader before the next flag-clobber would see
          different flags).
        - REG_B is a 32-bit GP register.
        """
        out = list(lines)
        addr_taken_per_line = self._compute_addr_taken_per_line(out)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            if (
                a.kind != "instr"
                or a.op != "xor"
            ):
                i += 1
                continue
            a_parts = _operands_split(a.operands)
            if a_parts is None:
                i += 1
                continue
            a_dst, a_src = a_parts
            a_dst_low = a_dst.strip().lower()
            a_src_low = a_src.strip().lower()
            if (
                a_dst_low not in PeepholeOptimizer._GP32
                or a_dst_low != a_src_low
            ):
                i += 1
                continue
            # B: mov [m], REG_A
            b = out[i + 1]
            if b.kind != "instr" or b.op != "mov":
                i += 1
                continue
            b_parts = _operands_split(b.operands)
            if b_parts is None:
                i += 1
                continue
            b_dst, b_src = b_parts
            if b_src.strip().lower() != a_dst_low:
                i += 1
                continue
            b_dst_clean = self._strip_size_prefix(b_dst.strip())
            slot_off = self._ebp_offset(b_dst_clean)
            if slot_off is None:
                i += 1
                continue
            # Slot must not be address-taken.
            if slot_off in addr_taken_per_line[i]:
                i += 1
                continue
            # Walk forward to find a load from same slot.
            load_idx = None
            chain_safe = True
            for j in range(i + 2, min(len(out), i + 12)):
                ln = out[j]
                if ln.kind == "label":
                    chain_safe = False
                    break
                if ln.kind != "instr":
                    continue
                if ln.op in ("call", "ret", "leave", "iret",
                             "iretd", "retf", "retn", "jmp",
                             "enter"):
                    chain_safe = False
                    break
                if ln.op.startswith("j") and ln.op != "jmp":
                    chain_safe = False
                    break
                # Check for write to [m].
                op_parts = _operands_split(ln.operands or "")
                if op_parts is not None:
                    dest = op_parts[0].strip()
                    if "[" in dest:
                        dest_off = self._ebp_offset(
                            self._strip_size_prefix(dest)
                        )
                        if dest_off is not None and dest_off == slot_off:
                            chain_safe = False
                            break
                else:
                    # Single-operand RMW like inc/dec on memory.
                    operand = (ln.operands or "").strip()
                    if "[" in operand:
                        op_off = self._ebp_offset(
                            self._strip_size_prefix(operand)
                        )
                        if (
                            op_off is not None
                            and op_off == slot_off
                            and ln.op in (
                                "inc", "dec", "neg", "not",
                            )
                        ):
                            chain_safe = False
                            break
                # LEA capturing &m.
                if ln.op == "lea" and op_parts is not None:
                    _, lea_src = op_parts
                    lea_off = self._ebp_offset(lea_src.strip())
                    if lea_off is not None and lea_off == slot_off:
                        chain_safe = False
                        break
                # Found a matching load?
                if ln.op == "mov" and op_parts is not None:
                    ld_dst, ld_src = op_parts
                    ld_dst_low = ld_dst.strip().lower()
                    ld_src_clean = self._strip_size_prefix(
                        ld_src.strip()
                    )
                    ld_off = self._ebp_offset(ld_src_clean)
                    if (
                        ld_dst_low in PeepholeOptimizer._GP32
                        and ld_off is not None
                        and ld_off == slot_off
                    ):
                        load_idx = j
                        load_dst = ld_dst_low
                        break
            if not chain_safe or load_idx is None:
                i += 1
                continue
            # Flags safety: from load_idx + 1 walk forward; until
            # first flag-clobber or fence, no flag-reader allowed.
            if not self._flags_safe_after(out, load_idx + 1):
                i += 1
                continue
            # Rewrite. Replace load with xor.
            indent = self._extract_indent(out[load_idx].raw)
            new_raw = f"{indent}xor     {load_dst}, {load_dst}"
            out[load_idx] = Line(
                raw=new_raw, kind="instr",
                op="xor", operands=f"{load_dst}, {load_dst}",
            )
            self.stats["zero_load_after_zero_store"] = (
                self.stats.get("zero_load_after_zero_store", 0) + 1
            )
            # Don't advance i; let the loop revisit. Since we replaced
            # the load with a register operation, no cascade is
            # immediately enabled at i, but a later optimization could.
            i += 1
        return out

    def _function_takes_ebp_addr(
        self, lines: list[Line], cur_idx: int, off: int
    ) -> bool:
        """Scan the enclosing function for any `lea reg, [ebp ± off]`
        that captures the address of the slot at the given offset.

        Walks outward from cur_idx until hitting a function-defining
        label (a top-level label whose name starts with `_` and isn't
        a `.local`). Conservatively returns True on syntactic match;
        we don't try to prove that the captured address doesn't
        escape — any escape opens the door to mid-chain mutation.
        """
        # Find function start: walk backward to the most recent
        # top-level label (kind=label, label starts with `_`, no `.`).
        start = 0
        for k in range(cur_idx - 1, -1, -1):
            ln = lines[k]
            if ln.kind == "label":
                lab = ln.label
                if lab and lab.startswith("_") and "." not in lab:
                    start = k + 1
                    break
        # Find function end: walk forward to the next top-level
        # label (or end of file).
        end = len(lines)
        for k in range(cur_idx + 1, len(lines)):
            ln = lines[k]
            if ln.kind == "label":
                lab = ln.label
                if lab and lab.startswith("_") and "." not in lab:
                    end = k
                    break
        # Build the X-pattern text we're matching against.
        sign = "+" if off > 0 else "-"
        target = f"[ebp {sign} {abs(off)}]"
        for k in range(start, end):
            ln = lines[k]
            if ln.kind != "instr" or ln.op != "lea":
                continue
            opp = _operands_split(ln.operands)
            if opp is None:
                continue
            _, src = opp
            src_norm = src.strip().lower()
            if src_norm == target:
                return True
        return False

    @staticmethod
    def _strip_size_prefix(text: str) -> str:
        """Strip optional NASM size prefix (`dword`, `word`, `byte`,
        `qword`) from a memory operand text."""
        s = text.strip()
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if s.lower().startswith(prefix):
                return s[len(prefix):].lstrip()
        return s

    def _pass_label_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG1, LABEL; mov REG2, [REG1]`` into
        ``mov REG2, [LABEL]`` using x86's disp32 absolute addressing.

        Saves 1 byte per match (5+2=7 bytes → 6 bytes for disp32-
        absolute mov). Common in global-variable access where the
        codegen emits a label load followed by a deref.

        Also handles the scaled-index case:
        ``mov REG1, LABEL; mov REG2, [REG1 + IDX*SCALE]`` →
        ``mov REG2, [LABEL + IDX*SCALE]``. Saves 1 byte (drops the
        REG1 load; the SIB-form mov adds disp32 to the addressing
        mode).

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov REG1, LABEL`` where LABEL is a non-numeric,
          non-memory expression (label or label-arithmetic).
        - Line B: ``mov REG2, [REG1]`` or ``mov REG2, [REG1 + IDX*SCALE]``
          (no offset on REG1).
        - REG1 either == REG2 (overwritten by the load) or dead
          after Line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            label = ap[1].strip()
            r2 = bp[0].strip().lower()
            src = bp[1].strip()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(r2)
            ):
                i += 1
                continue
            # LABEL must not be a number, register, or memory ref.
            if (
                "[" in label
                or self._is_general_register(label.lower())
            ):
                i += 1
                continue
            try:
                int(label)
                # numeric — that's `mov reg, IMM`. Not a label.
                i += 1
                continue
            except ValueError:
                pass
            # Source: `[REG1]` or `[REG1 + IDX*SCALE]`. Strip size.
            src_stripped = src
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if src_stripped.lower().startswith(prefix):
                    src_stripped = src_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            # Plain `[REG1]`.
            m_plain = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", src_stripped
            )
            # SIB form `[REG1 + IDX*SCALE]`.
            m_sib = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\+\s*"
                r"([a-zA-Z]+)\s*\*\s*([1248])\s*\]$",
                src_stripped,
            )
            if m_plain:
                base = m_plain.group(1).lower()
                sib_tail = ""
            elif m_sib:
                base = m_sib.group(1).lower()
                idx = m_sib.group(2).lower()
                scale = m_sib.group(3)
                # Don't fire when idx == r1 (the SIB also references
                # r1). Reordering: if idx is the label-loaded reg, we
                # can't rewrite — the index would be the label value.
                if idx == r1:
                    i += 1
                    continue
                sib_tail = f" + {idx}*{scale}"
            else:
                i += 1
                continue
            if base != r1:
                i += 1
                continue
            # Liveness: r1 dead after B (or r1 == r2).
            if r1 != r2 and not self._reg_dead_after(
                out, i + 2, r1
            ):
                i += 1
                continue
            # Build rewrite: `mov r2, [LABEL[+ idx*scale]]`.
            indent = self._extract_indent(b.raw)
            new_src = f"[{label}{sib_tail}]"
            new_raw = f"{indent}mov     {r2}, {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{r2}, {new_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["label_load_collapse"] = (
                self.stats.get("label_load_collapse", 0) + 1
            )
            continue
        return out

    def _pass_pop_index_push_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the array-index-then-push pattern that uses a
        prior `pop` for the base.

        Pattern:
            shl IDX, N           ; scale index
            pop BASE             ; restore previously-pushed base
            add IDX, BASE        ; compute address
            push dword [IDX]     ; push arr[i]

        Rewrite:
            pop BASE
            push dword [BASE + IDX*SCALE]

        Saves 4 bytes per match (drops shl + add; the SIB-form push
        adds 1 byte over the plain `push dword [reg]` form).

        Conditions:
        - 4 consecutive instr lines.
        - Line A: ``shl IDX, N`` for N ∈ {1, 2, 3}.
        - Line B: ``pop BASE``.
        - Line C: ``add IDX, BASE``.
        - Line D: ``push dword [IDX]``.
        - IDX dead after Line D (the SIB-form push reads IDX but
          doesn't write; original ended with IDX = base + idx*scale,
          new ends with IDX = unscaled index — different post-state).
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            d = out[i + 3]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "pop"
                and c.kind == "instr" and c.op == "add"
                and d.kind == "instr" and d.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            cp = _operands_split(c.operands)
            if ap is None or cp is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            base_reg = b.operands.strip().lower()
            c_dst = cp[0].strip().lower()
            c_src = cp[1].strip().lower()
            if (
                not self._is_general_register(idx_reg)
                or not self._is_general_register(base_reg)
                or c_dst != idx_reg
                or c_src != base_reg
                or idx_reg == base_reg
            ):
                i += 1
                continue
            # Line D's operand: `dword [IDX]`.
            d_op = d.operands.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if d_op.lower().startswith(prefix):
                    d_op_stripped = d_op[len(prefix):].lstrip()
                    break
            else:
                d_op_stripped = d_op
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", d_op_stripped
            )
            if mem_re is None or mem_re.group(1).lower() != idx_reg:
                i += 1
                continue
            # IDX dead after line D.
            if not self._reg_dead_after(out, i + 4, idx_reg):
                i += 1
                continue
            # Build rewrite.
            scale = SCALE[n]
            indent_b = self._extract_indent(b.raw)
            indent_d = self._extract_indent(d.raw)
            new_pop = Line(
                raw=b.raw,
                kind="instr", op="pop", operands=base_reg,
            )
            new_push_src = (
                f"dword [{base_reg} + {idx_reg}*{scale}]"
            )
            new_push = Line(
                raw=f"{indent_d}push    {new_push_src}",
                kind="instr",
                op="push",
                operands=new_push_src,
            )
            out = out[:i] + [new_pop, new_push] + out[i + 4:]
            self.stats["pop_index_push_collapse"] = (
                self.stats.get(
                    "pop_index_push_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_pop_index_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Sister of ``_pass_pop_index_push_collapse`` for the LOAD
        end. Collapse the array-index-then-load pattern that uses a
        prior ``pop`` for the base.

        Pattern:
            shl IDX, N           ; scale index
            pop BASE             ; restore previously-pushed base
            add IDX, BASE        ; compute address
            mov DST, [IDX]       ; load arr[i]

        Rewrite:
            pop BASE
            mov DST, [BASE + IDX*SCALE]

        Saves 5 bytes per match (drops shl + add; the SIB-form mov
        adds 1 byte over the plain ``mov DST, [reg]`` form).

        Conditions:
        - 4 consecutive instr lines.
        - Line A: ``shl IDX, N`` for N ∈ {1, 2, 3}.
        - Line B: ``pop BASE``.
        - Line C: ``add IDX, BASE``.
        - Line D: ``mov DST, [IDX]``.
        - IDX dead after Line D unless IDX == DST (then DST kills
          IDX's register naturally — it's the load target).
        - DST may be IDX, BASE, or different — all valid SIB forms.

        Common in compound-assign chains where the rhs is an array
        load through a previously-saved base — e.g. ``s += arr[i]``
        (after compound_assign_collapse fired) or the index-load
        directly inside any expression.
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            d = out[i + 3]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "pop"
                and c.kind == "instr" and c.op == "add"
                and d.kind == "instr" and d.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            cp = _operands_split(c.operands)
            dp = _operands_split(d.operands)
            if ap is None or cp is None or dp is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            base_reg = b.operands.strip().lower()
            c_dst = cp[0].strip().lower()
            c_src = cp[1].strip().lower()
            if (
                not self._is_general_register(idx_reg)
                or not self._is_general_register(base_reg)
                or c_dst != idx_reg
                or c_src != base_reg
                or idx_reg == base_reg
            ):
                i += 1
                continue
            # Line D: mov DST, [IDX]
            d_dst = dp[0].strip().lower()
            d_src = dp[1].strip()
            if not self._is_general_register(d_dst):
                i += 1
                continue
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if d_src.lower().startswith(prefix):
                    d_src = d_src[len(prefix):].lstrip()
                    break
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", d_src
            )
            if mem_re is None or mem_re.group(1).lower() != idx_reg:
                i += 1
                continue
            # IDX must be dead after the load — unless it's the
            # load target (in which case the load itself is the
            # final write).
            if d_dst != idx_reg:
                if not self._reg_dead_after(out, i + 4, idx_reg):
                    i += 1
                    continue
            # Build rewrite.
            scale = SCALE[n]
            indent_d = self._extract_indent(d.raw)
            new_pop = Line(
                raw=b.raw,
                kind="instr", op="pop", operands=base_reg,
            )
            new_load_src = (
                f"[{base_reg} + {idx_reg}*{scale}]"
            )
            new_load = Line(
                raw=f"{indent_d}mov     {d_dst}, {new_load_src}",
                kind="instr",
                op="mov",
                operands=f"{d_dst}, {new_load_src}",
            )
            out = out[:i] + [new_pop, new_load] + out[i + 4:]
            self.stats["pop_index_load_collapse"] = (
                self.stats.get(
                    "pop_index_load_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_push_pop_to_mov(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace ``push X; ... stack-balanced chain ... ; pop REG``
        with ``mov REG, X`` (placed where the pop was), dropping the
        push entirely.

        Saves 1 byte per match — push X + pop reg vs. mov reg, X.

        X may be:
        - A numeric immediate or label/label-arithmetic expression
          (the original case).
        - A memory operand (``[ebp ± N]``, ``[label]``, etc.) where
          the chain doesn't write to memory aliasing X.

        Conditions:
        - First instr: ``push X``.
        - Walk forward, tracking stack depth. Match the FIRST
          ``pop REG`` at depth 0.
        - Chain between push and pop must not access [esp + N], call,
          or ret/leave/enter. Any jump fences the rewrite.
        - If X is memory, the chain must not write to memory that
          may alias X.

        Common after pop_index_*_collapse drops the indexing chain
        but leaves the original `push BASE` and `pop BASE_REG` —
        which now have no chain between them or a trivial chain.
        Also fires on struct-copy retptr-save patterns.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            a = out[i]
            if not (a.kind == "instr" and a.op == "push"):
                i += 1
                continue
            push_op = a.operands.strip()
            # Skip register pushes — those are saving register values
            # and the codegen needs the round-trip.
            if self._is_general_register(push_op.lower()):
                i += 1
                continue
            # Optional size keyword strip.
            stripped = push_op
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):].lstrip()
                    break
            # Is it a memory operand?
            push_is_mem = stripped.startswith("[")
            # For memory operands, restrict to ebp-relative literal
            # offsets (`[ebp + N]` / `[ebp - N]`). ebp is callee-saved
            # and frame-relative; the chain can't change ebp without
            # invoking control flow we already bail on. Other forms
            # like `[eax + ecx*4]` reference registers the chain may
            # modify, making the post-pop value differ from the
            # pushed value.
            if push_is_mem and self._ebp_offset(stripped) is None:
                i += 1
                continue
            # Walk forward looking for matching pop.
            depth = 1  # we're past the push
            match_idx = None
            failed = False
            j = i + 1
            while j < len(out):
                ln = out[j]
                if ln.kind == "label":
                    # Label is an external entry point — control
                    # flow may reach the pop without our push.
                    failed = True
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                if ln.op == "push":
                    depth += 1
                    j += 1
                    continue
                if ln.op == "pop":
                    if depth == 1:
                        # This matches our push.
                        target_reg = ln.operands.strip().lower()
                        if not self._is_general_register(target_reg):
                            failed = True
                        else:
                            match_idx = j
                        break
                    depth -= 1
                    j += 1
                    continue
                # Stack-affecting ops fence the search.
                if ln.op in {
                    "call", "ret", "iret", "iretd", "retf",
                    "retn", "leave", "enter",
                }:
                    failed = True
                    break
                if ln.op.startswith("j"):
                    # Any jump fences (control flow may not reach pop).
                    failed = True
                    break
                # Operand-references-esp fence.
                if "esp" in (ln.operands or "").lower():
                    failed = True
                    break
                # If X is memory, bail on chain instructions that
                # may write to memory aliasing X.
                if push_is_mem:
                    ops_text = ln.operands or ""
                    if (
                        ops_text
                        and ln.op
                        not in self._READ_ONLY_FIRST_MEM
                    ):
                        first_op = (
                            ops_text.split(",", 1)[0].strip()
                            if "," in ops_text
                            else ops_text.strip()
                        )
                        if "[" in first_op:
                            if not self._mem_disjoint(
                                self._strip_size_prefix(first_op),
                                self._strip_size_prefix(stripped),
                            ):
                                failed = True
                                break
                j += 1
            if failed or match_idx is None:
                i += 1
                continue
            # Build rewrite: drop push at i, replace pop at
            # match_idx with `mov target_reg, push_op`.
            indent = self._extract_indent(out[match_idx].raw)
            new_mov = Line(
                raw=f"{indent}mov     {target_reg}, {stripped}",
                kind="instr",
                op="mov",
                operands=f"{target_reg}, {stripped}",
            )
            new_out = (
                out[:i] + out[i + 1: match_idx]
                + [new_mov] + out[match_idx + 1:]
            )
            out = new_out
            self.stats["push_pop_to_mov"] = (
                self.stats.get("push_pop_to_mov", 0) + 1
            )
            # Don't advance i (might match again from the same i).
            continue
        return out

    def _pass_dead_push_pop_pair(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop adjacent ``push reg1; pop reg2`` pairs when the popped
        register is dead after.

        Pattern:
            push reg1     (any GP32 register)
            pop  reg2     (any GP32 register)

        Cases:
        - reg1 == reg2: no-op (push then pop same register restores
          its value); drop both unconditionally.
        - reg1 != reg2 AND reg2 dead after: drop both. The pair sets
          reg2 = reg1 transiently; if reg2 isn't read before its
          next overwrite, the entire pair is dead.

        Saves 2 bytes per match (push reg = 1 byte, pop reg = 1 byte).

        Note: ``push_pop_to_mov`` (which handles non-register pushes)
        explicitly skips register pushes and the codegen's
        ``push reg; pop reg2`` save-restore pattern. My pass picks up
        the residual case where the popped register's transient value
        isn't actually used.

        Common after slice 37 (imm_store_then_imm_load_collapse) — the
        codegen sometimes emits ``push eax; pop edx`` to save a value
        into EDX when EDX is conventionally scratch but the surrounding
        code never actually reads EDX.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            a = lines[i]
            if (a.kind == "instr" and a.op == "push"
                    and i + 1 < len(lines)):
                a_op = a.operands.strip().lower()
                if a_op in PeepholeOptimizer._GP32:
                    b = lines[i + 1]
                    if b.kind == "instr" and b.op == "pop":
                        b_op = b.operands.strip().lower()
                        if b_op in PeepholeOptimizer._GP32:
                            # Case 1: same register — true no-op.
                            same = (a_op == b_op)
                            # Case 2: reg2 dead after.
                            dead = False
                            if not same:
                                dead = self._reg_dead_after(
                                    lines, i + 2, b_op,
                                    treat_as_scratch=True,
                                )
                            if same or dead:
                                self.stats[
                                    "dead_push_pop_pair"
                                ] = (
                                    self.stats.get(
                                        "dead_push_pop_pair", 0
                                    ) + 1
                                )
                                i += 2
                                continue
            out.append(a)
            i += 1
        return out

    def _pass_sib_const_index_fold(
        self, lines: list[Line]
    ) -> list[Line]:
        """Fold a constant-index SIB-form load. When the index
        register is loaded with a constant immediately before a
        SIB-form load that uses it, fold the constant into the
        displacement.

        Pattern (with optional displacement):
            mov IDX, IMM
            mov DST, [BASE + IDX*SCALE]
            mov DST, [BASE + IDX*SCALE + DISP]
            mov DST, [BASE + IDX*SCALE - DISP]

        Rewrite (drop the index load; fold IMM*SCALE into disp):
            mov DST, [BASE + (IMM*SCALE)]
            mov DST, [BASE + (IMM*SCALE + DISP)]
            mov DST, [BASE + (IMM*SCALE - DISP)]

        When the new displacement is 0, emit `[BASE]`. When BASE is
        the same register as IDX (rare), the rewrite is unsafe
        (BASE's value would change after the rewrite drops the
        const-load) and we bail.

        Saves 5 bytes per match (drops 5-byte `mov reg, imm32`).

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov IDX, IMM`` where IDX is a 32-bit GP reg
          and IMM is a numeric literal.
        - Line B: ``mov DST, [BASE + IDX*SCALE [DISP]]`` (with
          DISP optional, ±N form).
        - IDX dead after Line B (unless IDX == DST, in which
          case the load target overwrites IDX naturally).
        """
        SCALE_VALUES = {1, 2, 4, 8}
        # Loads using SIB form: mov, movsx, movzx.
        LOAD_OPS = {"mov", "movsx", "movzx"}
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr"
                and a.op in {"mov", "xor"}
                and b.kind == "instr" and b.op in LOAD_OPS
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            # Determine constant value of idx_reg.
            if a.op == "xor":
                # `xor reg, reg` zeros the register.
                if ap[1].strip().lower() != idx_reg:
                    i += 1
                    continue
                imm_val = 0
            else:
                # `mov reg, IMM` — must be a numeric literal.
                imm_str = ap[1].strip()
                try:
                    if imm_str.lower().startswith("0x"):
                        imm_val = int(imm_str, 16)
                    elif imm_str.lower().startswith("-0x"):
                        imm_val = -int(imm_str[1:], 16)
                    else:
                        imm_val = int(imm_str)
                except ValueError:
                    i += 1
                    continue
            # Parse Line B.
            dst_reg = bp[0].strip().lower()
            b_src = bp[1].strip()
            if not self._is_general_register(dst_reg):
                i += 1
                continue
            # Strip size prefix.
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_src.lower().startswith(prefix):
                    size_prefix = b_src[:len(prefix)]
                    b_src = b_src[len(prefix):].lstrip()
                    break
            # Match SIB form.
            sib_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\+\s*([a-zA-Z]+)\s*\*"
                r"\s*(\d+)"
                r"(?:\s*([+-])\s*(\d+))?"
                r"\s*\]$",
                b_src,
            )
            if sib_re is None:
                i += 1
                continue
            base_reg = sib_re.group(1).lower()
            sib_idx = sib_re.group(2).lower()
            try:
                scale = int(sib_re.group(3))
            except ValueError:
                i += 1
                continue
            if scale not in SCALE_VALUES:
                i += 1
                continue
            disp_sign = sib_re.group(4)
            disp_str = sib_re.group(5)
            disp = 0
            if disp_str is not None:
                try:
                    disp = int(disp_str)
                except ValueError:
                    i += 1
                    continue
                if disp_sign == "-":
                    disp = -disp
            # SIB index reg must match our IDX.
            if sib_idx != idx_reg:
                i += 1
                continue
            # BASE must NOT be IDX (otherwise BASE's value changes
            # after we drop the const-load).
            if base_reg == idx_reg:
                i += 1
                continue
            if not self._is_general_register(base_reg):
                i += 1
                continue
            # IDX dead after, unless DST == IDX.
            if dst_reg != idx_reg:
                if not self._reg_dead_after(out, i + 2, idx_reg):
                    i += 1
                    continue
            # Compute new displacement.
            new_disp = imm_val * scale + disp
            indent = self._extract_indent(b.raw)
            if new_disp == 0:
                new_src = f"{size_prefix}[{base_reg}]"
            elif new_disp > 0:
                new_src = (
                    f"{size_prefix}[{base_reg} + {new_disp}]"
                )
            else:
                new_src = (
                    f"{size_prefix}[{base_reg} - {-new_disp}]"
                )
            opname = b.op  # mov / movsx / movzx — preserve.
            spacer = " " * max(8 - len(opname), 1)
            new_raw = (
                f"{indent}{opname}{spacer}{dst_reg}, {new_src}"
            )
            new_line = Line(
                raw=new_raw, kind="instr", op=opname,
                operands=f"{dst_reg}, {new_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["sib_const_index_fold"] = (
                self.stats.get("sib_const_index_fold", 0) + 1
            )
            continue
        return out

    def _pass_push_const_index_fold(
        self, lines: list[Line]
    ) -> list[Line]:
        """Sister of `sib_const_index_fold` for the `push` form. Fold
        a constant-index SIB-form push into the displacement.

        Pattern (with optional displacement):
            mov IDX, IMM    (or xor IDX, IDX for IMM=0)
            push <size?> [BASE + IDX*SCALE]
            push <size?> [BASE + IDX*SCALE + DISP]
            push <size?> [BASE + IDX*SCALE - DISP]

        Rewrite (drop the index load; fold IMM*SCALE into disp):
            push <size?> [BASE + (IMM*SCALE)]
            push <size?> [BASE + (IMM*SCALE + DISP)]

        Saves 2 bytes per match for `xor reg, reg` (drops 2-byte op).
        Saves 5 bytes per match for `mov reg, IMM` (drops 5-byte op).

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov IDX, IMM`` (numeric literal) or
          ``xor IDX, IDX``.
        - Line B: ``push <size?> [BASE + IDX*SCALE [DISP]]``.
        - IDX dead after Line B.
        - BASE != IDX (otherwise BASE's value would change after
          dropping the const-load).
        """
        SCALE_VALUES = {1, 2, 4, 8}
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr"
                and a.op in {"mov", "xor"}
                and b.kind == "instr" and b.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            # Determine constant value of idx_reg.
            if a.op == "xor":
                if ap[1].strip().lower() != idx_reg:
                    i += 1
                    continue
                imm_val = 0
            else:
                imm_str = ap[1].strip()
                try:
                    if imm_str.lower().startswith("0x"):
                        imm_val = int(imm_str, 16)
                    elif imm_str.lower().startswith("-0x"):
                        imm_val = -int(imm_str[1:], 16)
                    else:
                        imm_val = int(imm_str)
                except ValueError:
                    i += 1
                    continue
            # Parse Line B's operand. Push has no destination — the
            # operand is the source (memory).
            push_src = b.operands.strip()
            # Strip size prefix.
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if push_src.lower().startswith(prefix):
                    size_prefix = push_src[:len(prefix)]
                    push_src = push_src[len(prefix):].lstrip()
                    break
            # Match SIB form.
            sib_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\+\s*([a-zA-Z]+)\s*\*"
                r"\s*(\d+)"
                r"(?:\s*([+-])\s*(\d+))?"
                r"\s*\]$",
                push_src,
            )
            if sib_re is None:
                i += 1
                continue
            base_reg = sib_re.group(1).lower()
            sib_idx = sib_re.group(2).lower()
            try:
                scale = int(sib_re.group(3))
            except ValueError:
                i += 1
                continue
            if scale not in SCALE_VALUES:
                i += 1
                continue
            disp_sign = sib_re.group(4)
            disp_str = sib_re.group(5)
            disp = 0
            if disp_str is not None:
                try:
                    disp = int(disp_str)
                except ValueError:
                    i += 1
                    continue
                if disp_sign == "-":
                    disp = -disp
            if sib_idx != idx_reg:
                i += 1
                continue
            if base_reg == idx_reg:
                i += 1
                continue
            if not self._is_general_register(base_reg):
                i += 1
                continue
            # IDX dead after the push (push doesn't write IDX, so
            # IDX retains its constant value). If IDX is read after,
            # the rewrite is unsafe.
            if not self._reg_dead_after(out, i + 2, idx_reg):
                i += 1
                continue
            # Compute new displacement.
            new_disp = imm_val * scale + disp
            indent = self._extract_indent(b.raw)
            if new_disp == 0:
                new_src = f"{size_prefix}[{base_reg}]"
            elif new_disp > 0:
                new_src = (
                    f"{size_prefix}[{base_reg} + {new_disp}]"
                )
            else:
                new_src = (
                    f"{size_prefix}[{base_reg} - {-new_disp}]"
                )
            new_raw = f"{indent}push    {new_src}"
            new_line = Line(
                raw=new_raw, kind="instr", op="push",
                operands=new_src,
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["push_const_index_fold"] = (
                self.stats.get("push_const_index_fold", 0) + 1
            )
            continue
        return out

    def _pass_byte_stores_to_dword(
        self, lines: list[Line]
    ) -> list[Line]:
        """Pack 4 consecutive ``mov byte [ebp +- N + i], imm8`` stores
        (i = 0, 1, 2, 3) into a single ``mov dword [ebp +- N], imm32``.

        Each `mov byte [m], imm8` is 4 bytes (`C6 ... imm8`). The
        equivalent `mov dword [m], imm32` is 7 bytes. Saves 9 bytes
        per 4-store group (16 → 7).

        Common in `char arr[N] = "string"` initialization where the
        codegen emits per-byte stores for each character.

        Conditions:
        - 4 consecutive instr lines.
        - All match ``mov byte [<base> ± N + j], imm8`` with the
          same base and consecutive offsets j = 0, 1, 2, 3 (or
          j = N, N+1, N+2, N+3 for any starting point).
        - All imm8 values are integer literals.

        Limited to ``[ebp ± N]`` addressing for safety. Other forms
        (`[reg + N]` register-base, label addresses, etc.) skipped.
        """
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            chunk = out[i:i + 4]
            packed = self._try_pack_byte_stores(chunk)
            if packed is None:
                i += 1
                continue
            # Replace 4 lines with 1.
            out = out[:i] + [packed] + out[i + 4:]
            self.stats["byte_stores_to_dword"] = (
                self.stats.get("byte_stores_to_dword", 0) + 1
            )
            # Don't advance — try to pack the next 4 from i again
            # (in case the next group is also packable).
            continue
        return out

    @staticmethod
    def _try_pack_byte_stores(
        chunk: list[Line],
    ) -> Line | None:
        """If the 4 lines form a packable byte-store group, return
        the replacement dword-store Line. Otherwise None."""
        if len(chunk) != 4:
            return None
        offsets: list[int] = []
        values: list[int] = []
        sign = "+"
        base_disp = 0
        for idx, ln in enumerate(chunk):
            if ln.kind != "instr" or ln.op != "mov":
                return None
            parts = _operands_split(ln.operands)
            if parts is None:
                return None
            dest, src = parts
            dest_norm = dest.strip()
            # Must be `byte [ebp + N]` or `byte [ebp - N]`.
            m = re.fullmatch(
                r"byte\s+\[\s*ebp\s*([+-])\s*(\d+)\s*\]",
                dest_norm,
                re.IGNORECASE,
            )
            if not m:
                return None
            disp = int(m.group(2)) if m.group(1) == "+" else -int(m.group(2))
            offsets.append(disp)
            try:
                values.append(int(src.strip()) & 0xFF)
            except ValueError:
                return None
            if idx == 0:
                sign = m.group(1)
                base_disp = disp
        # Offsets must be consecutive: base, base+1, base+2, base+3.
        if offsets != [base_disp + j for j in range(4)]:
            return None
        # Pack into little-endian dword.
        packed = (
            values[0]
            | (values[1] << 8)
            | (values[2] << 16)
            | (values[3] << 24)
        )
        # Build the rewrite. Use a positive offset+sign that matches
        # the base. The base is at offsets[0], which is base_disp.
        indent = re.match(r"^[ \t]*", chunk[0].raw).group(0)
        if base_disp >= 0:
            addr = f"[ebp + {base_disp}]"
        else:
            addr = f"[ebp - {-base_disp}]"
        new_raw = f"{indent}mov     dword {addr}, {packed}"
        return Line(
            raw=new_raw,
            kind="instr",
            op="mov",
            operands=f"dword {addr}, {packed}",
        )

    def _pass_label_push_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG, LABEL; push dword [REG]`` into
        ``push dword [LABEL]`` using x86's disp32-absolute push.

        Sister of ``label_load_collapse`` for the push case. Saves
        1 byte per match. Common in `printf("%d", glob);` arg-setup.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov REG, LABEL`` (LABEL is non-numeric, non-mem,
          non-register expression).
        - Line B: ``push dword [REG]``.
        - REG dead after Line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            label = ap[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            if (
                "[" in label
                or self._is_general_register(label.lower())
            ):
                i += 1
                continue
            try:
                int(label)
                i += 1
                continue
            except ValueError:
                pass
            # Push source: `dword [REG]`.
            b_op = b.operands.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_op.lower().startswith(prefix):
                    b_op = b_op[len(prefix):].lstrip()
                    break
            mem_re = re.match(r"^\[\s*([a-zA-Z]+)\s*\]$", b_op)
            if mem_re is None or mem_re.group(1).lower() != r1:
                i += 1
                continue
            # REG dead after.
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            indent = self._extract_indent(b.raw)
            new_src = f"dword [{label}]"
            new_raw = f"{indent}push    {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="push",
                operands=new_src,
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["label_push_collapse"] = (
                self.stats.get("label_push_collapse", 0) + 1
            )
            continue
        return out

    def _pass_label_store_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG, LABEL; mov <size> [REG], SRC`` into
        ``mov <size> [LABEL], SRC`` using x86's disp32-absolute store.

        Sister of ``label_load_collapse`` for the store case. Saves
        1 byte per match. Common in `glob = constant;` patterns where
        the codegen first loads the label address then stores into it.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov REG, LABEL`` (LABEL is non-numeric, non-mem,
          non-register expression).
        - Line B: ``mov <size> [REG], SRC`` (size in dword/word/byte;
          plain `[REG]` deref, no offset/SIB).
        - REG dead after Line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            label = ap[1].strip()
            b_dst = bp[0].strip()
            b_src = bp[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            if (
                "[" in label
                or self._is_general_register(label.lower())
            ):
                i += 1
                continue
            try:
                int(label)
                i += 1
                continue
            except ValueError:
                pass
            # Line B's dest must be `<size> [REG]` (with size keyword).
            m = re.match(
                r"^(dword|word|byte|qword)\s+\[\s*([a-zA-Z]+)\s*\]$",
                b_dst,
                re.IGNORECASE,
            )
            if m is None:
                i += 1
                continue
            size_kw = m.group(1).lower()
            mem_reg = m.group(2).lower()
            if mem_reg != r1:
                i += 1
                continue
            # SRC must not reference REG (else dropping the address
            # load breaks). E.g. `mov reg, LABEL; mov [reg], reg`
            # is `*LABEL = LABEL` which doesn't make sense to fold
            # without further analysis.
            if self._references_reg_family(b_src, r1):
                i += 1
                continue
            # REG dead after line B.
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            indent = self._extract_indent(b.raw)
            new_dst = f"{size_kw} [{label}]"
            new_raw = f"{indent}mov     {new_dst}, {b_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{new_dst}, {b_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["label_store_collapse"] = (
                self.stats.get("label_store_collapse", 0) + 1
            )
            continue
        return out

    def _pass_lea_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``lea REG, [ebp ± N]; mov REG2, [REG + M]`` (or
        ``[REG]``) into ``mov REG2, [ebp ± (N+M)]``.

        Saves 2-3 bytes per match by eliminating the LEA and folding
        the offset arithmetic into the load's addressing. Common in
        local struct member access where the codegen emits an
        explicit `lea` to compute the struct base, then `mov reg,
        [base + offset]` for each member.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``lea REG, [ebp ± N]`` (stack-relative base).
        - Line B: ``mov REG2, [REG]`` or ``mov REG2, [REG + M]``
          (plain or offset deref of REG).
        - REG dead after Line B (or REG == REG2).
        - Combined offset N+M is representable as a signed 32-bit
          displacement (always true for sane stack frames).
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            lea_src = ap[1].strip()
            r2 = bp[0].strip().lower()
            b_src = bp[1].strip()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(r2)
            ):
                i += 1
                continue
            # LEA src must be `[ebp ± N]` (stack-relative).
            m_lea = re.match(
                r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                lea_src,
                re.IGNORECASE,
            )
            if m_lea is None:
                i += 1
                continue
            n_disp = int(m_lea.group(2))
            if m_lea.group(1) == "-":
                n_disp = -n_disp
            # Mov src: `[REG]` or `[REG + M]` or `[REG - M]`.
            b_src_stripped = b_src
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_src_stripped.lower().startswith(prefix):
                    b_src_stripped = b_src_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            m_plain = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", b_src_stripped
            )
            m_disp = re.match(
                r"^\[\s*([a-zA-Z]+)\s*([+-])\s*(\d+)\s*\]$",
                b_src_stripped,
            )
            if m_plain:
                base = m_plain.group(1).lower()
                m_disp_val = 0
            elif m_disp:
                base = m_disp.group(1).lower()
                m_disp_val = int(m_disp.group(3))
                if m_disp.group(2) == "-":
                    m_disp_val = -m_disp_val
            else:
                i += 1
                continue
            if base != r1:
                i += 1
                continue
            # Liveness: REG dead after, or REG == REG2.
            if r1 != r2 and not self._reg_dead_after(
                out, i + 2, r1
            ):
                i += 1
                continue
            # Build combined offset.
            combined = n_disp + m_disp_val
            sign = "+" if combined >= 0 else "-"
            new_src = f"[ebp {sign} {abs(combined)}]"
            # Preserve the size keyword from the original mov.
            size_kw = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_src.lower().startswith(prefix):
                    size_kw = prefix
                    break
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}mov     {r2}, {size_kw}{new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{r2}, {size_kw}{new_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_load_collapse"] = (
                self.stats.get("lea_load_collapse", 0) + 1
            )
            continue
        return out

    def _pass_lea_sib_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Extension of `lea_load_collapse` that handles SIB-form
        addressing AND allows intermediate independent instructions.

        Pattern:
            lea     REG, [ebp ± N]                ; A
            ... independent intermediate ops ...   ; don't touch REG
            mov     REG2, [REG + IDX*SCALE]        ; B (or [REG] / [REG + M])

        Rewrite to:
            ... independent intermediate ops ...
            mov     REG2, [ebp ± N + IDX*SCALE]    ; (or [ebp ± (N+M)])
        Drop A.

        Saves 3 bytes per match (drops the 3-byte lea).

        Common in local array indexing:
            lea     eax, [ebp - 20]      ; address of arr[0]
            mov     ecx, [ebp - 32]      ; load i
            mov     eax, [eax + ecx*4]   ; arr[i]
        Collapses to `mov ecx, [ebp - 32]; mov eax, [ebp - 20 + ecx*4]`.

        Conditions:
        - Line A: ``lea REG, [ebp ± N]`` (stack-relative base).
        - Intermediate lines: must not write REG, not read REG (we
          care about REG's value at line B, not earlier reads, but
          the existing `lea_load_collapse` runs first for the
          adjacent case so we only fire when there's at least one
          intermediate).
        - Line B: ``mov REG2, [REG ± M]`` (plain or with disp) OR
          ``mov REG2, [REG + IDX*SCALE]`` (SIB form, possibly with
          additional disp).
        - REG dead after line B (or REG == REG2 — overwritten by load).
        - Combined offset N+M (if any) is representable as imm32.
        """
        out = list(lines)
        i = 0
        max_intermediate = 8
        while i < len(out):
            a = out[i]
            if not (a.kind == "instr" and a.op == "lea"):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            lea_src = ap[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            # LEA src must be `[ebp ± N]`.
            m_lea = re.match(
                r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                lea_src,
                re.IGNORECASE,
            )
            if m_lea is None:
                i += 1
                continue
            n_disp = int(m_lea.group(2))
            if m_lea.group(1) == "-":
                n_disp = -n_disp
            # Walk forward past independent instructions; find a load
            # using r1 as base. Bail if any intermediate touches r1.
            j = i + 1
            steps = 0
            while j < len(out) and steps < max_intermediate:
                ln = out[j]
                if ln.kind in ("blank", "comment"):
                    j += 1
                    continue
                if ln.kind != "instr":
                    break
                # Intermediate must not touch r1 (read or write).
                if self._references_reg_family(ln.operands, r1):
                    # If the FIRST encounter of r1 in operands is a
                    # mov reading [r1...] — that's our target.
                    if ln.op == "mov" and self._uses_reg_as_base(ln, r1):
                        break
                    # Any other reference (write to r1, use as plain
                    # source register, etc.) — bail.
                    break
                steps += 1
                j += 1
            else:
                i += 1
                continue
            if j >= len(out):
                i += 1
                continue
            b = out[j]
            if not (b.kind == "instr" and b.op == "mov"):
                i += 1
                continue
            if not self._uses_reg_as_base(b, r1):
                i += 1
                continue
            # Need at least one intermediate (else lea_load_collapse
            # would've handled it).
            if j == i + 1:
                # Adjacent — let lea_load_collapse handle it (this
                # pass extends to non-adjacent / SIB).
                # But also try SIB form for adjacent case.
                pass
            bp = _operands_split(b.operands)
            if bp is None:
                i += 1
                continue
            r2 = bp[0].strip().lower()
            b_src = bp[1].strip()
            if not self._is_general_register(r2):
                i += 1
                continue
            # Liveness: r1 dead after line B, OR r1 == r2 (overwritten).
            if r1 != r2 and not self._reg_dead_after(out, j + 1, r1):
                i += 1
                continue
            # Parse the address inside b_src and substitute.
            new_b_src = self._fold_lea_into_addr(b_src, r1, n_disp)
            if new_b_src is None:
                i += 1
                continue
            # Build new line.
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}mov     {r2}, {new_b_src}"
            new_b = Line(
                raw=new_raw, kind="instr",
                op="mov", operands=f"{r2}, {new_b_src}",
            )
            out = out[:i] + out[i + 1:j] + [new_b] + out[j + 1:]
            self.stats["lea_sib_load_collapse"] = (
                self.stats.get("lea_sib_load_collapse", 0) + 1
            )
            # Don't advance i; the substitution may enable more.
            continue
        return out

    @staticmethod
    def _uses_reg_as_base(line: Line, reg: str) -> bool:
        """Does `line` use `reg` as the BASE register in its memory
        addressing? Looks for `[reg]`, `[reg + ...]`, `[reg - ...]`.
        """
        if line.kind != "instr":
            return False
        ops = line.operands
        for m in re.finditer(r"\[([^\]]+)\]", ops):
            inner = m.group(1).strip()
            # Strip optional size prefix (already outside the brackets,
            # so this loop sees just the inside).
            # Match a bare reg, or reg followed by + / - / ...
            m_b = re.match(
                rf"^\s*({reg})\s*([+\-]|$)",
                inner,
                re.IGNORECASE,
            )
            if m_b:
                return True
        return False

    @staticmethod
    def _fold_lea_into_addr(
        b_src: str, lea_reg: str, lea_disp: int,
    ) -> str | None:
        """Substitute `[lea_reg + ...]` with `[ebp ± lea_disp + ...]`
        in `b_src` (the memory operand of a mov/op).

        Handles forms:
        - `[lea_reg]` → `[ebp ± lea_disp]`
        - `[lea_reg + M]` → `[ebp ± (lea_disp + M)]`
        - `[lea_reg + IDX*SCALE]` → `[ebp ± lea_disp + IDX*SCALE]`
        - `[lea_reg + IDX*SCALE + M]` → `[ebp ± (lea_disp+M) + IDX*SCALE]`

        Preserves any size keyword (dword/word/byte/qword).
        """
        s = b_src
        # Strip size prefix if present.
        size_kw = ""
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if s.lower().startswith(prefix):
                size_kw = prefix
                s = s[len(prefix):].lstrip()
                break
        if not (s.startswith("[") and s.endswith("]")):
            return None
        inner = s[1:-1].strip()
        # Patterns we accept:
        # [lea_reg]
        # [lea_reg + M]
        # [lea_reg - M]
        # [lea_reg + IDX*SCALE]
        # [lea_reg + IDX*SCALE + M]
        # [lea_reg + IDX*SCALE - M]
        pat_plain = re.compile(
            rf"^\s*({lea_reg})\s*$",
            re.IGNORECASE,
        )
        pat_disp = re.compile(
            rf"^\s*({lea_reg})\s*([+-])\s*(\d+)\s*$",
            re.IGNORECASE,
        )
        pat_sib = re.compile(
            rf"^\s*({lea_reg})\s*\+\s*(\w+)\s*\*\s*(\d+)"
            rf"(?:\s*([+-])\s*(\d+))?\s*$",
            re.IGNORECASE,
        )
        def fmt_disp(d: int) -> str:
            sign = "+" if d >= 0 else "-"
            return f"ebp {sign} {abs(d)}"

        if pat_plain.match(inner):
            new_inner = fmt_disp(lea_disp)
        elif (m := pat_disp.match(inner)):
            extra = int(m.group(3))
            if m.group(2) == "-":
                extra = -extra
            combined = lea_disp + extra
            new_inner = fmt_disp(combined)
        elif (m := pat_sib.match(inner)):
            idx = m.group(2)
            scale = int(m.group(3))
            extra = 0
            if m.group(4) is not None:
                extra = int(m.group(5))
                if m.group(4) == "-":
                    extra = -extra
            combined = lea_disp + extra
            sign = "+" if combined >= 0 else "-"
            new_inner = (
                f"ebp + {idx}*{scale} {sign} {abs(combined)}"
                if combined != 0
                else f"ebp + {idx}*{scale}"
            )
        else:
            return None
        return f"{size_kw}[{new_inner}]"

    def _pass_lea_sib_label_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse `lea REG, [LABEL + REG*scale (+ DISP_A)?]; <op>
        ..., [REG (+ DISP_B)?]` into a single op with the SIB form
        directly: `<op> ..., [LABEL + REG*scale + (DISP_A + DISP_B)]`.

        Drops the 6-byte+ LEA instruction. Saves 6+ bytes per match.

        Common shape: `if (g[i] == target)` or other operations on
        a global array element where the codegen materializes the
        address with `lea REG, [_g + REG*4]` then accesses via [REG].

        Pattern:
        - Line A: `lea REG, [LABEL + REG*N (+ DISP_A)?]` where:
          - REG is a 32-bit GP register
          - LABEL is a non-numeric, non-register expression
          - N ∈ {1, 2, 4, 8} (SIB scale factor; note: N=1 is the
            implicit scale, but we accept it explicitly)
          - DISP_A is optional integer displacement

        - Line B: an op whose memory operand is `[REG (+ DISP_B)?]`
          (plain or with displacement; no SIB).

        Rewrite: drop A, replace B's `[REG ± DISP_B]` operand with
        `[LABEL + REG*scale + (DISP_A ± DISP_B)]`.

        Conditions:
        - The op in B must support a memory-form operand (excludes
          ops that strictly use registers).
        - REG dead after B (treat_as_scratch=True since the lea was
          setting up an address scratch). Or REG == DST of B (load
          overwrites REG).
        - [REG ± DISP_B] is the ONLY memory operand (no other regs
          referenced in addressing).

        Supported B ops (memory-source operations):
        - `mov DST, [REG]` (load via REG)
        - `cmp/test [REG], X` (read-and-compare via REG)
        - `add/sub/and/or/xor/inc/dec [REG], X` (RMW via REG)
        - `push <size?> [REG]` (push from REG's deref)
        """
        SUPPORTED_OPS = {
            "mov", "cmp", "test", "add", "sub", "and", "or", "xor",
            "inc", "dec", "push",
        }
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op in SUPPORTED_OPS
            ):
                i += 1
                continue
            # Parse A: `lea REG, [LABEL + REG*N (+ DISP_A)?]`.
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            lea_src = ap[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            # Match `[LABEL + REG*N]` or `[LABEL + REG*N + D]` or
            # `[LABEL + REG*N - D]`.
            m = re.match(
                r"^\[\s*([^\[\]]+?)\s*\+\s*([a-zA-Z]+)\s*\*\s*"
                r"(\d+)\s*(?:([+-])\s*(\d+))?\s*\]$",
                lea_src,
            )
            if m is None:
                i += 1
                continue
            label = m.group(1).strip()
            sib_reg = m.group(2).lower()
            scale = int(m.group(3))
            disp_a_sign = m.group(4)
            disp_a_val = m.group(5)
            disp_a = 0
            if disp_a_sign and disp_a_val:
                disp_a = int(disp_a_val)
                if disp_a_sign == "-":
                    disp_a = -disp_a
            if scale not in (1, 2, 4, 8):
                i += 1
                continue
            if sib_reg != r1:
                i += 1
                continue
            # LABEL must be non-numeric, non-register.
            if self._is_general_register(label.lower()):
                i += 1
                continue
            try:
                int(label)
                i += 1
                continue
            except ValueError:
                pass
            try:
                int(label, 16)
                i += 1
                continue
            except ValueError:
                pass
            # Inspect B's operand structure.
            # For ops that take 2 operands (mov, cmp, test, add, sub,
            # and, or, xor): one of them must be `[REG (+ M)?]`.
            # For 1-op (inc, dec, push): the operand must be
            # `[REG (+ M)?]`.
            bp = _operands_split(b.operands)
            mem_operand_text = None
            mem_operand_idx = None  # 0 = first, 1 = second, None = N/A
            if bp is None:
                # 1-operand op.
                opnd = b.operands.strip()
                mem_operand_text = opnd
                mem_operand_idx = 0
            else:
                # 2-operand op.
                op0 = bp[0].strip()
                op1 = bp[1].strip()
                if op0.startswith("[") or op0.lower().startswith(
                    ("dword ", "word ", "byte ", "qword ")
                ) and "[" in op0:
                    mem_operand_text = op0
                    mem_operand_idx = 0
                elif op1.startswith("[") or op1.lower().startswith(
                    ("dword ", "word ", "byte ", "qword ")
                ) and "[" in op1:
                    mem_operand_text = op1
                    mem_operand_idx = 1
                else:
                    i += 1
                    continue
            if mem_operand_text is None:
                i += 1
                continue
            # Strip size prefix.
            size_kw = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if mem_operand_text.lower().startswith(prefix):
                    size_kw = mem_operand_text[:len(prefix)]
                    mem_operand_text = mem_operand_text[
                        len(prefix):
                    ].lstrip()
                    break
            # Match `[REG]` or `[REG ± M]`. No SIB form allowed in B.
            mb_plain = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$",
                mem_operand_text,
            )
            mb_disp = re.match(
                r"^\[\s*([a-zA-Z]+)\s*([+-])\s*(\d+)\s*\]$",
                mem_operand_text,
            )
            if mb_plain:
                base_reg = mb_plain.group(1).lower()
                disp_b = 0
            elif mb_disp:
                base_reg = mb_disp.group(1).lower()
                disp_b = int(mb_disp.group(3))
                if mb_disp.group(2) == "-":
                    disp_b = -disp_b
            else:
                i += 1
                continue
            if base_reg != r1:
                i += 1
                continue
            # Determine DST register (for the load case where REG ==
            # DST, REG's post-state is overwritten anyway).
            dst_reg = None
            if b.op == "mov" and bp is not None and mem_operand_idx == 1:
                # `mov DST, [REG]` (load).
                dst_candidate = bp[0].strip().lower()
                if self._is_general_register(dst_candidate):
                    dst_reg = dst_candidate
            # REG dead after B, OR REG == DST (load overwrites).
            if dst_reg != r1:
                if not self._reg_dead_after(
                    out, i + 2, r1, treat_as_scratch=True,
                ):
                    i += 1
                    continue
            # Build the new memory operand.
            combined_disp = disp_a + disp_b
            if combined_disp == 0:
                new_mem_inner = f"{label} + {r1}*{scale}"
            elif combined_disp > 0:
                new_mem_inner = (
                    f"{label} + {r1}*{scale} + {combined_disp}"
                )
            else:
                new_mem_inner = (
                    f"{label} + {r1}*{scale} - {-combined_disp}"
                )
            new_mem_text = f"{size_kw}[{new_mem_inner}]"
            # Reconstruct B's operands with the substituted memory.
            if bp is None:
                # 1-op.
                new_operands = new_mem_text
            else:
                if mem_operand_idx == 0:
                    new_operands = f"{new_mem_text}, {bp[1].strip()}"
                else:
                    new_operands = f"{bp[0].strip()}, {new_mem_text}"
            indent = self._extract_indent(b.raw)
            opname = b.op
            spacer = " " * max(8 - len(opname), 1)
            new_raw = f"{indent}{opname}{spacer}{new_operands}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op=opname,
                operands=new_operands,
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_sib_label_load_collapse"] = (
                self.stats.get(
                    "lea_sib_label_load_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_lea_self_sib_into_load(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the codegen's struct-array element-access pattern.

        Pattern (2 instrs, possibly with intermediates between):
            lea     REG, [REG + IDX*SCALE]            ; A
            ... independent intermediate ops ...      ; (don't touch REG/IDX)
            mov/movsx/movzx DST, [REG (+ DISP2)?]      ; B

        (LEA's source can also include a +/- DISP1.) Rewrite to:
            ... intermediate ...
            mov/movsx/movzx DST, [REG + IDX*SCALE + (DISP1+DISP2)]

        Drops A. Saves 3 bytes per match. Common in
        ``arr[i].member`` for struct arrays whose sizeof doesn't fit
        x86 SIB scale (sizeof not in {1, 2, 4, 8} — codegen emits
        ``mul``-then-add) AND for sizeof-fitting struct arrays where
        ``arr[i]`` is followed by a non-zero member offset.

        Conditions:
        - A: ``lea REG, [REG + IDX*SCALE (+ DISP1)?]`` — LEA's
          destination is also referenced inside its source (the
          self-modifying form). IDX must be a different reg from
          REG. Scale ∈ {1, 2, 4, 8}.
        - Up to 8 intermediate instructions, none of which write or
          read REG, and none of which write IDX. (IDX may be read.)
        - B: ``mov/movsx/movzx DST, [REG (+ DISP2)?]`` — REG used
          as plain base, no SIB on B's address. (B with its own
          SIB would need two scaled indices in the result, which
          x86 modrm doesn't support — bail.)
        - DST == REG OR REG dead after B (treat_as_scratch=False —
          REG might be EAX as the return value, which we mustn't
          drop the address-of-something-just-loaded).
        """
        out = list(lines)
        i = 0
        max_intermediate = 8
        # Match `[REG + IDX*SCALE]` or `[REG + IDX*SCALE + DISP]` /
        # `[REG + IDX*SCALE - DISP]`.
        lea_sib_re = re.compile(
            r"^\s*\[\s*(\w+)\s*\+\s*(\w+)\s*\*\s*(\d+)"
            r"(?:\s*([+-])\s*(\d+))?\s*\]\s*$",
            re.IGNORECASE,
        )
        # B's address: `[REG]` or `[REG + DISP]` / `[REG - DISP]`,
        # with optional size prefix.
        load_re = re.compile(
            r"^\s*(?:(dword|word|byte|qword)\s+)?"
            r"\[\s*(\w+)(?:\s*([+-])\s*(\d+))?\s*\]\s*$",
            re.IGNORECASE,
        )
        while i < len(out):
            a = out[i]
            if not (a.kind == "instr" and a.op == "lea"):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            if not self._is_general_register(r1):
                i += 1
                continue
            m_lea = lea_sib_re.match(ap[1])
            if m_lea is None:
                i += 1
                continue
            base = m_lea.group(1).lower()
            idx = m_lea.group(2).lower()
            scale = int(m_lea.group(3))
            disp1 = 0
            if m_lea.group(4) is not None:
                disp1 = int(m_lea.group(5))
                if m_lea.group(4) == "-":
                    disp1 = -disp1
            # LEA must be self-modifying: dst == base.
            if base != r1:
                i += 1
                continue
            # IDX must be a different reg, and scale must fit.
            if idx == r1 or scale not in (1, 2, 4, 8):
                i += 1
                continue
            if not self._is_general_register(idx):
                i += 1
                continue
            # Walk forward looking for a load using r1 as base.
            j = i + 1
            steps = 0
            while j < len(out) and steps < max_intermediate:
                ln = out[j]
                if ln.kind in ("blank", "comment"):
                    j += 1
                    continue
                if ln.kind != "instr":
                    break
                # Intermediate must not touch r1.
                if self._references_reg_family(ln.operands, r1):
                    if (
                        ln.op in ("mov", "movsx", "movzx")
                        and self._uses_reg_as_base(ln, r1)
                    ):
                        break
                    break
                # Intermediate must not WRITE idx (reads OK).
                if self._instr_writes_reg(ln, idx):
                    break
                steps += 1
                j += 1
            else:
                i += 1
                continue
            if j >= len(out):
                i += 1
                continue
            b = out[j]
            if not (
                b.kind == "instr"
                and b.op in ("mov", "movsx", "movzx")
            ):
                i += 1
                continue
            bp = _operands_split(b.operands)
            if bp is None:
                i += 1
                continue
            r2 = bp[0].strip().lower()
            if not self._is_general_register(r2):
                i += 1
                continue
            m_b = load_re.match(bp[1])
            if m_b is None:
                # Maybe B has SIB; bail (would need 2 scaled idx).
                i += 1
                continue
            size_kw = m_b.group(1) or ""
            b_base = m_b.group(2).lower()
            if b_base != r1:
                # B's base is some other register; not our pattern.
                i += 1
                continue
            disp2 = 0
            if m_b.group(3) is not None:
                disp2 = int(m_b.group(4))
                if m_b.group(3) == "-":
                    disp2 = -disp2
            # Liveness: r1 dead after B, OR r1 == r2 (overwritten).
            if r1 != r2 and not self._reg_dead_after(out, j + 1, r1):
                i += 1
                continue
            # Build collapsed B: `[r1 + idx*scale + (disp1+disp2)]`.
            combined_disp = disp1 + disp2
            disp_text = ""
            if combined_disp > 0:
                disp_text = f" + {combined_disp}"
            elif combined_disp < 0:
                disp_text = f" - {-combined_disp}"
            new_addr = f"[{r1} + {idx}*{scale}{disp_text}]"
            if size_kw:
                new_addr = f"{size_kw} {new_addr}"
            indent = self._extract_indent(b.raw)
            new_raw = (
                f"{indent}{b.op:<7s} {r2}, {new_addr}"
            )
            new_b = Line(
                raw=new_raw, kind="instr",
                op=b.op, operands=f"{r2}, {new_addr}",
            )
            out = out[:i] + out[i + 1:j] + [new_b] + out[j + 1:]
            self.stats["lea_self_sib_into_load"] = (
                self.stats.get("lea_self_sib_into_load", 0) + 1
            )
            continue
        return out

    def _pass_lea_offset_fold(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``lea REG, [ebp ± N]; add REG, M`` (or ``sub REG,
        M``) into ``lea REG, [ebp ± (N ± M)]``.

        Saves 3 bytes per match (drops the add/sub). LEA doesn't set
        flags, while add/sub does — so the rewrite is only safe when
        the add's flags are dead.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``lea REG, [ebp ± N]`` (stack-relative).
        - Line B: ``add REG, M`` or ``sub REG, M`` (numeric M).
        - Flags after Line B must be safe (dead before any reader).
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op in ("add", "sub")
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            lea_src = ap[1].strip()
            b_dst = bp[0].strip().lower()
            if not self._is_general_register(r1) or b_dst != r1:
                i += 1
                continue
            try:
                imm = int(bp[1].strip())
            except ValueError:
                i += 1
                continue
            # LEA src must be `[ebp ± N]`.
            m = re.match(
                r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                lea_src,
                re.IGNORECASE,
            )
            if m is None:
                i += 1
                continue
            n_disp = int(m.group(2))
            if m.group(1) == "-":
                n_disp = -n_disp
            # Apply the add/sub.
            if b.op == "add":
                combined = n_disp + imm
            else:
                combined = n_disp - imm
            # Flags must be dead after line B (LEA doesn't set flags
            # but add/sub does; if flags were used by a downstream
            # Jcc, dropping the add changes behavior).
            if not self._flags_safe_after(out, i + 2):
                i += 1
                continue
            sign = "+" if combined >= 0 else "-"
            new_src = f"[ebp {sign} {abs(combined)}]"
            indent = self._extract_indent(a.raw)
            new_raw = f"{indent}lea     {r1}, {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="lea",
                operands=f"{r1}, {new_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_offset_fold"] = (
                self.stats.get("lea_offset_fold", 0) + 1
            )
            continue
        return out

    def _pass_lea_forward_to_reg(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``lea REG1, BASE; mov REG2, REG1`` into
        ``lea REG2, BASE`` when REG1 is dead after.

        Saves 2 bytes per match (drops the register copy). Common
        after the codegen emits an address-into-EAX then transfers
        to a different register for further use.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``lea REG1, BASE``.
        - Line B: ``mov REG2, REG1`` (register copy).
        - REG1 != REG2.
        - REG1 dead after line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            base = ap[1].strip()
            b_dst = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(b_dst)
                or b_src != r1
                or b_dst == r1
            ):
                i += 1
                continue
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            indent = self._extract_indent(a.raw)
            new_raw = f"{indent}lea     {b_dst}, {base}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="lea",
                operands=f"{b_dst}, {base}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_forward_to_reg"] = (
                self.stats.get("lea_forward_to_reg", 0) + 1
            )
            continue
        return out

    def _pass_lea_store_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``lea REG, [ebp ± N]; mov <size> [REG + M], SRC``
        (or ``[REG]``) into ``mov <size> [ebp ± (N+M)], SRC``.

        Saves 3 bytes per match. Common in local struct member
        assignment where the codegen emits an explicit `lea` to
        compute the struct base, then `mov [base + offset], value`
        for each member.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``lea REG, [ebp ± N]`` (stack-relative base).
        - Line B: ``mov <size> [REG]`` or ``mov <size> [REG + M]``
          with SRC.
        - REG dead after line B.
        - SRC must not reference REG.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            lea_src = ap[1].strip()
            b_dst = bp[0].strip()
            b_src = bp[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            # LEA src must be `[ebp ± N]`.
            m_lea = re.match(
                r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                lea_src,
                re.IGNORECASE,
            )
            if m_lea is None:
                i += 1
                continue
            n_disp = int(m_lea.group(2))
            if m_lea.group(1) == "-":
                n_disp = -n_disp
            # Mov dest must be `<size> [REG]` or `<size> [REG ± M]`.
            m_dst_plain = re.match(
                r"^(dword|word|byte|qword)\s+\[\s*([a-zA-Z]+)\s*\]$",
                b_dst,
                re.IGNORECASE,
            )
            m_dst_disp = re.match(
                r"^(dword|word|byte|qword)\s+\[\s*([a-zA-Z]+)"
                r"\s*([+-])\s*(\d+)\s*\]$",
                b_dst,
                re.IGNORECASE,
            )
            if m_dst_plain:
                size_kw = m_dst_plain.group(1).lower()
                base_reg = m_dst_plain.group(2).lower()
                m_disp_val = 0
            elif m_dst_disp:
                size_kw = m_dst_disp.group(1).lower()
                base_reg = m_dst_disp.group(2).lower()
                m_disp_val = int(m_dst_disp.group(4))
                if m_dst_disp.group(3) == "-":
                    m_disp_val = -m_disp_val
            else:
                i += 1
                continue
            if base_reg != r1:
                i += 1
                continue
            # SRC must not reference REG.
            if self._references_reg_family(b_src, r1):
                i += 1
                continue
            # REG dead after line B.
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            combined = n_disp + m_disp_val
            sign = "+" if combined >= 0 else "-"
            new_dst = f"{size_kw} [ebp {sign} {abs(combined)}]"
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}mov     {new_dst}, {b_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{new_dst}, {b_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_store_collapse"] = (
                self.stats.get("lea_store_collapse", 0) + 1
            )
            continue
        return out

    def _pass_dead_stack_store(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop ``mov <size> [ebp ± N], V1`` when a later
        ``mov <size> [ebp ± N], V2`` overwrites it before any read.

        Saves up to 7 bytes per match (drops a dword-imm store).

        Common after struct/array init followed by member assignment:
        ``struct point p = {1, 2, 3}; p.x = 100;`` emits two stores
        to [ebp - 12] back-to-back; the first is dead.

        Conservative scan:
        - Read of [ebp ± N] in any instruction's operands → store is
          alive, bail.
        - Call → bail (callee MIGHT have access to &x via prior
          address-take + global; can't tell without escape analysis).
        - LEA producing an address that might alias [ebp ± N] → bail
          (the lea'd reg could be used for indirect access later).
          Conservative: any LEA that references [ebp ± N] bails.
          Other LEAs (label addresses) don't alias stack slots.
        - Conditional or unconditional jump → bail (control-flow
          boundary).
        - Label → bail (predecessor may differ).
        - Memory write through a register pointer → bail (might
          alias).
        - Reaching another `mov <size> [ebp ± N], V2` (same offset)
          → fire: drop the first store.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.kind != "instr":
                out.append(line)
                i += 1
                continue
            n_disp = self._stack_store_offset(line)
            if n_disp is None:
                out.append(line)
                i += 1
                continue
            # Found a stack-imm-or-reg store. Scan forward.
            if self._dead_stack_store_check(lines, i, n_disp):
                self.stats["dead_stack_store"] = (
                    self.stats.get("dead_stack_store", 0) + 1
                )
                i += 1  # Drop the line.
                continue
            out.append(line)
            i += 1
        return out

    def _pass_dead_unused_slot_stores(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop stores to ``[ebp ± N]`` slots that are never read or
        address-taken anywhere in the function.

        Per-function analysis: identify each slot N where:
        - No read reference (`mov reg, [ebp - N]`,
          `cmp reg, [ebp - N]`, `push dword [ebp - N]`, etc.).
        - No `lea reg, [ebp - N]` (address-taken would let reads
          happen indirectly).
        - No RMW (`add [ebp - N], X`, `inc dword [ebp - N]`, etc.).

        Then drop all `mov [ebp ± N], SRC` (pure overwrite stores)
        to that slot.

        Common shape: a function with local `int d = ...; if (cond)
        d = -d; return d;` where the codegen stores `d` to a frame
        slot for "completeness" but never reads it back (the value
        flows through EAX). The codegen pass doesn't currently do
        this dead-store elimination, so peephole picks it up.

        Saves 3+ bytes per dropped store.
        """
        # Build per-function reference set.
        func_starts: list[int] = []
        for k, ln in enumerate(lines):
            if ln.kind == "label" and not ln.label.startswith("."):
                func_starts.append(k)
        if not func_starts:
            return list(lines)
        func_starts.append(len(lines))

        out = list(lines)
        # Collect indices to drop.
        drop_indices: set[int] = set()
        for fi in range(len(func_starts) - 1):
            fstart = func_starts[fi]
            fend = func_starts[fi + 1]
            # Heuristic: only analyze functions with a real prologue
            # (`enter` or `push ebp; mov ebp, esp`). Functions
            # without a frame can't have stack-relative slots in
            # the conventional sense; synthetic test asm fragments
            # without a prologue are skipped.
            has_prologue = False
            for k in range(fstart, min(fstart + 5, fend)):
                ln = lines[k]
                if ln.kind != "instr":
                    continue
                if ln.op == "enter":
                    has_prologue = True
                    break
                if (
                    ln.op == "push"
                    and ln.operands.strip().lower() == "ebp"
                ):
                    has_prologue = True
                    break
            if not has_prologue:
                continue
            # If the function has any `lea reg, [ebp ± N]` or
            # `mov reg, ebp`, OR any SIB-form `[ebp + idx*K (+ N)]`
            # access, the stack frame is potentially accessible
            # through register-based indirect access (or through
            # a runtime-computed index that we can't analyze
            # statically). Bail on the whole function.
            address_taken = False
            for k in range(fstart, fend):
                ln = lines[k]
                if ln.kind != "instr":
                    continue
                if ln.op == "lea":
                    if "ebp" in ln.operands.lower():
                        address_taken = True
                        break
                if ln.op == "mov":
                    parts = _operands_split(ln.operands)
                    if parts is not None:
                        src = parts[1].strip().lower()
                        if src == "ebp":
                            address_taken = True
                            break
                # Check for SIB-form `[ebp + idx*K (+ N)]` or
                # `[ebp + idx (+ N)]` access. The literal form
                # `[ebp + N]` and `[ebp - N]` are SAFE (we
                # analyze those by offset). Anything else with
                # `ebp` inside `[...]` means indirect access
                # through a runtime-variable index.
                ops_text = ln.operands or ""
                # Find all `[...]` brackets containing `ebp`.
                for m in re.finditer(
                    r"\[([^\]]*ebp[^\]]*)\]",
                    ops_text, re.IGNORECASE,
                ):
                    inner = m.group(1).strip().lower()
                    # Safe form: `ebp + N` or `ebp - N` or
                    # `ebp` (no offset; would mean disp=0).
                    safe = re.fullmatch(
                        r"\s*ebp\s*([+-]\s*\d+)?\s*",
                        inner,
                    )
                    if safe is None:
                        address_taken = True
                        break
                if address_taken:
                    break
            if address_taken:
                continue
            # Scan all lines in [fstart, fend).
            # Track per-slot state: live (had read or address-take)
            # vs only-stored.
            slot_has_read: set[int] = set()
            slot_stores: dict[int, list[int]] = {}
            slot_has_other: set[int] = set()  # has RMW or ambiguous use

            for k in range(fstart, fend):
                ln = lines[k]
                if ln.kind != "instr":
                    continue
                # Check operands for [ebp ± N] references.
                ops_text = ln.operands
                # Find all [ebp ± N] references.
                slot_refs = re.findall(
                    r"\[\s*ebp\s*([+-])\s*(\d+)\s*\]",
                    ops_text,
                    re.IGNORECASE,
                )
                referenced_slots = set()
                for sign, num in slot_refs:
                    n = int(num)
                    if sign == "-":
                        n = -n
                    referenced_slots.add(n)
                # Check if this is a pure-store to one of these.
                store_offset = self._stack_store_offset(ln)
                # `lea` reads its memory operand as an address —
                # treat as address-taken (mark slot as having
                # other use).
                if ln.op == "lea":
                    for slot in referenced_slots:
                        slot_has_other.add(slot)
                    continue
                # If the store_offset is set, this is a pure store.
                # Other slot references in this instruction are
                # READS (e.g., `mov [ebp - 4], [ebp - 8]` is invalid
                # in x86 anyway, so this case shouldn't arise).
                if store_offset is not None:
                    slot_stores.setdefault(
                        store_offset, []
                    ).append(k)
                    # Other slots referenced (the source) are reads.
                    for slot in referenced_slots:
                        if slot != store_offset:
                            slot_has_read.add(slot)
                    continue
                # If this is an RMW (`add [ebp - N], src` /
                # `inc/dec dword [ebp - N]` / `xor [ebp - N], src` /
                # etc.), it's a read-and-write. Mark as other.
                # Detect by: dest is `[ebp - N]` and op is not `mov`
                # or `lea`.
                parts = _operands_split(ops_text)
                if parts is not None:
                    dest = parts[0].strip()
                    m = re.match(
                        r"^(?:dword|word|byte|qword)?\s*"
                        r"\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                        dest,
                        re.IGNORECASE,
                    )
                    if m is not None:
                        n = int(m.group(2))
                        if m.group(1) == "-":
                            n = -n
                        # RMW or other dest write.
                        slot_has_other.add(n)
                        # Other operands (the source) are reads.
                        for slot in referenced_slots:
                            if slot != n:
                                slot_has_read.add(slot)
                        continue
                # 1-op forms like `inc dword [ebp - N]`,
                # `dec dword [ebp - N]`, `not dword [ebp - N]`,
                # `neg dword [ebp - N]`, `pop dword [ebp - N]`.
                if ln.op in {
                    "inc", "dec", "neg", "not", "pop", "push",
                }:
                    op_text = ops_text.strip()
                    m1 = re.match(
                        r"^(?:dword|word|byte|qword)?\s*"
                        r"\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                        op_text,
                        re.IGNORECASE,
                    )
                    if m1 is not None:
                        n = int(m1.group(2))
                        if m1.group(1) == "-":
                            n = -n
                        # inc/dec/neg/not = RMW; pop = pure write
                        # (overwrites slot from stack); push = read.
                        if ln.op == "push":
                            slot_has_read.add(n)
                        elif ln.op == "pop":
                            # pop writes the slot. Treat as pure
                            # store (we could collect it but for
                            # simplicity treat as other use to be
                            # safe — pop has stack side effects).
                            slot_has_other.add(n)
                        else:
                            slot_has_other.add(n)
                        continue
                # Otherwise: any other instruction with a slot
                # reference reads it.
                for slot in referenced_slots:
                    slot_has_read.add(slot)

            # For each slot with stores and no reads/other uses,
            # drop the stores.
            for slot_n, store_lines in slot_stores.items():
                if (
                    slot_n in slot_has_read
                    or slot_n in slot_has_other
                ):
                    continue
                # All stores to this slot are dead.
                for k in store_lines:
                    drop_indices.add(k)

        if not drop_indices:
            return out
        new_out = []
        for k, ln in enumerate(out):
            if k in drop_indices:
                self.stats["dead_unused_slot_stores"] = (
                    self.stats.get(
                        "dead_unused_slot_stores", 0
                    ) + 1
                )
                continue
            new_out.append(ln)
        return new_out

    def _pass_slot_constant_propagation(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace `mov reg, [ebp ± N]` with `mov reg, IMM` when the
        slot is a "singleton-write IMM" — i.e., written exactly once
        in the function with `mov dword [ebp ± N], IMM_LITERAL` and
        never modified elsewhere (no RMW, no `lea` for address-taken
        access, no SIB-form ebp access).

        Per-function analysis. After this pass, downstream chain
        folding (slices 35/39/40) can fold subsequent imm-op chains
        on the loaded register.

        Conditions for marking a slot as constant:
        - Exactly one writer: `mov dword [ebp ± N], IMM_LITERAL`.
        - No `lea reg, [ebp ± N]` (would expose address for indirect
          writes).
        - No RMW operations on the slot (`inc/dec/and/or/xor/add/sub
          dword [ebp ± N], ...`).
        - No SIB-form `[ebp + reg*K (+ disp)]` accessing variable
          offsets.
        - Function must have a proper prologue (so we know it's a
          real function, not a synthetic snippet).

        Rewrite: for each `mov reg, [ebp ± N]` (no displacement)
        where the slot is constant, replace with `mov reg, IMM`.
        """
        out = list(lines)
        ranges = PeepholeOptimizer._function_ranges(out)
        for start, end in ranges:
            # Skip ranges without a real prologue.
            has_prologue = False
            for k in range(start, end):
                ln = out[k]
                if ln.kind == "instr" and (
                    ln.op == "enter"
                    or (ln.op == "push" and "ebp" in (ln.operands or "").lower())
                ):
                    has_prologue = True
                    break
            if not has_prologue:
                continue
            # Bail on SIB-form ebp access or lea of ebp-rel.
            sib_or_lea = False
            for k in range(start, end):
                ln = out[k]
                if ln.kind != "instr":
                    continue
                ops = ln.operands or ""
                if "[" in ops and "ebp" in ops.lower():
                    # Check for SIB-form (contains '*' inside brackets).
                    bracket_text = ops[
                        ops.index("[") + 1: ops.index("]")
                        if "]" in ops else len(ops)
                    ]
                    if "*" in bracket_text:
                        sib_or_lea = True
                        break
                if (ln.op == "lea"
                        and "ebp" in ops.lower()):
                    # Could be `lea reg, [ebp ± N]` or
                    # `lea reg, [ebp + reg*K]` etc. Bail.
                    sib_or_lea = True
                    break
            if sib_or_lea:
                continue
            # First-reference check: for each slot, find the first
            # line that references the slot (read OR write). If first
            # ref is a write of an IMM literal, record the IMM. If
            # first ref is a read or non-IMM write, the slot's initial
            # value is undefined or caller-set; we won't propagate.
            slot_first_is_imm_write: dict[int, int] = {}
            slot_seen: set[int] = set()
            for k in range(start, end):
                ln = out[k]
                if ln.kind != "instr":
                    continue
                ops = ln.operands or ""
                if "ebp" not in ops.lower():
                    continue
                for m in _EBP_OFFSET_RE.finditer(ops):
                    sign = m.group(1)
                    n = int(m.group(2))
                    offset = -n if sign == "-" else n
                    if offset >= 0:
                        continue  # parameter, skip
                    if offset in slot_seen:
                        continue
                    slot_seen.add(offset)
                    # Record IMM write only when THIS line writes the
                    # slot from an IMM literal.
                    if ln.op == "mov":
                        parts = _operands_split(ln.operands)
                        if parts is not None:
                            dst, src = parts
                            dst_t = dst.strip()
                            for prefix in ("dword ", "word ", "byte ",
                                           "qword "):
                                if dst_t.lower().startswith(prefix):
                                    dst_t = dst_t[len(prefix):].lstrip()
                                    break
                            dst_off = self._ebp_offset(dst_t)
                            if dst_off == offset:
                                imm_val = self._parse_imm_value(
                                    src.strip()
                                )
                                if imm_val is not None:
                                    slot_first_is_imm_write[offset] = imm_val
            # Collect per-slot writers (with entry-block flag) and RMW.
            slot_writers: dict[int, list[tuple[int, int, bool]]] = {}
            slot_rmw: set[int] = set()
            for k in range(start, end):
                ln = out[k]
                if ln.kind != "instr":
                    continue
                op = ln.op
                # Check stores to [ebp ± N]: `mov <size?> [ebp ± N], src`.
                if op == "mov":
                    parts = _operands_split(ln.operands)
                    if parts is not None:
                        dst, src = parts
                        dst_t = dst.strip()
                        # Strip optional size prefix.
                        for prefix in ("dword ", "word ", "byte ",
                                       "qword "):
                            if dst_t.lower().startswith(prefix):
                                dst_t = dst_t[len(prefix):].lstrip()
                                break
                        offset = self._ebp_offset(dst_t)
                        if offset is not None:
                            # Skip parameters: positive offsets are
                            # caller-pushed values; pre-write reads see
                            # the caller's value, not our IMM.
                            if offset > 0:
                                slot_rmw.add(offset)
                                continue
                            # Only IMM-literal source is a "constant
                            # store"; reg/mem source = non-constant.
                            imm_val = self._parse_imm_value(src.strip())
                            if imm_val is None:
                                slot_rmw.add(offset)
                            else:
                                slot_writers.setdefault(
                                    offset, []
                                ).append((k, imm_val))
                # RMW ops with [ebp ± N] dest.
                elif op in ("add", "sub", "and", "or", "xor",
                            "inc", "dec", "shl", "shr", "sar",
                            "rol", "ror", "rcl", "rcr",
                            "not", "neg"):
                    parts = _operands_split(ln.operands)
                    dst_t = None
                    if parts is not None:
                        dst_t = parts[0].strip()
                    elif ln.operands:
                        dst_t = ln.operands.strip()
                    if dst_t is not None:
                        for prefix in ("dword ", "word ", "byte ",
                                       "qword "):
                            if dst_t.lower().startswith(prefix):
                                dst_t = dst_t[len(prefix):].lstrip()
                                break
                        offset = self._ebp_offset(dst_t)
                        if offset is not None:
                            slot_rmw.add(offset)
            # Determine constant slots.
            #   - All writers write the same IMM.
            #   - First reference of the slot is a write of that IMM
            #     (so any read sees the writer's value, never the
            #     pre-write garbage / caller value).
            constant_slots: dict[int, int] = {}
            for offset, writes in slot_writers.items():
                if offset in slot_rmw:
                    continue
                imms = {w[1] for w in writes}
                if len(imms) != 1:
                    continue
                # First-reference of this slot must be a write with
                # the same IMM as all writers.
                first_imm = slot_first_is_imm_write.get(offset)
                if first_imm is None:
                    continue
                if first_imm != writes[0][1]:
                    continue
                constant_slots[offset] = writes[0][1]
            if not constant_slots:
                continue
            # Replace `mov reg, [ebp ± N]` (no disp, no SIB) with
            # `mov reg, IMM` for constant slots.
            for k in range(start, end):
                ln = out[k]
                if ln.kind != "instr" or ln.op != "mov":
                    continue
                parts = _operands_split(ln.operands)
                if parts is None:
                    continue
                dst, src = parts
                dst_low = dst.strip().lower()
                if dst_low not in PeepholeOptimizer._GP32:
                    continue
                src_t = src.strip()
                # Source must be a plain `[ebp ± N]` (no size prefix
                # since reg-load infers from dest).
                offset = self._ebp_offset(src_t)
                if offset is None:
                    continue
                if offset not in constant_slots:
                    continue
                imm = constant_slots[offset]
                indent = self._extract_indent(ln.raw)
                new_raw = f"{indent}mov     {dst_low}, {imm}"
                new_line = Line(
                    raw=new_raw, kind="instr",
                    op="mov", operands=f"{dst_low}, {imm}",
                )
                out[k] = new_line
                self.stats["slot_constant_propagation"] = (
                    self.stats.get(
                        "slot_constant_propagation", 0
                    ) + 1
                )
        return out

    @staticmethod
    def _stack_store_offset(line: Line) -> int | None:
        """If `line` is `mov [ebp ± N], SRC`, return N. The size
        prefix (`dword`/`word`/`byte`/`qword`) is optional — when
        omitted, NASM infers the size from the source register
        (eax/ax/al/ah). Without size and with non-register source,
        we can't infer; return None.

        Otherwise None.
        """
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest = parts[0].strip()
        # Form 1: with explicit size prefix.
        m = re.match(
            r"^(dword|word|byte|qword)\s+\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
            dest,
            re.IGNORECASE,
        )
        if m is not None:
            n = int(m.group(3))
            if m.group(2) == "-":
                n = -n
            return n
        # Form 2: no size prefix — `mov [ebp ± N], SRC` where SRC is
        # a register or a label (label means imm32 → dword, but NASM
        # actually requires size for that case; in practice the
        # codegen only emits sizeless form for register sources).
        m = re.match(
            r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
            dest,
            re.IGNORECASE,
        )
        if m is not None:
            # Check that the source is a register (size inferred).
            src = parts[1].strip()
            if src.lower() in PeepholeOptimizer._GP32:
                # 32-bit reg → dword store.
                n = int(m.group(2))
                if m.group(1) == "-":
                    n = -n
                return n
            # Sub-register → word/byte store; partial overwrites
            # don't fully kill the prior dword. Conservative: skip.
        return None

    def _dead_stack_store_check(
        self, lines: list[Line], start_idx: int, target_offset: int,
    ) -> bool:
        """Return True if the store at lines[start_idx] is dead —
        an overwrite to the same offset is reached before any read."""
        target_re = re.compile(
            rf"\[\s*ebp\s*[+-]\s*\d+\s*\]",
            re.IGNORECASE,
        )
        # Build the exact text to look for in operands when scanning
        # for reads. Both `[ebp + N]` and `[ebp - N]` (with original
        # sign) are valid — but normalized through the offset.
        for j in range(start_idx + 1, min(len(lines), start_idx + 30)):
            ln = lines[j]
            if ln.kind == "label":
                # Control-flow boundary.
                return False
            if ln.kind != "instr":
                continue
            op = ln.op
            ops = ln.operands
            # Calls bail.
            if op == "call":
                return False
            # Jumps bail (control flow).
            if op in {"jmp", "ret", "iret", "iretd", "retf",
                      "retn", "leave"} or op.startswith("j"):
                return False
            # Find any [ebp ± M] reference in this line.
            for m in target_re.finditer(ops):
                ref = m.group(0)
                rm = re.match(
                    r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                    ref,
                    re.IGNORECASE,
                )
                if not rm:
                    continue
                ref_off = int(rm.group(2))
                if rm.group(1) == "-":
                    ref_off = -ref_off
                if ref_off != target_offset:
                    continue
                # Same offset — does the line WRITE or READ it?
                if op == "mov":
                    parts = _operands_split(ops)
                    if parts is None:
                        return False
                    dest = parts[0].strip()
                    src = parts[1].strip()
                    # Dest matches our target offset → overwrite.
                    if (
                        ref in dest
                        and self._stack_store_offset(ln)
                        == target_offset
                    ):
                        return True
                    # Src refs target offset → read.
                    if ref in src:
                        return False
                    # Otherwise (other ops at this offset), conservative.
                    return False
                # Any other op (add/sub/etc.) reading or writing the
                # offset is conservative — bail.
                return False
            # Memory write through register (e.g. mov [eax], X) might
            # alias a stack slot if EAX was lea'd from one earlier.
            # Conservative: bail on any indirect mem write.
            if op == "mov":
                parts = _operands_split(ops)
                if parts is not None:
                    dest = parts[0].strip()
                    if "[" in dest and "ebp" not in dest.lower():
                        # Indirect memory write.
                        return False
            # LEA referencing target offset → bail.
            if op == "lea" and target_re.search(ops):
                for m in target_re.finditer(ops):
                    ref = m.group(0)
                    rm = re.match(
                        r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                        ref,
                        re.IGNORECASE,
                    )
                    if rm:
                        ref_off = int(rm.group(2))
                        if rm.group(1) == "-":
                            ref_off = -ref_off
                        if ref_off == target_offset:
                            return False
            # Otherwise, continue scanning.
        return False

    def _pass_reg_copy_addr_forward(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REGB, REGA; <instr using [REGB...]>``
        into ``<instr using [REGA...]>`` when REGB is dead after.

        Pattern (2 consecutive instr lines):
            mov     REGB, REGA       ; register copy
            <op>    ..., [REGB...]   ; address uses REGB as base

        Rewrite to:
            <op>    ..., [REGA...]   ; substituted

        Saves 2 bytes per match (drops the 2-byte register copy).

        Common after `*p = V` lowering: codegen emits
            lea     eax, addr
            mov     [ebp - N], eax       ; store p
            mov     ecx, eax             ; setup for deref
            mov     dword [ecx], V       ; *p = V
        The `mov ecx, eax; mov [ecx], V` pair collapses to
        `mov [eax], V`.

        Conditions:
        - Line A: ``mov REGB, REGA`` where REGA, REGB are GP32 regs,
          REGA != REGB.
        - Line B: an instruction whose operand contains ``[REGB...]``
          (REGB used as base register in memory addressing).
        - REGB dead after line B (we're dropping its only writer).
        - Line B's operand can be safely rewritten by replacing
          REGB with REGA in the [...] expression.
        - REGA isn't being modified in line B (would change its
          value before the addressing is resolved). Conservative:
          REGA must not be a destination of line B.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    a_dst, a_src = ap
                    a_dst_low = a_dst.strip().lower()
                    a_src_low = a_src.strip().lower()
                    if (
                        a_dst_low in PeepholeOptimizer._GP32
                        and a_src_low in PeepholeOptimizer._GP32
                        and a_dst_low != a_src_low
                    ):
                        b = lines[i + 1]
                        if (
                            b.kind == "instr"
                            and self._instr_uses_reg_in_addr(
                                b, a_dst_low,
                            )
                            and not self._instr_writes_reg(
                                b, a_src_low,
                            )
                            and self._reg_dead_after(
                                lines, i + 2, a_dst_low,
                            )
                        ):
                            new_b = self._substitute_addr_reg(
                                b, a_dst_low, a_src_low,
                            )
                            if new_b is not None:
                                out.append(new_b)
                                self.stats[
                                    "reg_copy_addr_forward"
                                ] = (
                                    self.stats.get(
                                        "reg_copy_addr_forward", 0,
                                    ) + 1
                                )
                                i += 2
                                continue
            out.append(line)
            i += 1
        return out

    def _instr_uses_reg_in_addr(self, line: Line, reg: str) -> bool:
        """Does any operand of `line` contain `[reg...]` (reg used
        as memory base or index)?"""
        if line.kind != "instr":
            return False
        ops = line.operands
        # Find all `[...]` substrings and check if any reference reg.
        for m in re.finditer(r"\[[^\]]*\]", ops):
            if _references_register(m.group(0), reg):
                return True
        return False

    @staticmethod
    def _instr_writes_reg(line: Line, reg: str) -> bool:
        """Does this instruction write to `reg` (as destination)?"""
        if line.kind != "instr":
            return False
        op = line.op
        # mov, lea, movsx, movzx — first operand is destination.
        if op in {"mov", "lea", "movsx", "movzx", "add", "sub",
                  "and", "or", "xor", "imul", "shl", "shr",
                  "sar", "rol", "ror", "rcl", "rcr", "adc",
                  "sbb", "neg", "not", "inc", "dec"}:
            parts = _operands_split(line.operands)
            if parts is None:
                # Single-operand op (neg, not, inc, dec).
                operand = line.operands.strip().lower()
                return operand == reg.lower()
            return parts[0].strip().lower() == reg.lower()
        # pop reg writes to reg.
        if op == "pop":
            return line.operands.strip().lower() == reg.lower()
        # call may write eax (cdecl return).
        if op == "call" and reg.lower() in {"eax", "ecx", "edx"}:
            return True
        return False

    def _substitute_addr_reg(
        self, line: Line, old_reg: str, new_reg: str,
    ) -> Line | None:
        """Substitute `old_reg` with `new_reg` in any `[...]`
        addressing in `line`'s operands. Returns the new Line, or
        None if substitution would change semantics in an unexpected
        way.
        """
        # Build a regex that finds [...] groups and substitutes
        # `old_reg` with `new_reg` inside, using word boundaries.
        def repl(match: re.Match) -> str:
            inner = match.group(0)
            # Substitute old_reg with new_reg inside the brackets,
            # case-insensitively, using word boundaries.
            return re.sub(
                rf"\b{old_reg}\b",
                new_reg,
                inner,
                flags=re.IGNORECASE,
            )

        new_operands = re.sub(
            r"\[[^\]]*\]",
            repl,
            line.operands,
        )
        if new_operands == line.operands:
            # Substitution didn't change anything (shouldn't happen
            # if _instr_uses_reg_in_addr returned True).
            return None
        # Reconstruct the line preserving leading whitespace.
        indent = self._extract_indent(line.raw)
        spacer = " " * max(1, 8 - len(line.op))
        new_raw = f"{indent}{line.op}{spacer}{new_operands}"
        return Line(
            raw=new_raw, kind="instr",
            op=line.op, operands=new_operands,
        )

    def _pass_value_forward_to_reg(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG1, SRC; mov REG2, REG1`` into
        ``mov REG2, SRC`` when REG1 is dead after.

        Saves 2 bytes per match (drops the `mov reg2, reg1` transfer;
        the new `mov reg2, SRC` is the same length as the original
        `mov reg1, SRC`).

        Common after label_offset_fold leaves
        ``mov eax, _label + N; mov ebx, eax`` — the first instruction's
        value is being forwarded through EAX to EBX, but EAX is dead
        after.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov REG1, SRC`` where SRC is not a memory operand
          (immediates, labels, label-arithmetic, register sources).
        - Line B: ``mov REG2, REG1`` (register copy).
        - REG1 != REG2 (else line B is a self-mov, handled by
          self_mov_elimination).
        - REG1 dead after line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            src = ap[1].strip()
            b_dst = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(b_dst)
            ):
                i += 1
                continue
            # Line B must be a register copy from r1.
            if b_src != r1 or b_dst == r1:
                i += 1
                continue
            # SRC may be a memory operand. The rewrite `mov reg2, [m]`
            # is the same width as the original `mov reg1, [m]`, so
            # we drop the 2-byte register transfer for free. Memory
            # source must not reference REG2 (would self-reference
            # after substitution).
            if "[" in src:
                if _references_register(src, b_dst):
                    i += 1
                    continue
            # SRC must not be the target register either (would be
            # forwarding `mov r2, r2` which is a self-mov).
            if src.lower() == b_dst:
                i += 1
                continue
            # REG1 dead after line B.
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            # Rewrite: replace lines A and B with `mov b_dst, src`.
            indent = self._extract_indent(a.raw)
            new_raw = f"{indent}mov     {b_dst}, {src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{b_dst}, {src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["value_forward_to_reg"] = (
                self.stats.get("value_forward_to_reg", 0) + 1
            )
            continue
        return out

    def _pass_dead_store_before_push(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop ``mov [m], REG`` when followed by ``push REG`` (or
        ``push dword [m]``) and [m] isn't read again before function
        exit.

        Common shape: codegen saves a call result to a frame slot
        immediately before pushing it as an argument:
        ``mov [ebp - 12], eax; push eax; ...; call _printf; ...; ret``
        — the slot's value is captured in the push, but the slot
        itself is never read again. The store is dead.

        Saves 3 bytes per match (the dropped ``mov [m], REG``).

        Conditions:
        - Line A: ``mov [m], REG`` where m is an ebp-offset and REG
          is a 32-bit GP reg.
        - Line B: ``push REG`` (same REG) OR ``push dword [m]``
          (same memory).
        - In the rest of the function (until next global label or
          ret), [m] is never read AND not address-taken.
        - No backward jump targeting before the store (loop body
          might re-enter and read [m]).

        This is a restricted form of dead-store-elimination at
        function exit. The pre-condition `push REG` distinguishes
        the save-then-push pattern from synthetic unit-test cases
        where a store is followed only by `ret`.
        """
        out = list(lines)
        # Find function boundaries.
        func_starts: list[int] = []
        for k, ln in enumerate(out):
            if ln.kind == "label" and not ln.label.startswith("."):
                func_starts.append(k)
        if not func_starts:
            return out
        func_starts.append(len(out))
        for fi in range(len(func_starts) - 1):
            fstart = func_starts[fi]
            fend = func_starts[fi + 1]
            # Stop at section directives.
            for k in range(fstart, fend):
                if out[k].kind == "directive" and out[k].raw.strip(
                ).lower().startswith("section"):
                    fend = k
                    break
            # Per-function label_pos for backward-jump check.
            label_pos: dict[str, int] = {}
            for k in range(fstart, fend):
                ln = out[k]
                if ln.kind == "label":
                    label_pos[ln.label] = k
            # Check function for any complex ebp use (lea, SIB).
            ebp_re_full = re.compile(
                r"\[\s*ebp\b[^\]]*\]", re.IGNORECASE
            )
            ebp_re_simple = re.compile(
                r"^\[\s*ebp\s*[+-]\s*\d+\s*\]$", re.IGNORECASE
            )
            any_complex_ebp = False
            for k in range(fstart, fend):
                ln = out[k]
                if ln.kind != "instr":
                    continue
                for m in ebp_re_full.finditer(ln.operands):
                    expr = m.group(0)
                    if not ebp_re_simple.match(expr):
                        any_complex_ebp = True
                        break
                if any_complex_ebp:
                    break
                if ln.op == "lea":
                    offsets = self._extract_ebp_offsets(
                        ln.operands
                    )
                    if offsets:
                        any_complex_ebp = True
                        break
            if any_complex_ebp:
                continue
            # For each store followed by a matching push, check if
            # the slot is dead.
            i = fstart
            while i < fend:
                ln = out[i]
                if ln.kind != "instr":
                    i += 1
                    continue
                n_disp = self._stack_store_offset(ln)
                if n_disp is None:
                    i += 1
                    continue
                # Get the register source from the store.
                parts = _operands_split(ln.operands)
                if parts is None:
                    i += 1
                    continue
                store_dest = parts[0].strip()
                store_src = parts[1].strip().lower()
                if store_src not in PeepholeOptimizer._GP32:
                    i += 1
                    continue
                # Find the next instruction.
                j = i + 1
                while j < fend:
                    nl = out[j]
                    if nl.kind in ("blank", "comment"):
                        j += 1
                        continue
                    break
                if j >= fend:
                    i += 1
                    continue
                nl = out[j]
                # Must be `push REG` (same as store_src). The case
                # `push dword [m]` would actually READ [m] — not a
                # candidate for elimination.
                if nl.kind != "instr" or nl.op != "push":
                    i += 1
                    continue
                push_op = nl.operands.strip().lower()
                if push_op != store_src:
                    i += 1
                    continue
                # Now check that [n_disp] isn't read in the rest of
                # the function. The store and push are at i and j.
                # Scan from j+1 to fend.
                slot_alive = False
                for k in range(j + 1, fend):
                    bln = out[k]
                    if bln.kind != "instr":
                        continue
                    # Check for read of [ebp ± n_disp] in operands.
                    offsets = self._extract_ebp_offsets(
                        bln.operands
                    )
                    if n_disp in offsets:
                        slot_alive = True
                        break
                    # Check for backward jump landing at or before
                    # the store (loop body re-entry).
                    if bln.op == "jmp" or bln.op.startswith("j"):
                        target = _branch_target(bln)
                        if target is not None:
                            tpos = label_pos.get(target)
                            if (
                                tpos is not None
                                and tpos <= i
                                and tpos >= fstart
                            ):
                                slot_alive = True
                                break
                if slot_alive:
                    i += 1
                    continue
                # Drop the store.
                out[i] = None  # type: ignore
                self.stats["dead_store_before_push"] = (
                    self.stats.get("dead_store_before_push", 0) + 1
                )
                i += 1
        return [ln for ln in out if ln is not None]

    @staticmethod
    def _extract_ebp_offsets(text: str) -> list[int]:
        """Find all `[ebp +/- N]` literals in `text` and return their
        signed offsets (negative for [ebp - N], positive for
        [ebp + N]). Tolerates optional size prefix and whitespace."""
        result: list[int] = []
        for m in re.finditer(
            r"\[\s*ebp\s*([+-])\s*(\d+)\s*\]",
            text, re.IGNORECASE,
        ):
            n = int(m.group(2))
            if m.group(1) == "-":
                n = -n
            result.append(n)
        return result

    def _pass_redundant_mem_load_via_xfer(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop ``mov R2, [m]`` when an earlier ``mov [m], R1; mov
        R2, R1`` chain already established R2 = [m]'s value, and
        nothing in between modified R2 or [m].

        Saves 3 bytes per match (the dropped memory load).

        Common shape: codegen patterns like
            mov [ebp - 4], eax
            mov ecx, eax
            ...some chain that doesn't touch ECX or [ebp - 4]...
            mov ecx, [ebp - 4]   <-- redundant: ECX still equals [ebp - 4]

        The codegen emits the reload because it doesn't track
        register aliases across passes; this peephole catches the
        cross-register equivalence.

        Conditions:
        - Pattern start: line A `mov [m], R1` immediately followed
          by line B `mov R2, R1` (R1 != R2, m is ebp-offset).
        - Pattern end: line C `mov R2, [m]` (same m).
        - Between B and C: no write to R2 (or sub-regs), no write
          to [m].
        - Between A and B: must be back-to-back (already implied).

        Aliasing analysis: register-base derefs (`[reg]`,
        `[reg + N]`) are treated as non-aliasing with stack slot
        [m] PROVIDED [m]'s address isn't taken via `lea` anywhere
        in the function (precomputed as a per-function set).
        """
        out = list(lines)
        # Pre-compute per-function set of address-taken stack
        # slots. Any `lea reg, [ebp - N]` causes [ebp - N] to
        # potentially escape into a register, after which any
        # deref `[reg]` could alias [ebp - N].
        addr_taken_by_func: list[tuple[int, int, set[int]]] = []
        # (fstart, fend, addr_taken_set)
        # Find function boundaries.
        func_starts: list[int] = []
        for k, ln in enumerate(out):
            if ln.kind == "label" and not ln.label.startswith("."):
                func_starts.append(k)
        if not func_starts:
            return out
        func_starts.append(len(out))
        for fi in range(len(func_starts) - 1):
            fstart = func_starts[fi]
            fend = func_starts[fi + 1]
            for k in range(fstart, fend):
                if out[k].kind == "directive" and out[k].raw.strip(
                ).lower().startswith("section"):
                    fend = k
                    break
            addr_taken: set[int] = set()
            for k in range(fstart, fend):
                ln = out[k]
                if ln.kind == "instr" and ln.op == "lea":
                    for off in self._extract_ebp_offsets(
                        ln.operands
                    ):
                        addr_taken.add(off)
            addr_taken_by_func.append(
                (fstart, fend, addr_taken)
            )
        def find_func(idx: int) -> set[int]:
            for fs, fe, at in addr_taken_by_func:
                if fs <= idx < fe:
                    return at
            return set()
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            # Line A: mov [m], R1
            a_dst = ap[0].strip()
            a_src = ap[1].strip().lower()
            if not (a_dst.startswith("[") and a_dst.endswith("]")):
                i += 1
                continue
            if not self._is_general_register(a_src):
                i += 1
                continue
            # m must be an ebp-offset (for safe aliasing analysis).
            if not self._is_ebp_offset_mem(a_dst):
                i += 1
                continue
            r1 = a_src
            mem_addr = a_dst
            # Line B: mov R2, R1
            b_dst = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(b_dst)
                or b_src != r1
                or b_dst == r1
            ):
                i += 1
                continue
            r2 = b_dst
            # Scan forward for `mov R2, [m]`. Bail on any write to
            # R2 or to [m].
            found_redundant_idx = None
            scan_count = 0
            j = i + 2
            while j < len(out) and scan_count <= 16:
                ln = out[j]
                if ln.kind in ("blank", "comment"):
                    j += 1
                    continue
                if ln.kind != "instr":
                    break
                # Check if this is the candidate redundant load.
                if ln.op == "mov":
                    p = _operands_split(ln.operands)
                    if p is not None:
                        ld_dst = p[0].strip().lower()
                        ld_src = p[1].strip()
                        if (
                            ld_dst == r2
                            and ld_src == mem_addr
                        ):
                            found_redundant_idx = j
                            break
                # Bail on writes to R2 (any form: mov R2, X; pop R2;
                # arithmetic with R2 as dest; etc.).
                p = _operands_split(ln.operands)
                if p is not None:
                    dst_low = p[0].strip().lower()
                    if dst_low == r2:
                        break
                if ln.op == "pop":
                    if ln.operands.strip().lower() == r2:
                        break
                # Sub-register writes also clobber R2.
                sub_regs_r2 = self._sub_regs_for(r2)
                if p is not None and p[0].strip().lower() in sub_regs_r2:
                    break
                # Bail on writes to [m] (besides line A itself).
                if (
                    ln.op == "mov"
                    and p is not None
                    and "[" in p[0]
                ):
                    dest_str = p[0].strip()
                    # Strip optional size prefix.
                    dest_chk = dest_str
                    for prefix in (
                        "dword ", "word ", "byte ", "qword "
                    ):
                        if dest_chk.lower().startswith(prefix):
                            dest_chk = dest_chk[
                                len(prefix):
                            ].lstrip()
                            break
                    # If dest is also ebp-offset, do regular
                    # disjoint check.
                    if self._is_ebp_offset_mem(dest_chk):
                        if not self._mem_disjoint(
                            dest_str, mem_addr
                        ):
                            break
                    else:
                        # Register-base deref. Could alias [m]
                        # only if m's address has been taken
                        # via lea anywhere in the function.
                        m_offset = self._ebp_offset(mem_addr)
                        addr_taken = find_func(j)
                        if (
                            m_offset is not None
                            and m_offset in addr_taken
                        ):
                            break
                # Bail on calls (might clobber regs/memory through
                # globals).
                if ln.op == "call":
                    break
                # Bail on instructions that implicitly clobber R2.
                if (
                    ln.op in PeepholeOptimizer._IMPLICIT_REG_USERS
                    and self._references_reg_family(
                        ln.operands, r2
                    )
                ):
                    break
                # Conditional or unconditional control flow breaks
                # the scan (we don't track across CFG boundaries).
                if (
                    ln.op == "jmp"
                    or ln.op.startswith("j")
                    or ln.op in (
                        "ret", "iret", "retf", "retn", "leave",
                        "enter",
                    )
                ):
                    break
                scan_count += 1
                j += 1
            if found_redundant_idx is None:
                i += 1
                continue
            # Drop line at found_redundant_idx.
            del out[found_redundant_idx]
            self.stats["redundant_mem_load_via_xfer"] = (
                self.stats.get("redundant_mem_load_via_xfer", 0) + 1
            )
            # Don't advance i — re-check from current position.
        return out

    @staticmethod
    def _sub_regs_for(reg: str) -> set[str]:
        """Sub-register names for a 32-bit GP reg."""
        m = {
            "eax": {"al", "ah", "ax"},
            "ebx": {"bl", "bh", "bx"},
            "ecx": {"cl", "ch", "cx"},
            "edx": {"dl", "dh", "dx"},
            "esi": {"si"},
            "edi": {"di"},
            "ebp": {"bp"},
            "esp": {"sp"},
        }
        return m.get(reg, set())

    def _pass_dup_load_chain_to_copy(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse duplicate load chains. Pattern (4 adjacent instr
        lines):
            mov R1, [m]
            mov R1, [R1] OR mov R1, [R1 + N]
            mov R2, [m]
            mov R2, [R2] OR mov R2, [R2 + N]
        →
            mov R1, [m]
            mov R1, [R1] (or [R1 + N])
            mov R2, R1

        Saves 4 bytes per match (drops two mem-loads, adds one
        reg-mov).

        Common shape: `p->x * p->x`-style code where the same
        struct member is accessed twice through different
        registers. The codegen evaluates each operand independently
        without memoization.

        Conditions:
        - R1 != R2 (else self-mov, irrelevant).
        - Both must be 32-bit GP regs.
        - The same memory operand `[m]` must be used in both chains.
        - The deref offset (if any) must match between B and D.
        - Lines must be back-to-back (no intermediate instrs).
        - No write to `[m]` happens between (since they're
          back-to-back, none can).
        """
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            d = out[i + 3]
            # All four must be `mov` instructions.
            if not all(
                ln.kind == "instr" and ln.op == "mov"
                for ln in (a, b, c, d)
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            cp = _operands_split(c.operands)
            dp = _operands_split(d.operands)
            if any(p is None for p in (ap, bp, cp, dp)):
                i += 1
                continue
            # A: mov R1, [m]
            r1 = ap[0].strip().lower()
            a_src = ap[1].strip()
            if (
                not self._is_general_register(r1)
                or not (a_src.startswith("[") and a_src.endswith("]"))
            ):
                i += 1
                continue
            # B: mov R1, [R1] or [R1 + N]
            b_dst = bp[0].strip().lower()
            b_src = bp[1].strip()
            if b_dst != r1:
                i += 1
                continue
            # Match `[r1]` or `[r1 + N]` or `[r1 - N]`.
            b_match = re.match(
                r"^\[\s*" + re.escape(r1)
                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                b_src, re.IGNORECASE,
            )
            if b_match is None:
                i += 1
                continue
            offset_b = 0
            if b_match.group(1) is not None:
                offset_b = int(b_match.group(2))
                if b_match.group(1) == "-":
                    offset_b = -offset_b
            # C: mov R2, [m] (same as A's source)
            r2 = cp[0].strip().lower()
            c_src = cp[1].strip()
            if (
                not self._is_general_register(r2)
                or r1 == r2
                or c_src != a_src
            ):
                i += 1
                continue
            # D: mov R2, [R2] or [R2 + N] (same offset as B)
            d_dst = dp[0].strip().lower()
            d_src = dp[1].strip()
            if d_dst != r2:
                i += 1
                continue
            d_match = re.match(
                r"^\[\s*" + re.escape(r2)
                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                d_src, re.IGNORECASE,
            )
            if d_match is None:
                i += 1
                continue
            offset_d = 0
            if d_match.group(1) is not None:
                offset_d = int(d_match.group(2))
                if d_match.group(1) == "-":
                    offset_d = -offset_d
            indent = self._extract_indent(c.raw)
            if offset_b == offset_d:
                # Same offset: drop C and D, replace with mov R2, R1.
                # End state: R1 == R2 == *(base + N).
                new_raw = f"{indent}mov     {r2}, {r1}"
                new_line = Line(
                    raw=new_raw, kind="instr", op="mov",
                    operands=f"{r2}, {r1}",
                )
                out = out[:i + 2] + [new_line] + out[i + 4:]
                self.stats["dup_load_chain_to_copy"] = (
                    self.stats.get("dup_load_chain_to_copy", 0) + 1
                )
            else:
                # Different offsets: drop C only; insert `mov R2, R1`
                # BEFORE B (so R2 has base before R1 is clobbered by
                # B's deref). After this, B clobbers R1 to *(base+N1)
                # and D reads R2 (still = base) for *(base+N2).
                # Saves 1 byte: 3-byte ebp-rel `mov R2, [m]` →
                # 2-byte `mov R2, R1`.
                new_raw = f"{indent}mov     {r2}, {r1}"
                new_line = Line(
                    raw=new_raw, kind="instr", op="mov",
                    operands=f"{r2}, {r1}",
                )
                out = (
                    out[:i + 1]
                    + [new_line, b, d]
                    + out[i + 4:]
                )
                self.stats["dup_load_chain_to_copy_diff_off"] = (
                    self.stats.get(
                        "dup_load_chain_to_copy_diff_off", 0
                    ) + 1
                )
            # Don't advance — re-check from current pos.
        return out

    def _pass_copy_save_deref_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the codegen's "save pointer before deref" idiom:

            mov R1, [SRC]
            mov R2, R1
            mov R1, [R1] or [R1 +/- N1]
            mov R2, [R2 +/- N2]
        →
            mov R2, [SRC]
            mov R1, [R2] or [R2 +/- N1]
            mov R2, [R2 +/- N2]

        Saves 2 bytes per match (drops the `mov R2, R1` register
        copy).

        Common shape: `s->a + s->b`-style code where the codegen
        evaluates s->a (clobbering the base pointer in R1) and needs
        the original pointer for s->b (saved in R2).
        """
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            d = out[i + 3]
            if not all(
                ln.kind == "instr" and ln.op == "mov"
                for ln in (a, b, c, d)
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            cp = _operands_split(c.operands)
            dp = _operands_split(d.operands)
            if any(p is None for p in (ap, bp, cp, dp)):
                i += 1
                continue
            # A: mov R1, [SRC]
            r1 = ap[0].strip().lower()
            a_src = ap[1].strip()
            if (
                not self._is_general_register(r1)
                or not (a_src.startswith("[") and a_src.endswith("]"))
            ):
                i += 1
                continue
            # B: mov R2, R1
            r2 = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(r2)
                or r2 == r1
                or b_src != r1
            ):
                i += 1
                continue
            # C: mov R1, [R1] or [R1 +/- N1]
            c_dst = cp[0].strip().lower()
            c_src = cp[1].strip()
            if c_dst != r1:
                i += 1
                continue
            c_match = re.match(
                r"^\[\s*" + re.escape(r1)
                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                c_src, re.IGNORECASE,
            )
            if c_match is None:
                i += 1
                continue
            # D: mov R2, [R2 +/- N2]
            d_dst = dp[0].strip().lower()
            d_src = dp[1].strip()
            if d_dst != r2:
                i += 1
                continue
            d_match = re.match(
                r"^\[\s*" + re.escape(r2)
                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                d_src, re.IGNORECASE,
            )
            if d_match is None:
                i += 1
                continue
            indent_a = self._extract_indent(a.raw)
            indent_c = self._extract_indent(c.raw)
            new_a_raw = f"{indent_a}mov     {r2}, {a_src}"
            new_a = Line(
                raw=new_a_raw, kind="instr", op="mov",
                operands=f"{r2}, {a_src}",
            )
            if c_match.group(1) is None:
                new_c_addr = f"[{r2}]"
            else:
                new_c_addr = (
                    f"[{r2} {c_match.group(1)} {c_match.group(2)}]"
                )
            new_c_raw = f"{indent_c}mov     {r1}, {new_c_addr}"
            new_c = Line(
                raw=new_c_raw, kind="instr", op="mov",
                operands=f"{r1}, {new_c_addr}",
            )
            out = out[:i] + [new_a, new_c, d] + out[i + 4:]
            self.stats["copy_save_deref_collapse"] = (
                self.stats.get("copy_save_deref_collapse", 0) + 1
            )
        return out

    def _pass_dup_load_to_copy(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse adjacent same-memory loads. Pattern (2 lines):
            mov R1, [m]
            mov R2, [m]    ; same memory textually
        →
            mov R1, [m]
            mov R2, R1

        Saves bytes for any non-trivial memory operand:
        - ebp-rel `[ebp ± N]` (3-4 bytes): saves 1-2 bytes
        - label `[_glob]` (5+ bytes): saves 3+ bytes
        - SIB form `[_g + reg*4]` (7+ bytes): saves 5+ bytes

        Common after `lea_sib_label_load_collapse` or other passes
        produce two adjacent same-memory loads, e.g., `g[i] + g[i]`
        or `pts[i].x; ...; pts[i].x` patterns.

        Conditions:
        - A: `mov R1, [m]` where R1 is a 32-bit GP reg.
        - B: `mov R2, [m]` where R2 is a 32-bit GP reg, R2 != R1.
        - The memory operands match textually after whitespace
          normalization (and after stripping any size prefixes).
        - The memory operand does NOT reference R1 (else after A
          modifies R1, B's address would be different).
        - This is the simpler 2-line case. Handled here BEFORE
          `dup_load_chain_to_copy` runs (which expects a 4-line
          chained-deref pattern).
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            a_src = ap[1].strip()
            r2 = bp[0].strip().lower()
            b_src = bp[1].strip()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(r2)
                or r1 == r2
            ):
                i += 1
                continue
            # Strip optional size prefix.
            def strip_size(s: str) -> str:
                for p in ("dword ", "word ", "byte ", "qword "):
                    if s.lower().startswith(p):
                        return s[len(p):].lstrip()
                return s
            a_src_norm = strip_size(a_src)
            b_src_norm = strip_size(b_src)
            if not (
                a_src_norm.startswith("[")
                and a_src_norm.endswith("]")
                and b_src_norm.startswith("[")
                and b_src_norm.endswith("]")
            ):
                i += 1
                continue
            # Compare textually after whitespace-normalizing.
            a_inner = " ".join(a_src_norm[1:-1].split())
            b_inner = " ".join(b_src_norm[1:-1].split())
            if a_inner != b_inner:
                i += 1
                continue
            # Memory operand must not reference R1 (else after A
            # writes R1, the address that B reads is different).
            if self._references_reg_family(a_src, r1):
                i += 1
                continue
            # Rewrite B as `mov R2, R1`.
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}mov     {r2}, {r1}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{r2}, {r1}",
            )
            out = out[:i + 1] + [new_line] + out[i + 2:]
            self.stats["dup_load_to_copy"] = (
                self.stats.get("dup_load_to_copy", 0) + 1
            )
            # Don't advance i; the new line might combine with
            # subsequent passes' targets.
            i += 1
            continue
        return out

    def _pass_dup_load_with_intermediate(
        self, lines: list[Line]
    ) -> list[Line]:
        """Eliminate the second of two same-source loads when
        separated by intermediate code that doesn't disturb [m] or
        the second register. Pattern:

            mov R1, [m]              ; A
            mov R1, [R1] or [R1+N1]  ; B
            ... K >= 1 instructions, no labels/jumps/calls ...
              (must not modify [m] or read/write R2)
            mov R2, [m]              ; D (drop)
            mov R2, [R2] or [R2+N2]  ; E
        →
            mov R2, [m]              ; load directly to R2
            mov R1, [R2] or [R2+N1]  ; deref via R2 to R1
            ... intermediate ...     ; (R2 still holds m)
            mov R2, [R2] or [R2+N2]  ; deref via R2

        Saves 3 bytes per match (drops the second `mov R2, [m]`)
        for ebp-relative m. Common in linked-list traversal where
        the loop body uses ptr twice — once for `head->val`, once
        for `head = head->next`.

        Conditions:
        - m is a literal `[ebp ± N]` slot (else aliasing analysis
          isn't sound).
        - R1 != R2, both 32-bit GP regs.
        - Intermediate doesn't modify [m]: stores to other slots
          are OK; register-base writes are OK iff [m]'s offset
          isn't address-taken anywhere in the function.
        - Intermediate doesn't read or write R2 (in original R2
          holds its pre-A value; in rewrite R2 holds `m`'s value).
        - No control flow in intermediate (calls, jumps, labels)
          — those would invalidate the cross-block assumption.
        """
        out = list(lines)
        addr_taken_per_line = self._compute_addr_taken_per_line(out)
        MAX_INTERMEDIATE = 16
        i = 0
        while i + 4 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            m_text = ap[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            if not (m_text.startswith("[") and m_text.endswith("]")):
                i += 1
                continue
            if self._ebp_offset(m_text) is None:
                i += 1
                continue
            if bp[0].strip().lower() != r1:
                i += 1
                continue
            b_match = re.match(
                r"^\[\s*" + re.escape(r1)
                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                bp[1].strip(), re.IGNORECASE,
            )
            if b_match is None:
                i += 1
                continue
            # Forward scan for D = `mov R2, [m]` then E = deref R2.
            j = i + 2
            d_idx = None
            r2 = None
            e_match = None
            failed = False
            intermediate_count = 0
            while j < len(out) and intermediate_count <= MAX_INTERMEDIATE:
                line = out[j]
                if line.kind == "label" or line.kind == "directive":
                    failed = True
                    break
                if line.kind != "instr":
                    j += 1
                    continue
                # Test if this is D.
                if line.op == "mov":
                    lp = _operands_split(line.operands)
                    if lp is not None:
                        cand_dst = lp[0].strip().lower()
                        cand_src = lp[1].strip()
                        if (
                            self._is_general_register(cand_dst)
                            and cand_dst != r1
                            and cand_src == m_text
                        ):
                            # Check D+1 is E.
                            if j + 1 < len(out):
                                e = out[j + 1]
                                if (
                                    e.kind == "instr"
                                    and e.op == "mov"
                                ):
                                    ep = _operands_split(e.operands)
                                    if ep is not None:
                                        if (
                                            ep[0].strip().lower()
                                            == cand_dst
                                        ):
                                            e_m = re.match(
                                                r"^\[\s*"
                                                + re.escape(cand_dst)
                                                + r"(?:\s*([+-])"
                                                + r"\s*(\d+))?\s*\]$",
                                                ep[1].strip(),
                                                re.IGNORECASE,
                                            )
                                            if e_m is not None:
                                                r2 = cand_dst
                                                d_idx = j
                                                e_match = e_m
                                                break
                # Not D. Check intermediate constraints.
                if line.op in {"call", "ret", "iret", "iretd",
                               "retn", "retf", "leave", "enter"}:
                    failed = True
                    break
                if line.op.startswith("j"):
                    failed = True
                    break
                ops_text = line.operands or ""
                # Check whether this line modifies [m].
                # Two-operand stores: dest is a memory operand.
                # One-operand RMW (inc/dec/neg/not/shl/shr/sar/etc.):
                # operand is the modified location.
                two_op = _operands_split(ops_text)
                if two_op is not None:
                    dest_op = two_op[0].strip()
                    if "[" in dest_op:
                        if not self._mem_disjoint_with_taken(
                            m_text, dest_op,
                            addr_taken_per_line[j],
                        ):
                            failed = True
                            break
                else:
                    if line.op in {
                        "inc", "dec", "neg", "not", "shl", "shr",
                        "sar", "sal", "rol", "ror", "rcl", "rcr",
                        "pop",
                    }:
                        op_only = (line.operands or "").strip()
                        if "[" in op_only:
                            if not self._mem_disjoint_with_taken(
                                m_text, op_only,
                                addr_taken_per_line[j],
                            ):
                                failed = True
                                break
                intermediate_count += 1
                j += 1
            if failed or d_idx is None or r2 is None:
                i += 1
                continue
            # Need at least 1 intermediate instr. K=0 is handled by
            # the existing dup_load_chain_to_copy passes.
            if d_idx == i + 2:
                i += 1
                continue
            # Second constraint: R2 not read/written in intermediate.
            r2_safe = True
            for k in range(i + 2, d_idx):
                line = out[k]
                if line.kind != "instr":
                    continue
                ops_text = line.operands or ""
                if self._references_reg_family(ops_text, r2):
                    r2_safe = False
                    break
                # Implicit-reg writers (cdq writes EDX, mul/idiv
                # writes EDX:EAX, etc.) — bail on R2 in {eax, ecx,
                # edx} if any implicit-reg user appears.
                if (
                    line.op in PeepholeOptimizer._IMPLICIT_REG_USERS
                    and r2 in {"eax", "ecx", "edx", "esi", "edi"}
                ):
                    r2_safe = False
                    break
            if not r2_safe:
                i += 1
                continue
            # All checks passed. Rewrite.
            indent_a = self._extract_indent(a.raw)
            indent_b = self._extract_indent(b.raw)
            new_a_raw = f"{indent_a}mov     {r2}, {m_text}"
            new_a = Line(
                raw=new_a_raw, kind="instr", op="mov",
                operands=f"{r2}, {m_text}",
            )
            if b_match.group(1) is None:
                new_b_addr = f"[{r2}]"
            else:
                new_b_addr = (
                    f"[{r2} {b_match.group(1)} "
                    f"{b_match.group(2)}]"
                )
            new_b_raw = f"{indent_b}mov     {r1}, {new_b_addr}"
            new_b = Line(
                raw=new_b_raw, kind="instr", op="mov",
                operands=f"{r1}, {new_b_addr}",
            )
            # Replace A and B with new_a and new_b; drop D; keep
            # everything else.
            out = (
                out[:i]
                + [new_a, new_b]
                + out[i + 2:d_idx]
                + out[d_idx + 1:]
            )
            addr_taken_per_line = self._compute_addr_taken_per_line(
                out
            )
            self.stats["dup_load_with_intermediate"] = (
                self.stats.get("dup_load_with_intermediate", 0) + 1
            )
        return out

    def _pass_dup_addr_compute_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop a 4-line address-recompute sequence when an
        immediately-preceding 3-line compute produced the same
        address, separated by a deref that doesn't clobber the
        address-holding register.

        Pattern (9 consecutive instr lines):
            A: mov R1, MEM1                ; load base
            C: imul R2, R2, K  OR  shl R2, N  ; scale (R2 already
                                              ; held the index)
            D: add R1, R2                  ; R1 = &elem
            E: mov R3, [R1 + off1]         ; first deref, R3 != R1
            F: mov R1, MEM1                ; (drop) recompute starts
            G: mov R2, MEM2                ; (drop) reload index
            H: imul R2, R2, K              ; (drop)
            I: add R1, R2                  ; (drop)
            J: mov R4, [R1 + off2]         ; second deref

        After:
            A, C, D, E, J  (drop F, G, H, I)

        Common in struct-array indexing in LOOPS where the struct
        size doesn't fit SIB scale (sizeof not in {1, 2, 4, 8}).
        The codegen evaluates `pts[i].x` and `pts[i].y` separately;
        the second evaluation rebuilds `&pts[i]` from scratch.

        For correctness, R2's value before A must equal what G
        loads. Verified by scanning backward within the same basic
        block from A: the most recent write to R2 must be a `mov
        R2, MEM2` that textually matches G.

        Saves 4 instructions per match (~13-17 bytes).
        """
        out = list(lines)
        drop_indices: set[int] = set()

        def norm_ops(ops: str) -> str:
            return " ".join(ops.split())

        def find_r2_pre_value(
            start_idx: int, r2: str
        ) -> str | None:
            """Scan backward from start_idx within the same basic
            block (until the previous label) for the most recent
            `mov R2, MEM_Y`. Return MEM_Y or None.

            Bails (returns None) on any non-mov write to R2 or any
            R2 reference that we can't confidently classify as
            read-only.
            """
            k = start_idx - 1
            while k >= 0:
                ln = out[k]
                if ln.kind == "label":
                    return None
                if ln.kind != "instr":
                    k -= 1
                    continue
                # Does this write R2 via a mov?
                if ln.op == "mov":
                    parts = _operands_split(ln.operands)
                    if parts is not None:
                        dst = parts[0].strip().lower()
                        if dst == r2:
                            src = parts[1].strip()
                            if (
                                src.startswith("[")
                                and src.endswith("]")
                            ):
                                return src
                            return None
                # Check for potential writes to R2 from other ops.
                ops_text = ln.operands or ""
                if not self._references_reg_family(ops_text, r2):
                    # No R2 mention; safe to continue.
                    k -= 1
                    continue
                # R2 referenced. Check forms:
                # - 1-op RMW (inc/dec/neg/not/shl/shr/sar/mul/div/
                #   imul/idiv on a single register or memory) where
                #   the operand IS R2 → write.
                # - 2-op ops where dest is R2 → write (except cmp/
                #   test which are read-only).
                # - 1-op ops with R2 as memory base (e.g.,
                #   `inc dword [ecx]`) → R2 is read, not written.
                # - call/iret/leave/etc. → may clobber R2 implicitly.
                if ln.op in {
                    "call", "iret", "iretd", "leave", "enter",
                    "ret", "retn", "retf",
                }:
                    return None
                parts = _operands_split(ops_text)
                if parts is not None:
                    # 2-operand form
                    dest = parts[0].strip().lower()
                    if dest == r2:
                        if ln.op not in ("cmp", "test"):
                            return None
                else:
                    # 1-operand or no-operand form. If the single
                    # operand is exactly R2 (a register), it's an
                    # RMW.
                    op_text = ops_text.strip().lower()
                    if op_text == r2:
                        return None
                    # If R2 is mentioned inside a memory operand
                    # (`[ecx]`), it's a memory address — R2 is
                    # READ, not written. Continue.
                # Implicit-reg writers
                if ln.op in PeepholeOptimizer._IMPLICIT_REG_USERS:
                    if r2 in {"eax", "ecx", "edx", "esi", "edi"}:
                        return None
                k -= 1
            return None

        i = 0
        while i + 8 < len(out):
            block = out[i:i + 9]
            if not all(ln.kind == "instr" for ln in block):
                i += 1
                continue
            a, c, d, e, f, g, h, ii_line, j = block

            # A: mov R1, MEM1
            if a.op != "mov":
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            mem1 = ap[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            if not (mem1.startswith("[") and mem1.endswith("]")):
                i += 1
                continue

            # C: imul R2, R2, K  OR  shl R2, N
            r2 = None
            if c.op == "imul":
                parts = [
                    p.strip() for p in c.operands.split(",")
                ]
                if len(parts) != 3:
                    i += 1
                    continue
                if parts[0].lower() != parts[1].lower():
                    i += 1
                    continue
                r2 = parts[0].lower()
                if (
                    not self._is_general_register(r2)
                    or r2 == r1
                ):
                    i += 1
                    continue
                try:
                    int(parts[2], 0)
                except ValueError:
                    i += 1
                    continue
            elif c.op in ("shl", "sal"):
                cp = _operands_split(c.operands)
                if cp is None:
                    i += 1
                    continue
                r2 = cp[0].strip().lower()
                if (
                    not self._is_general_register(r2)
                    or r2 == r1
                ):
                    i += 1
                    continue
                try:
                    int(cp[1].strip(), 0)
                except ValueError:
                    i += 1
                    continue
            else:
                i += 1
                continue

            # D: add R1, R2
            if d.op != "add":
                i += 1
                continue
            dp = _operands_split(d.operands)
            if dp is None:
                i += 1
                continue
            if (
                dp[0].strip().lower() != r1
                or dp[1].strip().lower() != r2
            ):
                i += 1
                continue

            # E: mov R3, [R1 + off1] with R3 != R1 and R3 != R2
            if e.op != "mov":
                i += 1
                continue
            ep = _operands_split(e.operands)
            if ep is None:
                i += 1
                continue
            r3 = ep[0].strip().lower()
            if not self._is_general_register(r3):
                i += 1
                continue
            if r3 == r1 or r3 == r2:
                i += 1
                continue
            e_src = ep[1].strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if e_src.lower().startswith(prefix):
                    e_src = e_src[len(prefix):].lstrip()
                    break
            e_match = re.match(
                r"^\[\s*" + re.escape(r1)
                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                e_src, re.IGNORECASE,
            )
            if e_match is None:
                i += 1
                continue

            # F = A textually
            if f.op != a.op or norm_ops(f.operands) != norm_ops(a.operands):
                i += 1
                continue

            # G: mov R2, MEM2_X (R2 reloaded — recompute's index load)
            if g.op != "mov":
                i += 1
                continue
            gp = _operands_split(g.operands)
            if gp is None:
                i += 1
                continue
            if gp[0].strip().lower() != r2:
                i += 1
                continue
            mem2_x = gp[1].strip()
            if not (
                mem2_x.startswith("[") and mem2_x.endswith("]")
            ):
                i += 1
                continue

            # H = C textually
            if h.op != c.op or norm_ops(h.operands) != norm_ops(c.operands):
                i += 1
                continue
            # I = D textually
            if (
                ii_line.op != d.op
                or norm_ops(ii_line.operands) != norm_ops(d.operands)
            ):
                i += 1
                continue

            # J: mov R4, [R1 + off2]
            if j.op != "mov":
                i += 1
                continue
            jp = _operands_split(j.operands)
            if jp is None:
                i += 1
                continue
            j_src = jp[1].strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if j_src.lower().startswith(prefix):
                    j_src = j_src[len(prefix):].lstrip()
                    break
            j_match = re.match(
                r"^\[\s*" + re.escape(r1)
                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                j_src, re.IGNORECASE,
            )
            if j_match is None:
                i += 1
                continue

            # Verify: scan backward from A for the most recent
            # write to R2. It must be `mov R2, MEM2_Y` where
            # MEM2_Y matches G's source MEM2_X.
            r2_pre_src = find_r2_pre_value(i, r2)
            if r2_pre_src is None:
                i += 1
                continue
            if norm_ops(r2_pre_src) != norm_ops(mem2_x):
                i += 1
                continue

            # All conditions met. Drop F, G, H, I.
            for k in range(i + 4, i + 8):
                drop_indices.add(k)
            i += 9
            continue

        if not drop_indices:
            return out
        new_out = []
        for k, ln in enumerate(out):
            if k in drop_indices:
                self.stats["dup_addr_compute_collapse"] = (
                    self.stats.get(
                        "dup_addr_compute_collapse", 0
                    ) + 1
                )
                continue
            new_out.append(ln)
        return new_out

    def _pass_dead_addr_recompute(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop a redundant 3-or-4-line address recompute when an
        earlier identical compute in the same basic block produced the
        same address still held in R1.

        Compute pattern:
            A: mov R1, MEM1
            [G: mov R2, MEM2  (optional explicit index reload)]
            C: imul R2, R2, K  OR  shl R2, N
            D: add R1, R2

        Two computes are equivalent iff they have the same R1, MEM1,
        R2, K, scale_op, and effective MEM2 (G's source if explicit,
        else the most recent backward write to R2 — must be `mov R2,
        MEM2`).

        Drop the later compute (3 or 4 lines, ~9-15 bytes) when:
        - An earlier equivalent compute exists in the current basic
          block.
        - R1 is not written between the earlier D and the later A
          (intermediate stores via [R1 + off] are fine — they don't
          modify R1).
        - The MEM1 slot is not written between.
        - The effective MEM2 slot is not written between.
        - The R2 register is restored to its MEM2 value before the
          later C reads it (verified by the find_r2_pre_value scan
          on the later compute's start).

        Differs from dup_addr_compute_collapse: that pass requires a
        rigid 9-line pattern with E being a single load between two
        compute sequences. This pass is more flexible — intermediate
        code can be any number of stores, rhs evaluations, etc.

        Common shape: struct-array WRITE loops where the codegen
        emits the address compute once per member assignment.

            pts[i].x = 1; pts[i].y = 2; pts[i].z = 3;

        produces three identical address computes; the 2nd and 3rd
        are dead.
        """
        out = list(lines)
        drop_indices: set[int] = set()

        def norm(s: str) -> str:
            return " ".join(s.split())

        def is_basic_block_break(ln: Line) -> bool:
            if ln.kind == "label":
                return True
            if ln.kind != "instr":
                return False
            if ln.op in (
                "call", "jmp", "ret", "retn", "retf", "iret",
                "iretd", "leave",
            ):
                return True
            if ln.op.startswith("j") and ln.op != "jmp":
                return True
            return False

        def parse_scale_reg(c_ln: Line) -> str | None:
            if c_ln.kind != "instr":
                return None
            if c_ln.op == "imul":
                parts = [p.strip() for p in c_ln.operands.split(",")]
                if (
                    len(parts) == 3
                    and parts[0].lower() == parts[1].lower()
                ):
                    r = parts[0].lower()
                    if self._is_general_register(r):
                        try:
                            int(parts[2], 0)
                            return r
                        except ValueError:
                            return None
                return None
            if c_ln.op in ("shl", "sal"):
                cp = _operands_split(c_ln.operands)
                if cp is not None:
                    r = cp[0].strip().lower()
                    if self._is_general_register(r):
                        try:
                            int(cp[1].strip(), 0)
                            return r
                        except ValueError:
                            return None
                return None
            return None

        def parse_scale_full(
            c_ln: Line, expected_r: str
        ) -> tuple[int, str] | None:
            r = parse_scale_reg(c_ln)
            if r != expected_r:
                return None
            if c_ln.op == "imul":
                parts = [
                    p.strip() for p in c_ln.operands.split(",")
                ]
                return (int(parts[2], 0), "imul")
            cp = _operands_split(c_ln.operands)
            return (int(cp[1].strip(), 0), c_ln.op)

        def find_r2_pre_value(
            start_idx: int, r2: str
        ) -> str | None:
            k = start_idx - 1
            while k >= 0:
                ln = out[k]
                if ln.kind == "label":
                    return None
                if ln.kind != "instr":
                    k -= 1
                    continue
                if ln.op == "mov":
                    parts = _operands_split(ln.operands)
                    if parts is not None:
                        dst = parts[0].strip().lower()
                        if dst == r2:
                            src = parts[1].strip()
                            if (
                                src.startswith("[")
                                and src.endswith("]")
                            ):
                                return src
                            return None
                ops_text = ln.operands or ""
                if not self._references_reg_family(ops_text, r2):
                    k -= 1
                    continue
                if ln.op in {
                    "call", "iret", "iretd", "leave", "enter",
                    "ret", "retn", "retf",
                }:
                    return None
                parts = _operands_split(ops_text)
                if parts is not None:
                    dest = parts[0].strip().lower()
                    if dest == r2:
                        if ln.op not in ("cmp", "test"):
                            return None
                else:
                    op_text = ops_text.strip().lower()
                    if op_text == r2:
                        return None
                if (
                    ln.op
                    in PeepholeOptimizer._IMPLICIT_REG_USERS
                ):
                    if r2 in {
                        "eax", "ecx", "edx", "esi", "edi",
                    }:
                        return None
                k -= 1
            return None

        def detect_compute(
            idx: int,
        ) -> tuple | None:
            """Return (length, r1, mem1_norm, r2, K, scale_op,
            mem2_norm) or None.

            length is 3 (no G) or 4 (with G).
            """
            if idx + 2 >= len(out):
                return None
            a = out[idx]
            if a.kind != "instr" or a.op != "mov":
                return None
            ap = _operands_split(a.operands)
            if ap is None:
                return None
            r1 = ap[0].strip().lower()
            mem1 = ap[1].strip()
            if not self._is_general_register(r1):
                return None
            if not (
                mem1.startswith("[") and mem1.endswith("]")
            ):
                return None

            # Try 4-line: A, G, C, D
            if idx + 3 < len(out):
                g = out[idx + 1]
                c = out[idx + 2]
                d = out[idx + 3]
                if (
                    g.kind == "instr"
                    and g.op == "mov"
                    and c.kind == "instr"
                    and d.kind == "instr"
                ):
                    gp = _operands_split(g.operands)
                    if gp is not None:
                        r2 = gp[0].strip().lower()
                        mem2 = gp[1].strip()
                        if (
                            self._is_general_register(r2)
                            and r2 != r1
                            and mem2.startswith("[")
                            and mem2.endswith("]")
                        ):
                            cinfo = parse_scale_full(c, r2)
                            if cinfo is not None and d.op == "add":
                                dp = _operands_split(d.operands)
                                if (
                                    dp is not None
                                    and dp[0].strip().lower() == r1
                                    and dp[1].strip().lower() == r2
                                ):
                                    return (
                                        4, r1, norm(mem1), r2,
                                        cinfo[0], cinfo[1],
                                        norm(mem2),
                                    )

            # Try 3-line: A, C, D
            c = out[idx + 1]
            d = out[idx + 2]
            if c.kind != "instr" or d.kind != "instr":
                return None
            r2 = parse_scale_reg(c)
            if r2 is None or r2 == r1:
                return None
            cinfo = parse_scale_full(c, r2)
            if cinfo is None:
                return None
            if d.op != "add":
                return None
            dp = _operands_split(d.operands)
            if dp is None:
                return None
            if (
                dp[0].strip().lower() != r1
                or dp[1].strip().lower() != r2
            ):
                return None
            mem2_eff = find_r2_pre_value(idx, r2)
            if mem2_eff is None:
                return None
            return (
                3, r1, norm(mem1), r2,
                cinfo[0], cinfo[1], norm(mem2_eff),
            )

        def line_writes_reg(
            ln: Line, reg: str
        ) -> bool:
            if ln.kind != "instr":
                return False
            if ln.op == "call":
                return reg in ("eax", "ecx", "edx")
            ops_text = ln.operands or ""
            parts = _operands_split(ops_text)
            if parts is not None:
                dst = parts[0].strip().lower()
                if dst == reg:
                    if ln.op in ("cmp", "test"):
                        return False
                    return True
                # If dst is mem and op is `lea reg, ...`,
                # check the dest separately.
                if ln.op == "lea":
                    if dst == reg:
                        return True
            else:
                # Single-op form: inc/dec/neg/mul/div/etc.
                op_text = ops_text.strip().lower()
                if op_text == reg:
                    return True
            # Implicit-reg-writers
            if (
                ln.op in PeepholeOptimizer._IMPLICIT_REG_USERS
                and reg in {"eax", "ecx", "edx", "esi", "edi"}
            ):
                return True
            return False

        def line_writes_mem(
            ln: Line, mem_norm: str
        ) -> bool:
            """Conservatively detect whether ln writes to the slot
            mem_norm. For ebp-relative slots, indirect stores via
            [reg] are assumed to alias unless the addr-taken set
            doesn't contain mem_norm's offset (handled by caller).
            """
            if ln.kind != "instr":
                return False
            if ln.op == "call":
                return True  # call could write any global slot
            ops_text = ln.operands or ""
            parts = _operands_split(ops_text)
            if parts is not None:
                dst = parts[0].strip()
                # Strip size prefix if any
                for prefix in ("dword ", "word ", "byte ", "qword "):
                    if dst.lower().startswith(prefix):
                        dst = dst[len(prefix):].strip()
                        break
                if dst.startswith("[") and dst.endswith("]"):
                    if ln.op in ("cmp", "test", "lea"):
                        return False
                    if norm(dst) == mem_norm:
                        return True
                    # Conservative: register-base derefs might alias
                    # ebp-relative slots only if address-taken (caller
                    # decides to skip pass entirely if so).
                    return False  # conservative; caller handles
            else:
                op_text = ops_text.strip()
                # `inc dword [m]` / `inc [m]`
                for prefix in ("dword ", "word ", "byte ", "qword "):
                    if op_text.lower().startswith(prefix):
                        op_text = op_text[len(prefix):].strip()
                        break
                if (
                    op_text.startswith("[")
                    and op_text.endswith("]")
                ):
                    # 1-op RMW on memory
                    if norm(op_text) == mem_norm:
                        return True
                    return False
            return False

        def has_indirect_mem_write(ln: Line) -> bool:
            """Detect if ln writes through a register-base memory
            operand (could alias address-taken stack slots).
            """
            if ln.kind != "instr":
                return False
            if ln.op == "call":
                return True
            if ln.op in (
                "ret", "retn", "retf", "iret", "iretd",
                "leave", "enter",
            ):
                return False
            ops_text = ln.operands or ""
            parts = _operands_split(ops_text)
            if parts is not None:
                dst = parts[0].strip()
                for prefix in (
                    "dword ", "word ", "byte ", "qword ",
                ):
                    if dst.lower().startswith(prefix):
                        dst = dst[len(prefix):].strip()
                        break
                if (
                    dst.startswith("[")
                    and dst.endswith("]")
                ):
                    if ln.op in ("cmp", "test", "lea"):
                        return False
                    inner = dst[1:-1].strip().lower()
                    # If inner mentions ebp without a register-base
                    # only (just `ebp [+- N]`), it's a direct stack
                    # slot, not indirect.
                    # If it mentions any non-ebp gp register, it's
                    # indirect.
                    has_other_reg = any(
                        re.search(rf"\b{r}\b", inner)
                        for r in (
                            "eax", "ecx", "edx", "ebx",
                            "esi", "edi", "esp",
                        )
                    )
                    return has_other_reg
            else:
                op_text = ops_text.strip()
                for prefix in (
                    "dword ", "word ", "byte ", "qword ",
                ):
                    if op_text.lower().startswith(prefix):
                        op_text = op_text[len(prefix):].strip()
                        break
                if (
                    op_text.startswith("[")
                    and op_text.endswith("]")
                ):
                    inner = op_text[1:-1].strip().lower()
                    has_other_reg = any(
                        re.search(rf"\b{r}\b", inner)
                        for r in (
                            "eax", "ecx", "edx", "ebx",
                            "esi", "edi", "esp",
                        )
                    )
                    return has_other_reg
            return False

        def is_ebp_relative(mem_norm: str) -> bool:
            """Return True iff mem_norm is exactly `[ebp + N]` or
            `[ebp - N]` (literal disp8/disp32, no register index).
            """
            inner = mem_norm.strip()
            if not (inner.startswith("[") and inner.endswith("]")):
                return False
            return bool(
                re.match(
                    r"^\[\s*ebp\s*([+-]\s*\d+)?\s*\]$",
                    inner,
                    re.IGNORECASE,
                )
            )

        def ebp_offset(mem_norm: str) -> int | None:
            m = re.match(
                r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                mem_norm.strip(),
                re.IGNORECASE,
            )
            if m is None:
                # Maybe `[ebp]` (offset 0)?
                m2 = re.match(
                    r"^\[\s*ebp\s*\]$",
                    mem_norm.strip(),
                    re.IGNORECASE,
                )
                if m2 is not None:
                    return 0
                return None
            sign = 1 if m.group(1) == "+" else -1
            return sign * int(m.group(2))

        # Compute address-taken slots per line (for indirect-write
        # alias check).
        addr_taken_per_line = self._compute_addr_taken_per_line(out)

        # Find compute starts.
        computes = []
        i = 0
        while i < len(out):
            info = detect_compute(i)
            if info is not None:
                computes.append((i, info))
                i += info[0]
            else:
                i += 1

        # For each later compute, find an earlier equivalent compute
        # in the same basic block with no intervening writes.
        for j in range(1, len(computes)):
            later_idx, later_info = computes[j]
            if later_idx in drop_indices:
                continue
            (
                later_len, r1, mem1_norm, r2, K, scale_op,
                mem2_norm,
            ) = later_info

            # Equivalence key excludes length (G optional vs not).
            later_key = later_info[1:]

            for k in range(j - 1, -1, -1):
                earlier_idx, earlier_info = computes[k]
                if earlier_idx in drop_indices:
                    continue
                # Compare keys (ignore length).
                if earlier_info[1:] != later_key:
                    continue
                earlier_end = earlier_idx + earlier_info[0]
                # Check basic block: no terminator/label between
                # earlier_end and later_idx.
                blocked = False
                for m in range(earlier_end, later_idx):
                    if is_basic_block_break(out[m]):
                        blocked = True
                        break
                if blocked:
                    break

                # Check no writes to R1, MEM1 slot, MEM2 slot.
                bad = False
                mem1_off = (
                    ebp_offset(mem1_norm)
                    if is_ebp_relative(mem1_norm)
                    else None
                )
                mem2_off = (
                    ebp_offset(mem2_norm)
                    if is_ebp_relative(mem2_norm)
                    else None
                )
                for m in range(earlier_end, later_idx):
                    ln = out[m]
                    if line_writes_reg(ln, r1):
                        bad = True
                        break
                    if line_writes_mem(ln, mem1_norm):
                        bad = True
                        break
                    if line_writes_mem(ln, mem2_norm):
                        bad = True
                        break
                    # Indirect memory write check: if mem1 or mem2 is
                    # ebp-relative AND that slot's offset is
                    # address-taken, an indirect write could alias.
                    if has_indirect_mem_write(ln):
                        addr_taken = addr_taken_per_line[m]
                        if (
                            mem1_off is not None
                            and mem1_off in addr_taken
                        ):
                            bad = True
                            break
                        if (
                            mem2_off is not None
                            and mem2_off in addr_taken
                        ):
                            bad = True
                            break
                        # Non-ebp-relative MEM1 or MEM2 (e.g.
                        # `[_glob]`) — conservatively assume indirect
                        # might alias.
                        if (
                            mem1_off is None
                            or mem2_off is None
                        ):
                            bad = True
                            break
                if bad:
                    continue

                # Verify R2 still has effective MEM2 value at later C.
                # later C is at later_idx + (later_len - 2).
                later_c_idx = later_idx + (later_len - 2)
                later_pre_val = find_r2_pre_value(later_c_idx, r2)
                if (
                    later_pre_val is None
                    or norm(later_pre_val) != mem2_norm
                ):
                    continue

                # All conditions met — drop later compute.
                for m in range(later_idx, later_idx + later_len):
                    drop_indices.add(m)
                break

        if not drop_indices:
            return out
        new_out = []
        for k, ln in enumerate(out):
            if k in drop_indices:
                self.stats["dead_addr_recompute"] = (
                    self.stats.get("dead_addr_recompute", 0) + 1
                )
                continue
            new_out.append(ln)
        return new_out

    def _pass_uncollapse_cmp_when_reload(
        self, lines: list[Line]
    ) -> list[Line]:
        """Reverse ``cmp_load_collapse`` when an immediate reload of
        the same memory follows. Pattern (3 consecutive instr lines):

            cmp <size> [m], OPND   ; A
            jcc L                  ; B
            mov reg, [m]           ; C — reload of [m] (same address)
        →
            mov reg, [m]           ; load
            cmp reg, OPND          ; (or `test reg, reg` if OPND == 0)
            jcc L

        Saves 2 bytes per match in the common ``cmp dword [m], 0`` →
        ``test reg, reg`` case (4-byte cmp-imm8 + 3-byte mov-rel32 →
        3-byte mov-rel32 + 2-byte test). Saves 1 byte for non-zero
        cmp-immediate or cmp-register forms.

        Why this pattern arises: ``cmp_load_collapse`` greedily folds
        ``mov reg, [m]; cmp reg, X`` into ``cmp dword [m], X`` when
        reg is dead at the cmp. But if a downstream basic block
        reloads the same [m] into a register, the load was needed
        anyway — un-folding back to ``mov reg, [m]; cmp reg, X``
        leaves the value usable.

        Conditions:
        - 3 adjacent instr lines with op = cmp/test, jcc, mov.
        - Both A and C reference the same memory operand.
        - The mov's destination is a 32-bit GP register.
        - The cmp/test is on dword (explicit ``dword`` prefix or NASM
          inferred from a register operand). Conservative: require
          either explicit `dword` OR cmp's second operand is a
          32-bit register or numeric literal (since byte/word forms
          would have a sub-register or `byte`/`word` prefix).
        - Mem operand has no register that conflicts with reg.
        - reg is dead at jcc's ENTRY: must be dead at jcc target
          (else target sees rewrite's M[m] instead of original
          pre-A value) — `_reg_dead_after` from the jcc covers
          both target and fallthrough paths.
        """
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op in ("cmp", "test")
                and b.kind == "instr"
                and b.op.startswith("j") and b.op != "jmp"
                and c.kind == "instr"
                and c.op in ("mov", "movsx", "movzx")
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            cp = _operands_split(c.operands)
            if ap is None or cp is None:
                i += 1
                continue
            a_mem = ap[0].strip()
            a_opnd = ap[1].strip()
            # Strip optional size prefix on A's mem operand for
            # comparison with C's mem operand. Track whether the
            # cmp was explicitly dword/byte/word-sized.
            a_size = None
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if a_mem.lower().startswith(prefix):
                    a_size = prefix.strip()
                    a_mem = a_mem[len(prefix):].lstrip()
                    break
            if not (a_mem.startswith("[") and a_mem.endswith("]")):
                i += 1
                continue
            # C: mov/movsx/movzx reg, [m] (with optional size prefix)
            c_dst = cp[0].strip().lower()
            c_src = cp[1].strip()
            if not self._is_general_register(c_dst):
                i += 1
                continue
            # Strip size prefix on C's source.
            c_size = None
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if c_src.lower().startswith(prefix):
                    c_size = prefix.strip()
                    c_src = c_src[len(prefix):].lstrip()
                    break
            if c_src != a_mem:
                i += 1
                continue
            # Size matching: cmp's size must match C's load size.
            # For mov: A must be dword (or no prefix + reg operand).
            # For movsx/movzx: A's size must match C's source size.
            if c.op == "mov":
                if a_size is not None and a_size != "dword":
                    i += 1
                    continue
                if a_size is None:
                    # Be conservative: the second operand must be a
                    # 32-bit GP register.
                    if not self._is_general_register(a_opnd.lower()):
                        i += 1
                        continue
            else:
                # movsx/movzx — both A and C must have explicit
                # matching size.
                if c_size is None or a_size != c_size:
                    i += 1
                    continue
                # Restrict to byte/word loads.
                if c_size not in ("byte", "word"):
                    i += 1
                    continue
                # For sub-word movsx/movzx, only the cmp-with-0
                # rewrite is safe across all jcc — non-zero immediate
                # operands have potentially-different signed/unsigned
                # interpretation between byte cmp and 32-bit cmp.
                # Restrict to OPND == 0.
                if a_opnd.strip() != "0":
                    i += 1
                    continue
                # For movzx (zero-extension), SF differs from byte
                # cmp's SF. Restrict to ZF-only Jcc.
                if c.op == "movzx":
                    if b.op not in self._ZF_ONLY_FLAG_READERS:
                        i += 1
                        continue
            # The mem operand must not reference c_dst (or its sub-
            # aliases) — for the dword `mov` case. For movsx/movzx
            # of byte/word, the load reads [m] BEFORE writing reg,
            # and `_reg_dead_after` covers the post-jcc state, so
            # the rewrite is safe even when [m] references reg.
            if c.op == "mov" and self._references_reg_family(a_mem, c_dst):
                i += 1
                continue
            # The cmp/test second operand must not reference c_dst's
            # family either (would conflict with reg's new value).
            if self._references_reg_family(a_opnd, c_dst):
                i += 1
                continue
            # Reg must be dead at jcc target only (not fallthrough).
            # Reasoning: in the rewrite, the fallthrough state of reg
            # is the same as the original (C runs on the fallthrough
            # in both versions, just hoisted earlier in mine). Only
            # the JCC-TAKEN path differs — original keeps reg's pre-A
            # value, my rewrite has reg = M[m]. So reg must be dead
            # at the jcc target.
            #
            # For mov: stay with the existing fallthrough+target check
            # (`_reg_dead_after`) since the dword-mov branch doesn't
            # need the looser semantics, and the existing tests assume
            # this stricter check.
            #
            # For movsx/movzx: only target-path-dead is required.
            if c.op == "mov":
                if not self._reg_dead_after(out, i + 1, c_dst):
                    i += 1
                    continue
            else:
                # movsx/movzx: check just the jcc target path.
                target = b.operands.strip()
                target_idx = self._find_label_idx(out, target, i + 1)
                if target_idx is None:
                    i += 1
                    continue
                if not self._reg_dead_after(
                    out, target_idx + 1, c_dst,
                ):
                    i += 1
                    continue
            # All checks passed. Build the rewrite.
            indent_a = self._extract_indent(a.raw)
            indent_c = self._extract_indent(c.raw)
            # New first line: the load (mov / movsx / movzx).
            if c.op == "mov":
                new_load_raw = f"{indent_c}mov     {c_dst}, {a_mem}"
                new_load_ops = f"{c_dst}, {a_mem}"
            else:
                new_load_raw = (
                    f"{indent_c}{c.op}   {c_dst}, {c_size} {a_mem}"
                )
                new_load_ops = f"{c_dst}, {c_size} {a_mem}"
            new_load = Line(
                raw=new_load_raw, kind="instr", op=c.op,
                operands=new_load_ops,
            )
            # New second line: cmp/test against the operand.
            # Special case: cmp/test with 0 → use test reg, reg
            # (2-byte) instead of cmp reg, 0 (3-byte).
            opnd_norm = a_opnd.strip()
            if a.op == "cmp" and opnd_norm == "0":
                new_cmp_raw = f"{indent_a}test    {c_dst}, {c_dst}"
                new_cmp = Line(
                    raw=new_cmp_raw, kind="instr", op="test",
                    operands=f"{c_dst}, {c_dst}",
                )
            else:
                new_cmp_raw = (
                    f"{indent_a}{a.op}     {c_dst}, {opnd_norm}"
                )
                new_cmp = Line(
                    raw=new_cmp_raw, kind="instr", op=a.op,
                    operands=f"{c_dst}, {opnd_norm}",
                )
            # B unchanged.
            out = (
                out[:i]
                + [new_load, new_cmp, b]
                + out[i + 3:]
            )
            self.stats["uncollapse_cmp_when_reload"] = (
                self.stats.get("uncollapse_cmp_when_reload", 0) + 1
            )
        return out

    def _pass_redundant_cmp_at_label(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop a duplicate cmp/test at a label whose only predecessor
        is a Jcc with a matching cmp. Pattern:

            cmp [m], X         ; A
            jcc L              ; B
            ... unconditional terminator before L ...
            L:
            cmp [m], X         ; D — duplicate of A
            jcc2 L2            ; (or anything else)
        →
            cmp [m], X         ; A
            jcc L              ; B
            ... terminator ...
            L:
            jcc2 L2            ; (cmp at L dropped)

        Saves 4-7 bytes per match (drops one cmp/test instruction).
        Common in if-elif chains where each branch tests the same
        expression against different constants.

        The flags from A still hold at L's entry because:
        - L has only one predecessor: the Jcc B.
        - The path A → B → L doesn't modify flags (jcc itself
          doesn't write flags; the cmp is right before the jcc).
        - L is preceded by an unconditional terminator, so no
          text-fallthrough into L.

        Conditions:
        - A is `cmp` or `test`, B is Jcc, both adjacent (B at A+1).
        - B's target L is referenced exactly twice in the file
          (definition + the one Jcc operand).
        - L is preceded by an unconditional terminator (so no
          fallthrough; only the Jcc reaches L).
        - The first instruction after L's definition is D, with
          op == A's op and operands == A's operands.
        - No flag-clobbering instructions between L's def and D
          (in practice they're adjacent).
        """
        out = list(lines)
        # Build label-reference textual count.
        label_pattern = re.compile(r"\.[A-Za-z_]\w*")
        counts: dict[str, int] = {}
        for line in out:
            for match in label_pattern.finditer(line.raw):
                counts[match.group()] = (
                    counts.get(match.group(), 0) + 1
                )
        # Walk and find cmp/test followed by jcc.
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op in ("cmp", "test")
                and b.kind == "instr"
                and b.op.startswith("j") and b.op != "jmp"
            ):
                i += 1
                continue
            target = _branch_target(b)
            if target is None or not target.startswith("."):
                i += 1
                continue
            if counts.get(target, 0) != 2:
                i += 1
                continue
            target_idx = self._find_label_idx(out, target, i)
            if target_idx is None:
                i += 1
                continue
            if not self._preceded_by_terminator(out, target_idx):
                i += 1
                continue
            # Find the first instruction after the label. Bail if we
            # encounter an intermediate label that has back-edges
            # (count > 1) — that means another control path can reach
            # D with different flags. The classic break case is a loop
            # head: `.L4_endif: / .L5_while_top: / cmp; jle; …; jmp
            # .L5_while_top`. The flags from A reach D on the first
            # entry, but the back-jump arrives with whatever flags the
            # loop body left (often clobbered by idiv/etc.).
            d_idx = None
            saw_back_edge = False
            for k in range(target_idx + 1, len(out)):
                ln = out[k]
                if ln.kind == "label":
                    inner_label = ln.raw.strip().rstrip(":")
                    if counts.get(inner_label, 0) > 1:
                        saw_back_edge = True
                        break
                    continue
                if ln.kind in ("blank", "comment"):
                    continue
                if ln.kind != "instr":
                    break  # directive/data
                d_idx = k
                break
            if saw_back_edge or d_idx is None:
                i += 1
                continue
            d = out[d_idx]
            if not (
                d.kind == "instr" and d.op == a.op
                and (d.operands or "").strip()
                    == (a.operands or "").strip()
            ):
                i += 1
                continue
            # All checks passed. Drop D.
            out = out[:d_idx] + out[d_idx + 1:]
            self.stats["redundant_cmp_at_label"] = (
                self.stats.get("redundant_cmp_at_label", 0) + 1
            )
            # Don't advance — re-check from current pos in case
            # cascading effects (e.g., another duplicate cmp pattern).
        return out

    def _pass_op_mem_to_reg_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``OP reg, [m]; mov [m], reg`` into
        ``OP [m], reg`` for commutative OPs (add, and, or, xor).

        Saves 3 bytes per match (drops the redundant store back to
        [m]; the merged in-place OP has the same encoded length as
        the original OP-with-mem-source).

        Common shape: `sum += helper()` where helper's return is in
        EAX, then we add [sum] and store back. The codegen emits
        ``add eax, [sum]; mov [sum], eax`` which directly merges.

        Conditions:
        - OP ∈ {add, and, or, xor} (commutative — `reg + [m]` ==
          `[m] + reg` so the rewrite preserves the value at [m]).
          sub/cmp/imul/shifts don't qualify due to operand-order or
          form differences.
        - A's source [m] textually equals B's destination [m]
          (after stripping optional size prefixes).
        - reg is the same in A's destination and B's source.
        - reg is dead after B (since the rewrite leaves reg
          unchanged whereas the original modified it via the OP).
        """
        out = list(lines)
        COMMUTATIVE_OPS = {"add", "and", "or", "xor"}
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op in COMMUTATIVE_OPS
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            a_dst = ap[0].strip().lower()
            a_src = ap[1].strip()
            if not self._is_general_register(a_dst):
                i += 1
                continue
            # Strip optional size prefix on A's source.
            a_src_clean = a_src
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if a_src_clean.lower().startswith(prefix):
                    a_src_clean = a_src_clean[len(prefix):].lstrip()
                    break
            if not (
                a_src_clean.startswith("[")
                and a_src_clean.endswith("]")
            ):
                i += 1
                continue
            # B: mov [m], reg
            b_dst = bp[0].strip()
            b_src = bp[1].strip().lower()
            if b_src != a_dst:
                i += 1
                continue
            b_dst_clean = b_dst
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_dst_clean.lower().startswith(prefix):
                    b_dst_clean = b_dst_clean[len(prefix):].lstrip()
                    break
            if not (
                b_dst_clean.startswith("[")
                and b_dst_clean.endswith("]")
            ):
                i += 1
                continue
            if b_dst_clean != a_src_clean:
                i += 1
                continue
            # reg must be dead after B (since the rewrite leaves
            # reg unchanged whereas the OP would have modified it).
            if not self._reg_dead_after(out, i + 2, a_dst):
                i += 1
                continue
            # All checks passed. Rewrite to single in-place OP.
            indent = self._extract_indent(a.raw)
            spacer = " " * max(1, 8 - len(a.op))
            new_raw = (
                f"{indent}{a.op}{spacer}{a_src_clean}, {a_dst}"
            )
            new_line = Line(
                raw=new_raw, kind="instr", op=a.op,
                operands=f"{a_src_clean}, {a_dst}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["op_mem_to_reg_collapse"] = (
                self.stats.get("op_mem_to_reg_collapse", 0) + 1
            )
        return out

    def _pass_push_pop_to_free_reg(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace ``push <SRC>; ...chain...; pop REG; <consumer
        using REG>`` with ``mov FREE_REG, <SRC>; ...chain...;
        <consumer using FREE_REG>``, dropping the push and pop
        entirely.

        Saves 4 bytes per match (drops 3-byte push + 1-byte pop).

        Common shape: dot-product or other dual-array loops where
        the codegen pushes one operand (computed via SIB-form
        load), then computes the other operand in EAX (clobbering
        EAX), then pops the saved operand and combines:

            push    dword [eax + ecx*4]   ; save a[i]
            mov     eax, [ebp + 12]       ; b base
            mov     eax, [eax + ecx*4]    ; b[i]
            pop     ecx                   ; restore a[i] in ecx
            imul    eax, ecx              ; eax = a[i] * b[i]

        With EDX free, this becomes:

            mov     edx, [eax + ecx*4]    ; save a[i] in EDX
            mov     eax, [ebp + 12]
            mov     eax, [eax + ecx*4]    ; b[i]
            imul    eax, edx              ; eax = a[i] * b[i]

        Differs from ``push_pop_to_mov`` (which also drops the
        push/pop): that pass places the mov at the POP position,
        reading SRC there. If SRC references registers the chain
        modified, the post-chain values are wrong — so it
        restricts to ebp-relative SRC. My pass places the mov at
        the PUSH position, reading SRC there (pre-chain), then
        carries the value through the chain in a free register.
        Works for arbitrary SRC including SIB-form memory.

        Conditions:
        - Chain is stack-balanced (no unbalanced inner pushes/pops).
        - Chain has no labels, jumps, calls, ret/leave/enter.
        - Chain doesn't reference ESP (else dropping the push
          changes ESP-relative addressing in chain).
        - A free GP register exists (unused in the function);
          prefer EDX (cdecl scratch).
        - Free register isn't referenced in the chain (automatic
          since it's unused in the function).
        - Consumer is the immediately-next instruction after pop.
        - Consumer reads REG (the pop's target).
        - REG dead after the consumer.
        """
        out = list(lines)
        unused_per_line = self._compute_unused_regs_per_line(out)
        # Preference order for the free register: EDX (cdecl
        # scratch, no callee-save cost) > caller-saved scratch.
        FREE_REG_ORDER = ["edx", "esi", "edi", "ebx"]
        i = 0
        while i + 2 < len(out):
            a = out[i]
            if not (a.kind == "instr" and a.op == "push"):
                i += 1
                continue
            push_op = a.operands.strip()
            # Strip optional size prefix.
            stripped = push_op
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):].lstrip()
                    break
            # We don't fire for register pushes (those are saving
            # callee values for restore, distinct codegen pattern).
            if self._is_general_register(push_op.lower()):
                i += 1
                continue
            if self._is_general_register(stripped.lower()):
                i += 1
                continue
            # Walk forward looking for matching pop, tracking depth.
            depth = 1
            match_idx = None
            failed = False
            j = i + 1
            while j < len(out):
                ln = out[j]
                if ln.kind == "label":
                    failed = True
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                if ln.op == "push":
                    depth += 1
                    j += 1
                    continue
                if ln.op == "pop":
                    if depth == 1:
                        match_idx = j
                        break
                    depth -= 1
                    j += 1
                    continue
                if ln.op in {
                    "call", "ret", "iret", "iretd", "retf",
                    "retn", "leave", "enter",
                }:
                    failed = True
                    break
                if ln.op.startswith("j"):
                    failed = True
                    break
                if "esp" in (ln.operands or "").lower():
                    failed = True
                    break
                j += 1
            if failed or match_idx is None:
                i += 1
                continue
            # The pop must target a GP register.
            pop_line = out[match_idx]
            target_reg = pop_line.operands.strip().lower()
            if not self._is_general_register(target_reg):
                i += 1
                continue
            # The next instruction after pop is the consumer. It
            # must reference target_reg AND target_reg must be dead
            # after it.
            cons_idx = None
            for k in range(match_idx + 1, len(out)):
                ln = out[k]
                if ln.kind in ("blank", "comment"):
                    continue
                if ln.kind != "instr":
                    break
                cons_idx = k
                break
            if cons_idx is None:
                i += 1
                continue
            consumer = out[cons_idx]
            if not self._references_reg_family(
                consumer.operands or "", target_reg,
            ):
                i += 1
                continue
            # Disallow consumers that have implicit-reg semantics
            # we can't safely retarget (cdq/idiv/mul/etc. use
            # specific registers).
            if not PeepholeOptimizer._has_explicit_operands_only(
                consumer
            ):
                i += 1
                continue
            if not self._reg_dead_after(out, cons_idx + 1, target_reg):
                i += 1
                continue
            # Pick a free register.
            unused = unused_per_line[i] or set()
            free_reg = None
            for cand in FREE_REG_ORDER:
                if cand in unused and cand != target_reg:
                    free_reg = cand
                    break
            if free_reg is None:
                i += 1
                continue
            # Build the rewrite. The SRC operand in the original
            # push must not reference target_reg (else after we
            # drop the push, the source reads the SAME register
            # at the new mov position — fine since we're at the
            # push position, pre-chain registers). But we should
            # also make sure SRC doesn't reference free_reg (it
            # doesn't, since free_reg is unused in function).
            indent = self._extract_indent(a.raw)
            new_mov = Line(
                raw=f"{indent}mov     {free_reg}, {stripped}",
                kind="instr", op="mov",
                operands=f"{free_reg}, {stripped}",
            )
            # Retarget the consumer: replace target_reg with
            # free_reg in its operands, including sub-aliases.
            new_consumer_operands = self._retarget_reg_in_operands(
                consumer.operands or "", target_reg, free_reg,
            )
            cons_indent = self._extract_indent(consumer.raw)
            cons_op = consumer.op
            spacer = " " * max(1, 8 - len(cons_op))
            new_consumer = Line(
                raw=(
                    f"{cons_indent}{cons_op}{spacer}"
                    f"{new_consumer_operands}"
                ),
                kind="instr", op=cons_op,
                operands=new_consumer_operands,
            )
            out = (
                out[:i] + [new_mov]
                + out[i + 1:match_idx]
                # drop pop
                + [new_consumer]
                + out[cons_idx + 1:]
            )
            unused_per_line = self._compute_unused_regs_per_line(out)
            self.stats["push_pop_to_free_reg"] = (
                self.stats.get("push_pop_to_free_reg", 0) + 1
            )
            # Don't advance — re-check from current pos.
        return out

    @staticmethod
    def _retarget_reg_in_operands(
        operands: str, src_reg: str, dst_reg: str,
    ) -> str:
        """Replace src_reg (and its sub-aliases) with dst_reg (and
        the corresponding sub-aliases) in operands string."""
        REG_FAMILIES = {
            "eax": ("eax", "ax", "al", "ah"),
            "ebx": ("ebx", "bx", "bl", "bh"),
            "ecx": ("ecx", "cx", "cl", "ch"),
            "edx": ("edx", "dx", "dl", "dh"),
            "esi": ("esi", "si"),
            "edi": ("edi", "di"),
            "ebp": ("ebp", "bp"),
            "esp": ("esp", "sp"),
        }
        # Map src family member at index i → dst family member at
        # index i. If the destination family has fewer aliases
        # (no high-byte for esi/edi/ebp/esp), we drop those mappings.
        src_family = REG_FAMILIES.get(src_reg, (src_reg,))
        dst_family = REG_FAMILIES.get(dst_reg, (dst_reg,))
        result = operands
        # Replace each alias. Use word-boundary regex.
        for k, src_alias in enumerate(src_family):
            if k >= len(dst_family):
                break
            dst_alias = dst_family[k]
            result = re.sub(
                r"\b" + re.escape(src_alias) + r"\b",
                dst_alias, result, flags=re.IGNORECASE,
            )
        return result

    def _pass_redundant_recompute_after_cmp(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop a duplicate 2-instruction load+SIB-deref sequence
        immediately after a cmp/test+jcc that doesn't modify the
        registers involved. Pattern:

            mov reg, [m]                   ; A
            ...intermediate (writes idx_reg, not reg)...
            mov reg, [reg + idx_reg*scale] ; B (SIB load)
            cmp reg, X (or test reg, reg)  ; C
            jcc L                          ; D
            mov reg, [m]                   ; E — duplicate of A
            mov reg, [reg + idx_reg*scale] ; F — duplicate of B
            [next instr uses reg = arr[idx]]

        Drop E and F. Saves 6 bytes per match.

        Why safe: cmp/test/jcc don't modify reg or idx_reg (just
        flags). At the start of F, reg = M[m] (from E), idx_reg
        unchanged. F dereferences M[M[m] + idx_reg*scale]. Same
        address as B did, same memory unchanged → same value. So
        at end of F, reg has the same value it had after B.
        Dropping E+F leaves reg unchanged from after B (cmp/jcc
        don't touch it), giving the same final reg value at the
        next instruction.

        Common shape: `if (arr[i] > m) m = arr[i];` and ternary
        forms `cond ? arr[i] : default;` — the codegen evaluates
        arr[i] once for the comparison, then evaluates it AGAIN
        for the value to assign. Redundant.

        Conditions:
        - A: ``mov reg, [m]`` with m a memory operand.
        - Up to 4 intermediate instructions between A and B that
          don't write reg or modify [m]. (Typical: a `mov idx_reg,
          [m_idx]` setting up the SIB index.)
        - B: ``mov reg, [reg + idx_reg*scale]`` or with displacement.
        - C: cmp/test that reads reg (immediately after B).
        - D: jcc (immediately after C).
        - E: textually identical to A (immediately after D).
        - F: textually identical to B (immediately after E).
        """
        out = list(lines)
        MAX_INTERMEDIATE = 4
        i = 0
        while i < len(out):
            a = out[i]
            if not (a.kind == "instr" and a.op == "mov"):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            a_dst = ap[0].strip().lower()
            a_src = ap[1].strip()
            if not self._is_general_register(a_dst):
                i += 1
                continue
            if not (a_src.startswith("[") and a_src.endswith("]")):
                i += 1
                continue
            # Scan forward up to MAX_INTERMEDIATE instructions for B.
            # Intermediate must not write a_dst (or modify [m]).
            j = i + 1
            int_count = 0
            b_idx = None
            scan_failed = False
            while j < len(out) and int_count <= MAX_INTERMEDIATE:
                ln = out[j]
                if ln.kind in ("blank", "comment"):
                    j += 1
                    continue
                if ln.kind in ("label", "directive"):
                    scan_failed = True
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                # Could this be B?
                if ln.op == "mov":
                    lp = _operands_split(ln.operands)
                    if lp is not None:
                        b_dst_cand = lp[0].strip().lower()
                        b_src_cand = lp[1].strip()
                        if b_dst_cand == a_dst:
                            sib_match = re.match(
                                r"^\[\s*" + re.escape(a_dst)
                                + r"\s*\+\s*([a-z]+)\*([0-9]+)"
                                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                                b_src_cand, re.IGNORECASE,
                            )
                            if sib_match is not None:
                                b_idx = j
                                break
                # Otherwise: intermediate instruction. Must not
                # write a_dst.
                if self._instr_writes_reg(ln, a_dst):
                    scan_failed = True
                    break
                int_count += 1
                j += 1
            if scan_failed or b_idx is None:
                i += 1
                continue
            b = out[b_idx]
            bp = _operands_split(b.operands)
            b_src = bp[1].strip()
            sib_match = re.match(
                r"^\[\s*" + re.escape(a_dst)
                + r"\s*\+\s*([a-z]+)\*([0-9]+)"
                + r"(?:\s*([+-])\s*(\d+))?\s*\]$",
                b_src, re.IGNORECASE,
            )
            idx_reg = sib_match.group(1).lower()
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            # Now expect C, D, E, F adjacent after B.
            if b_idx + 4 >= len(out):
                i += 1
                continue
            c = out[b_idx + 1]
            d = out[b_idx + 2]
            e = out[b_idx + 3]
            f = out[b_idx + 4]
            if not all(
                ln.kind == "instr" for ln in (c, d, e, f)
            ):
                i += 1
                continue
            if c.op not in ("cmp", "test"):
                i += 1
                continue
            if not (d.op.startswith("j") and d.op != "jmp"):
                i += 1
                continue
            if not (e.op == "mov" and f.op == "mov"):
                i += 1
                continue
            # E identical to A.
            ep = _operands_split(e.operands)
            fp = _operands_split(f.operands)
            cp = _operands_split(c.operands)
            if ep is None or fp is None or cp is None:
                i += 1
                continue
            if (
                ep[0].strip().lower() != a_dst
                or ep[1].strip() != a_src
            ):
                i += 1
                continue
            # F identical to B.
            if (
                fp[0].strip().lower() != a_dst
                or fp[1].strip() != b_src
            ):
                i += 1
                continue
            # C reads reg.
            if cp[0].strip().lower() != a_dst:
                i += 1
                continue
            # All checks passed. Drop E and F.
            out = out[:b_idx + 3] + out[b_idx + 5:]
            self.stats["redundant_recompute_after_cmp"] = (
                self.stats.get(
                    "redundant_recompute_after_cmp", 0
                ) + 1
            )
            # Don't advance — re-check from current pos.
        return out

    @staticmethod
    def _instr_writes_reg(line: Line, reg32: str) -> bool:
        """Does this instruction write to reg32 (or a sub-alias)?
        Conservative: returns True for mov/lea/movsx/movzx with
        reg as dest, single-operand RMW (inc/dec/neg/not) on reg,
        2-operand ops (add/sub/and/or/xor/shl/shr/etc.) with reg
        as dest, pop reg, and any implicit-write op (cdq/idiv/etc.
        that writes EAX/EDX)."""
        if line.kind != "instr":
            return False
        op = line.op
        family = PeepholeOptimizer._REG_FAMILY.get(reg32, ())
        ops = line.operands or ""
        # Parse operands.
        parts = _operands_split(ops)
        if parts is not None:
            dest = parts[0].strip().lower()
            # 2-operand instructions write to dest (except cmp/test).
            if op in {"cmp", "test"}:
                return False
            # mov/lea/movsx/movzx/add/sub/and/or/xor/imul/shl/shr/sar/etc.
            if dest in family:
                return True
            return False
        # Single-operand or no operands.
        if op == "pop":
            single = ops.strip().lower()
            return single in family
        if op in {"inc", "dec", "neg", "not", "shl", "shr", "sar",
                  "rol", "ror", "rcl", "rcr"}:
            single = ops.strip().lower()
            # Could be mem or reg.
            return single in family
        # Implicit writes (cdq, mul, div, idiv, lodsd, etc.)
        if op == "cdq" or op == "cwd":
            return reg32 == "edx"
        if op in {"mul", "div", "idiv"}:
            # 1-operand form writes EDX:EAX.
            return reg32 in ("eax", "edx")
        if op == "imul":
            # 1-operand: writes EDX:EAX. 2/3-op: writes the dest.
            if "," not in ops:
                return reg32 in ("eax", "edx")
            return False  # already handled by the operand split above
        if op in {"lodsd", "lodsw", "lodsb"}:
            return reg32 in ("eax", "esi")
        if op in {"stosd", "stosw", "stosb"}:
            return reg32 == "edi"
        if op in {"call"}:
            # cdecl: EAX, ECX, EDX clobbered.
            return reg32 in ("eax", "ecx", "edx")
        return False

    def _pass_xfer_store_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov R2, R1; mov [m], R2`` into ``mov [m], R1``
        when R2 is dead after.

        Saves 2 bytes per match (drops the register transfer).

        Common shape: codegen emits `mov eax, edx; mov [m], eax`
        after `cdq; idiv` to store the remainder via EAX. The
        transfer through EAX is unnecessary — `mov [m], edx`
        works directly.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov R2, R1`` (R2 != R1, both 32-bit GP regs).
        - Line B: ``mov [m], R2`` (store R2, plain memory dest).
        - The memory operand of B doesn't reference R2 (else changing
          the source register changes the address too).
        - R2 is dead after line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r2 = ap[0].strip().lower()
            r1 = ap[1].strip().lower()
            if (
                not self._is_general_register(r2)
                or not self._is_general_register(r1)
                or r1 == r2
            ):
                i += 1
                continue
            b_dst = bp[0].strip()
            b_src = bp[1].strip().lower()
            # B must be `mov [m], R2`.
            if b_src != r2:
                i += 1
                continue
            # Strip optional size prefix to inspect dst.
            dst_stripped = b_dst
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if dst_stripped.lower().startswith(prefix):
                    dst_stripped = dst_stripped[len(prefix):].lstrip()
                    break
            if not (
                dst_stripped.startswith("[")
                and dst_stripped.endswith("]")
            ):
                i += 1
                continue
            # B's address operand must not reference R2 (else changing
            # source from R2 to R1 would change the address too — but
            # we don't change the address, so this would still produce
            # different semantics).
            if self._references_reg_family(dst_stripped, r2):
                i += 1
                continue
            # R2 dead after line B.
            if not self._reg_dead_after(out, i + 2, r2):
                i += 1
                continue
            # Apply: rewrite line B to use R1 as source, drop line A.
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}mov     {b_dst}, {r1}"
            new_line = Line(
                raw=new_raw, kind="instr", op="mov",
                operands=f"{b_dst}, {r1}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["xfer_store_collapse"] = (
                self.stats.get("xfer_store_collapse", 0) + 1
            )
            continue
        return out

    def _pass_xfer_load_store_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the codegen's xfer-then-load-then-store pattern.

        Pattern (3 consecutive instr lines):
            A: mov XFER, BASE       ; xfer the address
            B: mov BASE, SRC        ; load value into BASE
            C: mov [XFER], BASE     ; store BASE through XFER

        Rewrite (2 lines):
            A': mov XFER, SRC       ; load value into XFER
            C': mov [BASE], XFER    ; store XFER through BASE

        Saves 1 instruction (~2 bytes).

        Common shape: codegen pattern after computing an address
        in BASE — `mov ECX, EAX; mov EAX, [ebp + N]; mov [ECX], EAX`.
        The xfer to ECX is to free EAX for the value load. We can
        instead load the value directly into ECX and use EAX as
        the address.

        Conditions:
        - A: `mov XFER, BASE` (register-to-register copy).
        - B: `mov BASE, SRC` where SRC doesn't reference XFER (else
          rewrite's `mov XFER, SRC` would self-reference).
        - C: `mov [XFER], BASE` (plain store with XFER as address).
        - XFER and BASE both dead after C (the rewrite swaps which
          register holds the value vs the address; downstream reads
          of either would observe different values).
        - C's address operand uses XFER directly (not a SIB form
          with XFER as base/index — too many cases to handle here).
        """
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
                and c.kind == "instr" and c.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            cp = _operands_split(c.operands)
            if ap is None or bp is None or cp is None:
                i += 1
                continue
            xfer = ap[0].strip().lower()
            base = ap[1].strip().lower()
            if (
                not self._is_general_register(xfer)
                or not self._is_general_register(base)
                or xfer == base
            ):
                i += 1
                continue
            # B: `mov BASE, SRC`.
            b_dst = bp[0].strip().lower()
            b_src = bp[1].strip()
            if b_dst != base:
                i += 1
                continue
            # SRC must not reference XFER.
            if self._references_reg_family(b_src, xfer):
                i += 1
                continue
            # C: `mov [XFER], BASE`.
            c_dst = cp[0].strip()
            c_src = cp[1].strip().lower()
            if c_src != base:
                i += 1
                continue
            # Strip optional size prefix to inspect dst.
            c_size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if c_dst.lower().startswith(prefix):
                    c_size_prefix = c_dst[:len(prefix)]
                    c_dst = c_dst[len(prefix):].lstrip()
                    break
            # C's address must be exactly `[XFER]` (no SIB or disp).
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$",
                c_dst,
            )
            if mem_re is None:
                i += 1
                continue
            mem_reg = mem_re.group(1).lower()
            if mem_reg != xfer:
                i += 1
                continue
            # XFER dead after C — the rewrite leaves XFER = SRC value
            # instead of the original (= address).
            if not self._reg_dead_after(out, i + 3, xfer):
                i += 1
                continue
            # BASE dead after C — the rewrite leaves BASE = address
            # (its pre-A value) instead of the original (= SRC).
            if not self._reg_dead_after(
                out, i + 3, base, treat_as_scratch=True
            ):
                i += 1
                continue
            # Rewrite. A': `mov XFER, SRC`; C': `mov [BASE], XFER`.
            a_indent = self._extract_indent(a.raw)
            c_indent = self._extract_indent(c.raw)
            new_a_raw = f"{a_indent}mov     {xfer}, {b_src}"
            new_c_dst = f"{c_size_prefix}[{base}]"
            new_c_raw = f"{c_indent}mov     {new_c_dst}, {xfer}"
            new_a = Line(
                raw=new_a_raw, kind="instr", op="mov",
                operands=f"{xfer}, {b_src}",
            )
            new_c = Line(
                raw=new_c_raw, kind="instr", op="mov",
                operands=f"{new_c_dst}, {xfer}",
            )
            out = out[:i] + [new_a, new_c] + out[i + 3:]
            self.stats["xfer_load_store_collapse"] = (
                self.stats.get(
                    "xfer_load_store_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_load_add_xfer_forward(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov R1, SRC; add R1, IMM; mov R2, R1`` into
        ``mov R2, SRC; add R2, IMM`` when R1 is dead after.

        Saves 2 bytes (drops the register transfer). Sister of
        `value_forward_to_reg` that handles the case where the load
        is followed by a single arithmetic step before the transfer.

        Common shape: struct member address materialization. The
        codegen emits `mov R, [p_slot]; add R, offset; mov R2, R`
        to compute the address of `p->member` into R2. After this
        pass, the address goes directly to R2 — which lets later
        passes (disp_store_collapse, disp_load_collapse) fold the
        offset into the eventual deref.

        Conditions:
        - Three consecutive instr lines.
        - Line A: ``mov R1, SRC`` (any source).
        - Line B: ``add R1, IMM`` or ``sub R1, IMM`` (numeric).
        - Line C: ``mov R2, R1`` (register copy).
        - R1 != R2.
        - R1 dead after line C.
        - SRC must not reference R2 (would self-reference after
          rewrite).
        """
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op in ("add", "sub")
                and c.kind == "instr" and c.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            cp = _operands_split(c.operands)
            if ap is None or bp is None or cp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            src = ap[1].strip()
            b_dst = bp[0].strip().lower()
            b_imm = bp[1].strip()
            c_dst = cp[0].strip().lower()
            c_src = cp[1].strip().lower()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(c_dst)
            ):
                i += 1
                continue
            # Add/sub must be on R1 with a numeric immediate.
            if b_dst != r1:
                i += 1
                continue
            try:
                int(b_imm)
            except ValueError:
                i += 1
                continue
            # Line C must be R2 = R1, R1 != R2.
            if c_src != r1 or c_dst == r1:
                i += 1
                continue
            # SRC must not reference R2 (else changing dst reg from
            # R1 to R2 would alter which memory location is read).
            if "[" in src and _references_register(src, c_dst):
                i += 1
                continue
            # SRC mustn't be R2 itself (would create self-mov).
            if src.lower() == c_dst:
                i += 1
                continue
            # R1 dead after line C.
            if not self._reg_dead_after(out, i + 3, r1):
                i += 1
                continue
            # Rewrite: replace 3 lines with 2.
            indent_a = self._extract_indent(a.raw)
            indent_b = self._extract_indent(b.raw)
            new_a = Line(
                raw=f"{indent_a}mov     {c_dst}, {src}",
                kind="instr", op="mov",
                operands=f"{c_dst}, {src}",
            )
            new_b = Line(
                raw=f"{indent_b}{b.op}     {c_dst}, {b_imm}",
                kind="instr", op=b.op,
                operands=f"{c_dst}, {b_imm}",
            )
            out = out[:i] + [new_a, new_b] + out[i + 3:]
            self.stats["load_add_xfer_forward"] = (
                self.stats.get("load_add_xfer_forward", 0) + 1
            )
            continue
        return out

    def _pass_self_mov_elimination(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop `mov REG, REG` where dst and src are the same register
        — these are no-ops on x86 (no flag changes, no value change).

        Saves 2 bytes per match. Common after right_operand_retarget
        leaves a stale `mov ecx, ecx` (or other self-mov) when the
        retargeted chain ends with the same register the codegen would
        have transferred to.

        Conservative: only matches plain register-to-register movs
        where the operand strings are identical (after lowercasing).
        Doesn't drop `mov al, al` / `mov ah, ah` etc. (sub-register
        movs are the same shape but might be uncommon — keep simple).
        """
        out: list[Line] = []
        for line in lines:
            if (
                line.kind == "instr"
                and line.op == "mov"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_low = src.strip().lower()
                    if (
                        dest_low == src_low
                        and self._is_general_register(dest_low)
                    ):
                        self.stats["self_mov_elimination"] = (
                            self.stats.get(
                                "self_mov_elimination", 0
                            ) + 1
                        )
                        continue
            out.append(line)
        return out

    def _pass_cmp_load_promote(
        self, lines: list[Line]
    ) -> list[Line]:
        """Promote `mov REG1, [m]` at the loop top to load into REG2
        instead, when a later `mov REG2, [m]` (same memory) appears as
        the SIB index load. Drops the redundant REG2 load and
        retargets the cmp to use REG2.

        Pattern (5 consecutive instructions following a label):
            .L_xxx:
            A: mov REG1, [m]
            B: cmp REG1, X       (or test REG1, REG1)
            C: jcc L
            D: mov REG1, RHS     (RHS doesn't reference REG1)
            E: mov REG2, [m]     (same [m] as A, REG2 != REG1)

        Rewrite:
            .L_xxx:
            A': mov REG2, [m]    (promoted)
            B': cmp REG2, X      (retargeted)
            C: jcc L             (unchanged)
            D: mov REG1, RHS     (unchanged)
            (E dropped)

        Saves 3 bytes per match (the dropped E).

        Common shape: for-loops indexing into an array (`for (i=0;
        i<n; i++) ... arr[i] ...`). Codegen evaluates `i` for the
        cmp into EAX, then later loads `i` again into ECX for the
        SIB index. After the rewrite, the loop top loads `i`
        directly into ECX, skipping the second load.

        Conditions (all required):
        - A is immediately preceded by a label.
        - B is `cmp REG1, X` or `test REG1, REG1` immediately after A.
        - B's second operand X does NOT reference REG2 (else cmp's
          semantics change after retargeting).
        - C is jcc L immediately after B.
        - D is `mov REG1, RHS` immediately after C, where RHS does not
          reference REG1 (no self-RMW — D must overwrite REG1
          cleanly so REG1's pre-A value isn't observable).
        - E is `mov REG2, [m]` immediately after D, with same memory
          operand as A, REG2 != REG1.
        - At C's jump target L: both REG1 and REG2 are dead
          (overwritten before read), so the L path doesn't depend on
          REG1 holding [m] or REG2 holding pre-loop garbage.

        Both REG1 and REG2 must be 32-bit GP registers.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Need lookahead of 5 (A, B, C, D, E) plus a preceding
            # label. So check i >= 1 (label) and i + 4 < len (E).
            if not (
                i >= 1
                and i + 4 < len(lines)
                and lines[i - 1].kind == "label"
                and line.kind == "instr"
                and line.op == "mov"
            ):
                out.append(line)
                i += 1
                continue
            a = line
            b = lines[i + 1]
            c = lines[i + 2]
            d = lines[i + 3]
            e = lines[i + 4]
            # All must be instructions.
            if not all(
                x.kind == "instr" for x in (b, c, d, e)
            ):
                out.append(line)
                i += 1
                continue
            # Parse A: mov REG1, [m].
            ap = _operands_split(a.operands)
            if ap is None:
                out.append(line)
                i += 1
                continue
            reg1 = ap[0].strip().lower()
            a_src = ap[1].strip()
            if not self._is_general_register(reg1):
                out.append(line)
                i += 1
                continue
            if not (a_src.startswith("[") and a_src.endswith("]")):
                out.append(line)
                i += 1
                continue
            # Parse B: cmp REG1, X (or test REG1, REG1).
            if b.op not in ("cmp", "test"):
                out.append(line)
                i += 1
                continue
            bp = _operands_split(b.operands)
            if bp is None:
                out.append(line)
                i += 1
                continue
            b_first = bp[0].strip().lower()
            b_second = bp[1].strip()
            if b_first != reg1:
                out.append(line)
                i += 1
                continue
            # For test: second operand must equal first (test reg, reg).
            if b.op == "test" and b_second.strip().lower() != reg1:
                out.append(line)
                i += 1
                continue
            # Parse C: jcc.
            if not (
                c.op.startswith("j")
                and c.op != "jmp"
            ):
                out.append(line)
                i += 1
                continue
            jcc_target = c.operands.strip()
            # Parse D: mov/lea REG1, RHS, RHS doesn't reference REG1.
            # `lea reg, [m]` is also a fresh write to REG1 (the
            # bracket form reads memory operands like ebp/etc., not
            # REG1's value), so it qualifies the same as `mov`.
            if d.op not in ("mov", "lea"):
                out.append(line)
                i += 1
                continue
            dp = _operands_split(d.operands)
            if dp is None:
                out.append(line)
                i += 1
                continue
            d_dst = dp[0].strip().lower()
            d_src = dp[1].strip()
            if d_dst != reg1:
                out.append(line)
                i += 1
                continue
            if _references_register(d_src, reg1):
                out.append(line)
                i += 1
                continue
            # Parse E: mov REG2, [m] same as A, REG2 != REG1.
            if e.op != "mov":
                out.append(line)
                i += 1
                continue
            ep = _operands_split(e.operands)
            if ep is None:
                out.append(line)
                i += 1
                continue
            reg2 = ep[0].strip().lower()
            e_src = ep[1].strip()
            if not self._is_general_register(reg2):
                out.append(line)
                i += 1
                continue
            if reg2 == reg1:
                out.append(line)
                i += 1
                continue
            if e_src != a_src:
                out.append(line)
                i += 1
                continue
            # B's second operand X must not reference REG2.
            if _references_register(b_second, reg2):
                out.append(line)
                i += 1
                continue
            # At C's jump target, both REG1 and REG2 must be dead.
            target_idx = self._find_label_idx(
                lines, jcc_target, from_idx=i + 2,
            )
            if target_idx is None:
                out.append(line)
                i += 1
                continue
            if not self._reg_dead_after(
                lines, target_idx + 1, reg1,
            ):
                out.append(line)
                i += 1
                continue
            if not self._reg_dead_after(
                lines, target_idx + 1, reg2,
            ):
                out.append(line)
                i += 1
                continue
            # Build rewrite: A → mov REG2, [m]; B → cmp REG2, X; drop E.
            indent_a = self._extract_indent(a.raw)
            indent_b = self._extract_indent(b.raw)
            new_a_raw = f"{indent_a}mov     {reg2}, {a_src}"
            new_a = Line(
                raw=new_a_raw,
                kind="instr",
                op="mov",
                operands=f"{reg2}, {a_src}",
            )
            spacer = " " * max(1, 8 - len(b.op))
            new_b_raw = f"{indent_b}{b.op}{spacer}{reg2}, {b_second}"
            new_b = Line(
                raw=new_b_raw,
                kind="instr",
                op=b.op,
                operands=f"{reg2}, {b_second}",
            )
            out.append(new_a)
            out.append(new_b)
            out.append(c)
            out.append(d)
            # E is dropped.
            self.stats["cmp_load_promote"] = (
                self.stats.get("cmp_load_promote", 0) + 1
            )
            i += 5
        return out

    def _pass_dead_cleanup_before_leave(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop stack-cleanup instructions immediately before a
        function-exit ``leave; ret`` sequence.

        ``leave`` is ``mov esp, ebp; pop ebp`` — it restores ESP to
        the saved frame pointer regardless of any prior ``add esp, N``
        or scratch-register pops. So those cleanup instructions are
        dead.

        Saves 1-3 bytes per dropped instruction.

        Common shape: cdecl arg cleanup at the end of a function.
        ``f(g());`` lowers to:
            push g_args
            call _g
            add esp, X       ; cleanup g's args
            push eax         ; pass g's result
            call _f
            add esp, 4       ; cleanup f's arg
            (leave; ret)
        After ``add_esp_to_pop`` converts the small ``add esp, N``
        forms to ``pop ecx``, the trailing cleanup is one or more
        ``pop ecx``. Those are all dead.

        Droppable instructions:
        - ``pop ecx``, ``pop edx``: caller-saved scratch regs are
          dead at function exit. Both ECX and EDX values after the
          pops are clobbered (the caller doesn't expect them
          preserved per cdecl).
        - ``add esp, N``, ``sub esp, -N``: pure ESP adjustment, gone
          when ``leave`` runs.

        Pop EAX is NEVER dropped — EAX is the return value register
        and a ``pop eax`` immediately before ``leave`` writes the
        return value (sometimes deliberately).
        Pops of EBX/ESI/EDI/EBP are NEVER dropped — these are
        callee-saved registers being restored from the prologue's
        pushes; dropping them breaks the cdecl contract with the
        caller.

        Algorithm: find each ``leave`` followed (modulo blanks/labels)
        by ``ret``, walk backward through labels/blanks/comments to
        find consecutive droppable instructions, drop them.

        Labels between cleanup and leave (e.g. ``.epilogue:``) are
        unaffected — they may be early-return targets that bypass the
        cleanup. Both paths converge on ``leave; ret``.
        """
        droppable_pops = {"pop"}
        droppable_pop_targets = {"ecx", "edx"}

        def is_droppable(line: Line) -> bool:
            if line.kind != "instr":
                return False
            if line.op in droppable_pops:
                return (
                    line.operands.strip().lower()
                    in droppable_pop_targets
                )
            if line.op == "add":
                parts = _operands_split(line.operands)
                if parts is None:
                    return False
                dst = parts[0].strip().lower()
                src = parts[1].strip()
                if dst != "esp":
                    return False
                try:
                    val = int(src, 0)
                except ValueError:
                    return False
                return val > 0
            if line.op == "sub":
                # `sub esp, -N` is rare but equivalent to `add esp, N`.
                parts = _operands_split(line.operands)
                if parts is None:
                    return False
                dst = parts[0].strip().lower()
                src = parts[1].strip()
                if dst != "esp":
                    return False
                try:
                    val = int(src, 0)
                except ValueError:
                    return False
                return val < 0
            return False

        # Identify each `leave` followed by `ret` (with optional
        # blanks/comments between).
        out = list(lines)
        leave_idxs: list[int] = []
        for i, line in enumerate(out):
            if line.kind == "instr" and line.op == "leave":
                # Look forward for ret skipping blanks/comments.
                j = i + 1
                while (
                    j < len(out)
                    and out[j].kind in ("blank", "comment")
                ):
                    j += 1
                if (
                    j < len(out)
                    and out[j].kind == "instr"
                    and out[j].op in ("ret", "iret", "iretd",
                                      "retf", "retn")
                ):
                    leave_idxs.append(i)

        # Walk backward from each leave, marking droppable cleanup.
        # Process from last to first so indices stay valid as we drop.
        drop_set: set[int] = set()
        for leave_idx in reversed(leave_idxs):
            j = leave_idx - 1
            while j >= 0:
                line = out[j]
                if line.kind in ("label", "blank", "comment"):
                    j -= 1
                    continue
                if is_droppable(line):
                    drop_set.add(j)
                    j -= 1
                    continue
                break

        if not drop_set:
            return out

        new_out: list[Line] = []
        for k, line in enumerate(out):
            if k in drop_set:
                self.stats["dead_cleanup_before_leave"] = (
                    self.stats.get(
                        "dead_cleanup_before_leave", 0
                    ) + 1
                )
                continue
            new_out.append(line)
        return new_out

    def _pass_indirect_call_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG, MEM; call REG`` into
        ``call dword MEM``.

        x86's ``call r/m32`` form supports a memory operand directly,
        eliminating the intermediate register load. Saves 2 bytes
        per match for ebp-relative or SIB-form memory (3-byte mov
        dropped, mem-form call gains 1 byte vs reg-form), 1 byte for
        absolute-label memory (5-byte mov dropped, mem-form call
        gains 4 bytes).

        Common shape: function-pointer calls. Codegen evaluates the
        callee expression into EAX, then emits ``call eax``. After
        my pass, the load and call fuse into a single instruction.
        Particularly impactful for vtable-style dispatches where the
        function pointer is at a struct member offset —
        ``mov eax, [ebp + 8]; mov eax, [eax + 4]; call eax`` becomes
        ``mov eax, [ebp + 8]; call dword [eax + 4]``.

        Conditions:
        - Line A: ``mov REG, MEM`` where REG is a 32-bit GP register
          and MEM is a memory operand (``[...]`` form).
        - Line B: ``call REG`` (same REG, single operand).
        - Lines A and B are adjacent.

        Safety: even when MEM references REG (e.g.,
        ``mov eax, [eax + 4]; call eax``), the rewrite is correct.
        The original sequence loads ``[old_eax + 4]`` into eax then
        calls eax. The rewrite reads from ``[old_eax + 4]`` (eax
        still has its pre-mov value at the call site) and calls that
        target. End state matches: same callee invoked, same return
        value in eax post-call.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            if i + 1 >= len(lines):
                out.append(lines[i])
                i += 1
                continue
            a = lines[i]
            b = lines[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "call"
            ):
                out.append(lines[i])
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                out.append(lines[i])
                i += 1
                continue
            dest = ap[0].strip().lower()
            src = ap[1].strip()
            call_target = b.operands.strip().lower()
            if not self._is_general_register(dest) or call_target != dest:
                out.append(lines[i])
                i += 1
                continue
            # Source must be a memory operand `[...]`.
            if not (src.startswith("[") and src.endswith("]")):
                out.append(lines[i])
                i += 1
                continue
            # Build the rewritten call.
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}call    dword {src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="call",
                operands=f"dword {src}",
            )
            out.append(new_line)
            self.stats["indirect_call_collapse"] = (
                self.stats.get("indirect_call_collapse", 0) + 1
            )
            i += 2
        return out

    def _pass_jmp_to_next_label(self, lines: list[Line]) -> list[Line]:
        """P-jmp-to-next: `jmp X` immediately (modulo blanks/comments)
        followed by `X:` becomes a no-op — drop the `jmp`."""
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            target = _jmp_target(line)
            if target is None:
                out.append(line)
                i += 1
                continue
            # Look ahead for the next non-blank/non-comment line.
            j = i + 1
            while j < len(lines) and lines[j].kind in ("blank", "comment"):
                j += 1
            if j < len(lines) and lines[j].kind == "label" and lines[j].label == target:
                # Drop the jmp; keep blanks/comments (they may belong to
                # the label that follows).
                self.stats["jmp_to_next_label"] = (
                    self.stats.get("jmp_to_next_label", 0) + 1
                )
                i += 1
                continue
            out.append(line)
            i += 1
        return out

    def _pass_pure_leaf_drop_frame(
        self, lines: list[Line]
    ) -> list[Line]:
        """Phase F: drop the EBP frame for pure-leaf functions with
        params.

        Pattern (per function):
            _fn:
                    push    ebp
                    mov     ebp, esp
                    <body — only [ebp + N] reads, no push/pop/call,
                    no [ebp - N] (locals), no SIB-form ebp, no ebp
                    write outside the prologue>
            (.epilogue:)?
                    leave
                    ret

        Rewrite:
            _fn:
                    <body with [ebp + N] → [esp + (N - 4)]>
            (.epilogue:)?
                    ret

        Saves 4 bytes per fire (drops `push ebp` + `mov ebp, esp` +
        `leave`; the `ret` stays). Common on arithmetic helper
        functions (`int sq(int x) { return x * x; }`) where the
        peephole has already collapsed the stack-machine slack to a
        register-only body.
        """
        out = list(lines)
        ranges = self._function_ranges(out)
        # Build a flat list of replacements; apply at the end so we
        # don't disturb subsequent function-range indices mid-walk.
        replacements: list[tuple[int, int, list[Line]]] = []
        for start, end in ranges:
            # Need a global label at `start`; if the prelude range
            # has no global label, skip.
            if (
                out[start].kind != "label"
                or out[start].label.startswith(".")
            ):
                continue
            # The prologue must be one of:
            #   `push ebp; mov ebp, esp`           (no locals)
            #   `push ebp; mov ebp, esp; sub esp, K`  (with locals)
            #   `enter K, 0`                        (collapsed form)
            # Find the prologue end. We accept frame_size == 0 OR
            # frame_size != 0 with no `[ebp - N]` accesses in body
            # (the locals are dead — peephole's other passes
            # collapsed them away). When frame_size != 0 AND body
            # does access [ebp - N], we'd need static scratch — not
            # yet implemented, so bail.
            i = start + 1
            while i < end and out[i].kind in ("blank", "comment"):
                i += 1
            push_idx = -1
            mov_idx = -1
            enter_idx = -1
            if (
                i < end
                and out[i].kind == "instr"
                and out[i].op == "push"
                and out[i].operands.strip().lower() == "ebp"
            ):
                push_idx = i
                i += 1
                while i < end and out[i].kind in ("blank", "comment"):
                    i += 1
                if not (
                    i < end
                    and out[i].kind == "instr"
                    and out[i].op == "mov"
                    and out[i].operands.strip().lower() == "ebp, esp"
                ):
                    continue
                mov_idx = i
                body_start = mov_idx + 1
                # Optional `sub esp, K`.
                while i + 1 < end and out[i + 1].kind in (
                    "blank", "comment",
                ):
                    i += 1
                if (
                    i + 1 < end
                    and out[i + 1].kind == "instr"
                    and out[i + 1].op == "sub"
                    and re.match(
                        r"^\s*esp\s*,\s*\d+\s*$",
                        out[i + 1].operands or "", re.IGNORECASE,
                    )
                ):
                    body_start = i + 2
            elif (
                i < end
                and out[i].kind == "instr"
                and out[i].op == "enter"
            ):
                # enter K, 0 — accept.
                m_enter = re.match(
                    r"^\s*(\d+)\s*,\s*0\s*$",
                    out[i].operands or "", re.IGNORECASE,
                )
                if m_enter is None:
                    continue
                enter_idx = i
                body_start = i + 1
            else:
                continue
            # The function must end with `leave` then `ret` (with
            # optional blanks in between). Find them.
            j = end - 1
            while j >= body_start and out[j].kind in ("blank", "comment"):
                j -= 1
            if not (
                j >= body_start
                and out[j].kind == "instr"
                and out[j].op == "ret"
            ):
                continue
            ret_idx = j
            j -= 1
            while j >= body_start and out[j].kind in ("blank", "comment"):
                j -= 1
            if not (
                j >= body_start
                and out[j].kind == "instr"
                and out[j].op == "leave"
            ):
                continue
            leave_idx = j
            # Body: [body_start, leave_idx). Must be pure-leaf.
            if not self._body_is_pure_leaf_lines(
                out[body_start:leave_idx]
            ):
                continue
            # Body must reference [ebp + N] for at least one N (else
            # this is a no-param function and `skip_frame` already
            # caught it — re-running here would just re-emit). Bail
            # if no [ebp + N] references — saves no bytes.
            if not any(
                self._has_ebp_pos_offset(line)
                for line in out[body_start:leave_idx]
            ):
                continue
            # Build the rewrite: drop the prologue and `leave`.
            # Rewrite all body lines' [ebp + N] → [esp + (N - 4)].
            new_body = [
                self._rewrite_ebp_to_esp_in_line(line)
                for line in out[body_start:leave_idx]
            ]
            # The new function range:
            # [label] + (any blanks/comments before prologue) +
            # new_body + (any blanks between leave and ret + the ret).
            prologue_first = (
                push_idx if push_idx >= 0 else enter_idx
            )
            head = [out[start]] + out[start + 1:prologue_first]
            tail = out[leave_idx + 1:end]
            new_lines = head + new_body + tail
            replacements.append((start, end, new_lines))
            self.stats["pure_leaf_drop_frame"] = (
                self.stats.get("pure_leaf_drop_frame", 0) + 1
            )
        # Apply replacements in reverse so earlier indices remain valid.
        for start, end, new_lines in reversed(replacements):
            out = out[:start] + new_lines + out[end:]
        return out

    @staticmethod
    def _body_is_pure_leaf_lines(body: list[Line]) -> bool:
        """Check whether `body` (a list of Line objects) is a
        pure-leaf body — no ESP-modifying instructions, no calls,
        no [ebp - N] (locals) accesses, no SIB-form ebp accesses,
        no ebp writes / register-form refs."""
        DISALLOWED_OPS = {
            "push", "pop", "call", "enter", "leave",
            "pushf", "pushfd", "popf", "popfd",
            "pusha", "pushad", "popa", "popad",
            "ret", "retn", "retf", "iret", "iretd",
            "rep", "repe", "repz", "repne", "repnz",
            "loop", "loope", "loopne", "loopz", "loopnz",
            "int", "into",
        }
        for line in body:
            if line.kind != "instr":
                continue
            op = line.op
            if op in DISALLOWED_OPS:
                return False
            operands = line.operands or ""
            ops_lower = operands.lower()
            # `add esp, ...` / `sub esp, ...` / `mov esp, ...`
            if op in ("add", "sub") and re.match(
                r"^\s*esp\s*,", ops_lower,
            ):
                return False
            if op == "mov" and re.match(
                r"^\s*esp\s*,", ops_lower,
            ):
                return False
            if op == "xchg" and "esp" in ops_lower:
                return False
            # ebp written as a register (not via memory operand).
            if op == "mov" and re.match(
                r"^\s*ebp\s*,", ops_lower,
            ):
                return False
            if op in ("xor", "and", "or", "add", "sub", "imul") and re.match(
                r"^\s*ebp\s*,", ops_lower,
            ):
                return False
            # [ebp - N] (local access). Bail.
            if re.search(
                r"\[\s*ebp\s*-\s*\d+\s*\]", operands, re.IGNORECASE,
            ):
                return False
            # [ebp] / [ebp + 0] (saved-EBP slot from prologue). When
            # the prologue is dropped this slot ceases to exist —
            # the access would map to `[esp - 4]` (below the new
            # esp, undefined). Bail. Surfaced in uc386's libc
            # `_setjmp`, which reads `[ebp]` to capture the caller's
            # saved ebp into jmp_buf[12]; pre-fix the pass dropped
            # _setjmp's prologue but left the `[ebp]` access intact,
            # so jmp_buf[12] wound up holding the *caller's caller's*
            # ebp. longjmp then restored EBP to that wrong frame and
            # MicroPython's exception-traceback path crashed reading
            # `code_state->fun_bc->context` against the bogus ebp.
            if re.search(
                r"\[\s*ebp\s*(?:\+\s*0\s*)?\]", operands, re.IGNORECASE,
            ):
                return False
            # SIB-form ebp accesses.
            if re.search(
                r"\[\s*ebp\s*\+\s*\w+\s*\*",
                operands,
                re.IGNORECASE,
            ):
                return False
            # `ebp` as register operand (not in a memory operand).
            no_mem = re.sub(r"\[[^\]]*\]", "", operands)
            if re.search(r"\bebp\b", no_mem, re.IGNORECASE):
                return False
        return True

    @staticmethod
    def _has_ebp_pos_offset(line: Line) -> bool:
        if line.kind != "instr":
            return False
        return bool(re.search(
            r"\[\s*ebp\s*\+\s*\d+\s*\]",
            line.operands or "",
            re.IGNORECASE,
        ))

    def _pass_drop_dead_frame_alloc(
        self, lines: list[Line]
    ) -> list[Line]:
        """Slice Y v2: drop the stack-frame allocation when the
        function's body has zero `[ebp - N]` (local) accesses but
        does have calls. Keeps the `push ebp / mov ebp, esp` pair
        (so [ebp + N] param accesses survive across calls) but
        drops the `sub esp, K` (or rewrites `enter K, 0` to
        `push ebp; mov ebp, esp`).

        Pattern (per function):
            _fn:
                    push    ebp
                    mov     ebp, esp
                    sub     esp, K           ; ← drop this
                    <body — references [ebp + N] for params; may
                    have calls; NO [ebp - N] (locals dead)>
                    leave
                    ret
        Or:
            _fn:
                    enter   K, 0             ; ← rewrite to push/mov
                    <body>
                    leave
                    ret

        Saves 3 bytes per fire (sub esp K → 0) or 1 byte per fire
        (enter K, 0 = 4 bytes → push ebp; mov ebp, esp = 3 bytes).

        Sister of `pure_leaf_drop_frame`: that pass requires NO
        calls in body (so it can replace [ebp + N] with [esp + N-4]
        and drop the EBP frame entirely). This pass keeps EBP for
        param access so it works on non-leaf functions where
        peephole has eliminated all the local accesses.
        """
        out = list(lines)
        ranges = self._function_ranges(out)
        replacements: list[tuple[int, int, list[Line]]] = []
        for start, end in ranges:
            if (
                out[start].kind != "label"
                or out[start].label.startswith(".")
            ):
                continue
            i = start + 1
            while i < end and out[i].kind in ("blank", "comment"):
                i += 1
            # Three prologue forms:
            # 1. push ebp; mov ebp, esp; sub esp, K  → drop sub
            # 2. enter K, 0                          → rewrite to push/mov
            # Fast bail when neither matches.
            push_idx = -1
            mov_idx = -1
            sub_idx = -1
            enter_idx = -1
            if (
                i < end
                and out[i].kind == "instr"
                and out[i].op == "push"
                and out[i].operands.strip().lower() == "ebp"
            ):
                push_idx = i
                i += 1
                while i < end and out[i].kind in ("blank", "comment"):
                    i += 1
                if not (
                    i < end
                    and out[i].kind == "instr"
                    and out[i].op == "mov"
                    and out[i].operands.strip().lower() == "ebp, esp"
                ):
                    continue
                mov_idx = i
                i += 1
                while i < end and out[i].kind in ("blank", "comment"):
                    i += 1
                if not (
                    i < end
                    and out[i].kind == "instr"
                    and out[i].op == "sub"
                    and re.match(
                        r"^\s*esp\s*,\s*\d+\s*$",
                        out[i].operands or "", re.IGNORECASE,
                    )
                ):
                    continue
                sub_idx = i
                body_start = sub_idx + 1
            elif (
                i < end
                and out[i].kind == "instr"
                and out[i].op == "enter"
                and re.match(
                    r"^\s*\d+\s*,\s*0\s*$",
                    out[i].operands or "", re.IGNORECASE,
                )
            ):
                enter_idx = i
                body_start = enter_idx + 1
            else:
                continue
            # Function ends with `leave; ret`.
            j = end - 1
            while j >= body_start and out[j].kind in (
                "blank", "comment",
            ):
                j -= 1
            if not (
                j >= body_start
                and out[j].kind == "instr"
                and out[j].op == "ret"
            ):
                continue
            j -= 1
            while j >= body_start and out[j].kind in (
                "blank", "comment",
            ):
                j -= 1
            if not (
                j >= body_start
                and out[j].kind == "instr"
                and out[j].op == "leave"
            ):
                continue
            leave_idx = j
            # Body must have NO [ebp - N] accesses (locals dead).
            # Also no SIB-form ebp accesses (variable-offset, can't
            # reason about), and no [esp + N] accesses (would mean
            # the body uses ESP-relative scratch that depends on
            # the prologue's `sub esp` allocation — dropping the
            # alloc would shift those accesses to invalid memory).
            body_clean = True
            for k in range(body_start, leave_idx):
                ln = out[k]
                if ln.kind != "instr":
                    continue
                ops = ln.operands or ""
                if re.search(
                    r"\[\s*ebp\s*-\s*\d+\s*\]", ops, re.IGNORECASE,
                ):
                    body_clean = False
                    break
                if re.search(
                    r"\[\s*ebp\s*\+\s*\w+\s*\*", ops, re.IGNORECASE,
                ):
                    body_clean = False
                    break
                # `[esp + N]` (any positive offset) — body relies
                # on ESP relative to the prologue's allocated
                # scratch. Bail.
                if re.search(
                    r"\[\s*esp\s*\+\s*\w+", ops, re.IGNORECASE,
                ):
                    body_clean = False
                    break
                if re.search(
                    r"\[\s*esp\s*\]", ops, re.IGNORECASE,
                ):
                    body_clean = False
                    break
            if not body_clean:
                continue
            # Build the rewrite.
            if enter_idx >= 0:
                # Rewrite `enter K, 0` to `push ebp; mov ebp, esp`.
                indent = self._extract_indent(out[enter_idx].raw)
                push_line = Line(
                    raw=f"{indent}push    ebp",
                    kind="instr", op="push", operands="ebp",
                )
                mov_line = Line(
                    raw=f"{indent}mov     ebp, esp",
                    kind="instr", op="mov", operands="ebp, esp",
                )
                new_lines = (
                    out[start:enter_idx]
                    + [push_line, mov_line]
                    + out[enter_idx + 1:end]
                )
            else:
                # Drop the `sub esp, K` line.
                new_lines = (
                    out[start:sub_idx]
                    + out[sub_idx + 1:end]
                )
            replacements.append((start, end, new_lines))
            self.stats["drop_dead_frame_alloc"] = (
                self.stats.get("drop_dead_frame_alloc", 0) + 1
            )
        for start, end, new_lines in reversed(replacements):
            out = out[:start] + new_lines + out[end:]
        return out

    @staticmethod
    def _rewrite_ebp_to_esp_in_line(line: Line) -> Line:
        """Rewrite `[ebp + N]` to `[esp + (N - 4)]` in a single
        instruction's operands (and raw text). Other lines pass
        through unchanged."""
        if line.kind != "instr":
            return line
        def sub_one(m: re.Match) -> str:
            n = int(m.group(1))
            new_n = n - 4
            if new_n == 0:
                return "[esp]"
            return f"[esp + {new_n}]"
        new_operands = re.sub(
            r"\[\s*ebp\s*\+\s*(\d+)\s*\]",
            sub_one,
            line.operands or "",
            flags=re.IGNORECASE,
        )
        new_raw = re.sub(
            r"\[\s*ebp\s*\+\s*(\d+)\s*\]",
            sub_one,
            line.raw,
            flags=re.IGNORECASE,
        )
        return Line(
            raw=new_raw, kind="instr",
            op=line.op, operands=new_operands,
        )


def optimize(asm_text: str) -> str:
    """Convenience wrapper: run peephole and return the optimized asm."""
    return PeepholeOptimizer().optimize(asm_text)
