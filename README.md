# upeep386 — i386 NASM optimizer toolkit

Language-agnostic NASM-text optimizer for compilers targeting the
**Intel 386 (x86-32) processor**. Three composable passes:

* **peephole** — pattern-based instruction rewriting to fixed point
  (binop_collapse, setcc_jcc_collapse, mov_zero_to_xor, …).
* **asm_dce** — call/reference-graph dead-code elimination,
  walking reachability from `_start` / `_main` over the call
  graph of top-level functions and data labels.
* **libc_split** — parser for a monolithic libc.asm with per-
  function dep tracking, so a host compiler can include only the
  transitive closure of routines the user code references.

Sibling to [upeepz80](https://github.com/avwohl/upeepz80) (Z80
peephole) and [upeep80](https://github.com/avwohl/upeep80) (8080
peephole); same split pattern — once the optimizer's pattern set
crystallized inside a host compiler, it became a standalone package
shared across compilers for the same target.

## Origin

Extracted from [uc386](https://github.com/avwohl/uc386) (the C23 →
i386/DOS compiler) for shared use by
[ucpp386](https://github.com/avwohl/ucpp386) (the C++ sibling).
Both compilers emit NASM in the same dialect and benefit from the
same rewrites — vtables and templates in ucpp386 make the asm-DCE
cascade and the inline-load collapses *more* important, not less,
than they are for C.

## Install

	pip install upeep386

Or from source:

	pip install -e .

## Quick start

	from upeep386 import optimize, dce, parse_libc

	asm = compiler.emit(...)
	asm = optimize(asm)          # peephole patterns to fixed point
	asm = dce(asm)               # drop unreachable functions / labels
	# parse_libc(libc_asm) lets the host include only the libc
	# routines the user actually references.

## Pattern list (current)

Runs to fixed point. Patterns the optimizer applies:

- `dead_after_terminator` — drop instructions between unconditional
  `jmp`/`ret` and the next label/directive.
- `jmp_to_next_label` — drop `jmp X` immediately followed by `X:`.
- `binop_collapse` — replace 4-line stack-machine right-operand
  transfer with a single `mov ecx, src`.
- `store_collapse` — drop push/pop pair around a store when src is
  a single-instruction load.
- `leave_collapse` — `mov esp, ebp; pop ebp` → `leave`.
- `imm_store_collapse` — `mov eax, IMM; mov [addr], eax; mov eax, X`
  → `mov dword [addr], IMM; mov eax, X`.
- `setcc_jcc_collapse` — drop the `setCC al; movzx eax, al; test
  eax, eax; jz/jnz` boolean-then-branch sequence into a single
  conditional jump.
- `push_immediate` — `mov eax, IMM; push eax` → `push IMM` when the
  next instruction overwrites EAX.
- `ecx_binop_collapse` — `mov ecx, src; OP eax, ecx` → `OP eax, src`.
- `mov_zero_to_xor` — `mov reg, 0` → `xor reg, reg` (when flag-safe).
- `store_load_collapse` — drop the redundant load after a store of
  the same register to the same address.

See `upeep386/peephole.py` docstring for the complete pattern
description and bytes saved per pattern.

## Input contract

Input is exactly what uc386 / ucpp386 emit. The optimizer assumes:

- Section directives at column 0: `section .text` / `section .data`
  / `section .bss`.
- Top-level labels at column 0: `_name:` (multi-underscore prefixes
  allowed for `__start`, `___builtin_*`).
- Local labels inside functions: `.local:`.
- cdecl calling convention; `EAX` is the return register; `EBP` is
  the frame pointer.
- No macro expansion required at this layer — input is post-
  preprocessor literal NASM.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
