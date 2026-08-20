# Changelog

All notable changes to upeep386 are documented here.

## 0.2.1 — 2026-08-20

Two peephole miscompiles, both in `upeep386/peephole.py`. Both are reached by
asm shapes a C or C++ front end does not normally emit but a hand-written libc
does: an 80-bit `long double` load, and a helper that takes its arguments in
registers rather than on the stack. Neither bug bites ordinary cdecl compiler
output — but the second fix changes what that output optimizes to. Callee
liveness is now read out of the callee's own body instead of assumed from the
calling convention, and where the analysis cannot prove a caller-saved register
is dead it keeps it live, so some folds across calls no longer fire. The output
is still correct; it can come out a few instructions longer. A 1,171-line
uc386 codegen dump optimized to 960 lines under 0.2.0 and to 964 under 0.2.1.

### Fixed
- **`_pass_fpu_op_collapse` no longer collapses an 80-bit load into an x87
  arithmetic instruction.** The pass rewrote `fld <mem>` followed by
  `faddp`/`fmulp`/`fsubp`/`fdivp`/`fsubrp`/`fdivrp st1, st0` into the single
  memory form (`fadd <mem>` and so on) without looking at the operand size.
  `fld` has an m80fp form — `fld tword [esi]` — but no x87 arithmetic
  instruction has one; FADD/FMUL/FSUB/FDIV take m32fp or m64fp only. So
  `fld tword [esi]; fmulp st1, st0` was rewritten to `fmul tword [esi]`, which
  has no encoding, and NASM rejected the whole file with "invalid operand
  sizes". This produced a failed build rather than wrong bytes, but it took the
  entire translation unit with it: the case that surfaced it is `_pf_pow10` in
  ucpp386's hand-written `libc_min.asm`, which walks a table of 80-bit long
  doubles in exactly this shape, so any build pulling in the printf float path
  stopped assembling. A `tword` operand on the `fld` now suppresses the collapse
  (`_is_m80_operand`); 32- and 64-bit sources collapse as before.

- **A `call` is no longer assumed to clobber EAX/ECX/EDX when the callee reads
  them first.** `_reg_dead_after` and `_is_pure_reg_write` — the liveness
  helpers that some seventy rewrite sites across the pass set consult — treated
  any direct call as killing the caller-saved registers. That holds for cdecl,
  which passes arguments on the stack, but not for a callee with a
  register-passing convention, and the register the caller had just loaded was
  then treated as dead and its setup folded away. For `libc_min.asm`'s
  `_pf_field(eax = text, ecx = length, edx = prefix length)`, this

  ```asm
          mov     eax, _pf_scratch
          mov     ecx, ebx
          sub     ecx, eax
          call    _pf_field
  ```

  became

  ```asm
          mov     ecx, ebx
          sub     ecx, _pf_scratch
          call    _pf_field
  ```

  — the `mov eax, _pf_scratch` folded into the `sub` and disappeared, leaving
  `_pf_field` to read whatever EAX happened to hold. The result assembles and
  links; it reads a wild address at run time, reported downstream as any
  `printf` of a number faulting. `_pf_outn(esi = ptr, ecx = count)` and
  `_pf_outrep(dl = char, ecx = count)` have the same shape.

  The previous guard was `_CALLS_READ_EAX`, a hardcoded list of five uc386
  symbols covering EAX only, so it could not express an ECX or EDX argument at
  all. It is superseded by `_compute_call_live_in`, which derives the answer
  from the asm being optimized: for every direct `call` target whose body is
  present in the text, which of EAX/ECX/EDX does the callee read before writing?
  Every target starts fully live and a register is dropped only once the body
  provably writes it before any read, so the initial state and every refinement
  round are over-approximations and the loop (capped at
  `_CALL_LIVE_IN_ROUNDS = 3`) can be cut short without becoming unsafe — only
  less precise. The map is rebuilt before each sweep of the pass set, since an
  earlier sweep can rewrite a callee body and change what it reads on entry. A
  call whose target is not defined in the text — an `extern` — is absent
  from the map and keeps the cdecl rule; a target that is defined in the text
  is analyzed, ordinary compiler output included. `_CALLS_READ_EAX` is retained
  as an override on top of the computed map.

  The practical cost is conservatism. A fold across a call into a
  register-reading callee no longer fires — that fold was the bug — and
  neither does one across a call the analysis cannot clear, which includes
  ordinary cdecl functions. Anything the forward scan cannot walk leaves the
  register assumed live: a loop, a branch it will not follow, or a body longer
  than its twenty-instruction window. In a uc386 codegen dump of a small C
  program, `_str_len`, `_sum_range`, `_fill` and `_main` each came back with a
  register spuriously live-in on that account, and the optimized output grew by
  four instructions. A suppressed fold also changes what later passes see, so a
  difference can go the other way: on `libc_min.asm`, 0.2.1 rewrites
  `xor eax, eax; mov [mem], al` to `mov byte [mem], 0` in eight places where
  0.2.0 left the pair standing.

### Added
- **Four regression tests in `tests/test_peephole.py`.** An 80-bit `fld` that
  must not collapse, a call into an EAX-reading callee whose argument setup must
  survive, the mirror case of a cdecl callee that writes EAX before reading it
  (where the fold must still fire, so the new analysis is not a blanket "assume
  live"), and an `_pf_outrep`-shaped callee reading ECX and EDX, which the old
  EAX-only allowlist could not describe.

### Changed
- **`PeepholeOptimizer._is_pure_reg_write` is an instance method rather than a
  static one**, since it now consults the per-instance live-in map. The public
  surface is unchanged: `optimize`, `PeepholeOptimizer`, `dce`, `parse_asm` and
  `parse_libc` keep their signatures, and only code reaching into the private
  helpers is affected.

This file starts at 0.2.1; for anything before that, see the git log.
