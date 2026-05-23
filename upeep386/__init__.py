"""upeep386 — i386 NASM optimizer toolkit.

Language-agnostic NASM-text optimizer for compilers targeting flat-32
i386 / x86-32. Consumed by sibling compilers `uc386` (C23) and
`ucpp386` (C++). Mirrors the upeepz80 / upeep80 split — once the
pattern set crystallizes inside a host compiler, the optimizer
becomes a standalone package shared across compilers for the same
target.

Three composable passes, run in order by the host compiler:

    from upeep386 import optimize, dce, parse_libc

    asm = compiler.emit(...)
    asm = optimize(asm)          # peephole patterns to fixed point
    asm = dce(asm)               # call-graph + dead-label DCE
    # ... emit final asm; the host's libc may use parse_libc to
    # do per-function selective inclusion at bundling time.

Three modules:

* `peephole` — pattern-based NASM rewrites (binop_collapse,
  setcc_jcc_collapse, mov_zero_to_xor, …). Public API:
  `optimize(asm)` returns optimized text;
  `PeepholeOptimizer()` is the OO entry point with `.stats`.
* `asm_dce` — call/reference-graph dead-code elimination from
  `_start`/`_main`. Public API: `dce(asm)`; `parse_asm(asm)` for
  the raw reachability set.
* `libc_split` — parse a monolithic libc.asm into per-function
  units with their dependency graph; emit only the transitive
  closure of seed symbols. Public API: `parse_libc(asm)`.

The input shape is what uc386 / ucpp386 `CodeGenerator.generate`
emits: NASM `section .text`, `bits 32`, top-level `_name:`
labels, cdecl calling convention.
"""

__version__ = "0.2.0"
__author__ = "upeep386 project"

from .peephole import (
    PeepholeOptimizer,
    optimize,
)
from .asm_dce import (
    ParsedAsm,
    dce,
    parse_asm,
)
from .libc_split import (
    LibcDataLabel,
    LibcFunction,
    ParsedLibc,
    parse_libc,
)

__all__ = [
    # peephole
    "PeepholeOptimizer",
    "optimize",
    # asm_dce
    "ParsedAsm",
    "dce",
    "parse_asm",
    # libc_split
    "LibcDataLabel",
    "LibcFunction",
    "ParsedLibc",
    "parse_libc",
    "__version__",
]
