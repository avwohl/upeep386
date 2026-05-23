"""upeep386 — peephole optimizer for i386 NASM assembly.

Language-agnostic NASM-text rewriter for compilers targeting flat-32
i386 / x86-32. Consumed by sibling compilers `uc386` (C23) and
`ucpp386` (C++). Mirrors the upeepz80 / upeep80 split — once the
pattern set crystallizes inside a host compiler, the optimizer
becomes a standalone package shared across compilers for that
target.

Public API:

    from upeep386 import optimize          # returns optimized asm text
    from upeep386 import PeepholeOptimizer # OO entry point with .stats

The input shape is what uc386/ucpp386 CodeGenerator.generate emits:
NASM `section .text`, `bits 32`, top-level `_name:` labels, cdecl
calling convention. See peephole.py docstring for the full pattern
list.
"""

__version__ = "0.1.0"
__author__ = "upeep386 project"

from .peephole import (
    PeepholeOptimizer,
    optimize,
)

__all__ = [
    "PeepholeOptimizer",
    "optimize",
    "__version__",
]
