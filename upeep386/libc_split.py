"""Parser for `lib/i386_dos_libc.asm` that supports selective inclusion.

Every uc386-compiled binary today embeds the entire monolithic libc
asm — 3500+ lines, ~6 KB of code, regardless of whether the user
called `printf` or just `return 0;`. This module parses that asm into
per-function units with their dependency graph, then picks only the
functions transitively needed by the user code.

The model mirrors uc80's `runtime.py` but adapted to NASM conventions
(section .text/.data/.bss instead of CSEG/DSEG/COMMON, no PUBLIC, etc).

Usage:

    parsed = parse_libc(libc_text)
    needed = parsed.transitive_closure({"_printf"})
    minimal = parsed.emit(needed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Top-level function label: `_name:` at column 0, optionally followed
# by an instruction on the same line (e.g.,
# `___builtin_memcpy:        jmp _memcpy`). Accepts multi-underscore
# prefixes like `___builtin_*`.
_FUNC_LABEL_RE = re.compile(r"^(_[_A-Za-z0-9]+):(.*)$")

# Data/BSS label: `_name:` or `__name:` (single colon, optionally with
# data directive on the same line). Same pattern; we disambiguate by
# section context.
_DATA_LABEL_RE = _FUNC_LABEL_RE

# Local label: `.local_name:`. Belongs to the most recent top-level
# label; the optimizer doesn't need to track it separately.
_LOCAL_LABEL_RE = re.compile(r"^\.[A-Za-z0-9_]+:")

# Section directive: `section .text` / `section .data` / `section .bss`.
# Optionally indented, but in libc.asm they're at column 0.
_SECTION_RE = re.compile(r"^\s*section\s+(\.\w+)\s*$")


@dataclass
class LibcFunction:
    """A function definition extracted from libc.asm."""

    name: str
    """The exported label (e.g., `_printf`, `__builtin_memcpy`)."""

    source: list[str]
    """Lines of asm comprising this function. The first line is the
    `name:` label."""

    deps: set[str] = field(default_factory=set)
    """Direct references to other libc symbols (functions, globals,
    bss labels) found in this function's body. Used to compute the
    transitive closure."""


@dataclass
class LibcDataLabel:
    """A label in `.data` or `.bss`. Subject to the transitive-closure
    walk like functions — included only when reachable from a referenced
    symbol."""

    name: str
    section: str  # ".data" or ".bss"
    source: list[str]

    deps: set[str] = field(default_factory=set)
    """Direct references to other libc symbols. Data labels can
    reference others (e.g. `__heap_ptr: dd __heap`)."""


@dataclass
class ParsedLibc:
    """Parsed libc structure with dependency graph."""

    header: list[str]
    """Lines before the first label (comments, top-of-file docs, the
    initial `section .text` directive). Always emitted."""

    functions: dict[str, LibcFunction]
    """name → function. Includes the synthetic builtin trampolines
    like `___builtin_memcpy: jmp _memcpy`."""

    data_labels: dict[str, LibcDataLabel]
    """name → data/bss block. Always included unconditionally."""

    data_section_lines: list[str]
    """Raw `.data` section lines including its `section .data`
    directive. Emitted when any data label is needed."""

    bss_section_lines: list[str]
    """Raw `.bss` section lines. Always emitted."""

    def transitive_closure(self, initial: set[str]) -> set[str]:
        """Walk `deps` starting from `initial`, returning the set of
        all libc symbol names (functions + data/bss labels) transitively
        needed.

        `initial` is the set of names referenced by user code (e.g.,
        derived from `extern _printf` declarations). Names not present
        in `self.functions` or `self.data_labels` are ignored (probably
        user-defined or truly unused references).

        Data labels can themselves reference other libc symbols (e.g.
        `__heap_ptr: dd __heap` references `__heap`). Those references
        are tracked too.
        """
        needed: set[str] = set()
        worklist = [n for n in initial
                    if n in self.functions or n in self.data_labels]
        while worklist:
            name = worklist.pop()
            if name in needed:
                continue
            needed.add(name)
            if name in self.functions:
                deps = self.functions[name].deps
            elif name in self.data_labels:
                deps = self.data_labels[name].deps
            else:
                continue
            for dep in deps:
                if dep in needed:
                    continue
                if dep in self.functions or dep in self.data_labels:
                    worklist.append(dep)
        return needed

    def emit(self, needed: set[str]) -> str:
        """Produce the asm text containing only `needed` symbols
        (functions + data/bss labels).

        Functions are emitted in their original order from the source.
        Data and BSS sections are emitted only when they have at
        least one needed label.
        """
        out: list[str] = []
        out.extend(self.header)
        # Functions in original order.
        for name, fn in self.functions.items():
            if name in needed:
                out.extend(fn.source)
        # Data section: emit only needed labels.
        needed_data = [d for n, d in self.data_labels.items()
                       if d.section == ".data" and n in needed]
        if needed_data:
            out.append("        section .data")
            for d in needed_data:
                out.extend(d.source)
        # BSS section: emit only needed labels.
        needed_bss = [d for n, d in self.data_labels.items()
                      if d.section == ".bss" and n in needed]
        if needed_bss:
            out.append("        section .bss")
            for d in needed_bss:
                out.extend(d.source)
        return "\n".join(out) + "\n"


def parse_libc(text: str) -> ParsedLibc:
    """Split libc asm into per-function units with dep graphs.

    The libc.asm has this shape:
        ; comments
        section .text
        _func1:
            ; body
        _func2:
            ; body
        section .bss
        _data1: resb N
        section .data
        _data2: dd 0
        section .text
        _func3:
            ; body

    We collect:
    - Header (everything up to the first function label).
    - Functions (each `_name:` block of `.text` content).
    - Data section (`.data` + `.bss` content emitted as-is).

    Functions span from their label up to (but not including) the
    next top-level label OR a `section .data` / `section .bss`
    directive. A `section .text` directive inside the libc resumes
    function parsing at the next function label.
    """
    lines = text.splitlines()
    header: list[str] = []
    functions: dict[str, LibcFunction] = {}
    data_labels: dict[str, LibcDataLabel] = {}
    data_section: list[str] = []
    bss_section: list[str] = []

    state = "header"  # header → text → data → bss → text → ...
    current_func: LibcFunction | None = None
    current_data: LibcDataLabel | None = None

    def finalize_func() -> None:
        nonlocal current_func
        if current_func is not None:
            current_func.deps = _extract_deps(current_func.source)
            functions[current_func.name] = current_func
            current_func = None

    for line in lines:
        sec_match = _SECTION_RE.match(line)
        if sec_match:
            sec = sec_match.group(1)
            finalize_func()
            current_data = None
            if sec == ".text":
                state = "text"
                # Don't add this `section .text` to anything if we're
                # mid-libc — the header already has the initial one.
                if not header or all(
                    not _SECTION_RE.match(l) for l in header
                ):
                    header.append(line)
                # If we already have a header section directive, this
                # new `section .text` (after data/bss) doesn't need to
                # be emitted again — NASM tolerates section toggling.
                continue
            if sec == ".data":
                state = "data"
                data_section.append(line)
                continue
            if sec == ".bss":
                state = "bss"
                bss_section.append(line)
                continue

        if state == "header":
            # Headers: comments, blanks, the initial `section .text`.
            # First non-comment / non-blank / non-section line that's
            # a function label flips us into text mode.
            stripped = line.strip()
            if stripped and not stripped.startswith(";"):
                m = _FUNC_LABEL_RE.match(line)
                if m:
                    state = "text"
                    # Fall through to the text branch below.
                else:
                    # Could be the implicit `section .text` (handled
                    # by _SECTION_RE above) — leave for now.
                    header.append(line)
                    continue
            else:
                header.append(line)
                continue

        if state == "text":
            m = _FUNC_LABEL_RE.match(line)
            if m:
                finalize_func()
                name = m.group(1)
                current_func = LibcFunction(name=name, source=[line])
                continue
            if current_func is not None:
                current_func.source.append(line)
                continue
            # Text content before any function (rare — typically just
            # blank lines after `section .text`).
            header.append(line)
            continue

        if state == "data":
            data_section.append(line)
            m = _DATA_LABEL_RE.match(line)
            if m:
                name = m.group(1)
                current_data = LibcDataLabel(name=name, section=".data",
                                              source=[line])
                data_labels[name] = current_data
            elif current_data is not None:
                current_data.source.append(line)
            continue

        if state == "bss":
            bss_section.append(line)
            m = _DATA_LABEL_RE.match(line)
            if m:
                name = m.group(1)
                current_data = LibcDataLabel(name=name, section=".bss",
                                              source=[line])
                data_labels[name] = current_data
            elif current_data is not None:
                current_data.source.append(line)
            continue

    finalize_func()
    # Extract deps for each data label.
    for label in data_labels.values():
        label.deps = _extract_deps(label.source)
        label.deps.discard(label.name)  # don't self-reference
    return ParsedLibc(
        header=header,
        functions=functions,
        data_labels=data_labels,
        data_section_lines=data_section,
        bss_section_lines=bss_section,
    )


# References to libc symbols inside a function body. Captures:
# - `call _name`  (direct call)
# - `jmp _name` (used by builtin trampolines)
# - `mov reg, _name` (function-pointer / data-address load)
# - `lea reg, [_name]` (likewise)
# - `push _name` (likewise)
# - `mov reg, [_name]` (data load)
# - `[_name]` operand (memory ref)
#
# We capture any bareword starting with `_` that looks like a label.
# This is over-approximate (it'll match e.g. `mov eax, 0x4C00` if we
# weren't careful), but `_` is reliable: numeric literals don't start
# with underscore.
_DEP_REFERENCE_RE = re.compile(r"\b(_[_A-Za-z][A-Za-z0-9_]*)\b")


def _extract_deps(source: list[str]) -> set[str]:
    """Find references to other libc-style labels inside a function
    body (excluding the function's own label and standard registers).

    Comments are stripped first so e.g. `; calls _printf` doesn't
    create a false dependency."""
    deps: set[str] = set()
    if not source:
        return deps
    own_name = ""
    m = _FUNC_LABEL_RE.match(source[0])
    if m:
        own_name = m.group(1)
    for line in source:
        # Strip inline comments.
        idx = line.find(";")
        if idx >= 0:
            line = line[:idx]
        for tok in _DEP_REFERENCE_RE.findall(line):
            if tok == own_name:
                continue
            # Skip local labels (would have been `.foo`, but our regex
            # only catches `_foo`-style — local labels are filtered).
            deps.add(tok)
    return deps
