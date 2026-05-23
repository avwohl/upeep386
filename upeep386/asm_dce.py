"""Dead-code elimination on uc386 codegen output.

Walks reachability from the entry points (``_start`` / ``_main`` / any
``global _name``) over the call/reference graph of top-level functions
and data labels. Functions and data labels not reachable get dropped.

This catches:
- Top-level functions whose only callers were themselves removed by
  uc_core's AST optimizer (cascade through the call graph).
- ``static`` functions that the AST optimizer's per-translation-unit
  unreferenced-decl pass missed (e.g., a static called only from a
  removed branch).
- Data labels (`.data` / `.bss`) referenced only by dead functions.

Intra-function dead-code (unreachable blocks within a function) is
NOT handled here — the existing peephole passes (``dead_after_terminator``,
``unreferenced_label_removal``) cover that.

The expected input shape is exactly what ``CodeGenerator.generate``
emits — see the codegen for the layout. Header comments, ``bits 32``,
``section .text``, ``global _start``, then ``_start:`` / ``__start:``,
then user functions, then ``.data`` / ``.bss`` sections.

Public API:

    parsed = parse_asm(text)
    reachable = parsed.reachable_set()
    minimal = parsed.emit(reachable)

Or shorthand:

    dce_text = dce(text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Top-level label: `_name:` at column 0, possibly followed by an
# instruction on the same line. Accepts multi-underscore prefixes
# like `__start`, `___builtin_*`.
_TOP_LABEL_RE = re.compile(r"^(_[_A-Za-z0-9]+):(.*)$")

# Section directive.
_SECTION_RE = re.compile(r"^\s*section\s+(\.\w+)\s*$")

# `global _name` directive.
_GLOBAL_RE = re.compile(r"^\s*global\s+(_[_A-Za-z0-9]+)\s*$")

# Symbol references inside instruction operands or data initializers.
# Match identifiers starting with `_` followed by ident chars. Excludes
# refs inside comments.
_SYMBOL_REF_RE = re.compile(r"\b(_[_A-Za-z0-9]+)\b")


@dataclass
class AsmFunction:
    """A function definition in the user codegen."""

    name: str
    source: list[str]  # lines, including the `name:` label line
    deps: set[str] = field(default_factory=set)


@dataclass
class AsmData:
    """A data/bss label."""

    name: str
    section: str  # ".data" or ".bss"
    source: list[str]
    deps: set[str] = field(default_factory=set)


@dataclass
class ParsedAsm:
    """Parsed user codegen with dependency graph.

    The header includes top-of-file directives (``bits 32``,
    ``section .text``, ``global _start``) and any leading comments.
    The header is always emitted unchanged.

    ``entry_points`` is the set of names that are roots for the
    reachability walk: ``_start`` (the driver entry), ``_main`` (the
    user's entry), plus any name that appears in a ``global`` directive
    in the header.
    """

    header: list[str]
    functions: dict[str, AsmFunction]
    data_labels: dict[str, AsmData]
    data_section_header: list[str]
    """Lines that precede the first data-section label (the
    `section .data` directive)."""
    bss_section_header: list[str]
    """Lines that precede the first bss-section label (the
    `section .bss` directive)."""
    trailer: list[str]
    """Anything after the last recognized block (typically empty)."""
    entry_points: set[str]

    def reachable_set(self) -> set[str]:
        """Return the names of all functions and data labels reachable
        from the entry points via the dep graph."""
        reached: set[str] = set()
        worklist = [
            n for n in self.entry_points
            if n in self.functions or n in self.data_labels
        ]
        while worklist:
            name = worklist.pop()
            if name in reached:
                continue
            reached.add(name)
            if name in self.functions:
                deps = self.functions[name].deps
            elif name in self.data_labels:
                deps = self.data_labels[name].deps
            else:
                continue
            for dep in deps:
                if dep in reached:
                    continue
                if dep in self.functions or dep in self.data_labels:
                    worklist.append(dep)
        return reached

    def emit(self, reachable: set[str]) -> str:
        """Produce asm text containing only reachable functions and
        data labels."""
        # Names referenced from anything we're emitting — used to
        # decide which `extern _name` declarations to keep. Includes
        # `reachable` itself (a fn may be both defined and self-ref'd
        # in deps) plus the union of deps of every reachable item.
        referenced: set[str] = set(reachable)
        for name in reachable:
            if name in self.functions:
                referenced.update(self.functions[name].deps)
            elif name in self.data_labels:
                referenced.update(self.data_labels[name].deps)

        out: list[str] = []
        # Filter `extern _name` lines: keep only those referenced from
        # something we're actually emitting. Without this, declarations
        # pulled in by header transitive includes (math.h declares
        # ~150 extern functions; awk uses 4) bloat the asm and trip
        # nasm's `-f bin` "binary output format does not support external
        # references" because the bundled libc doesn't define them.
        for line in self.header:
            stripped = line.strip()
            if stripped.startswith("extern "):
                name = stripped[7:].strip().rstrip(",")
                if name not in referenced:
                    continue
            out.append(line)
        for name, fn in self.functions.items():
            if name in reachable:
                out.extend(fn.source)
        # Data section: emit only when at least one needed label.
        needed_data = [
            (n, d) for n, d in self.data_labels.items()
            if d.section == ".data" and n in reachable
        ]
        if needed_data:
            out.extend(self.data_section_header)
            for _, d in needed_data:
                out.extend(d.source)
        # BSS section: emit only when at least one needed label OR
        # when the BSS-zero-init labels (`_bss_zero_start` /
        # `_bss_zero_end`) are present (the runtime always references
        # them when BSS init is enabled).
        needed_bss = [
            (n, d) for n, d in self.data_labels.items()
            if d.section == ".bss" and n in reachable
        ]
        if needed_bss:
            out.extend(self.bss_section_header)
            for _, d in needed_bss:
                out.extend(d.source)
        out.extend(self.trailer)
        # Preserve trailing newline.
        return "\n".join(out) + "\n" if out and out[-1] != "" else "\n".join(out)


def parse_asm(text: str) -> ParsedAsm:
    """Parse user codegen output.

    Recognizes:
    - Header (everything before the first top-level label).
    - Functions (each ``_name:`` block in ``.text``).
    - Data labels (each ``_name:`` block in ``.data`` / ``.bss``).

    A top-level block runs from its label until the next top-level
    label OR a section directive.
    """
    lines = text.splitlines()
    n = len(lines)
    i = 0

    # Phase 1: header — until we see the first top-level label
    # (which is ``_start:``) inside the .text section.
    header: list[str] = []
    section = ""
    entry_points: set[str] = {"_start", "_main"}

    # Look for the first label that's not inside a section header
    # block; everything before is header.
    while i < n:
        line = lines[i]
        m_sec = _SECTION_RE.match(line)
        if m_sec is not None:
            section = m_sec.group(1)
            header.append(line)
            i += 1
            continue
        m_glob = _GLOBAL_RE.match(line)
        if m_glob is not None:
            entry_points.add(m_glob.group(1))
            header.append(line)
            i += 1
            continue
        # If we see a top-level label, stop building header.
        if section == ".text" and _TOP_LABEL_RE.match(line):
            break
        header.append(line)
        i += 1

    functions: dict[str, AsmFunction] = {}
    data_labels: dict[str, AsmData] = {}
    data_section_header: list[str] = []
    bss_section_header: list[str] = []

    # Phase 2: walk through the rest collecting blocks.
    pending_section_header: list[str] | None = None
    while i < n:
        line = lines[i]
        m_sec = _SECTION_RE.match(line)
        if m_sec is not None:
            section = m_sec.group(1)
            if section == ".data":
                pending_section_header = data_section_header
                pending_section_header.append(line)
            elif section == ".bss":
                pending_section_header = bss_section_header
                pending_section_header.append(line)
            else:
                pending_section_header = None
            i += 1
            continue
        m_glob = _GLOBAL_RE.match(line)
        if m_glob is not None:
            # NASM `global _name` declarations can appear anywhere.
            # Collect them as entry points but otherwise drop the line
            # (they're metadata, not code; we re-emit `global _start`
            # via the header already).
            entry_points.add(m_glob.group(1))
            i += 1
            continue
        m_lbl = _TOP_LABEL_RE.match(line)
        if m_lbl is None:
            # Stray line. If section is .data/.bss and we haven't seen
            # a label there yet, append to the section header. In .text
            # this would be unusual; ignore for now.
            if pending_section_header is not None:
                pending_section_header.append(line)
            elif section == ".text":
                # Could be a blank line between functions; absorb into
                # whatever follows. Drop it for now (it'll appear before
                # the next function as part of parsing context).
                pass
            i += 1
            continue
        # Found a top-level label.
        name = m_lbl.group(1)
        # Collect this block: from this line until the next top-level
        # label or section directive.
        block: list[str] = [line]
        i += 1
        while i < n:
            ln = lines[i]
            if (
                _TOP_LABEL_RE.match(ln)
                or _SECTION_RE.match(ln)
                or _GLOBAL_RE.match(ln)
            ):
                break
            block.append(ln)
            i += 1
        # Skip trailing pure-blank lines that belong to the gap before
        # the next block (we'll let the next block carry them, or drop).
        # Actually keep them — they're part of this block's tail.
        deps = _extract_deps(block, name)
        if section == ".text":
            # `_start` and `__start` may share a body — `__start:` is
            # an alias right under `_start:`. Treat them both as the
            # same block keyed by `_start`. Detect this by checking
            # if the previous line just opened the `_start:` block.
            if name == "__start" and "_start" in functions:
                # Append to _start's source.
                functions["_start"].source.extend(block)
                # Don't merge deps — we want the body's references.
                functions["_start"].deps |= deps
            else:
                if name in functions:
                    # Duplicate function definition — append (defensive).
                    functions[name].source.extend(block)
                    functions[name].deps |= deps
                else:
                    functions[name] = AsmFunction(
                        name=name, source=block, deps=deps,
                    )
        elif section in (".data", ".bss"):
            if name in data_labels:
                data_labels[name].source.extend(block)
                data_labels[name].deps |= deps
            else:
                data_labels[name] = AsmData(
                    name=name, section=section, source=block, deps=deps,
                )

    # `_start` is always an entry point; ensure it's there if the asm
    # has it. `_main` is also an entry by convention (driver guarantees
    # it).
    if "_start" in functions:
        entry_points.add("_start")

    # Reserved entry points: BSS-zero-init labels (`_bss_zero_start` /
    # `_bss_zero_end`) referenced by `_start`'s prologue when BSS init
    # is enabled. They appear in `_start`'s deps anyway.

    # Address-alias detection: when a function's source is just its
    # label line (no instructions, no data — it falls through to the
    # NEXT function's body), treat it as an alias for that next
    # function. The trampoline pattern this handles:
    #     ___builtin_snprintf:
    #             jmp _snprintf
    # When peephole's `jmp_to_next_label` collapses the `jmp`
    # because `_snprintf` is now adjacent (after asm DCE drops
    # intervening libc functions), we get:
    #     ___builtin_snprintf:
    #     _snprintf:
    #             push ebp ...
    # `___builtin_snprintf:` is now a label-only definition that
    # aliases `_snprintf`. Without this fix, asm DCE would drop
    # `_snprintf` (no deps inbound) and leave `___builtin_snprintf`
    # falling into whatever's next — the wrong function.
    func_names = list(functions.keys())
    for k, name in enumerate(func_names):
        fn = functions[name]
        # Body-is-empty: only the label line(s), no instruction
        # lines. The label itself is `name:` possibly followed
        # by whitespace; everything else should be blank or
        # comment.
        has_instruction = False
        for line in fn.source:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            # Strip trailing comments.
            comment_idx = stripped.find(";")
            if comment_idx >= 0:
                stripped = stripped[:comment_idx].rstrip()
            if not stripped:
                continue
            # The label line itself: `name:` (possibly with the
            # `:` followed by whitespace). NASM also allows
            # `name:    instruction` on the same line; that has
            # an instruction.
            m_self = _TOP_LABEL_RE.match(stripped)
            if m_self is not None:
                # Check whether anything follows the colon.
                rest = m_self.group(2).strip()
                if rest:
                    has_instruction = True
                    break
                # else just the label — no instruction
                continue
            # Non-label, non-comment, non-blank line: it's an
            # instruction or directive.
            has_instruction = True
            break
        if has_instruction:
            continue
        # Empty-body function. If there's a next function, it's
        # the alias target.
        if k + 1 < len(func_names):
            next_name = func_names[k + 1]
            fn.deps.add(next_name)

    return ParsedAsm(
        header=header,
        functions=functions,
        data_labels=data_labels,
        data_section_header=data_section_header,
        bss_section_header=bss_section_header,
        trailer=[],
        entry_points=entry_points,
    )


def _extract_deps(source: list[str], self_name: str) -> set[str]:
    """Find symbol references in the function body. Skips:
    - The function's own label line (its leading ``_name:``).
    - Comments (``;`` to end of line).
    - Local labels (``.LX:``) — those are intra-function.
    """
    deps: set[str] = set()
    for line in source:
        # Strip any comment.
        comment_idx = line.find(";")
        if comment_idx >= 0:
            line = line[:comment_idx]
        # Strip a leading top-level label that matches self_name.
        # (We don't want to pick up `_helper_used:` as a dep when we're
        # parsing _helper_used itself.)
        m_lbl = _TOP_LABEL_RE.match(line)
        if m_lbl is not None and m_lbl.group(1) == self_name:
            line = m_lbl.group(2)
        for m in _SYMBOL_REF_RE.finditer(line):
            ref = m.group(1)
            if ref != self_name:
                deps.add(ref)
    return deps


def dce(text: str) -> str:
    """One-shot: parse, walk, emit."""
    parsed = parse_asm(text)
    reachable = parsed.reachable_set()
    return parsed.emit(reachable)
