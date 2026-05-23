"""Tests for the libc asm parser and selective inclusion."""

from upeep386.libc_split import (
    LibcFunction,
    ParsedLibc,
    _extract_deps,
    parse_libc,
)


# ── parse_libc: basic structure ──────────────────────────────────


def test_parse_simple_two_functions():
    text = (
        "; libc header\n"
        "        section .text\n"
        "_foo:\n"
        "        ret\n"
        "_bar:\n"
        "        ret\n"
    )
    parsed = parse_libc(text)
    assert "_foo" in parsed.functions
    assert "_bar" in parsed.functions
    assert len(parsed.functions["_foo"].source) == 2  # label + ret
    assert len(parsed.functions["_bar"].source) == 2


def test_parse_separates_data_section():
    text = (
        "        section .text\n"
        "_func:\n"
        "        ret\n"
        "section .data\n"
        "_global1: dd 0\n"
        "section .text\n"
        "_func2:\n"
        "        ret\n"
    )
    parsed = parse_libc(text)
    assert "_func" in parsed.functions
    assert "_func2" in parsed.functions
    assert "_global1" in parsed.data_labels
    # Data section is preserved verbatim.
    assert any(".data" in l for l in parsed.data_section_lines)
    assert any("_global1" in l for l in parsed.data_section_lines)


def test_parse_separates_bss_section():
    text = (
        "        section .text\n"
        "_func:\n"
        "        ret\n"
        "section .bss\n"
        "_buffer: resb 256\n"
    )
    parsed = parse_libc(text)
    assert "_buffer" in parsed.data_labels
    assert parsed.data_labels["_buffer"].section == ".bss"
    assert any("resb 256" in l for l in parsed.bss_section_lines)


def test_parse_handles_inline_label_and_instruction():
    """Trampoline-style: `_alias: jmp _target` on one line."""
    text = (
        "        section .text\n"
        "___builtin_memcpy:        jmp _memcpy\n"
        "_memcpy:\n"
        "        ret\n"
    )
    parsed = parse_libc(text)
    assert "___builtin_memcpy" in parsed.functions
    assert "_memcpy" in parsed.functions
    # Trampoline depends on _memcpy.
    assert "_memcpy" in parsed.functions["___builtin_memcpy"].deps


# ── _extract_deps ────────────────────────────────────────────────


def test_extract_deps_finds_call():
    src = ["_foo:", "        call    _bar", "        ret"]
    deps = _extract_deps(src)
    assert deps == {"_bar"}


def test_extract_deps_finds_data_load():
    src = ["_foo:", "        mov     eax, [_stdout]", "        ret"]
    deps = _extract_deps(src)
    assert "_stdout" in deps


def test_extract_deps_finds_jmp_trampoline():
    src = ["_alias:        jmp _target"]
    deps = _extract_deps(src)
    assert deps == {"_target"}


def test_extract_deps_skips_self_reference():
    """A function's own label shouldn't count as a dependency
    (otherwise every function would depend on itself, breaking the
    closure walk in subtle ways)."""
    src = [
        "_loop:",
        "        dec     ecx",
        "        jnz     _loop",
        "        ret",
    ]
    deps = _extract_deps(src)
    assert "_loop" not in deps


def test_extract_deps_ignores_inline_comments():
    src = ["_foo:", "        ret             ; calls _bar internally"]
    deps = _extract_deps(src)
    # Comment mentions `_bar` but that's not a real dependency.
    assert "_bar" not in deps


# ── transitive_closure ───────────────────────────────────────────


def test_transitive_closure_simple_chain():
    parsed = ParsedLibc(
        header=[],
        functions={
            "_a": LibcFunction("_a", source=[], deps={"_b"}),
            "_b": LibcFunction("_b", source=[], deps={"_c"}),
            "_c": LibcFunction("_c", source=[], deps=set()),
            "_unused": LibcFunction("_unused", source=[], deps=set()),
        },
        data_labels={},
        data_section_lines=[],
        bss_section_lines=[],
    )
    needed = parsed.transitive_closure({"_a"})
    assert needed == {"_a", "_b", "_c"}
    assert "_unused" not in needed


def test_transitive_closure_handles_cycles():
    """Mutually-recursive functions don't loop forever."""
    parsed = ParsedLibc(
        header=[],
        functions={
            "_a": LibcFunction("_a", source=[], deps={"_b"}),
            "_b": LibcFunction("_b", source=[], deps={"_a"}),
        },
        data_labels={},
        data_section_lines=[],
        bss_section_lines=[],
    )
    needed = parsed.transitive_closure({"_a"})
    assert needed == {"_a", "_b"}


def test_transitive_closure_includes_data_labels():
    """Data labels are now part of the closure — `_printf` references
    `_stdout`, so `_stdout` is needed."""
    from upeep386.libc_split import LibcDataLabel
    parsed = ParsedLibc(
        header=[],
        functions={
            "_printf": LibcFunction("_printf", source=[], deps={"_stdout"}),
        },
        data_labels={
            "_stdout": LibcDataLabel("_stdout", section=".data", source=[]),
        },
        data_section_lines=[],
        bss_section_lines=[],
    )
    needed = parsed.transitive_closure({"_printf"})
    assert needed == {"_printf", "_stdout"}


def test_transitive_closure_skips_unknown_seeds():
    parsed = ParsedLibc(
        header=[],
        functions={"_a": LibcFunction("_a", source=[], deps=set())},
        data_labels={},
        data_section_lines=[],
        bss_section_lines=[],
    )
    needed = parsed.transitive_closure({"_a", "_user_defined_func"})
    assert needed == {"_a"}


# ── emit ─────────────────────────────────────────────────────────


def test_emit_includes_only_needed_functions():
    parsed = parse_libc(
        "; header\n"
        "        section .text\n"
        "_foo:\n"
        "        ret\n"
        "_bar:\n"
        "        ret\n"
        "_baz:\n"
        "        ret\n"
    )
    out = parsed.emit({"_foo", "_baz"})
    assert "_foo:" in out
    assert "_baz:" in out
    assert "_bar:" not in out


def test_emit_includes_data_when_referenced():
    """When a needed function references a data label, the data
    section is emitted with that label."""
    parsed = parse_libc(
        "        section .text\n"
        "_func:\n"
        "        mov eax, [_global]\n"
        "        ret\n"
        "section .data\n"
        "_global: dd 42\n"
        "section .bss\n"
        "_scratch: resb 16\n"
    )
    needed = parsed.transitive_closure({"_func"})
    assert "_global" in needed
    assert "_scratch" not in needed  # not referenced
    out = parsed.emit(needed)
    assert "section .data" in out
    assert "_global: dd 42" in out
    assert "_scratch: resb 16" not in out


def test_emit_skips_unreferenced_data_and_bss():
    """If no needed function references a data label, drop the entire
    section."""
    parsed = parse_libc(
        "        section .text\n"
        "_func:\n"
        "        ret\n"
        "section .data\n"
        "_global: dd 42\n"
        "section .bss\n"
        "_scratch: resb 16\n"
    )
    out = parsed.emit({"_func"})
    assert "section .data" not in out
    assert "_global" not in out
    assert "section .bss" not in out
    assert "_scratch" not in out


# ── Integration with real libc ───────────────────────────────────


