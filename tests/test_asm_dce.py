"""Tests for src/uc386/asm_dce.py — post-codegen dead-code elimination."""
from __future__ import annotations

from upeep386.asm_dce import parse_asm, dce


def _wrap(body: str) -> str:
    """Wrap a function-body fragment in the standard codegen header."""
    return (
        "; uc386 codegen output\n"
        "        bits 32\n"
        "        section .text\n"
        "        global _start\n"
        "\n"
        + body
    )


def test_parse_asm_basic():
    text = _wrap(
        "_start:\n"
        "__start:\n"
        "        call    _main\n"
        "        mov     ah, 4Ch\n"
        "        int     21h\n"
        "_main:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    parsed = parse_asm(text)
    # Both _start and _main are recognized.
    assert "_start" in parsed.functions
    assert "_main" in parsed.functions
    # _start's body merges with __start's (alias).
    assert "__start" not in parsed.functions
    # _main is in the entry-points set by default.
    assert "_main" in parsed.entry_points
    assert "_start" in parsed.entry_points


def test_dce_drops_unreferenced_function():
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        call    _used\n"
        "        ret\n"
        "_used:\n"
        "        ret\n"
        "_unused:\n"
        "        ret\n"
    )
    out = dce(text)
    assert "_unused:" not in out
    assert "_used:" in out
    assert "_main:" in out


def test_dce_keeps_transitively_reachable():
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        call    _level1\n"
        "        ret\n"
        "_level1:\n"
        "        call    _level2\n"
        "        ret\n"
        "_level2:\n"
        "        ret\n"
        "_dead_at_root:\n"
        "        call    _level1\n"  # _level1 is referenced from here too,
        "        ret\n"               # but _dead_at_root has no incoming.
    )
    out = dce(text)
    assert "_level2:" in out  # transitively reachable via main → l1 → l2
    assert "_dead_at_root:" not in out  # no incoming edge to _dead_at_root


def test_dce_keeps_global_marked():
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "        global  _exposed\n"
        "_main:\n"
        "        ret\n"
        "_exposed:\n"  # has no caller, but is `global` → entry point
        "        ret\n"
    )
    parsed = parse_asm(text)
    assert "_exposed" in parsed.entry_points
    out = dce(text)
    assert "_exposed:" in out


def test_dce_drops_data_label_with_dead_owner():
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        ret\n"
        "_unused_fn:\n"
        "        mov     eax, [_unused_data]\n"
        "        ret\n"
        "        section .data\n"
        "_unused_data:\n"
        "        dd      42\n"
    )
    out = dce(text)
    assert "_unused_fn:" not in out
    assert "_unused_data:" not in out
    # The .data section header itself should be dropped when no needed
    # data labels remain.
    assert "section .data" not in out


def test_dce_keeps_data_label_referenced_by_live_fn():
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        mov     eax, [_live_data]\n"
        "        ret\n"
        "        section .data\n"
        "_live_data:\n"
        "        dd      42\n"
    )
    out = dce(text)
    assert "_live_data:" in out
    assert "section .data" in out


def test_dce_data_label_chain():
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        mov     eax, [_root_data]\n"
        "        ret\n"
        "        section .data\n"
        "_root_data:\n"
        "        dd      _next_data\n"
        "_next_data:\n"
        "        dd      99\n"
        "_orphan_data:\n"
        "        dd      0\n"
    )
    out = dce(text)
    assert "_root_data:" in out
    # _next_data is referenced from _root_data's initializer (transitive).
    assert "_next_data:" in out
    # _orphan_data has no inbound reference.
    assert "_orphan_data:" not in out


def test_dce_drops_bss_when_unreferenced():
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        ret\n"
        "        section .bss\n"
        "_unused_bss:\n"
        "        resb    16\n"
    )
    out = dce(text)
    assert "_unused_bss:" not in out
    assert "section .bss" not in out


def test_dce_keeps_bss_when_referenced():
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        mov     eax, [_used_bss]\n"
        "        ret\n"
        "        section .bss\n"
        "_used_bss:\n"
        "        resd    1\n"
    )
    out = dce(text)
    assert "_used_bss:" in out
    assert "section .bss" in out


def test_dce_keeps_bss_zero_start_end_when_referenced_by_start():
    """If `_start` references `_bss_zero_start` and `_bss_zero_end`
    for BSS init, they must be kept even when no other label needs
    them — they're just markers."""
    text = _wrap(
        "_start:\n"
        "        cld\n"
        "        xor     eax, eax\n"
        "        mov     edi, _bss_zero_start\n"
        "        mov     ecx, _bss_zero_end\n"
        "        sub     ecx, edi\n"
        "        rep stosb\n"
        "        call    _main\n"
        "_main:\n"
        "        ret\n"
        "        section .bss\n"
        "_bss_zero_start:\n"
        "_bss_zero_end:\n"
    )
    out = dce(text)
    assert "_bss_zero_start:" in out
    assert "_bss_zero_end:" in out


def test_dce_no_change_when_all_reachable():
    """If every function/label is reachable, output equals input
    (modulo whitespace normalization)."""
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        call    _helper\n"
        "        ret\n"
        "_helper:\n"
        "        ret\n"
    )
    out = dce(text)
    assert "_start:" in out
    assert "_main:" in out
    assert "_helper:" in out


def test_dce_drops_function_referencing_only_dead_data():
    """A function that only references a dead-from-roots data label
    is itself dead (no path from root reaches the function)."""
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        ret\n"
        "_dead_fn:\n"
        "        mov     eax, [_dead_data]\n"
        "        ret\n"
        "        section .data\n"
        "_dead_data:\n"
        "        dd      0\n"
    )
    out = dce(text)
    assert "_dead_fn:" not in out
    assert "_dead_data:" not in out


def test_dce_static_call_reference():
    """`_static_dead` with no caller should be dropped even though
    it has its own deps."""
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        ret\n"
        "_static_dead:\n"
        "        mov     eax, [_only_used_by_dead]\n"
        "        ret\n"
        "        section .data\n"
        "_only_used_by_dead:\n"
        "        dd      0\n"
    )
    out = dce(text)
    assert "_static_dead:" not in out
    assert "_only_used_by_dead:" not in out


def test_dce_handles_no_data_section():
    """A program with no .data is fine."""
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    out = dce(text)
    assert "_main:" in out
    assert "section .data" not in out


def test_dce_function_pointer_reference_is_address_take():
    """A `mov reg, _func` is an address-take; keeps _func live."""
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        mov     eax, _callback\n"  # address-take
        "        ret\n"
        "_callback:\n"
        "        ret\n"
    )
    out = dce(text)
    assert "_callback:" in out


def test_dce_jump_table_entries_reach_functions():
    """A `dd _label` in `.data` (jump table) keeps `_label` live."""
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        mov     eax, [_dispatch]\n"
        "        ret\n"
        "        section .data\n"
        "_dispatch:\n"
        "        dd      _case0, _case1, _case2\n"
        "_case0:\n"  # placed in .data — ok if treated as data label
        "        dd      0\n"
        "_case1:\n"
        "        dd      1\n"
        "_case2:\n"
        "        dd      2\n"
    )
    out = dce(text)
    assert "_dispatch:" in out
    # Each case is referenced via _dispatch's data, so all reachable.
    assert "_case0:" in out
    assert "_case1:" in out
    assert "_case2:" in out


def test_dce_ignores_local_labels():
    """`.LX:` labels are intra-function; shouldn't show up in deps."""
    text = _wrap(
        "_start:\n"
        "        call    _main\n"
        "_main:\n"
        "        jz      .L_else\n"
        "        ret\n"
        ".L_else:\n"
        "        ret\n"
    )
    parsed = parse_asm(text)
    # _main's deps shouldn't contain `.L_else` (it's not a `_*` symbol).
    assert all(not d.startswith(".") for d in parsed.functions["_main"].deps)
