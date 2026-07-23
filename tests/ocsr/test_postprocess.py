"""Tests for the conservative SMILES postprocessor."""

from glyph.ocsr.postprocess import light_postprocess, strip_h2_fragments


def test_strip_h2_simple():
    assert strip_h2_fragments("CCO.[HH]") == "CCO"
    assert strip_h2_fragments("CCO.[HH].[HH]") == "CCO"


def test_strip_h2_keeps_real_counterions():
    assert strip_h2_fragments("CCO.[I-]") == "CCO.[I-]"
    assert strip_h2_fragments("C.C") == "C.C"
    assert strip_h2_fragments("[Na+].[Cl-]") == "[Na+].[Cl-]"


def test_strip_h2_safe_on_empty_or_pure_hh():
    assert strip_h2_fragments("") == ""
    # Pure [HH] string should not collapse to empty: defensively kept.
    assert strip_h2_fragments("[HH]") == "[HH]"


def test_light_postprocess_alias():
    assert light_postprocess("c1ccccc1.[HH]") == "c1ccccc1"
