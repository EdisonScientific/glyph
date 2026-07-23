#!/usr/bin/env python3
"""Import-sanity tests: confirm core modules can be imported without side effects.

These are intentionally cheap — they catch trivial regressions
(missing imports, syntax errors, accidental top-level I/O) early.
"""

import importlib

import pytest

# Modules safe to import without heavy / optional deps loaded.
SAFE_MODULES = [
    "glyph",
    "glyph.markush",
    "glyph.markush.eval",
    "glyph.markush.eval.common",
    "glyph.markush.train",
    "glyph.markush.data",
]


@pytest.mark.parametrize("modname", SAFE_MODULES)
def test_module_imports_cleanly(modname):
    """Every safe module should import without raising."""
    importlib.import_module(modname)


def test_markush_has_version():
    import glyph.markush

    assert isinstance(glyph.markush.__version__, str)
    assert glyph.markush.__version__  # non-empty


def test_eval_common_public_api_present():
    """The expected public functions exist and are callable in the
    consolidated ``glyph.markush.eval.common`` module."""
    from glyph.markush.eval import common

    for name in (
        "extract_cxsmiles_and_stable",
        "strip_r_tags",
        "extract_r_labels",
        "split_cxsmiles",
        "parse_stable",
        "clean_prediction",
        "prediction_to_cxsmiles_opt",
    ):
        fn = getattr(common, name, None)
        assert callable(fn), f"{name} missing or not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
