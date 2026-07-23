"""Import-sanity test — package modules import without side effects."""

import importlib


def test_top_level_import():
    pkg = importlib.import_module("glyph.ocsr")
    assert pkg.__version__


def test_submodule_imports():
    for name in [
        "glyph.ocsr.config",
        "glyph.ocsr.model",
        "glyph.ocsr.smiles_tokenizer",
        "glyph.ocsr.eval",
        "glyph.ocsr.full_eval",
        "glyph.ocsr.predict",
        "glyph.ocsr.cli",
    ]:
        importlib.import_module(name)
