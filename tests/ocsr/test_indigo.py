"""Boundary tests for the vendored Indigo renderer.

These tests exercise the vendored Indigo .so files. They require host system
libraries (libfreetype, libfontconfig, libXrender). On hosts missing those,
the rendering tests are skipped; the load-only test still runs.
"""

from __future__ import annotations

import pytest


def _indigo_loads():
    try:
        from glyph.ocsr import _indigo  # noqa: F401

        return True
    except Exception:
        return False


def _indigo_renderer_works():
    try:
        from glyph.ocsr.indigo_renderer import IndigoRendererBoundary

        rb = IndigoRendererBoundary(image_size=64)
        rb.render_smiles_to_png("c1ccccc1")
        return True
    except Exception:
        return False


_SKIP_RENDER = "Indigo renderer requires libfreetype/libfontconfig/libXrender on the host."


@pytest.mark.skipif(not _indigo_loads(), reason="vendored Indigo not loadable on this host")
def test_indigo_loads_and_round_trips_smiles():
    from glyph.ocsr import _indigo

    ix = _indigo.Indigo()
    assert isinstance(ix.version(), str)
    mol = ix.loadMolecule("CCO")
    assert mol.smiles() == "CCO"


@pytest.mark.skipif(not _indigo_renderer_works(), reason=_SKIP_RENDER)
def test_indigo_renderer_boundary_emits_png():
    from glyph.ocsr.indigo_renderer import IndigoRendererBoundary

    rb = IndigoRendererBoundary(image_size=128)
    png = rb.render_smiles_to_png("c1ccccc1")
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert 256 < len(png) < 200_000


@pytest.mark.skipif(not _indigo_renderer_works(), reason=_SKIP_RENDER)
def test_indigo_renderer_handles_complex_smiles():
    from glyph.ocsr.indigo_renderer import IndigoRendererBoundary

    rb = IndigoRendererBoundary(image_size=192)
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    png = rb.render_smiles_to_png(aspirin)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not _indigo_renderer_works(), reason=_SKIP_RENDER)
def test_indigo_renderer_rejects_empty_smiles():
    from glyph.ocsr.indigo_renderer import IndigoRendererBoundary

    rb = IndigoRendererBoundary(image_size=64)
    with pytest.raises(ValueError):
        rb.render_smiles_to_png("")


@pytest.mark.skipif(not _indigo_renderer_works(), reason=_SKIP_RENDER)
def test_indigo_render_then_pil_decode_round_trip():
    from glyph.ocsr.indigo_renderer import IndigoRendererBoundary

    rb = IndigoRendererBoundary(image_size=96)
    img = rb.render_smiles_to_pil("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    assert img.mode == "RGB"
    assert img.size[0] >= 32 and img.size[1] >= 32
