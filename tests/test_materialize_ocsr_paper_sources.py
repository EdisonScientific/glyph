import importlib.util
import sys
from pathlib import Path

import pytest

from glyph.data.markush_render import (
    RenderJob,
    _ensure_indigo_pythonpath,
    pseudo_markush_smiles_reason,
    render_result_to_row,
    render_smiles_job,
)


def _indigo_renderer_available() -> bool:
    """True when the real glyph.ocsr Indigo renderer can be imported and run."""
    if importlib.util.find_spec("rdkit") is None:
        return False
    try:
        from glyph.ocsr.indigo_renderer import IndigoRendererBoundary

        IndigoRendererBoundary(image_size=64).version()
    except Exception:
        return False
    return True


def _render_job(tmp_path: Path, *, smiles: str = "CCO") -> RenderJob:
    return RenderJob(
        source_name="unit",
        source_path="unit.csv",
        source_row_index=3,
        sample_index=7,
        smiles=smiles,
        output_path=str(tmp_path / "render.png"),
        row={"SMILES": smiles, "pubchem_cid": "123"},
    )


def test_rdkit_render_backend_writes_png_and_metadata(tmp_path: Path) -> None:
    result = render_smiles_job(
        _render_job(tmp_path),
        width=128,
        height=128,
        force=True,
        backend="rdkit",
    )

    assert result.ok, result.error
    assert result.render_backend == "rdkit"
    assert Path(result.output_path).read_bytes().startswith(b"\x89PNG")

    row = render_result_to_row(result, {"pubchem_cid": "123"})
    assert row["smiles"] == "CCO"
    assert row["metadata"]["render_backend"] == "rdkit"


def test_indigo_render_backend_uses_real_glyph_renderer(tmp_path: Path) -> None:
    if not _indigo_renderer_available():
        pytest.skip("glyph.ocsr Indigo renderer (Linux x86-64 + rdkit) not available")

    result = render_smiles_job(
        _render_job(tmp_path),
        width=64,
        height=64,
        force=True,
        backend="indigo",
    )

    assert result.ok, result.error
    assert result.render_backend == "indigo"
    # The real Indigo renderer emits a genuine PNG image.
    assert Path(result.output_path).read_bytes().startswith(b"\x89PNG")


def test_ensure_indigo_pythonpath_prepends_path(tmp_path: Path) -> None:
    staged = tmp_path / "staged_src"
    staged.mkdir()
    resolved = str(staged.resolve())
    original = list(sys.path)
    try:
        _ensure_indigo_pythonpath(str(staged))
        assert sys.path[0] == resolved
        # Idempotent: an already-present path is not duplicated.
        _ensure_indigo_pythonpath(str(staged))
        assert sys.path.count(resolved) == 1
    finally:
        sys.path[:] = original


def test_indigo_render_backend_requires_square_size(tmp_path: Path) -> None:
    result = render_smiles_job(
        _render_job(tmp_path),
        width=64,
        height=96,
        force=True,
        backend="indigo",
    )

    assert not result.ok
    assert "square render dimensions" in result.error


def test_pseudo_markush_smiles_filter_rejects_placeholder_targets() -> None:
    assert pseudo_markush_smiles_reason("CC*") == "pseudo_markush_dummy_atom"
    assert pseudo_markush_smiles_reason("c1cc([Ar])ccc1") == "pseudo_markush_placeholder_atom"
    assert pseudo_markush_smiles_reason("CC[Y]") == "pseudo_markush_placeholder_atom"
    assert pseudo_markush_smiles_reason("CCO") == ""
