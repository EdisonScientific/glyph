"""Import-sanity tests for the tiny renderer/dataset wrapper."""

import importlib

import pytest
import torch

from glyph.ocsr.data import TrainRow


def _renderer_available():
    try:
        importlib.import_module("glyph.ocsr.renderer").render_smiles_to_png_bytes("CCO", size=64)
        return True
    except Exception:
        return False


_SKIP_REASON = (
    "RDKit MolDraw2D requires system libs (libXrender, libfreetype, fontconfig); skipping headless."
)


@pytest.mark.skipif(not _renderer_available(), reason=_SKIP_REASON)
def test_renderer_returns_png_bytes():
    from glyph.ocsr.renderer import render_smiles_to_png_bytes

    png = render_smiles_to_png_bytes("CCO", size=128)
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not _renderer_available(), reason=_SKIP_REASON)
def test_dataset_returns_tensor_and_label():
    from glyph.ocsr.dataset import TinySmilesDataset

    rows = [
        TrainRow(smiles="CCO", image_id="ethanol"),
        TrainRow(smiles="c1ccccc1", image_id="benzene"),
    ]
    ds = TinySmilesDataset(rows, image_size=96)
    img, smi, image_id = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 96, 96)
    assert torch.is_floating_point(img)
    assert smi == "CCO"
    assert image_id == "ethanol"
    assert img.max().item() <= 1.0 + 1e-6
    assert img.min().item() >= -1.0 - 1e-6
