"""Tests for the real-benchmark image loader.

The first three tests use synthetic in-memory PNGs so they run with no I/O
dependence. The fourth test exercises 2 real cached HF rows; it is skipped
if the parquet snapshot is not on disk in this environment.
"""

from __future__ import annotations

import io
import os

import pytest
import torch
from PIL import Image

from glyph.ocsr.bench_image import (
    _decode_png_bytes_to_rgb,
    _pil_to_chw_tensor,
    load_benchmark_samples,
    stack_batch,
)


def _make_synthetic_png_bytes(width: int = 64, height: int = 32, mode: str = "1") -> bytes:
    """Make a tiny in-memory PNG matching the HF benchmark's 1-bit format."""
    img = Image.new(mode, (width, height), 1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_png_bytes_to_rgb_handles_1bit():
    png = _make_synthetic_png_bytes(mode="1")
    img = _decode_png_bytes_to_rgb(png)
    assert img.mode == "RGB"
    assert img.size == (64, 32)


def test_decode_png_bytes_to_rgb_rejects_empty():
    with pytest.raises(ValueError):
        _decode_png_bytes_to_rgb(b"")


def test_pil_to_chw_tensor_resizes_and_normalizes():
    png = _make_synthetic_png_bytes(width=10, height=20)
    img = _decode_png_bytes_to_rgb(png)
    tensor = _pil_to_chw_tensor(img, size=48)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 48, 48)
    # White-on-white synthetic image -> after [-1,1] normalization, values ~ 1.0
    assert tensor.max().item() <= 1.0 + 1e-6
    assert tensor.min().item() >= -1.0 - 1e-6


@pytest.mark.skipif(
    os.environ.get("GLYPH_RUN_HF_TESTS", "") not in {"1", "true", "yes"},
    reason="opt-in test: set GLYPH_RUN_HF_TESTS=1 to download the HF benchmark parquet",
)
def test_load_two_real_benchmark_samples():
    samples = load_benchmark_samples(n=2, image_size=128)
    assert len(samples) == 2
    for s in samples:
        assert s.image.shape == (3, 128, 128)
        assert isinstance(s.image_id, str) and s.image_id
        # smiles can be None for excluded rows; first 2 parseable rows have it.
        assert isinstance(s.smiles, str) and s.smiles
    batch = stack_batch(samples)
    assert batch["images"].shape == (2, 3, 128, 128)
    assert len(batch["image_ids"]) == 2
