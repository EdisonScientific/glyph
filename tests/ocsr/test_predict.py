"""Public OCSR predictor behavior."""

from types import SimpleNamespace

import pytest
import torch

from glyph.ocsr.predict import OCSRPredictor, _reconcile_quantize_mode


def _unloaded_predictor() -> OCSRPredictor:
    predictor = OCSRPredictor.__new__(OCSRPredictor)
    predictor.recipe = SimpleNamespace(input_size=8)
    predictor.device = torch.device("cpu")
    predictor._input_dtype = torch.float32
    predictor.model = object()
    predictor.tokenizer = object()
    predictor.max_len = 16
    return predictor


def test_predict_batch_postprocesses_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "glyph.ocsr.predict._preprocess_image",
        lambda image, input_size: torch.zeros(3, input_size, input_size),
    )
    monkeypatch.setattr(
        "glyph.ocsr.predict._greedy_batch",
        lambda model, images, tokenizer, max_len: ["raw-decode"] * len(images),
    )
    seen: list[str] = []

    def fake_postprocess(smiles: str) -> str:
        seen.append(smiles)
        return f"processed:{smiles}"

    monkeypatch.setattr("glyph.ocsr.predict.postprocess_smiles", fake_postprocess)
    predictor = _unloaded_predictor()

    assert predictor.predict_batch(["a.png", "b.png"]) == [
        "processed:raw-decode",
        "processed:raw-decode",
    ]
    assert seen == ["raw-decode", "raw-decode"]


def test_predict_batch_can_disable_postprocessing(monkeypatch) -> None:
    monkeypatch.setattr(
        "glyph.ocsr.predict._preprocess_image",
        lambda image, input_size: torch.zeros(3, input_size, input_size),
    )
    monkeypatch.setattr(
        "glyph.ocsr.predict._greedy_batch",
        lambda model, images, tokenizer, max_len: ["raw-decode"] * len(images),
    )
    monkeypatch.setattr(
        "glyph.ocsr.predict.postprocess_smiles",
        lambda smiles: (_ for _ in ()).throw(AssertionError("postprocess should be disabled")),
    )

    assert _unloaded_predictor().predict_batch(["a.png"], postprocess=False) == ["raw-decode"]


class TestReconcileQuantizeMode:
    """Requested vs stored INT8 mode reconciliation (both full and decoder)."""

    def test_no_quantization_passthrough(self) -> None:
        assert _reconcile_quantize_mode(None, None) is None

    @pytest.mark.parametrize("mode", ["full", "decoder"])
    def test_fp32_checkpoint_uses_requested_mode(self, mode: str) -> None:
        # No stored mode: the requested mode is applied on the fly.
        assert _reconcile_quantize_mode(mode, None) == mode

    @pytest.mark.parametrize("mode", ["full", "decoder"])
    def test_matching_stored_mode_is_accepted(self, mode: str) -> None:
        assert _reconcile_quantize_mode(mode, mode) == mode

    @pytest.mark.parametrize(
        ("requested", "stored"),
        [("decoder", "full"), ("full", "decoder")],
    )
    def test_mismatch_is_rejected(self, requested: str, stored: str) -> None:
        with pytest.raises(ValueError, match="pre-quantized checkpoint cannot be"):
            _reconcile_quantize_mode(requested, stored)

    @pytest.mark.parametrize("stored", ["full", "decoder"])
    def test_unspecified_request_defers_to_stored_mode(self, stored: str) -> None:
        # A pre-quantized checkpoint loaded without an explicit request uses its
        # baked-in mode.
        assert _reconcile_quantize_mode(None, stored) == stored
