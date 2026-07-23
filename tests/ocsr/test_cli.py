"""CLI tests for the unified Glyph Typer application."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from glyph.cli import app
from glyph.markush import MarkushPrediction
from glyph.markush import cli as markush_cli
from glyph.markush.inference import DEFAULT_MARKUSH_BASE_MODEL
from glyph.ocsr.predict import DEFAULT_HF_REPO


def test_top_level_cli_exposes_model_families() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "ocsr" in result.output
    assert "markush" in result.output


def test_ocsr_predict_uses_released_checkpoint_by_default(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakePredictor:
        def __init__(
            self,
            checkpoint: str | None,
            *,
            device: str | None,
            quantize: str | None,
            precision: str | None,
        ) -> None:
            calls["init"] = {
                "checkpoint": checkpoint,
                "device": device,
                "quantize": quantize,
                "precision": precision,
            }

        def predict_batch(
            self,
            images: list[object],
            *,
            batch_size: int,
            postprocess: bool,
        ) -> list[str]:
            calls["predict"] = {
                "images": [str(image) for image in images],
                "batch_size": batch_size,
                "postprocess": postprocess,
            }
            return ["CCO"]

    monkeypatch.setattr("glyph.ocsr.predict.OCSRPredictor", FakePredictor)
    image = tmp_path / "molecule.png"
    image.write_bytes(b"fake")

    result = CliRunner().invoke(app, ["ocsr", "predict", str(image), "--json"])

    assert result.exit_code == 0, result.output
    assert calls["init"] == {
        "checkpoint": None,
        "device": None,
        "quantize": None,
        "precision": None,
    }
    assert DEFAULT_HF_REPO == "EdisonScientific/OCSRGlyph"
    assert calls["predict"] == {
        "images": [str(image)],
        "batch_size": 8,
        "postprocess": True,
    }
    assert json.loads(result.output) == {"image": str(image), "smiles": "CCO"}


def test_markush_released_adapter_download_uses_hugging_face(monkeypatch) -> None:
    calls: list[str] = []

    def fake_download(repo_id: str) -> str:
        calls.append(repo_id)
        return "/cache/markush-adapter"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)

    assert markush_cli._download_released_adapter() == "/cache/markush-adapter"
    assert calls == ["EdisonScientific/MarkushGlyph"]


def test_markush_predict_uses_released_adapter_and_base_by_default(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakePredictor:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def predict(self, image: object, **kwargs: object) -> MarkushPrediction:
            calls["image"] = str(image)
            calls["predict_kwargs"] = kwargs
            return MarkushPrediction.from_raw(str(image), "<markush><cxsmi>CC</cxsmi></markush>")

    monkeypatch.delenv("MARKUSH_BASE_MODEL", raising=False)
    monkeypatch.setattr(
        "glyph.markush.cli._download_released_adapter", lambda: "/cache/markush-adapter"
    )
    monkeypatch.setattr("glyph.markush.cli.MarkushPredictor", FakePredictor)
    image = tmp_path / "claim.png"
    image.write_bytes(b"fake")

    result = CliRunner().invoke(app, ["markush", "predict", str(image), "--json"])

    assert result.exit_code == 0, result.output
    assert calls["init"]["checkpoint"] == "/cache/markush-adapter"
    assert calls["init"]["base_model"] == DEFAULT_MARKUSH_BASE_MODEL
    assert calls["image"] == str(image)
    assert calls["predict_kwargs"] == {"majority_vote": False, "vote_k": 8}
    assert json.loads(result.output)["markush_xml"] == (
        "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"
    )
