"""Unit tests for the frontier-model baseline generator.

These tests never call a hosted model. The litellm call is injected via a fake
``completion_fn``, so the parsing, checkpoint/resume, artifact-schema, and
scoring wiring are all exercised offline. litellm is not imported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glyph.markush.eval import baselines


def test_resolve_and_slug_known_and_passthrough() -> None:
    assert baselines.resolve_litellm_model("claude-opus-4-8") == "anthropic/claude-opus-4-8"
    assert baselines.resolve_litellm_model("gpt-5.6-sol") == "openai/gpt-5.6-sol"
    # A full litellm string passes through untouched.
    assert baselines.resolve_litellm_model("openai/gpt-4.1") == "openai/gpt-4.1"
    assert baselines.model_slug("claude-opus-4-8") == "claude-opus-4-8"
    assert baselines.model_slug("openai/gpt-5.6-sol") == "gpt-5.6-sol"


def test_apply_prompt_variant() -> None:
    prompt = "Extract the Markush structure from this image."
    assert baselines.apply_prompt_variant(prompt, baselines.VARIANT_DEFAULT) == prompt
    formatted = baselines.apply_prompt_variant(prompt, baselines.VARIANT_FORMAT)
    assert formatted.startswith(prompt)
    assert "<markush><cxsmi>" in formatted
    assert "Return ONLY" in formatted
    with pytest.raises(ValueError):
        baselines.apply_prompt_variant(prompt, "nope")


def test_iter_variants() -> None:
    assert list(baselines.iter_variants("default")) == ["default"]
    assert list(baselines.iter_variants("format")) == ["format"]
    assert list(baselines.iter_variants("both")) == ["default", "format"]
    with pytest.raises(ValueError):
        list(baselines.iter_variants("bogus"))


def test_image_data_url_and_messages(tmp_path: Path) -> None:
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    url = baselines.image_data_url(png)
    assert url.startswith("data:image/png;base64,")
    messages = baselines.build_messages(png, "hello")
    assert messages[0]["role"] == "user"
    kinds = [part["type"] for part in messages[0]["content"]]
    assert kinds == ["image_url", "text"]
    assert messages[0]["content"][1]["text"] == "hello"


def test_parse_prediction_matches_eval_extraction() -> None:
    raw = "<think>reason</think><markush><cxsmi>[*]C(=O)O |$R1;;;$|</cxsmi><stable></stable></markush>"
    prediction, cxsmiles = baselines.parse_prediction(raw)
    assert "<think>" not in prediction
    assert cxsmiles == "[*]C(=O)O |$R1;;;$|"


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 20
    total_tokens = 30


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


def test_call_frontier_model_success_and_error() -> None:
    def ok_fn(**kwargs: Any) -> _FakeResponse:
        assert kwargs["model"] == "anthropic/claude-opus-4-8"
        return _FakeResponse("<markush><cxsmi>CC</cxsmi><stable></stable></markush>")

    call = baselines.call_frontier_model("claude-opus-4-8", [], completion_fn=ok_fn)
    assert call.error == ""
    assert "<cxsmi>CC</cxsmi>" in call.raw
    assert call.usage["completion_tokens"] == 20

    def boom_fn(**kwargs: Any) -> _FakeResponse:
        raise RuntimeError("rate limited")

    failed = baselines.call_frontier_model("claude-opus-4-8", [], completion_fn=boom_fn)
    assert failed.raw == ""
    assert "rate limited" in failed.error


def _samples(n: int = 3) -> list[baselines.BaselineSample]:
    return [
        baselines.BaselineSample(
            index=i,
            sample_id=str(i),
            image_path=f"/nonexistent/{i}.png",
            prompt="Extract the Markush structure from this image.",
            ground_truth="CC",
        )
        for i in range(n)
    ]


def test_generate_predictions_checkpoint_and_resume(tmp_path: Path, monkeypatch: Any) -> None:
    # Avoid touching the filesystem for image encoding.
    monkeypatch.setattr(baselines, "image_data_url", lambda path: "data:image/png;base64,AAAA")

    calls = {"n": 0}

    def flaky_fn(**kwargs: Any) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse("<markush><cxsmi>CC</cxsmi><stable></stable></markush>")

    samples = _samples(3)
    checkpoint = tmp_path / "default.checkpoint.jsonl"
    config = baselines.GenerationConfig(
        model="claude-opus-4-8", benchmark="ip5m", concurrency=1, num_retries=0
    )

    rows = baselines.generate_predictions(
        config, checkpoint_path=checkpoint, completion_fn=flaky_fn, progress=False, samples=samples
    )
    assert len(rows) == 3
    assert calls["n"] == 3
    assert all(row["error"] == "" for row in rows)

    # Re-run: everything already complete, so no new calls happen.
    rows2 = baselines.generate_predictions(
        config, checkpoint_path=checkpoint, completion_fn=flaky_fn, progress=False, samples=samples
    )
    assert calls["n"] == 3  # unchanged
    assert len(rows2) == 3
    assert [r["sample_id"] for r in rows2] == ["0", "1", "2"]


def test_errors_are_retried_on_resume(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(baselines, "image_data_url", lambda path: "data:image/png;base64,AAAA")

    samples = _samples(3)
    checkpoint = tmp_path / "default.checkpoint.jsonl"

    # Seed a checkpoint with one errored row.
    checkpoint.write_text(
        json.dumps(
            {"index": 1, "sample_id": "1", "error": "boom", "prediction": "", "ground_truth": "CC"}
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = baselines.load_checkpoint(checkpoint)
    assert not baselines._is_completed(loaded["1"])

    pending = baselines._pending_samples(samples, loaded)
    # id 1 (errored) is pending; ids 0/2 unattempted are pending; nothing done yet.
    assert {s.sample_id for s in pending} == {"0", "1", "2"}


def test_build_predictions_json_schema(tmp_path: Path) -> None:
    rows = [
        {
            "index": 0,
            "sample_id": "0",
            "image_path": "/x/0.png",
            "ground_truth": "CC",
            "prediction": "<markush><cxsmi>CC</cxsmi><stable></stable></markush>",
            "prediction_cxsmiles": "CC",
            "error": "",
        }
    ]
    config = baselines.GenerationConfig(
        model="claude-opus-4-8", benchmark="ip5m", variant="default"
    )
    out = tmp_path / "default.predictions.json"
    baselines.build_predictions_json(rows, out, config=config)
    payload = json.loads(out.read_text())
    assert set(payload) >= {"config", "aggregate", "samples", "timing", "inference_stats"}
    sample = payload["samples"][0]
    # Fields the DS4SD official scorer joins/reads.
    assert sample["sample_id"] == "0"
    assert sample["prediction"].startswith("<markush>")
    assert payload["config"]["model"] == "anthropic/claude-opus-4-8"
    assert payload["config"]["markush_prompt_policy"] == baselines.BASELINE_PROMPT_POLICY


def test_write_grading_jsonls_shapes(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "7",
            "prediction": "<markush><cxsmi>CC</cxsmi><stable></stable></markush>",
            "ground_truth": "CC",
        }
    ]
    pred = tmp_path / "grading_predictions.jsonl"
    gt = tmp_path / "grading_gt.jsonl"
    baselines.write_grading_jsonls(rows, predictions_path=pred, ground_truth_path=gt)
    pred_row = json.loads(pred.read_text().splitlines()[0])
    gt_row = json.loads(gt.read_text().splitlines()[0])
    assert pred_row == {
        "id": "7",
        "prediction": "<markush><cxsmi>CC</cxsmi><stable></stable></markush>",
    }
    assert gt_row == {"id": "7", "cxsmiles_opt": "CC"}


def test_score_strict_ours_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    rows = [
        {
            "sample_id": "0",
            "prediction": "<markush><cxsmi>CC</cxsmi><stable></stable></markush>",
            "ground_truth": "CC",
        },
        {
            "sample_id": "1",
            "prediction": "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>",
            "ground_truth": "c1ccccc1",
        },
    ]
    pred = tmp_path / "grading_predictions.jsonl"
    gt = tmp_path / "grading_gt.jsonl"
    baselines.write_grading_jsonls(rows, predictions_path=pred, ground_truth_path=gt)
    result = baselines.score_strict_ours(pred, gt, output_path=tmp_path / "strict_ours.json")
    ours = result["summary"]["ours"]
    assert ours["size"] == 2
    # One exact match (CC == CC), one mismatch.
    assert ours["cxsmi_equality_count"] == 1
    assert 49.0 < ours["cxsmi_equality_pct"] < 51.0
