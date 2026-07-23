from __future__ import annotations

import json
from typing import ClassVar

from typer.testing import CliRunner

from glyph.markush import MarkushPrediction, MarkushPredictor
from glyph.markush import extract_components as public_extract_components
from glyph.markush import match_concrete_smiles as public_match_concrete_smiles
from glyph.markush import normalize_claim as public_normalize_claim
from glyph.markush import parse_claim as public_parse_claim
from glyph.markush import strict_equal as public_strict_equal
from glyph.markush.cli import app
from glyph.markush.grading import grade, grade_both
from glyph.markush.inference import (
    DEFAULT_MARKUSH_ADAPTER,
    DEFAULT_MARKUSH_BASE_MODEL,
    OCSR_NO_EXTENSION_PROMPT,
    _looks_like_hf_repo_id,
    _resolve_adapter,
    _resolve_base_model,
)
from glyph.markush.targets import check_cxsmiles_round_trip, normalize_markush_target


def test_ocsr_no_extension_prompt_matches_validated_training_prompt() -> None:
    assert OCSR_NO_EXTENSION_PROMPT == (
        "Extract the Markush structure from this image.\n"
        "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
        "the <cxsmi> value must use canonicalized cxsmiles_opt.\n"
        "This image is an ordinary molecule with no Markush extension annotations; "
        "return only the base molecule in <cxsmi> and leave <stable> empty."
    )


def test_markush_prediction_normalizes_wrapped_output() -> None:
    result = MarkushPrediction.from_raw(
        "claim.png",
        "<markush><cxsmi>CC</cxsmi><stable>R1=Cl</stable></markush>",
    )

    assert result.image == "claim.png"
    assert result.cxsmiles_opt == "CC"
    assert result.stable == "R1=Cl"
    assert result.markush_xml == "<markush><cxsmi>CC</cxsmi><stable>R1=Cl</stable></markush>"
    # The standard-CXSMILES field is derived from the opt form.
    assert result.cxsmiles == "CC"
    assert result.vote == {}


def test_markush_prediction_normalizes_bare_cxsmiles() -> None:
    result = MarkushPrediction.from_raw("claim.png", "CC")

    assert result.cxsmiles_opt == "CC"
    assert result.cxsmiles == "CC"
    assert result.stable == ""
    assert result.markush_xml == "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"


def test_markush_prediction_derives_standard_cxsmiles_with_labels() -> None:
    result = MarkushPrediction.from_raw(
        "claim.png",
        "<markush><cxsmi><r>R1</r>CC</cxsmi><stable></stable></markush>",
    )

    # The opt <cxsmi> form is preserved verbatim (eval/repro contract) while the
    # standard |$...$| form is derived alongside it.
    assert result.cxsmiles_opt == "<r>R1</r>CC"
    assert result.markush_xml == "<markush><cxsmi><r>R1</r>CC</cxsmi><stable></stable></markush>"
    assert result.cxsmiles == "*CC |$R1;;$|"


def test_markush_prediction_standard_cxsmiles_falls_back_to_empty() -> None:
    # An unparseable core must not raise; the standard field is empty while the
    # opt form (and XML) still round-trips the raw model output.
    result = MarkushPrediction.from_raw("claim.png", "not_a_smiles_(((")

    assert result.cxsmiles == ""
    assert result.cxsmiles_opt == "not_a_smiles_((("


def test_markush_prediction_attaches_vote_metadata() -> None:
    result = MarkushPrediction.from_raw(
        "claim.png",
        "CC",
        vote={"n_candidates": 8, "vote_count": 5, "selection_source": "majority_vote"},
    )

    assert result.vote["n_candidates"] == 8
    assert result.vote["vote_count"] == 5
    assert result.asdict()["vote"]["selection_source"] == "majority_vote"


def test_normalize_markush_target_canonicalizes_and_checks_roundtrip() -> None:
    result = normalize_markush_target("C(C)O")

    assert result.valid is True
    assert result.cxsmiles_opt == "CCO"
    assert result.markush_xml == "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
    assert result.target_format == "canonicalized cxsmiles_opt"
    assert result.canonicalization_status == "canonicalized"
    assert result.converter_check is not None
    assert result.converter_check.valid is True


def test_check_cxsmiles_round_trip_preserves_markush_graph() -> None:
    result = check_cxsmiles_round_trip("<r>R1</r>CC")

    assert result.valid is True
    assert result.standard == "*CC |$R1;;$|"
    assert result.roundtrip_opt == "<r>R1</r>CC"


def test_markush_predictor_defaults_base_model(monkeypatch) -> None:
    monkeypatch.delenv("MARKUSH_BASE_MODEL", raising=False)

    predictor = MarkushPredictor(checkpoint="adapter")

    assert predictor.base_model == DEFAULT_MARKUSH_BASE_MODEL


def test_markush_predictor_honors_base_model_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MARKUSH_BASE_MODEL", "/models/local-base")

    predictor = MarkushPredictor(checkpoint="adapter")

    assert predictor.base_model == "/models/local-base"
    # An explicit base_model still wins over the environment override.
    assert MarkushPredictor(base_model="Qwen/Qwen3.5-2B-Base").base_model == "Qwen/Qwen3.5-2B-Base"


def test_looks_like_hf_repo_id_detects_repo_vs_local(tmp_path) -> None:
    # org/name with no suffix and no local path → repo id.
    assert _looks_like_hf_repo_id("EdisonScientific/MarkushGlyph") is True
    # A bare name (no slash) is a local reference, not a repo id.
    assert _looks_like_hf_repo_id("adapter") is False
    # A path with a file suffix is not a repo id.
    assert _looks_like_hf_repo_id("EdisonScientific/model.safetensors") is False
    # An existing local directory (even with a slash) is a local path.
    local_dir = tmp_path / "checkpoint-100"
    local_dir.mkdir()
    assert _looks_like_hf_repo_id(str(local_dir)) is False


def test_resolve_base_model_prefers_explicit_then_env_then_default(monkeypatch) -> None:
    monkeypatch.delenv("MARKUSH_BASE_MODEL", raising=False)
    assert _resolve_base_model(None) == DEFAULT_MARKUSH_BASE_MODEL
    assert _resolve_base_model("") == DEFAULT_MARKUSH_BASE_MODEL
    monkeypatch.setenv("MARKUSH_BASE_MODEL", "/models/env-base")
    assert _resolve_base_model(None) == "/models/env-base"
    assert _resolve_base_model("Qwen/Qwen3.5-2B-Base") == "Qwen/Qwen3.5-2B-Base"


def test_resolve_adapter_downloads_default_when_omitted(monkeypatch) -> None:
    calls: list[str] = []

    def fake_snapshot_download(repo_id: str) -> str:
        calls.append(repo_id)
        return "/cache/markush-adapter"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert _resolve_adapter(None) == "/cache/markush-adapter"
    assert calls == [DEFAULT_MARKUSH_ADAPTER]


def test_resolve_adapter_downloads_repo_id(monkeypatch) -> None:
    calls: list[str] = []

    def fake_snapshot_download(repo_id: str) -> str:
        calls.append(repo_id)
        return "/cache/downloaded"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert _resolve_adapter("EdisonScientific/MarkushGlyph") == "/cache/downloaded"
    assert calls == ["EdisonScientific/MarkushGlyph"]


def test_resolve_adapter_uses_local_path_without_download(monkeypatch, tmp_path) -> None:
    def fail_snapshot_download(repo_id: str) -> str:  # pragma: no cover - must not run
        raise AssertionError(f"unexpected download of {repo_id}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_snapshot_download)

    local_dir = tmp_path / "checkpoint-100"
    local_dir.mkdir()

    assert _resolve_adapter(str(local_dir)) == str(local_dir)


def test_resolve_adapter_reraises_auth_error_with_guidance(monkeypatch) -> None:
    def fake_snapshot_download(repo_id: str) -> str:
        raise RuntimeError("401 Client Error: Unauthorized for url")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    try:
        _resolve_adapter("EdisonScientific/MarkushGlyph")
    except RuntimeError as exc:
        assert "Authentication required" in str(exc)
        assert "huggingface-cli login" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected an auth error to be surfaced")


def test_markush_predictor_resolves_adapter_before_loading(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_snapshot_download(repo_id: str) -> str:
        calls["repo_id"] = repo_id
        return str(tmp_path / "downloaded-adapter")

    def fake_load_model_and_processor(
        base_model: str,
        checkpoint: str | None,
        device: str,
        *,
        dtype: str,
        merge_adapter: bool,
    ) -> tuple[object, object]:
        calls["load_checkpoint"] = checkpoint
        calls["load_base_model"] = base_model
        return object(), object()

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        "glyph.markush.eval.checkpoint.load_model_and_processor",
        fake_load_model_and_processor,
    )

    # Checkpoint omitted → default adapter downloaded and passed to the loader.
    predictor = MarkushPredictor(base_model="base")
    predictor.load()

    assert calls["repo_id"] == DEFAULT_MARKUSH_ADAPTER
    assert calls["load_checkpoint"] == str(tmp_path / "downloaded-adapter")
    assert calls["load_base_model"] == "base"
    assert predictor.checkpoint == str(tmp_path / "downloaded-adapter")


def test_markush_predictor_uses_checkpoint_eval_inference(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    model = object()
    processor = object()

    def fake_load_model_and_processor(
        base_model: str,
        checkpoint: str | None,
        device: str,
        *,
        dtype: str,
        merge_adapter: bool,
    ) -> tuple[object, object]:
        calls["load"] = {
            "base_model": base_model,
            "checkpoint": checkpoint,
            "device": device,
            "dtype": dtype,
            "merge_adapter": merge_adapter,
        }
        return model, processor

    def fake_generate_one(
        loaded_model: object,
        loaded_processor: object,
        record: object,
        *,
        max_new_tokens: int,
        device: str,
    ) -> str:
        calls["generate"] = {
            "model": loaded_model,
            "processor": loaded_processor,
            "record": record,
            "max_new_tokens": max_new_tokens,
            "device": device,
        }
        return "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"

    monkeypatch.setattr(
        "glyph.markush.eval.checkpoint.load_model_and_processor",
        fake_load_model_and_processor,
    )
    monkeypatch.setattr("glyph.markush.eval.checkpoint.generate_one", fake_generate_one)

    image = tmp_path / "claim.png"
    predictor = MarkushPredictor(
        checkpoint="adapter",
        base_model="base",
        device="cpu",
        dtype="float32",
        merge_adapter=False,
        max_new_tokens=17,
        prompt="paper prompt",
    )

    result = predictor.predict(image)

    assert calls["load"] == {
        "base_model": "base",
        "checkpoint": "adapter",
        "device": "cpu",
        "dtype": "float32",
        "merge_adapter": False,
    }
    generate_call = calls["generate"]
    assert generate_call["model"] is model
    assert generate_call["processor"] is processor
    assert generate_call["max_new_tokens"] == 17
    assert generate_call["device"] == "cpu"
    record = generate_call["record"]
    assert record.sample_id == str(image)
    assert record.image_path == str(image)
    assert record.prompt == "paper prompt"
    assert result.markush_xml == "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"


def test_grade_single_prediction_uses_public_shape() -> None:
    xml = "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"

    result = grade(xml, xml, protocol="ours")

    assert result["protocol"] == "ours"
    assert result["metrics"]["n_samples"] == 1
    assert result["sample"]["cxsmi_equality"] is True


def test_grade_both_returns_both_protocols() -> None:
    xml = "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"

    result = grade_both(xml, xml)

    assert set(result["metrics"]) == {"ours", "mg2"}
    assert result["sample"]["ours"]["cxsmi_equality"] is True
    assert result["sample"]["mg2"]["cxsmi_equality"] is True


def test_top_level_exports_conservative_fto_helpers() -> None:
    claim = "<markush><cxsmi><r>R1</r>c1ccccc1</cxsmi><stable></stable></markush>"

    parsed = public_parse_claim(claim)
    components = public_extract_components(claim)
    match = public_match_concrete_smiles("Cc1ccccc1", claim)

    assert parsed.r_labels == ("R1",)
    assert components["backbone_smiles"] == "*c1ccccc1"
    assert public_normalize_claim("CC") == "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"
    assert public_strict_equal("CC", "<markush><cxsmi>CC</cxsmi><stable></stable></markush>")
    assert match.supported is True
    assert match.matches is True


def test_grade_cli_outputs_json(tmp_path) -> None:
    gt = tmp_path / "gt.jsonl"
    pred = tmp_path / "pred.jsonl"
    xml = "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"
    gt.write_text(json.dumps({"id": "0", "annotation": xml}) + "\n", encoding="utf-8")
    pred.write_text(json.dumps({"id": "0", "prediction": xml}) + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "grade",
            "--ground-truth",
            str(gt),
            "--predictions",
            str(pred),
            "--protocol",
            "both",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload["metrics"]) == {"ours", "mg2"}


def test_predict_batch_cli_outputs_jsonl(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {"images": [], "predict_kwargs": []}

    class FakePredictor:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def predict_many(self, images: list[object], **kwargs: object) -> list[MarkushPrediction]:
            calls["images"] = [str(image) for image in images]
            calls["predict_kwargs"] = kwargs
            return [
                MarkushPrediction.from_raw(str(image), "<markush><cxsmi>CC</cxsmi></markush>")
                for image in images
            ]

    monkeypatch.setattr("glyph.markush.cli.MarkushPredictor", FakePredictor)
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"fake")
    image_b.write_bytes(b"fake")

    result = CliRunner().invoke(
        app,
        [
            "predict-batch",
            "--checkpoint",
            "adapter",
            "--base-model",
            "base",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-new-tokens",
            "17",
            "--batch-size",
            "16",
            "--no-merge-adapter",
            str(image_a),
            str(image_b),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["init"] == {
        "checkpoint": "adapter",
        "base_model": "base",
        "device": "cpu",
        "dtype": "float32",
        "merge_adapter": False,
        "max_new_tokens": 17,
        "prompt": (
            "Extract the Markush structure from this image.\n"
            "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
            "the <cxsmi> value must use canonicalized cxsmiles_opt."
        ),
    }
    assert calls["images"] == [str(image_a), str(image_b)]
    # Greedy batch options thread through to the predictor.
    assert calls["predict_kwargs"] == {
        "batch_size": 16,
        "prompt": (
            "Extract the Markush structure from this image.\n"
            "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
            "the <cxsmi> value must use canonicalized cxsmiles_opt."
        ),
        "max_new_tokens": 17,
    }
    rows = [json.loads(line) for line in result.output.splitlines()]
    assert [row["image"] for row in rows] == [str(image_a), str(image_b)]
    assert [row["markush_xml"] for row in rows] == [
        "<markush><cxsmi>CC</cxsmi><stable></stable></markush>",
        "<markush><cxsmi>CC</cxsmi><stable></stable></markush>",
    ]
    # JSONL rows are a superset carrying both CXSMILES representations.
    assert all("cxsmiles" in row and "cxsmiles_opt" in row for row in rows)


def test_predict_batch_cli_keeps_majority_vote_sequential(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakePredictor:
        def __init__(self, **kwargs: object) -> None:
            pass

        def predict(self, image: object, **kwargs: object) -> MarkushPrediction:
            calls.append((str(image), kwargs))
            return MarkushPrediction.from_raw(str(image), "CC")

    monkeypatch.setattr("glyph.markush.cli.MarkushPredictor", FakePredictor)
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    for image in images:
        image.write_bytes(b"fake")

    result = CliRunner().invoke(
        app,
        [
            "predict-batch",
            "--checkpoint",
            "adapter",
            "--base-model",
            "base",
            "--majority-vote",
            "--vote-k",
            "5",
            "--batch-size",
            "16",
            *(str(image) for image in images),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (str(images[0]), {"majority_vote": True, "vote_k": 5}),
        (str(images[1]), {"majority_vote": True, "vote_k": 5}),
    ]
    assert [json.loads(line)["image"] for line in result.output.splitlines()] == [
        str(image) for image in images
    ]


def test_normalize_target_cli_outputs_json() -> None:
    result = CliRunner().invoke(app, ["normalize-target", "C(C)O", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["valid"] is True
    assert payload["cxsmiles_opt"] == "CCO"
    assert payload["target_format"] == "canonicalized cxsmiles_opt"
    assert payload["converter_check"]["valid"] is True


def test_top_level_cli_exposes_paper_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "predict" in result.output
    assert "predict-batch" in result.output
    assert "normalize-target" in result.output
    assert "grade" in result.output
    assert "eval-checkpoint" in result.output


class _FormatFakePredictor:
    """Predictor stand-in returning a fixed labelled Markush prediction."""

    last_predict_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        pass

    def predict(self, image: object, **kwargs: object) -> MarkushPrediction:
        type(self).last_predict_kwargs = kwargs
        return MarkushPrediction.from_raw(
            str(image),
            "<markush><cxsmi><r>R1</r>CC</cxsmi><stable></stable></markush>",
        )


def _invoke_predict(monkeypatch, tmp_path, extra_args: list[str]):
    monkeypatch.setattr("glyph.markush.cli.MarkushPredictor", _FormatFakePredictor)
    image = tmp_path / "claim.png"
    image.write_bytes(b"fake")
    return CliRunner().invoke(
        app,
        [
            "predict",
            "--checkpoint",
            "adapter",
            "--base-model",
            "base",
            "--device",
            "cpu",
            *extra_args,
            str(image),
        ],
    )


def test_predict_text_defaults_to_standard_cxsmiles(monkeypatch, tmp_path) -> None:
    result = _invoke_predict(monkeypatch, tmp_path, [])

    assert result.exit_code == 0, result.output
    # Default --format is standard |$...$| CXSMILES, not the opt <r> form or XML.
    assert result.output.strip() == "*CC |$R1;;$|"


def test_predict_text_format_cxsmiles_opt(monkeypatch, tmp_path) -> None:
    result = _invoke_predict(monkeypatch, tmp_path, ["--format", "cxsmiles_opt"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "<r>R1</r>CC"


def test_predict_json_carries_both_representations(monkeypatch, tmp_path) -> None:
    result = _invoke_predict(monkeypatch, tmp_path, ["--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cxsmiles_opt"] == "<r>R1</r>CC"
    assert payload["cxsmiles"] == "*CC |$R1;;$|"
    # The <cxsmi> inside markush_xml stays the opt form (eval/repro contract).
    assert payload["markush_xml"] == (
        "<markush><cxsmi><r>R1</r>CC</cxsmi><stable></stable></markush>"
    )


def test_predict_majority_vote_flags_thread_through(monkeypatch, tmp_path) -> None:
    result = _invoke_predict(monkeypatch, tmp_path, ["--majority-vote", "--vote-k", "4"])

    assert result.exit_code == 0, result.output
    assert _FormatFakePredictor.last_predict_kwargs == {"majority_vote": True, "vote_k": 4}


def test_predict_greedy_is_default(monkeypatch, tmp_path) -> None:
    result = _invoke_predict(monkeypatch, tmp_path, [])

    assert result.exit_code == 0, result.output
    assert _FormatFakePredictor.last_predict_kwargs == {"majority_vote": False, "vote_k": 8}
