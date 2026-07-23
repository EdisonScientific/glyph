from __future__ import annotations

import torch
from PIL import Image

from glyph.markush.eval.fast_eval import (
    EvalSample,
    ForceOutputPrefixLogitsProcessor,
    GeneratedCandidate,
    PredictionResult,
    build_parser,
    generate_batch,
    run_mcs_eval,
    select_majority_vote_candidate,
)
from glyph.markush.eval.metrics_mcs import evaluate_markush_mcs


def test_force_output_prefix_logits_processor_forces_initial_tokens():
    processor = ForceOutputPrefixLogitsProcessor(prompt_len=3, prefix_token_ids=[4, 5])
    scores = torch.zeros((2, 8))

    first = processor(torch.ones((2, 3), dtype=torch.long), scores)
    assert torch.isneginf(first[:, :4]).all()
    assert first[:, 4].tolist() == [0.0, 0.0]
    assert torch.isneginf(first[:, 5:]).all()

    second = processor(torch.ones((2, 4), dtype=torch.long), scores)
    assert torch.isneginf(second[:, :5]).all()
    assert second[:, 5].tolist() == [0.0, 0.0]
    assert torch.isneginf(second[:, 6:]).all()

    after_prefix = processor(torch.ones((2, 5), dtype=torch.long), scores)
    assert after_prefix is scores


def test_fast_eval_parser_accepts_skip_mcs():
    args = build_parser().parse_args(
        [
            "--base-model",
            "model",
            "--eval-data",
            "eval.jsonl",
            "--skip-mcs",
        ]
    )

    assert args.skip_mcs is True


def test_fast_eval_parser_accepts_oracle_sampling_flags():
    args = build_parser().parse_args(
        [
            "--base-model",
            "model",
            "--eval-data",
            "eval.jsonl",
            "--do-sample",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--top-k",
            "50",
            "--num-return-sequences",
            "8",
            "--seed",
            "123",
        ]
    )

    assert args.do_sample is True
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.top_k == 50
    assert args.num_return_sequences == 8
    assert args.seed == 123


def test_fast_eval_parser_accepts_markush_prompt_policy():
    args = build_parser().parse_args(
        [
            "--base-model",
            "model",
            "--eval-data",
            "eval.jsonl",
            "--markush-prompt-policy",
            "visible_sg",
        ]
    )

    assert args.markush_prompt_policy == "visible_sg"


def test_generate_batch_uses_left_padding_and_restores_training_processor():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 9
        padding_side = "right"

        def batch_decode(self, token_ids, skip_special_tokens=True):
            return ["CCO" for _ in range(token_ids.shape[0])]

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()
            self.padding_sides_seen: list[str] = []

        def apply_chat_template(self, *args, **kwargs):
            return "<image>prompt"

        def __call__(self, *, text, images, padding, return_tensors):
            self.padding_sides_seen.append(self.tokenizer.padding_side)
            assert self.tokenizer.padding_side == "left"
            return {
                "input_ids": torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.long),
            }

    class _Model:
        def __init__(self, processor: _Processor) -> None:
            self.processor = processor

        def generate(self, **kwargs):
            assert self.processor.tokenizer.padding_side == "left"
            input_ids = kwargs["input_ids"]
            suffix = torch.tensor([[4], [4]], dtype=input_ids.dtype)
            return torch.cat([input_ids, suffix], dim=1)

    processor = _Processor()
    samples = [
        EvalSample(index=0, sample_id="a", image_path=None, prompt="p0", ground_truth="CCO"),
        EvalSample(index=1, sample_id="b", image_path=None, prompt="p1", ground_truth="CCN"),
    ]

    predictions = generate_batch(
        _Model(processor),
        processor,
        samples,
        [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
        max_new_tokens=4,
        device="cpu",
    )

    assert predictions == ["CCO", "CCO"]
    assert processor.padding_sides_seen == ["left"]
    assert processor.tokenizer.padding_side == "right"


def test_run_mcs_eval_uses_high_level_markush_evaluator():
    results = [
        PredictionResult(
            index=0,
            sample_id="s0",
            image_path=None,
            ground_truth="CCO",
            prediction="<markush><cxsmi>C(C)O</cxsmi><stable></stable></markush>",
            prediction_cxsmiles="C(C)O",
        )
    ]
    expected = evaluate_markush_mcs(
        [
            {
                "id": "s0",
                "prediction": "<markush><cxsmi>C(C)O</cxsmi><stable></stable></markush>",
                "cxsmiles_opt": "C(C)O",
            }
        ],
        [{"id": "s0", "cxsmiles_opt": "CCO"}],
        include_missing=True,
        return_details=False,
        protocol="ours",
    )

    aggregate, details = run_mcs_eval(results, protocol="ours")

    assert aggregate == expected
    assert details[0] is not None
    assert details[0]["id"] == "s0"
    assert details[0]["cxsmi_equality"] is True


def test_fast_eval_parser_accepts_majority_vote_sampling_flags():
    args = build_parser().parse_args(
        [
            "--base-model",
            "model",
            "--eval-data",
            "eval.jsonl",
            "--do-sample",
            "--num-return-sequences",
            "4",
            "--majority-vote",
        ]
    )

    assert args.majority_vote is True


def test_select_majority_vote_candidate_uses_modal_signature_then_likelihood():
    candidates = [
        GeneratedCandidate(
            prediction="a",
            prediction_cxsmiles="CCO",
            canonical_signature="sig-a",
            valid_for_vote=True,
            mean_logprob=-0.9,
        ),
        GeneratedCandidate(
            prediction="invalid",
            prediction_cxsmiles="",
            valid_for_vote=False,
            vote_error="missing",
            mean_logprob=-0.1,
        ),
        GeneratedCandidate(
            prediction="b1",
            prediction_cxsmiles="CCN",
            canonical_signature="sig-b",
            valid_for_vote=True,
            mean_logprob=-0.2,
        ),
        GeneratedCandidate(
            prediction="a2",
            prediction_cxsmiles="OCC",
            canonical_signature="sig-a",
            valid_for_vote=True,
            mean_logprob=-0.8,
        ),
        GeneratedCandidate(
            prediction="b2",
            prediction_cxsmiles="NCC",
            canonical_signature="sig-b",
            valid_for_vote=True,
            mean_logprob=-0.3,
        ),
    ]

    selected, stats = select_majority_vote_candidate(candidates)

    assert selected is not None
    assert selected.prediction == "b1"
    assert stats["selected_canonical_signature"] == "sig-b"
    assert stats["invalid_count"] == 1
    assert stats["vote_count"] == 2
    assert stats["tied"] is True


def test_generate_majority_vote_reuses_consensus_voting(tmp_path):
    from glyph.markush.eval.fast_eval import generate_majority_vote

    # Four sampled candidates: two decode to the same molecule (CCO / OCC),
    # one to a different graph (CCN), one is unparseable. The modal graph wins.
    decoded = [
        "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>",
        "<markush><cxsmi>OCC</cxsmi><stable></stable></markush>",
        "<markush><cxsmi>CCN</cxsmi><stable></stable></markush>",
        "not a molecule (((",
    ]

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 9
        padding_side = "left"

        def batch_decode(self, token_ids, skip_special_tokens=True):
            return list(decoded)

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

        def apply_chat_template(self, *args, **kwargs):
            return "<image>prompt"

        def __call__(self, *, text, images, padding, return_tensors):
            return {
                "input_ids": torch.zeros((1, 3), dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

    class _Model:
        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            num = kwargs["num_return_sequences"]
            rows = input_ids.repeat(num, 1)
            suffix = torch.full((num, 1), 4, dtype=input_ids.dtype)
            return torch.cat([rows, suffix], dim=1)

    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(image_path)

    selected, vote_stats, candidates = generate_majority_vote(
        _Model(),
        _Processor(),
        str(image_path),
        "prompt",
        vote_k=4,
        max_new_tokens=4,
        device="cpu",
        seed=0,
    )

    assert len(candidates) == 4
    assert vote_stats["n_candidates"] == 4
    assert vote_stats["vote_count"] == 2  # CCO + OCC share a canonical graph
    assert vote_stats["invalid_count"] == 1
    assert selected is not None
    assert selected.prediction_cxsmiles in {"CCO", "OCC"}


def test_generate_majority_vote_rejects_nonpositive_vote_k(tmp_path):
    from glyph.markush.eval.fast_eval import generate_majority_vote

    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(image_path)

    try:
        generate_majority_vote(object(), object(), str(image_path), "prompt", vote_k=0)
    except ValueError as exc:
        assert "vote_k" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected vote_k <= 0 to raise ValueError")
