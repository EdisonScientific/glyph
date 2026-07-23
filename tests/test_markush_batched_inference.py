from __future__ import annotations

from typing import Any

import torch

from glyph.markush.inference import MarkushPredictor


def test_predict_many_matches_sequential_exactly_for_variable_inputs(monkeypatch) -> None:
    """Greedy batching preserves raw text and parsed CXSMILES for 8 images."""

    raw_by_image = {
        f"image-{index}.png": (
            "<markush><cxsmi>" + "C" * (index + 1) + "</cxsmi><stable></stable></markush>"
        )
        for index in range(8)
    }
    calls: list[dict[str, Any]] = []

    class _Tokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 9

        @staticmethod
        def decode(token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return raw_by_image[f"image-{int(token_ids[0])}.png"]

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

        @staticmethod
        def apply_chat_template(messages: list[dict[str, Any]], **kwargs: Any) -> str:
            prompt = messages[0]["content"][1]["text"]
            return "<image>" + prompt

        def __call__(
            self,
            *,
            text: list[str],
            images: list[object],
            return_tensors: str,
            padding: bool,
        ) -> dict[str, torch.Tensor]:
            assert self.tokenizer.padding_side == "left"
            assert return_tensors == "pt"
            assert padding is True
            calls.append({"text": text, "images": images})
            lengths = [len(value) + row for row, value in enumerate(text)]
            width = max(lengths)
            input_ids = torch.zeros((len(text), width), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row, length in enumerate(lengths):
                input_ids[row, -length:] = 1
                attention_mask[row, -length:] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _Model:
        @staticmethod
        def generate(**kwargs: Any) -> torch.Tensor:
            input_ids = kwargs["input_ids"]
            assert kwargs["attention_mask"].shape == input_ids.shape
            assert kwargs["do_sample"] is False
            suffix = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(1)
            return torch.cat([input_ids, suffix], dim=1)

    def fake_process_vision_info(
        message_batches: list[list[dict[str, Any]]],
        *,
        image_patch_size: int,
    ) -> tuple[list[str], None]:
        assert image_patch_size == 14
        return [messages[0]["content"][0]["image"] for messages in message_batches], None

    monkeypatch.setattr("qwen_vl_utils.process_vision_info", fake_process_vision_info)
    monkeypatch.setattr(
        "glyph.markush.eval.checkpoint.generate_one",
        lambda model, processor, record, **kwargs: raw_by_image[record.image_path],
    )
    predictor = MarkushPredictor(checkpoint="adapter", device="cpu")
    predictor._model = _Model()
    predictor._processor = _Processor()
    images = list(raw_by_image)

    batched = predictor.predict_many(images, batch_size=8, prompt="override prompt")
    sequential = [predictor.predict(image, prompt="override prompt") for image in images]

    assert [result.raw for result in batched] == [result.raw for result in sequential]
    assert [result.cxsmiles_opt for result in batched] == [
        result.cxsmiles_opt for result in sequential
    ]
    assert [result.asdict() for result in batched] == [result.asdict() for result in sequential]
    assert len(calls) == 1
    assert calls[0]["text"] == ["<image>override prompt"] * 8
    assert predictor._processor.tokenizer.padding_side == "right"


def test_predict_many_chunks_generation_and_rejects_bad_batch_size(monkeypatch) -> None:
    predictor = MarkushPredictor(checkpoint="adapter")
    predictor._model = object()
    predictor._processor = object()
    chunks: list[list[str]] = []

    def fake_generate(
        model: object,
        processor: object,
        images: list[str],
        **kwargs: Any,
    ) -> list[str]:
        chunks.append(images)
        return ["CC"] * len(images)

    monkeypatch.setattr("glyph.markush.inference._generate_greedy_batch", fake_generate)

    results = predictor.predict_many([f"{index}.png" for index in range(9)], batch_size=4)

    assert chunks == [
        ["0.png", "1.png", "2.png", "3.png"],
        ["4.png", "5.png", "6.png", "7.png"],
        ["8.png"],
    ]
    assert [result.cxsmiles_opt for result in results] == ["CC"] * 9

    try:
        predictor.predict_many(["0.png"], batch_size=0)
    except ValueError as exc:
        assert str(exc) == "batch_size must be positive"
    else:  # pragma: no cover
        raise AssertionError("expected batch_size validation")
