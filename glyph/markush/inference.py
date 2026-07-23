"""Public image inference API for Markush transcription."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from glyph.markush.data.pipeline import opt_to_standard_cxsmiles
from glyph.markush.eval import common

DEFAULT_MARKUSH_PROMPT = (
    "Extract the Markush structure from this image.\n"
    "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
    "the <cxsmi> value must use canonicalized cxsmiles_opt."
)
OCSR_NO_EXTENSION_PROMPT = (
    "Extract the Markush structure from this image.\n"
    "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
    "the <cxsmi> value must use canonicalized cxsmiles_opt.\n"
    "This image is an ordinary molecule with no Markush extension annotations; "
    "return only the base molecule in <cxsmi> and leave <stable> empty."
)
# Default model sources. These resolve from the Hugging Face Hub at the repo's
# current ``main`` (no revision is pinned here), so headline reproduction assumes
# the currently-published weights rather than a byte-exact revision.
DEFAULT_MARKUSH_ADAPTER = "EdisonScientific/MarkushGlyph"
DEFAULT_MARKUSH_BASE_MODEL = "Qwen/Qwen3.5-2B-Base"

# Default number of sampled candidates for majority-vote decoding.  The sampling
# recipe itself (temperature/top-p/seed) is owned by ``eval.fast_eval`` and
# resolved there so MV@8 reproduces the eval headline (see ``predict``).
DEFAULT_VOTE_K = 8


def _standard_cxsmiles(cxsmiles_opt: str) -> str:
    """Convert ``cxsmiles_opt`` to standard ``|$...$|`` CXSMILES, never raising."""

    if not cxsmiles_opt:
        return ""
    try:
        return opt_to_standard_cxsmiles(cxsmiles_opt)
    except Exception:
        return ""


def _looks_like_hf_repo_id(checkpoint: str | Path) -> bool:
    """Return ``True`` when ``checkpoint`` looks like a Hugging Face repo id.

    Mirrors the OCSR predictor heuristic: a value that is not an existing local
    path, contains a ``/`` (``org/name``), and has no file suffix.
    """

    value = str(checkpoint)
    path = Path(value)
    return not path.exists() and "/" in value and path.suffix == ""


def _download_adapter(repo_id: str) -> str:
    """Snapshot-download a released Markush adapter repo, returning its local dir."""

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id)
    except Exception as exc:
        message = str(exc)
        if "401" in message or "403" in message or "Unauthorized" in message:
            raise RuntimeError(
                f"Authentication required to download from {repo_id}. "
                "Run `huggingface-cli login` or set HF_TOKEN environment variable."
            ) from exc
        raise


def _resolve_adapter(checkpoint: str | Path | None) -> str:
    """Resolve ``checkpoint`` to a LOCAL adapter directory.

    ``None`` downloads the default released adapter
    (:data:`DEFAULT_MARKUSH_ADAPTER`); a Hugging Face repo id is
    snapshot-downloaded; an existing local path is returned as-is. The returned
    directory always contains ``adapter_config.json`` for a released adapter, so
    :func:`glyph.markush.eval.checkpoint.load_model_and_processor` reliably takes
    the LoRA-adapter branch instead of the full-model branch.
    """

    if checkpoint is None:
        return _download_adapter(DEFAULT_MARKUSH_ADAPTER)
    if _looks_like_hf_repo_id(checkpoint):
        return _download_adapter(str(checkpoint))
    return str(checkpoint)


def _resolve_base_model(base_model: str | Path | None) -> str:
    """Resolve the base model id, honoring ``MARKUSH_BASE_MODEL`` then the default."""

    if base_model:
        return str(base_model)
    return os.environ.get("MARKUSH_BASE_MODEL") or DEFAULT_MARKUSH_BASE_MODEL


@dataclass(slots=True)
class MarkushPrediction:
    """Normalized result returned by :class:`MarkushPredictor`."""

    image: str
    raw: str
    markush_xml: str
    cxsmiles_opt: str
    cxsmiles: str
    stable: str
    vote: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        image: str | Path,
        raw: str,
        *,
        vote: dict[str, Any] | None = None,
    ) -> MarkushPrediction:
        """Normalize raw model output into the public result shape."""

        cleaned = common.clean_prediction(raw)
        cxsmiles_opt, stable = common.extract_cxsmiles_and_stable(cleaned, require_wrapper=False)
        if cxsmiles_opt:
            markush_xml = (
                f"<markush><cxsmi>{cxsmiles_opt}</cxsmi><stable>{stable}</stable></markush>"
            )
        else:
            cxsmiles_opt = common.prediction_to_cxsmiles_opt(cleaned)
            stable = ""
            markush_xml = (
                f"<markush><cxsmi>{cxsmiles_opt}</cxsmi><stable></stable></markush>"
                if cxsmiles_opt
                else ""
            )
        return cls(
            image=str(image),
            raw=cleaned,
            markush_xml=markush_xml,
            cxsmiles_opt=cxsmiles_opt,
            cxsmiles=_standard_cxsmiles(cxsmiles_opt),
            stable=stable,
            vote=dict(vote) if vote else {},
        )

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def _generate_greedy_batch(
    model: Any,
    processor: Any,
    images: list[str | Path],
    *,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """Generate one greedy completion per image with a single model call."""

    import torch
    from qwen_vl_utils import process_vision_info

    from glyph.markush.eval.checkpoint import (
        ENDOFTEXT_TOKEN_ID,
        IM_END_TOKEN_ID,
        _build_messages,
        _processor_image_patch_size,
        _processor_image_pixel_kwargs,
    )
    from glyph.markush.eval.fast_eval import temporary_tokenizer_padding_side

    image_patch_size = _processor_image_patch_size(processor)
    image_pixel_kwargs = _processor_image_pixel_kwargs(processor)
    message_batches = [
        _build_messages(
            str(image),
            prompt,
            image_pixel_kwargs=image_pixel_kwargs,
        )
        for image in images
    ]

    text_prompts: list[str] = []
    for messages in message_batches:
        try:
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                text_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                text_prompt = processor.apply_chat_template(messages, tokenize=False)
        text_prompts.append(text_prompt)

    image_inputs, _ = process_vision_info(
        message_batches,
        image_patch_size=image_patch_size,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    with temporary_tokenizer_padding_side(processor, "left"):
        inputs = processor(
            text=text_prompts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        prompt_len = inputs["input_ids"].shape[1]
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(
            tokenizer, "eos_token_id", None
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID],
                use_cache=True,
            )

        generated = output_ids[:, prompt_len:]
        raw_predictions = [
            tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in generated
        ]

    if len(raw_predictions) != len(images):
        raise RuntimeError(
            f"Expected {len(images)} generated sequence(s), got {len(raw_predictions)}"
        )
    return [common.clean_prediction(raw) for raw in raw_predictions]


class MarkushPredictor:
    """Lazy single-image and batched predictor for a Markush VLM checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        base_model: str | Path | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        merge_adapter: bool = True,
        max_new_tokens: int = 1024,
        prompt: str = DEFAULT_MARKUSH_PROMPT,
    ) -> None:
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.base_model = _resolve_base_model(base_model)
        self.device = device
        self.dtype = dtype
        self.merge_adapter = merge_adapter
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model and processor are already resident in memory."""

        return self._model is not None and self._processor is not None

    def load(self) -> MarkushPredictor:
        """Load the checkpoint and processor if needed.

        Resolves the adapter to a LOCAL directory first — downloading the
        default released adapter when ``checkpoint`` is omitted, or a Hugging
        Face repo id — so ``load_model_and_processor`` always sees an on-disk
        ``adapter_config.json`` and loads LoRA weights on top of the base model.
        """

        if self.is_loaded:
            return self

        resolved_adapter = _resolve_adapter(self.checkpoint)
        self.checkpoint = resolved_adapter

        from glyph.markush.eval.checkpoint import load_model_and_processor

        self._model, self._processor = load_model_and_processor(
            self.base_model,
            resolved_adapter,
            self.device,
            dtype=self.dtype,
            merge_adapter=self.merge_adapter,
        )
        return self

    def predict(
        self,
        image: str | Path,
        *,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        majority_vote: bool = False,
        vote_k: int = DEFAULT_VOTE_K,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> MarkushPrediction:
        """Predict canonicalized Markush CXSMILES for one image path.

        With ``majority_vote=False`` (default) this is a single greedy decode
        (``do_sample=False``).  With ``majority_vote=True`` it samples ``vote_k``
        candidates and returns the modal strict-graph consensus winner, reusing
        the eval self-consistency path (:func:`glyph.markush.eval.fast_eval.
        generate_majority_vote`).  ``temperature``/``top_p``/``seed`` default to
        the canonical MV recipe from ``fast_eval`` (which reproduces the IP5-M
        MV@8 headline) when left as ``None``.
        """

        self.load()
        resolved_prompt = prompt or self.prompt
        resolved_max_new_tokens = max_new_tokens or self.max_new_tokens

        if majority_vote:
            from glyph.markush.eval.fast_eval import (
                MV_DEFAULT_SEED,
                MV_DEFAULT_TEMPERATURE,
                MV_DEFAULT_TOP_P,
                generate_majority_vote,
            )

            selected, vote_stats, _candidates = generate_majority_vote(
                self._model,
                self._processor,
                str(image),
                resolved_prompt,
                vote_k=vote_k,
                temperature=MV_DEFAULT_TEMPERATURE if temperature is None else temperature,
                top_p=MV_DEFAULT_TOP_P if top_p is None else top_p,
                max_new_tokens=resolved_max_new_tokens,
                device=self.device,
                seed=MV_DEFAULT_SEED if seed is None else seed,
            )
            raw = selected.prediction if selected is not None else ""
            return MarkushPrediction.from_raw(image, raw, vote=vote_stats)

        from glyph.markush.eval.checkpoint import EvalRecord, generate_one

        record = EvalRecord(
            sample_id=str(image),
            image_path=str(image),
            cxsmiles_opt="",
            annotation="",
            prompt=resolved_prompt,
        )
        raw = generate_one(
            self._model,
            self._processor,
            record,
            max_new_tokens=resolved_max_new_tokens,
            device=self.device,
        )
        return MarkushPrediction.from_raw(image, raw)

    def predict_many(
        self,
        images: list[str | Path],
        *,
        batch_size: int = 8,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> list[MarkushPrediction]:
        """Greedily predict images in order with one model call per batch.

        Inputs are left-padded so variable prompt and image-token lengths retain
        decoder-only generation semantics. ``prompt`` overrides the predictor
        default for every image in this call.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not images:
            return []

        self.load()
        resolved_prompt = prompt or self.prompt
        resolved_max_new_tokens = max_new_tokens or self.max_new_tokens
        predictions: list[MarkushPrediction] = []
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            raw_predictions = _generate_greedy_batch(
                self._model,
                self._processor,
                batch_images,
                prompt=resolved_prompt,
                max_new_tokens=resolved_max_new_tokens,
                device=self.device,
            )
            predictions.extend(
                MarkushPrediction.from_raw(image, raw)
                for image, raw in zip(batch_images, raw_predictions, strict=True)
            )
        return predictions


__all__ = [
    "DEFAULT_MARKUSH_ADAPTER",
    "DEFAULT_MARKUSH_BASE_MODEL",
    "DEFAULT_MARKUSH_PROMPT",
    "OCSR_NO_EXTENSION_PROMPT",
    "MarkushPrediction",
    "MarkushPredictor",
]
