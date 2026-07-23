"""Inference utilities for image-to-SMILES prediction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .baseline_config import BaselineRecipe
from .eval import canonical_smiles
from .model import OCSRModel, OCSRModelConfig
from .postprocess import postprocess_smiles
from .smiles_tokenizer import CharSmilesTokenizer

# Default checkpoint source. Downloaded from the Hugging Face Hub at the repo's
# current ``main`` (no revision is pinned here), so headline reproduction assumes
# the currently-published weights rather than a byte-exact revision.
DEFAULT_HF_REPO = "EdisonScientific/OCSRGlyph"
DEFAULT_HF_FILENAME = "model.pth"
DEFAULT_HF_FILENAME_INT8 = "model_int8.pth"
MAX_DECODE_LEN = 256


ImageInput = str | Path | Image.Image


def _cuda_is_usable() -> bool:
    """Check that CUDA is available AND the current GPU is supported by this PyTorch build."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        return False


def _recipe_dict(state: dict[str, Any]) -> dict[str, Any]:
    recipe = state.get("recipe") or {}
    if is_dataclass(recipe):
        return asdict(recipe)
    if isinstance(recipe, dict):
        return dict(recipe)
    return {}


def _build_recipe(values: dict[str, Any]) -> BaselineRecipe:
    """Construct a BaselineRecipe from checkpoint metadata plus safe defaults."""

    recipe = BaselineRecipe(backbone_pretrained=False)
    for name in (
        "encoder_name",
        "input_size",
        "embed_dim",
        "dec_num_layers",
        "dec_attn_heads",
        "dec_ff_dim",
        "max_target_len",
        "vocab_size",
        "batch_size",
        "seed",
        "pad_id",
        "sos_id",
        "eos_id",
    ):
        if name in values and values[name] is not None:
            setattr(recipe, name, values[name])
    if "max_len" in values and "max_target_len" not in values:
        recipe.max_target_len = values["max_len"]
    recipe.backbone_pretrained = False
    return recipe


def _build_model(recipe: BaselineRecipe) -> OCSRModel:
    cfg = OCSRModelConfig(
        encoder_name=recipe.encoder_name,
        input_size=recipe.input_size,
        embed_dim=recipe.embed_dim,
        dec_num_layers=recipe.dec_num_layers,
        dec_attn_heads=recipe.dec_attn_heads,
        dec_ff_dim=recipe.dec_ff_dim,
        max_len=min(recipe.max_target_len, MAX_DECODE_LEN),
        vocab_size=recipe.vocab_size,
        batch_size=recipe.batch_size,
        seed=recipe.seed,
        pad_id=recipe.pad_id,
        sos_id=recipe.sos_id,
        eos_id=recipe.eos_id,
        backbone_pretrained=False,
    )
    return OCSRModel(cfg)


def _looks_like_hf_repo_id(checkpoint: str | Path) -> bool:
    value = str(checkpoint)
    path = Path(value)
    return not path.exists() and "/" in value and path.suffix == ""


def _download_hf_checkpoint(repo_id: str, quantize_mode: str | None) -> str:
    from huggingface_hub import hf_hub_download

    # The hosted INT8 artifact is a single fixed-mode file (``model_int8.pth``);
    # both ``full`` and ``decoder`` requests resolve to it. Its baked-in mode is
    # reconciled against the requested mode after load (see
    # ``_reconcile_quantize_mode``), which rejects an unsatisfiable mismatch.
    filename = DEFAULT_HF_FILENAME_INT8 if quantize_mode else DEFAULT_HF_FILENAME
    try:
        return hf_hub_download(repo_id, filename)
    except Exception as exc:
        message = str(exc)
        if "401" in message or "403" in message or "Unauthorized" in message:
            raise RuntimeError(
                f"Authentication required to download from {repo_id}. "
                "Run `huggingface-cli login` or set HF_TOKEN environment variable."
            ) from exc
        raise


def _resolve_checkpoint(checkpoint: str | Path | None, quantize_mode: str | None) -> str | Path:
    if checkpoint is None:
        return _download_hf_checkpoint(DEFAULT_HF_REPO, quantize_mode=quantize_mode)
    if isinstance(checkpoint, str) and _looks_like_hf_repo_id(checkpoint):
        return _download_hf_checkpoint(checkpoint, quantize_mode=quantize_mode)
    return checkpoint


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    path = Path(image)
    with Image.open(path) as img:
        img.load()
        return img.convert("RGB")


def _preprocess_image(image: ImageInput, input_size: int) -> torch.Tensor:
    """Return a CHW tensor using the benchmark/training [-1, 1] normalization."""

    img = _load_image(image)
    if img.size != (input_size, input_size):
        img = img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _greedy_batch(
    model: OCSRModel,
    images: torch.Tensor,
    tokenizer: CharSmilesTokenizer,
    max_len: int,
) -> list[str]:
    model.eval()
    device = images.device
    with torch.no_grad():
        memory = model.encoder(images)
        ids = torch.full((images.shape[0], 1), tokenizer.SOS_ID, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            logits = model.decoder(ids, memory)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if bool((nxt.squeeze(-1) == tokenizer.EOS_ID).all().item()):
                break
    return [tokenizer.decode(r) for r in ids.cpu().tolist()]


def _quantization_mode_arg(quantize: bool | str | None) -> str | None:
    if quantize is None or quantize is False:
        return None
    if quantize is True:
        return "full"
    if quantize in {"full", "decoder"}:
        return quantize
    raise ValueError("quantize must be False, None, True, 'full', or 'decoder'")


def _precision_arg(precision: str | None) -> str | None:
    if precision is None or precision == "auto":
        return None
    if precision in {"fp32", "fp16", "bf16"}:
        return precision
    raise ValueError("precision must be None, 'auto', 'fp32', 'fp16', or 'bf16'")


def _precision_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _quantize_model(model: OCSRModel, mode: str = "full") -> None:
    """Apply INT8 dynamic quantization.

    Args:
        mode: "full" (encoder+decoder) or "decoder" (decoder only).
    """

    if mode not in {"full", "decoder"}:
        raise ValueError("mode must be 'full' or 'decoder'")
    if mode == "full":
        model.encoder = torch.ao.quantization.quantize_dynamic(
            model.encoder,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    model.decoder = torch.ao.quantization.quantize_dynamic(
        model.decoder,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def _quantization_mode(state: dict[str, Any]) -> str | None:
    """Return the quantization mode stored in checkpoint, or None."""

    q = state.get("quantized", "")
    if q == "dynamic_full_int8":
        return "full"
    if q == "dynamic_decoder_int8":
        return "decoder"
    if state.get("decoder_dynamic_quantized"):
        return "decoder"
    return None


def _reconcile_quantize_mode(requested: str | None, stored: str | None) -> str | None:
    """Reconcile the requested quantize mode with a checkpoint's stored mode.

    A pre-quantized INT8 checkpoint bakes in a fixed mode (``"full"`` or
    ``"decoder"``); it cannot be re-quantized to a different mode. The hosted
    ``model_int8.pth`` artifact is a single fixed-mode file, so requesting the
    *other* mode (e.g. ``quantize="decoder"`` against a ``full`` int8 upload)
    cannot be honoured. Detect that mismatch and reject it with an actionable
    message instead of silently loading the stored mode.

    Returns the effective mode to use: the stored mode when the checkpoint is
    pre-quantized, otherwise the requested mode (applied on the fly).
    """

    if stored is not None:
        if requested is not None and requested != stored:
            raise ValueError(
                f"Requested quantize={requested!r} but the checkpoint is already "
                f"INT8-quantized as {stored!r}. A pre-quantized checkpoint cannot be "
                f"re-quantized to a different mode; request quantize={stored!r} to use it "
                f"as-is, or start from an fp32 checkpoint and quantize to {requested!r}."
            )
        return stored
    return requested


class OCSRPredictor:
    """Image → SMILES predictor.

    Security: the checkpoint is deserialized with ``torch.load(...,
    weights_only=False)``, which unpickles arbitrary Python objects and can
    execute code embedded in a malicious ``.pth``. Only load checkpoints from
    sources you trust.

    Args:
        checkpoint: Path to a ``.pth`` checkpoint, or a HuggingFace repo id
            such as ``EdisonScientific/OCSRGlyph``. If omitted, the
            default OCSRGlyph checkpoint is downloaded.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
        quantize: False/None for floating-point inference, True/"full" for
            encoder+decoder INT8, or "decoder" for decoder-only INT8. Dynamic
            quantization is CPU-only, so auto device selection uses CPU when
            quantization is enabled. When the checkpoint is already INT8
            (e.g. the hosted ``model_int8.pth``), the requested mode must match
            the checkpoint's baked-in mode — a pre-quantized checkpoint cannot
            be re-quantized to a different mode.
        precision: "fp32", "fp16", "bf16", or None/"auto". Auto uses fp16 on
            CUDA and fp32 on CPU.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        quantize: bool | str | None = False,
        precision: str | None = None,
    ):
        quantize_mode = _quantization_mode_arg(quantize)
        precision_arg = _precision_arg(precision)
        if quantize_mode and precision_arg in {"fp16", "bf16"}:
            raise ValueError(
                "Cannot combine INT8 quantization with fp16/bf16 — quantization is CPU-only"
            )

        resolved = _resolve_checkpoint(checkpoint, quantize_mode=quantize_mode)
        state = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError("Checkpoint must be a dict containing a 'model' state_dict")

        checkpoint_quantize_mode = _quantization_mode(state)
        # Requested vs stored mode must agree: a pre-quantized INT8 checkpoint is
        # a single fixed-mode artifact and cannot be re-quantized to another mode.
        _reconcile_quantize_mode(quantize_mode, checkpoint_quantize_mode)
        if checkpoint_quantize_mode and precision_arg in {"fp16", "bf16"}:
            raise ValueError(
                "Cannot combine INT8 quantization with fp16/bf16 — quantization is CPU-only"
            )
        if (quantize_mode or checkpoint_quantize_mode) and device == "cuda":
            raise ValueError("INT8 dynamic quantization is only supported on CPU")
        if device is None:
            device = (
                "cpu"
                if quantize_mode or checkpoint_quantize_mode
                else "cuda"
                if _cuda_is_usable()
                else "cpu"
            )

        self.device = torch.device(device)
        if self.device.type != "cuda" and precision_arg in {"fp16", "bf16"}:
            raise ValueError("fp16/bf16 precision is only supported on CUDA")
        self._precision = precision_arg or ("fp16" if self.device.type == "cuda" else "fp32")
        self._input_dtype = _precision_dtype(self._precision)
        self.checkpoint = Path(resolved)
        self.recipe = _build_recipe(_recipe_dict(state))
        self.tokenizer = CharSmilesTokenizer()
        self.max_len = min(self.recipe.max_target_len, MAX_DECODE_LEN)

        model = _build_model(self.recipe)
        if checkpoint_quantize_mode:
            _quantize_model(model, checkpoint_quantize_mode)
            model.load_state_dict(state["model"])
            model.to(self.device)
        else:
            model.to(self.device)
            model.load_state_dict(state["model"])
            if quantize_mode:
                model.to("cpu")
                _quantize_model(model, quantize_mode)
                self.device = torch.device("cpu")
                self._precision = "fp32"
                self._input_dtype = torch.float32
        if not (quantize_mode or checkpoint_quantize_mode):
            if self._precision == "fp16":
                model = model.half()
            elif self._precision == "bf16":
                model = model.to(torch.bfloat16)
        model.eval()
        self.model = model

    def predict(self, image: ImageInput, *, postprocess: bool = True) -> str:
        """Predict SMILES from one image.

        Deterministic OCSR postprocessing is enabled by default so this public
        surface matches the reproduced evaluation pipeline. Pass
        ``postprocess=False`` to retain the prior canonicalization-only path.
        """

        return self.predict_batch([image], batch_size=1, postprocess=postprocess)[0]

    def predict_batch(
        self,
        images: Sequence[ImageInput],
        batch_size: int = 8,
        *,
        postprocess: bool = True,
    ) -> list[str]:
        """Predict SMILES in batches, applying OCSR postprocessing by default."""

        if not images:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        predictions: list[str] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = torch.stack(
                [_preprocess_image(image, self.recipe.input_size) for image in chunk],
                dim=0,
            ).to(device=self.device, dtype=self._input_dtype)
            decoded = _greedy_batch(self.model, batch, self.tokenizer, self.max_len)
            if postprocess:
                predictions.extend(postprocess_smiles(smiles) or smiles for smiles in decoded)
            else:
                predictions.extend(canonical_smiles(smiles) or smiles for smiles in decoded)
        return predictions

    @staticmethod
    def quantize_checkpoint(
        input_path: str,
        output_path: str,
        mode: str = "full",
    ) -> dict[str, float | int | str]:
        """Apply INT8 dynamic quantization and save a new checkpoint.

        Security: ``input_path`` is deserialized with ``torch.load(...,
        weights_only=False)``, which unpickles arbitrary Python objects. Only
        quantize checkpoints from sources you trust.
        """

        if mode not in {"full", "decoder"}:
            raise ValueError("mode must be 'full' or 'decoder'")

        input_file = Path(input_path)
        output_file = Path(output_path)
        state = torch.load(input_file, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError("Checkpoint must be a dict containing a 'model' state_dict")

        recipe = _build_recipe(_recipe_dict(state))
        model = _build_model(recipe)
        checkpoint_quantize_mode = _quantization_mode(state)
        if checkpoint_quantize_mode:
            _quantize_model(model, checkpoint_quantize_mode)
            model.load_state_dict(state["model"])
            if checkpoint_quantize_mode != mode:
                raise ValueError(
                    f"Cannot convert {checkpoint_quantize_mode!r} quantized checkpoint "
                    f"to {mode!r}; start from an fp32 checkpoint."
                )
        else:
            model.load_state_dict(state["model"])
            _quantize_model(model, mode)
        model.eval()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        out_state = dict(state)
        out_state["model"] = model.state_dict()
        out_state["decoder_dynamic_quantized"] = mode == "decoder"
        out_state["quantized"] = f"dynamic_{mode}_int8"
        torch.save(out_state, output_file)

        input_size = input_file.stat().st_size
        output_size = output_file.stat().st_size
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "input_size_bytes": input_size,
            "output_size_bytes": output_size,
            "compression_ratio": output_size / input_size if input_size else 0.0,
            "size_reduction_bytes": input_size - output_size,
            "mode": mode,
        }
