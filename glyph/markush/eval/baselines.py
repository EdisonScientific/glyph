"""Frontier-model baselines for the Markush benchmarks, via litellm.

This module feeds the SAME benchmark input MarkushGlyph sees (the benchmark
image plus the exact ``visible_sg`` Markush prompt) to a frontier model through
`litellm <https://github.com/BerriAI/litellm>`_, parses the raw text with the
SAME extraction the offline eval uses (:mod:`glyph.markush.eval.common`), and
writes a ``predictions.json`` in the exact schema
``glyph markush eval benchmark`` emits. That predictions file scores unchanged
under both shipping contracts:

* the official DS4SD ``get_scores`` grader (``tools/mg2_scorer/score_benchmark.sh``), and
* the strict/ours graph grader (:func:`glyph.markush.grading.grade_files`,
  ``protocol="both"``).

Nothing here re-implements a scorer. The prompt, ground truth, and prediction
parsing are all pulled from the model's own eval path so the frontier baseline
and MarkushGlyph are measured through one pipeline.

Durability
----------
Predictions are checkpointed per row to a JSONL as they complete, so a run is
resumable: on restart, rows already predicted are skipped and only failed or
unattempted rows are retried. API calls (the slow, failure-prone step) use
litellm retries with exponential backoff for rate limits, and no completed
prediction is lost on a crash.

litellm is intentionally NOT a core runtime dependency. It is declared under the
optional ``baseline`` extra and imported lazily, so importing this module (for
parsing, checkpointing, or scoring) never requires it.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from glyph.markush.data.prompt_policies import MARKUSH_PROMPT_POLICY_VISIBLE_SG
from glyph.markush.eval import common
from glyph.markush.eval.benchmark_data import (
    MARKUSH_BENCHMARK_NAMES,
    materialize_benchmark,
    normalize_benchmark_name,
)
from glyph.markush.inference import DEFAULT_MARKUSH_PROMPT

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

#: Friendly codenames mapped to their exact litellm ``provider/model`` strings.
#: A value already containing ``/`` is treated as a litellm string and passed
#: through untouched, so any provider litellm supports works without an alias.
MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4-8": "anthropic/claude-opus-4-8",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "claude-fable-5": "anthropic/claude-fable-5",
}


def resolve_litellm_model(name: str) -> str:
    """Resolve a codename or litellm string to the exact litellm model string."""

    key = name.strip()
    if "/" in key:
        return key
    return MODEL_ALIASES.get(key, key)


def model_slug(name: str) -> str:
    """Return a stable filesystem slug for a model (provider prefix stripped)."""

    resolved = resolve_litellm_model(name)
    return resolved.split("/", 1)[-1]


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

#: The exact prompt policy MarkushGlyph is evaluated under. The per-row prompt
#: already carries the target-format line; the policy only pins the Sg guidance.
BASELINE_PROMPT_POLICY = MARKUSH_PROMPT_POLICY_VISIBLE_SG

VARIANT_DEFAULT = "default"
VARIANT_FORMAT = "format"
PROMPT_VARIANTS = (VARIANT_DEFAULT, VARIANT_FORMAT)

#: Appended for the ``format`` variant. The base prompt already names the target
#: envelope; models overwhelmingly return it cleanly, so the residual failures
#: are the CXSMILES *dialect* inside ``<cxsmi>``. The official DS4SD contract
#: decodes the ``cxsmiles_opt`` inline-label dialect (``<r>LABEL</r>`` written
#: immediately before the atom a group attaches to, R-group fragments joined by
#: ``.``), not the standard ``[1*] |$...$|`` form. This variant spells out that
#: dialect — the same contract MarkushGlyph is trained against, described with a
#: neutral example that reveals no answer — and forbids extra text.
FORMAT_VARIANT_INSTRUCTION = (
    "\n\nReturn ONLY one line, in exactly this format, and nothing else. "
    "No prose, no explanation, no code fences, no surrounding text:\n"
    "<markush><cxsmi>CXSMILES_OPT</cxsmi><stable>STABLE</stable></markush>\n"
    "CXSMILES_OPT must use the cxsmiles_opt inline-label dialect:\n"
    "- Mark every variable group with an inline tag <r>LABEL</r> placed "
    "immediately before the atom it attaches to, using the label drawn in the "
    "image (for example <r>R1</r>, <r>X</r>); use <r>_AP</r> for a bare "
    "attachment point with no printed label.\n"
    "- Write each pendant R-group as its own '.'-separated fragment, for "
    "example: <r>R1</r>C.<r>R2</r>C.<r>R3</r>Nc1ccc(<r>X</r>)cc1\n"
    "- Use only real element symbols; never invent placeholder atoms "
    "(no A, X, Y, Z, L, Q, w as atoms — those belong inside <r>...</r> tags).\n"
    "- Append an |Sg:...| block ONLY when the image shows a polymer bracket.\n"
    "- Do NOT use [1*]/[2*] dummy atoms or a trailing |$...$| label block.\n"
    "STABLE lists R-group label definitions if the image shows any; leave it as "
    "<stable></stable> when there are none."
)


def apply_prompt_variant(prompt: str, variant: str) -> str:
    """Return the model-facing prompt for a variant.

    ``default`` returns the exact MarkushGlyph prompt unchanged. ``format``
    appends :data:`FORMAT_VARIANT_INSTRUCTION`.
    """

    if variant == VARIANT_DEFAULT:
        return prompt
    if variant == VARIANT_FORMAT:
        return prompt + FORMAT_VARIANT_INSTRUCTION
    raise ValueError(f"Unknown prompt variant {variant!r}; expected one of: {PROMPT_VARIANTS}")


# ---------------------------------------------------------------------------
# Image encoding + message construction
# ---------------------------------------------------------------------------

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def image_data_url(image_path: str | Path) -> str:
    """Encode an image file as a base64 ``data:`` URL for a multimodal message."""

    path = Path(image_path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def build_messages(image_path: str | Path, prompt: str) -> list[dict[str, Any]]:
    """Build a one-turn multimodal litellm message: image first, then prompt."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Frontier model call
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelCall:
    """Outcome of one frontier-model call."""

    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str = ""


CompletionFn = Callable[..., Any]


def _default_completion_fn() -> CompletionFn:
    """Return ``litellm.completion``, importing litellm lazily.

    litellm is an optional extra; importing it only when a call is actually made
    keeps the base runtime (parsing, checkpointing, scoring) litellm-free.
    """

    try:
        import litellm
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised without the extra
        raise ModuleNotFoundError(
            "litellm is required for frontier baselines. Install the optional extra: "
            "`uv pip install 'glyph[baseline]'` or `pip install litellm`."
        ) from exc
    # Drop provider-unsupported params (for example a fixed temperature on
    # reasoning models) rather than raising, so one call signature spans
    # heterogeneous providers.
    litellm.drop_params = True
    return litellm.completion


def call_frontier_model(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 8000,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    timeout: float = 180.0,
    num_retries: int = 6,
    completion_fn: CompletionFn | None = None,
) -> ModelCall:
    """Call a frontier model via litellm and return the raw text + usage.

    Retries with exponential backoff are delegated to litellm (``num_retries``).
    A raised exception after retries is captured in :attr:`ModelCall.error`
    rather than propagated, so a single bad row never aborts a long run; the
    caller decides whether to persist or retry it.
    """

    fn = completion_fn or _default_completion_fn()
    kwargs: dict[str, Any] = {
        "model": resolve_litellm_model(model),
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "num_retries": num_retries,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    started = time.time()
    try:
        response = fn(**kwargs)
    except Exception as exc:
        return ModelCall(
            error=f"{type(exc).__name__}: {exc}", latency_s=round(time.time() - started, 3)
        )

    latency = round(time.time() - started, 3)
    try:
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        return ModelCall(
            error=f"malformed response: {type(exc).__name__}: {exc}", latency_s=latency
        )

    usage: dict[str, Any] = {}
    raw_usage = getattr(response, "usage", None)
    if raw_usage is not None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(raw_usage, key, None)
            if value is not None:
                usage[key] = value
    return ModelCall(raw=raw, usage=usage, latency_s=latency)


def parse_prediction(raw: str) -> tuple[str, str]:
    """Parse raw model text into ``(prediction, prediction_cxsmiles)``.

    Uses the exact eval extraction (:func:`glyph.markush.eval.common.clean_prediction`
    and :func:`~glyph.markush.eval.common.prediction_to_cxsmiles_opt`) so the
    frontier output is normalized identically to MarkushGlyph decodes.
    """

    prediction = common.clean_prediction(raw)
    prediction_cxsmiles = common.prediction_to_cxsmiles_opt(prediction)
    return prediction, prediction_cxsmiles


# ---------------------------------------------------------------------------
# Sample loading (exact MarkushGlyph prompt + ground truth)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BaselineSample:
    """One benchmark row prepared for a frontier baseline call."""

    index: int
    sample_id: str
    image_path: str
    prompt: str
    ground_truth: str


def load_baseline_samples(
    benchmark: str,
    *,
    limit: int | None = None,
    dest_root: str | Path | None = None,
    explicit_parquet: str | Path | None = None,
) -> list[BaselineSample]:
    """Load benchmark rows with the exact MarkushGlyph prompt and ground truth.

    Reuses :func:`glyph.markush.eval.benchmark_data.materialize_benchmark` and
    :func:`glyph.markush.eval.fast_eval.load_eval_samples` under the frozen
    ``visible_sg`` prompt policy, so the prompt string per row is byte-identical
    to the MarkushGlyph benchmark run.
    """

    # fast_eval imports torch at module load; keep it lazy so importing this
    # module for parsing/scoring never requires the ML stack.
    from glyph.markush.eval import fast_eval

    canonical = normalize_benchmark_name(benchmark)
    eval_jsonl, image_root = materialize_benchmark(
        canonical, dest_root, explicit_parquet=explicit_parquet
    )
    eval_samples = fast_eval.load_eval_samples(
        eval_jsonl,
        image_root,
        DEFAULT_MARKUSH_PROMPT,
        limit=limit,
        markush_prompt_policy=BASELINE_PROMPT_POLICY,
    )
    return [
        BaselineSample(
            index=sample.index,
            sample_id=sample.sample_id,
            image_path=str(sample.image_path),
            prompt=sample.prompt,
            ground_truth=sample.ground_truth,
        )
        for sample in eval_samples
    ]


# ---------------------------------------------------------------------------
# Per-row checkpointing (resumable)
# ---------------------------------------------------------------------------


def load_checkpoint(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a checkpoint JSONL into ``{sample_id: row}`` (last write wins)."""

    checkpoint_path = Path(path)
    rows: dict[str, dict[str, Any]] = {}
    if not checkpoint_path.is_file():
        return rows
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sample_id = row.get("sample_id")
        if sample_id is not None:
            rows[str(sample_id)] = row
    return rows


def _is_completed(row: dict[str, Any] | None) -> bool:
    """A row is done when it holds a successful (non-error) response."""

    return bool(row) and not row.get("error")


def _pending_samples(
    samples: Iterable[BaselineSample],
    checkpoint: dict[str, dict[str, Any]],
) -> list[BaselineSample]:
    return [s for s in samples if not _is_completed(checkpoint.get(s.sample_id))]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GenerationConfig:
    """Knobs for a single (model, benchmark, variant) generation run."""

    model: str
    benchmark: str
    variant: str = VARIANT_DEFAULT
    max_tokens: int = 8000
    temperature: float | None = None
    reasoning_effort: str | None = None
    timeout: float = 180.0
    num_retries: int = 6
    concurrency: int = 8


def _predict_one(
    sample: BaselineSample,
    config: GenerationConfig,
    *,
    completion_fn: CompletionFn | None,
) -> dict[str, Any]:
    prompt = apply_prompt_variant(sample.prompt, config.variant)
    messages = build_messages(sample.image_path, prompt)
    call = call_frontier_model(
        config.model,
        messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
        timeout=config.timeout,
        num_retries=config.num_retries,
        completion_fn=completion_fn,
    )
    prediction, prediction_cxsmiles = parse_prediction(call.raw)
    return {
        "index": sample.index,
        "sample_id": sample.sample_id,
        "image_path": sample.image_path,
        "ground_truth": sample.ground_truth,
        "model": resolve_litellm_model(config.model),
        "variant": config.variant,
        "prompt": prompt,
        "raw": call.raw,
        "prediction": prediction,
        "prediction_cxsmiles": prediction_cxsmiles,
        "usage": call.usage,
        "latency_s": call.latency_s,
        "error": call.error,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def generate_predictions(
    config: GenerationConfig,
    *,
    checkpoint_path: str | Path,
    limit: int | None = None,
    dest_root: str | Path | None = None,
    explicit_parquet: str | Path | None = None,
    completion_fn: CompletionFn | None = None,
    progress: bool = True,
    samples: list[BaselineSample] | None = None,
) -> list[dict[str, Any]]:
    """Generate (or resume) per-row predictions, checkpointing as we go.

    Returns the full ordered list of checkpoint rows (previously done + newly
    generated). Only successful rows are treated as complete on resume; rows
    that errored are retried on the next run.
    """

    if samples is None:
        samples = load_baseline_samples(
            config.benchmark, limit=limit, dest_root=dest_root, explicit_parquet=explicit_parquet
        )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(checkpoint_path)
    pending = _pending_samples(samples, done)

    write_lock = threading.Lock()
    handle = checkpoint_path.open("a", encoding="utf-8")

    def _flush(row: dict[str, Any]) -> None:
        with write_lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done[row["sample_id"]] = row

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(
                total=len(pending),
                desc=f"{model_slug(config.model)}/{config.benchmark}/{config.variant}",
            )
        except ModuleNotFoundError:
            bar = None

    try:
        if config.concurrency <= 1:
            for sample in pending:
                _flush(_predict_one(sample, config, completion_fn=completion_fn))
                if bar is not None:
                    bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
                futures = [
                    pool.submit(_predict_one, sample, config, completion_fn=completion_fn)
                    for sample in pending
                ]
                # Flush each row the moment it completes (not in submission order),
                # so an interrupt can never strand a finished-but-unwritten result.
                for future in as_completed(futures):
                    _flush(future.result())
                    if bar is not None:
                        bar.update(1)
    finally:
        if bar is not None:
            bar.close()
        handle.close()

    ordered = sorted(
        (done[s.sample_id] for s in samples if s.sample_id in done),
        key=lambda row: row.get("index", 0),
    )
    return ordered


# ---------------------------------------------------------------------------
# Artifact assembly (predictions.json + grading JSONLs)
# ---------------------------------------------------------------------------


def _prediction_sample_payload(row: dict[str, Any]) -> dict[str, Any]:
    """One ``samples[]`` entry matching the fast_eval PredictionResult schema."""

    prediction = row.get("prediction", "")
    prediction_cxsmiles = row.get("prediction_cxsmiles", "")
    return {
        "index": row.get("index", 0),
        "sample_id": row.get("sample_id"),
        "image_path": row.get("image_path"),
        "ground_truth": row.get("ground_truth", ""),
        "prediction": prediction,
        "prediction_cxsmiles": prediction_cxsmiles,
        "candidates": [
            {
                "prediction": prediction,
                "prediction_cxsmiles": prediction_cxsmiles,
                "logprob_sum": None,
                "token_count": None,
                "mean_logprob": None,
                "canonical_signature": "",
                "valid_for_vote": bool(prediction_cxsmiles),
                "vote_error": "",
            }
        ],
        "majority_vote": {"selection_source": "frontier_baseline"},
        "error": row.get("error", ""),
        "gpu": None,
        "mcs_score": None,
    }


def build_predictions_json(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    config: GenerationConfig,
) -> Path:
    """Write a ``predictions.json`` in the ``glyph markush eval benchmark`` schema.

    The top-level ``samples[]`` list carries the same per-row fields the offline
    eval emits, so ``tools/mg2_scorer/score_benchmark.sh`` scores it unchanged.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "generator": "glyph.markush.eval.baselines",
            "model": resolve_litellm_model(config.model),
            "benchmark": normalize_benchmark_name(config.benchmark),
            "prompt_variant": config.variant,
            "markush_prompt_policy": BASELINE_PROMPT_POLICY,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "num_retries": config.num_retries,
            "concurrency": config.concurrency,
        },
        "aggregate": {"size": len(rows), "skipped_mcs": True},
        "majority_vote_stats": {},
        "timing": {},
        "inference_stats": {
            "n_samples": len(rows),
            "n_errors": sum(1 for row in rows if row.get("error")),
        },
        "samples": [_prediction_sample_payload(row) for row in rows],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_grading_jsonls(
    rows: list[dict[str, Any]],
    *,
    predictions_path: str | Path,
    ground_truth_path: str | Path,
) -> tuple[Path, Path]:
    """Write the prediction + ground-truth JSONLs consumed by ``grade_files``.

    Prediction rows are ``{id, prediction}``; ground-truth rows are
    ``{id, cxsmiles_opt}`` — the exact shapes MarkushGlyph's strict/ours scoring
    joins (see :func:`glyph.markush.eval.fast_eval.run_mcs_eval`).
    """

    predictions_path = Path(predictions_path)
    ground_truth_path = Path(ground_truth_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

    with predictions_path.open("w", encoding="utf-8") as pred_handle:
        for row in rows:
            pred_handle.write(
                json.dumps(
                    {"id": row.get("sample_id"), "prediction": row.get("prediction", "")},
                    ensure_ascii=False,
                )
                + "\n"
            )
    with ground_truth_path.open("w", encoding="utf-8") as gt_handle:
        for row in rows:
            gt_handle.write(
                json.dumps(
                    {"id": row.get("sample_id"), "cxsmiles_opt": row.get("ground_truth", "")},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return predictions_path, ground_truth_path


# ---------------------------------------------------------------------------
# Scoring (reuse both shipping contracts unchanged)
# ---------------------------------------------------------------------------


def _count_true(details: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in details if row.get(key))


def score_strict_ours(
    predictions_jsonl: str | Path,
    ground_truth_jsonl: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score with the strict/ours grader via :func:`grade_files` (``both``).

    Returns a compact ``{ours, mg2}`` count/size/pct summary alongside the full
    ``grade_files`` payload.
    """

    from glyph.markush.grading import grade_files

    payload = grade_files(
        predictions_jsonl,
        ground_truth_jsonl,
        protocol="both",
        output_path=output_path,
    )
    per_sample = payload.get("per_sample", {})
    summary: dict[str, Any] = {}
    for protocol in ("ours", "mg2"):
        details = per_sample.get(protocol, [])
        size = len(details)
        count = _count_true(details, "cxsmi_equality")
        summary[protocol] = {
            "cxsmi_equality_count": count,
            "size": size,
            "cxsmi_equality_pct": (100.0 * count / size) if size else 0.0,
        }
    return {"summary": summary, "metrics": payload.get("metrics", {})}


def score_ds4sd(
    benchmark: str,
    predictions_json: str | Path,
    *,
    output_json: str | Path | None = None,
    repo_root: str | Path | None = None,
    gt_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Score with the official DS4SD grader via ``tools/mg2_scorer/score_benchmark.sh``.

    Runs the shipped script unmodified (it fetches the pinned DS4SD scorer and
    builds its py3.11 venv), then reads back the headline exact-match count/size
    from the scored JSON.

    The benchmark ground truth is resolved here (via
    :func:`glyph.markush.eval.benchmark_data.resolve_benchmark_gt`) and always
    passed through the script's ``--gt-jsonl`` override, so scoring never depends
    on the base ``uv run`` environment carrying the Markush stack.
    """

    import subprocess

    from glyph.markush.eval.benchmark_data import resolve_benchmark_gt

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    script = root / "tools" / "mg2_scorer" / "score_benchmark.sh"
    if not script.is_file():
        raise FileNotFoundError(f"DS4SD scorer script not found: {script}")

    predictions_json = Path(predictions_json)
    if output_json is None:
        output_json = predictions_json.with_name(predictions_json.stem + "_ds4sd_scored.json")
    output_json = Path(output_json)

    resolved_gt = Path(gt_jsonl) if gt_jsonl is not None else resolve_benchmark_gt(benchmark)

    cmd = [
        str(script),
        "--benchmark",
        normalize_benchmark_name(benchmark),
        "--input-json",
        str(predictions_json),
        "--output-json",
        str(output_json),
        "--gt-jsonl",
        str(resolved_gt),
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        # The vendored DS4SD scorer imports matplotlib; force a headless backend
        # so it runs regardless of the caller's MPLBACKEND (for example a Jupyter
        # inline backend that is invalid in the scorer's own venv).
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"DS4SD scorer failed (exit {completed.returncode}).\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    scored = json.loads(output_json.read_text(encoding="utf-8"))
    exact_counts = scored.get("public_mg2", {}).get("exact_counts", {})
    count = exact_counts.get("cxsmi_equality_count")
    size = exact_counts.get("size")
    pct = (100.0 * count / size) if count is not None and size else 0.0
    return {
        "cxsmi_equality_count": count,
        "size": size,
        "cxsmi_equality_pct": pct,
        "exact_counts": exact_counts,
        "output_json": str(output_json),
    }


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def cell_dir(out_dir: str | Path, model: str, benchmark: str) -> Path:
    """Return the durable directory for one (model, benchmark) cell."""

    return Path(out_dir) / model_slug(model) / normalize_benchmark_name(benchmark)


def run_baseline(
    config: GenerationConfig,
    *,
    out_dir: str | Path,
    limit: int | None = None,
    dest_root: str | Path | None = None,
    explicit_parquet: str | Path | None = None,
    completion_fn: CompletionFn | None = None,
    score: bool = True,
    score_ds4sd_contract: bool = True,
    progress: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate + persist + score one (model, benchmark, variant) cell.

    Writes, under ``<out_dir>/<model_slug>/<benchmark>/``:
    ``<variant>.checkpoint.jsonl`` (per-row, resumable), ``<variant>.predictions.json``
    (DS4SD-scorable), the two grading JSONLs, the per-contract score files, and a
    ``<variant>.summary.json`` roll-up (also returned).
    """

    cell = cell_dir(out_dir, config.model, config.benchmark)
    cell.mkdir(parents=True, exist_ok=True)
    variant = config.variant

    checkpoint_path = cell / f"{variant}.checkpoint.jsonl"
    rows = generate_predictions(
        config,
        checkpoint_path=checkpoint_path,
        limit=limit,
        dest_root=dest_root,
        explicit_parquet=explicit_parquet,
        completion_fn=completion_fn,
        progress=progress,
    )

    predictions_json = cell / f"{variant}.predictions.json"
    build_predictions_json(rows, predictions_json, config=config)
    grading_pred = cell / f"{variant}.grading_predictions.jsonl"
    grading_gt = cell / f"{variant}.grading_gt.jsonl"
    write_grading_jsonls(rows, predictions_path=grading_pred, ground_truth_path=grading_gt)

    summary: dict[str, Any] = {
        "model": resolve_litellm_model(config.model),
        "model_slug": model_slug(config.model),
        "benchmark": normalize_benchmark_name(config.benchmark),
        "variant": variant,
        "n_rows": len(rows),
        "n_errors": sum(1 for row in rows if row.get("error")),
        "n_parsed": sum(1 for row in rows if row.get("prediction_cxsmiles")),
        "predictions_json": str(predictions_json),
        "checkpoint": str(checkpoint_path),
        "scores": {},
    }

    if score:
        strict = score_strict_ours(
            grading_pred, grading_gt, output_path=cell / f"{variant}.strict_ours.json"
        )
        summary["scores"]["strict_ours"] = strict["summary"]
        if score_ds4sd_contract:
            ds4sd = score_ds4sd(
                config.benchmark,
                predictions_json,
                output_json=cell / f"{variant}.ds4sd_scored.json",
                repo_root=repo_root,
            )
            summary["scores"]["ds4sd"] = {
                "cxsmi_equality_count": ds4sd["cxsmi_equality_count"],
                "size": ds4sd["size"],
                "cxsmi_equality_pct": ds4sd["cxsmi_equality_pct"],
            }

    (cell / f"{variant}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def iter_variants(variant: str) -> Iterator[str]:
    """Yield the prompt variant(s) selected by a CLI ``--variant`` value."""

    if variant == "both":
        yield from PROMPT_VARIANTS
    else:
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of: default, format, both")
        yield variant


# ---------------------------------------------------------------------------
# Consolidation (one durable table over every cell)
# ---------------------------------------------------------------------------

#: Published MarkushGlyph headline under the official DS4SD paired contract
#: (see docs/reproducing.md), used as the reference row in the consolidated
#: table. ``(correct, denominator)`` for the greedy decode; ip5m also lists the
#: MV@8 number in ``note``.
MARKUSHGLYPH_REFERENCE: dict[str, dict[str, Any]] = {
    "ip5m": {"ds4sd": (511, 878), "note": "MV@8 532/878"},
    "m2s": {"ds4sd": (63, 103), "note": ""},
    "uspto_markush": {"ds4sd": (44, 74), "note": ""},
}

#: Models covered by the frontier baseline (claude-fable-5 is intentionally
#: omitted: it is gated on workspace data-retention and was not run).
CONSOLIDATION_MODELS: tuple[str, ...] = ("claude-opus-4-8", "gpt-5.6-sol")


def _pct(count: Any, size: Any) -> float | None:
    if count is None or not size:
        return None
    return round(100.0 * count / size, 2)


def _score_triplet(summary: dict[str, Any]) -> dict[str, Any]:
    scores = summary.get("scores", {})
    ds = scores.get("ds4sd", {})
    ours = scores.get("strict_ours", {}).get("ours", {})
    mg2 = scores.get("strict_ours", {}).get("mg2", {})

    def one(block: dict[str, Any]) -> dict[str, Any]:
        count = block.get("cxsmi_equality_count")
        size = block.get("size")
        return {"count": count, "size": size, "pct": _pct(count, size)}

    return {"ds4sd": one(ds), "ours": one(ours), "mg2": one(mg2)}


def consolidate_baselines(
    out_dir: str | Path,
    *,
    models: tuple[str, ...] = CONSOLIDATION_MODELS,
    benchmarks: tuple[str, ...] = MARKUSH_BENCHMARK_NAMES,
    variants: tuple[str, ...] = PROMPT_VARIANTS,
) -> dict[str, Any]:
    """Scan every ``<out_dir>/<model>/<benchmark>/<variant>.summary.json`` cell.

    Reads the per-cell summaries (which are never clobbered) into one structured
    record set. Cells with no summary yet are marked ``status="running"`` so an
    in-progress run renders cleanly. Returns a dict with ``cells`` and the
    MarkushGlyph ``reference`` block.
    """

    out_dir = Path(out_dir)
    cells: list[dict[str, Any]] = []
    for model in models:
        for benchmark in benchmarks:
            for variant in variants:
                summary_path = out_dir / model_slug(model) / benchmark / f"{variant}.summary.json"
                record: dict[str, Any] = {
                    "model": model_slug(model),
                    "benchmark": benchmark,
                    "variant": variant,
                }
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    record["status"] = "done"
                    record["n_rows"] = summary.get("n_rows")
                    record["n_errors"] = summary.get("n_errors")
                    record["n_parsed"] = summary.get("n_parsed")
                    record["scores"] = _score_triplet(summary)
                else:
                    record["status"] = "running"
                    record["scores"] = {}
                cells.append(record)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "out_dir": str(out_dir),
        "models": [model_slug(m) for m in models],
        "benchmarks": list(benchmarks),
        "variants": list(variants),
        "reference_markushglyph": {
            bench: {
                "ds4sd_count": ref["ds4sd"][0],
                "ds4sd_size": ref["ds4sd"][1],
                "ds4sd_pct": _pct(*ref["ds4sd"]),
                "note": ref.get("note", ""),
            }
            for bench, ref in MARKUSHGLYPH_REFERENCE.items()
        },
        "cells": cells,
    }


def render_summary_markdown(consolidated: dict[str, Any]) -> str:
    """Render the consolidated record set as a markdown table."""

    def cell_str(block: dict[str, Any]) -> str:
        count, size, pct = block.get("count"), block.get("size"), block.get("pct")
        if count is None or size is None:
            return "n/a"
        return f"{count}/{size} ({pct:.2f}%)"

    lines: list[str] = []
    lines.append("# Frontier-model Markush baselines")
    lines.append("")
    lines.append(f"Generated: {consolidated.get('generated_at', '')}")
    lines.append("")
    lines.append(
        "Both models run at **provider-default reasoning** (no reasoning_effort / "
        "temperature overrides) — an out-of-the-box baseline. `default` = the exact "
        "`visible_sg` prompt MarkushGlyph gets; `format` = same prompt plus the "
        "`cxsmiles_opt` `<r>`-dialect spec. **Reading:** DS4SD-official is a dialect "
        "floor (both models near-zero on `default` because neither emits the `<r>` "
        "opt dialect); **ours-strict is the fair capability measure.**"
    )
    lines.append("")
    lines.append(
        "| Model | Benchmark | Variant | DS4SD official (MG2) | ours-strict | our-mg2 | status |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for cell in consolidated["cells"]:
        scores = cell.get("scores", {})
        if cell["status"] != "done":
            note = "running (~2-3h, default reasoning)"
            lines.append(
                f"| {cell['model']} | {cell['benchmark']} | {cell['variant']} | n/a | n/a | n/a | {note} |"
            )
            continue
        lines.append(
            f"| {cell['model']} | {cell['benchmark']} | {cell['variant']} | "
            f"{cell_str(scores.get('ds4sd', {}))} | {cell_str(scores.get('ours', {}))} | "
            f"{cell_str(scores.get('mg2', {}))} | done |"
        )
    lines.append("")
    lines.append("## MarkushGlyph reference (DS4SD official, greedy)")
    lines.append("")
    lines.append("| Benchmark | Correct / N | % | note |")
    lines.append("|---|---|---|---|")
    for bench, ref in consolidated["reference_markushglyph"].items():
        lines.append(
            f"| {bench} | {ref['ds4sd_count']}/{ref['ds4sd_size']} | "
            f"{ref['ds4sd_pct']:.2f}% | {ref.get('note', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_consolidated(out_dir: str | Path, **kwargs: Any) -> tuple[Path, Path]:
    """Write ``summary_all.json`` + ``summary_all.md`` under ``out_dir``.

    Idempotent: re-run any time to refresh the consolidated table as more cells
    complete. Returns ``(json_path, markdown_path)``.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    consolidated = consolidate_baselines(out_dir, **kwargs)
    json_path = out_dir / "summary_all.json"
    md_path = out_dir / "summary_all.md"
    json_path.write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_summary_markdown(consolidated), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "BASELINE_PROMPT_POLICY",
    "CONSOLIDATION_MODELS",
    "FORMAT_VARIANT_INSTRUCTION",
    "MARKUSHGLYPH_REFERENCE",
    "MODEL_ALIASES",
    "PROMPT_VARIANTS",
    "VARIANT_DEFAULT",
    "VARIANT_FORMAT",
    "BaselineSample",
    "GenerationConfig",
    "ModelCall",
    "apply_prompt_variant",
    "build_messages",
    "build_predictions_json",
    "call_frontier_model",
    "cell_dir",
    "consolidate_baselines",
    "generate_predictions",
    "image_data_url",
    "iter_variants",
    "load_baseline_samples",
    "load_checkpoint",
    "model_slug",
    "parse_prediction",
    "render_summary_markdown",
    "resolve_litellm_model",
    "run_baseline",
    "score_ds4sd",
    "score_strict_ours",
    "write_consolidated",
    "write_grading_jsonls",
]
