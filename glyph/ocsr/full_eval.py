"""Full USPTO OCSR benchmark evaluation under three stereochemistry conventions.

This runs the released OCSRGlyph checkpoint over every row of the complete
5,719-image USPTO benchmark assembled by :mod:`glyph.ocsr.benchmark` and grades
each prediction under three increasingly lenient stereochemistry conventions:

* **A -- canonical / full-stereo.** RDKit isomeric canonical SMILES must match
  exactly (tetrahedral chirality and double-bond cis/trans geometry).
* **B -- chirality-kept.** Strip ``/`` and ``\\`` (cis/trans), then canonicalize
  with tetrahedral chirality preserved. This mirrors MolScribe's
  ``canon_smiles``.
* **C -- graph / all-stereo-removed.** Remove all stereochemistry, then
  canonicalize for connectivity-only matching.

Predictions use the released light postprocess by default: disconnected
``[H]``/``[HH]`` hydrogen fragments are stripped, then RDKit-canonicalized by
:class:`glyph.ocsr.predict.OCSRPredictor`.

The greedy decode above produces the published headline numbers. An opt-in
self-consistency **majority-vote** mode (MV@K) is also available via
``run_full_uspto_eval(..., vote_k=[8])`` and the ``glyph ocsr eval
--majority-vote`` CLI flag; it never changes the greedy numbers.

Two denominators are reported: all 5,719 fixed benchmark rows, and the original
5,704-row Hugging Face labelled subset. The eight rows whose ground truth is
unscoreable can never be counted correct on the full denominator.

Reference numbers for the released OCSRGlyph checkpoint::

    Convention   Full 5719              Parseable 5704
    A (canon)    5362 / 5719 = 93.76%   5360 / 5704 = 93.97%
    B (chiral)   5372 / 5719 = 93.93%   5370 / 5704 = 94.14%
    C (graph)    5499 / 5719 = 96.15%   5496 / 5704 = 96.35%
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard torch import
    import torch

    from .predict import OCSRPredictor

from .benchmark import SOURCE_HF
from .eval import (
    OCSR_BENCHMARK_PARQUET_PATH,
    USPTO_BENCHMARK_PARSEABLE_ROWS,
    USPTO_BENCHMARK_TOTAL_ROWS,
    EvalResult,
    canonical_smiles,
    evaluate_pairs,
    load_uspto_benchmark,
)


def load_label_rows(benchmark_csv: Path | str) -> list[dict[str, str]]:
    """Load the complete-label CSV in row order (image id, SMILES, source)."""

    path = Path(benchmark_csv).expanduser()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "smiles", "source_of_label"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path} must have columns {sorted(required)}")
    return rows


def benchmark_label_rows(benchmark_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Derive complete-label rows from a self-contained benchmark parquet.

    Maps :func:`glyph.ocsr.eval.load_uspto_benchmark` output (``id``, ``smiles``,
    ``source_of_label``) to the ``image_id, smiles, source_of_label`` shape the
    grader consumes, so the hosted parquet is scored directly with no separate
    label CSV. Rows without a recorded ``source_of_label`` default to the labelled
    HF subset.
    """

    label_rows: list[dict[str, str]] = []
    for row in benchmark_rows:
        smiles = row.get("smiles")
        source = row.get("source_of_label")
        label_rows.append(
            {
                "image_id": str(row["id"]),
                "smiles": smiles.strip() if isinstance(smiles, str) else "",
                "source_of_label": source if isinstance(source, str) and source else SOURCE_HF,
            }
        )
    return label_rows


def load_images_by_id(parquet_path: Path | str | None = None) -> dict[str, bytes]:
    """Map ``image_id -> PNG bytes`` from the HF benchmark snapshot."""

    images: dict[str, bytes] = {}
    for row in load_uspto_benchmark(parquet_path):
        image_bytes = row.get("image_bytes")
        if image_bytes is not None:
            images[str(row["id"])] = image_bytes
    return images


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode benchmark image bytes to an RGB PIL image."""

    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return image.convert("RGB")


def run_inference(
    predictor: Any,
    label_rows: list[dict[str, str]],
    images_by_id: dict[str, bytes],
    *,
    batch_size: int,
    postprocess: bool,
) -> list[str]:
    """Return one prediction per label row, in order. Missing images yield ``""``."""

    predictions: list[str] = []
    total = len(label_rows)
    start = time.time()
    for offset in range(0, total, batch_size):
        chunk = label_rows[offset : offset + batch_size]
        batch_images: list[Image.Image] = []
        present: list[int] = []
        for local_idx, row in enumerate(chunk):
            image_bytes = images_by_id.get(row["image_id"])
            if image_bytes is None:
                continue
            batch_images.append(decode_image(image_bytes))
            present.append(local_idx)

        chunk_predictions = [""] * len(chunk)
        if batch_images:
            decoded = predictor.predict_batch(
                batch_images,
                batch_size=len(batch_images),
                postprocess=postprocess,
            )
            for local_idx, prediction in zip(present, decoded, strict=True):
                chunk_predictions[local_idx] = prediction
        predictions.extend(chunk_predictions)

        done = offset + len(chunk)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"  {done}/{total} ({elapsed:.0f}s, {rate:.1f} img/s, eta {eta:.0f}s)",
            flush=True,
        )
    return predictions


def _result_block(result: EvalResult, denominator: int) -> dict[str, Any]:
    """Return counts and percentages over ``denominator`` for A/B/C."""

    def pct(count: int) -> float:
        return round(100.0 * count / denominator, 4) if denominator else 0.0

    return {
        "denominator": denominator,
        "n_scoreable_gt": result.n,
        "n_unscoreable_gt": result.n_invalid_gt,
        "n_valid_pred": result.n_valid_pred,
        "A_canon": {"n_correct": result.n_correct, "pct": pct(result.n_correct)},
        "B_chirality": {
            "n_correct": result.n_correct_chirality,
            "pct": pct(result.n_correct_chirality),
        },
        "C_graph": {
            "n_correct": result.n_correct_graph,
            "pct": pct(result.n_correct_graph),
        },
    }


def grade_uspto_predictions(
    label_rows: list[dict[str, str]],
    predictions: list[str],
) -> dict[str, Any]:
    """Grade A/B/C on all rows and the HF-labelled subset."""

    all_ground_truth = [row["smiles"] for row in label_rows]
    subset_predictions = [
        prediction
        for prediction, row in zip(predictions, label_rows, strict=True)
        if row["source_of_label"] == SOURCE_HF
    ]
    subset_ground_truth = [
        row["smiles"] for row in label_rows if row["source_of_label"] == SOURCE_HF
    ]

    full_result = evaluate_pairs(predictions, all_ground_truth)
    subset_result = evaluate_pairs(subset_predictions, subset_ground_truth)

    n_scoreable_total = sum(
        1 for ground_truth in all_ground_truth if canonical_smiles(ground_truth) is not None
    )
    return {
        "full_5719": _result_block(full_result, len(label_rows)),
        "parseable_5704": _result_block(subset_result, len(subset_predictions)),
        "counts": {
            "total_rows": len(label_rows),
            "parseable_subset_rows": len(subset_predictions),
            "scoreable_gt_rows": n_scoreable_total,
            "unscoreable_gt_rows": len(label_rows) - n_scoreable_total,
        },
    }


# ---------- Majority-vote (self-consistency) decoding ---------------------
#
# The greedy decode above produces the published OCSRGlyph headline numbers.
# Majority vote is a separate, opt-in test-time-augmentation mode: for each
# image draw ``k`` stochastic samples in a single batched decode, canonicalize
# the valid samples with the same RDKit helper the greedy scorer uses, and pick
# the modal canonical SMILES *without looking at ground truth*. Ties break by
# mean model log-likelihood. Grading reuses ``grade_uspto_predictions`` so the
# A/B/C convention math is identical to the greedy path.


def parse_vote_ks(value: str | None) -> list[int]:
    """Parse a comma-separated majority-vote K spec (e.g. ``"4,8,16"``)."""

    if not value:
        return []
    ks = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if any(k <= 0 for k in ks):
        raise ValueError("all majority-vote K values must be > 0")
    return ks


def select_majority_vote(
    candidates: list[dict[str, Any]],
    *,
    fallback_prediction: str,
    k: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Select the non-oracle modal canonical SMILES from sampled candidates.

    Invalid samples are dropped, the modal canonical form among the first ``k``
    valid candidates wins, and ties break by mean model log-likelihood. When no
    candidate is valid the greedy ``fallback_prediction`` is returned so a row is
    never scored against ground truth.
    """

    subset = candidates[:k] if k is not None else candidates
    valid = [candidate for candidate in subset if candidate.get("canonical")]
    invalid_count = len(subset) - len(valid)
    if not valid:
        return fallback_prediction, {
            "vote_count": 0,
            "invalid_count": invalid_count,
            "invalid_rate": invalid_count / len(subset) if subset else 0.0,
            "tied": False,
            "all_invalid": True,
            "selected_canonical": canonical_smiles(fallback_prediction),
            "selection_source": "greedy_fallback",
        }

    counts = Counter(str(candidate["canonical"]) for candidate in valid)
    max_votes = max(counts.values())
    tied_forms = sorted(form for form, count in counts.items() if count == max_votes)
    scores_by_form: dict[str, list[float]] = defaultdict(list)
    for candidate in valid:
        scores_by_form[str(candidate["canonical"])].append(float(candidate["mean_logprob"]))

    def tie_key(form: str) -> tuple[float, float, str]:
        scores = scores_by_form[form]
        return (sum(scores) / len(scores), max(scores), form)

    selected = max(tied_forms, key=tie_key)
    return selected, {
        "vote_count": max_votes,
        "valid_count": len(valid),
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / len(subset) if subset else 0.0,
        "tied": len(tied_forms) > 1,
        "tie_size": len(tied_forms),
        "all_invalid": False,
        "selected_canonical": selected,
        "selection_source": "majority_vote",
        "mean_logprob": sum(scores_by_form[selected]) / len(scores_by_form[selected]),
    }


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    import torch

    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
    filtered = torch.full_like(logits, -float("inf"))
    return filtered.scatter(dim=-1, index=sorted_indices, src=sorted_logits)


def _sample_candidates(
    predictor: OCSRPredictor,
    images: torch.Tensor,
    *,
    k: int,
    temperature: float,
    top_p: float,
    max_len: int,
    generator: torch.Generator | None,
    apply_postprocess: bool,
) -> list[list[dict[str, Any]]]:
    """Draw ``k`` stochastic decodes per image in a single repeated batch."""

    import torch
    import torch.nn.functional as F

    from .postprocess import postprocess_smiles

    if k <= 0:
        raise ValueError("k must be > 0")
    model = predictor.model
    tokenizer = predictor.tokenizer
    model.eval()
    device = images.device
    batch_size = images.shape[0]
    with torch.no_grad():
        memory = model.encoder(images)
        memory = memory.repeat_interleave(k, dim=0)
        ids = torch.full((batch_size * k, 1), tokenizer.SOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size * k, dtype=torch.bool, device=device)
        seq_logprob = torch.zeros(batch_size * k, dtype=torch.float, device=device)
        token_counts = torch.zeros(batch_size * k, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = model.decoder(ids, memory)[:, -1, :]
            model_log_probs = F.log_softmax(logits.float(), dim=-1)
            sample_logits = _top_p_filter(logits.float() / temperature, top_p)
            sample_probs = torch.softmax(sample_logits, dim=-1)
            nxt = torch.multinomial(sample_probs, num_samples=1, generator=generator)
            step_logprob = model_log_probs.gather(1, nxt).squeeze(1)

            active = ~finished
            seq_logprob = seq_logprob + torch.where(
                active, step_logprob, torch.zeros_like(step_logprob)
            )
            token_counts = token_counts + active.long()
            nxt = torch.where(
                finished.unsqueeze(1),
                torch.full_like(nxt, tokenizer.EOS_ID),
                nxt,
            )
            ids = torch.cat([ids, nxt], dim=1)
            finished = finished | (nxt.squeeze(1) == tokenizer.EOS_ID)
            if bool(finished.all().item()):
                break

    decoded = [tokenizer.decode(row) for row in ids.cpu().tolist()]
    scores = seq_logprob.cpu().tolist()
    lengths = token_counts.cpu().tolist()
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    for flat_idx, raw in enumerate(decoded):
        normalized = postprocess_smiles(raw) if apply_postprocess else raw
        if normalized is None:
            normalized = raw
        canon = canonical_smiles(normalized)
        grouped[flat_idx // k].append(
            {
                "raw": raw,
                "prediction": normalized,
                "canonical": canon,
                "valid": canon is not None,
                "logprob_sum": float(scores[flat_idx]),
                "token_count": int(lengths[flat_idx]),
                "mean_logprob": float(scores[flat_idx]) / max(1, int(lengths[flat_idx])),
            }
        )
    return grouped


def run_sampling(
    predictor: OCSRPredictor,
    label_rows: list[dict[str, str]],
    images_by_id: dict[str, bytes],
    *,
    batch_size: int,
    max_vote_k: int,
    temperature: float,
    top_p: float,
    seed: int,
    postprocess: bool,
) -> list[list[dict[str, Any]]]:
    """Return ``max_vote_k`` sampled candidates per label row, in order.

    Rows whose image is missing yield an empty candidate list; the caller falls
    back to the greedy prediction for those rows.
    """

    import torch

    from .predict import _preprocess_image

    generator = torch.Generator(device=predictor.device)
    generator.manual_seed(seed)
    candidates_by_row: list[list[dict[str, Any]]] = []
    total = len(label_rows)
    start = time.time()
    for offset in range(0, total, batch_size):
        chunk = label_rows[offset : offset + batch_size]
        batch_images: list[Image.Image] = []
        present: list[int] = []
        for local_idx, row in enumerate(chunk):
            image_bytes = images_by_id.get(row["image_id"])
            if image_bytes is None:
                continue
            batch_images.append(decode_image(image_bytes))
            present.append(local_idx)

        chunk_candidates: list[list[dict[str, Any]]] = [[] for _ in chunk]
        if batch_images:
            tensor = torch.stack(
                [_preprocess_image(image, predictor.recipe.input_size) for image in batch_images],
                dim=0,
            ).to(device=predictor.device, dtype=predictor._input_dtype)
            grouped = _sample_candidates(
                predictor,
                tensor,
                k=max_vote_k,
                temperature=temperature,
                top_p=top_p,
                max_len=predictor.max_len,
                generator=generator,
                apply_postprocess=postprocess,
            )
            for local_idx, candidates in zip(present, grouped, strict=True):
                chunk_candidates[local_idx] = candidates
        candidates_by_row.extend(chunk_candidates)

        done = offset + len(chunk)
        elapsed = time.time() - start
        print(f"  [mv] {done}/{total} ({elapsed:.0f}s)", flush=True)
    return candidates_by_row


def grade_majority_vote(
    label_rows: list[dict[str, str]],
    greedy_predictions: list[str],
    candidates_by_row: list[list[dict[str, Any]]],
    vote_ks: list[int],
) -> dict[str, Any]:
    """Grade MV@K predictions for each K, reusing the A/B/C convention grader."""

    payload: dict[str, Any] = {}
    for k_value in vote_ks:
        vote_predictions: list[str] = []
        vote_records: list[dict[str, Any]] = []
        for greedy_pred, candidates in zip(greedy_predictions, candidates_by_row, strict=True):
            selected, record = select_majority_vote(
                candidates, fallback_prediction=greedy_pred, k=k_value
            )
            vote_predictions.append(selected)
            vote_records.append(record)
        grade = grade_uspto_predictions(label_rows, vote_predictions)
        n_records = len(vote_records)
        grade["vote_diagnostics"] = {
            "k": k_value,
            "invalid_sample_count": sum(record.get("invalid_count", 0) for record in vote_records),
            "tie_count": sum(1 for record in vote_records if record.get("tied")),
            "all_invalid_count": sum(1 for record in vote_records if record.get("all_invalid")),
            "n_rows": n_records,
        }
        payload[str(k_value)] = grade
    return payload


def run_full_uspto_eval(
    checkpoint: Path | str | None = None,
    benchmark: Path | str | None = None,
    benchmark_parquet: Path | str | None = None,
    device: str | None = None,
    precision: str | None = "fp32",
    batch_size: int = 32,
    postprocess: str = "light",
    limit: int | None = None,
    vote_k: list[int] | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    sample_seed: int = 20260625,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    """Run checkpoint inference and grade the full USPTO benchmark.

    When ``benchmark`` and ``benchmark_parquet`` are both omitted, labels and
    images are read directly from the hosted self-contained benchmark parquet
    (resolved by :func:`glyph.ocsr.eval._resolve_default_parquet`): no label CSV
    is needed. Passing ``benchmark`` (a complete-label CSV) or ``benchmark_parquet``
    overrides the respective source for back-compat. The report preserves the
    command-line JSON shape. Rows and predictions are also returned so callers can
    write or otherwise inspect per-row results.

    The greedy decode is always graded and reported. Passing ``vote_k`` (a list
    of positive K values) additionally runs opt-in self-consistency
    majority-vote decoding and attaches an MV@K report block; it never alters the
    greedy numbers.
    """

    from .predict import OCSRPredictor

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if precision not in {None, "auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be None, 'auto', 'fp32', 'fp16', or 'bf16'")
    if postprocess not in {"light", "none"}:
        raise ValueError("postprocess must be 'light' or 'none'")
    vote_ks = sorted({int(k) for k in vote_k}) if vote_k else []
    if any(k <= 0 for k in vote_ks):
        raise ValueError("all majority-vote K values must be > 0")
    if vote_ks:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
    max_vote_k = max(vote_ks) if vote_ks else 0

    parquet_path = Path(benchmark_parquet).expanduser() if benchmark_parquet else None

    if benchmark is not None:
        # Back-compat: an explicit label CSV drives labels; images come from the
        # standard parquet resolver.
        benchmark_csv = Path(benchmark).expanduser()
        label_rows = load_label_rows(benchmark_csv)
        label_source = str(benchmark_csv)
        print("[data] loading benchmark images...", flush=True)
        images_by_id = load_images_by_id(parquet_path)
    else:
        # Self-contained: score directly from the hosted benchmark parquet's
        # complete labels (no CSV join). Read the parquet once for both.
        print("[data] loading benchmark...", flush=True)
        benchmark_rows = load_uspto_benchmark(parquet_path)
        label_rows = benchmark_label_rows(benchmark_rows)
        images_by_id = {
            str(row["id"]): row["image_bytes"]
            for row in benchmark_rows
            if row.get("image_bytes") is not None
        }
        label_source = (
            str(parquet_path) if parquet_path is not None else OCSR_BENCHMARK_PARQUET_PATH
        )

    if limit is not None:
        label_rows = label_rows[:limit]
    print(f"[data] {len(label_rows)} label rows from {label_source}", flush=True)
    print(f"[data] {len(images_by_id)} images available", flush=True)

    checkpoint_description = checkpoint or "EdisonScientific/OCSRGlyph (default)"
    print(f"[model] loading checkpoint {checkpoint_description}", flush=True)
    predictor_precision = None if precision == "auto" else precision
    predictor = OCSRPredictor(checkpoint, device=device, precision=predictor_precision)
    print(f"[model] device={predictor.device} precision={predictor._precision}", flush=True)

    predictions = run_inference(
        predictor,
        label_rows,
        images_by_id,
        batch_size=batch_size,
        postprocess=(postprocess == "light"),
    )

    report = grade_uspto_predictions(label_rows, predictions)
    report["config"] = {
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "benchmark": str(benchmark) if benchmark is not None else None,
        "benchmark_source": label_source,
        "benchmark_parquet": str(benchmark_parquet) if benchmark_parquet is not None else None,
        "device": str(predictor.device),
        "precision": predictor._precision,
        "batch_size": batch_size,
        "postprocess": postprocess,
        "limit": limit,
        "n_rows_evaluated": len(label_rows),
        "expected_total_rows": USPTO_BENCHMARK_TOTAL_ROWS,
        "expected_parseable_rows": USPTO_BENCHMARK_PARSEABLE_ROWS,
    }

    if vote_ks:
        report["config"].update(
            {
                "vote_k": vote_ks,
                "temperature": temperature,
                "top_p": top_p,
                "sample_seed": sample_seed,
            }
        )
        print(f"[mv] sampling {max_vote_k} candidates/row for MV@{vote_ks}...", flush=True)
        candidates_by_row = run_sampling(
            predictor,
            label_rows,
            images_by_id,
            batch_size=batch_size,
            max_vote_k=max_vote_k,
            temperature=temperature,
            top_p=top_p,
            seed=sample_seed,
            postprocess=(postprocess == "light"),
        )
        report["majority_vote"] = grade_majority_vote(
            label_rows, predictions, candidates_by_row, vote_ks
        )

    return report, label_rows, predictions


def format_summary(report: dict[str, Any]) -> str:
    """Format the human-readable command-line summary."""

    counts = report["counts"]
    lines = [
        "",
        "=" * 72,
        "FULL USPTO OCSR EVAL -- three stereochemistry conventions",
        "=" * 72,
        (
            f"rows: {counts['total_rows']} total | "
            f"{counts['parseable_subset_rows']} parseable-subset | "
            f"{counts['scoreable_gt_rows']} scoreable GT | "
            f"{counts['unscoreable_gt_rows']} unscoreable GT"
        ),
    ]
    for scope, title in (("full_5719", "Full 5719"), ("parseable_5704", "Parseable 5704")):
        block = report[scope]
        denominator = block["denominator"]
        lines.extend(["-" * 72, f"{title}  (denominator = {denominator})"])
        for key, label in (
            ("A_canon", "A canonical/full-stereo"),
            ("B_chirality", "B chirality-kept       "),
            ("C_graph", "C graph/no-stereo      "),
        ):
            metric = block[key]
            lines.append(
                f"  {label}  {metric['n_correct']:5d} / {denominator}  = {metric['pct']:6.2f}%"
            )
    majority_vote = report.get("majority_vote")
    if majority_vote:
        for k_value in sorted(majority_vote, key=int):
            mv_block = majority_vote[k_value]
            lines.extend(["=" * 72, f"MAJORITY VOTE  MV@{k_value}  (greedy fallback per row)"])
            for scope, title in (("full_5719", "Full 5719"), ("parseable_5704", "Parseable 5704")):
                block = mv_block[scope]
                denominator = block["denominator"]
                lines.append(f"  {title}  (denominator = {denominator})")
                for key, label in (
                    ("A_canon", "A canonical/full-stereo"),
                    ("B_chirality", "B chirality-kept       "),
                    ("C_graph", "C graph/no-stereo      "),
                ):
                    metric = block[key]
                    lines.append(
                        f"    {label}  {metric['n_correct']:5d} / {denominator}"
                        f"  = {metric['pct']:6.2f}%"
                    )
    lines.append("=" * 72)
    return "\n".join(lines)


def write_eval_report(report: dict[str, Any], output: Path | str) -> Path:
    """Write the aggregate JSON report."""

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_per_row_predictions(
    label_rows: list[dict[str, str]],
    predictions: list[str],
    output: Path | str,
) -> Path:
    """Write per-row predictions as JSON Lines."""

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row, prediction in zip(label_rows, predictions, strict=True):
            handle.write(
                json.dumps(
                    {
                        "image_id": row["image_id"],
                        "source_of_label": row["source_of_label"],
                        "gt": row["smiles"],
                        "pred": prediction,
                    }
                )
                + "\n"
            )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI for the full USPTO OCSR eval.

    Mirrors the ``glyph ocsr eval`` Typer command and backs
    ``python -m glyph.ocsr.full_eval``.
    """

    parser = argparse.ArgumentParser(
        description="Full USPTO OCSR eval under conventions A/B/C.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pth checkpoint or HF repo id (default: released EdisonScientific/OCSRGlyph).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Optional complete-label CSV override (image_id, smiles, source_of_label). "
        "Default: score directly from the self-contained benchmark parquet.",
    )
    parser.add_argument(
        "--benchmark-parquet",
        type=str,
        default=None,
        help="Optional benchmark parquet override (images + labels). "
        "Default: the hosted self-contained parquet (auto-downloaded).",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "auto"],
        help="Inference precision. fp32 reproduces the frozen reference exactly; "
        "the shipping predictor default is fp16 on GPU (may differ by a few rows).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--postprocess",
        choices=["light", "none"],
        default="light",
        help="'light' = strip [H]/[HH] fragments + canonicalize (reported); 'none' = canonicalize only.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N rows (quick trial)."
    )
    parser.add_argument(
        "--majority-vote",
        action="store_true",
        help="Also run opt-in self-consistency majority-vote decoding (MV@K). "
        "Never changes the greedy headline numbers.",
    )
    parser.add_argument(
        "--vote-k",
        type=str,
        default="8",
        help="Comma-separated K value(s) for --majority-vote, e.g. '8' or '4,8,16'.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (MV).")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus top-p (MV).")
    parser.add_argument("--sample-seed", type=int, default=20260625, help="Sampling seed (MV).")
    parser.add_argument("--output", type=str, default=None, help="Output JSON report path.")
    parser.add_argument(
        "--per-row-out", type=str, default=None, help="Optional JSONL of per-row predictions."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the full USPTO OCSR eval from the command line."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")

    vote_ks = parse_vote_ks(args.vote_k) if args.majority_vote else None

    report, label_rows, predictions = run_full_uspto_eval(
        checkpoint=args.checkpoint,
        benchmark=args.benchmark,
        benchmark_parquet=args.benchmark_parquet,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        postprocess=args.postprocess,
        limit=args.limit,
        vote_k=vote_ks,
        temperature=args.temperature,
        top_p=args.top_p,
        sample_seed=args.sample_seed,
    )
    print(format_summary(report))

    if args.output:
        out_path = write_eval_report(report, args.output)
        print(f"[out] wrote {out_path}", flush=True)
    if args.per_row_out:
        per_row_path = write_per_row_predictions(label_rows, predictions, args.per_row_out)
        print(f"[out] wrote {per_row_path}", flush=True)


if __name__ == "__main__":
    main()
