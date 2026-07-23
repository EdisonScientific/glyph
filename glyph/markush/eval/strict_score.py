"""Strict parsed-graph ("ours") scoring for a benchmark predictions JSON.

Companion to the official DS4SD scorer under ``tools/mg2_scorer/``. Given the
same ``(benchmark, predictions.json)`` pair that
``tools/mg2_scorer/score_benchmark.sh`` scores under the official DS4SD
contract, this reports the paper's second column — "Strict parsed-graph
equality" — by joining predictions to the same auto-pulled ground truth by id
and grading with :func:`glyph.markush.grading.grade_rows` (``protocol="ours"``).

The graph-isomorphism scoring itself lives in :mod:`glyph.markush.grading` /
:mod:`glyph.markush.eval.metrics_mcs` and is reused unchanged; this module only
adapts the benchmark artifacts (a ``{"samples": [...]}`` predictions JSON and a
DS4SD-form ground-truth JSONL) into the ``id``/``prediction``/``cxsmiles_opt``
rows the grader expects.

The ground-truth file is the exact one the official scorer uses
(``ds4sd_mg2_predictions_878.jsonl`` for ip5m; the derived DS4SD-form GT for m2s
/ uspto_markush). The official scorer reads that file's RDKit ``gt_cxsmiles``
column; the strict grader reads the model-format ``gt_cxsmiles_opt`` column from
the same rows, so both headline numbers are computed against one authoritative
ground truth. ``None`` ground-truth rows are retained (they count as incorrect
and stay in the denominator), matching the official 878-row contract.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from glyph.markush.grading import grade_rows

#: Id fields consulted (priority order) to join predictions to ground truth.
DEFAULT_ID_FIELDS: tuple[str, ...] = ("sample_id", "id", "index")

#: Prediction fields consulted (priority order) on each predictions.json sample.
#: ``prediction`` carries the raw ``<markush>…`` envelope the model emits;
#: ``prediction_cxsmiles`` is the extracted inline CXSMILES fallback.
DEFAULT_PRED_FIELDS: tuple[str, ...] = (
    "prediction",
    "prediction_cxsmiles",
    "cxsmiles_opt",
    "cxsmiles",
)

#: Ground-truth fields consulted (priority order) on each GT row. The DS4SD GT
#: files carry the model-format target as ``gt_cxsmiles_opt``. Plain
#: ``cxsmiles_opt`` is deliberately excluded because in those files it is MG2's
#: own *prediction*, not the ground truth.
DEFAULT_GT_FIELDS: tuple[str, ...] = ("gt_cxsmiles_opt", "annotation")

#: GT source fields that already carry a full ``<markush>`` envelope / raw
#: annotation, mapped straight to the ``annotation`` row key the grader reads.
#: Everything else is treated as a bare inline CXSMILES (``cxsmiles_opt`` key).
_ANNOTATION_GT_FIELDS = frozenset({"annotation", "gt_annotation"})


@dataclass
class StrictScore:
    """Headline strict parsed-graph result for one benchmark predictions file."""

    benchmark: str | None
    count: int
    total: int
    percent: float
    invalid_gt_count: int
    n_valid: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def headline(self) -> str:
        name = self.benchmark or "benchmark"
        return (
            f"=== {name} strict parsed-graph (ours) score: "
            f"{self.count} / {self.total} = {self.percent:.2f}% ==="
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "count": self.count,
            "total": self.total,
            "percent": self.percent,
            "invalid_gt_count": self.invalid_gt_count,
            "n_valid": self.n_valid,
        }


def _first_nonempty(entry: dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first key whose value is a non-empty scalar, else ``None``."""

    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _first_present(entry: dict[str, Any], keys: Sequence[str]) -> tuple[str | None, Any]:
    """Return ``(field, value)`` for the first *present* key (value may be ``None``).

    Distinguishes "field present but ``None``" (a retained invalid-GT row) from
    "field absent" (a schema error), which matters for the 878-row contract.
    """

    for key in keys:
        if key in entry:
            return key, entry[key]
    return None, None


def load_prediction_samples(path: str | Path) -> list[dict[str, Any]]:
    """Load prediction rows from a ``glyph markush eval benchmark`` artifact.

    Accepts the ``{"config": …, "samples": [...]}`` JSON the benchmark flow
    writes, a bare JSON list, or a JSONL file.
    """

    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict):
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"{path} does not contain a 'samples' list")
    else:
        raise ValueError(f"Unsupported JSON root in {path}: {type(payload).__name__}")
    return [sample for sample in samples if isinstance(sample, dict)]


def load_ground_truth_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load DS4SD-form ground-truth rows from a JSON/JSONL file."""

    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("samples", payload.get("rows"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} does not contain a ground-truth list")
    else:
        raise ValueError(f"Unsupported JSON root in {path}: {type(payload).__name__}")
    return [row for row in rows if isinstance(row, dict)]


def build_grading_rows(
    samples: Sequence[dict[str, Any]],
    ground_truth_rows: Sequence[dict[str, Any]],
    *,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
    pred_fields: Sequence[str] = DEFAULT_PRED_FIELDS,
    gt_fields: Sequence[str] = DEFAULT_GT_FIELDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Adapt benchmark samples + DS4SD GT rows into rows for :func:`grade_rows`.

    Predictions are joined to ground truth by id; the returned rows share the
    same id order so the strict denominator matches the official scorer's
    (per-prediction, ``None`` GT retained as incorrect). Raises if a prediction
    id has no matching ground-truth row.
    """

    gt_by_id: dict[str, tuple[str | None, Any]] = {}
    for row in ground_truth_rows:
        gid = _first_nonempty(row, id_fields)
        if gid is None:
            continue
        gt_by_id[str(gid)] = _first_present(row, gt_fields)

    pred_rows: list[dict[str, Any]] = []
    aligned_gt_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sid = _first_nonempty(sample, id_fields)
        if sid is None:
            raise ValueError(
                f"prediction sample at index {index} has no id field {tuple(id_fields)}"
            )
        sid = str(sid)
        if sid not in gt_by_id:
            raise KeyError(
                f"prediction id {sid!r} (index {index}) has no ground-truth row; "
                "predictions and ground truth must cover the same ids"
            )
        gt_field, gt_value = gt_by_id[sid]
        if gt_field is None:
            raise KeyError(
                f"ground-truth row for id {sid!r} has none of the fields {tuple(gt_fields)}"
            )

        pred_rows.append({"id": sid, "prediction": _first_nonempty(sample, pred_fields)})
        gt_key = "annotation" if gt_field in _ANNOTATION_GT_FIELDS else "cxsmiles_opt"
        aligned_gt_rows.append({"id": sid, gt_key: gt_value})

    return pred_rows, aligned_gt_rows


def strict_score(
    samples: Sequence[dict[str, Any]],
    ground_truth_rows: Sequence[dict[str, Any]],
    *,
    benchmark: str | None = None,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
    pred_fields: Sequence[str] = DEFAULT_PRED_FIELDS,
    gt_fields: Sequence[str] = DEFAULT_GT_FIELDS,
    verbose: bool = False,
) -> tuple[StrictScore, dict[str, Any]]:
    """Grade in-memory benchmark samples against DS4SD-form GT rows.

    Returns ``(StrictScore, grading_payload)`` where ``grading_payload`` is the
    full :func:`grade_rows` result (``protocol="ours"`` metrics + per-sample
    detail).
    """

    pred_rows, aligned_gt_rows = build_grading_rows(
        samples,
        ground_truth_rows,
        id_fields=id_fields,
        pred_fields=pred_fields,
        gt_fields=gt_fields,
    )
    payload = grade_rows(
        pred_rows,
        aligned_gt_rows,
        protocol="ours",
        include_missing=True,
        return_details=True,
        verbose=verbose,
    )
    per_sample = payload.get("per_sample", [])
    metrics = payload.get("metrics", {})
    count = sum(1 for row in per_sample if row.get("cxsmi_equality"))
    total = len(per_sample)
    invalid_gt_count = sum(1 for row in per_sample if not row.get("valid_gt"))
    score = StrictScore(
        benchmark=benchmark,
        count=count,
        total=total,
        percent=(100.0 * count / total) if total else 0.0,
        invalid_gt_count=invalid_gt_count,
        n_valid=int(metrics.get("n_valid", 0)),
        metrics=metrics,
    )
    return score, payload


def strict_score_files(
    input_json: str | Path,
    gt_jsonl: str | Path | None = None,
    *,
    benchmark: str | None = None,
    output_json: str | Path | None = None,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
    pred_fields: Sequence[str] = DEFAULT_PRED_FIELDS,
    gt_fields: Sequence[str] = DEFAULT_GT_FIELDS,
    verbose: bool = False,
) -> StrictScore:
    """Strict-score a predictions JSON file against a ground-truth file.

    When ``gt_jsonl`` is omitted, the ground truth for ``benchmark`` is
    auto-pulled from ``EdisonScientific/glyph-datasets`` via
    :func:`glyph.markush.eval.benchmark_data.resolve_benchmark_gt`, matching the
    official scorer's auto-pull. Writes a JSON report to ``output_json`` when
    given.
    """

    if gt_jsonl is None:
        if benchmark is None:
            raise ValueError("pass --gt-jsonl or --benchmark to resolve the ground truth")
        from glyph.markush.eval.benchmark_data import resolve_benchmark_gt

        gt_jsonl = resolve_benchmark_gt(benchmark)

    samples = load_prediction_samples(input_json)
    ground_truth_rows = load_ground_truth_rows(gt_jsonl)
    score, payload = strict_score(
        samples,
        ground_truth_rows,
        benchmark=benchmark,
        id_fields=id_fields,
        pred_fields=pred_fields,
        gt_fields=gt_fields,
        verbose=verbose,
    )

    if output_json is not None:
        report = {
            "config": {
                "grader": "glyph.markush.grading strict parsed-graph (protocol=ours)",
                "benchmark": benchmark,
                "input_json": str(input_json),
                "gt_jsonl": str(gt_jsonl),
            },
            "strict": score.to_dict(),
            "grading": payload,
        }
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return score


def _parse_field_list(value: str) -> list[str]:
    fields = [part.strip() for part in value.split(",") if part.strip()]
    if not fields:
        raise ValueError("field list must not be empty")
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True, help="Predictions JSON (STEP 1).")
    parser.add_argument(
        "--gt-jsonl",
        type=Path,
        default=None,
        help="DS4SD-form ground truth. Omit to auto-pull for --benchmark.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark name (ip5m, m2s, uspto_markush); labels output and auto-pulls GT.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional strict-scoring report path.",
    )
    parser.add_argument(
        "--id-field",
        default=",".join(DEFAULT_ID_FIELDS),
        help="Comma-separated id field priority.",
    )
    parser.add_argument(
        "--pred-field",
        default=",".join(DEFAULT_PRED_FIELDS),
        help="Comma-separated prediction field priority.",
    )
    parser.add_argument(
        "--gt-field",
        default=",".join(DEFAULT_GT_FIELDS),
        help="Comma-separated ground-truth field priority.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Thin CLI mirror of :func:`strict_score_files`."""

    args = build_parser().parse_args(argv)
    score = strict_score_files(
        args.input_json,
        args.gt_jsonl,
        benchmark=args.benchmark,
        output_json=args.output_json,
        id_fields=_parse_field_list(args.id_field),
        pred_fields=_parse_field_list(args.pred_field),
        gt_fields=_parse_field_list(args.gt_field),
        verbose=args.verbose,
    )
    print(
        json.dumps(
            {
                "count": score.count,
                "total": score.total,
                "percent": round(score.percent, 4),
                "invalid_gt_count": score.invalid_gt_count,
                "n_valid": score.n_valid,
            },
            indent=2,
        )
    )
    print("\n" + score.headline())


if __name__ == "__main__":
    main()
