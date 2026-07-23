"""Public grading helpers for Markush predictions.

The paper path promotes checkpoints with the strict ``protocol="ours"``
contract from :mod:`glyph.markush.eval.metrics_mcs`, which applies graph-isomorphism
CXSMILES equality and stable-section scoring.  The legacy ``protocol="mg2"``
path is kept for historical MarkushGrapher-2 comparability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from glyph.markush.eval.metrics_mcs import (
    evaluate_markush_both_protocols,
    evaluate_markush_mcs,
    load_jsonl,
)

Protocol = Literal["ours", "mg2", "both"]


def _single_prediction_row(prediction: str) -> dict[str, str]:
    return {"id": "0", "prediction": prediction}


def _single_ground_truth_row(reference: str) -> dict[str, str]:
    return {"id": "0", "annotation": reference}


def grade(
    prediction: str,
    reference: str,
    *,
    protocol: Protocol = "ours",
    include_missing: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Grade one prediction against one reference Markush annotation.

    Parameters
    ----------
    prediction:
        Model output, either a full ``<markush>`` XML envelope or a bare
        CXSMILES-like string accepted by the evaluator.
    reference:
        Ground-truth Markush annotation.
    protocol:
        ``"ours"`` for the strict graph-isomorphism contract, ``"mg2"`` for the
        legacy MarkushGrapher-2-compatible scorer, or ``"both"``.
    """

    if protocol == "both":
        return grade_both(
            prediction,
            reference,
            include_missing=include_missing,
            verbose=verbose,
        )
    if protocol not in {"ours", "mg2"}:
        raise ValueError(f"Unknown grading protocol: {protocol}")

    metrics, details = evaluate_markush_mcs(
        [_single_prediction_row(prediction)],
        [_single_ground_truth_row(reference)],
        include_missing=include_missing,
        return_details=True,
        verbose=verbose,
        protocol=protocol,
    )
    return {
        "protocol": protocol,
        "metrics": metrics,
        "sample": details[0] if details else {},
    }


def grade_both(
    prediction: str,
    reference: str,
    *,
    include_missing: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Grade one prediction under both strict and MG2-compatible protocols."""

    metrics, details = evaluate_markush_both_protocols(
        [_single_prediction_row(prediction)],
        [_single_ground_truth_row(reference)],
        include_missing=include_missing,
        return_details=True,
        verbose=verbose,
    )
    return {
        "protocol": "both",
        "metrics": metrics,
        "sample": {name: rows[0] if rows else {} for name, rows in details.items()},
    }


def grade_rows(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    *,
    protocol: Protocol = "ours",
    include_missing: bool = True,
    verbose: bool = False,
    return_details: bool = True,
) -> dict[str, Any]:
    """Grade JSONL-style prediction and ground-truth rows.

    Prediction rows should contain ``id`` and ``prediction`` or ``cxsmiles_opt``.
    Ground-truth rows should contain ``id`` and ``annotation``, ``target``, or
    ``cxsmiles_opt``.  This matches the repository's offline eval artifacts.
    """

    if protocol == "both":
        if return_details:
            metrics, details = evaluate_markush_both_protocols(
                predictions,
                ground_truths,
                include_missing=include_missing,
                return_details=True,
                verbose=verbose,
            )
            return {"protocol": "both", "metrics": metrics, "per_sample": details}
        metrics = evaluate_markush_both_protocols(
            predictions,
            ground_truths,
            include_missing=include_missing,
            return_details=False,
            verbose=verbose,
        )
        return {"protocol": "both", "metrics": metrics}

    if protocol not in {"ours", "mg2"}:
        raise ValueError(f"Unknown grading protocol: {protocol}")

    if return_details:
        metrics, details = evaluate_markush_mcs(
            predictions,
            ground_truths,
            include_missing=include_missing,
            return_details=True,
            verbose=verbose,
            protocol=protocol,
        )
        return {"protocol": protocol, "metrics": metrics, "per_sample": details}

    metrics = evaluate_markush_mcs(
        predictions,
        ground_truths,
        include_missing=include_missing,
        return_details=False,
        verbose=verbose,
        protocol=protocol,
    )
    return {"protocol": protocol, "metrics": metrics}


def grade_files(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    *,
    protocol: Protocol = "ours",
    include_missing: bool = True,
    verbose: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Grade prediction and ground-truth JSONL files and optionally write JSON."""

    payload = grade_rows(
        load_jsonl(predictions_path),
        load_jsonl(ground_truth_path),
        protocol=protocol,
        include_missing=include_missing,
        verbose=verbose,
        return_details=True,
    )
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return payload


__all__ = [
    "Protocol",
    "grade",
    "grade_both",
    "grade_files",
    "grade_rows",
]
