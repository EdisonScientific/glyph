#!/usr/bin/env python3
"""Evaluate Markush structure predictions against ground truth.

Computes seven metrics for Markush structure recognition:
  1. string_em       — Exact string match on cxsmiles_opt
  2. backbone_match  — Molecular graph match via InChIKey (requires RDKit)
  3. r_accuracy      — Fraction of ground-truth R-group labels found in prediction
  4. extension_match — Exact match of CXSMILES extension (part after |)
  5. stable_accuracy — Structured comparison of substituent tables
  6. markush_em      — Full Markush accuracy (backbone + stable)
  7. valid_rate      — Fraction of outputs that are parseable

Input formats
-------------
Predictions JSONL (one per line):
    {"id": "EP90115748NWA2_23_0", "prediction": "<markush><cxsmi>...</cxsmi><stable>...</stable></markush>"}

Ground truth JSONL (one per line):
    {"id": "EP90115748NWA2_23_0", "cxsmiles_opt": "...", "annotation": "<markush><cxsmi>...</cxsmi><stable>...</stable></markush>"}

Usage
-----
    python eval_markush.py --predictions preds.jsonl --ground-truth gt.jsonl
    python eval_markush.py --predictions preds.jsonl --ground-truth gt.jsonl --output metrics.json --detailed
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from glyph.markush.eval import common

# ---------------------------------------------------------------------------
# Optional RDKit import
# ---------------------------------------------------------------------------

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem, RDLogger

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    _RDKIT_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MarkushPrediction:
    """Parsed Markush prediction/ground-truth with structured components."""

    raw: str
    cxsmiles_opt: str = ""
    backbone_smiles: str = ""  # cxsmiles_opt with <r>…</r> → *, before |
    r_labels: list = field(default_factory=list)  # ordered list of R-group labels
    r_labels_set: set = field(default_factory=set)  # unique R-group labels
    extensions: str = ""  # everything after | in cxsmiles_opt
    stable: dict = field(default_factory=dict)  # {R_label: [alternatives]}
    stable_raw: str = ""  # raw <stable>…</stable> content
    valid: bool = False
    parse_error: str = ""


# ---------------------------------------------------------------------------
# Regex patterns and parsing helpers
#
# These were duplicated across the eval modules and silently diverged in
# regex flags + fallback handling.  They now delegate to
# :mod:`glyph.markush.eval.common` (the single source of truth).  The private
# names are kept so existing tests/callers don't break — they're thin
# aliases over the shared API.
# ---------------------------------------------------------------------------

_RE_MARKUSH = common.RE_MARKUSH
_RE_R_TAG = common.RE_R_TAG


def _extract_cxsmiles_and_stable(raw: str) -> tuple[str, str]:
    """Extract cxsmiles_opt and stable text from a <markush> XML string.

    Returns (cxsmiles_opt, stable_raw).  Both may be empty strings.
    Requires the ``<markush>...</markush>`` wrapper; inputs containing
    only ``<cxsmi>`` return ``("", "")`` so the caller's lenient-mode
    fallback can decide what to do.
    """
    return common.extract_cxsmiles_and_stable(raw, require_wrapper=True)


def _strip_r_tags(cxsmiles_opt: str) -> str:
    """Replace every ``<r>LABEL</r>`` with ``*`` (wildcard atom).

    Example
    -------
    >>> _strip_r_tags("<r>R1</r>C=O")
    '*C=O'
    """
    return common.strip_r_tags(cxsmiles_opt)


def _extract_r_labels(cxsmiles_opt: str) -> list[str]:
    """Return the ordered list of R-group labels from <r>…</r> tags."""
    return common.extract_r_labels(cxsmiles_opt)


def _split_cxsmiles(cxsmiles_opt: str) -> tuple[str, str]:
    """Split CXSMILES into SMILES core and extension.

    Everything before the first top-level ``|`` is core SMILES;
    everything after (including additional ``|`` delimiters) is the extension.

    The ``<r>…</r>`` tags should already have been stripped (or not).
    """
    return common.split_cxsmiles(cxsmiles_opt)


def _parse_stable(stable_raw: str) -> dict[str, list[str]]:
    """Parse a ``<stable>`` section into {label: [alternatives]}.

    Format: ``R0:sub1<n>sub2<ns>R1:sub3<n>sub4``
    ``<ns>`` separates R-groups; ``<n>`` separates alternatives within one group.

    Only ``:`` is accepted as a label separator (strict mode).
    """
    return common.parse_stable(stable_raw, permissive=False)


def parse_markush_output(raw: str, *, strict: bool = True) -> MarkushPrediction:
    """Parse a model output or ground-truth string into structured components.

    Parameters
    ----------
    raw : str
        The raw string to parse.
    strict : bool
        If True (default, used for predictions), require ``<markush><cxsmi>``
        XML structure for the output to be considered valid.
        If False (used for GT fallback), accept plain SMILES strings.

    Returns
    -------
    MarkushPrediction
        Structured representation with validity flag.
    """
    pred = MarkushPrediction(raw=raw)

    if not raw or not isinstance(raw, str):
        pred.parse_error = "empty or non-string input"
        return pred

    raw = raw.strip()

    # Try to extract from <markush> wrapper
    cxsmi_str, stable_raw = _extract_cxsmiles_and_stable(raw)

    if cxsmi_str:
        pred.cxsmiles_opt = cxsmi_str
        pred.stable_raw = stable_raw
        pred.valid = True
    elif not strict:
        # Lenient fallback: treat the whole string as cxsmiles_opt (no XML wrapper)
        # Only used for ground-truth entries that supply cxsmiles_opt directly.
        if re.search(r"[A-Za-z0-9]", raw) and "<markush>" not in raw:
            pred.cxsmiles_opt = raw
            pred.valid = True
        else:
            pred.parse_error = "no <markush>/<cxsmi> tags found and not a valid SMILES"
            return pred
    else:
        # Strict mode: model output must have proper XML structure
        pred.parse_error = "missing <markush><cxsmi>...</cxsmi></markush> structure"
        return pred

    # Extract components from cxsmiles_opt
    pred.r_labels = _extract_r_labels(pred.cxsmiles_opt)
    pred.r_labels_set = set(pred.r_labels)

    # Backbone = r-tags replaced with *, split off extension
    stripped = _strip_r_tags(pred.cxsmiles_opt)
    core, ext = _split_cxsmiles(stripped)
    pred.backbone_smiles = core
    pred.extensions = ext

    # Parse stable section
    pred.stable = _parse_stable(pred.stable_raw)

    return pred


def parse_ground_truth_entry(entry: dict) -> MarkushPrediction:
    """Parse a ground-truth JSONL entry.

    The GT can provide ``cxsmiles_opt`` directly or via ``annotation``.
    If ``cxsmiles_opt`` is present it is used as the authoritative backbone;
    the ``annotation`` field (if present) supplies the ``<stable>`` section.
    """
    # If annotation is present, parse it for the full structure
    annotation = entry.get("annotation", "")
    cxsmiles_opt_direct = entry.get("cxsmiles_opt", "")

    if annotation:
        pred = parse_markush_output(annotation, strict=False)
        # Override cxsmiles_opt with the direct field if provided
        if cxsmiles_opt_direct:
            pred.cxsmiles_opt = cxsmiles_opt_direct
            pred.r_labels = _extract_r_labels(cxsmiles_opt_direct)
            pred.r_labels_set = set(pred.r_labels)
            stripped = _strip_r_tags(cxsmiles_opt_direct)
            core, ext = _split_cxsmiles(stripped)
            pred.backbone_smiles = core
            pred.extensions = ext
        pred.valid = True
        return pred
    elif cxsmiles_opt_direct:
        # Build a minimal MarkushPrediction from cxsmiles_opt only
        pred = MarkushPrediction(raw=cxsmiles_opt_direct)
        pred.cxsmiles_opt = cxsmiles_opt_direct
        pred.r_labels = _extract_r_labels(cxsmiles_opt_direct)
        pred.r_labels_set = set(pred.r_labels)
        stripped = _strip_r_tags(cxsmiles_opt_direct)
        core, ext = _split_cxsmiles(stripped)
        pred.backbone_smiles = core
        pred.extensions = ext
        pred.valid = True
        return pred
    else:
        return MarkushPrediction(raw="", parse_error="no cxsmiles_opt or annotation")


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def metric_string_em(pred: MarkushPrediction, gt: MarkushPrediction) -> bool:
    """Exact string match on cxsmiles_opt."""
    return pred.cxsmiles_opt == gt.cxsmiles_opt


def metric_backbone_match(pred: MarkushPrediction, gt: MarkushPrediction) -> bool | None:
    """Compare molecular backbones via InChIKey.

    Returns None if RDKit is unavailable or either SMILES is unparseable.

    Algorithm:
        1. Take backbone_smiles (already has * instead of <r>…</r>, no extension)
        2. Replace * (wildcard) with C (carbon) for RDKit parsing
        3. Parse with RDKit (sanitize=False)
        4. Remove stereochemistry
        5. Compare InChIKeys
    """
    if not _RDKIT_AVAILABLE:
        return None

    def _to_inchikey(smiles: str) -> str | None:
        if not smiles:
            return None
        # Replace wildcard atoms with carbon
        smi = smiles.replace("*", "C")
        try:
            params = Chem.SmilesParserParams()
            params.sanitize = False
            # Try strict first, then lenient
            mol = Chem.MolFromSmiles(smi, params)
            if mol is None:
                return None
            # Remove stereochemistry
            Chem.RemoveStereochemistry(mol)
            # Try to sanitize for InChI generation
            with contextlib.suppress(Exception):
                Chem.SanitizeMol(mol)
                # some molecules can't be sanitized but InChI may still work
            inchi = Chem.MolToInchi(mol)
            if inchi is None:
                return None
            return Chem.InchiToInchiKey(inchi)
        except Exception:
            return None

    ik_pred = _to_inchikey(pred.backbone_smiles)
    ik_gt = _to_inchikey(gt.backbone_smiles)

    if ik_pred is None or ik_gt is None:
        return None  # unparseable — can't compare

    return ik_pred == ik_gt


def metric_r_accuracy(pred: MarkushPrediction, gt: MarkushPrediction) -> float:
    """Fraction of ground-truth R-group labels found in the prediction.

    Returns 1.0 if GT has no R-group labels (vacuously correct).
    """
    if not gt.r_labels_set:
        return 1.0
    if not pred.r_labels_set:
        return 0.0
    matched = gt.r_labels_set & pred.r_labels_set
    return len(matched) / len(gt.r_labels_set)


def metric_r_precision(pred: MarkushPrediction, gt: MarkushPrediction) -> float:
    """Fraction of predicted R-group labels that are in the ground truth.

    Returns 1.0 if prediction has no R-group labels (vacuously correct).
    """
    if not pred.r_labels_set:
        return 1.0 if not gt.r_labels_set else 0.0
    matched = gt.r_labels_set & pred.r_labels_set
    return len(matched) / len(pred.r_labels_set)


def metric_r_f1(pred: MarkushPrediction, gt: MarkushPrediction) -> float:
    """F1 score for R-group label sets."""
    recall = metric_r_accuracy(pred, gt)
    precision = metric_r_precision(pred, gt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def metric_extension_match(pred: MarkushPrediction, gt: MarkushPrediction) -> bool:
    """Exact string match on the CXSMILES extension (part after |)."""
    return pred.extensions == gt.extensions


def metric_stable_accuracy(pred: MarkushPrediction, gt: MarkushPrediction) -> float:
    """Structured comparison of substituent tables.

    For each R-group in the ground truth, checks if:
      (a) the label exists in the prediction, and
      (b) the set of alternatives matches exactly.

    Returns the fraction of GT R-groups that fully match.
    Returns 1.0 if GT has no R-groups in the stable section (vacuously correct).
    """
    if not gt.stable:
        # Vacuously correct only when prediction also has empty stable; otherwise
        # penalize extra predicted entries.
        return 1.0 if not pred.stable else 0.0

    matched = 0
    for label, gt_alts in gt.stable.items():
        pred_alts = pred.stable.get(label, None)
        if pred_alts is None:
            continue
        # Normalize: strip whitespace and compare as sorted lists
        gt_set = sorted([a.strip() for a in gt_alts])
        pred_set = sorted([a.strip() for a in pred_alts])
        if gt_set == pred_set:
            matched += 1

    return matched / len(gt.stable)


def metric_stable_exact_match(pred: MarkushPrediction, gt: MarkushPrediction) -> bool:
    """True only if ALL R-groups in the stable section match perfectly."""
    if not gt.stable and not pred.stable:
        return True
    if not gt.stable or not pred.stable:
        return not gt.stable and not pred.stable
    return metric_stable_accuracy(pred, gt) == 1.0


def metric_markush_em(
    pred: MarkushPrediction,
    gt: MarkushPrediction,
    *,
    use_backbone: bool = True,
) -> bool | None:
    """Full Markush exact match: backbone + stable must match.

    If use_backbone is True and RDKit is available, uses backbone_match.
    Otherwise falls back to string exact match of cxsmiles_opt.

    Returns None if backbone comparison fails (RDKit unavailable/unparseable).
    """
    # Stable check
    stable_ok = metric_stable_exact_match(pred, gt)
    if not stable_ok:
        return False

    # Backbone check
    if use_backbone and _RDKIT_AVAILABLE:
        bb = metric_backbone_match(pred, gt)
        if bb is None:
            return None  # can't determine
        return bb
    else:
        # Fall back to string exact match of cxsmiles_opt
        return metric_string_em(pred, gt)


def _string_similarity(a: str, b: str) -> float:
    """Levenshtein-ratio string similarity (0-1).

    Uses a simple DP implementation to avoid external dependencies.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    len_a, len_b = len(a), len(b)
    # Optimize: if very different lengths, quick return
    max_len = max(len_a, len_b)
    if max_len > 5000:
        # For very long strings, use a simpler metric to avoid O(n²) cost
        common = set(a) & set(b)
        return len(common) / len(set(a) | set(b)) if (set(a) | set(b)) else 1.0

    # Standard Levenshtein via two-row DP
    prev = list(range(len_b + 1))
    curr = [0] * (len_b + 1)
    for i in range(1, len_a + 1):
        curr[0] = i
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    distance = prev[len_b]
    return 1.0 - distance / max_len


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Per-sample evaluation result."""

    sample_id: str
    valid_pred: bool = False
    valid_gt: bool = False
    string_em: bool = False
    backbone_match: bool | None = None
    r_accuracy: float = 0.0
    r_precision: float = 0.0
    r_f1: float = 0.0
    extension_match: bool = False
    stable_accuracy: float = 0.0
    stable_em: bool = False
    markush_em: bool | None = None
    string_similarity: float = 0.0
    pred_cxsmiles: str = ""
    gt_cxsmiles: str = ""
    parse_error: str = ""


def evaluate_sample(
    sample_id: str,
    pred: MarkushPrediction,
    gt: MarkushPrediction,
) -> SampleResult:
    """Evaluate a single prediction against ground truth."""
    result = SampleResult(
        sample_id=sample_id,
        valid_pred=pred.valid,
        valid_gt=gt.valid,
        pred_cxsmiles=pred.cxsmiles_opt,
        gt_cxsmiles=gt.cxsmiles_opt,
        parse_error=pred.parse_error,
    )

    if not pred.valid:
        return result

    result.string_em = metric_string_em(pred, gt)
    result.backbone_match = metric_backbone_match(pred, gt)
    result.r_accuracy = metric_r_accuracy(pred, gt)
    result.r_precision = metric_r_precision(pred, gt)
    result.r_f1 = metric_r_f1(pred, gt)
    result.extension_match = metric_extension_match(pred, gt)
    result.stable_accuracy = metric_stable_accuracy(pred, gt)
    result.stable_em = metric_stable_exact_match(pred, gt)
    result.markush_em = metric_markush_em(pred, gt)
    result.string_similarity = _string_similarity(pred.cxsmiles_opt, gt.cxsmiles_opt)

    return result


def compute_metrics(
    results: list[SampleResult],
) -> dict:
    """Compute aggregate metrics across all samples.

    Parameters
    ----------
    results : list[SampleResult]
        Per-sample evaluation results from :func:`evaluate_sample`.

    Returns
    -------
    dict
        Aggregate metrics with keys matching the seven target metrics
        plus supplementary statistics.
    """
    n = len(results)
    if n == 0:
        return {"n_samples": 0, "error": "no samples to evaluate"}

    # Valid rate
    valid_count = sum(1 for r in results if r.valid_pred)
    valid_rate = valid_count / n

    # String EM (only over valid predictions — but report both ways)
    string_em_all = sum(1 for r in results if r.string_em) / n
    string_em_valid = (
        sum(1 for r in results if r.valid_pred and r.string_em) / valid_count
        if valid_count > 0
        else 0.0
    )

    # Backbone match (only where computable)
    bb_results = [r for r in results if r.backbone_match is not None]
    bb_count = len(bb_results)
    backbone_match = (
        sum(1 for r in bb_results if r.backbone_match) / bb_count if bb_count > 0 else None
    )

    # R-group accuracy (macro average over valid predictions)
    r_acc = (
        sum(r.r_accuracy for r in results if r.valid_pred) / valid_count if valid_count > 0 else 0.0
    )
    r_prec = (
        sum(r.r_precision for r in results if r.valid_pred) / valid_count
        if valid_count > 0
        else 0.0
    )
    r_f1 = sum(r.r_f1 for r in results if r.valid_pred) / valid_count if valid_count > 0 else 0.0

    # Extension match
    ext_match_all = sum(1 for r in results if r.extension_match) / n
    ext_match_valid = (
        sum(1 for r in results if r.valid_pred and r.extension_match) / valid_count
        if valid_count > 0
        else 0.0
    )

    # Stable accuracy (macro average)
    stable_acc = (
        sum(r.stable_accuracy for r in results if r.valid_pred) / valid_count
        if valid_count > 0
        else 0.0
    )
    stable_em = sum(1 for r in results if r.stable_em) / n

    # Full Markush EM
    markush_results = [r for r in results if r.markush_em is not None]
    markush_em = sum(1 for r in markush_results if r.markush_em) / n if markush_results else None

    # String similarity
    string_sim = (
        sum(r.string_similarity for r in results if r.valid_pred) / valid_count
        if valid_count > 0
        else 0.0
    )

    metrics = {
        "n_samples": n,
        "n_valid": valid_count,
        "valid_rate": valid_rate,
        "string_em": string_em_all,
        "string_em_valid": string_em_valid,
        "backbone_match": backbone_match,
        "backbone_match_n": bb_count,
        "r_accuracy": r_acc,
        "r_precision": r_prec,
        "r_f1": r_f1,
        "extension_match": ext_match_all,
        "extension_match_valid": ext_match_valid,
        "stable_accuracy": stable_acc,
        "stable_em": stable_em,
        "markush_em": markush_em,
        "string_similarity": string_sim,
    }

    return metrics


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file, skipping blank lines."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                warnings.warn(f"{path}:{line_no}: invalid JSON — {e}", stacklevel=2)
    return entries


def format_metrics_table(metrics: dict) -> str:
    """Format metrics dict as a human-readable table."""
    lines = []
    lines.append("=" * 56)
    lines.append(f"  Markush Evaluation Results  (n={metrics['n_samples']})")
    lines.append("=" * 56)
    lines.append("")

    # Core metrics
    rows = [
        ("Valid Output Rate", "valid_rate", "%"),
        ("String Exact Match", "string_em", "%"),
        ("String EM (valid only)", "string_em_valid", "%"),
        ("Backbone Match (InChIKey)", "backbone_match", "%"),
        ("R-group Recall", "r_accuracy", "%"),
        ("R-group Precision", "r_precision", "%"),
        ("R-group F1", "r_f1", "%"),
        ("Extension Match", "extension_match", "%"),
        ("Extension Match (valid)", "extension_match_valid", "%"),
        ("Stable Accuracy (macro)", "stable_accuracy", "%"),
        ("Stable Exact Match", "stable_em", "%"),
        ("Full Markush EM", "markush_em", "%"),
        ("String Similarity (avg)", "string_similarity", "f"),
    ]

    for label, key, fmt in rows:
        val = metrics.get(key)
        if val is None:
            val_str = "N/A (RDKit not available)"
        elif fmt == "%":
            val_str = f"{val * 100:6.2f}%"
        else:
            val_str = f"{val:6.4f}"
        lines.append(f"  {label:<30s} {val_str}")

    lines.append("")

    # Counts
    lines.append(f"  {'Samples evaluated':<30s} {metrics['n_samples']:>6d}")
    lines.append(f"  {'Valid predictions':<30s} {metrics['n_valid']:>6d}")
    if metrics.get("backbone_match_n") is not None:
        lines.append(f"  {'Backbone comparisons':<30s} {metrics['backbone_match_n']:>6d}")
    lines.append("")
    lines.append("=" * 56)

    return "\n".join(lines)


def format_detailed_result(r: SampleResult) -> str:
    """Format a single sample result for detailed output."""
    lines = [f"--- {r.sample_id} ---"]
    lines.append(f"  valid:        {r.valid_pred}")
    if not r.valid_pred:
        lines.append(f"  parse_error:  {r.parse_error}")
        return "\n".join(lines)
    lines.append(f"  string_em:    {r.string_em}")
    lines.append(f"  backbone:     {r.backbone_match}")
    lines.append(f"  r_accuracy:   {r.r_accuracy:.3f}")
    lines.append(f"  r_f1:         {r.r_f1:.3f}")
    lines.append(f"  ext_match:    {r.extension_match}")
    lines.append(f"  stable_acc:   {r.stable_accuracy:.3f}")
    lines.append(f"  stable_em:    {r.stable_em}")
    lines.append(f"  markush_em:   {r.markush_em}")
    lines.append(f"  str_sim:      {r.string_similarity:.4f}")
    if not r.string_em:
        lines.append(f"  GT:  {r.gt_cxsmiles[:120]}{'…' if len(r.gt_cxsmiles) > 120 else ''}")
        lines.append(f"  PR:  {r.pred_cxsmiles[:120]}{'…' if len(r.pred_cxsmiles) > 120 else ''}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-level evaluation API (importable)
# ---------------------------------------------------------------------------


def evaluate_predictions(
    predictions: list[dict],
    ground_truths: list[dict],
    *,
    verbose: bool = False,
) -> tuple[dict, list[SampleResult]]:
    """Evaluate a list of predictions against ground truths.

    This is the primary API for programmatic use (e.g. training callbacks).

    Parameters
    ----------
    predictions : list[dict]
        Each dict has ``"id"`` and ``"prediction"`` keys.
    ground_truths : list[dict]
        Each dict has ``"id"`` and either ``"cxsmiles_opt"`` or ``"annotation"`` keys.
    verbose : bool
        If True, print per-sample results.

    Returns
    -------
    metrics : dict
        Aggregate metrics.
    sample_results : list[SampleResult]
        Per-sample evaluation details.
    """
    # Build GT lookup
    gt_by_id: dict[str, dict] = {}
    for entry in ground_truths:
        sid = entry.get("id")
        if sid is None:
            warnings.warn(f"Ground-truth entry missing 'id' field: {entry}", stacklevel=2)
            continue
        gt_by_id[sid] = entry

    sample_results: list[SampleResult] = []
    unmatched_preds = []

    for entry in predictions:
        sid = entry.get("id")
        if sid is None:
            warnings.warn(f"Prediction entry missing 'id' field: {entry}", stacklevel=2)
            continue

        gt_entry = gt_by_id.get(sid)
        if gt_entry is None:
            unmatched_preds.append(sid)
            continue

        # Parse prediction
        pred_raw = entry.get("prediction", "")
        pred = parse_markush_output(pred_raw)

        # Parse ground truth
        gt = parse_ground_truth_entry(gt_entry)

        # Evaluate
        result = evaluate_sample(sid, pred, gt)
        sample_results.append(result)

        if verbose:
            print(format_detailed_result(result))

    if unmatched_preds:
        warnings.warn(
            f"{len(unmatched_preds)} prediction(s) had no matching ground-truth entry. "
            f"First 5: {unmatched_preds[:5]}",
            stacklevel=2,
        )

    # Check for GT entries with no prediction
    pred_ids = {e.get("id") for e in predictions}
    missing_preds = [sid for sid in gt_by_id if sid not in pred_ids]
    if missing_preds:
        warnings.warn(
            f"{len(missing_preds)} ground-truth sample(s) had no prediction. "
            f"They are counted as invalid (score 0). First 5: {missing_preds[:5]}",
            stacklevel=2,
        )
        # Add as invalid results
        for sid in missing_preds:
            gt_entry = gt_by_id[sid]
            gt = parse_ground_truth_entry(gt_entry)
            result = SampleResult(
                sample_id=sid,
                valid_pred=False,
                valid_gt=gt.valid,
                gt_cxsmiles=gt.cxsmiles_opt,
                parse_error="no prediction provided",
            )
            sample_results.append(result)

    metrics = compute_metrics(sample_results)
    return metrics, sample_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Markush structure predictions against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL file with 'id' and 'prediction' fields",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="JSONL file with 'id' and 'cxsmiles_opt'/'annotation' fields",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for metrics (default: print to stdout)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print per-sample results for error analysis",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        default=True,
        help="Count GT samples with no prediction as invalid (default: True)",
    )
    parser.add_argument(
        "--no-include-missing",
        dest="include_missing",
        action="store_false",
        help="Only evaluate samples that have both prediction and GT",
    )
    args = parser.parse_args()

    # RDKit availability notice
    if not _RDKIT_AVAILABLE:
        print(
            "WARNING: RDKit not available. backbone_match and markush_em will be reported as N/A.",
            file=sys.stderr,
        )

    # Load data
    print(f"Loading predictions from {args.predictions}")
    predictions = load_jsonl(args.predictions)
    print(f"  → {len(predictions)} entries")

    print(f"Loading ground truth from {args.ground_truth}")
    ground_truths = load_jsonl(args.ground_truth)
    print(f"  → {len(ground_truths)} entries")

    # Evaluate
    metrics, sample_results = evaluate_predictions(
        predictions,
        ground_truths,
        verbose=args.detailed,
    )

    # If not including missing, filter them out and recompute
    if not args.include_missing:
        pred_ids = {e.get("id") for e in predictions}
        gt_ids = {e.get("id") for e in ground_truths}
        common_ids = pred_ids & gt_ids
        sample_results = [r for r in sample_results if r.sample_id in common_ids]
        metrics = compute_metrics(sample_results)

    # Print summary
    print()
    print(format_metrics_table(metrics))

    # Save to file
    if args.output:
        output_data = {
            "metrics": metrics,
            "per_sample": [
                {
                    "id": r.sample_id,
                    "valid": r.valid_pred,
                    "string_em": r.string_em,
                    "backbone_match": r.backbone_match,
                    "r_accuracy": r.r_accuracy,
                    "r_precision": r.r_precision,
                    "r_f1": r.r_f1,
                    "extension_match": r.extension_match,
                    "stable_accuracy": r.stable_accuracy,
                    "stable_em": r.stable_em,
                    "markush_em": r.markush_em,
                    "string_similarity": r.string_similarity,
                }
                for r in sample_results
            ],
        }
        Path(args.output).write_text(
            json.dumps(output_data, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
