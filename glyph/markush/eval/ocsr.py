"""Strict SMILES metrics for ordinary OCSR rows.

The unified Markush trainer can generate both Markush XML and bare molecule
SMILES. This module scores the bare-SMILES side without extracting content from
accidental Markush XML, so target-format confusion remains visible.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem, RDLogger

from glyph.markush.eval.common import clean_prediction, extract_cxsmiles_and_stable

RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]

GROUND_TRUTH_FIELDS = ("smiles", "gt_smiles", "target", "assistant_text", "label")
PREDICTION_FIELDS = ("prediction", "pred", "output", "smiles", "target", "assistant_text")
OCSR_PREDICTION_MODE_NATIVE_SMILES = "native_smiles"
OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION = "markush_no_extension"
OCSR_PREDICTION_MODE_ALIASES = {
    "native": OCSR_PREDICTION_MODE_NATIVE_SMILES,
    "smiles": OCSR_PREDICTION_MODE_NATIVE_SMILES,
    "bare_smiles": OCSR_PREDICTION_MODE_NATIVE_SMILES,
    OCSR_PREDICTION_MODE_NATIVE_SMILES: OCSR_PREDICTION_MODE_NATIVE_SMILES,
    "markush": OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
    "no_extension_markush": OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
    "markush_cxsmiles_no_extension": OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
    "cxsmiles_no_extension": OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
    OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION: OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
}
ALLOWED_MARKUSH_OUTPUT_TAGS = {"markush", "cxsmi", "stable"}
XML_TAG_RE = re.compile(r"</?\s*([a-zA-Z0-9_:-]+)\b[^>]*>")


@dataclass(slots=True)
class OCSREvaluationConfig:
    """Configuration for OCSR prediction parsing.

    ``native_smiles`` is the benchmark-compatible default: predictions must be
    bare RDKit-parseable SMILES. ``markush_no_extension`` is opt-in and accepts
    no-extension Markush/XML outputs by extracting the ``<cxsmi>`` payload,
    while rejecting any Markush-only annotations.
    """

    prediction_mode: str = OCSR_PREDICTION_MODE_NATIVE_SMILES

    def normalized_prediction_mode(self) -> str:
        """Return a canonical prediction-mode name or raise for unknown modes."""

        return normalize_ocsr_prediction_mode(self.prediction_mode)


@dataclass(slots=True)
class OCSRSmilesEvaluation:
    """Per-sample OCSR SMILES evaluation result."""

    sample_id: str
    prediction: str
    ground_truth: str
    pred_canonical: str = ""
    gt_canonical: str = ""
    pred_graph_canonical: str = ""
    gt_graph_canonical: str = ""
    valid_prediction: bool = False
    exact_match: bool = False
    graph_exact_match: bool = False
    markush_xml_prediction: bool = False
    markush_extension_prediction: bool = False
    extracted_prediction: str = ""
    prediction_mode: str = OCSR_PREDICTION_MODE_NATIVE_SMILES
    prediction_error: str = ""


@dataclass(slots=True)
class _PredictionParse:
    mol: Any | None = None
    extracted: str = ""
    markush_xml_prediction: bool = False
    markush_extension_prediction: bool = False
    prediction_error: str = ""


def normalize_ocsr_prediction_mode(mode: str | None) -> str:
    """Normalize OCSR evaluator mode aliases."""

    key = (mode or OCSR_PREDICTION_MODE_NATIVE_SMILES).strip().lower().replace("-", "_")
    normalized = OCSR_PREDICTION_MODE_ALIASES.get(key)
    if normalized is None:
        valid = ", ".join(
            sorted(
                {
                    OCSR_PREDICTION_MODE_NATIVE_SMILES,
                    OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
                }
            )
        )
        raise ValueError(f"Unknown OCSR prediction mode {mode!r}; expected one of: {valid}")
    return normalized


def looks_like_markush_xml(value: str | None) -> bool:
    """Return true when a prediction appears to be wrapped Markush/XML output."""

    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(tag in lowered for tag in ("<markush", "<cxsmi", "<stable", "<r>"))


def _looks_like_markush_or_cxsmiles(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    return looks_like_markush_xml(value) or "|" in value


def _strict_bare_smiles_mol(value: str | None) -> Any | None:
    """Parse a strict bare SMILES string, rejecting XML/CXSMILES wrappers."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if looks_like_markush_xml(stripped) or "|" in stripped:
        return None
    if any(char.isspace() for char in stripped):
        return None
    return Chem.MolFromSmiles(stripped)


def _canonical_from_mol(mol: Any | None) -> str | None:
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _graph_canonical_from_mol(mol: Any | None) -> str | None:
    if mol is None:
        return None
    graph_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(graph_mol)
    return Chem.MolToSmiles(graph_mol, canonical=True, isomericSmiles=True)


def canonical_smiles(value: str | None) -> str | None:
    """Return RDKit canonical isomeric SMILES for a strict bare SMILES input."""

    mol = _strict_bare_smiles_mol(value)
    return _canonical_from_mol(mol)


def graph_canonical_smiles(value: str | None) -> str | None:
    """Return canonical SMILES with stereochemistry removed for a strict bare SMILES input."""

    mol = _strict_bare_smiles_mol(value)
    return _graph_canonical_from_mol(mol)


def _has_disallowed_xml_tags(value: str) -> bool:
    """Return true when XML-like output contains tags beyond no-extension Markush."""

    for match in XML_TAG_RE.finditer(value):
        tag = match.group(1).split(":", 1)[-1].lower()
        if tag not in ALLOWED_MARKUSH_OUTPUT_TAGS:
            return True
    return False


def _has_markush_extension(value: str, *, stable: str = "") -> bool:
    """Return true for Markush/CXSMILES annotation content disallowed for OCSR."""

    if stable.strip():
        return True
    lowered = value.lower()
    if "<r" in lowered or "</r" in lowered:
        return True
    # Any CXSMILES extension section can carry Markush-only annotations
    # (R labels, Sg/M sections, atom props, etc.). The opt-in OCSR mode accepts
    # only the base molecule payload.
    return "|" in value


def _parse_native_prediction(value: str) -> _PredictionParse:
    markush_xml_prediction = looks_like_markush_xml(value)
    mol = _strict_bare_smiles_mol(value)
    return _PredictionParse(
        mol=mol,
        extracted=value if mol is not None else "",
        markush_xml_prediction=markush_xml_prediction,
        prediction_error=(
            ""
            if mol is not None
            else "markush_xml_prediction"
            if markush_xml_prediction
            else "invalid_smiles"
        ),
    )


def _parse_markush_no_extension_prediction(value: str) -> _PredictionParse:
    markush_like = _looks_like_markush_or_cxsmiles(value)
    if not markush_like:
        return _parse_native_prediction(value)

    markush_xml_prediction = looks_like_markush_xml(value)
    if _has_disallowed_xml_tags(value):
        return _PredictionParse(
            markush_xml_prediction=markush_xml_prediction,
            markush_extension_prediction=True,
            prediction_error="markush_extension_prediction",
        )

    if "|" in value and not markush_xml_prediction:
        return _PredictionParse(
            markush_xml_prediction=False,
            markush_extension_prediction=True,
            prediction_error="markush_extension_prediction",
        )

    cxsmi, stable = extract_cxsmiles_and_stable(value, require_wrapper=False)
    if not cxsmi:
        return _PredictionParse(
            markush_xml_prediction=markush_xml_prediction,
            prediction_error="markush_xml_prediction"
            if markush_xml_prediction
            else "invalid_smiles",
        )
    if _has_markush_extension(cxsmi, stable=stable):
        return _PredictionParse(
            extracted=cxsmi,
            markush_xml_prediction=markush_xml_prediction,
            markush_extension_prediction=True,
            prediction_error="markush_extension_prediction",
        )
    mol = _strict_bare_smiles_mol(cxsmi)
    return _PredictionParse(
        mol=mol,
        extracted=cxsmi if mol is not None else "",
        markush_xml_prediction=markush_xml_prediction,
        prediction_error="" if mol is not None else "invalid_smiles",
    )


def _parse_prediction(value: str, *, prediction_mode: str) -> _PredictionParse:
    if prediction_mode == OCSR_PREDICTION_MODE_NATIVE_SMILES:
        return _parse_native_prediction(value)
    if prediction_mode == OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION:
        return _parse_markush_no_extension_prediction(value)
    raise ValueError(f"Unsupported normalized OCSR prediction mode: {prediction_mode}")


def _row_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("id", "sample_id", "uid"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(fallback)


def _text_from_entry(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for key in fields:
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def evaluate_ocsr_smiles(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    *,
    include_missing: bool = True,
    return_details: bool = False,
    config: OCSREvaluationConfig | None = None,
    prediction_mode: str | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate OCSR predictions under an explicit prediction parsing contract.

    Ground-truth rows that cannot be parsed as strict bare SMILES are excluded
    from the denominator and counted as ``n_invalid_gt``. Missing predictions
    for parseable rows count as invalid, incorrect predictions when
    ``include_missing`` is true. By default predictions must be native bare
    SMILES; opt-in ``markush_no_extension`` mode accepts no-extension
    Markush/XML predictions after extracting their base molecule.
    """

    if config is not None and prediction_mode is not None:
        raise ValueError("Pass either config or prediction_mode, not both.")
    eval_config = config or OCSREvaluationConfig(
        prediction_mode=prediction_mode or OCSR_PREDICTION_MODE_NATIVE_SMILES
    )
    normalized_prediction_mode = eval_config.normalized_prediction_mode()

    pred_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(predictions):
        sample_id = _row_id(row, index)
        if sample_id in pred_by_id:
            warnings.warn(
                f"Duplicate OCSR prediction id {sample_id!r}; keeping last row.",
                stacklevel=2,
            )
        pred_by_id[sample_id] = row

    gt_ids: set[str] = set()
    details: list[OCSRSmilesEvaluation] = []
    n_total_gt = 0
    n_evaluated = 0
    n_invalid_gt = 0
    n_correct = 0
    n_correct_graph = 0
    n_valid_pred = 0
    n_missing_pred = 0
    n_markush_xml_pred = 0
    n_markush_extension_pred = 0

    for index, gt_row in enumerate(ground_truths):
        sample_id = _row_id(gt_row, index)
        gt_ids.add(sample_id)
        n_total_gt += 1
        gt_text = _text_from_entry(gt_row, GROUND_TRUTH_FIELDS)
        gt_canonical = canonical_smiles(gt_text)
        gt_graph = graph_canonical_smiles(gt_text)
        if gt_canonical is None or gt_graph is None:
            n_invalid_gt += 1
            details.append(
                OCSRSmilesEvaluation(
                    sample_id=sample_id,
                    prediction="",
                    ground_truth=gt_text,
                    prediction_mode=normalized_prediction_mode,
                    prediction_error="invalid_ground_truth",
                )
            )
            continue

        pred_row = pred_by_id.get(sample_id)
        if pred_row is None:
            if not include_missing:
                continue
            prediction = ""
            n_missing_pred += 1
            prediction_error = "missing_prediction"
        else:
            prediction = clean_prediction(_text_from_entry(pred_row, PREDICTION_FIELDS))
            prediction_error = ""

        n_evaluated += 1
        parsed_prediction = _parse_prediction(
            prediction,
            prediction_mode=normalized_prediction_mode,
        )
        if parsed_prediction.markush_xml_prediction:
            n_markush_xml_pred += 1
        if parsed_prediction.markush_extension_prediction:
            n_markush_extension_pred += 1
        pred_canonical = _canonical_from_mol(parsed_prediction.mol)
        pred_graph = _graph_canonical_from_mol(parsed_prediction.mol)
        valid_prediction = pred_canonical is not None
        if valid_prediction:
            n_valid_pred += 1
        elif not prediction_error:
            prediction_error = parsed_prediction.prediction_error or "invalid_smiles"
        exact = pred_canonical is not None and pred_canonical == gt_canonical
        graph_exact = pred_graph is not None and pred_graph == gt_graph
        if exact:
            n_correct += 1
        if graph_exact:
            n_correct_graph += 1
        details.append(
            OCSRSmilesEvaluation(
                sample_id=sample_id,
                prediction=prediction,
                ground_truth=gt_text,
                pred_canonical=pred_canonical or "",
                gt_canonical=gt_canonical,
                pred_graph_canonical=pred_graph or "",
                gt_graph_canonical=gt_graph,
                valid_prediction=valid_prediction,
                exact_match=exact,
                graph_exact_match=graph_exact,
                markush_xml_prediction=parsed_prediction.markush_xml_prediction,
                markush_extension_prediction=parsed_prediction.markush_extension_prediction,
                extracted_prediction=parsed_prediction.extracted,
                prediction_mode=normalized_prediction_mode,
                prediction_error=prediction_error,
            )
        )

    n_extra_pred = len([sample_id for sample_id in pred_by_id if sample_id not in gt_ids])
    denominator = n_evaluated or 1
    metrics: dict[str, Any] = {
        "n_samples": n_evaluated,
        "n_total_gt": n_total_gt,
        "n_invalid_gt": n_invalid_gt,
        "n_correct": n_correct,
        "n_correct_graph": n_correct_graph,
        "n_valid_pred": n_valid_pred,
        "n_invalid_pred": n_evaluated - n_valid_pred,
        "n_missing_pred": n_missing_pred,
        "n_extra_pred": n_extra_pred,
        "n_markush_xml_pred": n_markush_xml_pred,
        "n_markush_extension_pred": n_markush_extension_pred,
        "canon_exact_match": n_correct / denominator if n_evaluated else 0.0,
        "graph_exact_match": n_correct_graph / denominator if n_evaluated else 0.0,
        "valid_smiles_rate": n_valid_pred / denominator if n_evaluated else 0.0,
        "markush_xml_prediction_rate": (n_markush_xml_pred / denominator if n_evaluated else 0.0),
        "markush_extension_prediction_rate": (
            n_markush_extension_pred / denominator if n_evaluated else 0.0
        ),
        "prediction_mode": normalized_prediction_mode,
    }
    detail_dicts = [asdict(item) for item in details]
    return (metrics, detail_dicts) if return_details else metrics


def flatten_ocsr_metrics(metrics: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """Return slashed trainer/W&B keys for OCSR metrics."""

    return {f"{prefix}/{key}": value for key, value in metrics.items()}
