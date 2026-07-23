#!/usr/bin/env python3
"""InChIKey-oriented Markush metric helpers.

The mixed Markush pipeline uses the richer MCS evaluator in
:mod:`glyph.markush.eval.metrics_mcs` as the primary MG2-style metric. This module provides the plan's named
``metrics_inchi`` entry point and a lightweight backbone InChIKey comparison
for callers that only need the backbone part.
"""

from __future__ import annotations

import contextlib
from typing import Any

from glyph.markush.eval import common
from glyph.markush.eval.metrics_mcs import evaluate_markush_mcs

_RDKIT_AVAILABLE = False
_RDKIT_IMPORT_ERROR: BaseException | None = None
try:  # pragma: no cover - availability depends on environment
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]
    _RDKIT_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    Chem = None  # ty: ignore[invalid-assignment]
    _RDKIT_IMPORT_ERROR = exc


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        extra = f" ({_RDKIT_IMPORT_ERROR})" if _RDKIT_IMPORT_ERROR else ""
        raise RuntimeError(f"RDKit is required for InChIKey metrics{extra}")


def backbone_smiles(cxsmiles_or_xml: str | None) -> str:
    """Extract the backbone SMILES with local ``<r>`` tags replaced by ``*``."""

    if not cxsmiles_or_xml:
        return ""
    cxsmi, _stable = common.extract_cxsmiles_and_stable(cxsmiles_or_xml, require_wrapper=False)
    value = cxsmi or str(cxsmiles_or_xml)
    stripped = common.strip_r_tags(value)
    core, _ext = common.split_cxsmiles(stripped, strip_trailing_pipe=True)
    return core


def backbone_inchikey(cxsmiles_or_xml: str | None, *, ignore_stereo: bool = True) -> str | None:
    """Return a backbone InChIKey after wildcard-to-carbon substitution."""

    _require_rdkit()
    core = backbone_smiles(cxsmiles_or_xml)
    if not core:
        return None
    smiles = core.replace("*", "C")
    try:
        params = Chem.SmilesParserParams()
        params.sanitize = False
        params.strictCXSMILES = False
        mol = Chem.MolFromSmiles(smiles, params)
        if mol is None:
            return None
        if ignore_stereo:
            Chem.RemoveStereochemistry(mol)
            for bond in mol.GetBonds():
                bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
        with contextlib.suppress(Exception):
            Chem.SanitizeMol(mol)
        inchi = Chem.MolToInchi(mol, options="/SNon" if ignore_stereo else "")
        return Chem.InchiToInchiKey(inchi) if inchi else None
    except Exception:
        return None


def backbone_inchikey_match(
    prediction: str | None,
    ground_truth: str | None,
    *,
    ignore_stereo: bool = True,
) -> bool | None:
    """Compare two Markush backbones by InChIKey.

    Returns ``None`` when either side cannot be parsed.
    """

    pred_key = backbone_inchikey(prediction, ignore_stereo=ignore_stereo)
    gt_key = backbone_inchikey(ground_truth, ignore_stereo=ignore_stereo)
    if pred_key is None or gt_key is None:
        return None
    return pred_key == gt_key


def evaluate_predictions_inchi(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    *,
    include_missing: bool = True,
    return_details: bool = False,
    protocol: str = "mg2",
) -> dict[str, Any] | tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate Markush predictions with the primary MG2-style metric.

    This delegates to :func:`glyph.markush.eval.metrics_mcs.evaluate_markush_mcs`,
    which includes backbone InChI equality plus R-group, m-section, Sg-section,
    and stable-table breakdowns.  ``protocol="mg2"`` is the default here because
    this module reports the MG2-style metric; note that ``"mg2"`` is an
    *approximate* in-package replication (not byte-exact). Official/headline MG2
    numbers require the pinned DS4SD scorer (``tools/mg2_scorer/``).
    """

    return evaluate_markush_mcs(
        predictions,
        ground_truths,
        include_missing=include_missing,
        return_details=return_details,
        protocol=protocol,
    )
