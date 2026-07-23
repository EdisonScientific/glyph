#!/usr/bin/env python3
"""MCS-based evaluation for Markush predictions in the local XML/CXSMILES format.

This module ports the core ideas from MarkushGrapher-2's public evaluator
(`markushgrapher/utils/ocsr/utils_evaluation.py`) without depending on the
MarkushGrapher package stack.  The important difference from string exact match
is that CXSMILES atom order/canonicalization differences are handled by RDKit
MCS atom mapping before R-group, M-section, and Sg-section checks.

Prediction format supported here::

    <markush><cxsmi><r>R0</r>C.O=C1...</cxsmi><stable>R0:hydroxyl<n>carboxyl<ns></stable></markush>

The ``<r>...</r>`` tags are converted to wildcard atoms (``*``) before RDKit
parsing, while the labels are retained in an atom-index map for MCS-aligned
R-group scoring.

CLI::

    python eval_mcs.py --predictions preds.jsonl --ground-truth gt.jsonl --output metrics.json

Importable API::

    from eval_mcs import evaluate_markush_mcs

    metrics = evaluate_markush_mcs(predictions, ground_truths)
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from glyph.markush.eval import common
from glyph.markush.eval.markush_graph import markush_graph_equal

# ---------------------------------------------------------------------------
# Evaluation protocol toggle
# ---------------------------------------------------------------------------
#
# Two protocols are supported:
#
# - "ours": our stricter eval. Penalizes false-positive R/M/Sg sections,
#   requires bijection in `cxsmi_equality`, compares all SGroup props,
#   penalizes extra labels in `stable_equality`, treats unmapped GT M-ring
#   atoms as a failure, and ORs whole-molecule backbone InChI with the
#   fragment-aligned one. Default.
#
# - "mg2": an *approximate*, in-package replication of MarkushGrapher-2's
#   `markushgrapher/utils/ocsr/utils_evaluation.py`. Recall-only semantics,
#   PrepareMolForDrawing before InChI, SGroup compared by LABEL only, MG-style
#   stable normalization, `stable_equality` ignores hallucinated labels, etc.
#
# `"strict"` is an accepted alias for `"ours"`.
#
# The `"mg2"` protocol is APPROXIMATE, not authoritative: several
# MG2-replication deltas are NOT implemented under `"mg2"`, so the `"mg2"`
# numbers approximate rather than byte-match upstream. Still deferred:
# the aromatize-via-carbon trick (temporarily flip R-atoms to carbon before
# sanitize, then back); kekulization, `Chem.RemoveHs`, and
# `PrepareMolForDrawing` inside the per-fragment
# `compute_molecule_prediction_quality`; the `canonicalize_markush` CXSMILES
# pre-pass; and `fix_cxsmiles` abbreviation expansion. Of these,
# aromatize-via-carbon is the largest expected source of numerical divergence
# for R-groups in aromatic rings. The official IP5-M contract does not depend on
# this path: it uses the pinned upstream DS4SD scorer directly (see
# docs/reproducing.md and tools/mg2_scorer/).

EvalProtocol = Literal["ours", "mg2"]

_PROTOCOL_ALIASES: dict[str, EvalProtocol] = {
    "strict": "ours",  # accepted alias; callers may pass either name
}

_VALID_PROTOCOLS: tuple[EvalProtocol, ...] = ("ours", "mg2")


def _normalize_protocol(protocol: str) -> EvalProtocol:
    """Canonicalize a protocol name, accepting known aliases."""
    canon = _PROTOCOL_ALIASES.get(protocol, protocol)
    if canon not in _VALID_PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; expected one of {_VALID_PROTOCOLS}")
    return canon  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Optional RDKit import.  Importing this module should still work on machines
# without RDKit; actually running MCS evaluation requires it.
# ---------------------------------------------------------------------------

_RDKIT_AVAILABLE = False
_RDKIT_IMPORT_ERROR: BaseException | None = None
rdMolDraw2D = None  # type: ignore[assignment]  # noqa: N816
try:  # pragma: no cover - exercised by CLI/environment checks
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFMCS, rdmolfiles
    from rdkit.Chem.inchi import MolToInchi

    RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]
    _RDKIT_AVAILABLE = True

    # rdMolDraw2D needs X11 libs (libXrender) — optional on headless nodes.
    # Only used for PrepareMolForDrawing in MG2 protocol.
    with contextlib.suppress(ImportError):
        from rdkit.Chem.Draw import rdMolDraw2D
except Exception as exc:  # ImportError on normal missing installs; keep robust.
    Chem = None  # ty: ignore[invalid-assignment]
    DataStructs = None  # ty: ignore[invalid-assignment]
    rdFMCS = None  # ty: ignore[invalid-assignment]  # noqa: N816  (matches rdkit module name)
    rdmolfiles = None  # ty: ignore[invalid-assignment]
    MolToInchi = None  # ty: ignore[invalid-assignment]
    _RDKIT_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Parsing helpers for local Markush XML/CXSMILES format
#
# Regex constants and core extractors live in ``glyph.markush.eval.common``; the
# module-level names below are kept as thin aliases so existing callers and
# tests keep working.
# ---------------------------------------------------------------------------

_RE_CXSMI = common.RE_CXSMI
_RE_STABLE = common.RE_STABLE
_RE_R_TAG = common.RE_R_TAG
_RE_M_SECTION = common.RE_M_SECTION


@dataclass
class ParsedMarkush:
    """A parsed prediction or ground-truth Markush string."""

    raw: str
    cxsmiles: str = ""
    rdkit_cxsmiles: str = ""  # cxsmiles with <r>label</r> replaced by *
    stable_raw: str = ""
    stable: dict[str, list[str]] = field(default_factory=dict)
    atom_labels: dict[int, str] = field(default_factory=dict)  # RDKit atom idx -> R label
    valid: bool = False
    parse_error: str = ""


@dataclass
class SampleEvaluation:
    """Per-sample MCS evaluation result."""

    sample_id: str
    valid_pred: bool
    valid_gt: bool
    prediction_error: str = ""
    ground_truth_error: str = ""
    pred_cxsmiles: str = ""
    gt_cxsmiles: str = ""
    scores: dict[str, Any] = field(default_factory=dict)


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        extra = f" ({_RDKIT_IMPORT_ERROR})" if _RDKIT_IMPORT_ERROR else ""
        raise RuntimeError(
            "RDKit is required for MCS Markush evaluation but is not installed/importable"
            f"{extra}. Install it with `pip install rdkit` or `conda install -c conda-forge rdkit`."
        )


def _apply_position_graph_equality(
    scores: dict[str, Any],
    gt_smiles: str,
    predicted_smiles: str,
    *,
    remove_stereo: bool,
) -> None:
    """Stamp the position-aware Markush equality verdict onto a score dict.

    The legacy MCS code still supplies soft diagnostic metrics, but the
    default headline equality must be the full relation from
    :mod:`glyph.markush.eval.markush_graph`: same graph, same label atoms, same
    variable-attachment sets, same Sgroups, and same fragment count.
    """

    result = markush_graph_equal(gt_smiles, predicted_smiles, remove_stereo=remove_stereo)
    error = result.error or result.gt_error or result.pred_error
    scores["position_graph_equal"] = result.equal
    scores["position_graph_valid_gt"] = result.valid_gt
    scores["position_graph_valid_pred"] = result.valid_pred
    scores["position_graph_error"] = error
    scores["position_graph_states_visited"] = result.states_visited
    scores["cxsmi_equality"] = bool(result.equal and result.valid_gt and result.valid_pred)
    scores["cxsmiles_full_match"] = scores["cxsmi_equality"]
    if not result.valid_gt or not result.valid_pred:
        scores["valid"] = False


def _extract_cxsmiles_and_stable(raw: str) -> tuple[str, str]:
    """Extract the ``<cxsmi>`` and ``<stable>`` payloads from an output string.

    Unlike :func:`glyph.markush.eval.metrics_string._extract_cxsmiles_and_stable`,
    this helper does **not** require the ``<markush>`` wrapper — it
    matches ``<cxsmi>`` and ``<stable>`` independently anywhere in the
    input.
    """
    return common.extract_cxsmiles_and_stable(raw, require_wrapper=False)


def strip_r_tags(cxsmiles: str) -> str:
    """Replace every local ``<r>LABEL</r>`` marker with a wildcard atom ``*``."""
    return common.strip_r_tags(cxsmiles)


def _split_cxsmiles(cxsmiles: str) -> tuple[str, str]:
    """Return ``(SMILES core, CX extension)`` using the first ``|`` delimiter.

    A trailing ``|`` on the extension is stripped so that RDKit's expected
    ``SMILES |...|`` form yields a clean extension payload.
    """
    return common.split_cxsmiles(cxsmiles, strip_trailing_pipe=True)


def _rdkit_parseable_cxsmiles(cxsmiles: str) -> str:
    """Return CXSMILES in RDKit's expected ``SMILES |...|`` form.

    MarkushGrapher/local targets often omit the optional space before the CX
    extension (``CC|Sg:...|``).  RDKit's parser accepts many CX extensions only
    with that separating space, so normalize before parsing while leaving metric
    string comparisons untouched.
    """
    core, ext = _split_cxsmiles(cxsmiles)
    return f"{core} |{ext}|" if ext else core


def _parse_stable(stable_raw: str) -> dict[str, list[str]]:
    """Parse ``R0:alt1<n>alt2<ns>R1:alt`` into ``{label: [alts...]}``.

    The local training/evaluation format uses ``<ns>`` between rows and ``<n>``
    between alternatives.  This parser is intentionally conservative but also
    accepts ``=`` as a label separator for mildly malformed model outputs
    (``permissive=True`` in :func:`glyph.markush.eval.common.parse_stable`).
    """
    return common.parse_stable(stable_raw, permissive=True)


def _mol_from_cxsmiles(cxsmiles: str, *, sanitize: bool = True):
    """Parse a SMILES/CXSMILES string with RDKit's lenient CXSMILES settings."""
    _require_rdkit()
    if not cxsmiles:
        return None
    params = Chem.SmilesParserParams()
    params.strictCXSMILES = False
    params.sanitize = False
    params.removeHs = False
    try:
        mol = Chem.MolFromSmiles(_rdkit_parseable_cxsmiles(cxsmiles), params)
    except Exception:
        return None
    if mol is None:
        return None
    if sanitize:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
            )
        except Exception:
            return None
    return mol


def _labels_for_inserted_wildcards(cxsmiles: str) -> dict[int, str]:
    """Map stripped-string wildcard ordinal (1-based) to the originating R label."""
    labels_by_ordinal: dict[int, str] = {}
    wildcard_count = 0
    pos = 0
    core = _split_cxsmiles(cxsmiles)[0]
    for match in _RE_R_TAG.finditer(core):
        segment = core[pos : match.start()]
        wildcard_count += segment.count("*")
        wildcard_count += 1
        labels_by_ordinal[wildcard_count] = match.group(1).strip()
        pos = match.end()
    return labels_by_ordinal


def _tag_labels_by_atom_index(
    rdkit_cxsmiles: str, labels_by_ordinal: dict[int, str]
) -> dict[int, str]:
    """Assign local ``<r>`` labels to inserted wildcard atom indices.

    RDKit's CXSMILES parser exposes ``*`` atoms but not their origin.  To avoid
    shifting labels onto pre-existing wildcard atoms, ``labels_by_ordinal`` marks
    only wildcard ordinals that came from ``<r>`` replacements.
    """
    if not labels_by_ordinal or not _RDKIT_AVAILABLE:
        return {}
    mol = _mol_from_cxsmiles(rdkit_cxsmiles)
    if mol is None:
        return {}
    wildcard_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    labels: dict[int, str] = {}
    for ordinal, label in labels_by_ordinal.items():
        if 1 <= ordinal <= len(wildcard_indices):
            labels[wildcard_indices[ordinal - 1]] = label
    return labels


def _rdkit_atom_labels(mol, explicit_labels: dict[int, str] | None = None) -> dict[int, str]:
    """Collect R-group labels from explicit tag maps and RDKit atom properties."""
    labels: dict[int, str] = dict(explicit_labels or {})
    if mol is None:
        return labels
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in labels:
            continue
        if atom.HasProp("atomLabel"):
            labels[idx] = atom.GetProp("atomLabel")
        elif atom.HasProp("molFileAlias"):
            labels[idx] = atom.GetProp("molFileAlias")
        elif atom.GetAtomicNum() == 0 and atom.GetIsotope():
            labels[idx] = str(atom.GetIsotope())
    return labels


def parse_markush_text(raw: Any, *, strict: bool = False) -> ParsedMarkush:
    """Parse local Markush XML or a bare CXSMILES string.

    Parameters
    ----------
    raw:
        Model output, annotation XML, or a bare ``cxsmiles_opt`` value.
    strict:
        If true, require ``<cxsmi>...</cxsmi>``.  The CLI/API use lenient mode
        for both predictions and GT so JSONL rows with direct ``cxsmiles_opt``
        fields remain evaluable.
    """
    parsed = ParsedMarkush(raw="" if raw is None else str(raw))
    if not isinstance(raw, str) or not raw.strip():
        parsed.parse_error = "empty or non-string Markush value"
        return parsed

    raw = raw.strip()
    cxsmiles, stable_raw = _extract_cxsmiles_and_stable(raw)
    if not cxsmiles:
        if strict:
            parsed.parse_error = "missing <cxsmi>...</cxsmi> section"
            return parsed
        # Lenient fallback for bare cxsmiles_opt fields.
        if "<markush" in raw.lower() or "<cxsmi" in raw.lower():
            parsed.parse_error = "malformed Markush XML/CXSMILES wrapper"
            return parsed
        cxsmiles = raw

    labels_by_ordinal = _labels_for_inserted_wildcards(cxsmiles)
    rdkit_cxsmiles = strip_r_tags(cxsmiles)

    parsed.cxsmiles = cxsmiles
    parsed.rdkit_cxsmiles = rdkit_cxsmiles
    parsed.stable_raw = stable_raw
    parsed.stable = _parse_stable(stable_raw)
    parsed.atom_labels = _tag_labels_by_atom_index(rdkit_cxsmiles, labels_by_ordinal)
    parsed.valid = True
    return parsed


def _prediction_text_from_entry(entry: dict[str, Any]) -> str:
    for key in ("prediction", "pred", "output", "cxsmiles_opt", "annotation"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _ground_truth_text_from_entry(entry: dict[str, Any]) -> str:
    """Prefer clean ``cxsmiles_opt`` but preserve ``<stable>`` from annotation."""
    cxsmiles = entry.get("cxsmiles_opt")
    annotation = entry.get("annotation", "")
    if isinstance(cxsmiles, str) and cxsmiles.strip():
        _, stable_raw = _extract_cxsmiles_and_stable(
            annotation if isinstance(annotation, str) else ""
        )
        return f"<markush><cxsmi>{cxsmiles.strip()}</cxsmi><stable>{stable_raw}</stable></markush>"
    if isinstance(annotation, str) and annotation.strip():
        return annotation
    return ""


# ---------------------------------------------------------------------------
# Molecule scoring primitives
# ---------------------------------------------------------------------------


def _copy_with_wildcards_as_carbon(cxsmiles: str, *, remove_stereo: bool = True):
    mol = _mol_from_cxsmiles(cxsmiles)
    if mol is None:
        return None
    mol = copy.deepcopy(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            atom.SetIsotope(0)
        if atom.HasProp("atomLabel"):
            atom.ClearProp("atomLabel")
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
        for bond in mol.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    with contextlib.suppress(Exception):
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
        )
    return mol


def _inchi_for_backbone(
    cxsmiles: str, *, remove_stereo: bool = True, protocol: str = "ours"
) -> str | None:
    """InChI for a backbone after wildcard -> carbon and stereo removal.

    Parameters
    ----------
    protocol:
        ``"mg2"`` calls ``rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)``
        before InChI computation, matching MG2's
        ``compute_molecule_prediction_quality``.
        ``"ours"`` computes InChI directly without that draw-preparation step.
    """
    protocol = _normalize_protocol(protocol)
    if not cxsmiles:
        return None
    mol = _copy_with_wildcards_as_carbon(cxsmiles, remove_stereo=remove_stereo)
    if mol is None:
        return None
    try:
        if protocol == "mg2" and rdMolDraw2D is not None:
            rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
        options = "/SNon" if remove_stereo else ""
        inchi = MolToInchi(mol, options=options)
    except Exception:
        return None
    return inchi or None


def _fingerprint_tanimoto(predicted_cxsmiles: str, gt_cxsmiles: str) -> float:
    pred_mol = _copy_with_wildcards_as_carbon(predicted_cxsmiles)
    gt_mol = _copy_with_wildcards_as_carbon(gt_cxsmiles)
    if pred_mol is None or gt_mol is None:
        return 0.0
    try:
        return float(
            DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(gt_mol), Chem.RDKFingerprint(pred_mol)
            )
        )
    except Exception:
        return 0.0


def _replace_wildcards_for_fragment(
    fragment_smiles: str, *, remove_stereo: bool = True
) -> str | None:
    mol = _copy_with_wildcards_as_carbon(fragment_smiles, remove_stereo=remove_stereo)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_molecule_prediction_quality(
    predicted_smiles: str,
    gt_smiles: str,
    *,
    remove_stereo: bool = True,
) -> dict[str, Any]:
    """Small RDKit-only molecule quality helper used for aligned fragments."""
    scores = {
        "tanimoto": 0.0,
        "tanimoto1": False,
        "inchi_equality": False,
        "string_equality": predicted_smiles == gt_smiles,
        "valid": False,
    }
    pred_mol = _copy_with_wildcards_as_carbon(predicted_smiles, remove_stereo=remove_stereo)
    gt_mol = _copy_with_wildcards_as_carbon(gt_smiles, remove_stereo=remove_stereo)
    if pred_mol is None or gt_mol is None:
        return scores
    scores["valid"] = True
    try:
        tanimoto = float(
            DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(gt_mol), Chem.RDKFingerprint(pred_mol)
            )
        )
        scores["tanimoto"] = tanimoto
        scores["tanimoto1"] = tanimoto == 1.0
    except Exception:
        pass
    try:
        options = "/SNon" if remove_stereo else ""
        gt_inchi = MolToInchi(gt_mol, options=options)
        pred_inchi = MolToInchi(pred_mol, options=options)
        scores["inchi_equality"] = bool(gt_inchi and gt_inchi == pred_inchi)
    except Exception:
        pass
    return scores


def _fragment_mols_and_indices(mol) -> list[tuple[Any, tuple[int, ...]]]:
    fragments: list[tuple[Any, tuple[int, ...]]] = []
    parser_params = Chem.SmilesParserParams()
    parser_params.strictCXSMILES = False
    parser_params.sanitize = False
    parser_params.removeHs = False
    for atom_indices in Chem.GetMolFrags(mol):
        try:
            frag_smiles = rdmolfiles.MolFragmentToSmiles(mol, atomsToUse=atom_indices)
            frag_mol = Chem.MolFromSmiles(frag_smiles, parser_params)
            if frag_mol is not None:
                with contextlib.suppress(Exception):
                    Chem.SanitizeMol(
                        frag_mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                        ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
                    )
                fragments.append((frag_mol, tuple(atom_indices)))
        except Exception:
            continue
    return fragments


def _parse_m_sections(cxsmiles: str) -> list[dict[str, Any]]:
    """Parse CXSMILES positional-variation ``m:connector:ring.ring`` sections."""
    _, ext = _split_cxsmiles(cxsmiles)
    sections: list[dict[str, Any]] = []
    for match in _RE_M_SECTION.finditer(ext):
        connector = int(match.group(1))
        ring_atoms = [int(tok) for tok in match.group(2).split(".") if tok != ""]
        sections.append({"atom_connector": connector, "ring_atoms": ring_atoms})
    return sections


def _sgroup_signature(sgroup, *, protocol: str = "ours") -> dict[str, str]:
    """Comparable non-index SGroup properties.

    Atom indices are checked via the MCS mapping; this signature covers the
    frequency-variation type/label/connectivity metadata RDKit exposes from the
    CXSMILES Sg section.

    Parameters
    ----------
    protocol:
        ``"mg2"`` compares only the ``"LABEL"`` property, matching MG2.
        ``"ours"`` compares ALL SGroup properties (except ``"index"``).
    """
    protocol = _normalize_protocol(protocol)
    props: dict[str, str] = {}
    if protocol == "mg2":
        try:
            if sgroup.HasProp("LABEL"):
                props["LABEL"] = sgroup.GetProp("LABEL")
        except Exception:
            pass
        return props
    try:
        prop_names = list(sgroup.GetPropNames())
    except Exception:
        prop_names = []
    for name in prop_names:
        if name == "index":
            continue
        try:
            props[name] = sgroup.GetProp(name)
        except Exception:
            continue
    return props


def _align_fragments_by_mcs(
    predicted_mol,
    gt_mol,
    pred_atom_labels: dict[int, str],
    gt_atom_labels: dict[int, str],
    *,
    remove_stereo: bool,
    verbose: bool = False,
    protocol: str = "ours",
) -> tuple[dict[str, Any], dict[int, list[int]]]:
    """Align GT fragments to predicted fragments and build GT atom -> pred atom map.

    Parameters
    ----------
    protocol:
        ``"mg2"`` reproduces MG2's core-fragment exclusion-by-atom-count: any
        non-core GT fragment that *happens* to share the core's atom count is
        dropped from the weighted Tanimoto denominator (this is a quirk of
        MG's code, not a deliberate choice — preserved here for replication).
        ``"ours"`` excludes only the core itself by *index*, so non-core
        fragments with the same atom count still contribute.
    """
    protocol = _normalize_protocol(protocol)
    predicted_fragments = _fragment_mols_and_indices(predicted_mol)
    gt_fragments = _fragment_mols_and_indices(gt_mol)

    scores: dict[str, Any] = {
        "backbone_core_tanimoto": 0.0,
        "backbone_core_tanimoto1": False,
        "backbone_core_inchi_equality": False,
        "backbone_fragments_tanimoto": [],
        "backbone_fragments_tanimoto1": [],
        "backbone_fragments_inchi_equality": [],
        "backbone_fragments_tanimoto_reduced": 0.0,
        "backbone_fragments_tanimoto1_reduced": True,
        "backbone_fragments_inchi_equality_reduced": True,
        "num_fragments_gt": len(gt_fragments),
        "num_fragments_pred": len(predicted_fragments),
        "num_fragments_equal": len(gt_fragments) == len(predicted_fragments),
    }
    if not gt_fragments:
        return scores, {}

    core_gt_fragment_index = max(
        enumerate(gt_fragments), key=lambda item: item[1][0].GetNumAtoms()
    )[0]
    core_backbone_size = gt_fragments[core_gt_fragment_index][0].GetNumAtoms()
    if protocol == "mg2":
        # MG2's quirk: exclude *any* fragment whose atom count equals the
        # core's, not just the core itself (line 1510-1514 of MG2's
        # utils_evaluation.py). Preserved bug-for-bug for replication.
        fragments_backbone_total_size = sum(
            frag.GetNumAtoms()
            for frag, _ in gt_fragments
            if frag.GetNumAtoms() != core_backbone_size
        )
    else:
        fragments_backbone_total_size = sum(
            frag.GetNumAtoms()
            for i, (frag, _) in enumerate(gt_fragments)
            if i != core_gt_fragment_index
        )

    fragments_mapping: dict[int, list[Any]] = defaultdict(list)
    fragment_indices_mapping: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    predicted_current = list(predicted_fragments)

    for i_gt, (gt_fragment, gt_fragment_indices) in enumerate(gt_fragments):
        selected_indices: list[int] = []
        if predicted_current:
            mcs_sizes: list[int] = []
            for pred_fragment, _ in predicted_current:
                try:
                    mcs = rdFMCS.FindMCS([pred_fragment, gt_fragment], timeout=5)
                    mcs_sizes.append(int(mcs.numAtoms))
                except Exception:
                    mcs_sizes.append(0)
            max_size = max(mcs_sizes) if mcs_sizes else 0
            selected_indices = [i for i, size in enumerate(mcs_sizes) if size == max_size]

            # Tie-break using R labels in this GT fragment, matching IBM's intent
            # for fragments that differ only by wildcard labels.
            if len(selected_indices) > 1:
                filtered: list[int] = []
                gt_labels_in_fragment = {
                    idx: label
                    for idx, label in gt_atom_labels.items()
                    if idx in gt_fragment_indices
                }
                for selected_index in selected_indices:
                    _, pred_indices = predicted_current[selected_index]
                    keep = True
                    for _, gt_label in gt_labels_in_fragment.items():
                        if not any(
                            pred_atom_labels.get(pred_idx, "").lower() == gt_label.lower()
                            for pred_idx in pred_indices
                        ):
                            keep = False
                            break
                    if keep:
                        filtered.append(selected_index)
                if filtered:
                    selected_indices = filtered

            if len(selected_indices) > 1:
                # Fallback: prefer the smallest tied fragment.
                min_len = min(len(predicted_current[i][1]) for i in selected_indices)
                selected_indices = [
                    i for i in selected_indices if len(predicted_current[i][1]) == min_len
                ]

        if selected_indices:
            selected_index = selected_indices[0]
            pred_fragment, pred_fragment_indices = predicted_current[selected_index]
            try:
                predicted_fragment_smiles = Chem.MolToSmiles(pred_fragment)
            except Exception:
                predicted_fragment_smiles = ""
        else:
            selected_index = -1
            pred_fragment = None
            pred_fragment_indices = ()
            predicted_fragment_smiles = ""

        try:
            gt_fragment_smiles = Chem.MolToSmiles(gt_fragment)
        except Exception:
            gt_fragment_smiles = ""

        pred_fragment_backbone = _replace_wildcards_for_fragment(
            predicted_fragment_smiles, remove_stereo=remove_stereo
        )
        gt_fragment_backbone = _replace_wildcards_for_fragment(
            gt_fragment_smiles, remove_stereo=remove_stereo
        )
        fragment_score = compute_molecule_prediction_quality(
            pred_fragment_backbone or "", gt_fragment_backbone or "", remove_stereo=remove_stereo
        )

        if i_gt == core_gt_fragment_index:
            scores["backbone_core_tanimoto"] = round(float(fragment_score["tanimoto"]), 3)
            scores["backbone_core_tanimoto1"] = bool(fragment_score["tanimoto1"])
            scores["backbone_core_inchi_equality"] = bool(fragment_score["inchi_equality"])
        else:
            scores["backbone_fragments_tanimoto"].append(
                round(float(fragment_score["tanimoto"]), 3)
            )
            scores["backbone_fragments_tanimoto1"].append(bool(fragment_score["tanimoto1"]))
            scores["backbone_fragments_inchi_equality"].append(
                bool(fragment_score["inchi_equality"])
            )

        for idx in selected_indices:
            fragments_mapping[i_gt].append(predicted_current[idx][0])
            fragment_indices_mapping[i_gt].append(predicted_current[idx][1])

        if len(selected_indices) == 1 and selected_index >= 0:
            predicted_current = [
                item for i, item in enumerate(predicted_current) if i != selected_index
            ]

    if scores["backbone_fragments_tanimoto"]:
        scores["backbone_fragments_tanimoto_reduced"] = round(
            mean(scores["backbone_fragments_tanimoto"]), 3
        )
    scores["backbone_fragments_tanimoto1_reduced"] = all(scores["backbone_fragments_tanimoto1"])
    scores["backbone_fragments_inchi_equality_reduced"] = all(
        scores["backbone_fragments_inchi_equality"]
    )

    denominator = fragments_backbone_total_size + core_backbone_size
    if denominator:
        scores["tanimoto"] = round(
            (
                scores["backbone_fragments_tanimoto_reduced"] * fragments_backbone_total_size
                + scores["backbone_core_tanimoto"] * core_backbone_size
            )
            / denominator,
            3,
        )
    else:
        scores["tanimoto"] = 0.0
    scores["tanimoto1"] = all(
        [scores["backbone_fragments_tanimoto1_reduced"], scores["backbone_core_tanimoto1"]]
    )
    scores["inchi_equality"] = all(
        [
            scores["backbone_fragments_inchi_equality_reduced"],
            scores["backbone_core_inchi_equality"],
        ]
    )

    # Create a global GT atom -> possible prediction atom mapping via MCS matches
    # within aligned fragments.
    gt_to_pred_indices_mapping: dict[int, list[int]] = defaultdict(list)
    for i_gt, (gt_fragment, gt_fragment_indices) in enumerate(gt_fragments):
        for pred_fragment, pred_fragment_indices in zip(
            fragments_mapping.get(i_gt, []), fragment_indices_mapping.get(i_gt, []), strict=False
        ):
            try:
                mcs = rdFMCS.FindMCS([pred_fragment, gt_fragment], timeout=5)
                if not mcs.smartsString:
                    continue
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                if mcs_mol is None:
                    continue
                gt_matches = gt_mol.GetSubstructMatches(mcs_mol, uniquify=False, maxMatches=1000)
                pred_matches = predicted_mol.GetSubstructMatches(
                    mcs_mol, uniquify=False, maxMatches=1000
                )
            except Exception:
                continue

            pred_matches = [
                m for m in pred_matches if all(idx in pred_fragment_indices for idx in m)
            ]
            gt_matches = [m for m in gt_matches if all(idx in gt_fragment_indices for idx in m)]
            for gt_match in gt_matches:
                for pred_match in pred_matches:
                    for pred_idx, gt_idx in zip(pred_match, gt_match, strict=False):
                        if pred_idx not in gt_to_pred_indices_mapping[gt_idx]:
                            gt_to_pred_indices_mapping[gt_idx].append(pred_idx)

    if verbose:
        print("GT->prediction atom mapping:", dict(gt_to_pred_indices_mapping))
    return scores, gt_to_pred_indices_mapping


# ---------------------------------------------------------------------------
# MarkushGrapher-inspired public scoring functions
# ---------------------------------------------------------------------------


def compute_markush_prediction_quality(
    predicted_smiles: str,
    gt_smiles: str,
    *,
    pred_atom_labels: dict[int, str] | None = None,
    gt_atom_labels: dict[int, str] | None = None,
    remove_stereo: bool = True,
    remove_double_bond_stereo: bool = True,  # Kept for API compatibility.
    verbose: bool = False,
    protocol: str = "ours",
) -> dict[str, Any]:
    """Evaluate one CXSMILES prediction using MCS alignment.

    Parameters mirror the MarkushGrapher-2 function name, but this adaptation
    accepts either local ``<r>``-tagged CXSMILES or already stripped CXSMILES.
    ``pred_atom_labels``/``gt_atom_labels`` should be supplied by
    :func:`parse_markush_text` when local ``<r>`` tags are used.

    Parameters
    ----------
    protocol:
        ``"ours"`` (default): our stricter eval — ``cxsmi_equality`` requires
        bijection for R/M/Sg counts; SGroup comparison uses all properties;
        unmapped GT M-ring atoms count as failures; whole-molecule backbone
        InChI is OR'd with the fragment-aligned one.

        ``"mg2"``: MarkushGrapher-2 replication — ``cxsmi_equality`` uses GT
        recall only (extra predicted R/M/Sg ignored); ``PrepareMolForDrawing``
        before InChI; SGroup compared by ``LABEL`` only; unmapped GT M-ring
        atoms silently skipped (vacuously OK); only fragment-aligned
        ``inchi_equality`` is used for backbone equality.

        ``"strict"`` is an accepted alias for ``"ours"``.
    """
    protocol = _normalize_protocol(protocol)
    _require_rdkit()

    pred_parsed = parse_markush_text(predicted_smiles, strict=False)
    gt_parsed = parse_markush_text(gt_smiles, strict=False)
    predicted_cx = pred_parsed.rdkit_cxsmiles or strip_r_tags(predicted_smiles)
    gt_cx = gt_parsed.rdkit_cxsmiles or strip_r_tags(gt_smiles)

    scores: dict[str, Any] = {
        "backbone_core_tanimoto": 0.0,
        "backbone_core_tanimoto1": False,
        "backbone_core_inchi_equality": False,
        "backbone_fragments_tanimoto": [],
        "backbone_fragments_tanimoto1": [],
        "backbone_fragments_inchi_equality": [],
        "backbone_fragments_tanimoto_reduced": 0.0,
        "backbone_fragments_tanimoto1_reduced": True,
        "backbone_fragments_inchi_equality_reduced": True,
        "tanimoto": 0.0,
        "tanimoto_similarity": 0.0,
        "tanimoto1": False,
        "inchi_equality": False,
        "backbone_inchi_match": False,
        "string_equality": predicted_smiles == gt_smiles,
        "valid": False,
        "r_labels": None,
        "m_sections": None,
        "sg_sections": None,
        "r": None,
        "r_group_accuracy": None,
        "m": None,
        "m_section_match": None,
        "sg": None,
        "r_full_match": False,
        "m_full_match": False,
        "sg_full_match": False,
        "num_fragments_gt": 0,
        "num_fragments_pred": 0,
        "num_fragments_equal": False,
        "cxsmi_equality": False,
        "cxsmiles_full_match": False,
    }

    predicted_mol = _mol_from_cxsmiles(predicted_cx)
    gt_mol = _mol_from_cxsmiles(gt_cx)
    if predicted_mol is None or gt_mol is None:
        return scores

    pred_labels = _rdkit_atom_labels(
        predicted_mol, {**pred_parsed.atom_labels, **(pred_atom_labels or {})}
    )
    gt_labels = _rdkit_atom_labels(gt_mol, {**gt_parsed.atom_labels, **(gt_atom_labels or {})})

    # Direct full-backbone InChI comparison after wildcard -> carbon and stereo removal.
    pred_inchi = _inchi_for_backbone(predicted_cx, remove_stereo=remove_stereo, protocol=protocol)
    gt_inchi = _inchi_for_backbone(gt_cx, remove_stereo=remove_stereo, protocol=protocol)
    scores["backbone_inchi_match"] = bool(pred_inchi and gt_inchi and pred_inchi == gt_inchi)
    scores["tanimoto_similarity"] = _fingerprint_tanimoto(predicted_cx, gt_cx)

    fragment_scores, gt_to_pred = _align_fragments_by_mcs(
        predicted_mol,
        gt_mol,
        pred_labels,
        gt_labels,
        remove_stereo=remove_stereo,
        verbose=verbose,
        protocol=protocol,
    )
    scores.update(fragment_scores)
    # Preserve the direct full-molecule soft metric under the requested name;
    # MarkushGrapher's weighted fragment score remains under "tanimoto".
    scores["tanimoto_similarity"] = scores["tanimoto_similarity"] or scores.get("tanimoto", 0.0)

    # R-group labels via MCS atom mapping.
    if not gt_labels:
        scores["r_labels"] = None
        scores["r"] = None
        scores["r_group_accuracy"] = None
        scores["r_full_match"] = len(pred_labels) == 0
    else:
        r_results: list[bool] = []
        mapping_for_r = copy.deepcopy(gt_to_pred)
        matched_pred_r_indices: set[int] = set()
        for gt_idx, gt_label in gt_labels.items():
            correct = False
            for pred_idx in list(mapping_for_r.get(gt_idx, [])):
                if pred_labels.get(pred_idx, "").lower() == gt_label.lower():
                    correct = True
                    matched_pred_r_indices.add(pred_idx)
                    # Consume the predicted atom index so duplicated symmetric
                    # mappings cannot satisfy multiple GT labels.
                    for key in list(mapping_for_r):
                        mapping_for_r[key] = [idx for idx in mapping_for_r[key] if idx != pred_idx]
                    break
            r_results.append(correct)
        scores["r_labels"] = r_results
        scores["r"] = round(mean(r_results), 3) if r_results else 0.0
        scores["r_group_accuracy"] = scores["r"]
        scores["r_full_match"] = all(r_results) and len(matched_pred_r_indices) == len(
            pred_labels
        ) == len(gt_labels)

    # Positional variation (M sections) through the same atom mapping.
    gt_m_sections = _parse_m_sections(gt_cx)
    pred_m_sections = _parse_m_sections(predicted_cx)
    if not gt_m_sections:
        scores["m_sections"] = None
        scores["m"] = None
        scores["m_section_match"] = None
        scores["m_full_match"] = len(pred_m_sections) == 0
    else:
        m_results: list[bool] = []
        mapping_for_m = copy.deepcopy(gt_to_pred)
        matched_pred_m_indices: set[int] = set()
        for gt_section in gt_m_sections:
            section_correct = False
            for pred_idx, pred_section in enumerate(pred_m_sections):
                if pred_idx in matched_pred_m_indices:
                    continue
                connector_ok = pred_section["atom_connector"] in mapping_for_m.get(
                    gt_section["atom_connector"], []
                )
                if protocol == "mg2":
                    # MG2 silently skips GT ring atoms with no mapping
                    # (line 1858-1859 of utils_evaluation.py: `if ring_atom
                    # not in gt_to_pred_indices_mapping_m: continue`). An
                    # m-section with NO mapped ring atoms passes the ring
                    # check vacuously.
                    _ring_hits: list[bool] = []
                    for gt_ring_atom in gt_section["ring_atoms"]:
                        if gt_ring_atom not in mapping_for_m:
                            continue
                        _ring_hits.append(
                            any(
                                pred_atom_idx in pred_section["ring_atoms"]
                                for pred_atom_idx in mapping_for_m[gt_ring_atom]
                            )
                        )
                    ring_ok = all(_ring_hits)
                else:
                    # ours: unmapped GT ring atom = failure (stricter, more
                    # semantically correct).
                    ring_ok = all(
                        any(
                            pred_atom_idx in pred_section["ring_atoms"]
                            for pred_atom_idx in mapping_for_m.get(gt_ring_atom, [])
                        )
                        for gt_ring_atom in gt_section["ring_atoms"]
                    )
                if connector_ok and ring_ok:
                    section_correct = True
                    matched_pred_m_indices.add(pred_idx)
                    connector = pred_section["atom_connector"]
                    for key in list(mapping_for_m):
                        mapping_for_m[key] = [idx for idx in mapping_for_m[key] if idx != connector]
                    break
            m_results.append(section_correct)
        scores["m_sections"] = m_results
        scores["m"] = round(mean(m_results), 3) if m_results else 0.0
        scores["m_section_match"] = scores["m"]
        scores["m_full_match"] = all(m_results) and len(matched_pred_m_indices) == len(
            pred_m_sections
        ) == len(gt_m_sections)

    # Frequency variation (Sg sections/substance groups).
    try:
        gt_sgroups = list(Chem.rdchem.GetMolSubstanceGroups(gt_mol))
        pred_sgroups = list(Chem.rdchem.GetMolSubstanceGroups(predicted_mol))
    except Exception:
        # If extraction fails, do not silently count SGroups as absent.
        gt_sgroups = [None]
        pred_sgroups = []
    if not gt_sgroups:
        scores["sg_sections"] = None
        scores["sg"] = None
        scores["sg_section_match"] = None
        scores["sg_full_match"] = len(pred_sgroups) == 0
    else:
        sg_results: list[bool] = []
        mapping_for_sg = copy.deepcopy(gt_to_pred)
        matched_pred_sg_indices: set[int] = set()
        for gt_sgroup in gt_sgroups:
            if gt_sgroup is None:
                sg_results.append(False)
                continue
            gt_atoms = list(gt_sgroup.GetAtoms())
            gt_signature = _sgroup_signature(gt_sgroup, protocol=protocol)
            section_correct = False
            if all(atom_idx in mapping_for_sg for atom_idx in gt_atoms):
                for pred_idx, pred_sgroup in enumerate(pred_sgroups):
                    if pred_idx in matched_pred_sg_indices:
                        continue
                    pred_atoms = set(pred_sgroup.GetAtoms())
                    pred_signature = _sgroup_signature(pred_sgroup, protocol=protocol)
                    if pred_signature != gt_signature or len(pred_atoms) != len(gt_atoms):
                        continue
                    gt_to_pred_ok = all(
                        any(
                            pred_atom in mapping_for_sg.get(gt_atom, []) for pred_atom in pred_atoms
                        )
                        for gt_atom in gt_atoms
                    )
                    pred_to_gt_ok = all(
                        any(pred_atom in mapping_for_sg.get(gt_atom, []) for gt_atom in gt_atoms)
                        for pred_atom in pred_atoms
                    )
                    if gt_to_pred_ok and pred_to_gt_ok:
                        section_correct = True
                        matched_pred_sg_indices.add(pred_idx)
                        for key in list(mapping_for_sg):
                            mapping_for_sg[key] = [
                                idx for idx in mapping_for_sg[key] if idx not in pred_atoms
                            ]
                        break
            sg_results.append(section_correct)
        scores["sg_sections"] = sg_results
        scores["sg"] = round(mean(sg_results), 3) if sg_results else 0.0
        scores["sg_section_match"] = scores["sg"]
        scores["sg_full_match"] = all(sg_results) and len(matched_pred_sg_indices) == len(
            pred_sgroups
        ) == len(gt_sgroups)

    # CXSMILES full match: backbone + exact feature counts/sets + fragment count.
    # Protocol determines whether we require bijection (ours) or GT recall only (mg2),
    # and whether backbone equality may fall back to whole-molecule InChI (ours)
    # or must come from the fragment-aligned InChI alone (mg2).
    if protocol == "mg2":
        r_ok = (scores.get("r") == 1.0) or (scores.get("r") is None)
        m_ok = (scores.get("m") == 1.0) or (scores.get("m") is None)
        sg_ok = (scores.get("sg") == 1.0) or (scores.get("sg") is None)
        # MG2 uses only the fragment-aligned `inchi_equality` (line 1693-1696,
        # 1961 of utils_evaluation.py). Whole-molecule `backbone_inchi_match`
        # is not part of MG's cxsmi_equality.
        backbone_ok = bool(scores["inchi_equality"])
    else:
        r_ok = bool(scores.get("r_full_match"))
        m_ok = bool(scores.get("m_full_match"))
        sg_ok = bool(scores.get("sg_full_match"))
        backbone_ok = bool(scores["inchi_equality"] or scores["backbone_inchi_match"])
    scores["cxsmi_equality"] = bool(
        r_ok and m_ok and sg_ok and backbone_ok and scores["num_fragments_equal"]
    )
    scores["cxsmiles_full_match"] = scores["cxsmi_equality"]
    scores["valid"] = True
    if protocol != "mg2" and not pred_atom_labels and not gt_atom_labels:
        _apply_position_graph_equality(
            scores,
            gt_smiles,
            predicted_smiles,
            remove_stereo=remove_stereo,
        )
    return scores


def _stable_normalized_text(value: str) -> str:
    return value.lower().replace("a ", "").replace(" group", "").replace(" ", "").strip()


def _mg2_normalize_substituent(value: str) -> str:
    """MG2's stable-text normalizer: replace ``"a "`` and ``" group"`` only.

    Mirrors lines 622-625 of MG2's ``utils_evaluation.get_stable_score``.
    Notably *not* lowercased and *not* whitespace-stripped here — MG applies
    lowercase + inner-space-strip only later, in the permissive comparison
    pass (lines 647-653).
    """
    if not isinstance(value, str):
        return value
    return value.replace("a ", "").replace(" group", "")


def _mg2_permissive_text(value: str) -> str:
    """MG2's permissive comparison form: lowercase + strip inner whitespace."""
    if not isinstance(value, str):
        return value
    return value.lower().replace(" ", "")


def get_stable_score(
    gt_stable: dict[str, list[str]] | None,
    predicted_stable: dict[str, list[str]] | None,
    permissive: bool = True,
    verbose: bool = False,
    normalize: bool = True,
    *,
    protocol: str = "ours",
) -> dict[str, Any]:
    """Compare substituent tables (stable sections).

    This is an adapted, dependency-free version of MarkushGrapher's
    ``get_stable_score``.  It reports exact table equality plus per-label macro
    recall/precision over alternatives.

    Parameters
    ----------
    protocol:
        ``"ours"`` (default): our stricter eval — ``stable_equality`` requires
        that the prediction has no hallucinated labels beyond GT, and
        normalization is more aggressive (lowercases and strips inner
        whitespace inside the normalizer itself).

        ``"mg2"``: MG2 replication — ``stable_equality`` only checks GT recall
        (extra predicted labels are silently allowed, matching line 697-698
        of MG's ``utils_evaluation.py``); normalization replaces ``"a "`` /
        ``" group"`` only, with lowercase + space-strip happening only in the
        permissive comparison pass.
    """
    protocol = _normalize_protocol(protocol)
    gt_stable = gt_stable or {}
    predicted_stable = predicted_stable or {}
    scores: dict[str, Any] = {
        "stable_equality": False,
        "stable_recall": 0.0,
        "stable_precision": 0.0,
    }

    if verbose:
        print("gt_stable", gt_stable)
        print("predicted_stable", predicted_stable)

    if gt_stable == {}:
        if predicted_stable == {}:
            scores.update({"stable_equality": True, "stable_recall": 1.0, "stable_precision": 1.0})
        return scores

    norm_fn = _mg2_normalize_substituent if protocol == "mg2" else _stable_normalized_text
    permissive_fn = _mg2_permissive_text if protocol == "mg2" else _stable_normalized_text

    if normalize:
        remapped_prediction: dict[str, list[str]] = {}
        for label, predicted_substituents in predicted_stable.items():
            if label not in gt_stable:
                remapped_prediction[label] = predicted_substituents
                continue
            normalized_gt = [norm_fn(s) for s in gt_stable[label]]
            remapped: list[str] = []
            for substituent in predicted_substituents:
                if substituent in gt_stable[label]:
                    remapped.append(substituent)
                    continue
                normalized_pred = norm_fn(substituent)
                if normalized_pred in normalized_gt:
                    remapped.append(gt_stable[label][normalized_gt.index(normalized_pred)])
                else:
                    remapped.append(substituent)
            remapped_prediction[label] = remapped
        predicted_stable = remapped_prediction

    if permissive:
        gt_cmp = {k.lower(): [permissive_fn(v) for v in values] for k, values in gt_stable.items()}
        pred_cmp = {
            k.lower(): [permissive_fn(v) for v in values] for k, values in predicted_stable.items()
        }
    else:
        gt_cmp = gt_stable
        pred_cmp = predicted_stable

    perfect_match: list[bool] = []
    recall_rows: list[float] = []
    extra_predicted_labels = set(pred_cmp) - set(gt_cmp)
    for label, gt_values in gt_cmp.items():
        pred_values = pred_cmp.get(label)
        if pred_values is None:
            perfect_match.append(False)
            recall_rows.append(0.0 if gt_values else 1.0)
            continue
        perfect_match.append(set(gt_values) == set(pred_values))
        recall_rows.append(
            1.0
            if not gt_values
            else sum(value in pred_values for value in gt_values) / len(gt_values)
        )

    precision_rows: list[float] = []
    for label, pred_values in pred_cmp.items():
        if not pred_values:
            continue
        gt_values = gt_cmp.get(label)
        if gt_values is None:
            precision_rows.append(0.0)
        else:
            precision_rows.append(
                sum(value in gt_values for value in pred_values) / len(pred_values)
            )

    if protocol == "mg2":
        # MG2: extra predicted labels are silently ignored
        # (line 697-698 of utils_evaluation.py).
        scores["stable_equality"] = bool(perfect_match) and all(perfect_match)
    else:
        scores["stable_equality"] = all(perfect_match) and not extra_predicted_labels
    scores["stable_recall"] = round(mean(recall_rows), 3) if recall_rows else 0.0
    scores["stable_precision"] = round(mean(precision_rows), 3) if precision_rows else 0.0
    return scores


def get_scores(
    gt_smiles_list: list[str],
    predicted_smiles_list: list[str],
    training_smiles: set[str] | list[str] | tuple[str, ...] | None = None,
    gt_stable_list: list[dict[str, list[str]]] | None = None,
    predicted_stable_list: list[dict[str, list[str]]] | None = None,
    cxsmiles: bool = True,
    markush: bool = False,
    get_unreduced_scores: bool = True,
    verbose: bool = False,
    remove_stereo: bool = True,
    protocol: str = "ours",
):
    """Batch evaluator with MarkushGrapher-compatible naming.

    ``cxsmiles`` is kept for signature compatibility; this implementation is
    intended for CXSMILES/Markush strings and always uses MCS Markush scoring.

    Parameters
    ----------
    protocol:
        ``"ours"`` (default) — our stricter eval.
        ``"mg2"`` — MarkushGrapher-2 replication.
        ``"strict"`` is an accepted alias for ``"ours"``.
    """
    protocol = _normalize_protocol(protocol)
    _require_rdkit()
    if len(gt_smiles_list) != len(predicted_smiles_list):
        raise ValueError(
            "gt_smiles_list and predicted_smiles_list must have the same length; "
            f"got {len(gt_smiles_list)} and {len(predicted_smiles_list)}"
        )
    training_smiles = set(training_smiles or [])
    scores: dict[int, dict[str, Any] | None] = {}

    for idx, (gt_smiles, predicted_smiles) in enumerate(
        zip(gt_smiles_list, predicted_smiles_list, strict=False)
    ):
        gt_parsed = parse_markush_text(gt_smiles, strict=False)
        pred_parsed = parse_markush_text(predicted_smiles, strict=False)
        default_incorrect = {
            "tanimoto": 0.0,
            "tanimoto_similarity": 0.0,
            "valid": False,
            "inchi_equality": False,
            "backbone_inchi_match": False,
            "string_equality": pred_parsed.cxsmiles == gt_parsed.cxsmiles,
            "r": None,
            "r_group_accuracy": None,
            "m": None,
            "m_section_match": None,
            "sg": None,
            "sg_section_match": None,
            "num_fragments_gt": 0,
            "num_fragments_pred": 0,
            "num_fragments_equal": False,
            "cxsmi_equality": False,
            "cxsmiles_full_match": False,
            "markush_equality": False,
            "is_in_training": int(predicted_smiles in training_smiles) if predicted_smiles else 0,
        }
        if not gt_parsed.valid:
            scores[idx] = None
            continue
        if not pred_parsed.valid:
            scores[idx] = default_incorrect
            continue
        try:
            sample_score = compute_markush_prediction_quality(
                pred_parsed.rdkit_cxsmiles,
                gt_parsed.rdkit_cxsmiles,
                pred_atom_labels=pred_parsed.atom_labels,
                gt_atom_labels=gt_parsed.atom_labels,
                remove_stereo=remove_stereo,
                verbose=verbose,
                protocol=protocol,
            )
        except Exception as exc:
            if verbose:
                print(
                    f"Error in compute_markush_prediction_quality for index {idx}: {exc}",
                    file=sys.stderr,
                )
            sample_score = default_incorrect
        sample_score["is_in_training"] = (
            int(predicted_smiles in training_smiles) if predicted_smiles else 0
        )
        sample_score["string_equality"] = pred_parsed.cxsmiles == gt_parsed.cxsmiles
        if protocol != "mg2":
            _apply_position_graph_equality(
                sample_score,
                gt_parsed.cxsmiles,
                pred_parsed.cxsmiles,
                remove_stereo=remove_stereo,
            )
        scores[idx] = sample_score

    if markush:
        for idx in range(len(gt_smiles_list)):
            if scores.get(idx) is None:
                continue
            gt_stable = (
                gt_stable_list[idx] if gt_stable_list and idx < len(gt_stable_list) else None
            )
            pred_stable = (
                predicted_stable_list[idx]
                if predicted_stable_list and idx < len(predicted_stable_list)
                else None
            )
            stable_scores = get_stable_score(gt_stable, pred_stable, protocol=protocol)
            scores[idx].update(stable_scores)  # type: ignore[union-attr]
            scores[idx]["markush_equality"] = bool(
                scores[idx].get("cxsmi_equality") and stable_scores["stable_equality"]
            )  # type: ignore[union-attr]

    def _values(key: str, *, exclude_none: bool = False) -> list[float]:
        values: list[float] = []
        for score in scores.values():
            if score is None:
                continue
            value = score.get(key)
            if value is None and exclude_none:
                continue
            if value is None:
                value = 0.0
            values.append(float(value))
        return values

    reduced_scores: dict[str, Any] = {
        "size": len(scores),
        "invalid_gt": sum(score is None for score in scores.values()) / len(scores)
        if scores
        else 0.0,
        "valid": round(mean(_values("valid")), 3) if _values("valid") else 0.0,
        "tanimoto": round(mean(_values("tanimoto")), 3) if _values("tanimoto") else 0.0,
        "tanimoto_similarity": round(mean(_values("tanimoto_similarity")), 3)
        if _values("tanimoto_similarity")
        else 0.0,
        "inchi_equality": round(mean(_values("inchi_equality")), 3)
        if _values("inchi_equality")
        else 0.0,
        "backbone_inchi_match": round(mean(_values("backbone_inchi_match")), 3)
        if _values("backbone_inchi_match")
        else 0.0,
        "string_equality": round(mean(_values("string_equality")), 3)
        if _values("string_equality")
        else 0.0,
        "cxsmi_equality": round(mean(_values("cxsmi_equality")), 3)
        if _values("cxsmi_equality")
        else 0.0,
        "num_fragments_equal": round(mean(_values("num_fragments_equal")), 3)
        if _values("num_fragments_equal")
        else 0.0,
        "r": round(mean(_values("r", exclude_none=True)), 3)
        if _values("r", exclude_none=True)
        else None,
        "m": round(mean(_values("m", exclude_none=True)), 3)
        if _values("m", exclude_none=True)
        else None,
        "sg": round(mean(_values("sg", exclude_none=True)), 3)
        if _values("sg", exclude_none=True)
        else None,
        "r_size": len(_values("r", exclude_none=True)),
        "m_size": len(_values("m", exclude_none=True)),
        "sg_size": len(_values("sg", exclude_none=True)),
    }
    if markush:
        reduced_scores.update(
            {
                "stable_equality": round(mean(_values("stable_equality")), 3)
                if _values("stable_equality")
                else 0.0,
                "stable_recall": round(mean(_values("stable_recall")), 3)
                if _values("stable_recall")
                else 0.0,
                "stable_precision": round(mean(_values("stable_precision")), 3)
                if _values("stable_precision")
                else 0.0,
                "markush_equality": round(mean(_values("markush_equality")), 3)
                if _values("markush_equality")
                else 0.0,
            }
        )
    return (reduced_scores, scores) if get_unreduced_scores else reduced_scores


# ---------------------------------------------------------------------------
# High-level local API and CLI formatting
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _bool_mean(values: list[Any]) -> float | None:
    return mean(float(bool(v)) for v in values) if values else None


def _aggregate_sample_results(results: list[SampleEvaluation]) -> dict[str, Any]:
    n = len(results)
    valid_results = [r for r in results if r.valid_pred and r.valid_gt and r.scores.get("valid")]
    scores = [r.scores for r in results]

    r_values = [s["r_group_accuracy"] for s in scores if s.get("r_group_accuracy") is not None]
    m_values = [s["m_section_match"] for s in scores if s.get("m_section_match") is not None]
    sg_values = [s["sg_section_match"] for s in scores if s.get("sg_section_match") is not None]
    stable_values = [s["stable_equality"] for s in scores if s.get("gt_has_stable")]

    metrics: dict[str, Any] = {
        "n_samples": n,
        "n_valid": len(valid_results),
        "valid_rate": len(valid_results) / n if n else 0.0,
        "string_exact_match": _bool_mean([s.get("string_equality") for s in scores]) or 0.0,
        "string_equality": _bool_mean([s.get("string_equality") for s in scores]) or 0.0,
        "backbone_inchi_match": _bool_mean([s.get("backbone_inchi_match") for s in scores]) or 0.0,
        "backbone_inchi_match_n": sum(1 for s in scores if "backbone_inchi_match" in s),
        "r_group_accuracy": _safe_mean([float(v) for v in r_values]),
        "r_group_accuracy_n": len(r_values),
        "m_section_match": _safe_mean([float(v) for v in m_values]),
        "m_section_match_n": len(m_values),
        "sg_section_match": _safe_mean([float(v) for v in sg_values]),
        "sg_section_match_n": len(sg_values),
        "stable_match": _bool_mean(stable_values),
        "stable_match_n": len(stable_values),
        "stable_equality_all": _bool_mean([s.get("stable_equality") for s in scores]) or 0.0,
        "cxsmiles_full_match": _bool_mean([s.get("cxsmiles_full_match") for s in scores]) or 0.0,
        "cxsmi_equality": _bool_mean([s.get("cxsmi_equality") for s in scores]) or 0.0,
        "markush_full_match": _bool_mean([s.get("markush_full_match") for s in scores]) or 0.0,
        "markush_equality": _bool_mean([s.get("markush_full_match") for s in scores]) or 0.0,
        "tanimoto_similarity": _safe_mean(
            [float(s.get("tanimoto_similarity", 0.0)) for s in scores]
        )
        or 0.0,
    }
    # Short aliases matching MarkushGrapher's reduced keys.
    metrics["r"] = metrics["r_group_accuracy"]
    metrics["m"] = metrics["m_section_match"]
    metrics["sg"] = metrics["sg_section_match"]
    return metrics


def evaluate_markush_mcs(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    *,
    include_missing: bool = True,
    return_details: bool = False,
    verbose: bool = False,
    protocol: str = "ours",
) -> dict[str, Any] | tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate local Markush predictions with MCS-aligned metrics.

    Parameters
    ----------
    predictions:
        JSONL-style rows with ``id`` and ``prediction`` (or ``cxsmiles_opt``).
    ground_truths:
        JSONL-style rows with ``id`` and ``cxsmiles_opt`` and/or ``annotation``.
    include_missing:
        Count GT rows with no prediction as invalid zero-score samples.
    return_details:
        If true, return ``(metrics, per_sample_dicts)``.  The default matches
        the requested importable API and returns only aggregate metrics.
    protocol:
        ``"ours"`` (default) — our stricter eval.
        ``"mg2"`` — MarkushGrapher-2 replication.
        ``"strict"`` is an accepted alias for ``"ours"``.

    See also
    --------
    :func:`evaluate_markush_both_protocols` runs the same inputs through both
    ``"ours"`` and ``"mg2"`` and returns a nested dict — useful for tracking
    both numbers during training and offline eval.
    """
    protocol = _normalize_protocol(protocol)
    _require_rdkit()

    pred_by_id: dict[Any, dict[str, Any]] = {}
    for entry in predictions:
        sid = entry.get("id")
        if sid is None:
            warnings.warn(f"Prediction row missing id: {entry}", stacklevel=2)
            continue
        pred_by_id[sid] = entry

    gt_by_id: dict[Any, dict[str, Any]] = {}
    for entry in ground_truths:
        sid = entry.get("id")
        if sid is None:
            warnings.warn(f"Ground-truth row missing id: {entry}", stacklevel=2)
            continue
        gt_by_id[sid] = entry

    ids: list[Any]
    if include_missing:
        ids = list(gt_by_id.keys())
    else:
        ids = [sid for sid in gt_by_id if sid in pred_by_id]

    extra_predictions = [sid for sid in pred_by_id if sid not in gt_by_id]
    if extra_predictions:
        warnings.warn(
            f"{len(extra_predictions)} prediction(s) have no ground-truth row and were ignored. "
            f"First 5: {extra_predictions[:5]}",
            stacklevel=2,
        )

    results: list[SampleEvaluation] = []
    for sid in ids:
        gt_entry = gt_by_id[sid]
        pred_entry = pred_by_id.get(sid)
        gt = parse_markush_text(_ground_truth_text_from_entry(gt_entry), strict=False)
        if pred_entry is None:
            results.append(
                SampleEvaluation(
                    sample_id=str(sid),
                    valid_pred=False,
                    valid_gt=gt.valid,
                    prediction_error="no prediction provided",
                    ground_truth_error=gt.parse_error,
                    gt_cxsmiles=gt.cxsmiles,
                    scores={
                        "string_equality": False,
                        "backbone_inchi_match": False,
                        "cxsmi_equality": False,
                        "cxsmiles_full_match": False,
                        "stable_equality": False,
                        "gt_has_stable": bool(gt.stable),
                        "markush_full_match": False,
                        "tanimoto_similarity": 0.0,
                    },
                )
            )
            continue

        pred = parse_markush_text(_prediction_text_from_entry(pred_entry), strict=False)
        sample = SampleEvaluation(
            sample_id=str(sid),
            valid_pred=pred.valid,
            valid_gt=gt.valid,
            prediction_error=pred.parse_error,
            ground_truth_error=gt.parse_error,
            pred_cxsmiles=pred.cxsmiles,
            gt_cxsmiles=gt.cxsmiles,
        )

        if not pred.valid or not gt.valid:
            sample.scores = {
                "string_equality": False,
                "backbone_inchi_match": False,
                "cxsmi_equality": False,
                "cxsmiles_full_match": False,
                "stable_equality": False,
                "gt_has_stable": bool(gt.stable),
                "markush_full_match": False,
                "tanimoto_similarity": 0.0,
            }
            results.append(sample)
            continue

        try:
            mcs_scores = compute_markush_prediction_quality(
                pred.rdkit_cxsmiles,
                gt.rdkit_cxsmiles,
                pred_atom_labels=pred.atom_labels,
                gt_atom_labels=gt.atom_labels,
                verbose=verbose,
                protocol=protocol,
            )
        except Exception as exc:
            sample.prediction_error = f"RDKit/MCS scoring failed: {exc}"
            mcs_scores = {
                "valid": False,
                "string_equality": pred.cxsmiles == gt.cxsmiles,
                "backbone_inchi_match": False,
                "r_group_accuracy": None,
                "m_section_match": None,
                "sg_section_match": None,
                "cxsmi_equality": False,
                "cxsmiles_full_match": False,
                "tanimoto_similarity": 0.0,
            }

        stable_scores = get_stable_score(gt.stable, pred.stable, protocol=protocol)
        if protocol != "mg2":
            _apply_position_graph_equality(
                mcs_scores,
                gt.cxsmiles,
                pred.cxsmiles,
                remove_stereo=True,
            )
        mcs_scores.update(stable_scores)
        mcs_scores["gt_has_stable"] = bool(gt.stable)
        mcs_scores["markush_full_match"] = bool(
            mcs_scores.get("cxsmiles_full_match") and stable_scores["stable_equality"]
        )
        # Local string exact should compare original clean CXSMILES, not the stripped
        # wildcard strings passed to RDKit.
        mcs_scores["string_equality"] = pred.cxsmiles == gt.cxsmiles
        sample.scores = mcs_scores
        results.append(sample)

    metrics = _aggregate_sample_results(results)
    details = [
        {
            "id": result.sample_id,
            "valid_pred": result.valid_pred,
            "valid_gt": result.valid_gt,
            "prediction_error": result.prediction_error,
            "ground_truth_error": result.ground_truth_error,
            "pred_cxsmiles": result.pred_cxsmiles,
            "gt_cxsmiles": result.gt_cxsmiles,
            **result.scores,
        }
        for result in results
    ]
    return (metrics, details) if return_details else metrics


def evaluate_markush_both_protocols(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    *,
    include_missing: bool = True,
    return_details: bool = False,
    verbose: bool = False,
) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Evaluate the same predictions under both ``"ours"`` and ``"mg2"`` protocols.

    Convenience for callers (training callbacks, offline eval CLI) that want
    to record both numbers side-by-side.  Internally calls
    :func:`evaluate_markush_mcs` twice; the cost is dominated by RDKit/MCS
    work that is mostly identical across protocols, so wall-clock is roughly
    1.4-2x a single run rather than 2x.

    Returns
    -------
    metrics : dict
        ``{"ours": metrics_ours_dict, "mg2": metrics_mg2_dict}``.
    details : dict (only if ``return_details``)
        ``{"ours": per_sample_list, "mg2": per_sample_list}``.
    """
    out_metrics: dict[str, dict[str, Any]] = {}
    out_details: dict[str, list[dict[str, Any]]] = {}
    for protocol in ("ours", "mg2"):
        if return_details:
            metrics, details = evaluate_markush_mcs(  # type: ignore[misc]
                predictions,
                ground_truths,
                include_missing=include_missing,
                return_details=True,
                verbose=verbose,
                protocol=protocol,
            )
            out_metrics[protocol] = metrics
            out_details[protocol] = details
        else:
            metrics = evaluate_markush_mcs(  # type: ignore[misc]
                predictions,
                ground_truths,
                include_missing=include_missing,
                return_details=False,
                verbose=verbose,
                protocol=protocol,
            )
            out_metrics[protocol] = metrics  # type: ignore[assignment]
    if return_details:
        return out_metrics, out_details
    return out_metrics


def flatten_both_protocol_metrics(
    both_metrics: dict[str, dict[str, Any]],
    *,
    prefix: str = "eval",
) -> dict[str, Any]:
    """Flatten a ``{"ours": {...}, "mg2": {...}}`` dict into namespaced keys.

    Yields keys like ``eval/ours/cxsmi_equality`` and ``eval/mg2/cxsmi_equality``.
    Suited for W&B / Trainer log dicts which expect flat ``{str: scalar}`` maps.
    Non-scalar (None, dict, list) values are passed through; numeric values
    are kept as-is for downstream consumers to format.
    """
    flat: dict[str, Any] = {}
    for protocol, metrics in both_metrics.items():
        for key, value in metrics.items():
            flat[f"{prefix}/{protocol}/{key}"] = value
    return flat


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                warnings.warn(f"{path}:{line_no}: invalid JSON skipped: {exc}", stacklevel=2)
    return rows


def _format_percent(value: float | None, *, none_text: str = "N/A") -> str:
    if value is None:
        return none_text
    return f"{value * 100:5.1f}%"


def format_metrics_summary(metrics: dict[str, Any], *, label: str = "Markush MCS") -> str:
    n = metrics.get("n_samples", 0)
    lines = [f"=== {label} ({n} samples) ==="]
    lines.append(f"Backbone InChI match:  {_format_percent(metrics.get('backbone_inchi_match'))}")
    lines.append(
        f"R-group accuracy:      {_format_percent(metrics.get('r_group_accuracy'), none_text=f'N/A (0/{n} have R-groups)')}"
    )
    lines.append(
        f"M-section match:       {_format_percent(metrics.get('m_section_match'), none_text=f'N/A (0/{n} have M sections)')}"
    )
    lines.append(
        f"Sg-section match:      {_format_percent(metrics.get('sg_section_match'), none_text=f'N/A (0/{n} have Sg sections)')}"
    )
    stable_n = metrics.get("stable_match_n", 0)
    lines.append(
        f"Stable match:          {_format_percent(metrics.get('stable_match'), none_text=f'N/A ({stable_n}/{n} have stable)')}"
    )
    lines.append(f"CXSMILES full match:   {_format_percent(metrics.get('cxsmiles_full_match'))}")
    lines.append(f"Markush full match:    {_format_percent(metrics.get('markush_full_match'))}")
    lines.append(f"Tanimoto similarity:   {metrics.get('tanimoto_similarity', 0.0):.3f}")
    lines.append(f"String exact match:    {_format_percent(metrics.get('string_exact_match'))}")
    lines.append(
        f"Valid predictions:     {metrics.get('n_valid', 0)}/{n} ({metrics.get('valid_rate', 0.0) * 100:.1f}%)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCS-based Markush/CXSMILES evaluator.")
    parser.add_argument(
        "--predictions", required=True, help="JSONL with id and prediction/cxsmiles_opt fields"
    )
    parser.add_argument(
        "--ground-truth", required=True, help="JSONL with id and cxsmiles_opt/annotation fields"
    )
    parser.add_argument("--output", help="Write metrics and per-sample details to this JSON file")
    parser.add_argument(
        "--label", default="Markush MCS", help="Label used in the printed summary header"
    )
    parser.add_argument(
        "--no-include-missing",
        dest="include_missing",
        action="store_false",
        help="Only score IDs present in both files",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose MCS mapping diagnostics"
    )
    parser.add_argument(
        "--protocol",
        choices=("ours", "mg2", "both"),
        default="ours",
        help=(
            "Evaluation protocol. 'ours' is our stricter eval (default); 'mg2' is "
            "an APPROXIMATE in-package replication of MarkushGrapher-2's "
            "utils_evaluation.py (not byte-exact — official/headline numbers "
            "require the pinned DS4SD scorer in tools/mg2_scorer/); "
            "'both' runs both and prints/saves each."
        ),
    )
    args = parser.parse_args(argv)

    try:
        _require_rdkit()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    predictions = load_jsonl(args.predictions)
    ground_truths = load_jsonl(args.ground_truth)

    if args.protocol == "both":
        both_metrics, both_details = evaluate_markush_both_protocols(
            predictions,
            ground_truths,
            include_missing=args.include_missing,
            return_details=True,
            verbose=args.verbose,
        )
        for proto, metrics in both_metrics.items():
            print(format_metrics_summary(metrics, label=f"{args.label} [{proto}]"))
            print()
        if args.output:
            payload = {
                "metrics": both_metrics,
                "per_sample": both_details,
                "protocol": "both",
            }
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            print(f"Saved metrics (both protocols) to {args.output}")
        return 0

    metrics, details = evaluate_markush_mcs(
        predictions,
        ground_truths,
        include_missing=args.include_missing,
        return_details=True,
        verbose=args.verbose,
        protocol=args.protocol,
    )
    print(format_metrics_summary(metrics, label=f"{args.label} [{args.protocol}]"))

    if args.output:
        payload = {"metrics": metrics, "per_sample": details, "protocol": args.protocol}
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print(f"\nSaved metrics to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
