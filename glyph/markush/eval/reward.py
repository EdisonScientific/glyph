#!/usr/bin/env python3
"""Component-wise reward function for Markush CXSMILES generation.

The default reward is strict-graph first: exact position-aware Markush graph
isomorphism is the dominant terminal signal, while component-level shaping
keeps partial generations learnable.  The legacy MG2/MCS scores remain
available as opt-in bonuses but default to zero because they are recall-style
and can miss hallucinated optional sections such as extra Sg groups.

Usage:
    from glyph.markush.eval.reward import compute_reward
    reward = compute_reward(prediction, ground_truth)
"""

import contextlib
import re
from typing import Any

from glyph.markush.eval import common
from glyph.markush.eval.markush_graph import MarkushEqualityResult, markush_graph_equal

# Try RDKit for molecular validation
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.inchi import MolToInchi

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_markush(text: str) -> dict[str, Any]:
    """Parse a markush prediction into components."""
    result = {
        "valid_xml": False,
        "cxsmiles": "",
        "stable": "",
        "backbone": "",
        "extensions": "",
        "r_labels": set(),
        "sg_sections": [],
        "has_sg": False,
        "has_m": False,
    }

    if not text or not isinstance(text, str):
        return result

    # Use the canonical, case-insensitive extractors from glyph.markush.eval.common
    # so this module can't drift from metrics_mcs / predict / metrics_string.
    cxsmi_match = common.RE_CXSMI.search(text)
    stable_match = common.RE_STABLE.search(text)

    if cxsmi_match:
        result["valid_xml"] = True
        result["cxsmiles"] = cxsmi_match.group(1).strip()

    if stable_match:
        result["stable"] = stable_match.group(1).strip()

    if result["cxsmiles"]:
        # Split backbone from extensions (preserve `parse_markush`'s historical
        # behavior of NOT stripping whitespace around the split parts; the
        # downstream Sg/m matchers assume the raw extension payload).
        parts = result["cxsmiles"].split("|", 1)
        result["backbone"] = parts[0]
        result["extensions"] = parts[1] if len(parts) > 1 else ""

        # Extract R-group labels (case-insensitive via common.RE_R_TAG).
        result["r_labels"] = set(common.extract_r_labels(result["cxsmiles"]))

        # Check for Sg and m sections. `Sg:` is case-sensitive in the CXSMILES
        # spec and `m:` must NOT match e.g. `Sg:m:...` continuations — anchor
        # both at the start of the extensions or after a `,`.
        result["sg_sections"] = _parse_sg_sections(result["extensions"])
        result["has_sg"] = bool(result["sg_sections"])
        result["has_m"] = bool(re.search(r"(?:^|,)m:", result["extensions"]))

    return result


def reward_valid_parse(pred: dict) -> float:
    """R1: +1.0 if valid markush XML with parseable CXSMILES."""
    if not pred["valid_xml"]:
        return 0.0
    if not pred["cxsmiles"]:
        return 0.0
    if RDKIT_AVAILABLE:
        # Try to parse the backbone SMILES (case-insensitive <r> stripping
        # via common.strip_r_tags so reward and metrics agree).
        backbone = common.strip_r_tags(pred["backbone"])
        try:
            params = Chem.SmilesParserParams()
            params.sanitize = False
            mol = Chem.MolFromSmiles(backbone, params)
            return 1.0 if mol is not None else 0.5
        except Exception:
            return 0.5  # Valid XML but unparseable SMILES
    return 0.8  # Valid XML, can't check SMILES


def reward_backbone_match(pred: dict, gt: dict) -> float:
    """R2: +1.0 if backbone InChI matches after wildcard→C + stereo removal."""
    if (
        pred.get("backbone", "").strip()
        and pred.get("backbone", "").strip() == gt.get("backbone", "").strip()
    ):
        return 1.0
    if not RDKIT_AVAILABLE or not pred["backbone"] or not gt["backbone"]:
        return 0.0

    def to_inchi(backbone: str) -> str | None:
        mol = _backbone_to_mol(backbone)
        if mol is None:
            return None
        try:
            return MolToInchi(mol)
        except Exception:
            return None

    pred_inchi = to_inchi(pred["backbone"])
    gt_inchi = to_inchi(gt["backbone"])

    if pred_inchi and gt_inchi:
        return 1.0 if pred_inchi == gt_inchi else 0.0
    return 0.0


def _backbone_to_mol(backbone: str):
    if not RDKIT_AVAILABLE or not backbone:
        return None
    smi = common.RE_R_TAG.sub("C", backbone).replace("*", "C")
    try:
        params = Chem.SmilesParserParams()
        params.sanitize = False
        mol = Chem.MolFromSmiles(smi, params)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        with contextlib.suppress(Exception):
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def reward_backbone_tanimoto(pred: dict, gt: dict) -> float:
    """Continuous RDKit fingerprint similarity for backbone-only shaping."""
    if (
        pred.get("backbone", "").strip()
        and pred.get("backbone", "").strip() == gt.get("backbone", "").strip()
    ):
        return 1.0
    if not RDKIT_AVAILABLE or not pred["backbone"] or not gt["backbone"]:
        return 0.0

    pred_mol = _backbone_to_mol(pred["backbone"])
    gt_mol = _backbone_to_mol(gt["backbone"])
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


def reward_r_groups(pred: dict, gt: dict) -> float:
    """R3: R-group label accuracy (F1 between predicted and GT labels)."""
    if not gt["r_labels"]:
        return 1.0 if not pred["r_labels"] else 0.0
    if not pred["r_labels"]:
        return 0.0

    tp = len(pred["r_labels"] & gt["r_labels"])
    precision = tp / len(pred["r_labels"]) if pred["r_labels"] else 0
    recall = tp / len(gt["r_labels"]) if gt["r_labels"] else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _parse_sg_sections(extensions: str) -> list[str]:
    """Parse all Sg sections from CXSMILES extensions.

    Walks through the extension string section-by-section.  When an ``Sg:``
    marker is seen at the start (or after a ``,``), its content is captured
    until the next CXSMILES section-start marker (any entry of
    ``_SECTION_MARKERS``) or end-of-string.  This avoids the previous
    greedy-split bug where ``Sg:n:1,2:ht,m:1:2.3`` was treated as a single
    Sg section that swallowed the unrelated ``m:`` section.

    Returns the list of Sg payloads (without the ``Sg:`` prefix), preserving
    order of appearance.  Returns ``[]`` for empty input or when no ``Sg:``
    section is present.
    """
    return common.parse_sg_sections(extensions)


def reward_sg_section(pred: dict, gt: dict) -> float:
    """R4: Sg-section accuracy."""
    gt_sg_sections = list(gt.get("sg_sections") or _parse_sg_sections(gt.get("extensions", "")))
    pred_sg_sections = list(
        pred.get("sg_sections") or _parse_sg_sections(pred.get("extensions", ""))
    )
    gt_has_sg = bool(gt_sg_sections)
    pred_has_sg = bool(pred_sg_sections)

    if not gt_has_sg:
        return 1.0 if not pred_has_sg else 0.0
    if not pred_has_sg:
        return 0.0

    # Simple overlap score on individual sections
    gt_set = set(gt_sg_sections)
    pred_set = set(pred_sg_sections)
    if gt_set == pred_set:
        return 1.0
    overlap = len(gt_set & pred_set) / max(len(gt_set | pred_set), 1)
    return overlap


def reward_extra_sg(pred: dict, gt: dict) -> float:
    """Return 1.0 when the prediction hallucinates Sg but GT has none."""

    if not gt.get("valid_xml") or not gt.get("cxsmiles"):
        return 0.0
    return 1.0 if pred.get("has_sg") and not gt.get("has_sg") else 0.0


def _extract_m_section(extensions: str) -> str | None:
    """Extract the m-section content from CXSMILES extensions."""
    return common.extract_m_section(extensions)


def reward_m_section(pred: dict, gt: dict) -> float:
    """R5: m-section (positional variation) accuracy."""
    gt_has_m = bool(re.search(r"(?:^|,)m:", gt.get("extensions", "")))
    pred_has_m = pred["has_m"]

    if not gt_has_m:
        return 1.0 if not pred_has_m else 0.5
    if not pred_has_m:
        return 0.0

    # Both have m — check for exact match first
    gt_m = _extract_m_section(gt.get("extensions", ""))
    pred_m = _extract_m_section(pred.get("extensions", ""))

    if gt_m and pred_m and gt_m == pred_m:
        return 1.0

    return 0.5  # Partial credit for both having m-section


def reward_stable(pred: dict, gt: dict) -> float:
    """R6: Substituent table match."""
    if not gt["stable"]:
        return 1.0 if not pred["stable"] else 0.5
    if not pred["stable"]:
        return 0.0

    # Parse stable sections
    def parse_stable(text):
        entries = {}
        for group in text.split("<ns>"):
            group = group.strip()
            if ":" in group:
                label, rest = group.split(":", 1)
                alternatives = [a.strip() for a in rest.split("<n>")]
                entries[label.strip()] = set(alternatives)
        return entries

    gt_entries = parse_stable(gt["stable"])
    pred_entries = parse_stable(pred["stable"])

    if not gt_entries:
        return 0.5

    matched = 0
    for label, gt_alts in gt_entries.items():
        pred_alts = pred_entries.get(label)
        if pred_alts and gt_alts == pred_alts:
            matched += 1

    return matched / len(gt_entries)


def position_graph_scores(pred: dict, gt: dict) -> dict[str, Any]:
    """Return exact position-aware graph-isomorphism reward diagnostics."""

    if not pred.get("cxsmiles") or not gt.get("cxsmiles"):
        result = MarkushEqualityResult(
            equal=False,
            valid_gt=bool(gt.get("cxsmiles")),
            valid_pred=bool(pred.get("cxsmiles")),
        )
    else:
        # Match the default `ours` eval semantics: position-aware graph
        # isomorphism with stereochemistry removed unless explicitly handled by
        # the prepared target/eval path.
        result = markush_graph_equal(gt["cxsmiles"], pred["cxsmiles"])
    exact = 1.0 if result.equal and result.valid_gt and result.valid_pred else 0.0
    return {
        "structure": exact,
        "position_graph_equal": exact,
        "position_graph_valid_gt": 1.0 if result.valid_gt else 0.0,
        "position_graph_valid_pred": 1.0 if result.valid_pred else 0.0,
        "position_graph_states_visited": float(result.states_visited or 0),
        "position_graph_error": result.error or result.gt_error or result.pred_error or "",
    }


def reward_structure_match(pred: dict, gt: dict) -> float:
    """R7: exact position-aware Markush structure match."""

    return float(position_graph_scores(pred, gt)["structure"])


DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "valid": 0.5,
    "backbone": 1.0,
    "backbone_tanimoto": 0.0,
    "r_groups": 0.5,
    "sg": 1.0,
    "sg_extra": -2.0,
    "m": 0.5,
    "stable": 0.5,
    # Dominant optimization target: the position-aware graph-isomorphism
    # equality used by the `ours` evaluator.
    "structure": 5.0,
    # Downstream eval-aligned bonuses. These default to zero because the graph
    # and MCS evaluators are heavier than the shaped component reward.
    "mg2_cxsmi": 0.0,
    "mg2_stable": 0.0,
    "mg2_full": 0.0,
}


def _merged_reward_weights(weights: dict | None) -> dict[str, float]:
    if weights is None:
        return dict(DEFAULT_REWARD_WEIGHTS)
    merged = {**DEFAULT_REWARD_WEIGHTS, **weights}
    legacy_base_keys = {"valid", "backbone", "r_groups", "sg", "m", "stable"}
    if "structure" not in weights and legacy_base_keys <= set(weights):
        # Preserve the historical "all component keys specified" override
        # behavior used by tests and MG2-bonus targeting.
        merged["structure"] = 0.0
    if "sg_extra" not in weights and legacy_base_keys <= set(weights):
        merged["sg_extra"] = 0.0
    return merged


def reward_mg2_exact_scores(prediction: str, ground_truth: str) -> dict[str, float]:
    """Return exact-match bonuses from the MarkushGrapher-2-style evaluator."""
    try:
        from glyph.markush.eval.metrics_mcs import (
            compute_markush_prediction_quality,
            get_stable_score,
            parse_markush_text,
        )

        pred = parse_markush_text(prediction, strict=False)
        gt = parse_markush_text(ground_truth, strict=False)
        if not pred.valid or not gt.valid:
            return {"mg2_cxsmi": 0.0, "mg2_stable": 0.0, "mg2_full": 0.0}

        mcs_scores = compute_markush_prediction_quality(
            pred.rdkit_cxsmiles,
            gt.rdkit_cxsmiles,
            pred_atom_labels=pred.atom_labels,
            gt_atom_labels=gt.atom_labels,
            protocol="mg2",
        )
        stable_scores = get_stable_score(gt.stable, pred.stable, protocol="mg2")
    except Exception:
        return {"mg2_cxsmi": 0.0, "mg2_stable": 0.0, "mg2_full": 0.0}

    cxsmi = 1.0 if mcs_scores.get("cxsmi_equality") else 0.0
    stable = 1.0 if stable_scores.get("stable_equality") else 0.0
    return {
        "mg2_cxsmi": cxsmi,
        "mg2_stable": stable,
        "mg2_full": 1.0 if cxsmi and stable else 0.0,
    }


def compute_reward_breakdown(
    prediction: str,
    ground_truth: str,
    weights: dict | None = None,
) -> dict[str, Any]:
    """Compute scalar reward plus component scores for RL logging.

    Returns a float from 0 to approximately the sum of component weights
    (default weights sum to 8.0: 1.0 valid + 2.0 backbone + 1.0 r_groups +
    1.5 sg + 0.5 m + 1.0 stable + 1.0 exact structure).

    Component weights can be adjusted to focus training on specific features.
    Default weights emphasize backbone (most impactful) and Sg sections (biggest bottleneck).
    """
    weights = _merged_reward_weights(weights)

    pred = parse_markush(prediction)
    gt = parse_markush(ground_truth)

    # Invalid/non-XML completions should not receive vacuous credit for
    # "missing" optional sections (e.g. no Sg/m section when GT has none).
    # Without this guard, garbage text can receive non-zero reward.
    if not pred["valid_xml"] or not pred["cxsmiles"]:
        return {
            "total": 0.0,
            "valid": 0.0,
            "backbone": 0.0,
            "backbone_tanimoto": 0.0,
            "r_groups": 0.0,
            "sg": 0.0,
            "sg_extra": 0.0,
            "m": 0.0,
            "stable": 0.0,
            "structure": 0.0,
            "position_graph_equal": 0.0,
            "position_graph_valid_gt": 0.0,
            "position_graph_valid_pred": 0.0,
            "position_graph_states_visited": 0.0,
            "position_graph_error": "",
            "mg2_cxsmi": 0.0,
            "mg2_stable": 0.0,
            "mg2_full": 0.0,
            "weights": dict(weights),
        }

    r1 = reward_valid_parse(pred)
    graph_scores = position_graph_scores(pred, gt)
    graph_exact = float(graph_scores["structure"]) == 1.0
    r2 = 1.0 if graph_exact else reward_backbone_match(pred, gt)
    r2_tanimoto = 1.0 if graph_exact else reward_backbone_tanimoto(pred, gt)
    r3 = reward_r_groups(pred, gt)
    r4 = reward_sg_section(pred, gt)
    r4_extra = reward_extra_sg(pred, gt)
    r5 = reward_m_section(pred, gt)
    r6 = reward_stable(pred, gt)
    r7 = float(graph_scores["structure"])
    mg2_scores = (
        reward_mg2_exact_scores(prediction, ground_truth)
        if any(float(weights.get(key, 0.0)) for key in ("mg2_cxsmi", "mg2_stable", "mg2_full"))
        else {"mg2_cxsmi": 0.0, "mg2_stable": 0.0, "mg2_full": 0.0}
    )

    total = (
        weights["valid"] * r1
        + weights["backbone"] * r2
        + weights.get("backbone_tanimoto", 0.0) * r2_tanimoto
        + weights["r_groups"] * r3
        + weights["sg"] * r4
        + weights.get("sg_extra", 0.0) * r4_extra
        + weights["m"] * r5
        + weights["stable"] * r6
        + weights.get("structure", 0.0) * r7
        + weights["mg2_cxsmi"] * mg2_scores["mg2_cxsmi"]
        + weights["mg2_stable"] * mg2_scores["mg2_stable"]
        + weights["mg2_full"] * mg2_scores["mg2_full"]
    )

    return {
        "total": total,
        "valid": r1,
        "backbone": r2,
        "backbone_tanimoto": r2_tanimoto,
        "r_groups": r3,
        "sg": r4,
        "sg_extra": r4_extra,
        "m": r5,
        "stable": r6,
        "structure": r7,
        "position_graph_equal": graph_scores["position_graph_equal"],
        "position_graph_valid_gt": graph_scores["position_graph_valid_gt"],
        "position_graph_valid_pred": graph_scores["position_graph_valid_pred"],
        "position_graph_states_visited": graph_scores["position_graph_states_visited"],
        "position_graph_error": graph_scores["position_graph_error"],
        **mg2_scores,
        "weights": dict(weights),
    }


def compute_reward(prediction: str, ground_truth: str, weights: dict | None = None) -> float:
    """Compute total component reward for a prediction."""

    return float(compute_reward_breakdown(prediction, ground_truth, weights)["total"])


if __name__ == "__main__":
    # Quick test
    gt = "<markush><cxsmi><r>R1</r>C1=CC=CC=C1|Sg:n:1,2:ht</cxsmi><stable>R1:methyl<n>ethyl<ns></stable></markush>"

    # Perfect prediction
    perfect = gt
    print(f"Perfect: {compute_reward(perfect, gt):.2f}")

    # Good prediction (backbone correct, missing Sg)
    good = "<markush><cxsmi><r>R1</r>C1=CC=CC=C1</cxsmi><stable>R1:methyl<n>ethyl<ns></stable></markush>"
    print(f"Good (no Sg): {compute_reward(good, gt):.2f}")

    # Partial (wrong backbone, right R-groups)
    partial = "<markush><cxsmi><r>R1</r>C1=CC=C1</cxsmi><stable></stable></markush>"
    print(f"Partial: {compute_reward(partial, gt):.2f}")

    # Garbage
    garbage = "```json {}"
    print(f"Garbage: {compute_reward(garbage, gt):.2f}")
