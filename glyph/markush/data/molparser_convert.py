"""Conversion from validated MolParser E-SMILES to shared CXSMILES targets.

The first supported conversion track is intentionally narrow: MolParser
``<a>`` labels bound to wildcard atoms become standard CXSMILES atom labels.
MolParser ``<r>`` ring groups become labeled wildcard connector atoms with
standard CXSMILES ``m:`` multicenter attachments to the referenced ring atom
set.  The repo's existing standard-to-opt converter then produces the shared
``cxsmiles_opt`` form. Unsupported native constructs are classified as
native-only instead of being approximated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from glyph.markush.data.molparser_esmiles import MolParserESmiles, parse_molparser_esmiles
from glyph.markush.data.pipeline import (
    MOLPARSER_CONNECTION_POINT_PROP,
    opt_to_standard_cxsmiles,
    standard_to_opt_cxsmiles,
    validate_cxsmiles,
)
from glyph.markush.eval.markush_graph import markush_graph_equal, parse_markush_graph


@dataclass(frozen=True)
class MolParserConversionResult:
    """Non-throwing conversion result for one MolParser E-SMILES value."""

    status: str
    reason: str = ""
    detail: str = ""
    cxsmiles_standard: str = ""
    cxsmiles_opt: str = ""
    target_cxsmiles_format: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def converted(self) -> bool:
        return self.status == "converted_to_cxsmiles_opt"

    @property
    def native_only(self) -> bool:
        return self.status == "native_only"

    @property
    def excluded(self) -> bool:
        return self.status == "excluded"


def _native_only(reason: str, detail: str = "") -> MolParserConversionResult:
    return MolParserConversionResult(status="native_only", reason=reason, detail=detail)


def _excluded(reason: str, detail: str = "") -> MolParserConversionResult:
    return MolParserConversionResult(status="excluded", reason=reason, detail=detail)


def _unsupported_label_text(label: str) -> bool:
    # ``$...$`` atom-label sections use semicolon slots and are embedded inside
    # pipe-delimited CXSMILES extensions.  Keep the first converter to labels
    # that cannot disturb that grammar or the optimized ``<r>...</r>`` wrapper.
    return any(char in label for char in ";,|$<>")


def _unsupported_group_label(group_name: str) -> str:
    if not group_name:
        return "empty_atom_group_name"
    if group_name == "<dum>":
        return ""
    if _unsupported_label_text(group_name):
        return "unsupported_label_text"
    return ""


def _labels_and_m_sections_for_supported_subset(
    parsed: MolParserESmiles,
) -> tuple[str, list[str] | None, list[str], list[int], str, str]:
    assert parsed.mol is not None
    if parsed.circle_groups:
        return "", None, [], [], "contains_circle_groups", f"{len(parsed.circle_groups)} <c> tags"
    if not parsed.atom_groups and not parsed.ring_groups:
        return "", None, [], [], "no_atom_groups", "no convertible <a> or <r> labels"

    connector_count = len(parsed.ring_groups)
    labels = [""] * (connector_count + parsed.mol.GetNumAtoms())
    connection_point_atoms: list[int] = []
    m_sections: list[str] = []

    for connector_idx, group in enumerate(parsed.ring_groups):
        if group.group_name == "<dum>":
            connection_point_atoms.append(connector_idx)
        else:
            unsupported_reason = _unsupported_group_label(group.group_name)
            if unsupported_reason:
                return "", None, [], [], unsupported_reason, group.raw
            labels[connector_idx] = group.group_name
        endpoints = sorted(connector_count + atom_idx for atom_idx in group.atoms)
        endpoint_text = ".".join(str(atom_idx) for atom_idx in endpoints)
        m_sections.append(f"m:{connector_idx}:{endpoint_text}")

    for group in parsed.atom_groups:
        atom = parsed.mol.GetAtomWithIdx(group.atom_index)
        if group.group_name == "<dum>":
            if atom.GetAtomicNum() != 0:
                return "", None, [], [], "dum_connection_point_on_nonwildcard_atom", group.raw
            if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumExplicitHs():
                return "", None, [], [], "decorated_dum_connection_point", group.raw
            connection_point_atoms.append(connector_count + group.atom_index)
            continue
        unsupported_reason = _unsupported_group_label(group.group_name)
        if unsupported_reason:
            return "", None, [], [], unsupported_reason, group.raw
        if atom.GetAtomicNum() != 0:
            return "", None, [], [], "atom_group_on_non_wildcard_atom", group.raw
        output_atom_idx = connector_count + group.atom_index
        existing = labels[output_atom_idx]
        if existing and existing != group.group_name:
            return "", None, [], [], "multiple_atom_groups_on_atom", group.raw
        labels[output_atom_idx] = group.group_name

    if any(count > 1 for count in Counter(connection_point_atoms).values()):
        return "", None, [], [], "duplicate_dum_connection_point", "duplicate <dum> tag on atom"

    core = parsed.core_smiles
    if connector_count:
        core = ".".join([*(["*"] * connector_count), parsed.core_smiles])
    return core, labels, m_sections, connection_point_atoms, "", ""


def _connection_point_atom_prop_section(connection_point_atoms: list[int]) -> str:
    if not connection_point_atoms:
        return ""
    entries = [
        f"{atom_idx}.{MOLPARSER_CONNECTION_POINT_PROP}.1"
        for atom_idx in sorted(connection_point_atoms)
    ]
    return f"atomProp:{':'.join(entries)}"


def _standard_cxsmiles(
    core_smiles: str,
    labels: list[str],
    m_sections: list[str],
    connection_point_atoms: list[int],
) -> str:
    atom_prop_section = _connection_point_atom_prop_section(connection_point_atoms)
    if not any(labels) and not m_sections and not atom_prop_section:
        return core_smiles
    extension_parts = []
    if any(labels):
        extension_parts.append(f"${';'.join(labels)}$")
    if atom_prop_section:
        extension_parts.append(atom_prop_section)
    extension_parts.extend(m_sections)
    return f"{core_smiles} |{','.join(extension_parts)}|"


# MolParser conversions must preserve stereochemistry: markush_graph_equal
# defaults to remove_stereo=True, which would silently accept conversions that
# drop/alter stereocenters (e.g. [C@@H] -> C). All converter soundness gates
# below use stereo-sensitive equality so stereo-lossy rows stay native_only.
_STEREO_SENSITIVE = {"remove_stereo": False}


def _graph_error(left: str, right: str) -> str:
    result = markush_graph_equal(left, right, **_STEREO_SENSITIVE)
    return result.gt_error or result.pred_error or result.error or "shared graph mismatch"


def _requires_unsanitized_core_parse(parsed: MolParserESmiles) -> bool:
    return any(
        warning.startswith("core parsed with RDKit sanitize=False")
        for warning in parsed.parse_warnings
    )


def _decorated_wildcard_label_detail(parsed: MolParserESmiles) -> str:
    assert parsed.mol is not None
    for group in parsed.atom_groups:
        if group.group_name == "<dum>":
            continue
        atom = parsed.mol.GetAtomWithIdx(group.atom_index)
        if atom.GetAtomicNum() != 0:
            continue
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumExplicitHs():
            return group.raw
    return ""


def _atom_features(parsed: Any) -> tuple[tuple[Any, ...], ...]:
    assert parsed.mol is not None
    return tuple(
        (
            atom.GetAtomicNum(),
            atom.GetIsAromatic(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            atom.GetNumExplicitHs(),
        )
        for atom in parsed.mol.GetAtoms()
    )


def _standard_to_opt_mismatch_reason(standard: str, opt: str) -> tuple[str, str]:
    left = parse_markush_graph(standard)
    right = parse_markush_graph(opt)
    if left.valid and right.valid and left.atom_labels == right.atom_labels:
        if left.mol is not None and right.mol is not None:
            left_features = _atom_features(left)
            right_features = _atom_features(right)
            if len(left_features) == len(right_features):
                if left_features == right_features:
                    return "standard_to_opt_graph_mismatch", _graph_error(standard, opt)
                non_h_differences = [
                    idx
                    for idx, (left_atom, right_atom) in enumerate(
                        zip(left_features, right_features, strict=True)
                    )
                    if left_atom[:4] != right_atom[:4]
                ]
                explicit_h_differences = [
                    idx
                    for idx, (left_atom, right_atom) in enumerate(
                        zip(left_features, right_features, strict=True)
                    )
                    if left_atom[4] != right_atom[4]
                ]
                if explicit_h_differences and not non_h_differences:
                    return (
                        "standard_to_opt_atom_feature_mismatch",
                        "standard-to-opt conversion changes explicit hydrogen counts",
                    )
        return (
            "standard_to_opt_atom_feature_mismatch",
            "standard-to-opt conversion changes strict atom features",
        )
    return "standard_to_opt_graph_mismatch", _graph_error(standard, opt)


def convert_molparser_esmiles_to_cxsmiles(raw: Any) -> MolParserConversionResult:
    """Convert the conservative MolParser ``<a>`` subset to shared CXSMILES.

    Rows outside the proven subset return ``status='native_only'`` with a
    reason.  Parse failures return ``status='excluded'`` because they are not
    usable on either the native or converted track without manual repair.
    """

    parsed = parse_molparser_esmiles(raw)
    if not parsed.valid:
        return _excluded("parse_error", parsed.error)

    core_smiles, labels, m_sections, connection_point_atoms, reason, detail = (
        _labels_and_m_sections_for_supported_subset(parsed)
    )
    if labels is None:
        if reason == "empty_atom_group_name":
            return _excluded(reason, detail)
        return _native_only(reason, detail)

    decorated_detail = _decorated_wildcard_label_detail(parsed)
    if decorated_detail:
        return _native_only("decorated_wildcard_atom_label", decorated_detail)

    standard = _standard_cxsmiles(core_smiles, labels, m_sections, connection_point_atoms)
    validation = validate_cxsmiles(standard)
    if not validation.valid:
        if _requires_unsanitized_core_parse(parsed):
            return _native_only(
                "core_requires_unsanitized_parse",
                "; ".join(parsed.parse_warnings),
            )
        return _native_only("standard_cxsmiles_invalid", "; ".join(validation.errors))

    try:
        opt = standard_to_opt_cxsmiles(standard)
    except Exception as exc:
        return _native_only("standard_to_opt_failed", str(exc))

    standard_vs_opt = markush_graph_equal(standard, opt, **_STEREO_SENSITIVE)
    if not (standard_vs_opt.equal and standard_vs_opt.valid_gt and standard_vs_opt.valid_pred):
        mismatch_reason, mismatch_detail = _standard_to_opt_mismatch_reason(standard, opt)
        return _native_only(mismatch_reason, mismatch_detail)

    opt_reflexive = markush_graph_equal(opt, opt, **_STEREO_SENSITIVE)
    if not (opt_reflexive.equal and opt_reflexive.valid_gt and opt_reflexive.valid_pred):
        return _native_only("cxsmiles_opt_not_reflexive", _graph_error(opt, opt))

    try:
        roundtrip_standard = opt_to_standard_cxsmiles(opt)
        roundtrip_opt = standard_to_opt_cxsmiles(roundtrip_standard)
    except Exception as exc:
        return _native_only("cxsmiles_roundtrip_failed", str(exc))

    roundtrip = markush_graph_equal(opt, roundtrip_opt, **_STEREO_SENSITIVE)
    if not (roundtrip.equal and roundtrip.valid_gt and roundtrip.valid_pred):
        return _native_only("cxsmiles_roundtrip_graph_mismatch", _graph_error(opt, roundtrip_opt))

    return MolParserConversionResult(
        status="converted_to_cxsmiles_opt",
        cxsmiles_standard=standard,
        cxsmiles_opt=opt,
        target_cxsmiles_format="cxsmiles_opt",
        warnings=tuple(validation.warnings),
    )
