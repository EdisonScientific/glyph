"""Deterministic structured transcript targets for Markush multitask SFT.

The transcript is generated from an already-validated Markush target.  It is a
machine-readable auxiliary view of the same ``canonicalized cxsmiles_opt``
payload, followed by the normal final ``<markush>`` target so existing eval
extractors can continue to read the final answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from glyph.markush.eval import common
from glyph.markush.eval.markush_graph import ParsedMarkushGraph, parse_markush_graph

SCHEMA_VERSION = "markush_structured_transcript_v1"
DEFAULT_TARGET_FORMAT = "canonicalized cxsmiles_opt"


@dataclass(frozen=True)
class StructuredTranscriptResult:
    """Non-throwing transcript-generation result."""

    valid: bool
    payload: dict[str, Any] | None = None
    transcript_json: str = ""
    final_xml: str = ""
    content: str = ""
    error: str = ""


def _sorted_ints(values: Any) -> list[int]:
    return sorted(int(value) for value in values)


def _bond_order(bond: Chem.Bond) -> str:
    if bond.GetIsAromatic():
        return "aromatic"
    return str(bond.GetBondType()).lower()


def _atom_payload(parsed: ParsedMarkushGraph) -> list[dict[str, Any]]:
    assert parsed.mol is not None
    atoms: list[dict[str, Any]] = []
    for atom in parsed.mol.GetAtoms():
        label = parsed.atom_labels.get(atom.GetIdx())
        item: dict[str, Any] = {
            "idx": atom.GetIdx(),
            "element": "*" if atom.GetAtomicNum() == 0 else atom.GetSymbol(),
            "aromatic": bool(atom.GetIsAromatic()),
            "formal_charge": atom.GetFormalCharge(),
        }
        if label is not None:
            item["r_label"] = label
        atoms.append(item)
    return atoms


def _bond_payload(parsed: ParsedMarkushGraph) -> list[dict[str, Any]]:
    assert parsed.mol is not None
    return [
        {
            "begin": bond.GetBeginAtomIdx(),
            "end": bond.GetEndAtomIdx(),
            "order": _bond_order(bond),
        }
        for bond in parsed.mol.GetBonds()
    ]


def _fragments_payload(parsed: ParsedMarkushGraph) -> list[list[int]]:
    assert parsed.mol is not None
    return sorted([_sorted_ints(fragment) for fragment in Chem.GetMolFrags(parsed.mol)])


def _extensions_payload(parsed: ParsedMarkushGraph) -> dict[str, Any]:
    return {
        "r_labels": [
            {"atom": atom, "label": label}
            for atom, label in sorted(
                parsed.atom_labels.items(), key=lambda item: (item[1], item[0])
            )
        ],
        "connection_points": _sorted_ints(parsed.connection_points),
        "m_sections": [
            {"connector": section.connector, "atoms": _sorted_ints(section.atoms)}
            for section in sorted(
                parsed.m_sections,
                key=lambda section: (section.connector, tuple(_sorted_ints(section.atoms))),
            )
        ],
        "sg_sections": [
            {
                "kind": section.kind,
                "atoms": _sorted_ints(section.atoms),
                "fields": list(section.fields),
            }
            for section in sorted(
                parsed.sg_sections,
                key=lambda section: (
                    section.kind,
                    tuple(_sorted_ints(section.atoms)),
                    section.fields,
                ),
            )
        ],
        "bond_stereo_sections": [
            {"kind": section.kind, "begin": section.begin, "end": section.end}
            for section in sorted(
                parsed.bond_stereo_sections,
                key=lambda section: (section.kind, section.begin, section.end),
            )
        ],
        "component_groups": [
            [_sorted_ints(component) for component in group.components]
            for group in sorted(
                parsed.component_groups,
                key=lambda group: tuple(
                    tuple(_sorted_ints(component)) for component in group.components
                ),
            )
        ],
        "link_nodes": [
            {
                "center": section.center,
                "bounds": list(section.bounds),
                "outer_atoms": list(section.outer_atoms),
            }
            for section in sorted(
                parsed.link_nodes,
                key=lambda section: (section.center, section.bounds, section.outer_atoms),
            )
        ],
    }


def _final_xml(cxsmiles_opt: str, stable: str) -> str:
    return f"<markush><cxsmi>{cxsmiles_opt}</cxsmi><stable>{stable}</stable></markush>"


def structured_transcript_for_markush_target(
    target: Any,
    *,
    target_format: str = DEFAULT_TARGET_FORMAT,
) -> StructuredTranscriptResult:
    """Build a deterministic transcript from a Markush target.

    ``target`` may be a bare CXSMILES string or a ``<markush>`` wrapper.  This
    helper intentionally does not canonicalize.  Callers should run the
    existing canonicalization/graph-validation gates first, then generate this
    transcript from the canonicalized ``cxsmiles_opt`` that will be supervised.
    """

    cxsmiles_opt, stable = common.extract_cxsmiles_and_stable(target, require_wrapper=False)
    if not cxsmiles_opt and isinstance(target, str):
        cxsmiles_opt = target.strip()
    if not cxsmiles_opt:
        return StructuredTranscriptResult(valid=False, error="target has no CXSMILES payload")

    parsed = parse_markush_graph(cxsmiles_opt)
    if not parsed.valid:
        return StructuredTranscriptResult(valid=False, error=parsed.error)

    assert parsed.mol is not None
    fragments = _fragments_payload(parsed)
    final_xml = _final_xml(cxsmiles_opt, stable)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target_format": target_format,
        "graph": {
            "atom_count": parsed.mol.GetNumAtoms(),
            "bond_count": parsed.mol.GetNumBonds(),
            "fragment_count": len(fragments),
            "atoms": _atom_payload(parsed),
            "bonds": _bond_payload(parsed),
            "fragments": fragments,
        },
        "extensions": _extensions_payload(parsed),
        "stable": stable,
        "final": {
            "cxsmiles_opt": cxsmiles_opt,
        },
    }
    transcript_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return StructuredTranscriptResult(
        valid=True,
        payload=payload,
        transcript_json=transcript_json,
        final_xml=final_xml,
        content=f"{transcript_json}\n{final_xml}",
    )


def validate_structured_transcript_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    """Validate the internal consistency of a generated transcript payload."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        return False, "wrong schema_version"
    final = payload.get("final")
    if not isinstance(final, dict):
        return False, "missing final object"
    cxsmiles_opt = final.get("cxsmiles_opt")
    if not isinstance(cxsmiles_opt, str) or not cxsmiles_opt.strip():
        return False, "missing final.cxsmiles_opt"
    parsed = parse_markush_graph(cxsmiles_opt)
    if not parsed.valid:
        return False, parsed.error
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        return False, "missing graph object"
    assert parsed.mol is not None
    if graph.get("atom_count") != parsed.mol.GetNumAtoms():
        return False, "atom_count does not match parsed graph"
    if graph.get("bond_count") != parsed.mol.GetNumBonds():
        return False, "bond_count does not match parsed graph"
    if graph.get("fragment_count") != len(Chem.GetMolFrags(parsed.mol)):
        return False, "fragment_count does not match parsed graph"
    return True, ""


def validate_structured_transcript_content(content: str) -> tuple[bool, str]:
    """Validate a two-line transcript-then-final assistant target."""

    if not isinstance(content, str) or "\n" not in content:
        return False, "content must contain transcript JSON and final XML lines"
    transcript_json, final_xml = content.split("\n", 1)
    try:
        payload = json.loads(transcript_json)
    except json.JSONDecodeError as exc:
        return False, f"transcript JSON does not parse: {exc}"
    ok, error = validate_structured_transcript_payload(payload)
    if not ok:
        return False, error
    extracted, stable = common.extract_cxsmiles_and_stable(final_xml, require_wrapper=True)
    if extracted != payload["final"]["cxsmiles_opt"]:
        return False, "final XML CXSMILES does not match transcript final.cxsmiles_opt"
    if stable != payload.get("stable", ""):
        return False, "final XML stable table does not match transcript stable"
    return True, ""
