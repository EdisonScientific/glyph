"""Position-aware Markush graph equality.

This module implements the correctness relation for the CXSMILES structure
part without using the legacy MCS scorer.  It parses the backbone graph, binds
R labels to the exact atoms that carried them, parses atom-indexed CXSMILES
annotations, and accepts only when a full graph isomorphism carries every
annotation to the prediction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import permutations
from math import factorial, prod
from typing import Any

from rdkit import Chem, RDLogger

from glyph.markush.eval import common

RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]

_R_TAG = common.RE_R_TAG
_MARKER_BASE = 9000
_NUMERIC_INDEX_FIELD = re.compile(r"\d+(?:[,.]\d+)*")
_BOND_STEREO_MARKERS: tuple[str, ...] = ("c:", "t:", "ctu:")
_BOND_STEREO_MARKER_RE = re.compile(r"ctu:|c:|t:")
_SMILES_BOND_TOKENS = set("-=#:$/\\~")
_MOLPARSER_CONNECTION_POINT_PROP = "molparserConnectionPoint"


@dataclass(frozen=True)
class MSection:
    """A positional-variation attachment: connector atom + allowed atom set."""

    connector: int
    atoms: frozenset[int]


@dataclass(frozen=True)
class SGroupSection:
    """A CXSMILES ``Sg:`` section with atom membership and non-index fields."""

    kind: str
    atoms: frozenset[int]
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class BondStereoSection:
    """A CXSMILES ``c:``, ``t:``, or ``ctu:`` bond annotation."""

    kind: str
    begin: int
    end: int


@dataclass(frozen=True)
class ComponentGroup:
    """A CXSMILES ``f:`` group expressed as grouped atom fragments."""

    components: tuple[frozenset[int], ...]


@dataclass(frozen=True)
class LinkNodeSection:
    """A CXSMILES ``LN:`` repeat/link-node annotation."""

    center: int
    bounds: tuple[int, int]
    outer_atoms: tuple[int, ...] = ()


@dataclass
class ParsedMarkushGraph:
    """Parsed label-bound Markush graph used by the equality predicate."""

    raw: str
    core: str = ""
    extension: str = ""
    mol: Any | None = None
    atom_labels: dict[int, str] = field(default_factory=dict)
    connection_points: tuple[int, ...] = ()
    m_sections: tuple[MSection, ...] = ()
    sg_sections: tuple[SGroupSection, ...] = ()
    bond_stereo_sections: tuple[BondStereoSection, ...] = ()
    core_bond_stereo_sections: tuple[BondStereoSection, ...] = ()
    component_groups: tuple[ComponentGroup, ...] = ()
    link_nodes: tuple[LinkNodeSection, ...] = ()
    valid: bool = False
    error: str = ""


@dataclass(frozen=True)
class MarkushEqualityResult:
    """A non-throwing verdict for one GT/prediction pair."""

    equal: bool
    valid_gt: bool
    valid_pred: bool
    gt_error: str = ""
    pred_error: str = ""
    error: str = ""
    states_visited: int = 0


@dataclass(frozen=True)
class MarkushCanonicalResult:
    """A deterministic canonical signature for one parsed Markush graph."""

    valid: bool
    signature: str = ""
    error: str = ""
    states_visited: int = 0


class _SearchLimitExceededError(RuntimeError):
    pass


def networkx_available() -> bool:
    """Return whether the optional NetworkX VF2 backend can be imported."""

    try:
        import networkx  # noqa: F401
    except ImportError:
        return False
    return True


def split_cxsmiles(cxsmiles: str) -> tuple[str, str]:
    """Split standard or optimized CXSMILES into ``(core, extension)``."""

    value = (cxsmiles or "").strip()
    if "|" not in value:
        return value, ""
    core, rest = value.split("|", 1)
    if "|" in rest:
        rest = rest.rsplit("|", 1)[0]
    return core.strip(), rest.strip()


def split_extension_sections(extension: str) -> list[str]:
    """Split CXSMILES extension sections without splitting Sg atom lists."""

    return [marker + payload for marker, payload in common.parse_extension_sections(extension)]


def _read_ring_number(smiles: str, pos: int) -> tuple[str | None, int]:
    if pos >= len(smiles):
        return None, pos
    if smiles[pos].isdigit():
        return smiles[pos], pos + 1
    if smiles[pos] == "%" and pos + 2 < len(smiles) and smiles[pos + 1 : pos + 3].isdigit():
        return smiles[pos + 1 : pos + 3], pos + 3
    return None, pos


def _read_smiles_atom(smiles: str, pos: int) -> tuple[bool, int]:
    if pos >= len(smiles):
        return False, pos
    char = smiles[pos]
    if char == "[":
        end = smiles.find("]", pos + 1)
        return (end != -1), (end + 1 if end != -1 else pos + 1)
    if char == "*":
        return True, pos + 1
    if smiles.startswith(("Cl", "Br"), pos):
        return True, pos + 2
    if char in "BCNOPSFIbcnops":
        return True, pos + 1
    return False, pos


def _smiles_bond_atom_pairs_in_cx_order(smiles_core: str) -> list[tuple[int, int]]:
    """Return bond atom pairs in CXSMILES textual bond-index order."""

    pairs: list[tuple[int, int]] = []
    branch_stack: list[int | None] = []
    ring_open: dict[str, tuple[int, str | None]] = {}
    current_atom: int | None = None
    next_atom_idx = 0
    pending_bond: str | None = None
    disconnected = True
    i = 0

    while i < len(smiles_core):
        char = smiles_core[i]

        if char.isspace():
            i += 1
            continue
        if char == ".":
            disconnected = True
            pending_bond = None
            i += 1
            continue
        if char == "(":
            branch_stack.append(current_atom)
            i += 1
            continue
        if char == ")":
            current_atom = branch_stack.pop() if branch_stack else current_atom
            pending_bond = None
            disconnected = False
            i += 1
            continue
        if char in _SMILES_BOND_TOKENS:
            pending_bond = char
            i += 1
            continue

        ring_number, next_pos = _read_ring_number(smiles_core, i)
        if ring_number is not None:
            if current_atom is not None:
                if ring_number in ring_open:
                    other_atom, _open_bond = ring_open.pop(ring_number)
                    pairs.append((other_atom, current_atom))
                else:
                    ring_open[ring_number] = (current_atom, pending_bond)
            pending_bond = None
            disconnected = False
            i = next_pos
            continue

        is_atom, next_pos = _read_smiles_atom(smiles_core, i)
        if is_atom:
            atom_idx = next_atom_idx
            next_atom_idx += 1
            if current_atom is not None and not disconnected:
                pairs.append((current_atom, atom_idx))
            current_atom = atom_idx
            pending_bond = None
            disconnected = False
            i = next_pos
            continue

        i += 1

    return pairs


def _replace_r_tags_with_markers(core: str) -> tuple[str, dict[int, str]]:
    marker_to_label: dict[int, str] = {}
    next_marker = _MARKER_BASE

    def replace_label(label: str) -> str:
        nonlocal next_marker
        marker = next_marker
        next_marker += 1
        marker_to_label[marker] = label.strip()
        return f"[*:{marker}]"

    def replace_r_tag(match: re.Match[str]) -> str:
        return replace_label(match.group(1))

    marked = _R_TAG.sub(replace_r_tag, core)
    # IP5-M contains ``[Ar]`` as an aryl pseudo-label bonded into the graph.
    # RDKit interprets it as elemental argon and rejects the valence, so bind it
    # as a labeled wildcard.  Keep this intentionally narrow; bracketed real
    # elements such as [Si], [Sn], [Bi], [Te], [Co], and [H] remain elements.
    marked = re.sub(r"\[Ar\]", lambda _match: replace_label("Ar"), marked)
    return marked, marker_to_label


def _dollar_labels(sections: list[str]) -> tuple[dict[int, str], int]:
    for section in sections:
        if section.startswith("$") and section.endswith("$"):
            labels = section[1:-1].split(";")
            while labels and not labels[-1].strip():
                labels.pop()
            return (
                {idx: label.strip() for idx, label in enumerate(labels) if label.strip()},
                len(labels),
            )
    return {}, 0


def _atom_prop_labels(sections: list[str]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for section in sections:
        if not section.startswith("atomProp:"):
            continue
        for entry in section.removeprefix("atomProp:").split(":"):
            try:
                atom_idx_text, prop, value = entry.split(".", 2)
                atom_idx = int(atom_idx_text)
            except ValueError:
                continue
            if prop == "dummyLabel" and value.strip():
                labels[atom_idx] = value.strip()
    return labels


def _atom_prop_connection_points(sections: list[str]) -> tuple[int, ...]:
    connection_points = []
    for section in sections:
        if not section.startswith("atomProp:"):
            continue
        for entry in section.removeprefix("atomProp:").split(":"):
            try:
                atom_idx_text, prop, value = entry.split(".", 2)
                atom_idx = int(atom_idx_text)
            except ValueError:
                continue
            if prop == _MOLPARSER_CONNECTION_POINT_PROP and value.strip() not in {
                "",
                "0",
                "false",
                "False",
            }:
                connection_points.append(atom_idx)
    return tuple(connection_points)


def _bind_atom_label(
    parsed: ParsedMarkushGraph,
    atom_labels: dict[int, str],
    atom_idx: int,
    label: str,
) -> None:
    existing = atom_labels.get(atom_idx)
    if existing is not None and existing != label:
        parsed.error = f"conflicting atom labels for atom {atom_idx}: {existing!r} vs {label!r}"
        return
    atom_labels[atom_idx] = label


def _parse_int_list(text: str, separator: str) -> list[int]:
    if not text:
        return []
    return [int(part) for part in text.split(separator) if part != ""]


def _parse_m_section(section: str) -> MSection | None:
    parts = section.split(":")
    if len(parts) < 3:
        return None
    try:
        connector = int(parts[1])
        atoms = frozenset(_parse_int_list(parts[2], "."))
    except ValueError:
        return None
    return MSection(connector=connector, atoms=atoms)


def _parse_sgroup_section(section: str) -> SGroupSection | None:
    parts = section.rstrip(",").split(":")
    while parts and parts[-1] == "":
        parts.pop()
    if len(parts) < 3:
        return None
    try:
        atoms = frozenset(_parse_int_list(parts[2], ","))
    except ValueError:
        return None
    return SGroupSection(kind=parts[1], atoms=atoms, fields=tuple(parts[3:]))


def _cx_bond_index_to_atom_pair(marked_core: str, mol: Any) -> dict[int, tuple[int, int]]:
    mapping: dict[int, tuple[int, int]] = {}
    for cx_idx, (begin, end) in enumerate(_smiles_bond_atom_pairs_in_cx_order(marked_core)):
        if begin >= mol.GetNumAtoms() or end >= mol.GetNumAtoms():
            continue
        bond = mol.GetBondBetweenAtoms(begin, end)
        if bond is not None:
            mapping[cx_idx] = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return mapping


def _parse_bond_stereo_sections(
    parsed: ParsedMarkushGraph,
    sections: list[str],
    marked_core: str,
) -> tuple[BondStereoSection, ...]:
    assert parsed.mol is not None
    cx_to_pair = _cx_bond_index_to_atom_pair(marked_core, parsed.mol)
    result: list[BondStereoSection] = []
    for section in sections:
        matches = list(_BOND_STEREO_MARKER_RE.finditer(section))
        if not matches or matches[0].start() != 0:
            continue
        for idx, match in enumerate(matches):
            marker = match.group(0)
            kind = marker[:-1]
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section)
            values = section[match.end() : end].strip(",")
            for token in values.replace(".", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    cx_bond_idx = int(token)
                except ValueError:
                    parsed.error = f"malformed {marker} bond index: {token!r}"
                    return ()
                pair = cx_to_pair.get(cx_bond_idx)
                if pair is None:
                    parsed.error = f"{marker} bond index {cx_bond_idx} out of range"
                    return ()
                result.append(BondStereoSection(kind=kind, begin=pair[0], end=pair[1]))
    return tuple(result)


def _core_bond_stereo_sections(mol: Any) -> tuple[BondStereoSection, ...]:
    result: list[BondStereoSection] = []
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo == Chem.rdchem.BondStereo.STEREOE:
            kind = "t"
        elif stereo == Chem.rdchem.BondStereo.STEREOZ:
            kind = "c"
        else:
            continue
        result.append(
            BondStereoSection(
                kind=kind,
                begin=bond.GetBeginAtomIdx(),
                end=bond.GetEndAtomIdx(),
            )
        )
    return tuple(result)


def _parse_component_groups(
    parsed: ParsedMarkushGraph,
    sections: list[str],
) -> tuple[ComponentGroup, ...]:
    assert parsed.mol is not None
    fragments = [frozenset(indices) for indices in Chem.GetMolFrags(parsed.mol)]
    groups: list[ComponentGroup] = []
    for section in sections:
        if not section.startswith("f:"):
            continue
        payload = section.removeprefix("f:")
        for group_text in payload.split(","):
            component_sets: list[frozenset[int]] = []
            for token in group_text.split("."):
                token = token.strip()
                if not token:
                    continue
                try:
                    component_idx = int(token)
                except ValueError:
                    parsed.error = f"malformed f component index: {token!r}"
                    return ()
                if component_idx < 0 or component_idx >= len(fragments):
                    parsed.error = f"f component index {component_idx} out of range"
                    return ()
                component_sets.append(fragments[component_idx])
            if not component_sets:
                parsed.error = "empty f component group"
                return ()
            groups.append(
                ComponentGroup(
                    components=tuple(sorted(component_sets, key=lambda atoms: tuple(sorted(atoms))))
                )
            )
    return tuple(groups)


def _parse_link_nodes(sections: list[str]) -> tuple[LinkNodeSection, ...]:
    result: list[LinkNodeSection] = []
    for section in sections:
        if not section.startswith("LN:"):
            continue
        parts = section.removeprefix("LN:").split(":", 1)
        if len(parts) != 2:
            return (LinkNodeSection(center=-1, bounds=(-1, -1)),)
        try:
            center = int(parts[0])
            values = [int(token) for token in parts[1].split(".") if token != ""]
        except ValueError:
            return (LinkNodeSection(center=-1, bounds=(-1, -1)),)
        if len(values) < 2:
            return (LinkNodeSection(center=-1, bounds=(-1, -1)),)
        result.append(
            LinkNodeSection(
                center=center,
                bounds=(values[0], values[1]),
                outer_atoms=tuple(values[2:]),
            )
        )
    return tuple(result)


def _validate_indices(
    parsed: ParsedMarkushGraph,
    *,
    dollar_label_count: int | None = None,
) -> None:
    assert parsed.mol is not None
    n_atoms = parsed.mol.GetNumAtoms()
    if dollar_label_count is not None and dollar_label_count > n_atoms:
        parsed.error = f"$ atom-label section has {dollar_label_count} labels for {n_atoms} atoms"
        return
    for atom_idx in parsed.atom_labels:
        if atom_idx < 0 or atom_idx >= n_atoms:
            parsed.error = f"R label atom index {atom_idx} out of range for {n_atoms} atoms"
            return
    for section in parsed.m_sections:
        indices = {section.connector, *section.atoms}
        bad = [idx for idx in indices if idx < 0 or idx >= n_atoms]
        if bad:
            parsed.error = f"m section atom index {bad[0]} out of range for {n_atoms} atoms"
            return
    for atom_idx in parsed.connection_points:
        if atom_idx < 0 or atom_idx >= n_atoms:
            parsed.error = (
                f"connection point atom index {atom_idx} out of range for {n_atoms} atoms"
            )
            return
    for section in parsed.sg_sections:
        bad = [idx for idx in section.atoms if idx < 0 or idx >= n_atoms]
        if bad:
            parsed.error = f"Sg section atom index {bad[0]} out of range for {n_atoms} atoms"
            return
    for section in parsed.bond_stereo_sections:
        bad = [idx for idx in (section.begin, section.end) if idx < 0 or idx >= n_atoms]
        if bad:
            parsed.error = (
                f"{section.kind} bond atom index {bad[0]} out of range for {n_atoms} atoms"
            )
            return
    for group in parsed.component_groups:
        for component in group.components:
            bad = [idx for idx in component if idx < 0 or idx >= n_atoms]
            if bad:
                parsed.error = f"f component atom index {bad[0]} out of range for {n_atoms} atoms"
                return
    for section in parsed.link_nodes:
        indices = [section.center, *section.outer_atoms]
        bad = [idx for idx in indices if idx < 0 or idx >= n_atoms]
        if bad:
            parsed.error = f"LN atom index {bad[0]} out of range for {n_atoms} atoms"
            return


def parse_markush_graph(raw: Any, *, remove_stereo: bool = True) -> ParsedMarkushGraph:
    """Parse a Markush/CXSMILES string into a label-bound graph.

    ``<r>LABEL</r>`` tags are converted to RDKit wildcard atoms carrying a
    temporary atom-map marker, then the literal label is copied into
    ``atom_labels`` by atom index.  Standard ``$...$`` atom-label sections are
    also accepted so converter tests can compare standard and optimized forms.
    """

    parsed = ParsedMarkushGraph(raw="" if raw is None else str(raw))
    if not isinstance(raw, str) or not raw.strip():
        parsed.error = "empty or non-string Markush value"
        return parsed

    cxsmiles, _stable = common.extract_cxsmiles_and_stable(raw, require_wrapper=False)
    if not cxsmiles:
        if "<markush" in raw.lower() or "<cxsmi" in raw.lower():
            parsed.error = "malformed Markush XML/CXSMILES wrapper"
            return parsed
        cxsmiles = raw.strip()

    core, extension = split_cxsmiles(cxsmiles)
    sections = split_extension_sections(extension)
    marked_core, marker_to_label = _replace_r_tags_with_markers(core)

    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.strictCXSMILES = False
    try:
        mol = Chem.MolFromSmiles(marked_core, params)
    except Exception as exc:
        parsed.error = f"SMILES core does not parse: {exc}"
        return parsed
    if mol is None:
        parsed.error = "SMILES core does not parse"
        return parsed
    core_bond_stereo_sections = _core_bond_stereo_sections(mol)
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
        for bond in mol.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)

    atom_labels: dict[int, str] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
        marker = atom.GetAtomMapNum()
        if marker in marker_to_label:
            _bind_atom_label(parsed, atom_labels, atom.GetIdx(), marker_to_label[marker])
            atom.SetAtomMapNum(0)
    if parsed.error:
        return parsed

    dollar_labels, dollar_label_count = _dollar_labels(sections)
    for atom_idx, label in dollar_labels.items():
        _bind_atom_label(parsed, atom_labels, atom_idx, label)
        if parsed.error:
            return parsed
    for atom_idx, label in _atom_prop_labels(sections).items():
        _bind_atom_label(parsed, atom_labels, atom_idx, label)
        if parsed.error:
            return parsed

    parsed.core = core
    parsed.extension = extension
    parsed.mol = mol
    parsed.atom_labels = atom_labels
    parsed.core_bond_stereo_sections = core_bond_stereo_sections
    parsed.connection_points = _atom_prop_connection_points(sections)
    parsed.m_sections = tuple(
        section
        for section in (_parse_m_section(item) for item in sections if item.startswith("m:"))
        if section is not None
    )
    parsed.sg_sections = tuple(
        section
        for section in (_parse_sgroup_section(item) for item in sections if item.startswith("Sg:"))
        if section is not None
    )
    parsed.bond_stereo_sections = _parse_bond_stereo_sections(parsed, sections, marked_core)
    if parsed.error:
        return parsed
    parsed.component_groups = _parse_component_groups(parsed, sections)
    if parsed.error:
        return parsed
    parsed.link_nodes = _parse_link_nodes(sections)

    _validate_indices(parsed, dollar_label_count=dollar_label_count or None)
    if parsed.error:
        return parsed
    parsed.valid = True
    return parsed


def _atom_feature(
    parsed: ParsedMarkushGraph, atom_idx: int, *, include_stereo: bool
) -> tuple[Any, ...]:
    assert parsed.mol is not None
    atom = parsed.mol.GetAtomWithIdx(atom_idx)
    chiral = str(atom.GetChiralTag()) if include_stereo else ""
    return (
        atom.GetAtomicNum(),
        atom.GetIsAromatic(),
        atom.GetFormalCharge(),
        atom.GetIsotope(),
        atom.GetTotalNumHs(),
        chiral,
        parsed.atom_labels.get(atom_idx, ""),
    )


def _bond_feature(bond: Any, *, include_stereo: bool) -> tuple[Any, ...]:
    stereo = str(bond.GetStereo()) if include_stereo else ""
    return (
        str(bond.GetBondType()),
        bond.GetIsAromatic(),
        bond.GetIsConjugated(),
        stereo,
    )


def _annotation_invariant(parsed: ParsedMarkushGraph, atom_idx: int) -> tuple[Any, ...]:
    """Return local Markush annotation roles safe for isomorphism pruning."""

    pieces = []
    for section in parsed.m_sections:
        roles = []
        if atom_idx == section.connector:
            roles.append("connector")
        if atom_idx in section.atoms:
            roles.append("member")
        if roles:
            pieces.append(("m", tuple(roles), len(section.atoms)))
    if atom_idx in parsed.connection_points:
        pieces.append(("connection_point",))
    for section in parsed.sg_sections:
        if atom_idx not in section.atoms:
            continue
        stable_fields = tuple(
            field
            for idx, field in enumerate(section.fields)
            if idx < 2 or not _NUMERIC_INDEX_FIELD.fullmatch(field)
        )
        pieces.append(("Sg", section.kind, len(section.atoms), stable_fields))
    for group in parsed.component_groups:
        for component in group.components:
            if atom_idx in component:
                pieces.append(("f", len(component), len(group.components)))
                break
    for section in parsed.link_nodes:
        roles = []
        if atom_idx == section.center:
            roles.append("center")
        if atom_idx in section.outer_atoms:
            roles.append("outer")
        if roles:
            pieces.append(("LN", tuple(roles), section.bounds, len(section.outer_atoms)))
    return tuple(sorted(pieces))


def _node_feature(
    parsed: ParsedMarkushGraph, atom_idx: int, *, include_stereo: bool
) -> tuple[Any, ...]:
    return (
        _atom_feature(parsed, atom_idx, include_stereo=include_stereo),
        _annotation_invariant(parsed, atom_idx),
    )


def _neighbor_signature(
    parsed: ParsedMarkushGraph, atom_idx: int, *, include_stereo: bool
) -> tuple[Any, ...]:
    assert parsed.mol is not None
    atom = parsed.mol.GetAtomWithIdx(atom_idx)
    pieces = []
    for bond in atom.GetBonds():
        other_idx = bond.GetOtherAtomIdx(atom_idx)
        pieces.append(
            (
                _bond_feature(bond, include_stereo=include_stereo),
                _node_feature(parsed, other_idx, include_stereo=include_stereo),
            )
        )
    return tuple(sorted(pieces))


def _map_numeric_field(field: str, mapping: dict[int, int]) -> str:
    if not _NUMERIC_INDEX_FIELD.fullmatch(field):
        return field
    if "," in field:
        sep = ","
    elif "." in field:
        sep = "."
    else:
        sep = ""
    values = re.split(r"[,.]", field)
    if any(int(value) not in mapping for value in values if value != ""):
        return field
    mapped = [str(mapping[int(value)]) for value in values if value != ""]
    return sep.join(mapped) if sep else mapped[0]


def _m_signature(sections: tuple[MSection, ...], mapping: dict[int, int]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                mapping[section.connector],
                tuple(sorted(mapping[idx] for idx in section.atoms)),
            )
            for section in sections
        )
    )


def _sg_signature(sections: tuple[SGroupSection, ...], mapping: dict[int, int]) -> tuple[Any, ...]:
    def mapped_fields(section: SGroupSection) -> tuple[str, ...]:
        values = []
        for idx, raw_field in enumerate(section.fields):
            values.append(_map_numeric_field(raw_field, mapping) if idx >= 2 else raw_field)
        return tuple(values)

    return tuple(
        sorted(
            (
                section.kind,
                tuple(sorted(mapping[idx] for idx in section.atoms)),
                mapped_fields(section),
            )
            for section in sections
        )
    )


def _bond_stereo_signature(
    sections: tuple[BondStereoSection, ...],
    mapping: dict[int, int],
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                section.kind,
                tuple(sorted((mapping[section.begin], mapping[section.end]))),
            )
            for section in sections
        )
    )


def _effective_bond_stereo_sections(
    parsed: ParsedMarkushGraph,
    *,
    use_core_fallback: bool,
) -> tuple[BondStereoSection, ...]:
    sections = list(parsed.bond_stereo_sections)
    if use_core_fallback:
        explicit_pairs = {
            tuple(sorted((section.begin, section.end))) for section in parsed.bond_stereo_sections
        }
        sections.extend(
            section
            for section in parsed.core_bond_stereo_sections
            if tuple(sorted((section.begin, section.end))) not in explicit_pairs
        )
    return tuple(sections)


def _component_group_signature(
    groups: tuple[ComponentGroup, ...],
    mapping: dict[int, int],
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            tuple(
                sorted(
                    tuple(sorted(mapping[idx] for idx in component))
                    for component in group.components
                )
            )
            for group in groups
        )
    )


def _link_node_signature(
    sections: tuple[LinkNodeSection, ...],
    mapping: dict[int, int],
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                mapping[section.center],
                section.bounds,
                tuple(sorted(mapping[idx] for idx in section.outer_atoms)),
            )
            for section in sections
        )
    )


def _numeric_field_references_atom(field: str, atom_idx: int) -> bool:
    if not _NUMERIC_INDEX_FIELD.fullmatch(field):
        return False
    return any(int(value) == atom_idx for value in re.split(r"[,.]", field) if value != "")


def _removable_explicit_hydrogen_atoms(parsed: ParsedMarkushGraph) -> frozenset[int]:
    """Return explicit H atoms safe to collapse into ordinary implicit H counts.

    This is deliberately conservative. Hydrogens carrying direct Markush
    annotations remain exact. Sgroup membership is the one exception: an
    ordinary explicit H may be dropped from an Sgroup when its heavy-atom
    neighbor is in the same Sgroup, matching the observed strict_v3
    adjacent-Sg-[H] convention.
    """

    assert parsed.mol is not None
    explicit_h = {atom.GetIdx() for atom in parsed.mol.GetAtoms() if atom.GetAtomicNum() == 1}
    if not explicit_h:
        return frozenset()

    m_atoms = {section.connector for section in parsed.m_sections}
    for section in parsed.m_sections:
        m_atoms.update(section.atoms)
    connection_points = set(parsed.connection_points)
    link_node_atoms = {section.center for section in parsed.link_nodes}
    for section in parsed.link_nodes:
        link_node_atoms.update(section.outer_atoms)
    bond_stereo_atoms = {
        idx
        for section in (*parsed.bond_stereo_sections, *parsed.core_bond_stereo_sections)
        for idx in (section.begin, section.end)
    }

    removable: set[int] = set()
    for atom_idx in explicit_h:
        atom = parsed.mol.GetAtomWithIdx(atom_idx)
        if (
            atom.GetIsotope() != 0
            or atom.GetFormalCharge() != 0
            or atom.GetDegree() != 1
            or parsed.atom_labels.get(atom_idx)
            or atom_idx in m_atoms
            or atom_idx in connection_points
            or atom_idx in link_node_atoms
            or atom_idx in bond_stereo_atoms
        ):
            continue

        neighbor_idx = atom.GetNeighbors()[0].GetIdx()
        valid_sgroup_membership = True
        for section in parsed.sg_sections:
            if atom_idx not in section.atoms:
                continue
            if neighbor_idx not in section.atoms:
                valid_sgroup_membership = False
                break
            if any(_numeric_field_references_atom(field, atom_idx) for field in section.fields):
                valid_sgroup_membership = False
                break
        if valid_sgroup_membership:
            removable.add(atom_idx)
    return frozenset(removable)


def _without_removable_explicit_hydrogens(parsed: ParsedMarkushGraph) -> ParsedMarkushGraph:
    """Return a parsed graph with ordinary explicit H atoms folded into neighbors."""

    if not parsed.valid or parsed.mol is None:
        return parsed

    removable_h = _removable_explicit_hydrogen_atoms(parsed)
    if not removable_h:
        return parsed

    explicit_h = {atom.GetIdx() for atom in parsed.mol.GetAtoms() if atom.GetAtomicNum() == 1}
    if explicit_h - removable_h:
        return parsed

    mol = Chem.Mol(parsed.mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    try:
        normalized_mol = Chem.RemoveHs(mol)
    except Exception:
        return parsed

    old_to_new: dict[int, int] = {}
    for atom in normalized_mol.GetAtoms():
        marker = atom.GetAtomMapNum()
        if marker <= 0:
            return parsed
        old_to_new[marker - 1] = atom.GetIdx()
        atom.SetAtomMapNum(0)

    removed_by_rdkit = set(range(parsed.mol.GetNumAtoms())) - set(old_to_new)
    if removed_by_rdkit != set(removable_h):
        return parsed

    def remap_atom_set(indices: frozenset[int]) -> frozenset[int]:
        return frozenset(old_to_new[idx] for idx in indices if idx in old_to_new)

    remapped_sgroups = tuple(
        SGroupSection(section.kind, remap_atom_set(section.atoms), section.fields)
        for section in parsed.sg_sections
        if remap_atom_set(section.atoms)
    )
    remapped_component_groups = tuple(
        ComponentGroup(
            tuple(
                component
                for component in (remap_atom_set(component) for component in group.components)
                if component
            )
        )
        for group in parsed.component_groups
    )

    return ParsedMarkushGraph(
        raw=parsed.raw,
        core=parsed.core,
        extension=parsed.extension,
        mol=normalized_mol,
        atom_labels={old_to_new[idx]: label for idx, label in parsed.atom_labels.items()},
        connection_points=tuple(old_to_new[idx] for idx in parsed.connection_points),
        m_sections=tuple(
            MSection(
                old_to_new[section.connector],
                frozenset(old_to_new[idx] for idx in section.atoms),
            )
            for section in parsed.m_sections
        ),
        sg_sections=remapped_sgroups,
        bond_stereo_sections=tuple(
            BondStereoSection(section.kind, old_to_new[section.begin], old_to_new[section.end])
            for section in parsed.bond_stereo_sections
        ),
        core_bond_stereo_sections=tuple(
            BondStereoSection(section.kind, old_to_new[section.begin], old_to_new[section.end])
            for section in parsed.core_bond_stereo_sections
        ),
        component_groups=remapped_component_groups,
        link_nodes=tuple(
            LinkNodeSection(
                old_to_new[section.center],
                section.bounds,
                tuple(old_to_new[idx] for idx in section.outer_atoms),
            )
            for section in parsed.link_nodes
        ),
        valid=True,
    )


def _extension_features_match(
    gt: ParsedMarkushGraph,
    pred: ParsedMarkushGraph,
    gt_to_pred: dict[int, int],
) -> bool:
    assert pred.mol is not None
    identity = {idx: idx for idx in range(pred.mol.GetNumAtoms())}
    use_core_bond_stereo = bool(gt.bond_stereo_sections or pred.bond_stereo_sections)
    return (
        _m_signature(gt.m_sections, gt_to_pred) == _m_signature(pred.m_sections, identity)
        and _sg_signature(gt.sg_sections, gt_to_pred) == _sg_signature(pred.sg_sections, identity)
        and _bond_stereo_signature(
            _effective_bond_stereo_sections(
                gt,
                use_core_fallback=use_core_bond_stereo,
            ),
            gt_to_pred,
        )
        == _bond_stereo_signature(
            _effective_bond_stereo_sections(
                pred,
                use_core_fallback=use_core_bond_stereo,
            ),
            identity,
        )
        and _component_group_signature(gt.component_groups, gt_to_pred)
        == _component_group_signature(pred.component_groups, identity)
        and _link_node_signature(gt.link_nodes, gt_to_pred)
        == _link_node_signature(pred.link_nodes, identity)
    )


def _candidate_map(
    gt: ParsedMarkushGraph,
    pred: ParsedMarkushGraph,
    *,
    include_stereo: bool,
) -> dict[int, list[int]]:
    assert gt.mol is not None and pred.mol is not None
    pred_by_feature: dict[tuple[Any, ...], list[int]] = {}
    for atom in pred.mol.GetAtoms():
        idx = atom.GetIdx()
        feature = (
            _node_feature(pred, idx, include_stereo=include_stereo),
            atom.GetDegree(),
            _neighbor_signature(pred, idx, include_stereo=include_stereo),
        )
        pred_by_feature.setdefault(feature, []).append(idx)

    candidates: dict[int, list[int]] = {}
    for atom in gt.mol.GetAtoms():
        idx = atom.GetIdx()
        feature = (
            _node_feature(gt, idx, include_stereo=include_stereo),
            atom.GetDegree(),
            _neighbor_signature(gt, idx, include_stereo=include_stereo),
        )
        candidates[idx] = list(pred_by_feature.get(feature, []))
    return candidates


def _find_isomorphism(
    gt: ParsedMarkushGraph,
    pred: ParsedMarkushGraph,
    *,
    include_stereo: bool,
    max_states: int,
) -> tuple[dict[int, int] | None, int]:
    assert gt.mol is not None and pred.mol is not None
    candidates = _candidate_map(gt, pred, include_stereo=include_stereo)
    if any(not values for values in candidates.values()):
        return None, 0

    order = sorted(
        range(gt.mol.GetNumAtoms()),
        key=lambda idx: (len(candidates[idx]), -gt.mol.GetAtomWithIdx(idx).GetDegree(), idx),
    )
    mapping: dict[int, int] = {}
    used_pred: set[int] = set()
    states = 0

    def compatible(gt_idx: int, pred_idx: int) -> bool:
        for mapped_gt_idx, mapped_pred_idx in mapping.items():
            gt_bond = gt.mol.GetBondBetweenAtoms(gt_idx, mapped_gt_idx)
            pred_bond = pred.mol.GetBondBetweenAtoms(pred_idx, mapped_pred_idx)
            if gt_bond is None or pred_bond is None:
                if gt_bond is not None or pred_bond is not None:
                    return False
                continue
            if _bond_feature(gt_bond, include_stereo=include_stereo) != _bond_feature(
                pred_bond, include_stereo=include_stereo
            ):
                return False
        return True

    def search(pos: int) -> dict[int, int] | None:
        nonlocal states
        states += 1
        if states > max_states:
            raise _SearchLimitExceededError(f"isomorphism search exceeded {max_states} states")
        if pos == len(order):
            return dict(mapping) if _extension_features_match(gt, pred, mapping) else None

        gt_idx = order[pos]
        for pred_idx in candidates[gt_idx]:
            if pred_idx in used_pred or not compatible(gt_idx, pred_idx):
                continue
            mapping[gt_idx] = pred_idx
            used_pred.add(pred_idx)
            found = search(pos + 1)
            if found is not None:
                return found
            used_pred.remove(pred_idx)
            del mapping[gt_idx]
        return None

    return search(0), states


def _signature_for_mapping(
    parsed: ParsedMarkushGraph,
    old_to_canonical: dict[int, int],
    *,
    include_stereo: bool,
    use_core_bond_stereo_fallback: bool = False,
) -> str:
    """Serialize one complete atom labeling into a comparable signature."""

    assert parsed.mol is not None
    n_atoms = parsed.mol.GetNumAtoms()
    atoms_by_canonical = [0] * n_atoms
    for old_idx, canonical_idx in old_to_canonical.items():
        atoms_by_canonical[canonical_idx] = old_idx

    atom_entries = tuple(
        _node_feature(parsed, old_idx, include_stereo=include_stereo)
        for old_idx in atoms_by_canonical
    )
    bond_entries = []
    for bond in parsed.mol.GetBonds():
        begin = old_to_canonical[bond.GetBeginAtomIdx()]
        end = old_to_canonical[bond.GetEndAtomIdx()]
        bond_entries.append(
            (
                min(begin, end),
                max(begin, end),
                _bond_feature(bond, include_stereo=include_stereo),
            )
        )
    fragment_entries = tuple(
        sorted(
            tuple(sorted(old_to_canonical[idx] for idx in fragment))
            for fragment in Chem.GetMolFrags(parsed.mol)
        )
    )
    payload = (
        "markush-canonical-v1",
        atom_entries,
        tuple(sorted(bond_entries)),
        fragment_entries,
        _m_signature(parsed.m_sections, old_to_canonical),
        _sg_signature(parsed.sg_sections, old_to_canonical),
        _bond_stereo_signature(
            _effective_bond_stereo_sections(
                parsed,
                use_core_fallback=use_core_bond_stereo_fallback,
            ),
            old_to_canonical,
        ),
        _component_group_signature(parsed.component_groups, old_to_canonical),
        _link_node_signature(parsed.link_nodes, old_to_canonical),
    )
    return repr(payload)


def _atom_groups(
    parsed: ParsedMarkushGraph,
    *,
    include_stereo: bool,
) -> list[list[int]]:
    """Return deterministic feature-compatible atom groups for canonical search."""

    assert parsed.mol is not None
    groups: dict[tuple[Any, ...], list[int]] = {}
    for atom in parsed.mol.GetAtoms():
        idx = atom.GetIdx()
        feature = (
            _node_feature(parsed, idx, include_stereo=include_stereo),
            atom.GetDegree(),
            _neighbor_signature(parsed, idx, include_stereo=include_stereo),
        )
        groups.setdefault(feature, []).append(idx)
    return [groups[key] for key in sorted(groups)]


def _canonical_markush_signature_from_parsed(
    parsed: ParsedMarkushGraph,
    *,
    include_stereo: bool,
    max_states: int,
    use_core_bond_stereo_fallback: bool = False,
) -> MarkushCanonicalResult:
    assert parsed.mol is not None

    groups = _atom_groups(parsed, include_stereo=include_stereo)
    state_budget = prod(factorial(len(group)) for group in groups)
    if state_budget > max_states:
        return MarkushCanonicalResult(
            False,
            error=f"canonical search would require {state_budget} states > {max_states}",
            states_visited=max_states,
        )

    best: str | None = None
    states = 0

    def search(group_idx: int, next_canonical_idx: int, mapping: dict[int, int]) -> None:
        nonlocal best, states
        if group_idx == len(groups):
            states += 1
            signature = _signature_for_mapping(
                parsed,
                mapping,
                include_stereo=include_stereo,
                use_core_bond_stereo_fallback=use_core_bond_stereo_fallback,
            )
            if best is None or signature < best:
                best = signature
            return

        group = groups[group_idx]
        for ordering in permutations(group):
            additions = {
                old_idx: next_canonical_idx + offset for offset, old_idx in enumerate(ordering)
            }
            mapping.update(additions)
            search(group_idx + 1, next_canonical_idx + len(group), mapping)
            for old_idx in additions:
                del mapping[old_idx]

    search(0, 0, {})
    return MarkushCanonicalResult(True, signature=best or "", states_visited=states)


def canonical_markush_signature(
    raw: Any,
    *,
    remove_stereo: bool = True,
    max_states: int = 200_000,
) -> MarkushCanonicalResult:
    """Return a deterministic canonical signature for one Markush structure.

    This is a Strategy-A style cross-check for the VF2-style equality path:
    atoms are partitioned by exact local features, all labelings within each
    feature-compatible partition are enumerated, and the lexicographically
    minimal full graph + annotation serialization is returned.
    """

    parsed = parse_markush_graph(raw, remove_stereo=remove_stereo)
    if not parsed.valid:
        return MarkushCanonicalResult(False, error=parsed.error)
    assert parsed.mol is not None

    return _canonical_markush_signature_from_parsed(
        parsed,
        include_stereo=not remove_stereo,
        max_states=max_states,
    )


def canonical_markush_equal(
    gt_raw: Any,
    pred_raw: Any,
    *,
    remove_stereo: bool = True,
    max_states: int = 200_000,
) -> MarkushEqualityResult:
    """Compare Markush structures by deterministic canonical signatures."""

    gt_parsed = parse_markush_graph(gt_raw, remove_stereo=remove_stereo)
    pred_parsed = parse_markush_graph(pred_raw, remove_stereo=remove_stereo)
    if not gt_parsed.valid or not pred_parsed.valid:
        return MarkushEqualityResult(
            equal=False,
            valid_gt=gt_parsed.valid,
            valid_pred=pred_parsed.valid,
            gt_error=gt_parsed.error,
            pred_error=pred_parsed.error,
        )

    include_stereo = not remove_stereo
    use_core_bond_stereo = bool(gt_parsed.bond_stereo_sections or pred_parsed.bond_stereo_sections)
    gt = _canonical_markush_signature_from_parsed(
        gt_parsed,
        include_stereo=include_stereo,
        max_states=max_states,
        use_core_bond_stereo_fallback=use_core_bond_stereo,
    )
    pred = _canonical_markush_signature_from_parsed(
        pred_parsed,
        include_stereo=include_stereo,
        max_states=max_states,
        use_core_bond_stereo_fallback=use_core_bond_stereo,
    )
    if not gt.valid or not pred.valid:
        return MarkushEqualityResult(
            equal=False,
            valid_gt=gt.valid,
            valid_pred=pred.valid,
            gt_error=gt.error,
            pred_error=pred.error,
            states_visited=max(gt.states_visited, pred.states_visited),
        )
    return MarkushEqualityResult(
        equal=gt.signature == pred.signature,
        valid_gt=True,
        valid_pred=True,
        states_visited=max(gt.states_visited, pred.states_visited),
    )


def _rdkit_canonical_markush_signature_from_parsed(
    parsed: ParsedMarkushGraph,
    *,
    include_stereo: bool,
    use_core_bond_stereo_fallback: bool = False,
) -> MarkushCanonicalResult:
    assert parsed.mol is not None

    mol = Chem.Mol(parsed.mol)
    markush_features = {
        atom.GetIdx(): (
            parsed.atom_labels.get(atom.GetIdx(), ""),
            _annotation_invariant(parsed, atom.GetIdx()),
        )
        for atom in parsed.mol.GetAtoms()
    }
    feature_codes = {
        feature: idx + 1
        for idx, feature in enumerate(sorted(set(markush_features.values()), key=repr))
    }
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(feature_codes[markush_features[atom.GetIdx()]])

    ranks = list(
        Chem.CanonicalRankAtoms(
            mol,
            breakTies=True,
            includeChirality=include_stereo,
            includeIsotopes=True,
            includeAtomMaps=True,
        )
    )
    atoms_by_canonical = sorted(range(mol.GetNumAtoms()), key=lambda idx: (ranks[idx], idx))
    old_to_canonical = {
        old_idx: canonical_idx for canonical_idx, old_idx in enumerate(atoms_by_canonical)
    }
    return MarkushCanonicalResult(
        True,
        signature=_signature_for_mapping(
            parsed,
            old_to_canonical,
            include_stereo=include_stereo,
            use_core_bond_stereo_fallback=use_core_bond_stereo_fallback,
        ),
        states_visited=1,
    )


def rdkit_canonical_markush_signature(
    raw: Any,
    *,
    remove_stereo: bool = True,
) -> MarkushCanonicalResult:
    """Return a Markush signature ordered by RDKit canonical atom ranks.

    RDKit owns the chemistry normalization and canonical ranking here. Temporary
    atom-map numbers encode only Markush-local label/annotation roles so RDKit's
    ranker sees the same distinctions the equality relation must preserve. The
    returned signature still serializes exact `m:`/`Sg:` signatures after
    remapping; RDKit is not asked to interpret those CXSMILES extensions.
    """

    parsed = parse_markush_graph(raw, remove_stereo=remove_stereo)
    if not parsed.valid:
        return MarkushCanonicalResult(False, error=parsed.error)
    return _rdkit_canonical_markush_signature_from_parsed(
        parsed,
        include_stereo=not remove_stereo,
    )


def rdkit_canonical_markush_equal(
    gt_raw: Any,
    pred_raw: Any,
    *,
    remove_stereo: bool = True,
) -> MarkushEqualityResult:
    """Compare Markush structures using RDKit canonical atom ranking."""

    gt_parsed = parse_markush_graph(gt_raw, remove_stereo=remove_stereo)
    pred_parsed = parse_markush_graph(pred_raw, remove_stereo=remove_stereo)
    if not gt_parsed.valid or not pred_parsed.valid:
        return MarkushEqualityResult(
            equal=False,
            valid_gt=gt_parsed.valid,
            valid_pred=pred_parsed.valid,
            gt_error=gt_parsed.error,
            pred_error=pred_parsed.error,
        )

    include_stereo = not remove_stereo
    use_core_bond_stereo = bool(gt_parsed.bond_stereo_sections or pred_parsed.bond_stereo_sections)
    gt = _rdkit_canonical_markush_signature_from_parsed(
        gt_parsed,
        include_stereo=include_stereo,
        use_core_bond_stereo_fallback=use_core_bond_stereo,
    )
    pred = _rdkit_canonical_markush_signature_from_parsed(
        pred_parsed,
        include_stereo=include_stereo,
        use_core_bond_stereo_fallback=use_core_bond_stereo,
    )
    if not gt.valid or not pred.valid:
        return MarkushEqualityResult(
            equal=False,
            valid_gt=gt.valid,
            valid_pred=pred.valid,
            gt_error=gt.error,
            pred_error=pred.error,
            states_visited=max(gt.states_visited, pred.states_visited),
        )
    return MarkushEqualityResult(
        equal=gt.signature == pred.signature,
        valid_gt=True,
        valid_pred=True,
        states_visited=max(gt.states_visited, pred.states_visited),
    )


def _to_networkx_graph(parsed: ParsedMarkushGraph, *, include_stereo: bool):
    """Build a NetworkX graph with exact atom/bond feature attributes."""

    import networkx as nx

    assert parsed.mol is not None
    graph = nx.Graph()
    for atom in parsed.mol.GetAtoms():
        idx = atom.GetIdx()
        graph.add_node(idx, feature=_node_feature(parsed, idx, include_stereo=include_stereo))
    for bond in parsed.mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            feature=_bond_feature(bond, include_stereo=include_stereo),
        )
    return graph


def networkx_markush_equal(
    gt_raw: Any,
    pred_raw: Any,
    *,
    remove_stereo: bool = True,
    max_states: int = 200_000,
) -> MarkushEqualityResult:
    """Compare Markush structures with NetworkX's VF2 GraphMatcher.

    This optional backend is for differential validation and benchmarking. It
    uses NetworkX for the graph-isomorphism search, then applies the same
    Markush-specific extension check as the in-repo backend.
    """

    try:
        from networkx.algorithms import isomorphism as nx_iso
    except ImportError:
        return MarkushEqualityResult(
            equal=False,
            valid_gt=False,
            valid_pred=False,
            error="networkx is not installed",
        )

    gt = parse_markush_graph(gt_raw, remove_stereo=remove_stereo)
    pred = parse_markush_graph(pred_raw, remove_stereo=remove_stereo)
    if not gt.valid or not pred.valid:
        return MarkushEqualityResult(
            equal=False,
            valid_gt=gt.valid,
            valid_pred=pred.valid,
            gt_error=gt.error,
            pred_error=pred.error,
        )

    assert gt.mol is not None and pred.mol is not None
    if gt.mol.GetNumAtoms() != pred.mol.GetNumAtoms():
        return MarkushEqualityResult(False, True, True)
    if gt.mol.GetNumBonds() != pred.mol.GetNumBonds():
        return MarkushEqualityResult(False, True, True)
    if len(Chem.GetMolFrags(gt.mol)) != len(Chem.GetMolFrags(pred.mol)):
        return MarkushEqualityResult(False, True, True)

    include_stereo = not remove_stereo
    gt_graph = _to_networkx_graph(gt, include_stereo=include_stereo)
    pred_graph = _to_networkx_graph(pred, include_stereo=include_stereo)
    matcher = nx_iso.GraphMatcher(
        gt_graph,
        pred_graph,
        node_match=lambda left, right: left["feature"] == right["feature"],
        edge_match=lambda left, right: left["feature"] == right["feature"],
    )
    states = 0
    for mapping in matcher.isomorphisms_iter():
        states += 1
        if states > max_states:
            return MarkushEqualityResult(
                equal=False,
                valid_gt=True,
                valid_pred=True,
                error=f"networkx isomorphism search exceeded {max_states} mappings",
                states_visited=max_states,
            )
        if _extension_features_match(gt, pred, mapping):
            return MarkushEqualityResult(True, True, True, states_visited=states)
    return MarkushEqualityResult(False, True, True, states_visited=states)


def markush_graph_equal(
    gt_raw: Any,
    pred_raw: Any,
    *,
    remove_stereo: bool = True,
    normalize_hydrogens: bool = False,
    max_states: int = 200_000,
) -> MarkushEqualityResult:
    """Return TRUE iff GT and prediction denote the same Markush structure."""

    gt = parse_markush_graph(gt_raw, remove_stereo=remove_stereo)
    pred = parse_markush_graph(pred_raw, remove_stereo=remove_stereo)
    if not gt.valid or not pred.valid:
        return MarkushEqualityResult(
            equal=False,
            valid_gt=gt.valid,
            valid_pred=pred.valid,
            gt_error=gt.error,
            pred_error=pred.error,
        )

    if normalize_hydrogens:
        gt = _without_removable_explicit_hydrogens(gt)
        pred = _without_removable_explicit_hydrogens(pred)

    assert gt.mol is not None and pred.mol is not None
    if gt.mol.GetNumAtoms() != pred.mol.GetNumAtoms():
        return MarkushEqualityResult(False, True, True)
    if gt.mol.GetNumBonds() != pred.mol.GetNumBonds():
        return MarkushEqualityResult(False, True, True)
    if len(Chem.GetMolFrags(gt.mol)) != len(Chem.GetMolFrags(pred.mol)):
        return MarkushEqualityResult(False, True, True)

    include_stereo = not remove_stereo
    try:
        mapping, states = _find_isomorphism(
            gt,
            pred,
            include_stereo=include_stereo,
            max_states=max_states,
        )
    except _SearchLimitExceededError as exc:
        return MarkushEqualityResult(
            equal=False,
            valid_gt=True,
            valid_pred=True,
            error=str(exc),
            states_visited=max_states,
        )
    return MarkushEqualityResult(
        equal=mapping is not None,
        valid_gt=True,
        valid_pred=True,
        states_visited=states,
    )


def grade(gt_raw: Any, pred_raw: Any, **kwargs: Any) -> dict[str, Any]:
    """Dictionary wrapper used by validation and future scoring integrations."""

    result = markush_graph_equal(gt_raw, pred_raw, **kwargs)
    return {
        "valid_gt": result.valid_gt,
        "valid_pred": result.valid_pred,
        "cxsmi_equality": result.equal,
        "cxsmiles_full_match": result.equal,
        "error": result.error or result.gt_error or result.pred_error,
        "states_visited": result.states_visited,
    }
