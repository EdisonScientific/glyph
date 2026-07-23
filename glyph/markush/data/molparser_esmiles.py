"""Parser and native equality helpers for MolParser E-SMILES.

MolParser captions use ``SMILES<sep>EXTENSION``.  The SMILES prefix is parsed
with RDKit in its original atom order; the extension contains XML-like tags
whose indices refer to that original prefix.  This module keeps those
annotations explicit so later conversion to shared CXSMILES can be audited
instead of inferred from strings.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, RDLogger

from glyph.markush.data.canonicalize import clean_smiles_brackets

RDLogger.DisableLog("rdApp.*")

SEP = "<sep>"
_TAG_RE = re.compile(r"<([arc])>(.*?)</\1>")


@dataclass(frozen=True)
class MolParserTag:
    """One raw MolParser extension tag."""

    kind: str
    index: int
    group_name: str
    group_base: str
    modifier: str
    raw: str

    @property
    def is_connection_point(self) -> bool:
        return self.group_name == "<dum>"


@dataclass(frozen=True)
class AtomGroup:
    """An ``<a>`` group bound to one original SMILES atom index."""

    atom_index: int
    group_name: str
    group_base: str
    modifier: str
    raw: str
    is_connection_point: bool = False


@dataclass(frozen=True)
class RingGroup:
    """An ``<r>`` group bound to one original RDKit ring atom set."""

    ring_index: int
    atoms: frozenset[int]
    group_name: str
    group_base: str
    modifier: str
    raw: str


@dataclass(frozen=True)
class CircleGroup:
    """A ``<c>`` abstract ring annotation."""

    circle_index: int
    group_name: str
    group_base: str
    modifier: str
    raw: str


@dataclass
class MolParserESmiles:
    """Parsed MolParser E-SMILES with source-indexed annotations."""

    raw: str
    core_smiles: str = ""
    extension: str = ""
    mol: Any | None = None
    ring_atom_sets: tuple[frozenset[int], ...] = ()
    tags: tuple[MolParserTag, ...] = ()
    atom_groups: tuple[AtomGroup, ...] = ()
    ring_groups: tuple[RingGroup, ...] = ()
    circle_groups: tuple[CircleGroup, ...] = ()
    connection_points: tuple[int, ...] = ()
    parse_warnings: tuple[str, ...] = ()
    valid: bool = False
    error: str = ""


@dataclass(frozen=True)
class MolParserEqualityResult:
    """A non-throwing native E-SMILES equality verdict."""

    equal: bool
    valid_gt: bool
    valid_pred: bool
    gt_error: str = ""
    pred_error: str = ""
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


def _split_group_modifier(group_name: str) -> tuple[str, str]:
    text = group_name.strip()
    if "?" not in text or text == "<dum>":
        return text, ""
    base, modifier = text.split("?", 1)
    return base, "?" + modifier


def _parse_tag_body(kind: str, body: str, raw: str) -> tuple[MolParserTag | None, str]:
    if ":" not in body:
        return None, f"{raw}: missing ':' between index and group"
    index_text, group_name = body.split(":", 1)
    index_text = index_text.strip()
    group_name = group_name.strip()
    if not index_text:
        return None, f"{raw}: missing index"
    try:
        index = int(index_text)
    except ValueError:
        return None, f"{raw}: non-integer index {index_text!r}"
    if index < 0:
        return None, f"{raw}: negative index {index}"
    group_base, modifier = _split_group_modifier(group_name)
    return MolParserTag(kind, index, group_name, group_base, modifier, raw), ""


def _parse_core_mol(core: str, *, remove_stereo: bool) -> tuple[Any | None, str, tuple[str, ...]]:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.strictCXSMILES = False
    try:
        mol = Chem.MolFromSmiles(core, params)
    except Exception as exc:
        mol = None
        sanitized_error = str(exc)
    else:
        sanitized_error = ""
    if mol is None:
        params.sanitize = False
        try:
            mol = Chem.MolFromSmiles(core, params)
        except Exception as exc:
            return None, f"SMILES core does not parse: {exc}", ()
        if mol is None:
            return None, "SMILES core does not parse", ()
        warning = (
            f"core parsed with RDKit sanitize=False after sanitized parse failed: {sanitized_error}"
            if sanitized_error
            else "core parsed with RDKit sanitize=False after sanitized parse failed"
        )
        warnings = (warning,)
    else:
        warnings = ()
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
        for bond in mol.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    return mol, "", warnings


def parse_molparser_esmiles(
    raw: Any,
    *,
    require_sep: bool = True,
    remove_stereo: bool = True,
) -> MolParserESmiles:
    """Parse one MolParser ``SMILES<sep>EXTENSION`` value.

    The returned object is invalid, rather than raising, when the core SMILES or
    extension cannot be interpreted.
    """

    text = "" if raw is None else str(raw).strip()
    parsed = MolParserESmiles(raw=text)
    if not text:
        parsed.error = "empty or non-string MolParser E-SMILES value"
        return parsed
    if SEP not in text:
        if require_sep:
            parsed.error = "missing <sep> separator"
            return parsed
        core, extension = text, ""
    else:
        core, extension = text.split(SEP, 1)
    core = core.strip()
    extension = extension.strip()
    if not core:
        parsed.error = "missing core SMILES before <sep>"
        return parsed

    mol, error, core_warnings = _parse_core_mol(core, remove_stereo=remove_stereo)
    if error:
        parsed.error = error
        return parsed
    assert mol is not None
    ring_atom_sets = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())

    warnings: list[str] = list(core_warnings)
    tags: list[MolParserTag] = []
    atom_groups: list[AtomGroup] = []
    ring_groups: list[RingGroup] = []
    circle_groups: list[CircleGroup] = []
    pos = 0
    for match in _TAG_RE.finditer(extension):
        unknown = extension[pos : match.start()].strip()
        if unknown:
            parsed.error = f"unsupported extension text: {unknown!r}"
            return parsed
        pos = match.end()
        raw_tag = match.group(0)
        tag, error = _parse_tag_body(match.group(1), match.group(2).strip(), raw_tag)
        if error:
            parsed.error = error
            return parsed
        assert tag is not None
        if tag.group_name == "":
            warnings.append(f"{raw_tag}: empty group name")
        tags.append(tag)
        if tag.kind == "a":
            if tag.index >= mol.GetNumAtoms():
                parsed.error = f"{raw_tag}: atom index {tag.index} out of range"
                return parsed
            atom_groups.append(
                AtomGroup(
                    atom_index=tag.index,
                    group_name=tag.group_name,
                    group_base=tag.group_base,
                    modifier=tag.modifier,
                    raw=tag.raw,
                    is_connection_point=tag.is_connection_point,
                )
            )
        elif tag.kind == "r":
            if tag.index >= len(ring_atom_sets):
                parsed.error = f"{raw_tag}: ring index {tag.index} out of range"
                return parsed
            ring_groups.append(
                RingGroup(
                    ring_index=tag.index,
                    atoms=ring_atom_sets[tag.index],
                    group_name=tag.group_name,
                    group_base=tag.group_base,
                    modifier=tag.modifier,
                    raw=tag.raw,
                )
            )
        elif tag.kind == "c":
            circle_groups.append(
                CircleGroup(
                    circle_index=tag.index,
                    group_name=tag.group_name,
                    group_base=tag.group_base,
                    modifier=tag.modifier,
                    raw=tag.raw,
                )
            )
    trailing = extension[pos:].strip()
    if trailing:
        parsed.error = f"unsupported extension text: {trailing!r}"
        return parsed

    duplicate_counts = Counter((tag.kind, tag.index, tag.group_name) for tag in tags)
    for key, count in duplicate_counts.items():
        if count > 1:
            warnings.append(f"duplicate identical tag {key!r} occurs {count} times")
    binding_names: dict[tuple[str, int], set[str]] = {}
    for tag in tags:
        binding_names.setdefault((tag.kind, tag.index), set()).add(tag.group_name)
    for key, group_names in sorted(binding_names.items()):
        kind, index = key
        if kind in {"a", "c"} and len(group_names) > 1:
            names = ", ".join(repr(name) for name in sorted(group_names))
            parsed.error = f"duplicate incompatible <{kind}> annotations at index {index}: {names}"
            return parsed

    parsed.core_smiles = core
    parsed.extension = extension
    parsed.mol = mol
    parsed.ring_atom_sets = ring_atom_sets
    parsed.tags = tuple(tags)
    parsed.atom_groups = tuple(atom_groups)
    parsed.ring_groups = tuple(ring_groups)
    parsed.circle_groups = tuple(circle_groups)
    parsed.connection_points = tuple(
        group.atom_index for group in atom_groups if group.is_connection_point
    )
    parsed.parse_warnings = tuple(warnings)
    parsed.valid = True
    return parsed


def serialize_molparser_esmiles(
    parsed: MolParserESmiles, *, preserve_extension: bool = True
) -> str:
    """Serialize a valid parsed E-SMILES value back to MolParser text."""

    if not parsed.valid:
        raise ValueError(parsed.error or "cannot serialize invalid MolParser E-SMILES")
    extension = parsed.extension if preserve_extension else "".join(tag.raw for tag in parsed.tags)
    return f"{parsed.core_smiles}{SEP}{extension}"


def _strip_atom_maps(smiles: str) -> str:
    return re.sub(r":\d+(?=\])", "", smiles)


def _tag_text(kind: str, index: int, group_name: str) -> str:
    return f"<{kind}>{index}:{group_name}</{kind}>"


def _make_tag(kind: str, index: int, group_name: str) -> MolParserTag:
    group_base, modifier = _split_group_modifier(group_name)
    return MolParserTag(
        kind=kind,
        index=index,
        group_name=group_name,
        group_base=group_base,
        modifier=modifier,
        raw=_tag_text(kind, index, group_name),
    )


def _with_tags(parsed: MolParserESmiles, tags: list[MolParserTag]) -> str:
    return f"{parsed.core_smiles}{SEP}{''.join(tag.raw for tag in tags)}"


def _parse_smiles_with_maps(smiles: str) -> Any | None:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.strictCXSMILES = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is not None:
        return mol
    params.sanitize = False
    return Chem.MolFromSmiles(smiles, params)


def renumber_molparser_esmiles(
    raw: Any,
    new_order: list[int],
    *,
    remove_stereo: bool = False,
) -> str:
    """Rewrite atom order and remap MolParser extension indices.

    RDKit is used to perform the atom renumbering, but atom map numbers are
    temporarily assigned before serialization.  Parsing the mapped serialization
    gives the concrete ``old_atom_index -> output_atom_index`` map needed by
    MolParser extension tags.  Ring tags are remapped by carrying the original
    ring atom set through that map and finding the corresponding ring in the
    output molecule.
    """

    parsed = parse_molparser_esmiles(raw, remove_stereo=remove_stereo)
    if not parsed.valid:
        raise ValueError(parsed.error or "invalid MolParser E-SMILES")
    assert parsed.mol is not None
    if len(new_order) != parsed.mol.GetNumAtoms():
        raise ValueError("renumber order length does not match atom count")
    if sorted(new_order) != list(range(parsed.mol.GetNumAtoms())):
        raise ValueError("renumber order must be a permutation of atom indices")

    mol = Chem.Mol(parsed.mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    renumbered = Chem.RenumberAtoms(mol, new_order)
    smiles_with_maps = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=True)
    mapped_mol = _parse_smiles_with_maps(smiles_with_maps)
    if mapped_mol is None:
        raise ValueError("renumbered mapped SMILES does not parse")

    old_to_output = {
        atom.GetAtomMapNum() - 1: atom.GetIdx()
        for atom in mapped_mol.GetAtoms()
        if atom.GetAtomMapNum()
    }
    expected = set(range(parsed.mol.GetNumAtoms()))
    if set(old_to_output) != expected:
        missing = sorted(expected - set(old_to_output))
        raise ValueError(f"renumbered mapped SMILES lost atom-map indices: {missing}")

    output_ring_sets = tuple(frozenset(ring) for ring in mapped_mol.GetRingInfo().AtomRings())
    core_out = clean_smiles_brackets(_strip_atom_maps(smiles_with_maps))
    out_tags: list[MolParserTag] = []
    for tag in parsed.tags:
        if tag.kind == "a":
            out_tags.append(_make_tag(tag.kind, old_to_output[tag.index], tag.group_name))
        elif tag.kind == "r":
            original_ring = parsed.ring_atom_sets[tag.index]
            mapped_ring = frozenset(old_to_output[atom] for atom in original_ring)
            matches = [
                ring_index
                for ring_index, ring_atoms in enumerate(output_ring_sets)
                if ring_atoms == mapped_ring
            ]
            if not matches:
                raise ValueError(f"renumbered output lost ring atom set for tag {tag.raw}")
            out_tags.append(_make_tag(tag.kind, matches[0], tag.group_name))
        elif tag.kind == "c":
            out_tags.append(_make_tag(tag.kind, tag.index, tag.group_name))
        else:
            raise ValueError(f"unsupported MolParser tag kind {tag.kind!r}")
    return f"{core_out}{SEP}{''.join(tag.raw for tag in out_tags)}"


def molparser_invariance_variants(
    raw: Any,
    *,
    variants: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Generate MolParser E-SMILES variants that should compare equal."""

    parsed = parse_molparser_esmiles(raw, remove_stereo=False)
    if not parsed.valid:
        raise ValueError(parsed.error or "invalid MolParser E-SMILES")
    assert parsed.mol is not None
    n_atoms = parsed.mol.GetNumAtoms()
    if n_atoms == 0:
        return [], [("no_atoms", "core molecule has no atoms")]

    candidate_orders: list[tuple[str, list[int]]] = [
        ("identity_round_trip", list(range(n_atoms))),
        ("reverse_atom_order", list(reversed(range(n_atoms)))),
    ]
    if n_atoms > 2:
        candidate_orders.append(("rotate_atom_order", [*range(1, n_atoms), 0]))

    generated: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    seen_orders: set[tuple[int, ...]] = set()
    for name, order in candidate_orders:
        if len(generated) >= max(1, variants):
            break
        order_key = tuple(order)
        if order_key in seen_orders:
            continue
        seen_orders.add(order_key)
        try:
            variant = renumber_molparser_esmiles(raw, order)
        except Exception as exc:
            skipped.append((name, str(exc)))
            continue
        result = molparser_esmiles_equal(raw, variant)
        if result.equal and result.valid_gt and result.valid_pred:
            generated.append((name, variant))
        else:
            skipped.append(
                (
                    name,
                    result.gt_error
                    or result.pred_error
                    or result.error
                    or "RDKit atom-order rewrite did not preserve native E-SMILES semantics",
                )
            )
    return generated, skipped


def _safe_atom_metric(atom: Any, method_name: str) -> int:
    try:
        return int(getattr(atom, method_name)())
    except Exception:
        return 0


def _core_atom_signature(parsed: MolParserESmiles, atom_idx: int) -> tuple[Any, ...]:
    assert parsed.mol is not None
    atom = parsed.mol.GetAtomWithIdx(atom_idx)
    neighbors = []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        neighbors.append(
            (
                str(bond.GetBondType()),
                other.GetAtomicNum(),
                other.GetIsAromatic(),
                other.GetFormalCharge(),
                _safe_atom_metric(other, "GetNumRadicalElectrons"),
                _safe_atom_metric(other, "GetNumImplicitHs"),
                other.GetDegree(),
            )
        )
    return (
        atom.GetAtomicNum(),
        atom.GetIsAromatic(),
        atom.GetFormalCharge(),
        atom.GetIsotope(),
        _safe_atom_metric(atom, "GetNumRadicalElectrons"),
        _safe_atom_metric(atom, "GetNumImplicitHs"),
        atom.GetDegree(),
        tuple(sorted(neighbors)),
    )


def _alternate_group_name(group_name: str) -> str:
    if not group_name:
        return "ALT"
    candidate = f"{group_name}_ALT"
    return candidate if candidate != group_name else f"{group_name}_ALT2"


def molparser_sensitivity_variants(
    raw: Any,
    *,
    variants: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Generate valid semantic edits that should compare unequal."""

    parsed = parse_molparser_esmiles(raw, remove_stereo=False)
    if not parsed.valid:
        raise ValueError(parsed.error or "invalid MolParser E-SMILES")
    assert parsed.mol is not None
    n_atoms = parsed.mol.GetNumAtoms()
    tags = list(parsed.tags)
    generated: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    signatures = {idx: _core_atom_signature(parsed, idx) for idx in range(n_atoms)}

    def add_variant(name: str, edited_tags: list[MolParserTag]) -> None:
        if len(generated) >= max(1, variants):
            return
        edited = _with_tags(parsed, edited_tags)
        result = molparser_esmiles_equal(raw, edited)
        if not result.valid_gt or not result.valid_pred:
            skipped.append(
                (
                    name,
                    result.gt_error or result.pred_error or result.error or "invalid edit",
                )
            )
            return
        if result.equal:
            skipped.append((name, "semantic edit compared equal, likely by symmetry"))
            return
        reflexive = molparser_esmiles_equal(edited, edited)
        if not reflexive.equal:
            skipped.append(
                (
                    name,
                    reflexive.gt_error
                    or reflexive.pred_error
                    or reflexive.error
                    or "edited variant is not reflexive",
                )
            )
            return
        generated.append((name, edited))

    for pos, tag in enumerate(tags):
        if tag.kind != "a":
            continue
        destinations = sorted(
            (
                idx
                for idx in range(n_atoms)
                if idx != tag.index and signatures[idx] != signatures[tag.index]
            ),
            key=lambda idx: (signatures[idx], idx),
        )
        for dst in destinations:
            edited_tags = list(tags)
            edited_tags[pos] = _make_tag("a", dst, tag.group_name)
            add_variant("move_atom_group_to_distinct_atom", edited_tags)
            break
        break

    for pos, tag in enumerate(tags):
        if tag.kind == "a":
            edited_tags = list(tags)
            edited_tags[pos] = _make_tag("a", tag.index, _alternate_group_name(tag.group_name))
            add_variant("change_atom_group_name", edited_tags)
            break

    for pos, tag in enumerate(tags):
        if tag.kind != "r":
            continue
        ring_signatures = {
            idx: tuple(sorted(signatures[atom] for atom in atoms))
            for idx, atoms in enumerate(parsed.ring_atom_sets)
        }
        destinations = sorted(
            (
                idx
                for idx in range(len(parsed.ring_atom_sets))
                if idx != tag.index and ring_signatures[idx] != ring_signatures[tag.index]
            ),
            key=lambda idx: (ring_signatures[idx], idx),
        )
        for dst in destinations:
            edited_tags = list(tags)
            edited_tags[pos] = _make_tag("r", dst, tag.group_name)
            add_variant("move_ring_group_to_distinct_ring", edited_tags)
            break
        break

    for pos, tag in enumerate(tags):
        if tag.kind == "r":
            edited_tags = list(tags)
            edited_tags[pos] = _make_tag("r", tag.index, _alternate_group_name(tag.group_name))
            add_variant("change_ring_group_name", edited_tags)
            break

    for pos, tag in enumerate(tags):
        if tag.kind == "c":
            edited_tags = list(tags)
            edited_tags[pos] = _make_tag("c", tag.index, _alternate_group_name(tag.group_name))
            add_variant("change_circle_group_name", edited_tags)
            break

    if tags:
        edited_tags = list(tags)
        edited_tags.pop()
        add_variant("drop_extension_tag", edited_tags)

    if n_atoms > 0:
        edited_tags = list(tags)
        edited_tags.append(_make_tag("a", 0, "ADDED"))
        add_variant("add_atom_group_annotation", edited_tags)

    if not generated and not skipped:
        skipped.append(("no_safe_edit", "no valid semantic edit could be generated"))
    return generated, skipped


def _atom_feature(
    parsed: MolParserESmiles, atom_idx: int, *, include_stereo: bool
) -> tuple[Any, ...]:
    assert parsed.mol is not None
    atom = parsed.mol.GetAtomWithIdx(atom_idx)
    chiral = str(atom.GetChiralTag()) if include_stereo else ""
    return (
        atom.GetAtomicNum(),
        atom.GetIsAromatic(),
        atom.GetFormalCharge(),
        atom.GetIsotope(),
        atom.GetNumExplicitHs(),
        _safe_atom_metric(atom, "GetNumImplicitHs"),
        _safe_atom_metric(atom, "GetNumRadicalElectrons"),
        chiral,
    )


def _bond_feature(bond: Any, *, include_stereo: bool) -> tuple[Any, ...]:
    stereo = str(bond.GetStereo()) if include_stereo else ""
    return (
        str(bond.GetBondType()),
        bond.GetIsAromatic(),
        bond.GetIsConjugated(),
        stereo,
    )


def _atom_annotation(parsed: MolParserESmiles, atom_idx: int) -> tuple[Any, ...]:
    atom_tags = [
        (
            group.group_name,
            group.group_base,
            group.modifier,
            group.is_connection_point,
        )
        for group in parsed.atom_groups
        if group.atom_index == atom_idx
    ]
    ring_roles = [
        (
            "ring",
            group.group_name,
            group.group_base,
            group.modifier,
            len(group.atoms),
        )
        for group in parsed.ring_groups
        if atom_idx in group.atoms
    ]
    return tuple(sorted([*atom_tags, *ring_roles]))


def _node_feature(
    parsed: MolParserESmiles, atom_idx: int, *, include_stereo: bool
) -> tuple[Any, ...]:
    return (
        _atom_feature(parsed, atom_idx, include_stereo=include_stereo),
        _atom_annotation(parsed, atom_idx),
    )


def _neighbor_signature(
    parsed: MolParserESmiles, atom_idx: int, *, include_stereo: bool
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


def _candidate_map(
    gt: MolParserESmiles,
    pred: MolParserESmiles,
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


def _atom_group_signature(
    groups: tuple[AtomGroup, ...], mapping: dict[int, int]
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                mapping[group.atom_index],
                group.group_name,
                group.group_base,
                group.modifier,
                group.is_connection_point,
            )
            for group in groups
        )
    )


def _ring_group_signature(
    groups: tuple[RingGroup, ...], mapping: dict[int, int]
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                tuple(sorted(mapping[atom] for atom in group.atoms)),
                group.group_name,
                group.group_base,
                group.modifier,
            )
            for group in groups
        )
    )


def _circle_group_signature(groups: tuple[CircleGroup, ...]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                group.circle_index,
                group.group_name,
                group.group_base,
                group.modifier,
            )
            for group in groups
        )
    )


def _extension_features_match(
    gt: MolParserESmiles,
    pred: MolParserESmiles,
    gt_to_pred: dict[int, int],
) -> bool:
    assert pred.mol is not None
    pred_identity = {idx: idx for idx in range(pred.mol.GetNumAtoms())}
    return (
        _atom_group_signature(gt.atom_groups, gt_to_pred)
        == _atom_group_signature(pred.atom_groups, pred_identity)
        and _ring_group_signature(gt.ring_groups, gt_to_pred)
        == _ring_group_signature(pred.ring_groups, pred_identity)
        and _circle_group_signature(gt.circle_groups) == _circle_group_signature(pred.circle_groups)
    )


def _find_isomorphism(
    gt: MolParserESmiles,
    pred: MolParserESmiles,
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


def molparser_esmiles_equal(
    gt_raw: Any,
    pred_raw: Any,
    *,
    remove_stereo: bool = True,
    max_states: int = 200_000,
) -> MolParserEqualityResult:
    """Return true iff two MolParser E-SMILES strings denote the same native graph."""

    gt = parse_molparser_esmiles(gt_raw, remove_stereo=remove_stereo)
    pred = parse_molparser_esmiles(pred_raw, remove_stereo=remove_stereo)
    if not gt.valid or not pred.valid:
        return MolParserEqualityResult(
            equal=False,
            valid_gt=gt.valid,
            valid_pred=pred.valid,
            gt_error=gt.error,
            pred_error=pred.error,
        )

    assert gt.mol is not None and pred.mol is not None
    if gt.mol.GetNumAtoms() != pred.mol.GetNumAtoms():
        return MolParserEqualityResult(False, True, True)
    if gt.mol.GetNumBonds() != pred.mol.GetNumBonds():
        return MolParserEqualityResult(False, True, True)
    if len(Chem.GetMolFrags(gt.mol)) != len(Chem.GetMolFrags(pred.mol)):
        return MolParserEqualityResult(False, True, True)

    include_stereo = not remove_stereo
    try:
        mapping, states = _find_isomorphism(
            gt,
            pred,
            include_stereo=include_stereo,
            max_states=max_states,
        )
    except _SearchLimitExceededError as exc:
        return MolParserEqualityResult(
            equal=False,
            valid_gt=True,
            valid_pred=True,
            error=str(exc),
            states_visited=max_states,
        )
    return MolParserEqualityResult(
        equal=mapping is not None,
        valid_gt=True,
        valid_pred=True,
        states_visited=states,
    )


def networkx_molparser_esmiles_equal(
    gt_raw: Any,
    pred_raw: Any,
    *,
    remove_stereo: bool = True,
    max_states: int = 200_000,
) -> MolParserEqualityResult:
    """Compare native E-SMILES with NetworkX VF2 as a third-party check."""

    try:
        import networkx as nx
        from networkx.algorithms import isomorphism as nx_iso
    except ImportError:
        return MolParserEqualityResult(
            equal=False,
            valid_gt=False,
            valid_pred=False,
            error="networkx is not installed",
        )

    gt = parse_molparser_esmiles(gt_raw, remove_stereo=remove_stereo)
    pred = parse_molparser_esmiles(pred_raw, remove_stereo=remove_stereo)
    if not gt.valid or not pred.valid:
        return MolParserEqualityResult(
            equal=False,
            valid_gt=gt.valid,
            valid_pred=pred.valid,
            gt_error=gt.error,
            pred_error=pred.error,
        )

    assert gt.mol is not None and pred.mol is not None
    if gt.mol.GetNumAtoms() != pred.mol.GetNumAtoms():
        return MolParserEqualityResult(False, True, True)
    if gt.mol.GetNumBonds() != pred.mol.GetNumBonds():
        return MolParserEqualityResult(False, True, True)
    if len(Chem.GetMolFrags(gt.mol)) != len(Chem.GetMolFrags(pred.mol)):
        return MolParserEqualityResult(False, True, True)

    include_stereo = not remove_stereo

    def to_graph(parsed: MolParserESmiles):
        graph = nx.Graph()
        assert parsed.mol is not None
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

    matcher = nx_iso.GraphMatcher(
        to_graph(gt),
        to_graph(pred),
        node_match=lambda left, right: left["feature"] == right["feature"],
        edge_match=lambda left, right: left["feature"] == right["feature"],
    )
    states = 0
    for mapping in matcher.isomorphisms_iter():
        states += 1
        if states > max_states:
            return MolParserEqualityResult(
                equal=False,
                valid_gt=True,
                valid_pred=True,
                error=f"networkx isomorphism search exceeded {max_states} mappings",
                states_visited=max_states,
            )
        if _extension_features_match(gt, pred, mapping):
            return MolParserEqualityResult(True, True, True, states_visited=states)
    return MolParserEqualityResult(False, True, True, states_visited=states)


def _atom_label_keys_for_pair(
    left: MolParserESmiles,
    right: MolParserESmiles,
) -> dict[tuple[Any, ...], int]:
    keys: set[tuple[Any, ...]] = set()
    for parsed in (left, right):
        assert parsed.mol is not None
        for atom_idx in range(parsed.mol.GetNumAtoms()):
            key = _atom_group_signature_for_atom(parsed, atom_idx)
            if key:
                keys.add(key)
    return {key: idx + 1 for idx, key in enumerate(sorted(keys))}


def _atom_group_signature_for_atom(parsed: MolParserESmiles, atom_idx: int) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                group.group_name,
                group.group_base,
                group.modifier,
                group.is_connection_point,
            )
            for group in parsed.atom_groups
            if group.atom_index == atom_idx
        )
    )


def _rdkit_canonical_signature(
    parsed: MolParserESmiles,
    label_to_atom_map: dict[tuple[Any, ...], int],
    *,
    include_stereo: bool,
) -> str:
    assert parsed.mol is not None
    mol = Chem.Mol(parsed.mol)
    for atom in mol.GetAtoms():
        key = _atom_group_signature_for_atom(parsed, atom.GetIdx())
        atom.SetAtomMapNum(label_to_atom_map.get(key, 0))
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=include_stereo,
        allBondsExplicit=True,
        allHsExplicit=True,
    )


def rdkit_canonical_molparser_esmiles_equal(
    gt_raw: Any,
    pred_raw: Any,
    *,
    remove_stereo: bool = True,
) -> MolParserEqualityResult:
    """Compare applicable native E-SMILES rows with RDKit canonicalization.

    This backend is a differential check, not the primary equality definition.
    It is applicable when all MolParser annotations can be encoded as atom-map
    invariants: atom groups only, no ring groups, and no circle groups.  Ring
    and circle annotations are skipped because RDKit canonical atom ranking does
    not natively represent those set/abstract annotations without adding
    non-chemical graph structure.
    """

    gt = parse_molparser_esmiles(gt_raw, remove_stereo=remove_stereo)
    pred = parse_molparser_esmiles(pred_raw, remove_stereo=remove_stereo)
    if not gt.valid or not pred.valid:
        return MolParserEqualityResult(
            equal=False,
            valid_gt=gt.valid,
            valid_pred=pred.valid,
            gt_error=gt.error,
            pred_error=pred.error,
        )

    if gt.ring_groups or pred.ring_groups or gt.circle_groups or pred.circle_groups:
        return MolParserEqualityResult(
            equal=False,
            valid_gt=True,
            valid_pred=True,
            error="rdkit canonical backend not applicable to ring or circle annotations",
        )

    assert gt.mol is not None and pred.mol is not None
    if gt.mol.GetNumAtoms() != pred.mol.GetNumAtoms():
        return MolParserEqualityResult(False, True, True)
    if gt.mol.GetNumBonds() != pred.mol.GetNumBonds():
        return MolParserEqualityResult(False, True, True)
    if len(Chem.GetMolFrags(gt.mol)) != len(Chem.GetMolFrags(pred.mol)):
        return MolParserEqualityResult(False, True, True)

    label_to_atom_map = _atom_label_keys_for_pair(gt, pred)
    include_stereo = not remove_stereo
    try:
        gt_signature = _rdkit_canonical_signature(
            gt,
            label_to_atom_map,
            include_stereo=include_stereo,
        )
        pred_signature = _rdkit_canonical_signature(
            pred,
            label_to_atom_map,
            include_stereo=include_stereo,
        )
    except Exception as exc:
        return MolParserEqualityResult(False, True, True, error=str(exc))
    return MolParserEqualityResult(
        equal=gt_signature == pred_signature,
        valid_gt=True,
        valid_pred=True,
    )


def molparser_value_taxonomy(values: Iterable[Any]) -> dict[str, Any]:
    """Return parser/taxonomy counters for a sequence of raw E-SMILES values."""

    tag_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    modifier_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    rows_with_dum = 0
    valid = 0
    invalid = 0
    total = 0
    for value in values:
        total += 1
        parsed = parse_molparser_esmiles(value)
        if not parsed.valid:
            invalid += 1
            error_counts[parsed.error] += 1
            continue
        valid += 1
        if any(tag.is_connection_point for tag in parsed.tags):
            rows_with_dum += 1
        for tag in parsed.tags:
            tag_counts[tag.kind] += 1
            group_counts[tag.group_name] += 1
            if tag.modifier:
                modifier_counts[tag.modifier] += 1
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "rows_with_dum": rows_with_dum,
        "tag_counts": dict(sorted(tag_counts.items())),
        "top_groups": dict(group_counts.most_common(50)),
        "modifier_counts": dict(sorted(modifier_counts.items())),
        "error_counts": dict(error_counts.most_common()),
        "networkx_available": networkx_available(),
    }
