from __future__ import annotations

import pytest

from glyph.markush.data.molparser_esmiles import (
    molparser_esmiles_equal,
    molparser_invariance_variants,
    molparser_sensitivity_variants,
    networkx_available,
    networkx_molparser_esmiles_equal,
    parse_molparser_esmiles,
    rdkit_canonical_molparser_esmiles_equal,
    renumber_molparser_esmiles,
    serialize_molparser_esmiles,
)


def assert_native_equal(left: str, right: str) -> None:
    result = molparser_esmiles_equal(left, right)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.error == ""
    assert result.equal is True


def assert_native_not_equal(left: str, right: str) -> None:
    result = molparser_esmiles_equal(left, right)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.equal is False


def test_parse_empty_extension_keeps_sep_contract():
    parsed = parse_molparser_esmiles("CCO<sep>")

    assert parsed.valid, parsed.error
    assert parsed.core_smiles == "CCO"
    assert parsed.extension == ""
    assert parsed.tags == ()
    assert serialize_molparser_esmiles(parsed) == "CCO<sep>"


def test_parse_atom_group_and_dum_connection_point():
    parsed = parse_molparser_esmiles("C1(*)SCCS1<sep><a>1:<dum></a>")

    assert parsed.valid, parsed.error
    assert len(parsed.atom_groups) == 1
    group = parsed.atom_groups[0]
    assert group.atom_index == 1
    assert group.group_name == "<dum>"
    assert group.is_connection_point is True
    assert parsed.connection_points == (1,)


def test_parse_empty_atom_group_as_explicit_warning():
    parsed = parse_molparser_esmiles("*CC<sep><a>0:</a>")

    assert parsed.valid, parsed.error
    assert parsed.atom_groups[0].atom_index == 0
    assert parsed.atom_groups[0].group_name == ""
    assert "empty group name" in parsed.parse_warnings[0]


def test_parse_duplicate_identical_tag_as_explicit_warning():
    parsed = parse_molparser_esmiles("*CC<sep><a>0:R</a><a>0:R</a>")

    assert parsed.valid, parsed.error
    assert len(parsed.atom_groups) == 2
    assert "duplicate identical tag" in parsed.parse_warnings[0]


def test_parse_ring_and_circle_groups_with_modifier():
    parsed = parse_molparser_esmiles("c1ccccc1<sep><r>0:R?t</r><c>2:RingA</c>")

    assert parsed.valid, parsed.error
    assert len(parsed.ring_groups) == 1
    assert parsed.ring_groups[0].ring_index == 0
    assert parsed.ring_groups[0].atoms == frozenset(range(6))
    assert parsed.ring_groups[0].group_base == "R"
    assert parsed.ring_groups[0].modifier == "?t"
    assert parsed.circle_groups[0].circle_index == 2
    assert parsed.circle_groups[0].group_name == "RingA"


def test_parse_multiple_ring_groups_on_same_ring_as_native_annotations():
    parsed = parse_molparser_esmiles("c1ccccc1<sep><r>0:R</r><r>0:R2</r>")

    assert parsed.valid, parsed.error
    assert [group.group_name for group in parsed.ring_groups] == ["R", "R2"]
    assert {group.atoms for group in parsed.ring_groups} == {frozenset(range(6))}


@pytest.mark.parametrize(
    ("value", "error_part"),
    [
        ("CC", "missing <sep>"),
        ("CC<sep><a>9:R</a>", "atom index 9 out of range"),
        ("CC<sep><r>0:R</r>", "ring index 0 out of range"),
        ("CC<sep><a>bad:R</a>", "non-integer index"),
        ("CC<sep><a>0:R", "unsupported extension text"),
        ("CC<sep><a>0:R</a><a>0:R2</a>", "duplicate incompatible <a>"),
        ("CC<sep><c>0:A</c><c>0:B</c>", "duplicate incompatible <c>"),
    ],
)
def test_parse_invalid_cases_return_explicit_errors(value: str, error_part: str):
    parsed = parse_molparser_esmiles(value)

    assert not parsed.valid
    assert error_part in parsed.error


def test_native_equality_is_reflexive_and_cycle_consistent():
    value = "C1(*)SCCS1<sep><a>1:<dum></a>"
    parsed = parse_molparser_esmiles(value)
    serialized = serialize_molparser_esmiles(parsed, preserve_extension=False)

    assert_native_equal(value, value)
    assert_native_equal(value, serialized)


def test_native_equality_rejects_atom_group_on_distinct_atom():
    gt = "CCC<sep><a>0:R</a>"
    pred = "CCC<sep><a>1:R</a>"

    assert_native_not_equal(gt, pred)


def test_native_equality_accepts_atom_group_on_symmetric_atom():
    gt = "CCC<sep><a>0:R</a>"
    pred = "CCC<sep><a>2:R</a>"

    assert_native_equal(gt, pred)


def test_native_equality_rejects_group_name_change():
    gt = "CCC<sep><a>0:R</a>"
    pred = "CCC<sep><a>0:R2</a>"

    assert_native_not_equal(gt, pred)


def test_native_equality_rejects_ring_group_on_non_equivalent_ring():
    gt = "c1ccccc1.c1ncccc1<sep><r>0:R</r>"
    pred = "c1ccccc1.c1ncccc1<sep><r>1:R</r>"

    assert_native_not_equal(gt, pred)


def test_renumber_molparser_esmiles_remaps_atom_group_indices():
    value = "CCO<sep><a>2:OH</a>"
    variant = renumber_molparser_esmiles(value, [2, 1, 0])
    parsed_variant = parse_molparser_esmiles(variant)

    assert parsed_variant.valid, parsed_variant.error
    assert parsed_variant.atom_groups[0].atom_index == 0
    assert_native_equal(value, variant)


def test_renumber_molparser_esmiles_remaps_ring_group_indices():
    value = "c1ccccc1.C1CCCCC1<sep><r>0:Ar</r>"
    parsed_value = parse_molparser_esmiles(value)
    variant = renumber_molparser_esmiles(
        value, list(reversed(range(parsed_value.mol.GetNumAtoms())))
    )

    assert_native_equal(value, variant)


def test_molparser_invariance_variants_are_equal():
    value = "CCO.c1ccccc1<sep><a>2:OH</a><r>0:Ar</r>"
    generated, skipped = molparser_invariance_variants(value, variants=3)

    assert len(generated) == 3
    assert skipped == []
    for _name, variant in generated:
        assert_native_equal(value, variant)


def test_native_equality_preserves_rdkit_hydrogen_and_radical_state():
    assert_native_not_equal("[O]C<sep>", "OC<sep>")
    assert_native_not_equal("C[C@@](CC(CN(C)C)O)N<sep>", "CC(CC(CN(C)C)O)N<sep>")


def test_molparser_sensitivity_variants_are_valid_near_misses():
    value = "CCO.c1ccccc1<sep><a>2:OH</a><r>0:Ar</r><c>0:Cyc</c>"
    generated, skipped = molparser_sensitivity_variants(value, variants=5)

    assert len(generated) == 5
    assert all(name != "no_safe_edit" for name, _reason in skipped)
    for _name, variant in generated:
        assert_native_not_equal(value, variant)
        assert_native_equal(variant, variant)


def test_native_equality_rejects_circle_group_changes():
    value = "*CC<sep><a>0:R</a><c>0:RingA</c>"

    assert_native_not_equal(value, "*CC<sep><a>0:R</a><c>0:RingB</c>")
    assert_native_not_equal(value, "*CC<sep><a>0:R</a><c>1:RingA</c>")


@pytest.mark.skipif(not networkx_available(), reason="NetworkX is not installed")
def test_networkx_backend_agrees_on_known_native_cases():
    cases = [
        ("CCC<sep><a>0:R</a>", "CCC<sep><a>2:R</a>", True),
        ("CCC<sep><a>0:R</a>", "CCC<sep><a>1:R</a>", False),
        ("c1ccccc1<sep><r>0:R</r>", "c1ccccc1<sep><r>0:R</r>", True),
    ]

    for left, right, expected in cases:
        internal = molparser_esmiles_equal(left, right)
        networkx = networkx_molparser_esmiles_equal(left, right)

        assert internal.equal is expected
        assert networkx.valid_gt, networkx.gt_error
        assert networkx.valid_pred, networkx.pred_error
        assert networkx.error == ""
        assert networkx.equal is expected


def test_rdkit_canonical_backend_agrees_on_atom_annotation_cases():
    cases = [
        ("CCC<sep><a>0:R</a>", "CCC<sep><a>2:R</a>", True),
        ("CCC<sep><a>0:R</a>", "CCC<sep><a>1:R</a>", False),
        ("CCC<sep><a>0:R</a>", "CCC<sep><a>0:R2</a>", False),
        ("CCO<sep>", "OCC<sep>", True),
    ]

    for left, right, expected in cases:
        internal = molparser_esmiles_equal(left, right)
        canonical = rdkit_canonical_molparser_esmiles_equal(left, right)

        assert internal.equal is expected
        assert canonical.valid_gt, canonical.gt_error
        assert canonical.valid_pred, canonical.pred_error
        assert canonical.error == ""
        assert canonical.equal is expected


def test_rdkit_canonical_backend_reports_ring_and_circle_annotations_not_applicable():
    ring = rdkit_canonical_molparser_esmiles_equal(
        "c1ccccc1<sep><r>0:R</r>",
        "c1ccccc1<sep><r>0:R</r>",
    )
    circle = rdkit_canonical_molparser_esmiles_equal(
        "*CC<sep><a>0:R</a><c>0:RingA</c>",
        "*CC<sep><a>0:R</a><c>0:RingA</c>",
    )

    assert ring.valid_gt is True
    assert ring.valid_pred is True
    assert "not applicable" in ring.error
    assert circle.valid_gt is True
    assert circle.valid_pred is True
    assert "not applicable" in circle.error
