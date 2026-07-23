from __future__ import annotations

from glyph.markush.data.molparser_convert import convert_molparser_esmiles_to_cxsmiles
from glyph.markush.data.molparser_esmiles import (
    molparser_invariance_variants,
    molparser_sensitivity_variants,
)
from glyph.markush.data.pipeline import opt_to_standard_cxsmiles, standard_to_opt_cxsmiles
from glyph.markush.eval.markush_graph import markush_graph_equal


def assert_shared_equal(left: str, right: str) -> None:
    result = markush_graph_equal(left, right)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.error == ""
    assert result.equal is True


def assert_shared_not_equal(left: str, right: str) -> None:
    result = markush_graph_equal(left, right)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.equal is False


def test_convert_simple_atom_groups_to_standard_and_opt_cxsmiles():
    result = convert_molparser_esmiles_to_cxsmiles("*CC(*)<sep><a>0:R[1]</a><a>3:Ar?n</a>")

    assert result.converted
    assert result.cxsmiles_standard == "*CC(*) |$R[1];;;Ar?n$|"
    assert result.cxsmiles_opt == "<r>R[1]</r>CC(<r>Ar?n</r>)"
    assert result.target_cxsmiles_format == "cxsmiles_opt"
    assert_shared_equal(result.cxsmiles_standard, result.cxsmiles_opt)


def test_convert_rejects_nonwildcard_atom_group_as_native_only():
    result = convert_molparser_esmiles_to_cxsmiles("CCO<sep><a>2:OH</a>")

    assert result.native_only
    assert result.reason == "atom_group_on_non_wildcard_atom"


def test_standard_to_opt_preserves_nonwildcard_atom_labels_losslessly():
    # main's pipeline preserves dummyLabels on non-wildcard atoms as residual
    # atomProp entries rather than dropping them; the result stays graph-equal
    # to the source (lossless). The converter never emits such targets from a
    # MolParser <a> group (those are rejected as atom_group_on_non_wildcard_atom),
    # so this only documents the standard_to_opt contract.
    standard = "CCO |$;;OH$|"
    opt = standard_to_opt_cxsmiles(standard)

    assert opt == "CCO|atomProp:2.dummyLabel.OH"
    assert_shared_equal(standard, opt)


def test_convert_dum_connection_point_to_atomprop():
    result = convert_molparser_esmiles_to_cxsmiles("C1(*)SCCS1<sep><a>1:<dum></a>")

    assert result.converted
    assert result.cxsmiles_standard == "C1(*)SCCS1 |atomProp:1.molparserConnectionPoint.1|"
    assert result.cxsmiles_opt == "C1(*)SCCS1|atomProp:1.molparserConnectionPoint.1"
    assert_shared_equal(result.cxsmiles_standard, result.cxsmiles_opt)
    assert_shared_not_equal(result.cxsmiles_opt, "C1(*)SCCS1")


def test_convert_dum_connection_point_with_r_label_to_atomprop():
    result = convert_molparser_esmiles_to_cxsmiles("*C*<sep><a>0:<dum></a><a>2:R</a>")

    assert result.converted
    assert result.cxsmiles_standard == "*C* |$;;R$,atomProp:0.molparserConnectionPoint.1|"
    assert result.cxsmiles_opt == "*C<r>R</r>|atomProp:0.molparserConnectionPoint.1"
    assert_shared_equal(result.cxsmiles_standard, result.cxsmiles_opt)


def test_convert_ring_dum_connection_point_to_multicenter_atomprop():
    result = convert_molparser_esmiles_to_cxsmiles("c1ccccc1<sep><r>0:<dum></r>")

    assert result.converted
    assert result.cxsmiles_standard == (
        "*.c1ccccc1 |atomProp:0.molparserConnectionPoint.1,m:0:1.2.3.4.5.6|"
    )
    assert result.cxsmiles_opt == "*.c1ccccc1|atomProp:0.molparserConnectionPoint.1,m:0:1.2.3.4.5.6"
    assert_shared_equal(result.cxsmiles_standard, result.cxsmiles_opt)


def test_convert_rejects_nonwildcard_dum_connection_point_as_native_only():
    result = convert_molparser_esmiles_to_cxsmiles("CCO<sep><a>2:<dum></a>")

    assert result.native_only
    assert result.reason == "dum_connection_point_on_nonwildcard_atom"


def test_convert_rejects_decorated_dum_connection_point_as_native_only():
    result = convert_molparser_esmiles_to_cxsmiles("[*+]CC<sep><a>0:<dum></a>")

    assert result.native_only
    assert result.reason == "decorated_dum_connection_point"


def test_convert_ring_group_to_multicenter_cxsmiles():
    ring = convert_molparser_esmiles_to_cxsmiles("c1ccccc1<sep><r>0:R</r>")

    assert ring.converted
    assert ring.cxsmiles_standard == "*.c1ccccc1 |$R;;;;;;$,m:0:1.2.3.4.5.6|"
    assert ring.cxsmiles_opt == "<r>R</r>.c1ccccc1|m:0:1.2.3.4.5.6"
    assert_shared_equal(ring.cxsmiles_standard, ring.cxsmiles_opt)


def test_convert_mixed_atom_and_ring_groups_to_multicenter_cxsmiles():
    result = convert_molparser_esmiles_to_cxsmiles("*c1ccccc1<sep><a>0:X</a><r>0:R?t</r>")

    assert result.converted
    assert result.cxsmiles_standard == "*.*c1ccccc1 |$R?t;X;;;;;;$,m:0:2.3.4.5.6.7|"
    assert result.cxsmiles_opt == "<r>R?t</r>.<r>X</r>c1ccccc1|m:0:2.3.4.5.6.7"
    assert_shared_equal(result.cxsmiles_standard, result.cxsmiles_opt)


def test_convert_bracket_ar_pseudo_label_in_molparser_core():
    result = convert_molparser_esmiles_to_cxsmiles("[Ar]Nc1ccccc1C(N*)=O<sep><a>10:Q</a>")

    assert result.converted
    assert result.cxsmiles_standard == "[Ar]Nc1ccccc1C(N*)=O |$;;;;;;;;;;Q;$|"
    assert result.cxsmiles_opt == "<r>Ar</r>Nc1ccccc1C(N<r>Q</r>)=O"
    assert_shared_equal(result.cxsmiles_standard, result.cxsmiles_opt)


def test_convert_neutral_chiral_wildcard_atom_group():
    result = convert_molparser_esmiles_to_cxsmiles("[*@]C<sep><a>0:R</a>")

    assert result.converted
    assert result.cxsmiles_standard == "[*@]C |$R;$|"
    assert result.cxsmiles_opt == "<r>R</r>C"
    assert_shared_equal(result.cxsmiles_standard, result.cxsmiles_opt)


def test_convert_rejects_decorated_wildcard_atom_label_as_native_only():
    result = convert_molparser_esmiles_to_cxsmiles("[*+]CC<sep><a>0:R</a>")

    assert result.native_only
    assert result.reason == "decorated_wildcard_atom_label"
    assert result.detail == "<a>0:R</a>"


def test_convert_rejects_standard_to_opt_atom_feature_mismatch_as_native_only():
    result = convert_molparser_esmiles_to_cxsmiles(
        "*[C@@H](*)N[C@H](*)[C@@H](N)*<sep><a>0:Ph</a><a>2:R</a><a>5:CF3</a><a>8:CO2Et</a>"
    )

    assert result.native_only
    assert result.reason == "standard_to_opt_atom_feature_mismatch"
    assert "explicit hydrogen" in result.detail


def test_convert_rejects_unsanitized_core_as_native_only():
    result = convert_molparser_esmiles_to_cxsmiles(
        "*P12(*)(CC(=O)C(=O)O1)OC(=O)C(=O)O2<sep><a>0:R[10]</a><a>2:R[9]</a>"
    )

    assert result.native_only
    assert result.reason == "core_requires_unsanitized_parse"
    assert "sanitize=False" in result.detail


def test_convert_rejects_circle_groups_as_native_only():
    circle = convert_molparser_esmiles_to_cxsmiles("*CC<sep><a>0:R</a><c>0:RingA</c>")

    assert circle.native_only
    assert circle.reason == "contains_circle_groups"


def test_convert_rejects_empty_and_grammar_unsafe_labels():
    empty = convert_molparser_esmiles_to_cxsmiles("*CC<sep><a>0:</a>")
    unsafe = convert_molparser_esmiles_to_cxsmiles("*CC<sep><a>0:R;1</a>")

    assert empty.excluded
    assert empty.reason == "empty_atom_group_name"
    assert unsafe.native_only
    assert unsafe.reason == "unsupported_label_text"


def test_converted_target_round_trips_through_standard_and_opt_converters():
    result = convert_molparser_esmiles_to_cxsmiles("*CC(*)<sep><a>0:R1</a><a>3:R2</a>")
    roundtrip_standard = opt_to_standard_cxsmiles(result.cxsmiles_opt)
    roundtrip_opt = standard_to_opt_cxsmiles(roundtrip_standard)

    assert result.converted
    assert_shared_equal(result.cxsmiles_standard, roundtrip_standard)
    assert_shared_equal(result.cxsmiles_opt, roundtrip_opt)


def test_convert_metamorphic_invariance_variants_to_equal_shared_graphs():
    raw = "*CC(*)<sep><a>0:R1</a><a>3:R2</a>"
    base = convert_molparser_esmiles_to_cxsmiles(raw)
    variants, skipped = molparser_invariance_variants(raw, variants=3)

    assert base.converted
    assert skipped == []
    assert variants
    for _name, variant in variants:
        converted = convert_molparser_esmiles_to_cxsmiles(variant)
        assert converted.converted
        assert_shared_equal(base.cxsmiles_opt, converted.cxsmiles_opt)


def test_convert_ring_metamorphic_invariance_variants_to_equal_shared_graphs():
    raw = "*c1ccccc1<sep><a>0:X</a><r>0:R</r>"
    base = convert_molparser_esmiles_to_cxsmiles(raw)
    variants, skipped = molparser_invariance_variants(raw, variants=3)

    assert base.converted
    assert skipped == []
    assert variants
    for _name, variant in variants:
        converted = convert_molparser_esmiles_to_cxsmiles(variant)
        assert converted.converted
        assert_shared_equal(base.cxsmiles_opt, converted.cxsmiles_opt)


def test_convert_sensitivity_variants_to_unequal_shared_graphs_when_representable():
    raw = "*CC(*)<sep><a>0:R1</a><a>3:R2</a>"
    base = convert_molparser_esmiles_to_cxsmiles(raw)
    variants, _skipped = molparser_sensitivity_variants(raw, variants=5)
    converted_variants = []
    for name, variant in variants:
        converted = convert_molparser_esmiles_to_cxsmiles(variant)
        if converted.converted:
            converted_variants.append((name, converted))

    assert base.converted
    assert converted_variants
    for _name, converted in converted_variants:
        assert_shared_not_equal(base.cxsmiles_opt, converted.cxsmiles_opt)


def test_convert_ring_sensitivity_variants_to_unequal_shared_graphs_when_representable():
    raw = "c1ccccc1<sep><r>0:R</r>"
    base = convert_molparser_esmiles_to_cxsmiles(raw)
    variants, _skipped = molparser_sensitivity_variants(raw, variants=3)
    converted_variants = []
    for name, variant in variants:
        converted = convert_molparser_esmiles_to_cxsmiles(variant)
        if converted.converted:
            converted_variants.append((name, converted))

    assert base.converted
    assert converted_variants
    for _name, converted in converted_variants:
        assert_shared_not_equal(base.cxsmiles_opt, converted.cxsmiles_opt)
