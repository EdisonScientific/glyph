from __future__ import annotations

import pytest

from glyph.markush.data.canonicalize import reindex_extensions
from glyph.markush.data.pipeline import opt_to_standard_cxsmiles, standard_to_opt_cxsmiles
from glyph.markush.eval.markush_graph import (
    canonical_markush_equal,
    canonical_markush_signature,
    markush_graph_equal,
    networkx_available,
    networkx_markush_equal,
    rdkit_canonical_markush_equal,
    split_extension_sections,
)


def assert_equal(gt: str, pred: str) -> None:
    result = markush_graph_equal(gt, pred)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.error == ""
    assert result.equal is True


def assert_not_equal(gt: str, pred: str) -> None:
    result = markush_graph_equal(gt, pred)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.equal is False


def test_graph_equal_is_reflexive_for_labeled_markush_with_extensions():
    cxsmiles = "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6"

    assert_equal(cxsmiles, cxsmiles)


def test_graph_equal_accepts_fragment_order_and_kekule_serialization_variant():
    gt = "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6"
    pred = "<r>Ym</r>C.NC1=CC=CC=C1|m:1:3.4.5.6.7.8"

    assert_equal(gt, pred)


def test_canonical_signature_accepts_fragment_order_and_kekule_variant():
    gt = "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6"
    pred = "<r>Ym</r>C.NC1=CC=CC=C1|m:1:3.4.5.6.7.8"

    gt_sig = canonical_markush_signature(gt)
    pred_sig = canonical_markush_signature(pred)

    assert gt_sig.valid, gt_sig.error
    assert pred_sig.valid, pred_sig.error
    assert gt_sig.signature == pred_sig.signature
    assert canonical_markush_equal(gt, pred).equal is True


def test_graph_equal_rejects_variable_attachment_on_different_ring():
    gt = "CC.S1C=NC2=C1C=CC=C2|m:1:2.3.4.5.6"
    pred = "C1=CC2=C(C=C1)N=CS2.CC|m:9:0.1.2.3.4.5"

    assert_not_equal(gt, pred)
    assert canonical_markush_equal(gt, pred).equal is False


def test_graph_equal_rejects_label_swap_on_distinct_atoms():
    gt = "CC(<r>R1</r>)C<r>R2</r>"
    pred = "CC(<r>R2</r>)C<r>R1</r>"

    assert_not_equal(gt, pred)
    assert canonical_markush_equal(gt, pred).equal is False


def test_graph_equal_accepts_symmetry_equivalent_label_position():
    gt = "<r>R1</r>c1ccccc1"
    pred = "C1=CC=C(<r>R1</r>)C=C1"

    assert_equal(gt, pred)
    assert canonical_markush_equal(gt, pred).equal is True


def test_canonical_signature_reports_state_budget_errors():
    result = canonical_markush_signature("c1ccccc1", max_states=10)

    assert result.valid is False
    assert "states" in result.error


@pytest.mark.parametrize(
    ("gt", "pred", "expected"),
    [
        (
            "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6",
            "<r>Ym</r>C.NC1=CC=CC=C1|m:1:3.4.5.6.7.8",
            True,
        ),
        (
            "CC.S1C=NC2=C1C=CC=C2|m:1:2.3.4.5.6",
            "C1=CC2=C(C=C1)N=CS2.CC|m:9:0.1.2.3.4.5",
            False,
        ),
        (
            "CC(<r>R1</r>)C<r>R2</r>",
            "CC(<r>R2</r>)C<r>R1</r>",
            False,
        ),
        (
            "<r>R1</r>c1ccccc1",
            "C1=CC=C(<r>R1</r>)C=C1",
            True,
        ),
    ],
)
def test_rdkit_canonical_backend_agrees_with_internal_backend(
    gt: str,
    pred: str,
    expected: bool,
):
    internal = markush_graph_equal(gt, pred)
    rdkit_canonical = rdkit_canonical_markush_equal(gt, pred)

    assert internal.equal is expected
    assert rdkit_canonical.valid_gt, rdkit_canonical.gt_error
    assert rdkit_canonical.valid_pred, rdkit_canonical.pred_error
    assert rdkit_canonical.error == ""
    assert rdkit_canonical.equal is expected


@pytest.mark.skipif(not networkx_available(), reason="NetworkX is not installed")
@pytest.mark.parametrize(
    ("gt", "pred", "expected"),
    [
        (
            "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6",
            "<r>Ym</r>C.NC1=CC=CC=C1|m:1:3.4.5.6.7.8",
            True,
        ),
        (
            "CC.S1C=NC2=C1C=CC=C2|m:1:2.3.4.5.6",
            "C1=CC2=C(C=C1)N=CS2.CC|m:9:0.1.2.3.4.5",
            False,
        ),
        (
            "CC(<r>R1</r>)C<r>R2</r>",
            "CC(<r>R2</r>)C<r>R1</r>",
            False,
        ),
        (
            "<r>R1</r>c1ccccc1",
            "C1=CC=C(<r>R1</r>)C=C1",
            True,
        ),
    ],
)
def test_networkx_backend_agrees_with_internal_and_canonical_backends(
    gt: str,
    pred: str,
    expected: bool,
):
    internal = markush_graph_equal(gt, pred)
    canonical = canonical_markush_equal(gt, pred)
    networkx = networkx_markush_equal(gt, pred)
    rdkit_canonical = rdkit_canonical_markush_equal(gt, pred)

    assert internal.equal is expected
    assert canonical.equal is expected
    assert rdkit_canonical.valid_gt, rdkit_canonical.gt_error
    assert rdkit_canonical.valid_pred, rdkit_canonical.pred_error
    assert rdkit_canonical.error == ""
    assert rdkit_canonical.equal is expected
    assert networkx.valid_gt, networkx.gt_error
    assert networkx.valid_pred, networkx.pred_error
    assert networkx.error == ""
    assert networkx.equal is expected


def test_graph_equal_treats_bracket_ar_as_markush_pseudo_label():
    cxsmiles = "[Ar]<r>Z</r>C1CO1"

    assert_equal(cxsmiles, cxsmiles)


def test_reindex_extensions_preserves_sgroup_order_and_remaps_parent_atom():
    ext = "Sg:n:21,22,23:f:ht:20"
    old_to_new = {20: 5, 21: 30, 22: 31, 23: 6}

    assert reindex_extensions(ext, old_to_new) == "Sg:n:30,31,6:f:ht:5"


def test_reindex_extensions_remaps_atomprops():
    ext = "atomProp:2.molparserConnectionPoint.1:4.dummyLabel.R"
    old_to_new = {2: 8, 4: 1}

    assert reindex_extensions(ext, old_to_new) == (
        "atomProp:8.molparserConnectionPoint.1:1.dummyLabel.R"
    )


def test_graph_equal_keeps_numeric_sgroup_label_as_label_not_atom_index():
    cxsmiles = "CC|Sg:n:0:17:ht"

    assert_equal(cxsmiles, cxsmiles)


def test_graph_equal_ignores_trailing_empty_sgroup_fields_and_extension_comma():
    standard = "CC |Sg:n:0:m:ht:::,|"
    opt = "CC|Sg:n:0:m:ht"

    assert_equal(standard, opt)


def test_graph_equal_preserves_numeric_sgroup_metadata_that_is_not_atom_index():
    cxsmiles = "CC|Sg:n:0:x:ht:35"

    assert_equal(cxsmiles, cxsmiles)


def test_graph_equal_reads_standard_atom_labels_with_trailing_extension_comma():
    standard = "*C1=C(*)C(*)=C2C(=O)C(*)=C(*)C(*)=C2C1=O |$B16;;;B15;;A13;;;;;B17;;B18;;A14;;;$,|"
    opt = (
        "<r>B16</r>C1=C(<r>B15</r>)C(<r>A13</r>)=C2C(=O)"
        "C(<r>B17</r>)=C(<r>B18</r>)C(<r>A14</r>)=C2C1=O"
    )

    assert split_extension_sections("$B16;;;B15;;A13;;;;;B17;;B18;;A14;;;$,") == [
        "$B16;;;B15;;A13;;;;;B17;;B18;;A14;;;$"
    ]
    assert_equal(standard, opt)


def test_graph_equal_reads_atomprop_dummy_labels():
    standard = "C*OC |atomProp:1.dummyLabel.E|"
    opt = "C<r>E</r>OC"

    assert_equal(standard, opt)


def test_graph_equal_preserves_connection_point_atomprop():
    connection_point = "*C |atomProp:0.molparserConnectionPoint.1|"

    assert_equal(connection_point, "*C|atomProp:0.molparserConnectionPoint.1")
    assert_not_equal(connection_point, "*C")
    assert_not_equal(connection_point, "<r>_AP</r>C")
    assert canonical_markush_equal(connection_point, "*C").equal is False


def test_split_extension_sections_recognizes_non_sg_cx_sections():
    extension = "Sg:n:0,1:n:ht,f:0.1,LN:0:1.2,ctu:3,atomProp:2.dummyLabel.R1"

    assert split_extension_sections(extension) == [
        "Sg:n:0,1:n:ht",
        "f:0.1",
        "LN:0:1.2",
        "ctu:3",
        "atomProp:2.dummyLabel.R1",
    ]


def test_graph_equal_rejects_cx_bond_stereo_mismatch():
    assert_not_equal("CC=CC |c:1|", "CC=CC |t:1|")


def test_graph_equal_parses_compact_c_t_bond_stereo_section():
    assert_equal("C/C=C/C|c:0t:1|", "C/C=C/C|c:0,t:1|")


def test_graph_equal_accepts_slash_stereo_equivalent_to_cx_bond_stereo():
    comparators = [
        markush_graph_equal,
        canonical_markush_equal,
        rdkit_canonical_markush_equal,
    ]
    if networkx_available():
        comparators.append(networkx_markush_equal)

    for comparator in comparators:
        assert comparator("C/C=C/C", "CC=CC |t:1|").equal is True
        assert comparator("C/C=C\\C", "CC=CC |c:1|").equal is True
        assert comparator("C/C=C/C", "CC=CC |c:1|").equal is False
        assert comparator("C/C=C\\C", "CC=CC |t:1|").equal is False

    assert_equal("C/C=C/C", "CC=CC |t:1|")
    assert_equal("C/C=C\\C", "CC=CC |c:1|")


def test_graph_equal_still_ignores_plain_slash_stereo_without_cx_bond_stereo():
    comparators = [
        markush_graph_equal,
        canonical_markush_equal,
        rdkit_canonical_markush_equal,
    ]
    if networkx_available():
        comparators.append(networkx_markush_equal)

    for comparator in comparators:
        assert comparator("C/C=C/C", "CC=CC").equal is True
        assert comparator("CC=CC |t:1|", "CC=CC").equal is False


def test_graph_equal_maps_cx_bond_indexes_by_smiles_order():
    # Ring-closure bonds are indexed when the closure token is read in
    # CXSMILES order, not by RDKit's internal bond index.
    assert_equal("C1C.CC=1 |c:2|", "C=1C.CC1 |c:2|")


def test_graph_equal_requires_component_grouping_but_ignores_fragment_order():
    gt = "[Na+].[OH-].c1ccccc1 |f:0.1|"
    pred = "c1ccccc1.[OH-].[Na+] |f:1.2|"

    assert_equal(gt, pred)
    assert_not_equal(gt, "[Na+].[OH-].c1ccccc1")


def test_graph_equal_requires_link_node_annotation():
    assert_not_equal("NCC=O |LN:1:1.3|", "NCC=O")


def test_graph_equal_treats_numbered_wildcards_as_wildcards():
    standard = "[1*]CC |$R1$|"
    opt = "<r>R1</r>CC"

    assert_equal(standard, opt)


def test_graph_equal_allows_trailing_empty_dollar_labels_to_be_omitted():
    standard = "*C |$R1;$|"
    opt = "<r>R1</r>C"

    assert_equal(standard, opt)


def test_standard_to_opt_uses_dollar_atom_indices_not_dense_label_order():
    standard = "CC(*)C* |$;;R1;;R2$|"

    assert standard_to_opt_cxsmiles(standard) == "CC(<r>R1</r>)C<r>R2</r>"


def test_opt_to_standard_writes_dollar_labels_by_atom_index():
    opt = "CC(<r>R1</r>)C<r>R2</r>|m:1:0.1.3"

    assert opt_to_standard_cxsmiles(opt) == "CC(*)C* |$;;R1;;R2$,m:1:0.1.3|"


def test_opt_to_standard_converts_bracket_ar_pseudo_label():
    opt = "[Ar]<r>Z</r>C1CO1"

    assert_equal(opt, opt_to_standard_cxsmiles(opt))


def test_standard_to_opt_converts_bracket_ar_pseudo_label():
    standard = "[Ar]C* |$;;R$|"
    opt = standard_to_opt_cxsmiles(standard)

    assert opt == "<r>Ar</r>C<r>R</r>"
    assert_equal(standard, opt)


def test_standard_to_opt_converts_neutral_chiral_wildcard_label():
    standard = "[*@]C |$R;$|"
    opt = standard_to_opt_cxsmiles(standard)

    assert opt == "<r>R</r>C"
    assert_equal(standard, opt)


def test_standard_opt_round_trip_preserves_label_bound_structure():
    opt = "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6"
    standard = opt_to_standard_cxsmiles(opt)
    round_trip = standard_to_opt_cxsmiles(standard)

    assert_equal(opt, standard)
    assert_equal(opt, round_trip)


def test_standard_opt_round_trip_ignores_rdkit_explicit_h_bookkeeping():
    standard = "[1*][2*][C@@H]([3*])[4*] |$A;B;;D;E$|"
    opt = standard_to_opt_cxsmiles(standard)
    round_trip = opt_to_standard_cxsmiles(opt)

    assert_equal(standard, opt)
    assert_equal(standard, round_trip)


def test_graph_equal_can_normalize_explicit_h_atom_representation():
    gt = "N([H])C"
    pred = "NC"

    assert not markush_graph_equal(gt, pred).equal
    assert markush_graph_equal(gt, pred, normalize_hydrogens=True).equal


def test_graph_equal_normalizes_adjacent_sgroup_explicit_h_membership():
    gt = "[H]OC |Sg:n:0,1:m:ht:::,|"
    pred = "OC |Sg:n:0:m:ht:::,|"

    assert not markush_graph_equal(gt, pred).equal
    assert markush_graph_equal(gt, pred, normalize_hydrogens=True).equal


def test_graph_equal_does_not_normalize_non_adjacent_sgroup_h_membership():
    gt = "[H].OC |Sg:n:0,1:m:ht:::,|"
    pred = "OC |Sg:n:0:m:ht:::,|"

    assert not markush_graph_equal(gt, pred, normalize_hydrogens=True).equal


def test_graph_equal_supports_disconnected_fragment_order():
    assert markush_graph_equal("CC.O", "O.CC").equal
    assert not markush_graph_equal("CC.O", "CCO").equal


def test_legacy_convert_opt_to_standard_is_dependency_free():
    opt = "CC(<r>R1</r>)C<r>R2</r>"

    assert opt_to_standard_cxsmiles(opt) == "CC(*)C* |$;;R1;;R2$|"
