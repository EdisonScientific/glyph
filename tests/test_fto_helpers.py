from __future__ import annotations

from glyph.markush.fto import (
    extract_components,
    match_concrete_smiles,
    normalize_claim,
    parse_claim,
    strict_equal,
)


def test_parse_claim_extracts_core_components() -> None:
    claim = "<markush><cxsmi><r>R1</r>c1ccccc1</cxsmi><stable></stable></markush>"

    parsed = parse_claim(claim)

    assert parsed.cxsmiles_opt == "<r>R1</r>c1ccccc1"
    assert parsed.backbone_smiles == "*c1ccccc1"
    assert parsed.r_labels == ("R1",)
    assert parsed.stable == {}
    assert parsed.valid_graph is True


def test_extract_components_is_json_serializable_shape() -> None:
    components = extract_components("CC")

    assert components["cxsmiles_opt"] == "CC"
    assert components["markush_xml"] == "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"


def test_normalize_claim_wraps_bare_cxsmiles() -> None:
    assert normalize_claim("CC") == "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"


def test_strict_equal_uses_graph_equality() -> None:
    assert strict_equal("CC", "<markush><cxsmi>CC</cxsmi><stable></stable></markush>") is True


def test_match_concrete_smiles_supports_exact_non_markush_case() -> None:
    result = match_concrete_smiles("CCO", "CCO")

    assert result.supported is True
    assert result.matches is True
    assert result.protocol == "rdkit_exact_canonical_smiles"


def test_match_concrete_smiles_supports_unconstrained_rgroup_substructure() -> None:
    claim = "<markush><cxsmi><r>R1</r>c1ccccc1</cxsmi><stable></stable></markush>"

    positive = match_concrete_smiles("Cc1ccccc1", claim)
    negative = match_concrete_smiles("CCO", claim)

    assert positive.supported is True
    assert positive.matches is True
    assert negative.supported is True
    assert negative.matches is False


def test_match_concrete_smiles_returns_unsupported_for_specific_stable_constraints() -> None:
    claim = "<markush><cxsmi><r>R1</r>c1ccccc1</cxsmi><stable>R1:Cl</stable></markush>"

    result = match_concrete_smiles("Clc1ccccc1", claim)

    assert result.supported is False
    assert result.matches is None
    assert "specific_stable_substituent_constraints_not_supported" in result.unsupported_reasons
