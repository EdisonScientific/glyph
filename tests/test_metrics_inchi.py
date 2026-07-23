from __future__ import annotations

import pytest

from glyph.markush.eval.metrics_inchi import (
    _RDKIT_AVAILABLE,
    backbone_inchikey,
    backbone_inchikey_match,
    backbone_smiles,
    evaluate_predictions_inchi,
)

pytestmark = pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="metrics_inchi requires RDKit")


def test_backbone_smiles_extracts_xml_and_strips_extensions():
    assert backbone_smiles("<markush><cxsmi><r>R1</r>CC|Sg:n:1,2:n:ht</cxsmi></markush>") == "*CC"


def test_backbone_inchikey_ignores_r_label_names():
    assert backbone_inchikey("<r>R1</r>CC") == backbone_inchikey("<r>X2</r>CC")


def test_backbone_inchikey_match_returns_none_for_unparseable():
    assert backbone_inchikey_match("not-a-smiles", "CC") is None


def test_evaluate_predictions_inchi_delegates_to_mg2_protocol():
    preds = [{"id": "a", "prediction": "<markush><cxsmi>CC</cxsmi></markush>"}]
    gts = [{"id": "a", "cxsmiles_opt": "CC"}]

    metrics = evaluate_predictions_inchi(preds, gts)

    assert metrics["n_samples"] == 1
    assert metrics["cxsmi_equality"] == 1.0
