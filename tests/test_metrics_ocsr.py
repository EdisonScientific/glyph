from __future__ import annotations

from glyph.markush.eval.ocsr import (
    OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
    OCSR_PREDICTION_MODE_NATIVE_SMILES,
    OCSREvaluationConfig,
    canonical_smiles,
    evaluate_ocsr_smiles,
    flatten_ocsr_metrics,
    graph_canonical_smiles,
)


def test_ocsr_canonical_smiles_uses_rdkit_isomeric_canonicalization() -> None:
    assert canonical_smiles("C(C)O") == "CCO"
    assert canonical_smiles("N[C@@H](C)C(=O)O") != canonical_smiles("N[C@H](C)C(=O)O")


def test_ocsr_graph_canonical_smiles_ignores_stereo() -> None:
    assert graph_canonical_smiles("N[C@@H](C)C(=O)O") == graph_canonical_smiles("N[C@H](C)C(=O)O")


def test_evaluate_ocsr_smiles_counts_canonical_equivalence() -> None:
    metrics, details = evaluate_ocsr_smiles(
        [{"id": "a", "prediction": "C(C)O"}],
        [{"id": "a", "smiles": "CCO"}],
        return_details=True,
    )

    assert metrics["n_samples"] == 1
    assert metrics["n_correct"] == 1
    assert metrics["canon_exact_match"] == 1.0
    assert metrics["graph_exact_match"] == 1.0
    assert details[0]["pred_canonical"] == "CCO"


def test_evaluate_ocsr_smiles_does_not_extract_from_markush_xml() -> None:
    metrics, details = evaluate_ocsr_smiles(
        [{"id": "a", "prediction": "<markush><cxsmi>CCO</cxsmi></markush>"}],
        [{"id": "a", "smiles": "CCO"}],
        return_details=True,
    )

    assert metrics["n_samples"] == 1
    assert metrics["n_correct"] == 0
    assert metrics["n_valid_pred"] == 0
    assert metrics["n_markush_xml_pred"] == 1
    assert metrics["markush_xml_prediction_rate"] == 1.0
    assert metrics["prediction_mode"] == OCSR_PREDICTION_MODE_NATIVE_SMILES
    assert details[0]["prediction_error"] == "markush_xml_prediction"


def test_no_extension_markush_prediction_passes_only_in_opt_in_ocsr_mode() -> None:
    prediction = "<markush><cxsmi>C(C)O</cxsmi><stable></stable></markush>"

    native = evaluate_ocsr_smiles(
        [{"id": "a", "prediction": prediction}],
        [{"id": "a", "smiles": "CCO"}],
    )
    opt_in, details = evaluate_ocsr_smiles(
        [{"id": "a", "prediction": prediction}],
        [{"id": "a", "smiles": "CCO"}],
        config=OCSREvaluationConfig(prediction_mode=OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION),
        return_details=True,
    )

    assert native["n_correct"] == 0
    assert native["n_valid_pred"] == 0
    assert opt_in["n_correct"] == 1
    assert opt_in["n_correct_graph"] == 1
    assert opt_in["n_valid_pred"] == 1
    assert opt_in["n_markush_xml_pred"] == 1
    assert opt_in["n_markush_extension_pred"] == 0
    assert opt_in["prediction_mode"] == OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION
    assert details[0]["extracted_prediction"] == "C(C)O"
    assert details[0]["pred_canonical"] == "CCO"


def test_hallucinated_markush_extension_fails_in_opt_in_ocsr_mode() -> None:
    predictions = [
        {
            "id": "stable",
            "prediction": ("<markush><cxsmi>CCO</cxsmi><stable>R1: alkyl</stable></markush>"),
        },
        {
            "id": "sg",
            "prediction": "<markush><cxsmi>CCO |Sg:n:0,1:n:ht|</cxsmi></markush>",
        },
        {
            "id": "rgroup",
            "prediction": "<markush><cxsmi><r>R1</r>CCO</cxsmi></markush>",
        },
    ]

    metrics, details = evaluate_ocsr_smiles(
        predictions,
        [{"id": row["id"], "smiles": "CCO"} for row in predictions],
        prediction_mode=OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
        return_details=True,
    )

    assert metrics["n_samples"] == 3
    assert metrics["n_correct"] == 0
    assert metrics["n_valid_pred"] == 0
    assert metrics["n_markush_extension_pred"] == 3
    assert metrics["markush_extension_prediction_rate"] == 1.0
    assert {detail["prediction_error"] for detail in details} == {"markush_extension_prediction"}


def test_evaluate_ocsr_smiles_counts_missing_predictions_as_wrong() -> None:
    metrics = evaluate_ocsr_smiles([], [{"id": "a", "smiles": "CCO"}])

    assert metrics["n_samples"] == 1
    assert metrics["n_missing_pred"] == 1
    assert metrics["n_invalid_pred"] == 1
    assert metrics["canon_exact_match"] == 0.0


def test_evaluate_ocsr_smiles_excludes_invalid_ground_truth() -> None:
    metrics = evaluate_ocsr_smiles(
        [{"id": "a", "prediction": "CCO"}],
        [{"id": "a", "smiles": "not-smiles"}],
    )

    assert metrics["n_total_gt"] == 1
    assert metrics["n_invalid_gt"] == 1
    assert metrics["n_samples"] == 0
    assert metrics["canon_exact_match"] == 0.0


def test_flatten_ocsr_metrics_uses_prefix() -> None:
    flat = flatten_ocsr_metrics({"canon_exact_match": 0.5}, prefix="eval_ocsr_uspto")
    assert flat == {"eval_ocsr_uspto/canon_exact_match": 0.5}
