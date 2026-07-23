"""Tests for the canonicalization + evaluation contract.

These tests use only synthetic SMILES; they do not depend on the HF benchmark.
"""

from pathlib import Path

import pyarrow as pa
from rdkit import Chem

from glyph.ocsr.benchmark import default_uspto5719_benchmark_path
from glyph.ocsr.eval import (
    EVAL_CONTRACT_VERSION,
    canonical_smiles,
    evaluate_pairs,
    graph_canonical_smiles,
    hash_strings_sorted,
    load_uspto_benchmark,
)
from glyph.ocsr.full_eval import select_majority_vote
from glyph.ocsr.postprocess import postprocess_smiles


def test_eval_contract_version_is_v1():
    assert EVAL_CONTRACT_VERSION == "v1"


def test_default_complete_benchmark_uses_glyph_data_root(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    assert default_uspto5719_benchmark_path() == tmp_path / "uspto_ocsr_5719_complete.csv"


def test_canonical_smiles_basic():
    # Different ways to write benzene canonicalize to the same string.
    a = canonical_smiles("c1ccccc1")
    b = canonical_smiles("C1=CC=CC=C1")
    assert a is not None and b is not None and a == b


def test_canonical_smiles_invalid_returns_none():
    assert canonical_smiles("") is None
    assert canonical_smiles(None) is None  # type: ignore[arg-type]
    assert canonical_smiles("not_a_smiles") is None


def test_evaluate_pairs_simple_case():
    preds = ["c1ccccc1", "BOGUS", "CCO", "CCC"]
    gts = ["C1=CC=CC=C1", "C1=CC=CC=C1", "OCC", "CCN"]
    result = evaluate_pairs(preds, gts)
    assert result.n == 4
    assert result.n_correct == 2  # benzene + ethanol
    assert result.canon_exact_match == 0.5
    assert result.valid_smiles_rate == 0.75


def test_evaluate_pairs_excludes_invalid_gt():
    preds = ["CCO", "CCO"]
    gts = ["OCC", "not_a_smiles"]  # second GT excluded
    result = evaluate_pairs(preds, gts)
    assert result.n == 1
    assert result.n_correct == 1
    assert result.n_invalid_gt == 1
    assert result.canon_exact_match == 1.0


def test_graph_canon_ignores_stereo():
    # Same connectivity, different stereo -> graph match should be True
    a = graph_canonical_smiles("C[C@H](N)C(=O)O")
    b = graph_canonical_smiles("C[C@@H](N)C(=O)O")
    c = graph_canonical_smiles("CC(N)C(=O)O")
    assert a == b == c


def test_evaluate_pairs_distinguishes_all_stereochemistry_conventions():
    predictions = ["C/C=C/[C@H](F)Cl", "C[C@H](N)C(=O)O"]
    ground_truth = [r"C/C=C\[C@H](F)Cl", "C[C@@H](N)C(=O)O"]

    molecules = [Chem.MolFromSmiles(smiles) for smiles in predictions + ground_truth]
    assert all(molecule is not None for molecule in molecules)
    assert Chem.FindMolChiralCenters(molecules[0], includeCIP=True) == [(3, "R")]
    assert Chem.FindMolChiralCenters(molecules[2], includeCIP=True) == [(3, "R")]
    assert [
        bond.GetStereo()
        for molecule in (molecules[0], molecules[2])
        for bond in molecule.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    ] == [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]
    assert Chem.FindMolChiralCenters(molecules[1], includeCIP=True) == [(1, "S")]
    assert Chem.FindMolChiralCenters(molecules[3], includeCIP=True) == [(1, "R")]

    result = evaluate_pairs(predictions, ground_truth)

    assert result.n == 2
    assert result.n_correct == 0
    assert result.n_correct_chirality == 1
    assert result.n_correct_graph == 2
    assert result.canon_exact_match == 0.0
    assert result.chirality_exact_match == 0.5
    assert result.graph_exact_match == 1.0


def test_hash_strings_sorted_is_order_independent():
    assert hash_strings_sorted(["b", "a", "c"]) == hash_strings_sorted(["c", "a", "b"])


def test_arrow_ipc_cache_is_read_without_parquet_parser(tmp_path: Path):
    path = tmp_path / "benchmark.arrow"
    table = pa.table(
        {
            "id": ["row-1"],
            "image": [{"bytes": b"png", "path": "row-1.png"}],
            "mol": ["mol"],
            "smiles": ["CCO"],
            "selfies": ["selfies"],
            "inchi": ["inchi"],
        }
    )
    with path.open("wb") as handle, pa.ipc.new_stream(handle, table.schema) as writer:
        writer.write_table(table)

    rows = load_uspto_benchmark(path)

    assert rows == [
        {
            "id": "row-1",
            "smiles": "CCO",
            "selfies": "selfies",
            "inchi": "inchi",
            "mol": "mol",
            "source_of_label": None,
            "image_bytes": b"png",
        }
    ]


def test_postprocess_smiles_strips_isolated_hydrogen_fragments_and_canonicalizes():
    assert postprocess_smiles("C(C)O.[H].[HH]") == "CCO"


def test_select_majority_vote_drops_invalid_and_breaks_ties_by_likelihood():
    candidates = [
        {"canonical": "CCO", "mean_logprob": -0.8},
        {"canonical": None, "mean_logprob": -0.1},
        {"canonical": "CCN", "mean_logprob": -0.2},
        {"canonical": "CCO", "mean_logprob": -0.9},
        {"canonical": "CCN", "mean_logprob": -0.3},
    ]

    selected, stats = select_majority_vote(candidates, fallback_prediction="CCC")

    assert selected == "CCN"
    assert stats["invalid_count"] == 1
    assert stats["vote_count"] == 2
    assert stats["tied"] is True


def test_select_majority_vote_all_invalid_uses_non_oracle_greedy_fallback():
    selected, stats = select_majority_vote(
        [{"canonical": None, "mean_logprob": -1.0}],
        fallback_prediction="C(C)O",
    )

    assert selected == "C(C)O"
    assert stats["all_invalid"] is True
    assert stats["selection_source"] == "greedy_fallback"
