"""Unit tests for data-format adapters."""

import csv

from glyph.ocsr.data import (
    EvalRow,
    TrainRow,
    parse_pubchem_csv,
    parse_real_eval_csv,
    parse_real_eval_row,
    parse_uspto_csv,
)


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_pubchem_row_parses(tmp_path):
    path = tmp_path / "pubchem.csv"
    _write_csv(
        path,
        [
            {"SMILES": "CCO", "pubchem_cid": "702"},
            {"SMILES": "c1ccccc1", "pubchem_cid": "241"},
        ],
        fieldnames=["SMILES", "pubchem_cid"],
    )
    rows = parse_pubchem_csv(path)
    assert len(rows) == 2
    assert isinstance(rows[0], TrainRow)
    assert rows[0].smiles == "CCO"
    assert rows[0].image_id == "702"
    assert rows[0].image_path is None  # PubChem uses dynamic rendering


def test_uspto_row_parses_with_coords_and_edges(tmp_path):
    path = tmp_path / "uspto.csv"
    _write_csv(
        path,
        [
            {
                "file_path": "uspto_mol/images/US12345_0.png",
                "SMILES": "CC(=O)O",
                "node_coords": "[[0.1,0.2],[0.5,0.5],[0.9,0.2]]",
                "edges": "[[0,1,1],[1,2,2]]",
            },
        ],
        fieldnames=["file_path", "SMILES", "node_coords", "edges"],
    )
    rows = parse_uspto_csv(path)
    assert len(rows) == 1
    r = rows[0]
    assert r.smiles == "CC(=O)O"
    assert r.image_path == "uspto_mol/images/US12345_0.png"
    assert r.image_id == "US12345_0"
    assert r.node_coords == [[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]]
    assert r.edges == [[0, 1, 1], [1, 2, 2]]


def test_uspto_row_handles_missing_optional_columns(tmp_path):
    path = tmp_path / "uspto_mini.csv"
    _write_csv(
        path,
        [{"file_path": "x/y.png", "SMILES": "CCO"}],
        fieldnames=["file_path", "SMILES"],
    )
    rows = parse_uspto_csv(path)
    assert rows[0].node_coords is None
    assert rows[0].edges is None


def test_real_eval_row_parses(tmp_path):
    path = tmp_path / "real_USPTO.csv"
    _write_csv(
        path,
        [
            {
                "image_id": "US07314511-20080101-C00002",
                "file_path": "real/USPTO/US07314511-20080101-C00002.png",
                "SMILES": "CCO",
            },
            {
                "image_id": "blind",
                "file_path": "real/USPTO/blind.png",
                "SMILES": "",
            },
        ],
        fieldnames=["image_id", "file_path", "SMILES"],
    )
    rows = parse_real_eval_csv(path)
    assert len(rows) == 2
    assert isinstance(rows[0], EvalRow)
    assert rows[0].smiles == "CCO"
    assert rows[1].smiles is None  # empty becomes None


def test_real_eval_row_derives_image_id_from_path():
    row = parse_real_eval_row({"file_path": "real/USPTO/US01-X.png", "SMILES": "CCO"})
    assert row.image_id == "US01-X"
