from __future__ import annotations

import csv
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from glyph.data.ocsr_eval_jsonl import materialize_ocsr_eval


def _write_png(path: Path, *, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _png_bytes(*, size: tuple[int, int] = (7, 5)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_materialize_ocsr_eval_from_csv_canonicalizes_and_copies_images(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "source_images"
    _write_png(image_root / "row1.png", size=(11, 9))
    csv_path = tmp_path / "clef.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "file_path", "SMILES"])
        writer.writeheader()
        writer.writerow({"image_id": "row1", "file_path": "row1.png", "SMILES": "C(C)O"})

    stats = materialize_ocsr_eval(
        input_path=csv_path,
        output_dir=tmp_path / "out",
        dataset_name="clef",
        source_format="csv",
        image_root=image_root,
    )

    assert stats.total_rows == 1
    assert stats.written_rows == 1
    assert stats.copied_images == 1
    assert stats.min_width == 11
    assert stats.max_height == 9
    row = json.loads((tmp_path / "out" / "eval_ocsr_clef.jsonl").read_text())
    assert row["id"] == "row1"
    assert row["task"] == "ocsr"
    assert row["mode"] == "molecule_ocsr"
    assert row["source"] == "clef"
    assert row["split"] == "eval"
    assert row["benchmark"] == "clef"
    assert row["image_path"] == "images/clef/row1.png"
    assert row["smiles"] == "CCO"
    assert row["raw_smiles"] == "C(C)O"
    assert row["target_format"] == "SMILES"
    assert row["target_source"] == "smiles->rdkit_canonical_isomeric_smiles"
    assert row["prompt"] == (
        "Extract the molecular structure from this image as SMILES.\n"
        "Target format: return a SMILES string."
    )
    assert row["messages"] == [
        {
            "role": "user",
            "content": (
                "<image>\nExtract the molecular structure from this image as SMILES.\n"
                "Target format: return a SMILES string."
            ),
        },
        {"role": "assistant", "content": "CCO"},
    ]
    assert row["images"] == ["images/clef/row1.png"]
    assert (tmp_path / "out" / row["image_path"]).exists()


def test_materialize_ocsr_eval_from_csv_counts_skips(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "file_path", "SMILES"])
        writer.writeheader()
        writer.writerow({"image_id": "bad_smiles", "file_path": "bad.png", "SMILES": "not-smiles"})
        writer.writerow({"image_id": "missing_image", "file_path": "missing.png", "SMILES": "CCO"})

    stats = materialize_ocsr_eval(
        input_path=csv_path,
        output_dir=tmp_path / "out",
        dataset_name="bad",
        source_format="csv",
        image_root=tmp_path,
    )

    assert stats.total_rows == 2
    assert stats.written_rows == 0
    assert stats.skipped_invalid_smiles == 1
    assert stats.skipped_missing_image == 1
    assert (tmp_path / "out" / "eval_ocsr_bad.jsonl").read_text() == ""


def test_materialize_ocsr_eval_from_parquet_embedded_images(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "uspto.parquet"
    table = pyarrow.table(
        {
            "id": ["uspto_1"],
            "image": [{"bytes": _png_bytes(size=(13, 10))}],
            "smiles": ["C(C)O"],
        }
    )
    pq.write_table(table, parquet_path)

    stats = materialize_ocsr_eval(
        input_path=parquet_path,
        output_dir=tmp_path / "out",
        dataset_name="uspto",
        source_format="parquet",
    )

    assert stats.total_rows == 1
    assert stats.written_rows == 1
    assert stats.embedded_images == 1
    assert stats.min_width == 13
    row = json.loads((tmp_path / "out" / "eval_ocsr_uspto.jsonl").read_text())
    assert row["id"] == "uspto_1"
    assert row["split"] == "eval"
    assert row["benchmark"] == "uspto"
    assert row["image_path"] == "images/uspto/uspto_1.png"
    assert row["images"] == ["images/uspto/uspto_1.png"]
    assert row["smiles"] == "CCO"
    assert (tmp_path / "out" / row["image_path"]).exists()
