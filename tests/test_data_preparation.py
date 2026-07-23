from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from glyph.markush.data.cli import app as data_cli_app
from glyph.markush.data.example_modes import (
    MOLECULE_OCSR,
    MOLPARSER_CONVERTED_MARKUSH,
    MOLPARSER_NATIVE_MARKUSH,
    PATENT_IMAGE_MARKUSH,
    STRUCTURE_ONLY_MARKUSH,
    STRUCTURED_TRANSCRIPT_MARKUSH,
    TEXT_CONDITIONED_MARKUSH,
)
from glyph.markush.data.hf_publish import dataset_from_jsonl
from glyph.markush.data.importers.legacy import normalize_existing_record
from glyph.markush.data.importers.public_datasets import (
    DATASET_SPECS,
    config_validation_rows,
    convert_loaded_dataset,
    convert_molscribe_archive_dir,
    convert_split_dataset_rows,
    default_workers,
    extract_molscribe_archive_inputs,
    normalize_row,
    selected_dataset_names,
    split_name_from_expression,
)
from glyph.markush.data.prepare_sft import (
    MixWeights,
    build_training_example,
    discover_jsonl_inputs,
    eligible_modes,
    is_trainable_source,
    load_exclusion_keys,
    mix_examples,
    prepare_sft_jsonl,
    target_spec_for_record,
)
from glyph.markush.data.structured_transcripts import (
    SCHEMA_VERSION,
    validate_structured_transcript_content,
)
from glyph.markush.eval import common


def test_normalize_row_maps_mg2_fields_and_saves_image(tmp_path: Path):
    image = Image.new("RGB", (8, 8), "white")
    row = {
        "id": "US20100009286A1.pdf_4_2",
        "page_image": image,
        "cxsmiles_opt": "<r>R1</r>CC",
        "cxsmiles": "[1*]CC |$R1;;$|",
        "annotation": "<markush><cxsmi><r>R1</r>CC</cxsmi><stable></stable></markush>",
        "description": "wherein R1 is alkyl",
        "cells": [{"text": "R1", "bbox": [0, 0, 1, 1]}],
    }

    out = normalize_row(
        row,
        spec=DATASET_SPECS["mg1-synthetic"],
        split="train",
        index=0,
        images_dir=tmp_path / "images",
    )

    assert out["id"] == "US20100009286A1.pdf_4_2"
    assert out["source"] == "mg1_synthetic"
    assert out["image_has_text"] is True
    assert out["patent_id"] == "US20100009286A1"
    assert out["patent_office"] == "USPTO"
    # Portable source-id-keyed layout resolved under GLYPH_DATA_ROOT at load time.
    assert out["image"] == "images/mg1_synthetic/train/US20100009286A1.pdf_4_2.png"
    assert (tmp_path / out["image"]).exists()
    assert out["cells"][0]["text"] == "R1"
    assert out["metadata"]["source_split"] == "train"
    assert out["metadata"]["source_row_index"] == 0
    assert out["metadata"]["source_id_field"] == "id"
    assert out["metadata"]["source_image_field"] == "page_image"
    assert out["metadata"]["source_smiles_field"] is None


def test_normalize_row_preserves_molparser_esmiles_target(tmp_path: Path):
    image = Image.new("RGB", (8, 8), "white")
    row = {
        "image": image,
        "smiles": "CCO<sep><a>2:<dum></a>",
    }

    out = normalize_row(
        row,
        spec=DATASET_SPECS["molparser-markush"],
        split="train",
        index=0,
        images_dir=tmp_path / "images",
    )

    assert out["source"] == "molparser"
    assert out["id"] == "molparser-markush_train_000000"
    assert out["image"] == ("images/molparser/train/molparser-markush_train_000000.png")
    assert out["task"] == "markush"
    assert out["smiles"] == "CCO<sep><a>2:<dum></a>"
    assert out["molparser_esmiles"] == "CCO<sep><a>2:<dum></a>"
    assert out["metadata"]["source_split"] == "train"
    assert out["metadata"]["source_row_index"] == 0
    assert out["metadata"]["source_id_field"] is None
    assert out["metadata"]["source_image_field"] == "image"
    assert out["metadata"]["source_smiles_field"] == "smiles"


def test_normalize_row_uses_source_id_in_exported_image_path(tmp_path: Path):
    image = Image.new("RGB", (8, 8), "white")
    row = {
        "id": "stable-id",
        "image": image,
        "cxsmiles_opt": "CC",
    }

    out = normalize_row(
        row,
        spec=DATASET_SPECS["mg1-synthetic"],
        split="train",
        index=47,
        images_dir=tmp_path / "images",
    )

    assert out["id"] == "stable-id"
    assert out["image"] == "images/mg1_synthetic/train/stable-id.png"
    assert (tmp_path / out["image"]).exists()
    assert out["metadata"]["upstream_id"] == "stable-id"
    assert out["metadata"]["row_index"] == 47


def test_download_dataset_selection_all_wins():
    assert selected_dataset_names(["all"]) == list(DATASET_SPECS)
    assert selected_dataset_names(["ip5m", "all"]) == list(DATASET_SPECS)
    assert selected_dataset_names(["ip5m"]) == ["ip5m"]


def test_split_name_from_expression():
    assert split_name_from_expression("train[:1]") == "train"
    assert split_name_from_expression("test") == "test"
    assert split_name_from_expression(None) is None


def test_config_validation_rows_reports_missing_config_and_splits():
    specs = [DATASET_SPECS["ip5m"], DATASET_SPECS["m2s"]]
    rows = config_validation_rows(
        specs,
        configs_by_repo={
            "docling-project/MarkushGrapher-2-Datasets": ["ip5-markush"],
        },
        splits_by_name={"ip5m": ["test"]},
    )

    assert rows[0]["config_present"] is True
    assert rows[0]["required_splits_present"] is True
    assert rows[1]["config_present"] is False


def test_config_validation_rows_accepts_archive_file_specs():
    specs = [DATASET_SPECS["molscribe-ocsr"]]
    rows = config_validation_rows(
        specs,
        configs_by_repo={},
        files_by_repo={"yujieq/MolScribe": ["README.md", "uspto_mol.zip"]},
    )

    assert rows[0]["kind"] == "archive"
    assert rows[0]["archive_filename"] == "uspto_mol.zip"
    assert rows[0]["config_present"] is True


def test_convert_molscribe_archive_dir_to_ocsr_unified_schema(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    extract_root = tmp_path / "hf"
    image_dir = extract_root / "uspto_mol" / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(image_dir / "mol1.png")
    csv_path = extract_root / "uspto_mol" / "train_680k.csv"
    csv_path.write_text(
        "image_id,file_path,raw_SMILES,SMILES\n"
        "US07793668-20100914-C00001,uspto_mol/images/mol1.png,CCO,INVALID_R_GROUP\n",
        encoding="utf-8",
    )

    manifest = convert_molscribe_archive_dir(
        extract_root,
        spec=DATASET_SPECS["molscribe-ocsr"],
        output_dir=tmp_path / "out",
        limit=1,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "molscribe-ocsr" / "unified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["splits"] == {"train": 1}
    assert rows[0]["source"] == "molscribe"
    assert rows[0]["task"] == "ocsr"
    assert rows[0]["smiles"] == "CCO"
    assert rows[0]["metadata"]["source_smiles_field"] == "raw_SMILES"
    assert rows[0]["image_has_text"] is False
    assert rows[0]["image"] == ("images/molscribe/train/US07793668-20100914-C00001.png")
    assert (tmp_path / rows[0]["image"]).exists()

    passthrough_manifest = convert_molscribe_archive_dir(
        extract_root,
        spec=DATASET_SPECS["molscribe-ocsr"],
        output_dir=tmp_path / "out_passthrough",
        limit=1,
        save_images=False,
    )
    passthrough_rows = [
        json.loads(line)
        for line in (tmp_path / "out_passthrough" / "molscribe-ocsr" / "unified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert passthrough_manifest["save_images"] is False
    assert passthrough_rows[0]["image"] == str(image_dir / "mol1.png")


def test_extract_molscribe_archive_inputs_prefers_full_train_csv(tmp_path: Path):
    archive_path = tmp_path / "uspto_mol.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "uspto_mol/train_200k.csv",
            "file_path,mol_path,raw_SMILES,SMILES\n"
            "uspto_mol/images/mol200.TIF,uspto_mol/images/mol200.MOL,CC,R_GROUP\n",
        )
        archive.writestr(
            "uspto_mol/train_680k.csv",
            "file_path,mol_path,raw_SMILES,SMILES\n"
            "uspto_mol/images/mol680.TIF,uspto_mol/images/mol680.MOL,CCO,R_GROUP\n",
        )
        archive.writestr("uspto_mol/images/mol200.TIF", "image200")
        archive.writestr("uspto_mol/images/mol200.MOL", "mol200")
        archive.writestr("uspto_mol/images/mol680.TIF", "image680")
        archive.writestr("uspto_mol/images/mol680.MOL", "mol680")

    extract_root = tmp_path / "hf"
    extraction = extract_molscribe_archive_inputs(archive_path, extract_root)
    assert extraction["selected_csvs"] == ["uspto_mol/train_680k.csv"]
    assert extraction["referenced_images"] == 1
    assert (extract_root / "uspto_mol" / "train_680k.csv").exists()
    assert not (extract_root / "uspto_mol" / "train_200k.csv").exists()
    assert (extract_root / "uspto_mol" / "images" / "mol680.TIF").exists()
    assert not (extract_root / "uspto_mol" / "images" / "mol680.MOL").exists()
    assert not (extract_root / "uspto_mol" / "images" / "mol200.TIF").exists()

    manifest = convert_molscribe_archive_dir(
        extract_root,
        spec=DATASET_SPECS["molscribe-ocsr"],
        output_dir=tmp_path / "out_selected",
        save_images=False,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "out_selected" / "molscribe-ocsr" / "unified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["splits"] == {"train": 1}
    assert rows[0]["smiles"] == "CCO"
    assert rows[0]["image"].endswith("mol680.TIF")


def test_convert_loaded_dataset_supports_worker_image_export(tmp_path: Path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    # Images materialize under GLYPH_DATA_ROOT/images; the JSONL records the
    # portable relative path that resolves back against that root.
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    source_image = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(source_image)
    dataset = datasets.Dataset.from_list(
        [
            {
                "id": "row1",
                "image": str(source_image),
                "cxsmiles_opt": "CC",
            }
        ]
    )

    manifest = convert_loaded_dataset(
        dataset,
        spec=DATASET_SPECS["uspto-mol-m"],
        output_dir=tmp_path / "out",
        forced_split_name="train",
        save_images=True,
        workers=2,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "uspto-mol-m" / "unified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["splits"] == {"train": 1}
    assert rows[0]["source"] == "uspto_mol_m"
    assert rows[0]["image"] == "images/uspto_mol_m/train/row1.png"
    assert (tmp_path / rows[0]["image"]).exists()


def _sha_by_relpath(root: Path) -> dict[str, str]:
    """SHA-256 of every PNG under ``root``, keyed by its path relative to ``root``."""

    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.png"))
    }


def _distinct_image(index: int) -> Image.Image:
    """A small, deterministic, per-index-distinct RGB image (non-uniform content)."""

    image = Image.new("RGB", (8 + index, 9 + index), (index * 7 % 256, 40, 200 - index))
    for x in range(image.width):
        image.putpixel((x, x % image.height), ((x * 13) % 256, (x * 5) % 256, index % 256))
    return image


def test_default_workers_capped() -> None:
    workers = default_workers()
    assert 1 <= workers <= 8


def test_convert_split_dataset_rows_parallel_matches_serial(tmp_path: Path) -> None:
    """Parallel image materialization is byte-identical to the serial loop."""

    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_list(
        [{"id": f"row{i}", "image": _distinct_image(i), "cxsmiles_opt": f"C{i}"} for i in range(11)]
    ).cast_column("image", datasets.Image())
    spec = DATASET_SPECS["uspto-mol-m"]

    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_rows = convert_split_dataset_rows(
        dataset, spec=spec, split="train", images_dir=serial_dir, workers=1
    )
    parallel_rows = convert_split_dataset_rows(
        dataset, spec=spec, split="train", images_dir=parallel_dir, workers=4
    )

    # Same JSONL rows, in the same (input) order.
    assert parallel_rows == serial_rows
    assert [r["id"] for r in parallel_rows] == [f"row{i}" for i in range(11)]
    # Byte-identical PNG bytes regardless of worker count.
    serial_shas = _sha_by_relpath(serial_dir)
    assert len(serial_shas) == 11
    assert _sha_by_relpath(parallel_dir) == serial_shas
    # Byte-identical serialized JSONL payload.

    def _to_jsonl(rows: list[dict]) -> str:
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)

    assert _to_jsonl(parallel_rows) == _to_jsonl(serial_rows)


def test_convert_loaded_dataset_parallel_matches_serial_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end convert_loaded_dataset: identical unified.jsonl + manifest counts."""

    datasets = pytest.importorskip("datasets")
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    dataset = datasets.Dataset.from_list(
        [{"id": f"row{i}", "image": _distinct_image(i), "cxsmiles_opt": f"C{i}"} for i in range(7)]
    ).cast_column("image", datasets.Image())
    spec = DATASET_SPECS["uspto-mol-m"]

    kwargs = {"spec": spec, "forced_split_name": "train", "save_images": True}
    serial = convert_loaded_dataset(dataset, output_dir=tmp_path / "s", workers=1, **kwargs)
    parallel = convert_loaded_dataset(dataset, output_dir=tmp_path / "p", workers=4, **kwargs)

    assert serial["splits"] == parallel["splits"] == {"train": 7}
    serial_jsonl = (tmp_path / "s" / "uspto-mol-m" / "unified.jsonl").read_bytes()
    parallel_jsonl = (tmp_path / "p" / "uspto-mol-m" / "unified.jsonl").read_bytes()
    assert serial_jsonl == parallel_jsonl


def test_convert_split_dataset_rows_parallel_survives_corrupt_image(tmp_path: Path) -> None:
    """A corrupt source image is dropped (image=None) without aborting the pool,
    and the parallel result is byte-identical to the serial one."""

    datasets = pytest.importorskip("datasets")
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(5):
        path = src / f"img{i}.png"
        _distinct_image(i).save(path)
        paths.append(str(path))
    # Corrupt the 3rd source image: not decodable, must be skipped, not fatal.
    (src / "img2.png").write_bytes(b"\x89PNG\r\nnot-a-real-image")

    dataset = datasets.Dataset.from_list(
        [{"id": f"row{i}", "image": paths[i], "cxsmiles_opt": f"C{i}"} for i in range(5)]
    )
    spec = DATASET_SPECS["uspto-mol-m"]

    serial_rows = convert_split_dataset_rows(
        dataset, spec=spec, split="train", images_dir=tmp_path / "serial", workers=1
    )
    parallel_rows = convert_split_dataset_rows(
        dataset, spec=spec, split="train", images_dir=tmp_path / "parallel", workers=4
    )

    assert parallel_rows == serial_rows
    # Every row is still emitted, in order; only the corrupt image is dropped.
    assert [r["id"] for r in parallel_rows] == [f"row{i}" for i in range(5)]
    assert parallel_rows[2]["image"] is None
    assert all(parallel_rows[i]["image"] is not None for i in (0, 1, 3, 4))
    # The 4 decodable images materialize identically under either worker count.
    assert _sha_by_relpath(tmp_path / "serial") == _sha_by_relpath(tmp_path / "parallel")
    assert len(_sha_by_relpath(tmp_path / "parallel")) == 4


# --- keep-set (index-subset) materialization --------------------------------


def test_convert_split_dataset_rows_keep_ids_materializes_only_referenced(tmp_path: Path) -> None:
    """Only the index-referenced source_ids are materialized; kept output is
    byte-identical to the full pull, in both the serial and parallel paths."""

    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_list(
        [{"id": f"row{i}", "image": _distinct_image(i), "cxsmiles_opt": f"C{i}"} for i in range(8)]
    ).cast_column("image", datasets.Image())
    spec = DATASET_SPECS["uspto-mol-m"]

    full_dir = tmp_path / "full"
    full_rows = convert_split_dataset_rows(
        dataset, spec=spec, split="train", images_dir=full_dir, workers=1
    )
    full_by_id = {row["id"]: row for row in full_rows}
    full_shas = _sha_by_relpath(full_dir)

    keep = {"row1", "row4", "row6"}
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial = convert_split_dataset_rows(
        dataset, spec=spec, split="train", images_dir=serial_dir, workers=1, keep_ids=keep
    )
    parallel = convert_split_dataset_rows(
        dataset, spec=spec, split="train", images_dir=parallel_dir, workers=4, keep_ids=keep
    )

    # Never-referenced rows are dropped; kept rows stay in input order; the
    # parallel path agrees with the serial path row-for-row.
    assert [row["id"] for row in serial] == ["row1", "row4", "row6"]
    assert parallel == serial

    # A referenced id is present on disk; a non-referenced id is absent.
    serial_shas = _sha_by_relpath(serial_dir)
    assert set(serial_shas) == {f"uspto_mol_m/train/{rid}.png" for rid in keep}
    assert "uspto_mol_m/train/row0.png" not in serial_shas

    # Kept rows are byte-identical to the full pull (PNG bytes + JSONL row).
    for rid in keep:
        rel = f"uspto_mol_m/train/{rid}.png"
        assert serial_shas[rel] == full_shas[rel]
        assert next(row for row in serial if row["id"] == rid) == full_by_id[rid]
    assert _sha_by_relpath(parallel_dir) == serial_shas


def test_convert_loaded_dataset_keep_ids_reports_missing_coverage(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A referenced source_id absent from the upstream source is surfaced
    (warning + manifest count), not silently dropped."""

    datasets = pytest.importorskip("datasets")
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    dataset = datasets.Dataset.from_list(
        [{"id": f"row{i}", "image": _distinct_image(i), "cxsmiles_opt": f"C{i}"} for i in range(4)]
    ).cast_column("image", datasets.Image())
    spec = DATASET_SPECS["uspto-mol-m"]

    keep = {"row1", "row2", "GHOST"}
    with caplog.at_level("WARNING"):
        manifest = convert_loaded_dataset(
            dataset,
            spec=spec,
            output_dir=tmp_path / "out",
            forced_split_name="train",
            save_images=True,
            workers=1,
            keep_ids=keep,
        )

    assert (manifest["keep_requested"], manifest["keep_matched"], manifest["keep_missing"]) == (
        3,
        2,
        1,
    )
    assert "not found" in caplog.text
    assert "GHOST" in caplog.text
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "uspto-mol-m" / "unified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["id"] for row in rows] == ["row1", "row2"]


def _write_molscribe_zip(path: Path, count: int, *, with_image_id: bool) -> None:
    """Write a small MolScribe-style archive of TIF images + one train CSV."""

    header = (
        "image_id,file_path,raw_SMILES,SMILES" if with_image_id else "file_path,raw_SMILES,SMILES"
    )
    lines = [header]
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(count):
            member = f"uspto_mol/images/mol{i}.TIF"
            buffer = io.BytesIO()
            _distinct_image(i).save(buffer, format="TIFF")
            archive.writestr(member, buffer.getvalue())
            if with_image_id:
                lines.append(f"ID{i:03d},{member},C{'C' * i},RG{i}")
            else:
                lines.append(f"{member},C{'C' * i},RG{i}")
        archive.writestr("uspto_mol/train_680k.csv", "\n".join(lines) + "\n")


@pytest.mark.parametrize("with_image_id", [True, False])
def test_molscribe_keep_ids_extracts_and_materializes_only_referenced(
    tmp_path: Path, monkeypatch, with_image_id: bool
) -> None:
    """The MolScribe archive path unzips + materializes only index-referenced
    images (id from an ``image_id`` column or the image-path stem), byte-identical
    to the full pull, and surfaces a missing id."""

    spec = DATASET_SPECS["molscribe-ocsr"]
    archive = tmp_path / "uspto_mol.zip"
    _write_molscribe_zip(archive, 6, with_image_id=with_image_id)
    referenced = ("ID001", "ID004") if with_image_id else ("mol1", "mol4")
    absent = "ID000" if with_image_id else "mol0"

    # Baseline full pull under its own data root for byte-identity reference.
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path / "full"))
    full_root = tmp_path / "full_hf"
    extract_molscribe_archive_inputs(archive, full_root)
    convert_molscribe_archive_dir(full_root, spec=spec, output_dir=tmp_path / "full_out")
    full_shas = _sha_by_relpath(tmp_path / "full" / "images")
    assert len(full_shas) == 6

    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path / "sub"))
    keep = {*referenced, "GHOST"}
    sub_root = tmp_path / "sub_hf"
    extraction = extract_molscribe_archive_inputs(archive, sub_root, spec=spec, keep_ids=keep)
    manifest = convert_molscribe_archive_dir(
        sub_root, spec=spec, output_dir=tmp_path / "sub_out", keep_ids=keep
    )

    # Extraction unzips only the referenced images (2 of 6), not the whole tree.
    assert extraction["referenced_images"] == 2
    assert {p.name for p in sub_root.rglob("*.TIF")} == {"mol1.TIF", "mol4.TIF"}

    # Conversion materializes only the referenced ids; a non-referenced id is absent.
    rows = [
        json.loads(line)
        for line in (tmp_path / "sub_out" / spec.name / "unified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["id"] for row in rows] == list(referenced)
    sub_shas = _sha_by_relpath(tmp_path / "sub" / "images")
    assert set(sub_shas) == {f"molscribe/train/{rid}.png" for rid in referenced}
    assert f"molscribe/train/{absent}.png" not in sub_shas
    for rel, sha in sub_shas.items():
        assert sha == full_shas[rel]

    # The missing (index-referenced but upstream-absent) id is counted, not dropped.
    assert (manifest["keep_requested"], manifest["keep_matched"], manifest["keep_missing"]) == (
        3,
        2,
        1,
    )


def test_prepare_sft_prefers_combined_unified_jsonl(tmp_path: Path):
    combined = tmp_path / "all_unified.jsonl"
    combined.write_text("")
    nested = tmp_path / "ip5m" / "unified.jsonl"
    nested.parent.mkdir()
    nested.write_text("")

    assert discover_jsonl_inputs([tmp_path]) == [combined]


def test_prepare_sft_prefers_dataset_unified_jsonl_over_split_files(tmp_path: Path):
    dataset = tmp_path / "mg1-synthetic"
    dataset.mkdir()
    train = dataset / "train.jsonl"
    test = dataset / "test.jsonl"
    unified = dataset / "unified.jsonl"
    train.write_text("")
    test.write_text("")
    unified.write_text("")

    assert discover_jsonl_inputs([dataset]) == [unified]


def test_prepare_sft_excludes_eval_sources_by_default():
    train = {
        "id": "train_a",
        "source": "uspto_mol_m",
        "split": "train",
        "task": "markush",
        "image": "images/a.png",
        "image_has_text": False,
        "cxsmiles_opt": "CC",
    }
    eval_row = {
        **train,
        "id": "eval_ip5m",
        "source": "ip5m",
        "split": "test",
    }

    assert is_trainable_source(train)
    assert not is_trainable_source(eval_row)


def test_prepare_sft_mode_selection_and_text_dropout():
    record = {
        "id": "r1",
        "source": "uspto_mol_m",
        "split": "train",
        "task": "markush",
        "image": "images/r1.png",
        "image_has_text": False,
        "description": "wherein R1 is alkyl",
        "annotation": "<markush><cxsmi><r>R1</r>CC</cxsmi><stable>R1:alkyl</stable></markush>",
    }

    assert eligible_modes(record) == {STRUCTURE_ONLY_MARKUSH, TEXT_CONDITIONED_MARKUSH}
    c_example = build_training_example(
        record,
        TEXT_CONDITIONED_MARKUSH,
        text_dropout_rate=0.0,
        rng=random.Random(1),
    )
    assert "Patent text: wherein R1 is alkyl" in c_example["messages"][0]["content"]
    assert "the <cxsmi> value must use cxsmiles_opt" in c_example["messages"][0]["content"]
    dropped = build_training_example(
        record,
        TEXT_CONDITIONED_MARKUSH,
        text_dropout_rate=1.0,
        rng=random.Random(1),
    )
    assert dropped["effective_mode"] == STRUCTURE_ONLY_MARKUSH
    assert "Patent text:" not in dropped["messages"][0]["content"]


def test_prepare_sft_target_prefers_cxsmiles_opt_and_keeps_stable_text():
    record = {
        "id": "r1",
        "source": "mg1_synthetic",
        "split": "train",
        "task": "markush",
        "image": "images/r1.png",
        "image_has_text": True,
        "cxsmiles_opt": "<r>R1</r>CC",
        "annotation": (
            "<markush><cxsmi>WRONG_RENDERED_CXSMILES</cxsmi><stable>R1:alkyl</stable></markush>"
        ),
    }

    example = build_training_example(
        record,
        STRUCTURE_ONLY_MARKUSH,
        rng=random.Random(1),
    )

    assert example["messages"][1]["content"] == (
        "<markush><cxsmi><r>R1</r>CC</cxsmi><stable>R1:alkyl</stable></markush>"
    )
    assert example["target_source"] == "cxsmiles_opt"
    assert example["target_format"] == "cxsmiles_opt"
    assert "the <cxsmi> value must use cxsmiles_opt" in example["messages"][0]["content"]


def test_prepare_sft_normalizes_standard_cxsmiles_fallback_to_opt_convention():
    record = {
        "id": "r1",
        "source": "mg1_synthetic",
        "split": "train",
        "task": "markush",
        "image": "images/r1.png",
        "image_has_text": False,
        "cxsmiles": "*CC |$R1;;$|",
    }

    example = build_training_example(
        record,
        STRUCTURE_ONLY_MARKUSH,
        rng=random.Random(1),
    )

    assert example["messages"][1]["content"] == (
        "<markush><cxsmi><r>R1</r>CC</cxsmi><stable></stable></markush>"
    )
    assert example["target_source"] == "cxsmiles->cxsmiles_opt"
    assert example["target_format"] == "cxsmiles_opt"
    assert "the <cxsmi> value must use cxsmiles_opt" in example["messages"][0]["content"]


def test_prepare_sft_keeps_native_molparser_markush_separate_from_shared_target():
    record = {
        "id": "molparser-markush_train_000000",
        "source": "molparser",
        "split": "train",
        "task": "markush",
        "image": "images/molparser.png",
        "image_has_text": False,
        "smiles": "CCO<sep><a>2:<dum></a>",
        "metadata": {
            "dataset": "molparser-markush",
            "repo": "UniParser/MolParser-7M",
            "config": "sft_real",
            "eval_only": False,
        },
    }

    assert is_trainable_source(record)
    assert target_spec_for_record(record) is None
    native_target = target_spec_for_record(record, example_mode=MOLPARSER_NATIVE_MARKUSH)
    assert native_target is not None
    assert native_target.source == "molparser_esmiles"
    assert native_target.format_label == "MolParser E-SMILES"
    assert native_target.content == "CCO<sep><a>2:<dum></a>"
    assert eligible_modes(record) == {MOLPARSER_NATIVE_MARKUSH}


def test_prepare_sft_converts_supported_molparser_markush_to_explicit_mode():
    record = {
        "id": "molparser-markush_train_000001",
        "source": "molparser",
        "split": "train",
        "task": "markush",
        "image": "images/molparser.png",
        "image_has_text": False,
        "smiles": "*CC<sep><a>0:R[1]</a>",
        "molparser_esmiles": "*CC<sep><a>0:R[1]</a>",
        "metadata": {
            "dataset": "molparser-markush",
            "repo": "UniParser/MolParser-7M",
            "config": "sft_real",
            "eval_only": False,
        },
    }

    target = target_spec_for_record(record)
    assert target is not None
    assert target.source == "molparser_esmiles->cxsmiles_opt"
    assert target.format_label == "cxsmiles_opt"
    assert target.conversion_status == "converted_to_cxsmiles_opt"
    assert "<cxsmi><r>R[1]</r>CC</cxsmi>" in target.content
    assert eligible_modes(record) == {MOLPARSER_CONVERTED_MARKUSH, MOLPARSER_NATIVE_MARKUSH}

    example = build_training_example(record, MOLPARSER_CONVERTED_MARKUSH, rng=random.Random(1))
    assert example["mode"] == MOLPARSER_CONVERTED_MARKUSH
    assert example["target_source"] == "molparser_esmiles->cxsmiles_opt"
    assert example["target_format"] == "cxsmiles_opt"
    assert example["target_conversion_status"] == "converted_to_cxsmiles_opt"
    assert "the <cxsmi> value must use cxsmiles_opt" in example["messages"][0]["content"]


def test_prepare_sft_can_emit_native_molparser_markush_mode():
    record = {
        "id": "molparser-markush_train_000000",
        "source": "molparser",
        "split": "train",
        "task": "markush",
        "image": "images/molparser.png",
        "image_has_text": False,
        "molparser_esmiles": "C1(*)SCCS1<sep><a>1:<dum></a>",
        "metadata": {
            "dataset": "molparser-markush",
            "repo": "UniParser/MolParser-7M",
            "config": "sft_real",
            "eval_only": False,
        },
    }

    example = build_training_example(record, MOLPARSER_NATIVE_MARKUSH, rng=random.Random(1))

    assert example["mode"] == MOLPARSER_NATIVE_MARKUSH
    assert example["target_source"] == "molparser_esmiles"
    assert example["target_format"] == "MolParser E-SMILES"
    assert example["target_conversion_status"] == "native_molparser_esmiles"
    assert example["messages"][1]["content"] == "C1(*)SCCS1<sep><a>1:<dum></a>"
    assert "Target format: return MolParser E-SMILES" in example["messages"][0]["content"]

    examples, stats = mix_examples(
        [record],
        weights=MixWeights(
            structure_only_markush=0,
            patent_image_markush=0,
            text_conditioned_markush=0,
            molecule_ocsr=0,
            molparser_native_markush=1,
            molparser_converted_markush=0,
        ),
        total_samples=1,
        seed=3,
    )

    assert examples[0]["mode"] == MOLPARSER_NATIVE_MARKUSH
    assert stats["mode_counts"] == {MOLPARSER_NATIVE_MARKUSH: 1}
    assert stats["target_source_counts"] == {"molparser_esmiles": 1}
    assert stats["target_format_counts"] == {"MolParser E-SMILES": 1}
    assert stats["target_conversion_status_counts"] == {"native_molparser_esmiles": 1}


def test_mix_examples_includes_converted_molparser_only_when_weighted():
    record = {
        "id": "molparser-markush_train_000001",
        "source": "molparser",
        "split": "train",
        "task": "markush",
        "image": "images/molparser.png",
        "image_has_text": False,
        "molparser_esmiles": "*CC<sep><a>0:R</a>",
    }

    with pytest.raises(ValueError, match="No eligible training examples"):
        mix_examples([record], total_samples=None, seed=3)

    examples, stats = mix_examples(
        [record],
        weights=MixWeights(
            structure_only_markush=0,
            patent_image_markush=0,
            text_conditioned_markush=0,
            molecule_ocsr=0,
            molparser_native_markush=0,
            molparser_converted_markush=1,
        ),
        total_samples=1,
        seed=3,
    )

    assert len(examples) == 1
    assert examples[0]["mode"] == MOLPARSER_CONVERTED_MARKUSH
    assert stats["mode_counts"] == {MOLPARSER_CONVERTED_MARKUSH: 1}
    assert stats["target_source_counts"] == {"molparser_esmiles->cxsmiles_opt": 1}
    assert stats["target_conversion_status_counts"] == {"converted_to_cxsmiles_opt": 1}


def test_prepare_sft_prompt_accepts_explicit_target_format_override():
    record = {
        "id": "r1",
        "source": "mg1_synthetic",
        "split": "train",
        "task": "markush",
        "image": "images/r1.png",
        "image_has_text": False,
        "cxsmiles_opt": "<r>R1</r>CC",
        "target_cxsmiles_format": "canonicalized cxsmiles_opt",
    }

    example = build_training_example(
        record,
        STRUCTURE_ONLY_MARKUSH,
        rng=random.Random(1),
    )

    assert example["target_source"] == "cxsmiles_opt"
    assert example["target_format"] == "canonicalized cxsmiles_opt"
    assert "canonicalized cxsmiles_opt" in example["messages"][0]["content"]


def test_prepare_sft_can_emit_verified_canonicalized_markush_target():
    record = {
        "id": "r1",
        "source": "mg1_synthetic",
        "split": "train",
        "task": "markush",
        "image": "images/r1.png",
        "image_has_text": False,
        "cxsmiles_opt": "C(C)O",
    }

    example = build_training_example(
        record,
        STRUCTURE_ONLY_MARKUSH,
        canonicalize_markush=True,
        rng=random.Random(1),
    )

    assert example["messages"][1]["content"] == (
        "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
    )
    assert example["target_source"] == "cxsmiles_opt->canonicalized_cxsmiles_opt"
    assert example["target_format"] == "canonicalized cxsmiles_opt"
    assert example["target_canonicalization_status"] == "canonicalized"
    assert example["target_canonicalization_error"] == ""
    assert "canonicalized cxsmiles_opt" in example["messages"][0]["content"]


def test_prepare_sft_can_emit_structured_transcript_markush_target():
    record = {
        "id": "r1",
        "source": "mg1_synthetic",
        "split": "train",
        "task": "markush",
        "image": "images/r1.png",
        "image_has_text": False,
        "cxsmiles_opt": "<r>R1</r>CC",
    }

    example = build_training_example(
        record,
        STRUCTURED_TRANSCRIPT_MARKUSH,
        rng=random.Random(1),
    )

    target = example["messages"][1]["content"]
    transcript_json, final_xml = target.split("\n", 1)
    transcript = json.loads(transcript_json)
    ok, error = validate_structured_transcript_content(target)
    assert ok, error
    assert transcript["schema_version"] == SCHEMA_VERSION
    assert transcript["target_format"] == "canonicalized cxsmiles_opt"
    assert final_xml == "<markush><cxsmi><r>R1</r>CC</cxsmi><stable></stable></markush>"
    assert common.prediction_to_cxsmiles_opt(target) == "<r>R1</r>CC"
    assert example["mode"] == STRUCTURED_TRANSCRIPT_MARKUSH
    assert example["target_source"] == (
        "cxsmiles_opt->canonicalized_cxsmiles_opt->structured_transcript"
    )
    assert example["target_format"] == ("structured transcript + final canonicalized cxsmiles_opt")
    assert f"schema {SCHEMA_VERSION}" in example["messages"][0]["content"]
    assert (
        "final <cxsmi> value must use canonicalized cxsmiles_opt"
        in example["messages"][0]["content"]
    )


def test_prepare_sft_cli_exposes_canonicalized_target_flags(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "unified.jsonl").write_text(
        json.dumps(
            {
                "id": "good",
                "source": "uspto_mol_m",
                "split": "train",
                "task": "markush",
                "image": "images/good.png",
                "image_has_text": False,
                "cxsmiles_opt": "C(C)O",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"
    stats = tmp_path / "stats.json"

    result = CliRunner().invoke(
        data_cli_app,
        [
            "prepare-sft",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--stats-json",
            str(stats),
            "--total-samples",
            "1",
            "--structure-only-markush-fraction",
            "1",
            "--patent-image-markush-fraction",
            "0",
            "--text-conditioned-markush-fraction",
            "0",
            "--molecule-ocsr-fraction",
            "0",
            "--canonicalize-markush-targets",
            "--require-canonical-markush-targets",
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["target_format"] == "canonicalized cxsmiles_opt"
    assert row["messages"][1]["content"] == (
        "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
    )
    assert json.loads(stats.read_text(encoding="utf-8"))["target_format_counts"] == {
        "canonicalized cxsmiles_opt": 1
    }


def test_prepare_sft_cli_exposes_structured_transcript_fraction(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "unified.jsonl").write_text(
        json.dumps(
            {
                "id": "structured",
                "source": "uspto_mol_m",
                "split": "train",
                "task": "markush",
                "image": "images/structured.png",
                "image_has_text": False,
                "cxsmiles_opt": "<r>R1</r>CC",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"
    stats = tmp_path / "stats.json"

    result = CliRunner().invoke(
        data_cli_app,
        [
            "prepare-sft",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--stats-json",
            str(stats),
            "--total-samples",
            "1",
            "--structure-only-markush-fraction",
            "0",
            "--patent-image-markush-fraction",
            "0",
            "--text-conditioned-markush-fraction",
            "0",
            "--molecule-ocsr-fraction",
            "0",
            "--structured-transcript-markush-fraction",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    ok, error = validate_structured_transcript_content(row["messages"][1]["content"])
    assert ok, error
    assert row["mode"] == STRUCTURED_TRANSCRIPT_MARKUSH
    assert row["target_format"] == "structured transcript + final canonicalized cxsmiles_opt"
    assert common.prediction_to_cxsmiles_opt(row["messages"][1]["content"]) == "<r>R1</r>CC"
    assert json.loads(stats.read_text(encoding="utf-8"))["mode_counts"] == {
        STRUCTURED_TRANSCRIPT_MARKUSH: 1
    }


def test_generate_failure_seeds_cli_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    scaffold_pool = tmp_path / "pool.txt"
    output_jsonl = tmp_path / "seeds.jsonl"
    output_csv = tmp_path / "seeds.csv"
    stats_json = tmp_path / "stats.json"

    class FakeSample:
        focus = "sgroup"

    def fake_load_scaffold_pool(path: Path) -> list[str]:
        captured["scaffold_pool"] = path
        return ["CCO"]

    def fake_generate_failure_mode_decorated_samples(
        scaffolds,
        *,
        samples_per_focus,
        focuses,
        seed,
        max_attempts_per_sample,
        progress_interval,
    ):
        captured["generate_args"] = {
            "scaffolds": scaffolds,
            "samples_per_focus": samples_per_focus,
            "focuses": focuses,
            "seed": seed,
            "max_attempts_per_sample": max_attempts_per_sample,
            "progress_interval": progress_interval,
        }
        return [FakeSample()], {"generated": 1}

    def fake_write_failure_mode_seed_outputs(
        focused_samples,
        *,
        output_jsonl,
        output_csv,
        id_prefix,
    ) -> None:
        captured["write_args"] = {
            "focused_samples": focused_samples,
            "output_jsonl": output_jsonl,
            "output_csv": output_csv,
            "id_prefix": id_prefix,
        }
        output_jsonl.write_text('{"id":"failure_sgroup_000000"}\n', encoding="utf-8")
        output_csv.write_text("id,cxsmiles\nfailure_sgroup_000000,CC\n", encoding="utf-8")

    monkeypatch.setattr("glyph.markush.data.cli.load_scaffold_pool", fake_load_scaffold_pool)
    monkeypatch.setattr(
        "glyph.markush.data.cli.generate_failure_mode_decorated_samples",
        fake_generate_failure_mode_decorated_samples,
    )
    monkeypatch.setattr(
        "glyph.markush.data.cli.write_failure_mode_seed_outputs",
        fake_write_failure_mode_seed_outputs,
    )

    result = CliRunner().invoke(
        data_cli_app,
        [
            "generate",
            "failure-seeds",
            "--scaffold-pool",
            str(scaffold_pool),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--samples-per-focus",
            "2",
            "--focus",
            "sgroup",
            "--seed",
            "72",
            "--decorate-attempts-per-sample",
            "100",
            "--decorate-progress-interval",
            "10",
            "--id-prefix",
            "failmode",
            "--stats-json",
            str(stats_json),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["scaffold_pool"] == scaffold_pool
    assert captured["generate_args"] == {
        "scaffolds": ["CCO"],
        "samples_per_focus": 2,
        "focuses": ["sgroup"],
        "seed": 72,
        "max_attempts_per_sample": 100,
        "progress_interval": 10,
    }
    assert captured["write_args"]["output_jsonl"] == output_jsonl
    assert captured["write_args"]["output_csv"] == output_csv
    assert captured["write_args"]["id_prefix"] == "failmode"
    assert output_jsonl.exists()
    assert output_csv.exists()
    assert json.loads(stats_json.read_text(encoding="utf-8"))["focus_counts"] == {"sgroup": 1}


def test_prepare_sft_keeps_truthful_raw_target_when_canonicalization_falls_back():
    cxsmiles = (
        "<r>R1</r>C.<r>R2</r>C.<r>R3</r>C(C)(C)C(CC(=O)O)"
        "NC1=C(F)C=<r>X</r>C(C2=CNC3=<r>Y</r>C=<r>Z</r>C=C23)=N1"
        "|m:1:23.24.25.26.27.28,m:3:14.15.17.18.19.29"
    )
    record = {
        "id": "r1",
        "source": "mg1_synthetic",
        "split": "train",
        "task": "markush",
        "image": "images/r1.png",
        "image_has_text": False,
        "cxsmiles_opt": cxsmiles,
    }

    example = build_training_example(
        record,
        STRUCTURE_ONLY_MARKUSH,
        canonicalize_markush=True,
        rng=random.Random(1),
    )

    assert f"<cxsmi>{cxsmiles}</cxsmi>" in example["messages"][1]["content"]
    assert example["target_source"] == "cxsmiles_opt"
    assert example["target_format"] == "cxsmiles_opt"
    assert example["target_canonicalization_status"] == "fallback_unverified"
    assert example["target_canonicalization_error"]
    assert "the <cxsmi> value must use cxsmiles_opt" in example["messages"][0]["content"]


def test_prepare_sft_can_require_verified_canonical_markush_targets():
    rows = [
        {
            "id": "good",
            "source": "mg1_synthetic",
            "split": "train",
            "task": "markush",
            "image": "images/good.png",
            "image_has_text": False,
            "cxsmiles_opt": "C(C)O",
        },
        {
            "id": "bad",
            "source": "mg1_synthetic",
            "split": "train",
            "task": "markush",
            "image": "images/bad.png",
            "image_has_text": False,
            "cxsmiles_opt": "not_a_smiles_XYZ123",
        },
    ]

    examples, stats = mix_examples(
        rows,
        weights=MixWeights(
            structure_only_markush=1,
            patent_image_markush=0,
            text_conditioned_markush=0,
            molecule_ocsr=0,
        ),
        total_samples=1,
        seed=3,
        canonicalize_markush_targets=True,
        require_canonical_markush_targets=True,
    )

    assert len(examples) == 1
    assert examples[0]["source_id"] == "good"
    assert examples[0]["target_format"] == "canonicalized cxsmiles_opt"
    assert stats["skipped_records"] == 1
    assert stats["target_canonicalization_status_counts"] == {"canonicalized": 1}


def test_mix_examples_respects_requested_counts_with_oversampling():
    records = [
        {
            "id": "a",
            "source": "pipeline_generated",
            "split": "train",
            "task": "markush",
            "image": "images/a.png",
            "image_has_text": False,
            "cxsmiles_opt": "CC",
        },
        {
            "id": "b",
            "source": "pipeline_generated",
            "split": "train",
            "task": "markush",
            "image": "images/b.png",
            "image_has_text": True,
            "cxsmiles_opt": "CC",
        },
        {
            "id": "d",
            "source": "molscribe",
            "split": "train",
            "task": "ocsr",
            "image": "images/d.png",
            "smiles": "CCO",
        },
    ]

    examples, stats = mix_examples(
        records,
        weights=MixWeights(
            structure_only_markush=1,
            patent_image_markush=1,
            text_conditioned_markush=0,
            molecule_ocsr=1,
        ),
        total_samples=6,
        text_dropout_rate=0.0,
        seed=3,
    )

    assert len(examples) == 6
    assert stats["requested_counts"] == {
        STRUCTURE_ONLY_MARKUSH: 2,
        PATENT_IMAGE_MARKUSH: 2,
        TEXT_CONDITIONED_MARKUSH: 0,
        MOLECULE_OCSR: 2,
        MOLPARSER_NATIVE_MARKUSH: 0,
        MOLPARSER_CONVERTED_MARKUSH: 0,
        STRUCTURED_TRANSCRIPT_MARKUSH: 0,
    }
    assert stats["mode_counts"] == {
        STRUCTURE_ONLY_MARKUSH: 2,
        PATENT_IMAGE_MARKUSH: 2,
        MOLECULE_OCSR: 2,
    }
    assert TEXT_CONDITIONED_MARKUSH not in stats["shortages"]


def test_mix_examples_without_total_samples_emits_each_eligible_mode_once():
    records = [
        {
            "id": "patent",
            "source": "mg1_synthetic",
            "split": "train",
            "task": "markush",
            "image": "images/patent.png",
            "image_has_text": True,
            "cxsmiles_opt": "CC",
        },
        {
            "id": "structure",
            "source": "uspto_mol_m",
            "split": "train",
            "task": "markush",
            "image": "images/structure.png",
            "image_has_text": False,
            "cxsmiles_opt": "CCO",
        },
    ]

    examples, stats = mix_examples(
        records,
        weights=MixWeights(
            structure_only_markush=0,
            patent_image_markush=0,
            text_conditioned_markush=1,
            molecule_ocsr=0,
        ),
        total_samples=None,
        text_dropout_rate=0.0,
        seed=3,
    )

    assert len(examples) == 2
    assert stats["requested_counts"] == {
        STRUCTURE_ONLY_MARKUSH: 1,
        PATENT_IMAGE_MARKUSH: 1,
        TEXT_CONDITIONED_MARKUSH: 0,
        MOLECULE_OCSR: 0,
    }
    assert stats["mode_counts"] == {
        STRUCTURE_ONLY_MARKUSH: 1,
        PATENT_IMAGE_MARKUSH: 1,
    }
    assert stats["shortages"] == {}


def test_mix_examples_workers_match_serial_for_deterministic_rows():
    records = [
        {
            "id": "patent",
            "source": "mg1_synthetic",
            "split": "train",
            "task": "markush",
            "image": "images/patent.png",
            "image_has_text": True,
            "cxsmiles_opt": "CC",
        },
        {
            "id": "structure",
            "source": "uspto_mol_m",
            "split": "train",
            "task": "markush",
            "image": "images/structure.png",
            "image_has_text": False,
            "cxsmiles_opt": "CCO",
        },
        {
            "id": "ocsr",
            "source": "molscribe",
            "split": "train",
            "task": "ocsr",
            "image": "images/ocsr.png",
            "smiles": "CCN",
        },
    ]
    weights = MixWeights(
        structure_only_markush=1,
        patent_image_markush=1,
        text_conditioned_markush=0,
        molecule_ocsr=1,
    )

    serial_examples, serial_stats = mix_examples(
        records,
        weights=weights,
        total_samples=None,
        text_dropout_rate=0.0,
        seed=9,
        workers=1,
    )
    worker_examples, worker_stats = mix_examples(
        records,
        weights=weights,
        total_samples=None,
        text_dropout_rate=0.0,
        seed=9,
        workers=2,
    )

    def _stable(rows: list[dict]) -> list[dict]:
        return sorted(rows, key=lambda row: row["id"])

    assert _stable(worker_examples) == _stable(serial_examples)
    assert worker_stats == serial_stats


def test_mix_examples_rejects_invalid_worker_count():
    with pytest.raises(ValueError, match="workers must be positive"):
        mix_examples([], workers=0)


def test_mix_examples_does_not_report_zero_count_shortages():
    records = [
        {
            "id": "a",
            "source": "pipeline_generated",
            "split": "train",
            "task": "markush",
            "image": "images/a.png",
            "image_has_text": False,
            "cxsmiles_opt": "CC",
        }
    ]

    examples, stats = mix_examples(
        records,
        weights=MixWeights(
            structure_only_markush=1,
            patent_image_markush=0,
            text_conditioned_markush=0,
            molecule_ocsr=0,
        ),
        total_samples=1,
        text_dropout_rate=0.0,
        seed=3,
    )

    assert len(examples) == 1
    assert stats["shortages"] == {}


def test_prepare_sft_jsonl_shape(tmp_path: Path):
    record = {
        "id": "d1",
        "source": "molscribe",
        "split": "train",
        "task": "ocsr",
        "image": "images/d1.png",
        "smiles": "CCO",
    }
    example = build_training_example(record, MOLECULE_OCSR, rng=random.Random(1))

    out = tmp_path / "train.jsonl"
    out.write_text(json.dumps(example) + "\n")
    loaded = json.loads(out.read_text())
    assert loaded["messages"][0]["content"] == (
        "<image>\nExtract the molecular structure from this image as SMILES.\n"
        "Target format: return a SMILES string."
    )
    assert loaded["messages"][1]["content"] == "CCO"
    assert loaded["target_source"] == "smiles->rdkit_canonical_isomeric_smiles"
    assert loaded["target_format"] == "SMILES"
    assert loaded["target_canonicalization_status"] == "already_canonical"


def test_prepare_sft_canonicalizes_molecule_ocsr_smiles():
    record = {
        "id": "d1",
        "source": "molscribe",
        "split": "train",
        "task": "ocsr",
        "image": "images/d1.png",
        "smiles": "C(C)O",
    }

    target = target_spec_for_record(record, example_mode=MOLECULE_OCSR)
    assert target is not None
    assert target.content == "CCO"
    assert target.source == "smiles->rdkit_canonical_isomeric_smiles"
    assert target.format_label == "SMILES"
    assert target.canonicalization_status == "canonicalized"

    example = build_training_example(record, MOLECULE_OCSR, rng=random.Random(1))
    assert example["messages"][1]["content"] == "CCO"
    assert example["target_source"] == "smiles->rdkit_canonical_isomeric_smiles"
    assert example["target_canonicalization_status"] == "canonicalized"


def test_prepare_sft_rejects_invalid_molecule_ocsr_smiles():
    record = {
        "id": "d1",
        "source": "molscribe",
        "split": "train",
        "task": "ocsr",
        "image": "images/d1.png",
        "smiles": "not_a_smiles_XYZ123",
    }

    assert target_spec_for_record(record, example_mode=MOLECULE_OCSR) is None
    assert eligible_modes(record) == set()


def test_prepare_sft_excludes_audited_bad_rows(tmp_path: Path):
    data = tmp_path / "unified.jsonl"
    rows = [
        {
            "id": "bad",
            "source": "pipeline_generated",
            "split": "train",
            "task": "markush",
            "image": "images/bad.png",
            "image_has_text": False,
            "cxsmiles_opt": "<r>R1</r>bad",
        },
        {
            "id": "good",
            "source": "pipeline_generated",
            "split": "train",
            "task": "markush",
            "image": "images/good.png",
            "image_has_text": False,
            "cxsmiles_opt": "<r>R1</r>CC",
        },
    ]
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = tmp_path / "audit_report.json"
    report.write_text(
        json.dumps(
            {
                "training": {
                    "exclusion_candidates": [
                        {"source": "pipeline_generated", "split": "train", "id": "bad"}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    exclusion_list = tmp_path / "exclusions.json"
    exclusion_list.write_text(
        json.dumps([{"source": "pipeline_generated", "split": "train", "id": "bad"}]),
        encoding="utf-8",
    )

    assert load_exclusion_keys([report]) == {("pipeline_generated", "train", "bad")}
    assert load_exclusion_keys([exclusion_list]) == {("pipeline_generated", "train", "bad")}
    out = tmp_path / "sft.jsonl"
    stats = prepare_sft_jsonl(
        datasets=[data],
        output=out,
        weights=MixWeights(
            structure_only_markush=1,
            patent_image_markush=0,
            text_conditioned_markush=0,
            molecule_ocsr=0,
        ),
        total_samples=1,
        exclude_files=[report],
        oversample=False,
    )

    examples = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stats["excluded_records"] == 1
    assert examples[0]["source_id"] == "good"


def test_prepare_sft_consumes_molparser_materialized_exclusions(tmp_path: Path):
    data = tmp_path / "molparser_unified.jsonl"
    rows = [
        {
            "id": "bad",
            "source": "molparser",
            "split": "train",
            "task": "markush",
            "image": "images/bad.png",
            "image_has_text": False,
            "molparser_esmiles": "CCO<sep>",
        },
        {
            "id": "good",
            "source": "molparser",
            "split": "train",
            "task": "markush",
            "image": "images/good.png",
            "image_has_text": False,
            "molparser_esmiles": "CCN<sep>",
        },
    ]
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(
        json.dumps(
            [
                {
                    "source": "molparser",
                    "split": "train",
                    "id": "bad",
                    "reason": "parse_error",
                    "target_source": "molparser_esmiles",
                }
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "sft.jsonl"
    stats = prepare_sft_jsonl(
        datasets=[data],
        output=out,
        weights=MixWeights(
            structure_only_markush=0,
            patent_image_markush=0,
            text_conditioned_markush=0,
            molecule_ocsr=0,
            molparser_native_markush=1,
            molparser_converted_markush=0,
        ),
        total_samples=2,
        exclude_files=[exclusions],
        oversample=False,
    )

    examples = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stats["excluded_records"] == 1
    assert stats["exclude_keys"] == 1
    assert stats["pool_sizes"][MOLPARSER_NATIVE_MARKUSH] == 1
    assert stats["mode_counts"] == {MOLPARSER_NATIVE_MARKUSH: 1}
    assert examples[0]["source_id"] == "good"
    assert examples[0]["target_source"] == "molparser_esmiles"


def test_convert_existing_chat_jsonl_to_unified_record():
    record = {
        "id": "local_1",
        "messages": [
            {
                "role": "user",
                "content": "<image>\nPatent text: wherein R1 is alkyl\nExtract the Markush.",
            },
            {
                "role": "assistant",
                "content": (
                    "<markush><cxsmi><r>R1</r>CC</cxsmi><stable>R1:alkyl</stable></markush>"
                ),
            },
        ],
        "images": ["generated/pages/local_1.png"],
    }

    out = normalize_existing_record(
        record,
        source="pipeline_generated",
        split="train",
        task="markush",
        image_has_text=True,
        index=0,
    )

    assert out["image"] == "generated/pages/local_1.png"
    assert out["description"] == "wherein R1 is alkyl"
    assert out["cxsmiles_opt"] == "<r>R1</r>CC"
    assert out["image_has_text"] is True


def test_push_to_hf_builds_dataset_dict_from_jsonl(tmp_path: Path):
    data = tmp_path / "unified.jsonl"
    data.write_text(
        json.dumps({"id": "a", "split": "train", "image": "a.png"})
        + "\n"
        + json.dumps({"id": "b", "split": "test", "image": "b.png"})
        + "\n"
    )

    ds = dataset_from_jsonl([data])

    assert set(ds.keys()) == {"train", "test"}
    assert len(ds["train"]) == 1
    assert len(ds["test"]) == 1
