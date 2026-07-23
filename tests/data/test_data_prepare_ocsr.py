"""Unit tests for the OCSR release data-prep pipeline (``glyph data prepare --ocsr``).

These cover the orchestration without touching the network: layout creation,
provenance-sidecar contents, adjacent-ring mining on a tiny synthetic PubChem sample,
white-crop on one sample image, the large-download refusal, and idempotent
re-runs. The multi-GB source pulls and the 5719 benchmark build are exercised
live elsewhere (see the module docstring in ``glyph/data/ocsr.py``).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from glyph.data import layout
from glyph.data.ocsr import (
    PUBCHEM_CSV_NAME,
    _normalize_uspto_680k_layout,
    prepare_ocsr,
)
from glyph.data.ocsr_adjacent_ring import (
    default_workers as adjacent_ring_default_workers,
)
from glyph.data.ocsr_adjacent_ring import (
    has_adjacent_ring_stereo,
    mine_adjacent_ring,
)
from glyph.data.ocsr_white_crop import (
    background_to_white,
    smart_tight_crop,
    white_crop_image,
    white_crop_stereo_split,
)
from glyph.data.ocsr_white_crop import (
    default_workers as white_crop_default_workers,
)

# --- layout ------------------------------------------------------------------


def test_layout_matches_documented_data_layout(tmp_path: Path) -> None:
    assert layout.ocsr_dataset_dir("pubchem_1m", root=tmp_path) == (
        tmp_path / "datasets" / "ocsr" / "pubchem_1m"
    )
    assert layout.uspto_5719_benchmark_path(root=tmp_path) == (
        tmp_path / "benchmarks" / "ocsr" / "uspto_ocsr_5719_complete.csv"
    )
    assert set(layout.OCSR_DATASET_NAMES) == {
        "pubchem_1m",
        "uspto_680k",
        "stereo_200k_white",
        "adjacent_ring",
    }


def test_provenance_records_do_not_misattribute() -> None:
    stereo = layout.OCSR_PROVENANCE["stereo_200k_white"]
    assert stereo["this_work"] is False
    assert stereo["role"] == "external_preprocessed"
    assert stereo["citation"] == "zhang2026molsight"
    assert stereo["source_repo"] == "Robert-zwr/Stereo-200k"
    assert "did NOT" in stereo["attribution"]
    assert "Do NOT present" in stereo["attribution"]

    adjacent_ring = layout.OCSR_PROVENANCE["adjacent_ring"]
    assert adjacent_ring["this_work"] is True
    assert adjacent_ring["role"] == "derived_this_work"
    assert adjacent_ring["derived_from"] == "pubchem_1m"

    for name in ("pubchem_1m", "uspto_680k"):
        rec = layout.OCSR_PROVENANCE[name]
        assert rec["this_work"] is False
        assert rec["source_repo"] == "CYF200127/MolNexTR"

    assert layout.BENCHMARK_5719_PROVENANCE["expected_rows"] == 5719


def test_write_provenance_roundtrip(tmp_path: Path) -> None:
    path = layout.write_provenance(tmp_path / "d", layout.OCSR_PROVENANCE["pubchem_1m"])
    payload = json.loads(path.read_text())
    assert payload["dataset"] == "pubchem_1m"
    assert "prepared_at" in payload


# --- adjacent-ring mining ----------------------------------------------------


def _write_pubchem_csv(path: Path, smiles: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pubchem_cid", "InChI", "SMILES", "num_atoms"])
        writer.writeheader()
        for index, smi in enumerate(smiles):
            writer.writerow(
                {"pubchem_cid": str(index), "InChI": "", "SMILES": smi, "num_atoms": "0"}
            )


def test_mine_adjacent_ring_subset(tmp_path: Path) -> None:
    keep = "O[C@H]1CCCC[C@@H]1O"  # 2 adjacent ring stereocenters
    reject = ["CCO", "C[C@H](N)C(=O)O", "O[C@H]1CC[C@@H](O)CC1"]
    in_csv = tmp_path / "pubchem.csv"
    _write_pubchem_csv(in_csv, [keep, *reject])
    out_csv = tmp_path / "adjacent_ring.csv"

    result = mine_adjacent_ring(in_csv, out_csv, min_len=0, max_len=1000)

    assert result.scanned == 4
    assert result.kept == 1
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 1
    assert rows[0]["SMILES"] == keep
    # Output is a strict subset of the input schema (whole rows copied).
    assert set(rows[0].keys()) == {"pubchem_cid", "InChI", "SMILES", "num_atoms"}


def test_adjacent_ring_length_filter_and_max_rows(tmp_path: Path) -> None:
    keep = "O[C@H]1CCCC[C@@H]1O"
    in_csv = tmp_path / "pubchem.csv"
    _write_pubchem_csv(in_csv, [keep, keep, keep])
    out_csv = tmp_path / "adjacent_ring.csv"

    # Length filter excludes the (short) qualifying molecule.
    result = mine_adjacent_ring(in_csv, out_csv, min_len=50, max_len=180)
    assert result.kept == 0

    # max_rows caps the scan.
    result2 = mine_adjacent_ring(in_csv, tmp_path / "c2.csv", min_len=0, max_len=1000, max_rows=2)
    assert result2.scanned == 2


def test_has_adjacent_ring_stereo_handles_none() -> None:
    assert has_adjacent_ring_stereo(None) is False


def test_mine_adjacent_ring_parallel_matches_serial(tmp_path: Path) -> None:
    keep = "O[C@H]1CCCC[C@@H]1O"
    smiles = ([keep, "CCO", "C[C@H](N)C(=O)O", "O[C@H]1CC[C@@H](O)CC1"] * 8) + [keep]
    in_csv = tmp_path / "pubchem.csv"
    _write_pubchem_csv(in_csv, smiles)

    serial = mine_adjacent_ring(in_csv, tmp_path / "serial.csv", min_len=0, max_len=1000, workers=1)
    parallel = mine_adjacent_ring(
        in_csv, tmp_path / "parallel.csv", min_len=0, max_len=1000, workers=4
    )

    assert (serial.scanned, serial.kept) == (parallel.scanned, parallel.kept)
    assert serial.kept == 9  # one keeper per group of 8, plus the trailing one
    # Byte-identical output CSV (row order + content) regardless of worker count.
    assert (tmp_path / "serial.csv").read_bytes() == (tmp_path / "parallel.csv").read_bytes()


def test_mine_adjacent_ring_parallel_respects_max_rows(tmp_path: Path) -> None:
    in_csv = tmp_path / "pubchem.csv"
    _write_pubchem_csv(in_csv, ["O[C@H]1CCCC[C@@H]1O"] * 50)
    result = mine_adjacent_ring(
        in_csv, tmp_path / "out.csv", min_len=0, max_len=1000, max_rows=17, workers=4
    )
    assert result.scanned == 17


# --- white crop --------------------------------------------------------------


def _content_on_colored_bg() -> Image.Image:
    arr = np.full((40, 60, 3), (200, 210, 220), dtype=np.uint8)
    arr[15:25, 25:35] = (0, 0, 0)
    return Image.fromarray(arr)


def test_background_to_white_and_crop() -> None:
    img = _content_on_colored_bg()
    whitened = np.asarray(background_to_white(img))
    assert tuple(whitened[0, 0]) == (255, 255, 255)
    cropped = white_crop_image(img)
    assert cropped.size[0] < img.size[0]
    assert cropped.size[1] < img.size[1]


def test_background_to_white_noop_on_white_image() -> None:
    white = Image.new("RGB", (20, 20), (255, 255, 255))
    assert np.asarray(background_to_white(white)).min() == 255


def test_smart_tight_crop_all_white_returns_input() -> None:
    white = Image.new("RGB", (20, 20), (255, 255, 255))
    assert smart_tight_crop(white).size == white.size


def test_white_crop_stereo_split(tmp_path: Path) -> None:
    src_images = tmp_path / "stereo" / "train"
    src_images.mkdir(parents=True)
    _content_on_colored_bg().save(src_images / "0.png")
    _content_on_colored_bg().save(src_images / "1.png")
    src_csv = tmp_path / "stereo" / "train.csv"
    with src_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES", "file_path"])
        writer.writeheader()
        writer.writerow({"SMILES": "N/C=N/CC(=O)O", "file_path": "stereo/train/0.png"})
        writer.writerow({"SMILES": "CCO", "file_path": "stereo/train/1.png"})

    out_dir = tmp_path / "stereo_200k_white"
    result = white_crop_stereo_split(src_csv, src_csv.parent, out_dir, split="train")

    assert result.written == 2
    assert (out_dir / "train" / "0.png").exists()
    rows = list(csv.DictReader((out_dir / "train.csv").open()))
    assert rows[0]["file_path"] == "train/0.png"
    assert rows[0]["SMILES"] == "N/C=N/CC(=O)O"  # target copied unchanged

    # Idempotent: a re-run skips the already-cropped images.
    rerun = white_crop_stereo_split(src_csv, src_csv.parent, out_dir, split="train")
    assert rerun.written == 0
    assert rerun.skipped_existing == 2


def _write_stereo_split(tmp_path: Path, count: int) -> Path:
    """Build a Stereo-like source tree of ``count`` colored-bg images + a CSV."""

    src_images = tmp_path / "stereo" / "train"
    src_images.mkdir(parents=True)
    src_csv = tmp_path / "stereo" / "train.csv"
    with src_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES", "file_path"])
        writer.writeheader()
        for index in range(count):
            arr = np.full((30 + index, 44 + index, 3), (200, 205, 210), dtype=np.uint8)
            arr[10:20, 12:22] = (index % 200, 0, 0)
            Image.fromarray(arr).save(src_images / f"{index}.png")
            writer.writerow({"SMILES": f"C{index}", "file_path": f"stereo/train/{index}.png"})
    return src_csv


def _sha_by_name(image_dir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(image_dir.glob("*.png"))
    }


def test_default_workers_capped() -> None:
    for fn in (white_crop_default_workers, adjacent_ring_default_workers):
        workers = fn()
        assert 1 <= workers <= 8


def test_white_crop_stereo_split_parallel_matches_serial(tmp_path: Path) -> None:
    src_csv = _write_stereo_split(tmp_path, count=12)

    serial = white_crop_stereo_split(
        src_csv, src_csv.parent, tmp_path / "serial", split="train", workers=1
    )
    parallel = white_crop_stereo_split(
        src_csv, src_csv.parent, tmp_path / "parallel", split="train", workers=4
    )

    assert (serial.written, serial.failed, serial.missing_images) == (12, 0, 0)
    assert (parallel.written, parallel.failed) == (serial.written, serial.failed)
    # Byte-identical images and rewritten CSV regardless of worker count.
    assert _sha_by_name(tmp_path / "serial" / "train") == _sha_by_name(
        tmp_path / "parallel" / "train"
    )
    assert (tmp_path / "serial" / "train.csv").read_bytes() == (
        tmp_path / "parallel" / "train.csv"
    ).read_bytes()


def test_white_crop_stereo_split_parallel_survives_corrupt_image(tmp_path: Path) -> None:
    src_csv = _write_stereo_split(tmp_path, count=4)
    # Corrupt one source image; the pool must count it as failed, not abort.
    (tmp_path / "stereo" / "train" / "2.png").write_bytes(b"\x89PNG\r\nnot-an-image")

    result = white_crop_stereo_split(
        src_csv, src_csv.parent, tmp_path / "out", split="train", workers=4
    )

    assert result.written == 3
    assert result.failed == 1
    # The CSV still lists every non-empty row (matching the serial contract).
    rows = list(csv.DictReader((tmp_path / "out" / "train.csv").open()))
    assert [r["file_path"] for r in rows] == [f"train/{i}.png" for i in range(4)]


# --- orchestration -----------------------------------------------------------


def test_prepare_ocsr_dry_run_writes_nothing(tmp_path: Path) -> None:
    report = prepare_ocsr(data_root=tmp_path, dry_run=True)
    assert report.data_root == tmp_path
    statuses = {step.name: step.status for step in report.steps}
    assert all(status == "planned" for status in statuses.values())
    assert not (tmp_path / "datasets").exists()
    assert not (tmp_path / "benchmarks").exists()


def test_prepare_ocsr_refuses_large_and_skips_existing(tmp_path: Path) -> None:
    # Pre-materialize everything so no network is needed.
    pubchem = layout.ocsr_dataset_dir("pubchem_1m", root=tmp_path) / PUBCHEM_CSV_NAME
    _write_pubchem_csv(pubchem, ["O[C@H]1CCCC[C@@H]1O"])
    adjacent_ring_dir = layout.ocsr_dataset_dir("adjacent_ring", root=tmp_path)
    adjacent_ring_dir.mkdir(parents=True)
    (adjacent_ring_dir / "adjacent_ring.csv").write_text("SMILES\nCCO\n")
    bench = layout.uspto_5719_benchmark_path(root=tmp_path)
    bench.parent.mkdir(parents=True)
    bench.write_text("image_id,smiles,source_of_label\n")

    report = prepare_ocsr(data_root=tmp_path, allow_large=False)
    statuses = {step.name: step.status for step in report.steps}
    assert statuses["pubchem_1m"] == "skipped_exists"
    assert statuses["uspto_680k"] == "refused"
    assert statuses["stereo_200k_white"] == "refused"
    assert statuses["adjacent_ring"] == "skipped_exists"
    assert statuses["uspto_ocsr_5719"] == "skipped_exists"
    assert report.ok


def test_prepare_ocsr_mines_adjacent_ring_from_local_pubchem(tmp_path: Path) -> None:
    # pubchem present + benchmark present (so neither needs the network); adjacent_ring derives.
    pubchem = layout.ocsr_dataset_dir("pubchem_1m", root=tmp_path) / PUBCHEM_CSV_NAME
    _write_pubchem_csv(pubchem, ["O[C@H]1CCCC[C@@H]1O", "CCO"])
    bench = layout.uspto_5719_benchmark_path(root=tmp_path)
    bench.parent.mkdir(parents=True)
    bench.write_text("image_id,smiles,source_of_label\n")

    report = prepare_ocsr(
        data_root=tmp_path, allow_large=False, adjacent_ring_min_len=0, adjacent_ring_max_len=99
    )
    adjacent_ring_step = next(s for s in report.steps if s.name == "adjacent_ring")
    assert adjacent_ring_step.status == "materialized"
    assert adjacent_ring_step.counts["kept"] == 1

    provenance = json.loads(
        (layout.ocsr_dataset_dir("adjacent_ring", root=tmp_path) / "provenance.json").read_text()
    )
    assert provenance["this_work"] is True
    assert provenance["derived_from"] == "pubchem_1m"


# --- uspto_680k extract-layout normalization ---------------------------------


def _make_nested_uspto_extract(dataset_dir: Path) -> None:
    """Mimic ``train_uspto.zip`` extraction: train_uspto/train_uspto.csv + year tree."""

    nested = dataset_dir / "train_uspto"
    (nested / "2006" / "I20060606").mkdir(parents=True)
    img = nested / "2006" / "I20060606" / "US01-C001.TIF"
    Image.new("RGB", (32, 24), (255, 255, 255)).save(img)
    with (nested / "train_uspto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "SMILES"])
        writer.writeheader()
        writer.writerow({"file_path": "uspto_mol/2006/I20060606/US01-C001.TIF", "SMILES": "CCO"})


def test_normalize_uspto_680k_layout(tmp_path: Path) -> None:
    dataset_dir = layout.ocsr_dataset_dir("uspto_680k", root=tmp_path)
    _make_nested_uspto_extract(dataset_dir)

    moved = _normalize_uspto_680k_layout(dataset_dir)

    # CSV promoted to train.csv, images folded under images/, nested dir gone.
    assert moved == 2
    assert (dataset_dir / "train.csv").is_file()
    assert (dataset_dir / "images" / "2006" / "I20060606" / "US01-C001.TIF").is_file()
    assert not (dataset_dir / "train_uspto").exists()
    # The documented layout + default strip_prefix resolves the CSV's file_path:
    # uspto_mol/<year>/... --(strip uspto_mol/)--> images/<year>/...
    with (dataset_dir / "train.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    rel = row["file_path"].removeprefix("uspto_mol/")
    assert (dataset_dir / "images" / rel).is_file()
    # Idempotent: a second call is a no-op.
    assert _normalize_uspto_680k_layout(dataset_dir) == 0


def test_prepare_ocsr_heals_unnormalized_uspto_extract(tmp_path: Path) -> None:
    # An older un-normalized extract present (nested train_uspto/, no train.csv):
    # prepare heals it in place without any network / re-download.
    dataset_dir = layout.ocsr_dataset_dir("uspto_680k", root=tmp_path)
    _make_nested_uspto_extract(dataset_dir)
    # Satisfy the other steps offline so only uspto is exercised.
    pubchem = layout.ocsr_dataset_dir("pubchem_1m", root=tmp_path) / PUBCHEM_CSV_NAME
    _write_pubchem_csv(pubchem, ["CCO"])
    adjacent_ring_dir = layout.ocsr_dataset_dir("adjacent_ring", root=tmp_path)
    adjacent_ring_dir.mkdir(parents=True)
    (adjacent_ring_dir / "adjacent_ring.csv").write_text("SMILES\nCCO\n")
    bench = layout.uspto_5719_benchmark_path(root=tmp_path)
    bench.parent.mkdir(parents=True)
    bench.write_text("image_id,smiles,source_of_label\n")

    report = prepare_ocsr(data_root=tmp_path, allow_large=False)

    statuses = {step.name: step.status for step in report.steps}
    assert statuses["uspto_680k"] == "materialized"
    assert (dataset_dir / "train.csv").is_file()
    assert (dataset_dir / "images" / "2006" / "I20060606" / "US01-C001.TIF").is_file()


def test_normalize_uspto_680k_layout_merges_partial_images(tmp_path: Path) -> None:
    """A re-run over a PARTIAL images/ + stranded train_uspto/ merges, not skips.

    Reproduces the real interrupted-then-re-run symptom: a prior run already
    materialized part of ``images/2002`` (and left a partial ``train.csv``), while
    the remaining TIFs sit stranded in ``train_uspto/2002-2003``. The heal must fold
    the *union* under ``images/`` (nothing stranded), promote the CSV, and drop the
    nested tree -- rather than no-op on the pre-existing ``train.csv`` or skip the
    already-existing year-dir.
    """

    dataset_dir = layout.ocsr_dataset_dir("uspto_680k", root=tmp_path)

    # Prior partial run: one file already under images/2002 + an existing train.csv.
    already = dataset_dir / "images" / "2002" / "I20020101" / "existing.TIF"
    already.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(already)
    full_csv = "file_path,SMILES\nuspto_mol/2002/I20020202/stranded.TIF,CCO\n"
    existing_train = dataset_dir / "train.csv"
    existing_train.write_text(full_csv, encoding="utf-8")

    # Stranded extract: an identical CSV duplicate + a DIFFERENT 2002 file + a new 2003 file.
    nested = dataset_dir / "train_uspto"
    stranded_2002 = nested / "2002" / "I20020202" / "stranded.TIF"
    stranded_2002.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(stranded_2002)
    stranded_2003 = nested / "2003" / "I20030303" / "new.TIF"
    stranded_2003.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(stranded_2003)
    (nested / "train_uspto.csv").write_text(full_csv, encoding="utf-8")

    moved = _normalize_uspto_680k_layout(dataset_dir)

    assert moved == 2  # two stranded images; the identical CSV duplicate is discarded
    # The union of prior + stranded files all live under images/ (nothing stranded).
    assert already.is_file()  # pre-existing year-dir file merged, not clobbered
    assert (dataset_dir / "images" / "2002" / "I20020202" / "stranded.TIF").is_file()
    assert (dataset_dir / "images" / "2003" / "I20030303" / "new.TIF").is_file()
    # The existing identical CSV is retained and its nested duplicate is removed.
    assert existing_train.read_text(encoding="utf-8") == full_csv
    assert not nested.exists()
    # No files left behind anywhere except under images/ + the top-level train.csv.
    leftover = sorted(
        rel.as_posix()
        for p in dataset_dir.rglob("*")
        if p.is_file()
        for rel in [p.relative_to(dataset_dir)]
        if rel.name != "train.csv" and rel.parts[0] != "images"
    )
    assert leftover == []
    # Idempotent: a second call is a no-op.
    assert _normalize_uspto_680k_layout(dataset_dir) == 0


def test_normalize_uspto_680k_layout_preserves_differing_collision(tmp_path: Path) -> None:
    """A same-path collision with different bytes preserves both files."""

    dataset_dir = layout.ocsr_dataset_dir("uspto_680k", root=tmp_path)
    destination = dataset_dir / "images" / "2006" / "same.TIF"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"image-version-a")

    nested = dataset_dir / "train_uspto"
    conflicting = nested / "2006" / "same.TIF"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"image-version-b")

    conflicts: list[Path] = []
    assert _normalize_uspto_680k_layout(dataset_dir, conflicts=conflicts) == 0

    assert destination.read_bytes() == b"image-version-a"
    assert conflicting.read_bytes() == b"image-version-b"
    assert conflicts == [conflicting]
    assert nested.is_dir()
    # An unresolved conflict stays unchanged on retry rather than being overwritten.
    assert _normalize_uspto_680k_layout(dataset_dir) == 0
    assert destination.read_bytes() == b"image-version-a"
    assert conflicting.read_bytes() == b"image-version-b"
