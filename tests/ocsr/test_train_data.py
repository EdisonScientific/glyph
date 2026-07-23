"""Boundary tests for the production data classes."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from glyph.ocsr.smiles_tokenizer import CharSmilesTokenizer


def _indigo_renderer_works():
    try:
        from glyph.ocsr.indigo_renderer import IndigoRendererBoundary

        rb = IndigoRendererBoundary(image_size=64)
        rb.render_smiles_to_png("CCO")
        return True
    except Exception:
        return False


_SKIP_INDIGO = (
    "PubChemIndigoDataset requires Indigo native libs (libfreetype/libfontconfig/libXrender)."
)


def _write_tiny_pubchem(tmp_path: Path) -> Path:
    p = tmp_path / "tiny_pubchem.csv"
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pubchem_cid", "InChI", "SMILES", "num_atoms"])
        writer.writeheader()
        writer.writerows(
            [
                {"pubchem_cid": 702, "InChI": "", "SMILES": "CCO", "num_atoms": 3},
                {"pubchem_cid": 241, "InChI": "", "SMILES": "c1ccccc1", "num_atoms": 6},
                {
                    "pubchem_cid": 6322,
                    "InChI": "",
                    "SMILES": "CC(=O)Oc1ccccc1C(=O)O",
                    "num_atoms": 13,
                },
            ]
        )
    return p


@pytest.mark.skipif(not _indigo_renderer_works(), reason=_SKIP_INDIGO)
def test_pubchem_indigo_dataset_renders_and_tokenizes(tmp_path):
    from glyph.ocsr.train_data import PubChemIndigoDataset

    csv_path = _write_tiny_pubchem(tmp_path)
    tok = CharSmilesTokenizer()
    ds = PubChemIndigoDataset(csv_path=csv_path, tokenizer=tok, max_target_len=32)
    assert len(ds) == 3
    item = ds[1]
    assert item["image"].shape == (3, 384, 384)
    assert item["target_ids"].shape[0] == 32
    assert item["smiles"] == "c1ccccc1"


@pytest.mark.skipif(not _indigo_renderer_works(), reason=_SKIP_INDIGO)
def test_collate_batch_stacks_correctly(tmp_path):
    from glyph.ocsr.train_data import PubChemIndigoDataset, collate_batch

    csv_path = _write_tiny_pubchem(tmp_path)
    tok = CharSmilesTokenizer()
    ds = PubChemIndigoDataset(csv_path=csv_path, tokenizer=tok, max_target_len=24)
    batch = collate_batch([ds[i] for i in range(2)])
    assert batch["image"].shape == (2, 3, 384, 384)
    assert batch["target_ids"].shape == (2, 24)
    assert batch["smiles"] == ["CCO", "c1ccccc1"]


@pytest.mark.skipif(
    not os.environ.get("GLYPH_OCSR_USPTO_DATA_DIR"),
    reason="opt-in test: set GLYPH_OCSR_USPTO_DATA_DIR to a local USPTO/MolNexTR data root.",
)
def test_uspto_dataset_loads():
    from glyph.ocsr.train_data import UsptoMolDataset

    data_dir = Path(os.environ["GLYPH_OCSR_USPTO_DATA_DIR"])
    csvs = sorted(data_dir.rglob("*.csv"))
    csv_path = csvs[0]
    image_root = data_dir / "train_uspto" if (data_dir / "train_uspto").exists() else data_dir
    tok = CharSmilesTokenizer()
    ds = UsptoMolDataset(
        csv_path=csv_path,
        image_root=image_root,
        tokenizer=tok,
        max_target_len=64,
        image_size=224,
        max_rows=2,
        path_strip_prefix="uspto_mol/",
    )
    if len(ds) == 0:
        pytest.skip("USPTO CSV had no parseable rows yet")
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["target_ids"].shape[0] == 64


# ---------- Frozen enrichment CSV resolver (network-free tiers) ----------


def test_resolve_ocsr_train_csv_prefers_mirror_env(tmp_path: Path, monkeypatch):
    from glyph.ocsr.train_data import resolve_ocsr_train_csv

    mirror = tmp_path / "mirror"
    target = mirror / "ocsr" / "adjacent_ring.csv"
    target.parent.mkdir(parents=True)
    target.write_text("SMILES\nCCO\n")
    monkeypatch.setenv("GLYPH_OCSR_TRAIN_ROOT", str(mirror))
    assert resolve_ocsr_train_csv("adjacent_ring") == target


def test_resolve_ocsr_train_csv_uses_glyph_data_root_snapshot(tmp_path: Path, monkeypatch):
    from glyph.ocsr.train_data import resolve_ocsr_train_csv

    monkeypatch.delenv("GLYPH_OCSR_TRAIN_ROOT", raising=False)
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    snapshot = tmp_path / "datasets" / "ocsr" / "stereo_200k" / "train.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("SMILES,file_path\nCCO,stereo/train/0.png\n")
    assert resolve_ocsr_train_csv("stereo_200k_train") == snapshot


def test_resolve_ocsr_train_csv_accepts_repo_relative_path(tmp_path: Path, monkeypatch):
    from glyph.ocsr.train_data import resolve_ocsr_train_csv

    monkeypatch.delenv("GLYPH_OCSR_TRAIN_ROOT", raising=False)
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    snapshot = tmp_path / "datasets" / "ocsr" / "adjacent_ring.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("SMILES\nCCO\n")
    assert resolve_ocsr_train_csv("ocsr/adjacent_ring.csv") == snapshot


def test_data_path_arg_hub_sentinel_calls_resolver(tmp_path: Path, monkeypatch):
    from glyph.ocsr import train as ocsr_train

    resolved = tmp_path / "adjacent_ring.csv"
    resolved.write_text("SMILES\nCCO\n")
    monkeypatch.setattr(
        "glyph.ocsr.train_data.resolve_ocsr_train_csv",
        lambda name: resolved,
    )
    assert ocsr_train._data_path_arg("hub:adjacent_ring") == str(resolved)


def test_data_path_arg_plain_path_resolves_under_data_root(tmp_path: Path, monkeypatch):
    from glyph.ocsr import train as ocsr_train

    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    out = ocsr_train._data_path_arg("datasets/ocsr/x.csv")
    assert out == str(tmp_path / "datasets" / "ocsr" / "x.csv")
