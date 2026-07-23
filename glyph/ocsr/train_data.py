"""Training datasets aligned to the MolNexTR data-format contract.

This module wraps the PubChem (dynamic Indigo) and USPTO (pre-rendered PNGs)
splits into a single ``IterableDataset`` style consumer that returns the
tensors the trainer expects:

    (image: float32 [3, H, W] in [-1, 1], target_ids: int64 [max_len])

Two real datasets are provided:

* ``PubChemIndigoDataset`` — reads the 1M-row PubChem CSV, renders each row
  with our vendored Indigo on the fly. Uses ``epam.indigo.layout()`` so the
  SMILES does not need pre-stored coordinates.
* ``UsptoMolDataset`` — reads the USPTO 680k aux CSV + on-disk image dir,
  decodes the PNG, and tokenizes the canonical SMILES.

Both datasets honor a ``CharSmilesTokenizer`` for the SMILES-only baseline.
The full ``chartok_coords,edges`` multitask is intentionally not implemented
here; it will be added in a follow-up commit.
"""

from __future__ import annotations

import csv
import io
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import chemistry_safe_augment
from .indigo_renderer import IndigoRendererBoundary
from .smiles_tokenizer import CharSmilesTokenizer

# ---------- Frozen enrichment CSV auto-download --------------------------
#
# The OCSR enrichment CSVs are shipped FROZEN on the dataset repo
# ``EdisonScientific/glyph-datasets`` under ``ocsr/``. ``adjacent_ring`` (the
# original "This work" mined subset) carries a bare ``SMILES`` column and is
# rendered on the fly with the vendored Indigo, so it is fully self-contained.
# ``stereo_200k`` ships its ``SMILES``/``file_path`` CSV here too. This resolver
# mirrors the OCSR eval benchmark resolver
# (:func:`glyph.ocsr.eval._resolve_default_parquet`): a local mirror env, then a
# ``GLYPH_DATA_ROOT`` snapshot, then ``hf_hub_download``. It removes the manual
# ``huggingface-cli`` step so the trainer can pull the enrichment CSVs itself.

#: Dataset repo that hosts the frozen OCSR training CSVs.
GLYPH_DATASETS_REPO = "EdisonScientific/glyph-datasets"

#: Pin for the hosted CSVs. ``main`` resolves the published files; repin to the
#: exact upload commit SHA for byte-exact, reproducible fetches (same follow-up
#: as :data:`glyph.ocsr.eval.OCSR_BENCHMARK_REVISION`).
OCSR_TRAIN_REVISION = "86f8e17d75699d8ed0799eb6d100ceba3faac56e"

#: Env var pointing at a local mirror of the dataset-repo root (contains
#: ``ocsr/…``). Checked before the ``GLYPH_DATA_ROOT`` snapshot and the download.
OCSR_TRAIN_ROOT_ENV = "GLYPH_OCSR_TRAIN_ROOT"

#: Short keys → repo-relative paths for the frozen enrichment CSVs.
HOSTED_OCSR_TRAIN_CSVS: dict[str, str] = {
    "adjacent_ring": "ocsr/adjacent_ring.csv",
    "stereo_200k_train": "ocsr/stereo_200k/train.csv",
    "stereo_200k_val": "ocsr/stereo_200k/val.csv",
    "stereo_200k_white_train": "ocsr/stereo_200k_white/train.csv",
    "stereo_200k_white_val": "ocsr/stereo_200k_white/val.csv",
}


def resolve_ocsr_train_csv(name: str) -> Path:
    """Resolve a frozen OCSR enrichment CSV hosted on ``glyph-datasets``.

    ``name`` is either a short key from :data:`HOSTED_OCSR_TRAIN_CSVS`
    (e.g. ``"adjacent_ring"``) or a repo-relative path (``"ocsr/adjacent_ring.csv"``).

    Resolution order (mirrors the OCSR eval benchmark resolver):

    1. A local mirror under ``$GLYPH_OCSR_TRAIN_ROOT``.
    2. A ``$GLYPH_DATA_ROOT`` snapshot under ``datasets/``.
    3. ``hf_hub_download`` from :data:`GLYPH_DATASETS_REPO` at
       :data:`OCSR_TRAIN_REVISION`.
    """

    from glyph.paths import datasets_dir

    repo_path = HOSTED_OCSR_TRAIN_CSVS.get(name, name)

    mirror = os.environ.get(OCSR_TRAIN_ROOT_ENV)
    if mirror:
        candidate = Path(mirror).expanduser() / repo_path
        if candidate.is_file():
            return candidate

    snapshot = datasets_dir() / repo_path
    if snapshot.is_file():
        return snapshot

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=GLYPH_DATASETS_REPO,
            filename=repo_path,
            repo_type="dataset",
            revision=OCSR_TRAIN_REVISION,
        )
    )


def _normalize_pil_to_chw(img: Image.Image, image_size: int) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@dataclass
class IndigoRenderOptions:
    image_size: int = 384
    randomize: bool = True
    augment: bool = False
    p_clean: float = 0.3
    p_each: float = 0.4


class PubChemIndigoDataset(Dataset):
    """1M-row PubChem CSV; SMILES rendered on the fly with vendored Indigo."""

    def __init__(
        self,
        csv_path: Path | str,
        tokenizer: CharSmilesTokenizer,
        max_target_len: int,
        render_opts: IndigoRenderOptions | None = None,
        max_rows: int | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
        self.render_opts = render_opts or IndigoRenderOptions()
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            self.smiles = [str(row["SMILES"]) for row in csv.DictReader(handle)]
        if max_rows is not None:
            self.smiles = self.smiles[:max_rows]
        # Lazily-initialized per-process renderer (Indigo handles are not
        # safely shared across DataLoader workers).
        self._renderer: IndigoRendererBoundary | None = None

    def _get_renderer(self) -> IndigoRendererBoundary:
        if self._renderer is None:
            self._renderer = IndigoRendererBoundary(image_size=self.render_opts.image_size)
        return self._renderer

    def __len__(self) -> int:
        return len(self.smiles)

    def _maybe_randomize_options(self) -> None:
        if not self.render_opts.randomize:
            return
        ix = self._get_renderer().indigo
        rt = round(random.uniform(0.7, 1.5), 2)
        bw = round(random.uniform(1.0, 2.0), 2)
        ix.setOption("render-relative-thickness", rt)
        ix.setOption("render-bond-line-width", bw)
        ix.setOption(
            "render-implicit-hydrogens-visible",
            random.random() < 0.7,
        )
        # Drop "none" from stereo-style choices: hiding stereo cues at training
        # time hurts recognition of stereochemistry, so always render explicit
        # stereo cues during training.
        ix.setOption(
            "render-stereo-style",
            random.choice(["old", "ext"]),
        )

    def __getitem__(self, idx: int) -> dict:  # ty: ignore[invalid-method-override]
        # Indigo can fail on degenerate molecules (e.g. "stereocenters: angle
        # between bonds is too small"). When this happens we fall back to the
        # next row deterministically. We cap retries to avoid pathological
        # loops; if a small contiguous chunk is unrenderable we surface the
        # error to the caller.
        n = len(self.smiles)
        attempts = 0
        cur = idx
        while True:
            smi = self.smiles[cur]
            try:
                renderer = self._get_renderer()
                self._maybe_randomize_options()
                png = renderer.render_smiles_to_png(smi)
                img = Image.open(io.BytesIO(png))
                if self.render_opts.augment:
                    img = chemistry_safe_augment(
                        img.convert("RGB"),
                        p_clean=self.render_opts.p_clean,
                        p_each=self.render_opts.p_each,
                    )
                tensor = _normalize_pil_to_chw(img, self.render_opts.image_size)
                break
            except Exception:
                attempts += 1
                if attempts >= 16:
                    raise
                cur = (cur + 1) % n
        target_ids = self.tokenizer.encode_padded(smi, max_len=self.max_target_len)
        return {
            "image": tensor,
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "smiles": smi,
            "image_id": f"pubchem-{cur}",
        }


class UsptoMolDataset(Dataset):
    """USPTO 680k aux CSV + on-disk image files (TIF or PNG).

    The MolNexTR distribution stores `file_path` strings prefixed with
    ``uspto_mol/...`` but extracts the actual files under a different root
    (we observed extraction into ``train_uspto/...``). To keep the CSV
    untouched, callers can pass ``path_strip_prefix`` to remove a leading
    segment from every CSV path before joining with ``image_root``.

    The dataset also exposes ``compute_sample_weights`` which returns a
    per-row weight array based on training-data features only. Stereo-bearing
    SMILES and long SMILES are upweighted to support targeted fine-tuning.
    Crucially this uses the training CSV's SMILES column, never the eval
    benchmark labels.
    """

    def __init__(
        self,
        csv_path: Path | str,
        image_root: Path | str,
        tokenizer: CharSmilesTokenizer,
        max_target_len: int,
        image_size: int = 384,
        max_rows: int | None = None,
        path_strip_prefix: str | None = None,
        augment: bool = False,
        aug_p_clean: float = 0.3,
        aug_p_each: float = 0.4,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
        self.image_size = image_size
        self.path_strip_prefix = path_strip_prefix
        self.augment = augment
        self.aug_p_clean = aug_p_clean
        self.aug_p_each = aug_p_each
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        filtered: list[dict[str, str]] = []
        for row in rows:
            raw_smiles = row.get("raw_SMILES")
            smiles = raw_smiles if raw_smiles not in {None, ""} else row.get("SMILES")
            if smiles not in {None, ""}:
                filtered.append({**row, "SMILES": str(smiles)})
        self.rows = filtered[:max_rows] if max_rows is not None else filtered

    def __len__(self) -> int:
        return len(self.rows)

    def compute_sample_weights(
        self,
        stereo_weight: float = 4.0,
        long_threshold: int = 80,
        long_weight: float = 4.0,
    ) -> list[float]:
        """Return per-row sampling weights based on training features only.

        - Rows whose SMILES contain any of ``@``, ``/``, ``\\`` get
          ``stereo_weight``-fold weight.
        - Rows whose SMILES length is at least ``long_threshold`` get
          ``long_weight``-fold weight.
        Weights compose multiplicatively so a long stereo row gets the product.
        """

        weights: list[float] = []
        for row in self.rows:
            smiles = str(row["SMILES"])
            w = 1.0
            if any(ch in smiles for ch in ("@", "/", "\\")):
                w *= stereo_weight
            if len(smiles) >= long_threshold:
                w *= long_weight
            weights.append(w)
        return weights

    def _resolve(self, path: str) -> str:
        if self.path_strip_prefix and path.startswith(self.path_strip_prefix):
            path = path[len(self.path_strip_prefix) :].lstrip("/")
        if os.path.isabs(path):
            return path
        return str(self.image_root / path)

    def __getitem__(self, idx: int) -> dict:  # ty: ignore[invalid-method-override]
        n = len(self.rows)
        attempts = 0
        cur = idx
        while True:
            row = self.rows[cur]
            smi = str(row["SMILES"])
            path = str(row["file_path"])
            full = self._resolve(path)
            try:
                img = Image.open(full).convert("RGB")
                if self.augment:
                    img = chemistry_safe_augment(
                        img, p_clean=self.aug_p_clean, p_each=self.aug_p_each
                    )
                tensor = _normalize_pil_to_chw(img, self.image_size)
                break
            except Exception:
                attempts += 1
                if attempts >= 16:
                    raise
                cur = (cur + 1) % n
        target_ids = self.tokenizer.encode_padded(smi, max_len=self.max_target_len)
        image_id = str(row.get("image_id") or Path(path).stem)
        return {
            "image": tensor,
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "smiles": smi,
            "image_id": image_id,
        }


def collate_batch(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    target_ids = torch.stack([b["target_ids"] for b in batch], dim=0)
    return {
        "image": images,
        "target_ids": target_ids,
        "smiles": [b["smiles"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }
