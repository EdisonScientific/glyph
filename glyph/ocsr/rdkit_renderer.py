"""RDKit-based dynamic SMILES renderer.

Provides a Dataset that renders PubChem SMILES with RDKit at training time,
adding renderer-style diversity beyond the Indigo path. Optional per-call
randomization of bond width, font size, image size, and orientation.
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from torch.utils.data import Dataset

from .augment import chemistry_safe_augment
from .smiles_tokenizer import CharSmilesTokenizer


def _normalize_pil_to_chw(img: Image.Image, image_size: int) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def render_rdkit(smi: str, image_size: int = 384, randomize: bool = True) -> Image.Image:
    """Render SMILES with RDKit, producing a 384x384 RGB PIL image."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # fallback to all-white
        return Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    AllChem.Compute2DCoords(mol)  # ty: ignore[unresolved-attribute]
    drawer = rdMolDraw2D.MolDraw2DCairo(image_size, image_size)
    opts = drawer.drawOptions()
    if randomize:
        opts.bondLineWidth = random.choice([1, 2, 3])
        opts.fixedBondLength = random.uniform(20, 40)
        # Atom font choice
        opts.minFontSize = random.choice([10, 12, 14])
        opts.maxFontSize = random.choice([14, 18, 22])
        # Random orientation: rotate atoms about origin
        if random.random() < 0.5:
            angle = random.uniform(0, 360)
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                from math import cos, radians, sin

                a = radians(angle)
                x, y = p.x, p.y
                conf.SetAtomPosition(i, (x * cos(a) - y * sin(a), x * sin(a) + y * cos(a), p.z))
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png)).convert("RGB")


class RDKitRenderDataset(Dataset):
    """PubChem-format CSV rendered with RDKit on the fly."""

    def __init__(
        self,
        csv_path: Path | str,
        tokenizer: CharSmilesTokenizer,
        max_target_len: int,
        image_size: int = 384,
        max_rows: int | None = None,
        augment: bool = False,
        aug_p_clean: float = 0.3,
        aug_p_each: float = 0.4,
        randomize: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
        self.image_size = image_size
        self.augment = augment
        self.aug_p_clean = aug_p_clean
        self.aug_p_each = aug_p_each
        self.randomize = randomize
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            self.smiles = [str(row["SMILES"]) for row in csv.DictReader(handle)]
        if max_rows is not None:
            self.smiles = self.smiles[:max_rows]

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:  # ty: ignore[invalid-method-override]
        attempts = 0
        cur = idx
        n = len(self.smiles)
        while True:
            smi = self.smiles[cur]
            try:
                img = render_rdkit(smi, self.image_size, self.randomize)
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
        return {
            "image": tensor,
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "smiles": smi,
            "image_id": f"rdkit_{cur}",
        }
