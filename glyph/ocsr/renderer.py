"""Tiny RDKit-based image renderer for lightweight tests only.

This is **not** the production renderer. The production training recipe uses
EPAM Indigo's dynamic rendering per MolNexTR/MolScribe. We use RDKit here so
we can stand up a minimal end-to-end dataset check without taking a heavy
extra dependency. It remains useful only as a reference baseline alongside
the Indigo pipeline.
"""

from __future__ import annotations

import io

from rdkit import Chem
from rdkit.Chem import Draw


def render_smiles_to_png_bytes(smiles: str, size: int = 224) -> bytes:
    """Render a SMILES string into a square PNG byte string.

    Returns the PNG bytes. Raises ValueError on unparseable SMILES.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable SMILES for renderer: {smiles!r}")
    drawer = Draw.MolDraw2DCairo(size, size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_smiles_to_pil(smiles: str, size: int = 224):
    """Render a SMILES string into a PIL.Image (RGB, ``size`` x ``size``)."""

    from PIL import Image

    png = render_smiles_to_png_bytes(smiles, size=size)
    return Image.open(io.BytesIO(png)).convert("RGB")
