"""Thin wrapper around the vendored EPAM Indigo renderer.

Indigo is the public renderer used by both MolNexTR and MolScribe in their
``dynamic_indigo`` training data pipeline. We vendor it under
``glyph.ocsr._indigo`` so we do not depend on the upstream repos at runtime.
The license is Apache-2.0 (per file headers); compatible with our package.

Production training will use the same option set MolNexTR/MolScribe randomize
between batches (bond thickness, font family, label mode, implicit H, etc.).
This module exposes only the minimum needed at the rendering boundary; the
full randomized recipe is wired in the production training loop.
"""

from __future__ import annotations

from typing import Any

# The vendored Indigo shared libraries are Linux x86-64 only. This module (and
# thus the native import below) is loaded lazily from the training-data render
# path (``glyph.ocsr.train_data``); the OCSR inference/eval path never imports
# it, so inference and evaluation stay cross-platform.
from . import _indigo
from ._indigo.renderer import IndigoRenderer

_DEFAULT_RENDER_OPTIONS: dict[str, Any] = {
    "render-output-format": "png",
    "render-relative-thickness": 1.0,
    "render-bond-line-width": 1.5,
    "render-implicit-hydrogens-visible": True,
    "render-coloring": False,
    "render-stereo-style": "old",
    # Background and label defaults matching MolScribe/MolNexTR clean baseline.
    "render-background-color": "1, 1, 1",
    "render-base-color": "0, 0, 0",
}


class IndigoRendererBoundary:
    """Stable boundary API around Indigo's renderer.

    The class deliberately does *not* expose the full Indigo handle; callers
    that need the raw Indigo object should reach into ``self.indigo``. The
    boundary is what we promise to keep stable across training pipeline
    rewrites.
    """

    def __init__(self, image_size: int = 384, options: dict[str, Any] | None = None):
        self.indigo = _indigo.Indigo()
        self.renderer = IndigoRenderer(self.indigo)
        self.image_size = image_size
        merged = dict(_DEFAULT_RENDER_OPTIONS)
        if options:
            merged.update(options)
        for k, v in merged.items():
            self.indigo.setOption(k, v)
        # Always set image size as a 2-tuple option per Indigo's API.
        self.indigo.setOption("render-image-size", image_size, image_size)

    def version(self) -> str:
        return self.indigo.version()

    def render_smiles_to_png(self, smiles: str) -> bytes:
        if not smiles:
            raise ValueError("Cannot render empty SMILES")
        mol = self.indigo.loadMolecule(smiles)
        # Force a layout in case the SMILES has no coordinates.
        mol.layout()
        buf = self.renderer.renderToBuffer(mol)
        return bytes(buf)

    def render_smiles_to_pil(self, smiles: str):
        import io

        from PIL import Image

        png = self.render_smiles_to_png(smiles)
        return Image.open(io.BytesIO(png)).convert("RGB")
