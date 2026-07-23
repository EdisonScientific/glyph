"""SMILES postprocessing utilities for OCSR.

Deterministic string transformations that fix common model artifacts without
using ground truth.  The full postprocessor removes isolated hallucinated
hydrogen fragments and then RDKit-canonicalizes the result.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]


def strip_hydrogen_fragments(smiles: str | None) -> str | None:
    """Remove disconnected ``[H]``/``[HH]`` fragments from a SMILES string.

    Splits on ``.`` and drops parts equal to ``[H]`` or ``[HH]`` exactly.  This
    never mutates chemically meaningful bracketed hydrogens such as ``[nH]`` or
    ``[NH3+]``.
    """

    if not smiles:
        return smiles
    parts = str(smiles).split(".")
    kept = [part for part in parts if part.strip() not in {"[H]", "[HH]"}]
    if not kept:
        return smiles
    return ".".join(kept).strip(".")


def strip_h2_fragments(smiles: str) -> str:
    """Backward-compatible alias for the original light postprocessor."""

    cleaned = strip_hydrogen_fragments(smiles)
    return smiles if cleaned is None else cleaned


def canonical_smiles_safe(smiles: str | None) -> str | None:
    """Return RDKit canonical SMILES, or ``None`` if parsing fails."""

    if smiles is None:
        return None
    value = str(smiles).strip()
    if not value:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def postprocess_smiles(smiles: str | None) -> str | None:
    """Apply the deterministic OCSR cleanup used for the benchmark eval."""

    if smiles is None:
        return None
    cleaned = strip_hydrogen_fragments(smiles)
    canon = canonical_smiles_safe(cleaned)
    if canon is not None:
        return canon
    return canonical_smiles_safe(smiles)


def light_postprocess(smiles: str) -> str:
    """Conservative SMILES cleanup kept for existing callers."""

    return strip_h2_fragments(smiles)
