"""OCSR target normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


RDKIT_CANONICAL_ISOMERIC_SMILES_SOURCE_SUFFIX = "->rdkit_canonical_isomeric_smiles"
OCSR_TARGET_FORMAT = "SMILES"
OCSR_CANONICAL_STATUSES = {"canonicalized", "already_canonical"}


@dataclass(frozen=True)
class OCSRSmilesResult:
    """Guarded RDKit canonicalization result for ordinary molecule SMILES."""

    value: str | None
    changed: bool
    status: str
    error: str = ""


def canonicalize_ocsr_smiles_checked(value: str | None) -> OCSRSmilesResult:
    """Return RDKit-canonical isomeric SMILES for an ordinary OCSR target.

    This helper is intentionally stricter than Markush/CXSMILES evaluators:
    ordinary OCSR supervision should be bare SMILES, not Markush XML, R-tagged
    CXSMILES, or a CXSMILES extension payload.
    """

    if not isinstance(value, str):
        return OCSRSmilesResult(None, False, "invalid_smiles", "SMILES value is not a string")
    smiles = value.strip()
    if not smiles:
        return OCSRSmilesResult(None, False, "invalid_smiles", "SMILES value is empty")
    lowered = smiles.lower()
    if "<markush" in lowered or "<cxsmi" in lowered or "<r>" in lowered:
        return OCSRSmilesResult(
            None,
            False,
            "invalid_smiles",
            "OCSR target must be bare SMILES, not Markush XML/CXSMILES markup",
        )
    if "|" in smiles:
        return OCSRSmilesResult(
            None,
            False,
            "invalid_smiles",
            "OCSR target must be bare SMILES without CXSMILES extensions",
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return OCSRSmilesResult(None, False, "invalid_smiles", "RDKit could not parse SMILES")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    changed = canonical != smiles
    return OCSRSmilesResult(
        canonical,
        changed,
        "canonicalized" if changed else "already_canonical",
        "",
    )


def row_declares_ocsr_smiles(row: dict) -> bool:
    """Return whether a row belongs to the ordinary OCSR SMILES contract."""

    return (
        str(row.get("task") or "").lower() == "ocsr"
        or row.get("mode") == "molecule_ocsr"
        or row.get("target_format") == OCSR_TARGET_FORMAT
    )
