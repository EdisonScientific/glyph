#!/usr/bin/env python3
"""
Canonicalize CXSMILES training targets for deterministic Sg/M atom indices.

This script processes JSONL training files and canonicalizes the CXSMILES in each
record's assistant message. Canonicalization ensures:
- Core SMILES is in RDKit canonical form
- Fragment ordering is deterministic
- Sg atom indices point to the correct atoms after reordering
- M attachment point indices are correctly mapped
- R-group labels are preserved at the correct positions

Usage:
    python -m glyph.markush.data.canonicalize \\
        --input data/train.jsonl \\
        --output data/train_canonical.jsonl \\
        --report
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

from rdkit import Chem, RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

CX_SECTION_STARTS = (
    "atomProp:",
    "SgD:",
    "SgH:",
    "SgN:",
    "ctu:",
    "wU:",
    "wD:",
    "Sg:",
    "LN:",
    "m:",
    "c:",
    "t:",
    "f:",
    "w:",
    "S:",
    "r:",
    "@@:",
    "@:",
    "$",
    "(",
)


class ParsedCXSMILES(NamedTuple):
    """Parsed components of a CXSMILES string with R-group tags."""

    core: str  # SMILES without extension
    extension: str  # Everything after | (or empty)
    rgroups: list[tuple[int, int, str]]  # (start, end, label) for each <r>LABEL</r>


class CanonicalizeStats(NamedTuple):
    """Statistics from canonicalization."""

    total: int
    changed: int
    unchanged: int
    failed: int
    no_cxsmiles: int


class CanonicalizeResult(NamedTuple):
    """Result of a guarded CXSMILES canonicalization attempt."""

    value: str | None
    changed: bool
    accepted: bool
    fallback: bool
    status: str
    error: str | None = None


def parse_cxsmiles(cxsmiles_with_rgroups: str) -> ParsedCXSMILES:
    """
    Parse CXSMILES with R-group tags, separating core from extensions.

    Args:
        cxsmiles_with_rgroups: CXSMILES string possibly containing <r>LABEL</r> tags

    Returns:
        ParsedCXSMILES with core, extension, and rgroups list
    """
    if "|" in cxsmiles_with_rgroups:
        core, ext = cxsmiles_with_rgroups.split("|", 1)
    else:
        core, ext = cxsmiles_with_rgroups, ""

    rgroup_pattern = r"<r>([^<]+)</r>"
    rgroups = [(m.start(), m.end(), m.group(1)) for m in re.finditer(rgroup_pattern, core)]

    return ParsedCXSMILES(core, ext, rgroups)


def strip_rgroups(core: str) -> str:
    """Replace <r>LABEL</r> with [*] in SMILES string."""
    return re.sub(r"<r>[^<]+</r>", "[*]", core)


def get_atom_mapping(
    stripped_smiles: str,
) -> tuple[dict[int, int] | None, list[int] | None, str | None]:
    """
    Get mapping from original atom indices to canonical atom indices.

    Uses atom map numbers to track how atoms are reordered during canonicalization.

    Args:
        stripped_smiles: SMILES with [*] for R-groups

    Returns:
        Tuple of:
        - old_to_new: dict mapping original atom idx to canonical atom idx
        - wildcard_indices: list of original atom indices for [*] atoms (in order)
        - canonical_with_maps: canonical SMILES string with atom map numbers
        Returns (None, None, None) if parsing fails.
    """
    mol = Chem.MolFromSmiles(stripped_smiles)
    if mol is None:
        return None, None, None

    # Track wildcards and assign atom map numbers
    wildcard_indices = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
        if atom.GetSymbol() == "*":
            wildcard_indices.append(i)

    # Get canonical SMILES with atom maps
    canonical_with_maps = Chem.MolToSmiles(mol)

    # Parse canonical to build mapping
    mol2 = Chem.MolFromSmiles(canonical_with_maps)
    if mol2 is None:
        return None, None, None
    old_to_new = {}
    for new_idx, atom in enumerate(mol2.GetAtoms()):
        old_idx = atom.GetAtomMapNum() - 1
        old_to_new[old_idx] = new_idx

    return old_to_new, wildcard_indices, canonical_with_maps


def clean_smiles_brackets(smiles_with_brackets: str) -> str:
    """
    Clean up bracketed atoms in SMILES, stripping atom maps while preserving
    necessary H counts.

    - [*:N] -> [*] (strip atom maps from wildcards)
    - [CH3:N] -> C (common aliphatic atoms with default H)
    - [cH:N] -> c (aromatic atoms with default H)
    - [nH:N] -> [nH] (preserve explicit H on aromatic heteroatoms)
    - [NH2+:N] -> [NH2+] (preserve charges and H counts)

    Args:
        smiles_with_brackets: SMILES with bracketed atoms

    Returns:
        Cleaned SMILES with simplified notation
    """

    def simplify_atom(match: re.Match) -> str:
        full = match.group(0)
        inner = full[1:-1]  # Remove [ and ]

        # Handle wildcards - strip atom map only
        if inner.startswith("*"):
            if ":" in inner:
                return "[*]"
            return full

        # Remove atom map (:NUM)
        if ":" in inner:
            inner = inner.rsplit(":", 1)[0]

        # Parse charge (at the end)
        charge = ""
        charge_match = re.search(r"([+-]\d*)$", inner)
        if charge_match:
            charge = charge_match.group()
            inner = inner[: charge_match.start()]

        # Parse H count
        h_part = ""
        h_match = re.search(r"(H\d*)$", inner, re.IGNORECASE)
        if h_match:
            h_part = h_match.group()
            inner = inner[: h_match.start()]

        # Organic subset atoms
        organic_subset = {
            "C",
            "c",
            "N",
            "n",
            "O",
            "o",
            "S",
            "s",
            "P",
            "p",
            "F",
            "Cl",
            "Br",
            "I",
            "B",
            "b",
        }

        if inner not in organic_subset:
            # Non-organic atom - reconstruct with any H/charge info but WITHOUT atom map
            return f"[{inner}{h_part}{charge}]"

        # If there's a charge, must keep brackets
        if charge:
            return f"[{inner}{h_part}{charge}]"

        # Aromatic heteroatoms with explicit H must keep the H
        # This is critical for [nH], [oH], [sH] in heteroaromatic rings
        if inner in ("n", "o", "s") and h_part.upper() == "H":
            return f"[{inner}{h_part}]"

        # For common cases without charge, simplify
        # [CH3] -> C, [cH] -> c, [NH] -> N (aliphatic), etc.
        return inner

    return re.sub(r"\[[^\]]+\]", simplify_atom, smiles_with_brackets)


def reindex_extensions(ext: str, old_to_new: dict[int, int]) -> str:
    """
    Reindex atom indices in CXSMILES extensions (Sg, m sections).

    Handles:
    - Sg:TYPE:INDICES:LABEL:MODIFIER (e.g., Sg:n:11,12,13,14:Q:ht)
    - m:CONNECTOR:RING_ATOMS (e.g., m:17:1.2.3.13.14.15)

    Args:
        ext: Extension string (everything after |)
        old_to_new: Mapping from old to new atom indices

    Returns:
        Reindexed extension string
    """
    if not ext:
        return ""

    # Parse extension into sections
    # Sections are separated by commas, but Sg sections contain comma-separated indices
    # Use heuristic: a comma is a section separator if followed by Sg:, m:, or $
    parts = []
    current: list[str] = []
    i = 0

    while i < len(ext):
        c = ext[i]
        if c == ",":
            # Check if this comma is a section separator
            rest = ext[i + 1 :] if i + 1 < len(ext) else ""
            if rest.startswith(CX_SECTION_STARTS):
                # Section separator
                parts.append("".join(current))
                current = []
            else:
                # Part of current section (like indices comma)
                current.append(c)
        else:
            current.append(c)
        i += 1

    if current:
        parts.append("".join(current))

    new_parts = []

    for part in parts:
        if part.startswith("Sg:"):
            # Format: Sg:TYPE:INDICES:LABEL:MODIFIER or variations
            # e.g., Sg:n:11,12,13,14:Q:ht
            segments = part.split(":")
            if len(segments) >= 3:
                indices_str = segments[2]
                if indices_str:
                    old_indices = [int(x) for x in indices_str.split(",") if x]
                    new_indices = [old_to_new.get(i, i) for i in old_indices]
                    segments[2] = ",".join(str(i) for i in new_indices)
            for idx in range(5, len(segments)):
                if re.fullmatch(r"\d+(?:[,.]\d+)*", segments[idx]):
                    separator = "," if "," in segments[idx] else "." if "." in segments[idx] else ""
                    values = [int(x) for x in re.split(r"[,.]", segments[idx]) if x]
                    mapped = [str(old_to_new.get(value, value)) for value in values]
                    segments[idx] = separator.join(mapped) if separator else mapped[0]
            new_parts.append(":".join(segments))

        elif part.startswith("m:"):
            # Format: m:CONNECTOR:RING_ATOMS (dot-separated)
            # e.g., m:17:1.2.3.13.14.15
            segments = part.split(":")
            if len(segments) >= 2:
                connector = int(segments[1])
                segments[1] = str(old_to_new.get(connector, connector))
            if len(segments) >= 3:
                ring_atoms_str = segments[2]
                if ring_atoms_str:
                    ring_atoms = [int(x) for x in ring_atoms_str.split(".") if x]
                    new_ring = sorted([old_to_new.get(i, i) for i in ring_atoms])
                    segments[2] = ".".join(str(i) for i in new_ring)
            new_parts.append(":".join(segments))

        elif part.startswith("atomProp:"):
            entries = []
            for entry in part.removeprefix("atomProp:").split(":"):
                try:
                    atom_idx_text, prop, value = entry.split(".", 2)
                    atom_idx = int(atom_idx_text)
                except ValueError:
                    entries.append(entry)
                    continue
                entries.append(f"{old_to_new.get(atom_idx, atom_idx)}.{prop}.{value}")
            new_parts.append(f"atomProp:{':'.join(entries)}")

        elif part.startswith("$") and part.endswith("$"):
            # Atom labels - would need reordering based on new atom order
            # For now, pass through unchanged
            new_parts.append(part)
        else:
            new_parts.append(part)

    return ",".join(new_parts)


def reconstruct_rgroups(
    canonical_with_maps: str,
    rgroups: list[tuple[int, int, str]],
    wildcard_indices: list[int],
    old_to_new: dict[int, int],
) -> str:
    """
    Replace [*:N] placeholders with <r>LABEL</r> tags in canonical SMILES.

    Args:
        canonical_with_maps: Canonical SMILES with atom map numbers
        rgroups: List of (start, end, label) from original parsing
        wildcard_indices: List of original atom indices for wildcards
        old_to_new: Mapping from old to new atom indices

    Returns:
        SMILES with <r>LABEL</r> tags at correct positions
    """
    # Map original wildcard indices to labels
    orig_idx_to_label = {wildcard_indices[i]: rgroups[i][2] for i in range(len(rgroups))}

    # Build new index to label mapping
    new_idx_to_label = {}
    for orig_idx, label in orig_idx_to_label.items():
        new_idx = old_to_new[orig_idx]
        new_idx_to_label[new_idx] = label

    # Replace [*:N] with <r>LABEL</r>
    def replace_wildcard(match: re.Match) -> str:
        map_num = int(match.group(1))
        old_idx = map_num - 1
        new_idx = old_to_new[old_idx]
        if new_idx in new_idx_to_label:
            return f"<r>{new_idx_to_label[new_idx]}</r>"
        return "[*]"

    return re.sub(r"\[\*:(\d+)\]", replace_wildcard, canonical_with_maps)


def _canonicalize_candidate(cxsmiles_with_rgroups: str) -> str | None:
    """Return a candidate canonical CXSMILES string before graph verification."""

    parsed = parse_cxsmiles(cxsmiles_with_rgroups)

    if not parsed.rgroups:
        # No R-groups, just canonicalize directly
        mol = Chem.MolFromSmiles(parsed.core)
        if mol is None:
            return None

        # Get atom mapping for extensions
        old_to_new, _, _ = get_atom_mapping(parsed.core)
        if old_to_new is None:
            return None

        canonical = Chem.MolToSmiles(mol)

        if parsed.extension:
            new_ext = reindex_extensions(parsed.extension, old_to_new)
            return f"{canonical}|{new_ext}"
        return canonical

    # Strip R-groups
    stripped = strip_rgroups(parsed.core)

    # Get atom mapping
    old_to_new, wildcard_indices, canonical_with_maps = get_atom_mapping(stripped)
    if old_to_new is None:
        return None

    # Reconstruct R-groups in canonical SMILES
    result_with_maps = reconstruct_rgroups(
        canonical_with_maps, parsed.rgroups, wildcard_indices, old_to_new
    )

    # Clean up bracket notation
    result_cleaned = clean_smiles_brackets(result_with_maps)

    # Reindex extensions
    if parsed.extension:
        new_ext = reindex_extensions(parsed.extension, old_to_new)
        return f"{result_cleaned}|{new_ext}"

    return result_cleaned


def canonicalize_cxsmiles_checked(cxsmiles_with_rgroups: str) -> CanonicalizeResult:
    """Canonicalize CXSMILES and report whether the guarded rewrite was accepted.

    A single RDKit canonicalization pass is not always a fixed point for
    CXSMILES with multiple disconnected R-label fragments and indexed
    extensions. Treat a target as canonical only after guarded rewrites converge
    to an idempotent value.
    """

    original = cxsmiles_with_rgroups
    current = cxsmiles_with_rgroups
    seen = {current}
    max_passes = 8

    for pass_index in range(max_passes):
        candidate = _canonicalize_candidate(current)
        if candidate is None:
            return CanonicalizeResult(
                value=None if pass_index == 0 else original,
                changed=False,
                accepted=False,
                fallback=pass_index > 0,
                status="parse_failed" if pass_index == 0 else "fallback_parse_failed",
                error=f"parse_failed: {current[:100]}",
            )

        is_same, verify_msg = verify_molecule_unchanged(current, candidate)
        if not is_same:
            return CanonicalizeResult(
                value=original,
                changed=False,
                accepted=False,
                fallback=True,
                status="fallback_unverified",
                error=verify_msg,
            )

        if candidate == current:
            if current != original:
                original_same, original_verify_msg = verify_molecule_unchanged(original, current)
                if not original_same:
                    return CanonicalizeResult(
                        value=original,
                        changed=False,
                        accepted=False,
                        fallback=True,
                        status="fallback_unverified",
                        error=original_verify_msg,
                    )
            changed = original != current
            return CanonicalizeResult(
                value=current,
                changed=changed,
                accepted=True,
                fallback=False,
                status="canonicalized" if changed else "already_canonical",
            )

        if candidate in seen:
            return CanonicalizeResult(
                value=original,
                changed=False,
                accepted=False,
                fallback=True,
                status="fallback_nonconvergent",
                error="canonicalization entered a cycle before reaching a fixed point",
            )

        seen.add(candidate)
        current = candidate

    return CanonicalizeResult(
        value=original,
        changed=False,
        accepted=False,
        fallback=True,
        status="fallback_nonconvergent",
        error=f"canonicalization did not converge within {max_passes} passes",
    )


def canonicalize_cxsmiles(cxsmiles_with_rgroups: str) -> str | None:
    """
    Canonicalize CXSMILES while preserving R-group labels and reindexing extensions.

    Args:
        cxsmiles_with_rgroups: CXSMILES string with <r>LABEL</r> tags

    Returns:
        Canonical CXSMILES with R-groups at correct positions and reindexed extensions,
        the original string when the guarded rewrite is not structure-preserving,
        or None if parsing fails.
    """

    return canonicalize_cxsmiles_checked(cxsmiles_with_rgroups).value


def verify_molecule_unchanged(original: str, canonical: str) -> tuple[bool, str]:
    """
    Verify the molecule structure is preserved after canonicalization.

    Uses the position-aware Markush graph relation, so R-label, ``m:``, and
    ``Sg:`` atom bindings must survive canonicalization. This deliberately
    rejects RDKit SMILES rewrites that are lossy around wildcard/dummy atoms in
    aromatic systems.

    Args:
        original: Original CXSMILES
        canonical: Canonical CXSMILES

    Returns:
        Tuple of (is_same, message)
    """
    from glyph.markush.eval.markush_graph import markush_graph_equal

    result = markush_graph_equal(original, canonical)
    if not result.valid_gt:
        return False, f"Original CXSMILES failed to parse: {result.gt_error}"
    if not result.valid_pred:
        return False, f"Canonical CXSMILES failed to parse: {result.pred_error}"
    if not result.equal:
        return False, result.error or "canonicalization changed Markush graph"
    return True, "OK"


def extract_cxsmiles_from_content(content: str) -> str | None:
    """Extract CXSMILES from <cxsmi>...</cxsmi> tags in content."""
    match = re.search(r"<cxsmi>(.*?)</cxsmi>", content)
    return match.group(1) if match else None


def replace_cxsmiles_in_content(content: str, new_cxsmiles: str) -> str:
    """Replace CXSMILES in <cxsmi>...</cxsmi> tags."""

    # Use a function replacement to avoid regex escape issues with backslashes
    def replacer(match: re.Match) -> str:
        return f"<cxsmi>{new_cxsmiles}</cxsmi>"

    return re.sub(r"<cxsmi>.*?</cxsmi>", replacer, content)


def process_record(record: dict) -> tuple[dict, bool, str | None]:
    """
    Process a single training record, canonicalizing the CXSMILES if present.

    Args:
        record: Training record dict with 'messages' list

    Returns:
        Tuple of (processed_record, was_changed, error_message)
    """
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            cxsmiles = extract_cxsmiles_from_content(content)

            if cxsmiles is None:
                return record, False, "no_cxsmiles"

            # Canonicalize
            result = canonicalize_cxsmiles_checked(cxsmiles)
            if result.value is None:
                return record, False, result.error or f"parse_failed: {cxsmiles[:100]}"
            if result.fallback:
                return record, False, f"verify_failed: {result.error}"
            canonical = result.value

            # Check if changed
            if cxsmiles == canonical:
                return record, False, None

            # Update content
            new_content = replace_cxsmiles_in_content(content, canonical)
            msg["content"] = new_content

            return record, True, None

    return record, False, "no_assistant_message"


def process_file(
    input_path: Path, output_path: Path, report: bool = False, max_records: int | None = None
) -> CanonicalizeStats:
    """
    Process a JSONL file, canonicalizing all CXSMILES.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        report: Whether to print detailed statistics
        max_records: Maximum number of records to process (for testing)

    Returns:
        CanonicalizeStats with processing statistics
    """
    total = 0
    changed = 0
    unchanged = 0
    failed = 0
    no_cxsmiles = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line_num, line in enumerate(f_in, 1):
            if max_records and total >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON parse error: {e}")
                failed += 1
                f_out.write(line + "\n")
                continue

            processed, was_changed, error = process_record(record)

            if error:
                if error == "no_cxsmiles":
                    no_cxsmiles += 1
                elif error.startswith("parse_failed") or error.startswith("verify_failed"):
                    failed += 1
                    if report:
                        logger.warning(f"Line {line_num}: {error}")
                else:
                    unchanged += 1
            elif was_changed:
                changed += 1
            else:
                unchanged += 1

            f_out.write(json.dumps(processed) + "\n")

    stats = CanonicalizeStats(
        total=total, changed=changed, unchanged=unchanged, failed=failed, no_cxsmiles=no_cxsmiles
    )

    if report:
        print("\nCanonicalization Statistics:")
        print(f"  Total records:      {stats.total}")
        print(f"  Changed:            {stats.changed} ({100 * stats.changed / stats.total:.1f}%)")
        print(
            f"  Already canonical:  {stats.unchanged} ({100 * stats.unchanged / stats.total:.1f}%)"
        )
        print(f"  No CXSMILES:        {stats.no_cxsmiles}")
        print(f"  Failed:             {stats.failed}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Canonicalize CXSMILES in training JSONL files")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--report", action="store_true", help="Print detailed statistics")
    parser.add_argument(
        "--max-records", type=int, default=None, help="Maximum records to process (for testing)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    stats = process_file(args.input, args.output, report=args.report, max_records=args.max_records)

    if stats.failed > 0 and stats.failed / stats.total > 0.1:
        logger.warning(f"High failure rate: {stats.failed}/{stats.total}")
        sys.exit(1)


if __name__ == "__main__":
    main()
