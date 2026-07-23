#!/usr/bin/env python3
"""Scaffold curation and CXSMILES seed generation for Markush training data.

This module implements the dependency-light stages of the data plan:

1. Build a filtered scaffold pool from a PubChem ``CID-SMILES`` dump.
2. Add R-group wildcards with CXSMILES atom-label sections.
3. Add realistic Sg repeating-unit and m positional-variation annotations.
4. Validate generated CXSMILES and write renderer-ready seed manifests.

The seed manifests written here are consumed by an external renderer (which
needs a JVM/CDK toolchain not required by this module) via its
``--cxsmiles_file`` option.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import logging
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, FilterCatalog, rdMolDescriptors

from glyph.paths import data_root

RDLogger.DisableLog("rdApp.*")

LOGGER = logging.getLogger(__name__)

ALLOWED_ELEMENTS = frozenset({"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"})
RGROUP_LABEL_BASES = ("R", "X", "Y", "Z", "A", "B", "D", "E", "G", "J", "K", "L", "M", "Q")
MOLPARSER_CONNECTION_POINT_PROP = "molparserConnectionPoint"

# Weighted R-group label pools matching IP5-M distribution.
# Each entry maps a category name to (label_list, probability).
LABEL_POOLS: dict[str, tuple[list[str], float]] = {
    "standard_r": (["R", "R0"] + [f"R{i}" for i in range(1, 21)], 0.34),
    "lettered_r": (
        [f"R{c}" for c in "abcdefghijklmnopqrstuvwxyz"] + ["R'", "R''", "R'''"],
        0.14,
    ),
    "attachment": (["_AP"], 0.07),
    "single_letter": (list("GZEJLXYAQWBDKM"), 0.11),
    "letter_number": ([f"{c}{i}" for c in "XYZAELJQKMGDB" for i in range(1, 81)], 0.15),
    "chemical": (["OR", "CF3", "CN", "Me", "OH", "NH", "OMe", "NR", "SCH3"], 0.04),
    "subscripted": (
        [f"R{i}{s}" for i in range(1, 9) for s in "abcd"],
        0.04,
    ),
    "high_index": ([f"R{i}" for i in range(21, 50)], 0.03),
    "other": ([f"{c}{i}" for c in "MEGBDJKLQ" for i in range(1, 120)], 0.08),
}
_LABEL_POOL_CATEGORIES = list(LABEL_POOLS.keys())
_LABEL_POOL_WEIGHTS = [LABEL_POOLS[k][1] for k in _LABEL_POOL_CATEGORIES]
# MarkushGenerator's get_boxes() SVG parser assumes 3 elements per bracket
# group (left, right, label).  HH/EU connectivity renders a 4th element
# (label at both ends), crashing the counter-based key assignment.  Restrict
# to HT until the upstream parser is fixed.  See SG_RENDER_FAILURE_ANALYSIS.md.
SG_CONNECTIVITY = ("ht",)
_WILDCARD_TOKEN_RE = re.compile(r"\[(?:\d+)?\*(?:@{1,2})?\]|\[Ar\]|\*")
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


@dataclass(frozen=True)
class ScaffoldFilterConfig:
    """Drug-like scaffold filter used while streaming PubChem SMILES."""

    min_heavy_atoms: int = 3
    max_heavy_atoms: int = 80
    max_mol_weight: float = 600.0
    max_logp: float = 6.0
    max_hbd: int = 6
    max_hba: int = 12
    max_rotatable_bonds: int = 12
    max_tpsa: float = 160.0
    min_rings: int = 0
    allowed_elements: frozenset[str] = ALLOWED_ELEMENTS
    require_carbon: bool = True
    exclude_pains: bool = True


@dataclass(frozen=True)
class DecorationConfig:
    """Controls R-group, Sg, and m annotation generation."""

    min_rgroups: int = 1
    max_rgroups: int = 8
    high_rgroup_probability: float = 0.12  # probability of high-count R-group tail
    high_rgroup_min: int = 9
    high_rgroup_max: int = 22
    label_style: str = "mixed"  # "R_only" or "mixed"
    sg_probability: float = 0.30
    m_probability: float = 0.32
    max_sg_annotations: int = 2  # HT-only restriction stays (see SG_RENDER_FAILURE_ANALYSIS.md)
    max_m_annotations: int = 2
    min_sg_atoms: int = 2
    max_sg_atoms: int = 8
    prefer_chain_sg: bool = True
    strict_validation: bool = True
    zero_rgroup_probability: float = 0.13  # probability of Sg/m-only, no R-groups
    fragment_probability: float = 0.30  # probability of disconnected fragment notation


DECORATION_PROFILES: dict[str, DecorationConfig] = {
    "hybrid": DecorationConfig(),
    "eval_aligned": DecorationConfig(
        max_rgroups=8,
        high_rgroup_probability=0.18,
        high_rgroup_min=9,
        high_rgroup_max=24,
        sg_probability=0.25,
        m_probability=0.34,
        zero_rgroup_probability=0.15,
        fragment_probability=0.34,
    ),
    "patent_broad": DecorationConfig(
        max_rgroups=10,
        high_rgroup_probability=0.20,
        high_rgroup_min=9,
        high_rgroup_max=28,
        sg_probability=0.36,
        m_probability=0.30,
        zero_rgroup_probability=0.10,
        fragment_probability=0.26,
    ),
    "renderer_safe": DecorationConfig(
        max_rgroups=8,
        high_rgroup_probability=0.10,
        high_rgroup_min=9,
        high_rgroup_max=18,
        sg_probability=0.30,
        m_probability=0.18,
        zero_rgroup_probability=0.08,
        fragment_probability=0.25,
    ),
}


def decoration_config_for_profile(
    profile: str = "hybrid",
    **overrides: Any,
) -> DecorationConfig:
    """Return a named decoration profile with non-``None`` overrides applied."""

    if profile not in DECORATION_PROFILES:
        choices = ", ".join(sorted(DECORATION_PROFILES))
        raise ValueError(f"Unknown decoration profile {profile!r}; choose one of: {choices}")
    config = DECORATION_PROFILES[profile]
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **clean_overrides)


@dataclass
class ValidationResult:
    """Validation report for a standard or optimized CXSMILES string."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


@dataclass
class DecoratedMarkush:
    """A generated Markush seed with standard and optimized CXSMILES forms."""

    scaffold_smiles: str
    core_smiles: str
    cxsmiles: str
    cxsmiles_opt: str
    rgroup_labels: list[str]
    sg_annotations: list[dict[str, Any]]
    m_annotations: list[dict[str, Any]]
    validation: ValidationResult

    @property
    def has_sg(self) -> bool:
        return bool(self.sg_annotations)

    @property
    def has_m(self) -> bool:
        return bool(self.m_annotations)

    def to_record(self, sample_id: str) -> dict[str, Any]:
        return {
            "id": sample_id,
            "scaffold_smiles": self.scaffold_smiles,
            "core_smiles": self.core_smiles,
            "cxsmiles": self.cxsmiles,
            "cxsmiles_opt": self.cxsmiles_opt,
            "features": {
                "rgroup_count": len(self.rgroup_labels),
                "rgroup_labels": self.rgroup_labels,
                "has_sg": self.has_sg,
                "has_m": self.has_m,
                "sg_count": len(self.sg_annotations),
                "m_count": len(self.m_annotations),
            },
            "sg_annotations": self.sg_annotations,
            "m_annotations": self.m_annotations,
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True)
class FailureModeFocusSpec:
    """A focused synthetic bucket for current Markush eval failure modes."""

    name: str
    description: str
    config: DecorationConfig


@dataclass(frozen=True)
class FocusedDecoratedMarkush:
    """A decorated seed plus the failure-mode focus it was generated for."""

    focus: str
    sample: DecoratedMarkush


def _has_fragment_notation(sample: DecoratedMarkush) -> bool:
    core, _extension = split_cxsmiles(sample.cxsmiles_opt)
    return "." in core


FAILURE_MODE_FOCUS_SPECS: dict[str, FailureModeFocusSpec] = {
    "backbone": FailureModeFocusSpec(
        name="backbone",
        description="Backbone recognition with low annotation load.",
        config=DecorationConfig(
            min_rgroups=1,
            max_rgroups=3,
            high_rgroup_probability=0.0,
            sg_probability=0.0,
            m_probability=0.0,
            zero_rgroup_probability=0.0,
            fragment_probability=0.0,
        ),
    ),
    "rgroup": FailureModeFocusSpec(
        name="rgroup",
        description="High R-group count and eval-like label variety without Sg/m noise.",
        config=DecorationConfig(
            min_rgroups=5,
            max_rgroups=8,
            high_rgroup_probability=0.35,
            high_rgroup_min=9,
            high_rgroup_max=24,
            sg_probability=0.0,
            m_probability=0.0,
            zero_rgroup_probability=0.0,
            fragment_probability=0.0,
        ),
    ),
    "sgroup": FailureModeFocusSpec(
        name="sgroup",
        description="Isolated S-group annotation cases.",
        config=DecorationConfig(
            min_rgroups=1,
            max_rgroups=4,
            high_rgroup_probability=0.0,
            sg_probability=1.0,
            m_probability=0.0,
            max_sg_annotations=2,
            zero_rgroup_probability=0.0,
            fragment_probability=0.0,
        ),
    ),
    "msection": FailureModeFocusSpec(
        name="msection",
        description="Isolated positional-variation m-section cases.",
        config=DecorationConfig(
            min_rgroups=1,
            max_rgroups=4,
            high_rgroup_probability=0.0,
            sg_probability=0.0,
            m_probability=1.0,
            max_m_annotations=2,
            zero_rgroup_probability=0.0,
            fragment_probability=0.0,
        ),
    ),
    "fragment": FailureModeFocusSpec(
        name="fragment",
        description="Disconnected R-group fragment notation cases.",
        config=DecorationConfig(
            min_rgroups=2,
            max_rgroups=8,
            high_rgroup_probability=0.15,
            high_rgroup_min=9,
            high_rgroup_max=18,
            sg_probability=0.0,
            m_probability=0.0,
            zero_rgroup_probability=0.0,
            fragment_probability=1.0,
        ),
    ),
}
FAILURE_MODE_FOCUS_CHOICES = tuple(FAILURE_MODE_FOCUS_SPECS)


def _matches_failure_mode_focus(sample: DecoratedMarkush, focus: str) -> bool:
    """Return whether ``sample`` isolates the requested failure-mode focus."""

    fragment = _has_fragment_notation(sample)
    rgroup_count = len(sample.rgroup_labels)
    if focus == "backbone":
        return 1 <= rgroup_count <= 3 and not sample.has_sg and not sample.has_m and not fragment
    if focus == "rgroup":
        return rgroup_count >= 5 and not sample.has_sg and not sample.has_m and not fragment
    if focus == "sgroup":
        return sample.has_sg and not sample.has_m and not fragment
    if focus == "msection":
        return sample.has_m and not sample.has_sg and not fragment
    if focus == "fragment":
        return fragment and not sample.has_sg and not sample.has_m
    raise ValueError(f"Unknown failure-mode focus {focus!r}")


def _normalize_failure_mode_focuses(focuses: Iterable[str] | None) -> list[str]:
    if focuses is None:
        return list(FAILURE_MODE_FOCUS_CHOICES)
    normalized: list[str] = []
    for focus in focuses:
        key = focus.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in FAILURE_MODE_FOCUS_SPECS:
            choices = ", ".join(FAILURE_MODE_FOCUS_CHOICES)
            raise ValueError(f"Unknown failure-mode focus {focus!r}; choose one of: {choices}")
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("At least one failure-mode focus is required")
    return normalized


def generate_failure_mode_decorated_samples(
    scaffolds: list[str],
    *,
    samples_per_focus: int,
    focuses: Iterable[str] | None = None,
    seed: int = 42,
    max_attempts_per_sample: int = 100,
    progress_interval: int = 0,
) -> tuple[list[FocusedDecoratedMarkush], dict[str, Any]]:
    """Generate renderer-ready seeds balanced across failure-mode focus buckets."""

    if not scaffolds:
        raise ValueError("scaffolds cannot be empty")
    if samples_per_focus <= 0:
        raise ValueError("samples_per_focus must be positive")
    if max_attempts_per_sample <= 0:
        raise ValueError("max_attempts_per_sample must be positive")
    if progress_interval < 0:
        raise ValueError("progress_interval must be non-negative")

    focus_list = _normalize_failure_mode_focuses(focuses)
    rng = random.Random(seed)
    focused_samples: list[FocusedDecoratedMarkush] = []
    stats: dict[str, Any] = {
        "samples_per_focus": samples_per_focus,
        "requested_focuses": focus_list,
        "focuses": {},
        "requested": samples_per_focus * len(focus_list),
        "generated": 0,
        "attempted": 0,
        "failed": 0,
        "focus_rejected": 0,
    }

    for focus in focus_list:
        spec = FAILURE_MODE_FOCUS_SPECS[focus]
        generated_for_focus = 0
        attempted_for_focus = 0
        failed_for_focus = 0
        rejected_for_focus = 0
        max_attempts = samples_per_focus * max_attempts_per_sample
        while generated_for_focus < samples_per_focus and attempted_for_focus < max_attempts:
            attempted_for_focus += 1
            stats["attempted"] += 1
            scaffold = rng.choice(scaffolds)
            made = decorate_scaffold(scaffold, config=spec.config, rng=rng)
            if made is None:
                failed_for_focus += 1
                stats["failed"] += 1
                continue
            if not _matches_failure_mode_focus(made, focus):
                rejected_for_focus += 1
                stats["focus_rejected"] += 1
                continue
            focused_samples.append(FocusedDecoratedMarkush(focus=focus, sample=made))
            generated_for_focus += 1
            stats["generated"] += 1
            if progress_interval and generated_for_focus % progress_interval == 0:
                LOGGER.info(
                    "Generated %d/%d %s-focused seeds",
                    generated_for_focus,
                    samples_per_focus,
                    focus,
                )

        stats["focuses"][focus] = {
            "description": spec.description,
            "requested": samples_per_focus,
            "generated": generated_for_focus,
            "attempted": attempted_for_focus,
            "failed": failed_for_focus,
            "focus_rejected": rejected_for_focus,
        }

    return focused_samples, stats


def iter_cid_smiles(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(cid, smiles)`` rows from PubChem ``CID-SMILES`` text or gzip files."""

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            cid, smiles = parts
            if cid and smiles:
                yield cid, smiles


def build_pains_catalog() -> FilterCatalog.FilterCatalog:
    """Build RDKit's PAINS filter catalog."""

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def passes_scaffold_filter(
    mol: Chem.Mol,
    config: ScaffoldFilterConfig = ScaffoldFilterConfig(),
    pains_catalog: FilterCatalog.FilterCatalog | None = None,
) -> bool:
    """Return whether ``mol`` is suitable as a Markush scaffold seed."""

    n_atoms = mol.GetNumHeavyAtoms()
    if not (config.min_heavy_atoms <= n_atoms <= config.max_heavy_atoms):
        return False

    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if config.require_carbon and "C" not in symbols:
        return False
    if not symbols <= set(config.allowed_elements):
        return False

    try:
        if Descriptors.MolWt(mol) > config.max_mol_weight:
            return False
        if Descriptors.MolLogP(mol) > config.max_logp:
            return False
        if rdMolDescriptors.CalcNumHBD(mol) > config.max_hbd:
            return False
        if rdMolDescriptors.CalcNumHBA(mol) > config.max_hba:
            return False
        if rdMolDescriptors.CalcNumRotatableBonds(mol) > config.max_rotatable_bonds:
            return False
        if rdMolDescriptors.CalcTPSA(mol) > config.max_tpsa:
            return False
        if rdMolDescriptors.CalcNumRings(mol) < config.min_rings:
            return False
    except Exception:
        return False

    return not (config.exclude_pains and pains_catalog is not None and pains_catalog.HasMatch(mol))


def _dedupe_key(mol: Chem.Mol) -> str:
    """Use InChI when available, falling back to canonical SMILES."""

    try:
        return Chem.MolToInchi(mol)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def build_scaffold_pool(
    cid_smiles_path: str | Path,
    output_path: str | Path,
    *,
    target_count: int = 200_000,
    sample_rate: float = 0.01,
    seed: int = 42,
    filter_config: ScaffoldFilterConfig = ScaffoldFilterConfig(),
    progress_interval: int = 5_000_000,
) -> dict[str, int]:
    """Stream PubChem CID-SMILES and write a filtered scaffold pool.

    The implementation samples before RDKit parsing so it can scan the full
    PubChem dump without holding it in memory.  Output is one canonical SMILES
    per line.
    """

    if not (0 < sample_rate <= 1):
        raise ValueError("sample_rate must be in (0, 1]")
    if target_count <= 0:
        raise ValueError("target_count must be positive")

    rng = random.Random(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pains_catalog = build_pains_catalog() if filter_config.exclude_pains else None
    seen: set[str] = set()
    pool: list[str] = []
    stats = {
        "processed": 0,
        "sampled": 0,
        "parse_failed": 0,
        "filtered": 0,
        "duplicates": 0,
        "written": 0,
    }

    for _cid, smiles in iter_cid_smiles(cid_smiles_path):
        stats["processed"] += 1
        if progress_interval and stats["processed"] % progress_interval == 0:
            LOGGER.info(
                "processed=%d sampled=%d accepted=%d",
                stats["processed"],
                stats["sampled"],
                len(pool),
            )

        if rng.random() > sample_rate:
            continue
        stats["sampled"] += 1

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            stats["parse_failed"] += 1
            continue

        if not passes_scaffold_filter(mol, filter_config, pains_catalog):
            stats["filtered"] += 1
            continue

        key = _dedupe_key(mol)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)

        pool.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        if len(pool) >= target_count:
            break

    rng.shuffle(pool)
    with open(output_path, "w", encoding="utf-8") as handle:
        for smiles in pool:
            handle.write(smiles + "\n")

    stats["written"] = len(pool)
    return stats


def load_scaffold_pool(path: str | Path) -> list[str]:
    """Read a newline-delimited scaffold pool, dropping blank/comment lines."""

    scaffolds: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                scaffolds.append(line)
    return scaffolds


def find_rgroup_attachment_candidates(mol: Chem.Mol) -> list[int]:
    """Find atoms where replacing an implicit H with an R-group is plausible."""

    ring_info = mol.GetRingInfo()
    small_ring_atoms = {
        atom_idx for ring in ring_info.AtomRings() if len(ring) <= 3 for atom_idx in ring
    }
    candidates: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() in small_ring_atoms:
            continue
        if atom.GetSymbol() not in {"C", "N", "O", "S"}:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs() <= 0:
            continue
        candidates.append(atom.GetIdx())
    return candidates


def generate_rgroup_labels(n: int, style: str, rng: random.Random) -> list[str]:
    """Generate unique R-group labels matching IP5-M distribution."""

    if n <= 0:
        return []
    if style == "R_only":
        return [f"R{i + 1}" for i in range(n)]
    if style != "mixed":
        raise ValueError("label_style must be 'mixed' or 'R_only'")

    labels: list[str] = []
    used: set[str] = set()
    # Weighted sampling from IP5-M-matched label pools
    max_attempts = n * 20  # safety valve to avoid infinite loop
    attempts = 0
    while len(labels) < n and attempts < max_attempts:
        attempts += 1
        category = rng.choices(_LABEL_POOL_CATEGORIES, weights=_LABEL_POOL_WEIGHTS, k=1)[0]
        pool, _weight = LABEL_POOLS[category]
        label = rng.choice(pool)
        if label in used:
            continue
        used.add(label)
        labels.append(label)

    # Fallback: if we exhausted attempts (very unlikely), fill remaining with R{n}
    while len(labels) < n:
        fallback = f"R{len(labels) + 1}"
        if fallback not in used:
            used.add(fallback)
            labels.append(fallback)
        else:
            labels.append(f"R{rng.randint(50, 99)}")
    return labels


def _smiles_atom_output_order(mol: Chem.Mol) -> list[int]:
    if not mol.HasProp("_smilesAtomOutputOrder"):
        return list(range(mol.GetNumAtoms()))
    raw = mol.GetProp("_smilesAtomOutputOrder")
    try:
        order = ast.literal_eval(raw)
    except Exception:
        return list(range(mol.GetNumAtoms()))
    if not isinstance(order, list) or len(order) != mol.GetNumAtoms():
        return list(range(mol.GetNumAtoms()))
    return [int(x) for x in order]


def add_rgroups_to_molecule(
    mol: Chem.Mol,
    config: DecorationConfig,
    rng: random.Random,
) -> tuple[Chem.Mol, dict[int, str]] | None:
    """Attach wildcard atoms to randomly selected implicit-H sites."""

    candidates = find_rgroup_attachment_candidates(mol)
    if len(candidates) < config.min_rgroups:
        return None

    if rng.random() < config.high_rgroup_probability and len(candidates) >= config.high_rgroup_min:
        max_count = min(config.high_rgroup_max, len(candidates))
        n_rgroups = rng.randint(config.high_rgroup_min, max_count)
    else:
        max_count = min(config.max_rgroups, len(candidates))
        n_rgroups = rng.randint(config.min_rgroups, max_count)

    attachment_points = rng.sample(candidates, n_rgroups)
    labels = generate_rgroup_labels(n_rgroups, config.label_style, rng)

    editable = Chem.RWMol(mol)
    rgroup_map: dict[int, str] = {}
    for atom_idx, label in zip(attachment_points, labels, strict=True):
        wildcard = Chem.Atom(0)
        wildcard.SetNoImplicit(True)
        wildcard_idx = editable.AddAtom(wildcard)
        editable.AddBond(atom_idx, wildcard_idx, Chem.BondType.SINGLE)
        rgroup_map[wildcard_idx] = label

    decorated = editable.GetMol()
    try:
        decorated.UpdatePropertyCache(strict=False)
    except Exception:
        return None
    return decorated, rgroup_map


def _ring_atoms(mol: Chem.Mol) -> set[int]:
    return {atom_idx for ring in mol.GetRingInfo().AtomRings() for atom_idx in ring}


def _crossing_bond_count(mol: Chem.Mol, group: Iterable[int]) -> int:
    group_set = set(group)
    count = 0
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if (a in group_set) != (b in group_set):
            count += 1
    return count


def _indices_connected(mol: Chem.Mol, indices: Iterable[int]) -> bool:
    index_set = set(indices)
    if not index_set:
        return False

    start = next(iter(index_set))
    seen = {start}
    stack = [start]
    while stack:
        atom_idx = stack.pop()
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in index_set and n_idx not in seen:
                seen.add(n_idx)
                stack.append(n_idx)
    return seen == index_set


def _grow_contiguous_group(
    mol: Chem.Mol,
    start: int,
    excluded: set[int],
    min_size: int,
    max_size: int,
    rng: random.Random,
) -> list[int] | None:
    group = [start]
    frontier = [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(start).GetNeighbors()
        if neighbor.GetIdx() not in excluded
    ]
    target_size = rng.randint(min_size, max_size)

    while len(group) < target_size and frontier:
        rng.shuffle(frontier)
        atom_idx = frontier.pop(0)
        if atom_idx in excluded or atom_idx in group:
            continue
        group.append(atom_idx)
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in excluded and n_idx not in group:
                frontier.append(n_idx)

    if len(group) < min_size:
        return None
    return group


def _find_sg_candidate(
    mol: Chem.Mol,
    excluded: set[int],
    config: DecorationConfig,
    rng: random.Random,
) -> list[int] | None:
    ring_atoms = _ring_atoms(mol)
    all_candidates = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 0 and atom.GetIdx() not in excluded
    ]
    if len(all_candidates) < config.min_sg_atoms:
        return None

    chain_candidates = [idx for idx in all_candidates if idx not in ring_atoms]
    candidate_sets = (
        [chain_candidates, all_candidates] if config.prefer_chain_sg else [all_candidates]
    )

    for candidates in candidate_sets:
        if len(candidates) < config.min_sg_atoms:
            continue
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        for start in shuffled:
            group = _grow_contiguous_group(
                mol,
                start,
                excluded,
                config.min_sg_atoms,
                min(config.max_sg_atoms, len(all_candidates)),
                rng,
            )
            if not group:
                continue
            if _crossing_bond_count(mol, group) == 2:
                return sorted(group)
    return None


def generate_sg_label(rng: random.Random) -> str:
    """Generate a bracket subscript label seen in patent Markush structures."""

    roll = rng.random()
    if roll < 0.45:
        return rng.choice(("n", "m", "x", "y", "p", "q"))
    if roll < 0.65:
        return str(rng.randint(1, 20))
    if roll < 0.85:
        lo = rng.randint(0, 5)
        hi = lo + rng.randint(1, 10)
        return f"{lo}-{hi}"
    return rng.choice(("n+1", "n-1", ">=1", ">0", "n+m"))


def add_sg_annotations(
    mol: Chem.Mol,
    rgroup_atom_indices: Iterable[int],
    config: DecorationConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate Sg annotations over contiguous non-wildcard subgraphs."""

    if rng.random() > config.sg_probability:
        return []

    n_sg = 1 if config.max_sg_annotations <= 1 else rng.choices([1, 2], weights=[0.8, 0.2])[0]
    n_sg = min(n_sg, config.max_sg_annotations)
    excluded = set(rgroup_atom_indices)
    annotations: list[dict[str, Any]] = []

    for _ in range(n_sg):
        group = _find_sg_candidate(mol, excluded, config, rng)
        if group is None:
            break
        annotations.append(
            {
                "atom_ids": group,
                "label": generate_sg_label(rng),
                "connectivity": rng.choice(SG_CONNECTIVITY),
            }
        )
        excluded.update(group)

    return annotations


def _ring_containing_atom(rings: list[tuple[int, ...]], atom_idx: int) -> tuple[int, ...] | None:
    for ring in rings:
        if atom_idx in ring:
            return ring
    return None


def add_m_annotations(
    mol: Chem.Mol,
    rgroup_atom_indices: Iterable[int],
    config: DecorationConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate positional variation annotations on ring substituents."""

    if rng.random() > config.m_probability:
        return []

    rings = list(mol.GetRingInfo().AtomRings())
    if not rings:
        return []

    annotations: list[dict[str, Any]] = []
    used_centers: set[int] = set()

    rgroups = list(rgroup_atom_indices)
    rng.shuffle(rgroups)
    for rgroup_idx in rgroups:
        if len(annotations) >= config.max_m_annotations:
            break
        atom = mol.GetAtomWithIdx(rgroup_idx)
        if atom.GetAtomicNum() != 0:
            continue
        for neighbor in atom.GetNeighbors():
            ring = _ring_containing_atom(rings, neighbor.GetIdx())
            if ring is None or rgroup_idx in used_centers:
                continue
            n_endpoints = rng.randint(2, min(len(ring), 6))
            endpoints = rng.sample(list(ring), n_endpoints)
            if neighbor.GetIdx() not in endpoints:
                endpoints[0] = neighbor.GetIdx()
            annotations.append(
                {"center": rgroup_idx, "endpoints": sorted(endpoints), "source": "rgroup_ring"}
            )
            used_centers.add(rgroup_idx)
            break

    if annotations:
        return annotations

    shuffled_rings = list(rings)
    rng.shuffle(shuffled_rings)
    for ring in shuffled_rings:
        ring_set = set(ring)
        ring_atoms = list(ring)
        rng.shuffle(ring_atoms)
        for ring_atom_idx in ring_atoms:
            ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
            substituents = [
                neighbor.GetIdx()
                for neighbor in ring_atom.GetNeighbors()
                if neighbor.GetIdx() not in ring_set and neighbor.GetAtomicNum() != 0
            ]
            rng.shuffle(substituents)
            for center_idx in substituents:
                if center_idx in used_centers:
                    continue
                n_endpoints = rng.randint(2, min(len(ring), 6))
                endpoints = rng.sample(list(ring), n_endpoints)
                if ring_atom_idx not in endpoints:
                    endpoints[0] = ring_atom_idx
                annotations.append(
                    {
                        "center": center_idx,
                        "endpoints": sorted(endpoints),
                        "source": "substituent_ring",
                    }
                )
                used_centers.add(center_idx)
                if len(annotations) >= config.max_m_annotations:
                    return annotations
                break
            if annotations:
                return annotations

    return annotations


def assemble_cxsmiles(
    core_smiles: str,
    atom_output_order: list[int],
    rgroup_map: dict[int, str],
    sg_annotations: list[dict[str, Any]],
    m_annotations: list[dict[str, Any]],
) -> str:
    """Assemble standard CXSMILES using atom indices in SMILES output order."""

    old_to_output = {old_idx: out_idx for out_idx, old_idx in enumerate(atom_output_order)}
    n_atoms = len(atom_output_order)

    dollar_entries = ["" for _ in range(n_atoms)]
    for atom_idx, label in rgroup_map.items():
        if atom_idx in old_to_output:
            dollar_entries[old_to_output[atom_idx]] = label

    extension_parts = ["$" + ";".join(dollar_entries) + "$"]

    for sg in sg_annotations:
        mapped = sorted(old_to_output[idx] for idx in sg["atom_ids"] if idx in old_to_output)
        if len(mapped) != len(sg["atom_ids"]):
            continue
        atom_ids = ",".join(str(idx) for idx in mapped)
        label = sg.get("label", "n")
        connectivity = sg.get("connectivity", "ht")
        extension_parts.append(f"Sg:n:{atom_ids}:{label}:{connectivity}")

    for m_ann in m_annotations:
        center = m_ann["center"]
        endpoints = m_ann["endpoints"]
        if center not in old_to_output or any(idx not in old_to_output for idx in endpoints):
            continue
        mapped_center = old_to_output[center]
        mapped_endpoints = sorted(old_to_output[idx] for idx in endpoints)
        endpoint_text = ".".join(str(idx) for idx in mapped_endpoints)
        extension_parts.append(f"m:{mapped_center}:{endpoint_text}")

    return f"{core_smiles} |{','.join(extension_parts)}|"


def split_cxsmiles(cxsmiles: str) -> tuple[str, str]:
    """Split a standard or optimized CXSMILES into ``(core, extension)``."""

    value = cxsmiles.strip()
    if "|" not in value:
        return value, ""
    core, rest = value.split("|", 1)
    if "|" in rest:
        rest = rest.rsplit("|", 1)[0]
    return core.strip(), rest.strip()


def split_extension_sections(extension: str) -> list[str]:
    """Split CXSMILES extension sections without splitting Sg index lists."""

    if not extension:
        return []

    sections: list[str] = []
    current: list[str] = []
    for i, char in enumerate(extension):
        if char == ",":
            j = i + 1
            while j < len(extension) and extension[j] == ",":
                j += 1
            rest = extension[j:]
            if not rest or rest.startswith(CX_SECTION_STARTS):
                sections.append("".join(current))
                current = []
            else:
                current.append(char)
        else:
            current.append(char)
    if current:
        sections.append("".join(current))
    return [section for section in sections if section]


def _parse_int_list(text: str, separator: str) -> list[int]:
    return [int(part) for part in text.split(separator) if part != ""]


def _atom_prop_labels(section: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    if not section.startswith("atomProp:"):
        return labels
    for entry in section.removeprefix("atomProp:").split(":"):
        try:
            atom_idx_text, prop, value = entry.split(".", 2)
            atom_idx = int(atom_idx_text)
        except ValueError:
            continue
        if prop == "dummyLabel" and value.strip():
            labels[atom_idx] = value.strip()
    return labels


def _trim_trailing_empty_labels(labels: list[str]) -> list[str]:
    trimmed = list(labels)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed


def _add_label(labels_by_atom: dict[int, str], atom_idx: int, label: str) -> None:
    existing = labels_by_atom.get(atom_idx)
    if existing is not None and existing != label:
        raise ValueError(f"conflicting atom labels for atom {atom_idx}: {existing!r} vs {label!r}")
    labels_by_atom[atom_idx] = label


def _same_ring(mol: Chem.Mol, atom_indices: Iterable[int]) -> bool:
    atom_set = set(atom_indices)
    if not atom_set:
        return False
    return any(atom_set <= set(ring) for ring in mol.GetRingInfo().AtomRings())


def validate_cxsmiles(cxsmiles: str) -> ValidationResult:
    """Validate parseability and internal consistency of CXSMILES extensions."""

    result = ValidationResult()
    core, extension = split_cxsmiles(cxsmiles)
    core_for_parse = re.sub(r"<r>[^<]+</r>", "*", core)
    core_for_parse = re.sub(r"\[Ar\]", "*", core_for_parse)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.strictCXSMILES = False
    mol = Chem.MolFromSmiles(core_for_parse, params)
    if mol is None:
        result.add_error("SMILES core does not parse")
        return result

    n_atoms = mol.GetNumAtoms()
    sections = split_extension_sections(extension)
    label_section_seen = False

    for section in sections:
        if section.startswith("$"):
            label_section_seen = True
            if not section.endswith("$"):
                result.add_error("$ atom-label section is not closed")
                continue
            labels = _trim_trailing_empty_labels(section[1:-1].split(";"))
            if len(labels) > n_atoms:
                result.add_error(
                    f"$ atom-label section has {len(labels)} entries but molecule has {n_atoms} atoms"
                )
            continue

        if section.startswith("atomProp:"):
            label_section_seen = True
            for entry in section.removeprefix("atomProp:").split(":"):
                try:
                    atom_idx_text, _prop, _value = entry.split(".", 2)
                    atom_idx = int(atom_idx_text)
                except ValueError:
                    result.add_error(f"Malformed atomProp entry: {entry}")
                    continue
                if atom_idx < 0 or atom_idx >= n_atoms:
                    result.add_error(
                        f"atomProp atom index {atom_idx} out of range for {n_atoms} atoms"
                    )
            continue

        if section.startswith("Sg:"):
            parts = section.rstrip(",").split(":")
            while parts and parts[-1] == "":
                parts.pop()
            if len(parts) < 5:
                result.add_error(f"Malformed Sg section: {section}")
                continue
            try:
                atom_ids = _parse_int_list(parts[2], ",")
            except ValueError:
                result.add_error(f"Malformed Sg atom indices: {section}")
                continue
            if len(atom_ids) < 2:
                result.add_error(f"Sg section has fewer than two atoms: {section}")
                continue
            for atom_idx in atom_ids:
                if atom_idx < 0 or atom_idx >= n_atoms:
                    result.add_error(f"Sg atom index {atom_idx} out of range for {n_atoms} atoms")
            if result.errors:
                continue
            if any(mol.GetAtomWithIdx(idx).GetAtomicNum() == 0 for idx in atom_ids):
                result.add_error(f"Sg section includes wildcard atom: {section}")
            if not _indices_connected(mol, atom_ids):
                result.add_error(f"Sg atoms are not contiguous: {section}")
            crossing = _crossing_bond_count(mol, atom_ids)
            if crossing != 2:
                result.add_warning(
                    f"Sg section has {crossing} crossing bonds, expected 2: {section}"
                )
            continue

        if section.startswith("m:"):
            parts = section.split(":")
            if len(parts) < 3:
                result.add_error(f"Malformed m section: {section}")
                continue
            try:
                center = int(parts[1])
                endpoints = _parse_int_list(parts[2], ".")
            except ValueError:
                result.add_error(f"Malformed m atom indices: {section}")
                continue
            if center < 0 or center >= n_atoms:
                result.add_error(f"m center index {center} out of range for {n_atoms} atoms")
            if len(endpoints) < 2:
                result.add_error(f"m section has fewer than two endpoints: {section}")
            for endpoint in endpoints:
                if endpoint < 0 or endpoint >= n_atoms:
                    result.add_error(
                        f"m endpoint index {endpoint} out of range for {n_atoms} atoms"
                    )
            if not any(
                endpoint < 0 or endpoint >= n_atoms for endpoint in endpoints
            ) and not _same_ring(mol, endpoints):
                result.add_error(f"m endpoints are not on the same ring: {section}")
            continue

        result.add_warning(f"Unknown CXSMILES extension section: {section}")

    if extension and not label_section_seen:
        result.add_warning("CXSMILES extension has no $ atom-label section")

    return result


def standard_to_opt_cxsmiles(cxsmiles: str) -> str:
    """Convert generated standard CXSMILES into the training ``<r>`` form.

    This lightweight converter handles the exact standard form emitted by this
    module.  MarkushGenerator's ``CXSMILESTokenizer`` remains the preferred
    converter for CDK-generated MOL files because it can use renderer atom maps.
    """

    core, extension = split_cxsmiles(cxsmiles)
    sections = split_extension_sections(extension)
    dollar_labels: list[str] = []
    remaining_sections: list[str] = []
    labels_by_atom: dict[int, str] = {}
    preserved_atomprop_entries: list[str] = []

    for section in sections:
        if section.startswith("$") and section.endswith("$"):
            dollar_labels = _trim_trailing_empty_labels(section[1:-1].split(";"))
        elif section.startswith("atomProp:"):
            for atom_idx, label in _atom_prop_labels(section).items():
                _add_label(labels_by_atom, atom_idx, label)
            for entry in section.removeprefix("atomProp:").split(":"):
                try:
                    _atom_idx_text, prop, _value = entry.split(".", 2)
                except ValueError:
                    preserved_atomprop_entries.append(entry)
                    continue
                if prop != "dummyLabel":
                    preserved_atomprop_entries.append(entry)
        else:
            remaining_sections.append(section)

    for atom_idx, label in enumerate(dollar_labels):
        if label.strip():
            _add_label(labels_by_atom, atom_idx, label.strip())
    marker_base = 9000
    markers: list[int] = []
    implicit_marker_labels: dict[int, str] = {}

    def mark_wildcard(match: re.Match[str]) -> str:
        marker = marker_base + len(markers)
        markers.append(marker)
        if match.group(0) == "[Ar]":
            implicit_marker_labels[marker] = "Ar"
        return f"[*:{marker}]"

    marker_to_label: dict[int, str] = {}
    converted_label_atoms: set[int] = set()
    if labels_by_atom or "[Ar]" in core:
        marked_core = _WILDCARD_TOKEN_RE.sub(mark_wildcard, core)
        params = Chem.SmilesParserParams()
        params.removeHs = False
        params.strictCXSMILES = False
        mol = Chem.MolFromSmiles(marked_core, params)
        if mol is not None:
            for atom in mol.GetAtoms():
                marker = atom.GetAtomMapNum()
                if marker in markers:
                    label = labels_by_atom.get(
                        atom.GetIdx(),
                        implicit_marker_labels.get(marker, ""),
                    )
                    if label:
                        marker_to_label[marker] = label
                        converted_label_atoms.add(atom.GetIdx())

    residual_atomprop_entries = [
        f"{atom_idx}.dummyLabel.{label}"
        for atom_idx, label in sorted(labels_by_atom.items())
        if atom_idx not in converted_label_atoms
    ]
    atomprop_entries = [*residual_atomprop_entries, *preserved_atomprop_entries]
    if atomprop_entries:
        remaining_sections.insert(0, f"atomProp:{':'.join(atomprop_entries)}")

    replacement_idx = 0

    def replace_wildcard(match: re.Match[str]) -> str:
        nonlocal replacement_idx
        marker = marker_base + replacement_idx
        replacement_idx += 1
        label = marker_to_label.get(marker, "")
        if not label:
            return match.group(0)
        return f"<r>{label}</r>" if label else match.group(0)

    opt_core = _WILDCARD_TOKEN_RE.sub(replace_wildcard, core)
    if remaining_sections:
        return f"{opt_core}|{','.join(remaining_sections)}"
    return opt_core


def opt_to_standard_cxsmiles(cxsmiles_opt: str) -> str:
    """Convert optimized ``<r>LABEL</r>`` CXSMILES to ``|$...$|`` form.

    The conversion preserves the input atom order.  Temporary atom-map numbers
    are used only to discover which parsed atom each ``<r>`` tag created, so
    labels are written into the standard ``$`` section by atom index instead of
    by detached occurrence order.
    """

    core, extension = split_cxsmiles(cxsmiles_opt)
    marker_base = 9000
    marker_to_label: dict[int, str] = {}

    def make_marker(label: str) -> str:
        marker = marker_base + len(marker_to_label)
        marker_to_label[marker] = label.strip()
        return f"[*:{marker}]"

    def mark_rgroup(match: re.Match[str]) -> str:
        return make_marker(match.group(1))

    marked_core = _RGROUP_TAG_RE.sub(mark_rgroup, core)
    marked_core = re.sub(r"\[Ar\]", lambda _match: make_marker("Ar"), marked_core)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.strictCXSMILES = False
    mol = Chem.MolFromSmiles(marked_core, params)
    if mol is None:
        raise ValueError(f"SMILES core does not parse: {cxsmiles_opt}")

    labels = [""] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        marker = atom.GetAtomMapNum()
        if marker in marker_to_label:
            labels[atom.GetIdx()] = marker_to_label[marker]

    standard_core = _RGROUP_TAG_RE.sub("*", core)
    standard_core = re.sub(r"\[Ar\]", "*", standard_core)
    sections = [
        section for section in split_extension_sections(extension) if not section.startswith("$")
    ]
    if any(labels):
        sections.insert(0, "$" + ";".join(labels) + "$")
    if sections:
        return f"{standard_core} |{','.join(sections)}|"
    return standard_core


_RGROUP_TAG_RE = re.compile(r"<r>([^<]+)</r>")


def apply_fragment_notation(
    cxsmiles_opt: str,
    rgroup_labels: list[str],
    rng: random.Random,
) -> str:
    """Represent some R-groups as disconnected `.`-separated prefix fragments.

    With the given labels, randomly selects 1-3 to convert from inline
    ``<r>LABEL</r>`` wildcards to prefix fragments ``C<r>LABEL</r>.`` prepended
    before the main SMILES.  Single-R fragment cases are allowed because they
    occur in eval-style CXSMILES.
    """

    if not rgroup_labels:
        return cxsmiles_opt

    max_fragments = 1 if len(rgroup_labels) == 1 else min(3, len(rgroup_labels) - 1)
    n_fragments = rng.randint(1, max_fragments)
    fragment_labels = set(rng.sample(rgroup_labels, n_fragments))

    # Split opt cxsmiles into core and extension (if any)
    if "|" in cxsmiles_opt:
        core_part, ext_part = cxsmiles_opt.split("|", 1)
        ext_suffix = "|" + ext_part
    else:
        core_part = cxsmiles_opt
        ext_suffix = ""

    # Remove selected labels from inline, leave bare wildcard *
    def _replace_if_fragment(match: re.Match[str]) -> str:
        label = match.group(1)
        if label in fragment_labels:
            return "*"
        return match.group(0)

    modified_core = _RGROUP_TAG_RE.sub(_replace_if_fragment, core_part)

    # Build prefix fragments
    fragments = [f"C<r>{label}</r>" for label in rgroup_labels if label in fragment_labels]
    prefix = ".".join(fragments) + "."

    return prefix + modified_core + ext_suffix


def decorate_scaffold(
    scaffold_smiles: str,
    *,
    config: DecorationConfig = DecorationConfig(),
    rng: random.Random | None = None,
) -> DecoratedMarkush | None:
    """Decorate one scaffold with R-group, Sg, and m CXSMILES annotations."""

    rng = random.Random() if rng is None else rng
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        return None

    # P2 Gap #2: ~7% probability of 0-R-group (Sg/m-only) structures
    zero_rgroup = rng.random() < config.zero_rgroup_probability

    if zero_rgroup:
        decorated_mol = mol
        rgroup_map: dict[int, str] = {}
        rgroup_atom_indices: list[int] = []
    else:
        added = add_rgroups_to_molecule(mol, config, rng)
        if added is None:
            return None
        decorated_mol, rgroup_map = added
        rgroup_atom_indices = list(rgroup_map)

    sg_annotations = add_sg_annotations(decorated_mol, rgroup_atom_indices, config, rng)
    m_annotations = add_m_annotations(decorated_mol, rgroup_atom_indices, config, rng)

    # For 0-R-group structures, require at least one Sg or m annotation
    if zero_rgroup and not sg_annotations and not m_annotations:
        return None

    try:
        core_smiles = Chem.MolToSmiles(decorated_mol, canonical=False, isomericSmiles=True)
    except Exception:
        return None
    atom_output_order = _smiles_atom_output_order(decorated_mol)
    cxsmiles = assemble_cxsmiles(
        core_smiles,
        atom_output_order,
        rgroup_map,
        sg_annotations,
        m_annotations,
    )
    validation = validate_cxsmiles(cxsmiles)
    if config.strict_validation and not validation.valid:
        return None

    cxsmiles_opt = standard_to_opt_cxsmiles(cxsmiles)

    # P1 Gap #4: ~15% probability of disconnected fragment notation
    rgroup_labels = list(rgroup_map.values())
    if rgroup_labels and rng.random() < config.fragment_probability:
        cxsmiles_opt = apply_fragment_notation(cxsmiles_opt, rgroup_labels, rng)

    return DecoratedMarkush(
        scaffold_smiles=scaffold_smiles,
        core_smiles=core_smiles,
        cxsmiles=cxsmiles,
        cxsmiles_opt=cxsmiles_opt,
        rgroup_labels=rgroup_labels,
        sg_annotations=sg_annotations,
        m_annotations=m_annotations,
        validation=validation,
    )


def generate_decorated_samples(
    scaffolds: list[str],
    *,
    num_samples: int,
    seed: int = 42,
    config: DecorationConfig = DecorationConfig(),
    max_attempts_per_sample: int = 25,
    target_profile: dict[str, Any] | None = None,
    profile_histogram: str = "feature_combo",
    progress_interval: int = 0,
) -> tuple[list[DecoratedMarkush], dict[str, int]]:
    """Sample scaffolds with replacement and return valid decorated seeds."""

    from glyph.markush.data.distribution import profile_key_for_record, target_counts_from_profile

    if not scaffolds:
        raise ValueError("scaffolds cannot be empty")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if max_attempts_per_sample <= 0:
        raise ValueError("max_attempts_per_sample must be positive")
    if progress_interval < 0:
        raise ValueError("progress_interval must be non-negative")

    rng = random.Random(seed)
    samples: list[DecoratedMarkush] = []
    target_counts = (
        target_counts_from_profile(target_profile, num_samples, histogram=profile_histogram)
        if target_profile
        else {}
    )
    accepted_profile_counts: Counter[str] = Counter()
    stats = {
        "requested": num_samples,
        "attempted": 0,
        "generated": 0,
        "failed": 0,
        "profile_rejected": 0,
        "profile_fallback": 0,
    }

    max_total_attempts = num_samples * max_attempts_per_sample
    while len(samples) < num_samples and stats["attempted"] < max_total_attempts:
        made = None
        fallback = None
        for _ in range(max_attempts_per_sample):
            if stats["attempted"] >= max_total_attempts:
                break
            stats["attempted"] += 1
            scaffold = rng.choice(scaffolds)
            made = decorate_scaffold(scaffold, config=config, rng=rng)
            if made is None:
                stats["failed"] += 1
                continue
            if not target_counts:
                break
            key = profile_key_for_record(made.to_record("candidate"), profile_histogram)
            if accepted_profile_counts[key] >= target_counts.get(key, 0):
                fallback = made
                made = None
                stats["profile_rejected"] += 1
                continue
            accepted_profile_counts[key] += 1
            break
        if made is None:
            made = fallback
            if made is None:
                continue
            if target_counts:
                key = profile_key_for_record(made.to_record("candidate"), profile_histogram)
                accepted_profile_counts[key] += 1
                stats["profile_fallback"] += 1
        samples.append(made)
        if progress_interval and (
            len(samples) == num_samples or len(samples) % progress_interval == 0
        ):
            LOGGER.info(
                "Decorated %d/%d seeds after %d attempts "
                "(failed=%d, profile_rejected=%d, profile_fallback=%d)",
                len(samples),
                num_samples,
                stats["attempted"],
                stats["failed"],
                stats["profile_rejected"],
                stats["profile_fallback"],
            )

    stats["generated"] = len(samples)
    if target_counts:
        stats["profile_target_bins"] = len(target_counts)
        stats["profile_filled_bins"] = sum(
            1 for key, target in target_counts.items() if accepted_profile_counts[key] >= target
        )
    return samples, stats


def build_training_record(
    sample_id: str,
    image_relpath: str,
    cxsmiles_opt: str,
    *,
    description: str = "",
    stable: str = "",
) -> dict[str, Any]:
    """Build the JSONL shape expected by ``glyph.markush.train.sft.MarkushDataset``."""

    instruction = "Extract the Markush structure from this patent image."
    target_instruction = (
        "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
        "the <cxsmi> value must use cxsmiles_opt."
    )
    if description:
        user_content = f"<image>\nPatent text: {description}\n{instruction}\n{target_instruction}"
    else:
        user_content = f"<image>\n{instruction}\n{target_instruction}"

    stable = (stable or "").strip()
    stable_match = re.search(r"<stable>(.*?)</stable>", stable, flags=re.IGNORECASE | re.DOTALL)
    if stable_match:
        stable_section = f"<stable>{stable_match.group(1).strip()}</stable>"
    elif stable.lower().startswith("<markush"):
        stable_section = "<stable></stable>"
    elif stable.lower().startswith("<stable"):
        stable_section = stable
    elif stable:
        stable_section = f"<stable>{stable}</stable>"
    else:
        stable_section = ""
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": f"<markush><cxsmi>{cxsmiles_opt}</cxsmi>{stable_section}</markush>",
            },
        ],
        "images": [image_relpath],
    }


def load_render_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a renderer manifest from JSON list or JSONL format."""

    path = Path(path)
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("render manifest JSON must contain a list of records")
    return data


def image_relpath_for_training(
    image_path: str | Path,
    *,
    data_root: str | Path | None,
    manifest_dir: str | Path | None = None,
) -> str:
    """Convert a rendered page path to the JSONL image path used by SFT.

    ``MarkushDataset`` resolves ``images`` entries relative to the configured
    image root. When the rendered image lives under the caller's data directory,
    this helper stores the relative path from that directory. If it cannot prove
    that relationship, it falls back to the path as provided.
    """

    path = Path(image_path)
    if not path.is_absolute() and manifest_dir is not None:
        candidate = Path(manifest_dir) / path
        if candidate.exists():
            path = candidate

    if data_root is not None:
        try:
            rel = path.resolve().relative_to(Path(data_root).resolve())
            return rel.as_posix()
        except (OSError, ValueError):
            pass

    return str(image_path).replace(os.sep, "/")


def training_record_from_render_manifest_entry(
    entry: dict[str, Any],
    *,
    image_relpath: str,
    prompt_text_dropout: float = 0.0,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Convert one render manifest row into an SFT-ready training record."""

    if not (0.0 <= prompt_text_dropout <= 1.0):
        raise ValueError("prompt_text_dropout must be in [0, 1]")

    cxsmiles_opt = entry.get("cxsmiles_opt")
    if not cxsmiles_opt and entry.get("cxsmiles_input"):
        cxsmiles_opt = standard_to_opt_cxsmiles(entry["cxsmiles_input"])
    if not cxsmiles_opt:
        raise ValueError(f"manifest entry {entry.get('id', '<unknown>')} has no cxsmiles_opt")

    rng = random.Random() if rng is None else rng
    description = entry.get("description", "") or ""
    if description and rng.random() < prompt_text_dropout:
        description = ""

    return build_training_record(
        str(entry.get("id", "")),
        image_relpath,
        cxsmiles_opt,
        description=description,
        stable=entry.get("annotation", "") or "",
    )


def prepare_training_jsonl_from_manifest(
    manifest_path: str | Path,
    output_jsonl: str | Path,
    *,
    data_root: str | Path | None = "data",
    prompt_text_dropout: float = 0.0,
    seed: int = 42,
) -> dict[str, int]:
    """Convert renderer manifest output into SFT JSONL."""

    manifest_path = Path(manifest_path)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    entries = load_render_manifest(manifest_path)
    rng = random.Random(seed)
    stats = {
        "total": len(entries),
        "written": 0,
        "failed": 0,
        "dropped_prompt_text": 0,
    }

    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for entry in entries:
            page_path = entry.get("page_path") or entry.get("image_path")
            if not page_path:
                stats["failed"] += 1
                continue

            image_rel = image_relpath_for_training(
                page_path,
                data_root=data_root,
                manifest_dir=manifest_path.parent,
            )
            before_description = bool(entry.get("description"))
            try:
                record = training_record_from_render_manifest_entry(
                    entry,
                    image_relpath=image_rel,
                    prompt_text_dropout=prompt_text_dropout,
                    rng=rng,
                )
            except ValueError:
                stats["failed"] += 1
                continue

            after_description = "Patent text:" in record["messages"][0]["content"]
            if before_description and not after_description:
                stats["dropped_prompt_text"] += 1
            handle.write(json.dumps(record) + "\n")
            stats["written"] += 1

    return stats


def write_seed_outputs(
    samples: list[DecoratedMarkush],
    *,
    output_jsonl: str | Path,
    output_csv: str | Path | None = None,
    id_prefix: str = "mksh",
) -> None:
    """Write generated seeds as JSONL and optionally as renderer-compatible CSV."""

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = [sample.to_record(f"{id_prefix}_{idx:06d}") for idx, sample in enumerate(samples)]
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "scaffold_smiles",
                    "cxsmiles",
                    "cxsmiles_opt",
                    "rgroup_count",
                    "has_sg",
                    "has_m",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "id": record["id"],
                        "scaffold_smiles": record["scaffold_smiles"],
                        "cxsmiles": record["cxsmiles"],
                        "cxsmiles_opt": record["cxsmiles_opt"],
                        "rgroup_count": record["features"]["rgroup_count"],
                        "has_sg": record["features"]["has_sg"],
                        "has_m": record["features"]["has_m"],
                    }
                )


def write_failure_mode_seed_outputs(
    focused_samples: list[FocusedDecoratedMarkush],
    *,
    output_jsonl: str | Path,
    output_csv: str | Path | None = None,
    id_prefix: str = "failure",
) -> None:
    """Write focused generated seeds with failure-mode metadata."""

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    focus_counts: Counter[str] = Counter()
    for focused in focused_samples:
        idx = focus_counts[focused.focus]
        focus_counts[focused.focus] += 1
        record = focused.sample.to_record(f"{id_prefix}_{focused.focus}_{idx:06d}")
        record["failure_mode_focus"] = focused.focus
        record["failure_mode_description"] = FAILURE_MODE_FOCUS_SPECS[focused.focus].description
        records.append(record)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "failure_mode_focus",
                    "failure_mode_description",
                    "scaffold_smiles",
                    "cxsmiles",
                    "cxsmiles_opt",
                    "rgroup_count",
                    "has_sg",
                    "has_m",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "id": record["id"],
                        "failure_mode_focus": record["failure_mode_focus"],
                        "failure_mode_description": record["failure_mode_description"],
                        "scaffold_smiles": record["scaffold_smiles"],
                        "cxsmiles": record["cxsmiles"],
                        "cxsmiles_opt": record["cxsmiles_opt"],
                        "rgroup_count": record["features"]["rgroup_count"],
                        "has_sg": record["features"]["has_sg"],
                        "has_m": record["features"]["has_m"],
                    }
                )


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-count", type=int, default=200_000)
    parser.add_argument("--sample-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pains", action="store_true", help="Disable PAINS filtering")


def _add_decoration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decorate-attempts-per-sample",
        type=int,
        default=25,
        help="Candidate decoration attempts per requested sample before falling back",
    )
    parser.add_argument(
        "--decoration-profile",
        choices=sorted(DECORATION_PROFILES),
        default="hybrid",
        help="Named generator distribution profile",
    )
    parser.add_argument("--target-profile", type=str, default=None)
    parser.add_argument("--profile-histogram", type=str, default="feature_combo")
    parser.add_argument("--min-rgroups", type=int, default=None)
    parser.add_argument("--max-rgroups", type=int, default=None)
    parser.add_argument("--high-rgroup-probability", type=float, default=None)
    parser.add_argument("--high-rgroup-min", type=int, default=None)
    parser.add_argument("--high-rgroup-max", type=int, default=None)
    parser.add_argument("--label-style", choices=["mixed", "R_only"], default=None)
    parser.add_argument("--sg-probability", type=float, default=None)
    parser.add_argument("--m-probability", type=float, default=None)
    parser.add_argument("--zero-rgroup-probability", type=float, default=None)
    parser.add_argument("--fragment-probability", type=float, default=None)
    parser.add_argument("--id-prefix", type=str, default="mksh")
    parser.add_argument("--stats-json", type=str, default=None)


def _cmd_build_scaffold_pool(args: argparse.Namespace) -> None:
    cfg = ScaffoldFilterConfig(exclude_pains=not args.no_pains)
    stats = build_scaffold_pool(
        args.cid_smiles,
        args.output,
        target_count=args.target_count,
        sample_rate=args.sample_rate,
        seed=args.seed,
        filter_config=cfg,
    )
    print(json.dumps(stats, indent=2))


def _cmd_decorate(args: argparse.Namespace) -> None:
    from glyph.markush.data.distribution import load_profile

    scaffolds = load_scaffold_pool(args.scaffold_pool)
    cfg = decoration_config_for_profile(
        args.decoration_profile,
        min_rgroups=args.min_rgroups,
        max_rgroups=args.max_rgroups,
        high_rgroup_probability=args.high_rgroup_probability,
        high_rgroup_min=args.high_rgroup_min,
        high_rgroup_max=args.high_rgroup_max,
        label_style=args.label_style,
        sg_probability=args.sg_probability,
        m_probability=args.m_probability,
        zero_rgroup_probability=args.zero_rgroup_probability,
        fragment_probability=args.fragment_probability,
    )
    target_profile = load_profile(Path(args.target_profile)) if args.target_profile else None
    samples, stats = generate_decorated_samples(
        scaffolds,
        num_samples=args.num_samples,
        seed=args.seed,
        config=cfg,
        target_profile=target_profile,
        profile_histogram=args.profile_histogram,
        max_attempts_per_sample=args.decorate_attempts_per_sample,
    )
    write_seed_outputs(
        samples,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        id_prefix=args.id_prefix,
    )

    feature_stats = {
        "with_sg": sum(1 for sample in samples if sample.has_sg),
        "with_m": sum(1 for sample in samples if sample.has_m),
        "with_fragment": sum(
            1 for sample in samples if "." in split_cxsmiles(sample.cxsmiles_opt)[0]
        ),
        "zero_rgroup": sum(1 for sample in samples if not sample.rgroup_labels),
        "rgroup_only": sum(1 for sample in samples if not sample.has_sg and not sample.has_m),
    }
    stats.update(feature_stats)
    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


def _cmd_decorate_failure_modes(args: argparse.Namespace) -> None:
    scaffolds = load_scaffold_pool(args.scaffold_pool)
    focused_samples, stats = generate_failure_mode_decorated_samples(
        scaffolds,
        samples_per_focus=args.samples_per_focus,
        focuses=args.focus,
        seed=args.seed,
        max_attempts_per_sample=args.decorate_attempts_per_sample,
        progress_interval=args.decorate_progress_interval,
    )
    write_failure_mode_seed_outputs(
        focused_samples,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        id_prefix=args.id_prefix,
    )

    stats["output_jsonl"] = str(args.output_jsonl)
    stats["output_csv"] = str(args.output_csv) if args.output_csv else None
    stats["focus_counts"] = dict(Counter(sample.focus for sample in focused_samples))
    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


def _cmd_validate(args: argparse.Namespace) -> None:
    total = 0
    invalid = 0
    with open(args.input_jsonl, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            cxsmiles = record.get(args.field, "")
            report = validate_cxsmiles(cxsmiles)
            if not report.valid:
                invalid += 1
                print(f"{args.input_jsonl}:{line_no}: {report.errors}")
    print(json.dumps({"total": total, "invalid": invalid, "valid": total - invalid}, indent=2))
    if invalid:
        raise SystemExit(1)


def _cmd_prepare_training_jsonl(args: argparse.Namespace) -> None:
    stats = prepare_training_jsonl_from_manifest(
        args.manifest,
        args.output_jsonl,
        data_root=data_root(args.data_root),
        prompt_text_dropout=args.prompt_text_dropout,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))
    if stats["failed"]:
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Markush data curation and seed generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pool = subparsers.add_parser("build-scaffold-pool", help="Filter PubChem CID-SMILES")
    pool.add_argument("--cid-smiles", type=str, required=True, help="CID-SMILES or CID-SMILES.gz")
    pool.add_argument("--output", type=str, required=True, help="Output scaffold_pool.txt")
    _add_filter_args(pool)
    pool.set_defaults(func=_cmd_build_scaffold_pool)

    decorate = subparsers.add_parser("decorate", help="Generate Markush CXSMILES seeds")
    decorate.add_argument("--scaffold-pool", type=str, required=True)
    decorate.add_argument("--output-jsonl", type=str, required=True)
    decorate.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV with a 'cxsmiles' column for the renderer",
    )
    _add_decoration_args(decorate)
    decorate.set_defaults(func=_cmd_decorate)

    failure = subparsers.add_parser(
        "decorate-failure-modes",
        help="Generate renderer-ready seeds balanced across eval failure modes",
    )
    failure.add_argument("--scaffold-pool", type=str, required=True)
    failure.add_argument("--output-jsonl", type=str, required=True)
    failure.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV with cxsmiles/cxsmiles_opt columns for the renderer",
    )
    failure.add_argument("--samples-per-focus", type=int, required=True)
    failure.add_argument(
        "--focus",
        action="append",
        choices=FAILURE_MODE_FOCUS_CHOICES,
        default=None,
        help="Failure-mode focus to generate. Repeatable; defaults to all.",
    )
    failure.add_argument("--seed", type=int, default=42)
    failure.add_argument(
        "--decorate-attempts-per-sample",
        type=int,
        default=100,
        help="Candidate decoration attempts per requested sample in each focus bucket",
    )
    failure.add_argument("--decorate-progress-interval", type=int, default=0)
    failure.add_argument("--id-prefix", type=str, default="failure")
    failure.add_argument("--stats-json", type=str, default=None)
    failure.set_defaults(func=_cmd_decorate_failure_modes)

    validate = subparsers.add_parser("validate", help="Validate generated seed JSONL")
    validate.add_argument("--input-jsonl", type=str, required=True)
    validate.add_argument("--field", type=str, default="cxsmiles")
    validate.set_defaults(func=_cmd_validate)

    prepare = subparsers.add_parser(
        "prepare-training-jsonl",
        help="Convert render manifest output into SFT training JSONL",
    )
    prepare.add_argument("--manifest", type=str, required=True, help="manifest.json or JSONL")
    prepare.add_argument("--output-jsonl", type=str, required=True)
    prepare.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Data directory used to make rendered image paths relative (default: GLYPH_DATA_ROOT or ./data)",
    )
    prepare.add_argument(
        "--prompt-text-dropout",
        type=float,
        default=0.0,
        help="Static probability of dropping patent text from the user prompt",
    )
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=_cmd_prepare_training_jsonl)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
