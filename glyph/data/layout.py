"""Filesystem layout and provenance records for the release data pipeline.

This module is the single source of truth for *where* prepared datasets and
benchmarks live under ``$GLYPH_DATA_ROOT`` and *how* each artifact is
attributed. The orchestration in :mod:`glyph.data.ocsr` (and, later, the
Markush unit) reads these constants so the on-disk layout and the provenance
sidecars stay consistent with the documented layout and provenance.

Layout::

    $GLYPH_DATA_ROOT/
      datasets/ocsr/{pubchem_1m, uspto_680k, stereo_200k_white, adjacent_ring}
      datasets/markush/{mg1_synthetic, mg2_structure_only, molparser, ordinary_5pct,
                        adjacent_ring, prepared/}
      benchmarks/ocsr/uspto_ocsr_5719_complete.csv
      benchmarks/markush/{ip5m, m2s, uspto_markush}

Every materialized dataset directory carries a ``provenance.json`` sidecar so a
downstream reader can attribute the source without consulting this code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from glyph.paths import data_path

# --- Directory constants -----------------------------------------------------

DATASETS_OCSR = ("datasets", "ocsr")
BENCHMARKS_OCSR = ("benchmarks", "ocsr")

PUBCHEM_1M = "pubchem_1m"
USPTO_680K = "uspto_680k"
STEREO_200K_WHITE = "stereo_200k_white"
ADJACENT_RING = "adjacent_ring"

OCSR_DATASET_NAMES = (
    PUBCHEM_1M,
    USPTO_680K,
    STEREO_200K_WHITE,
    ADJACENT_RING,
)

#: The complete USPTO OCSR benchmark label sidecar filename.
USPTO_5719_BENCHMARK_FILENAME = "uspto_ocsr_5719_complete.csv"

PROVENANCE_FILENAME = "provenance.json"

# --- Markush directory constants -------------------------------------------

DATASETS_MARKUSH = ("datasets", "markush")
BENCHMARKS_MARKUSH = ("benchmarks", "markush")

MG1_SYNTHETIC = "mg1_synthetic"
MG2_STRUCTURE_ONLY = "mg2_structure_only"
MOLPARSER = "molparser"
ORDINARY_5PCT = "ordinary_5pct"
MARKUSH_PREPARED = "prepared"

#: The four training sources that feed the headline multitask mix, each keyed to
#: the :data:`glyph.markush.data.importers.public_datasets.DATASET_SPECS` name it
#: is imported from. The ``adjacent_ring`` slot shown in the §3
#: ``datasets/markush/*`` layout is intentionally absent here -- it maps to no
#: importer spec or recipe entry, and is distinct from the OCSR ``adjacent_ring``
#: dataset (:data:`ADJACENT_RING`, an adjacent-ring-stereo PubChem-1M subset)
#: (see :data:`MARKUSH_DATASET_TO_SPEC` and ``glyph/data/README.md`` for the
#: open provenance flag).
MARKUSH_DATASET_NAMES = (
    MG1_SYNTHETIC,
    MG2_STRUCTURE_ONLY,
    MOLPARSER,
    ORDINARY_5PCT,
)

#: Map each §3 ``datasets/markush/*`` directory to the importer spec name in
#: ``public_datasets.DATASET_SPECS``. The §3 directory names and the recipe's
#: importer names diverge for two entries (recorded here so nothing guesses):
#:   * ``mg2_structure_only``  <- importer ``uspto-mol-m`` (source ``uspto_mol_m``);
#:     the USPTO-MOL-M rows train in the ``structure_only_markush`` example mode.
#:   * ``ordinary_5pct``       <- importer ``molscribe-ocsr`` (source ``molscribe``);
#:     the "5% ordinary" no-extension Markush conversion.
MARKUSH_DATASET_TO_SPEC: dict[str, str] = {
    MG1_SYNTHETIC: "mg1-synthetic",
    MG2_STRUCTURE_ONLY: "uspto-mol-m",
    MOLPARSER: "molparser-markush",
    ORDINARY_5PCT: "molscribe-ocsr",
}

#: The three DS4SD Markush eval benchmarks, keyed to the
#: importer spec they are prepared from.
MARKUSH_BENCHMARK_NAMES = ("ip5m", "m2s", "uspto_markush")
MARKUSH_BENCHMARK_TO_SPEC: dict[str, str] = {
    "ip5m": "ip5m",
    "m2s": "m2s",
    "uspto_markush": "uspto-markush",
}

#: The canonical §3 output location for the assembled 5% multitask mix. The
#: training configs point at this directory.
MARKUSH_MULTITASK_MIX_DIRNAME = "multitask_5pct_seed74"
MARKUSH_MULTITASK_MIX_FILENAME = "train.jsonl"

# --- Pinned upstream revisions ----------------------------------------------
# Recorded so a re-fetch is reproducible. Sizes are the on-disk download size
# of the referenced upstream file(s), captured from the Hugging Face repo.

MOLNEXTR_REPO = "CYF200127/MolNexTR"
MOLNEXTR_REVISION = "9ac2da6f"
STEREO_REPO = "Robert-zwr/Stereo-200k"
STEREO_REVISION = "f6cdec49"
USPTO_BENCHMARK_REPO = "hheiden/USPTO_OCSR_benchmark"
USPTO_BENCHMARK_REVISION = "32379850"

# Markush upstream repos + pinned revisions.
#
# MarkushGrapher-2-Datasets is pinned to the authoritative IP5-M eval contract
# revision (n=878, fingerprint 15733712f9c574af). The same repo hosts the uspto-mol-m / m2s / uspto-markush
# configs, so the one pin covers all four MG2 configs.
#
# MG1 and MolParser did not record a Hugging Face dataset SHA in the original
# training manifests. They are pinned here to the public repositories' current
# ``main`` commits as deterministic fallback revisions. MolScribe did record its
# authoritative snapshot in the original import manifest.
MG1_REPO = "docling-project/MarkushGrapher-Datasets"
MG1_REVISION = "a6cf981dd100198ce87e8099402f0645b44080ed"
MG2_REPO = "docling-project/MarkushGrapher-2-Datasets"
MG2_REVISION = "e90868cdd3b62f1fe175d9ede9e0c65485c4a96c"
MG2_FINGERPRINT = "15733712f9c574af"
MOLPARSER_REPO = "UniParser/MolParser-7M"
MOLPARSER_REVISION = "f37e164f179c7cd8470fc879422c29af2c47874c"
MOLSCRIBE_REPO = "yujieq/MolScribe"
MOLSCRIBE_REVISION = "a0189776b7415b82795c7ee81eed311bf5c8724b"

#: Per-importer pinned revision, keyed by ``DATASET_SPECS`` spec name. The
#: source fetchers thread this into the importer so a re-fetch is reproducible.
MARKUSH_SPEC_REVISIONS: dict[str, str | None] = {
    "mg1-synthetic": MG1_REVISION,
    "uspto-mol-m": MG2_REVISION,
    "ip5m": MG2_REVISION,
    "m2s": MG2_REVISION,
    "uspto-markush": MG2_REVISION,
    "molparser-markush": MOLPARSER_REVISION,
    "molscribe-ocsr": MOLSCRIBE_REVISION,
}


# --- Provenance records ------------------------------------------------------
# AUTHORITATIVE provenance records.
# Do NOT relabel external data as "our upload"/"our render".

OCSR_PROVENANCE: dict[str, dict[str, Any]] = {
    PUBCHEM_1M: {
        "dataset": PUBCHEM_1M,
        "title": "PubChem-1M (MolNexTR public source)",
        "role": "external_public",
        "this_work": False,
        "attribution": (
            "Public MolNexTR training source. Redistributed by MolNexTR "
            "(Chen et al.); referenced here by manifest, not re-hosted by this work."
        ),
        "source_repo": MOLNEXTR_REPO,
        "source_repo_type": "dataset",
        "source_files": ["train_pubchem.csv"],
        "pinned_revision": MOLNEXTR_REVISION,
        "expected_rows": 1_000_000,
        "expected_download": "~210 MB (train_pubchem.csv)",
        "schema": "SMILES-only CSV (pubchem_cid, InChI, SMILES, num_atoms).",
        "rendering": "Images rendered on the fly with vendored EPAM Indigo at train time.",
    },
    USPTO_680K: {
        "dataset": USPTO_680K,
        "title": "USPTO-680K (MolNexTR public source)",
        "role": "external_public",
        "this_work": False,
        "attribution": (
            "Public MolNexTR USPTO auxiliary training source. Redistributed by "
            "MolNexTR (Chen et al.); referenced here by manifest, not re-hosted by this work."
        ),
        "source_repo": MOLNEXTR_REPO,
        "source_repo_type": "dataset",
        "source_files": ["train_uspto.zip"],
        "pinned_revision": MOLNEXTR_REVISION,
        "expected_rows": 680_220,
        "expected_download": "~2.1 GB (train_uspto.zip, images + CSV)",
        "schema": "CSV with file_path,SMILES(,node_coords,edges) + on-disk images.",
        "rendering": "Pre-rendered images shipped in the archive.",
    },
    STEREO_200K_WHITE: {
        "dataset": STEREO_200K_WHITE,
        "title": "Stereo-200K (white-cropped)",
        "role": "external_preprocessed",
        "this_work": False,
        "attribution": (
            "EXTERNAL dataset. Base images are the public Hugging Face dataset "
            "Robert-zwr/Stereo-200k; cite MolSight (zhang2026molsight). This work did "
            "NOT render or upload these images. The only contribution here is the "
            "white-background + tight-crop preprocessing (background_to_white + "
            "smart_tight_crop); SMILES targets are byte-identical to the source. "
            "Do NOT present stereo_200k_white as an original upload."
        ),
        "citation": "zhang2026molsight",
        "source_repo": STEREO_REPO,
        "source_repo_type": "dataset",
        "source_files": ["stereo.zip"],
        "pinned_revision": STEREO_REVISION,
        "expected_rows": 200_000,
        "expected_download": "~4.3 GB (stereo.zip)",
        "schema": "CSV with SMILES,file_path + PNG images.",
        "preprocessing": (
            "glyph.data.ocsr_white_crop.white_crop_image "
            "(corner-median background -> white, then largest-cluster tight crop, 3% pad)."
        ),
    },
    ADJACENT_RING: {
        "dataset": ADJACENT_RING,
        "title": "adjacent_ring (adjacent-ring-stereo subset of PubChem-1M)",
        "role": "derived_this_work",
        "this_work": True,
        "attribution": (
            "This work. A pure-logic mined SUBSET of PubChem-1M (see the pubchem_1m "
            "component) -- rows kept iff they have >=2 chiral centers with at least two "
            "adjacent in the same ring and SMILES length in [min_len, max_len]. It is NOT "
            "a standalone external dataset; it references pubchem_1m by construction and "
            "overlaps it 100%. This is the only 'This work' OCSR training data."
        ),
        "derived_from": PUBCHEM_1M,
        "generator": "glyph.data.ocsr_adjacent_ring.mine_adjacent_ring",
        "expected_rows": 64_752,
        "expected_download": "0 (derived locally from pubchem_1m)",
        "mining_rule": (
            "has_adjacent_ring_stereo AND min_len<=len(SMILES)<=max_len AND RDKit-parseable; "
            "defaults min_len=50, max_len=180 -> 64,752 of 1,000,000 kept (6.49%)."
        ),
    },
}

BENCHMARK_5719_PROVENANCE: dict[str, Any] = {
    "artifact": USPTO_5719_BENCHMARK_FILENAME,
    "title": "USPTO OCSR 5,719-row complete benchmark label sidecar",
    "role": "benchmark",
    "this_work": False,
    "attribution": (
        "Composed by glyph.ocsr.benchmark from the public benchmark "
        "hheiden/USPTO_OCSR_benchmark (5,704 labelled rows) plus 13 null labels "
        "recovered from the MolScribe (Qian et al., 2023) real/USPTO.csv release; "
        "2 rows remain unrecovered. 5704 + 13 + 2 = 5719. Per-label provenance is "
        "written to the CSV's source_of_label column and the .provenance.json sidecar."
    ),
    "benchmark_repo": USPTO_BENCHMARK_REPO,
    "benchmark_revision": USPTO_BENCHMARK_REVISION,
    "molscribe_source": "MolScribe real.zip -> real/USPTO.csv (redistributed via CYF200127/MolNexTR).",
    "expected_rows": 5719,
}


# --- Markush provenance records ----------------------------------------------
# AUTHORITATIVE. Every training
# source below is EXTERNAL public data referenced by manifest at a pinned
# revision -- NOT re-hosted or uploaded by this work. Do NOT relabel as an
# original upload. The only "this work" contributions are the local
# canonicalization/conversion transforms (recorded per-record), not the images.

MARKUSH_PROVENANCE: dict[str, dict[str, Any]] = {
    MG1_SYNTHETIC: {
        "dataset": MG1_SYNTHETIC,
        "title": "MG1 synthetic Markush (MarkushGrapher training set)",
        "role": "external_public",
        "this_work": False,
        "attribution": (
            "EXTERNAL. Public MarkushGrapher synthetic training data, redistributed "
            "by docling-project/MarkushGrapher-Datasets (config "
            "'markushgrapher-synthetic-training'). Referenced here by manifest at a "
            "pinned revision; NOT re-hosted or generated by this work. The headline "
            "mix uses these PUBLIC MG1 rows, not the local Java/CDK synthetic generator."
        ),
        "source_repo": MG1_REPO,
        "source_repo_type": "dataset",
        "importer_spec": "mg1-synthetic",
        "pinned_revision": MG1_REVISION,
        "expected_download": "~25 GB (image tree; --allow-large gated)",
        "training_mode": "patent_image_markush",
        "expected_rows_in_base_mix": 152_620,
    },
    MG2_STRUCTURE_ONLY: {
        "dataset": MG2_STRUCTURE_ONLY,
        "title": "USPTO-MOL-M structure-only Markush (MarkushGrapher-2)",
        "role": "external_public",
        "this_work": False,
        "attribution": (
            "EXTERNAL. Public USPTO-MOL-M data from docling-project/"
            "MarkushGrapher-2-Datasets (config 'uspto-mol-m-54k'). Referenced by "
            "manifest at a pinned revision; NOT re-hosted by this work. The layout "
            "names this directory 'mg2_structure_only' (the importer spec is "
            "'uspto-mol-m'); rows train in the 'structure_only_markush' example mode."
        ),
        "source_repo": MG2_REPO,
        "source_repo_type": "dataset",
        "importer_spec": "uspto-mol-m",
        "pinned_revision": MG2_REVISION,
        "expected_download": "~2.9 GB (image tree; --allow-large gated)",
        "training_mode": "structure_only_markush",
        "expected_rows_in_base_mix": 52_524,
    },
    MOLPARSER: {
        "dataset": MOLPARSER,
        "title": "MolParser-7M (real SFT subset)",
        "role": "external_public",
        "this_work": False,
        "attribution": (
            "EXTERNAL. Public MolParser-7M data from UniParser/MolParser-7M (config "
            "'sft_real'). Cite MolParser (Fan et al.). Referenced by manifest at a "
            "pinned revision; NOT re-hosted by this work. Converted locally to the "
            "unified cxsmiles_opt Markush target (molparser_converted_markush mode)."
        ),
        "source_repo": MOLPARSER_REPO,
        "source_repo_type": "dataset",
        "importer_spec": "molparser-markush",
        "pinned_revision": MOLPARSER_REVISION,
        "expected_download": "~2.0 GB (image tree; --allow-large gated)",
        "training_mode": "molparser_converted_markush",
        "expected_rows_in_base_mix": 44_856,
    },
    ORDINARY_5PCT: {
        "dataset": ORDINARY_5PCT,
        "title": "MolScribe OCSR ordinary molecules (5% no-extension Markush)",
        "role": "external_preprocessed",
        "this_work": False,
        "attribution": (
            "EXTERNAL base images. MolScribe USPTO ordinary-molecule OCSR data from "
            "yujieq/MolScribe (uspto_mol.zip). Cite MolScribe (Qian et al., 2023). "
            "Referenced by manifest at a pinned revision; NOT re-hosted by this work. "
            "The 'ordinary molecule as no-extension Markush' target conversion "
            "(molecule_ocsr_as_markush_no_extension) is this work's transform; the "
            "images and SMILES are external. Held out from the ordinary-molecule OCSR "
            "benchmarks; an 80-row decontamination is applied at mix time."
        ),
        "citation": "qian2023molscribe",
        "source_repo": MOLSCRIBE_REPO,
        "source_repo_type": "model",
        "importer_spec": "molscribe-ocsr",
        "pinned_revision": MOLSCRIBE_REVISION,
        "expected_download": "~2.2 GB (uspto_mol.zip; --allow-large gated)",
        "training_mode": "molecule_ocsr_as_markush_no_extension",
        "expected_rows_in_mix": 13_158,
    },
}

#: Provenance sidecar for the assembled 5% multitask mix under ``prepared/``.
MARKUSH_MULTITASK_MIX_PROVENANCE: dict[str, Any] = {
    "artifact": MARKUSH_MULTITASK_MIX_DIRNAME,
    "title": "MarkushGlyph 5% multitask SFT mix (263,158 rows, seed 74)",
    "role": "derived_this_work",
    "this_work": True,
    "attribution": (
        "Assembled by this work from the external sources above. 250,000 base "
        "(152,620 MG1-synthetic + 52,524 USPTO-MOL-M + 44,856 MolParser; == 200K "
        "broad-canonical + 50K checkpoint-mined-hard) + 13,158 MolScribe-OCSR rows "
        "converted to no-extension Markush ('5% ordinary') = 263,158, seed 74, with "
        "80 decontamination rows removed. The underlying images are EXTERNAL public "
        "data (see per-source provenance); only the mixture composition and the "
        "target conversions are this work. Image references are rewritten to "
        "$GLYPH_DATA_ROOT-relative paths."
    ),
    "seed": 74,
    "expected_total_rows": 263_158,
    "expected_base_rows": 250_000,
    "expected_ordinary_rows": 13_158,
    "expected_decontamination_removed": 80,
    "ocsr_count_formula": "round(0.05 / (1 - 0.05) * 250000) == 13158",
    "expected_source_counts": {
        "mg1_synthetic": 152_620,
        "uspto_mol_m": 52_524,
        "molparser": 44_856,
        "molscribe": 13_158,
    },
    "hard_partition": (
        "The 50K checkpoint-mined-hard slice is checkpoint-dependent and NOT "
        "regenerable from scratch; it is consumed as a "
        "frozen cached input, not re-derived by this pipeline."
    ),
}

#: Per-benchmark provenance for ``benchmarks/markush/{ip5m,m2s,uspto_markush}``.
MARKUSH_BENCHMARK_PROVENANCE: dict[str, dict[str, Any]] = {
    "ip5m": {
        "benchmark": "ip5m",
        "title": "IP5-M Markush eval (pinned DS4SD contract)",
        "role": "benchmark",
        "this_work": False,
        "attribution": (
            "EXTERNAL DS4SD benchmark. docling-project/MarkushGrapher-2-Datasets "
            "config 'ip5-markush' split 'test'. The AUTHORITATIVE verified contract: "
            "n=878 at pinned revision, fingerprint 15733712f9c574af. Not re-hosted."
        ),
        "source_repo": MG2_REPO,
        "source_repo_type": "dataset",
        "importer_spec": "ip5m",
        "pinned_revision": MG2_REVISION,
        "fingerprint": MG2_FINGERPRINT,
        "expected_rows": 878,
    },
    "m2s": {
        "benchmark": "m2s",
        "title": "M2S Markush eval",
        "role": "benchmark",
        "this_work": False,
        "attribution": (
            "EXTERNAL DS4SD benchmark. docling-project/MarkushGrapher-2-Datasets "
            "config 'm2s' split 'test'. Not re-hosted by this work."
        ),
        "source_repo": MG2_REPO,
        "source_repo_type": "dataset",
        "importer_spec": "m2s",
        "pinned_revision": MG2_REVISION,
        "expected_rows": 103,
    },
    "uspto_markush": {
        "benchmark": "uspto_markush",
        "title": "USPTO-Markush eval",
        "role": "benchmark",
        "this_work": False,
        "attribution": (
            "EXTERNAL DS4SD benchmark. docling-project/MarkushGrapher-2-Datasets "
            "config 'uspto-markush' split 'test'. Not re-hosted by this work."
        ),
        "source_repo": MG2_REPO,
        "source_repo_type": "dataset",
        "importer_spec": "uspto-markush",
        "pinned_revision": MG2_REVISION,
        "expected_rows": 74,
    },
}


# --- Path helpers ------------------------------------------------------------


def ocsr_datasets_dir(*parts: str, root: str | Path | None = None) -> Path:
    """Return ``$GLYPH_DATA_ROOT/datasets/ocsr[/parts...]``."""

    return data_path(*DATASETS_OCSR, *parts, root=root)


def ocsr_dataset_dir(name: str, *, root: str | Path | None = None) -> Path:
    """Return the target directory for one OCSR dataset."""

    return ocsr_datasets_dir(name, root=root)


def ocsr_benchmarks_dir(*parts: str, root: str | Path | None = None) -> Path:
    """Return ``$GLYPH_DATA_ROOT/benchmarks/ocsr[/parts...]``."""

    return data_path(*BENCHMARKS_OCSR, *parts, root=root)


def uspto_5719_benchmark_path(*, root: str | Path | None = None) -> Path:
    """Return the target path for the complete 5,719-row USPTO benchmark CSV."""

    return ocsr_benchmarks_dir(USPTO_5719_BENCHMARK_FILENAME, root=root)


def markush_datasets_dir(*parts: str, root: str | Path | None = None) -> Path:
    """Return ``$GLYPH_DATA_ROOT/datasets/markush[/parts...]``."""

    return data_path(*DATASETS_MARKUSH, *parts, root=root)


def markush_dataset_dir(name: str, *, root: str | Path | None = None) -> Path:
    """Return the target directory for one Markush source dataset."""

    return markush_datasets_dir(name, root=root)


def markush_benchmarks_dir(*parts: str, root: str | Path | None = None) -> Path:
    """Return ``$GLYPH_DATA_ROOT/benchmarks/markush[/parts...]``."""

    return data_path(*BENCHMARKS_MARKUSH, *parts, root=root)


def markush_benchmark_dir(name: str, *, root: str | Path | None = None) -> Path:
    """Return the target directory for one Markush eval benchmark."""

    return markush_benchmarks_dir(name, root=root)


def markush_prepared_dir(*parts: str, root: str | Path | None = None) -> Path:
    """Return ``$GLYPH_DATA_ROOT/datasets/markush/prepared[/parts...]``."""

    return markush_datasets_dir(MARKUSH_PREPARED, *parts, root=root)


def markush_multitask_mix_dir(*, root: str | Path | None = None) -> Path:
    """Return the canonical §3 directory for the assembled 5% multitask mix."""

    return markush_prepared_dir(MARKUSH_MULTITASK_MIX_DIRNAME, root=root)


def markush_multitask_mix_path(*, root: str | Path | None = None) -> Path:
    """Return the canonical §3 ``train.jsonl`` path for the 5% multitask mix."""

    return markush_multitask_mix_dir(root=root) / MARKUSH_MULTITASK_MIX_FILENAME


def write_provenance(directory: Path, record: dict[str, Any]) -> Path:
    """Write a ``provenance.json`` sidecar into ``directory`` and return its path.

    A ``prepared_at`` UTC timestamp is added so a reader can tell when the
    artifact was last materialized. Existing sidecars are overwritten (the
    record is deterministic apart from the timestamp).
    """

    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("prepared_at", datetime.now(UTC).isoformat())
    path = directory / PROVENANCE_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --- Step reporting ----------------------------------------------------------

# Status vocabulary for a single prepared artifact.
STATUS_MATERIALIZED = "materialized"
STATUS_SKIPPED_EXISTS = "skipped_exists"
STATUS_PLANNED = "planned"  # --dry-run
STATUS_REFUSED = "refused"  # needs --allow-large or a missing dependency
STATUS_ERROR = "error"


@dataclass
class PrepareStep:
    """Outcome of preparing one dataset/benchmark artifact."""

    name: str
    status: str
    detail: str = ""
    path: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "path": str(self.path) if self.path is not None else None,
            "counts": self.counts,
        }


@dataclass
class PrepareReport:
    """Aggregate result of a ``prepare`` run for one model family."""

    family: str
    data_root: Path
    steps: list[PrepareStep] = field(default_factory=list)

    def add(self, step: PrepareStep) -> PrepareStep:
        self.steps.append(step)
        return step

    @property
    def ok(self) -> bool:
        return all(step.status != STATUS_ERROR for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "data_root": str(self.data_root),
            "steps": [step.to_dict() for step in self.steps],
        }
