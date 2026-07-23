"""Unit tests for the Markush release data-prep pipeline (``glyph data prepare --markush``).

These cover the orchestration without touching the network: layout creation,
provenance-sidecar external attribution, the large-download refusal + dry-run
plan, benchmark eval-JSONL preparation with relative image emission, the
OCSR -> no-extension-Markush conversion + decontamination, the assembled 5%
multitask mix composition/ratio/seed determinism + ``$GLYPH_DATA_ROOT``-relative
image rewriting, and idempotent re-runs. The multi-GB HF source pulls and the
full 250K base are exercised live / documented elsewhere (see the module
docstring in ``glyph/data/markush.py`` and ``glyph/data/README.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

from glyph.data import layout
from glyph.data.markush import prepare_markush
from glyph.data.markush_eval_jsonl import relativize_image_ref
from glyph.data.markush_mix import build_mix, parse_partition, relativize_row_images
from glyph.data.markush_multitask import (
    assemble_multitask_mix,
    convert_ocsr_rows_to_no_extension_markush,
    count_rows,
    decontaminate_ocsr_rows,
    eval_canonical_smiles_set,
)

# --- layout ------------------------------------------------------------------


def test_layout_matches_release_contract(tmp_path: Path) -> None:
    assert layout.markush_dataset_dir("mg1_synthetic", root=tmp_path) == (
        tmp_path / "datasets" / "markush" / "mg1_synthetic"
    )
    assert layout.markush_benchmark_dir("ip5m", root=tmp_path) == (
        tmp_path / "benchmarks" / "markush" / "ip5m"
    )
    assert layout.markush_multitask_mix_path(root=tmp_path) == (
        tmp_path / "datasets" / "markush" / "prepared" / "multitask_5pct_seed74" / "train.jsonl"
    )
    assert set(layout.MARKUSH_DATASET_NAMES) == {
        "mg1_synthetic",
        "mg2_structure_only",
        "molparser",
        "ordinary_5pct",
    }
    assert set(layout.MARKUSH_BENCHMARK_NAMES) == {"ip5m", "m2s", "uspto_markush"}


def test_dataset_to_spec_mapping_records_the_naming_divergence() -> None:
    # Layout dir names diverge from the importer spec names for two entries.
    assert layout.MARKUSH_DATASET_TO_SPEC["mg2_structure_only"] == "uspto-mol-m"
    assert layout.MARKUSH_DATASET_TO_SPEC["ordinary_5pct"] == "molscribe-ocsr"
    assert layout.MARKUSH_DATASET_TO_SPEC["mg1_synthetic"] == "mg1-synthetic"
    assert layout.MARKUSH_DATASET_TO_SPEC["molparser"] == "molparser-markush"


# --- provenance: external attribution (hard hygiene requirement) -------------


def test_provenance_attributes_training_sources_as_external_not_our_upload() -> None:
    for name in ("mg1_synthetic", "mg2_structure_only", "molparser", "ordinary_5pct"):
        rec = layout.MARKUSH_PROVENANCE[name]
        assert rec["this_work"] is False, name
        assert rec["role"].startswith("external"), name
        assert "EXTERNAL" in rec["attribution"], name
        assert "NOT re-hosted" in rec["attribution"], name

    mg1 = layout.MARKUSH_PROVENANCE["mg1_synthetic"]
    assert mg1["source_repo"] == "docling-project/MarkushGrapher-Datasets"
    assert mg1["pinned_revision"] == "a6cf981dd100198ce87e8099402f0645b44080ed"
    assert "PUBLIC MG1 rows, not the local Java/CDK synthetic generator" in mg1["attribution"]

    mg2 = layout.MARKUSH_PROVENANCE["mg2_structure_only"]
    assert mg2["source_repo"] == "docling-project/MarkushGrapher-2-Datasets"
    assert mg2["pinned_revision"] == layout.MG2_REVISION

    molparser = layout.MARKUSH_PROVENANCE["molparser"]
    assert molparser["source_repo"] == "UniParser/MolParser-7M"
    assert molparser["pinned_revision"] == "f37e164f179c7cd8470fc879422c29af2c47874c"

    molscribe = layout.MARKUSH_PROVENANCE["ordinary_5pct"]
    assert molscribe["source_repo"] == "yujieq/MolScribe"
    assert molscribe["pinned_revision"] == "a0189776b7415b82795c7ee81eed311bf5c8724b"
    assert molscribe["citation"] == "qian2023molscribe"


def test_benchmark_provenance_pins_ip5m_contract() -> None:
    ip5m = layout.MARKUSH_BENCHMARK_PROVENANCE["ip5m"]
    assert ip5m["this_work"] is False
    assert ip5m["expected_rows"] == 878
    assert ip5m["pinned_revision"] == "e90868cdd3b62f1fe175d9ede9e0c65485c4a96c"
    assert ip5m["fingerprint"] == "15733712f9c574af"
    assert layout.MARKUSH_BENCHMARK_PROVENANCE["m2s"]["expected_rows"] == 103
    assert layout.MARKUSH_BENCHMARK_PROVENANCE["uspto_markush"]["expected_rows"] == 74


def test_mix_provenance_is_this_work_composition_over_external_images() -> None:
    mix = layout.MARKUSH_MULTITASK_MIX_PROVENANCE
    assert mix["this_work"] is True
    assert mix["seed"] == 74
    assert mix["expected_total_rows"] == 263_158
    assert mix["expected_source_counts"] == {
        "mg1_synthetic": 152_620,
        "uspto_mol_m": 52_524,
        "molparser": 44_856,
        "molscribe": 13_158,
    }
    assert "EXTERNAL public data" in mix["attribution"]


# --- image-path relativization (path-portability directive) ------------------


def test_relativize_image_ref_variants(tmp_path: Path) -> None:
    root = tmp_path / "root"
    # already relative -> unchanged
    assert relativize_image_ref("images/ip5m/test/a.png", root=root) == (
        "images/ip5m/test/a.png",
        False,
    )
    # under root -> made relative
    assert relativize_image_ref(str(root / "images/ip5m/test/a.png"), root=root) == (
        "images/ip5m/test/a.png",
        False,
    )
    # cluster absolute path outside root but with an images/ segment (mix relativizer)
    from glyph.data.markush_mix import relativize_image_ref as mix_relativize

    assert mix_relativize(
        "/tmp/glyphdata/x/images/mg1_synthetic/train/000001_a.png", root=root
    ) == (
        "images/mg1_synthetic/train/000001_a.png",
        False,
    )
    # unresolvable absolute -> flagged
    assert relativize_image_ref("/tmp/glyphdata/x/a.png", root=root) == (
        "/tmp/glyphdata/x/a.png",
        True,
    )


def test_relativize_row_images_rewrites_all_image_keys(tmp_path: Path) -> None:
    root = tmp_path / "root"
    row = {
        "images": ["/tmp/glyphdata/x/images/molparser/train/000001_a.png"],
        "image": "/tmp/glyphdata/x/images/molparser/train/000001_a.png",
    }
    rewritten, unresolved = relativize_row_images(row, root=root)
    assert unresolved is False
    assert rewritten["images"] == ["images/molparser/train/000001_a.png"]
    assert rewritten["image"] == "images/molparser/train/000001_a.png"


# --- OCSR -> no-extension-Markush conversion + decontamination ---------------


def _ocsr_unified_row(row_id: str, smiles: str) -> dict:
    return {
        "id": row_id,
        "source": "molscribe",
        "source_id": row_id,
        "task": "ocsr",
        "smiles": smiles,
        "images": [f"/tmp/glyphdata/o/images/ordinary_5pct/train/{row_id}.png"],
    }


def test_ocsr_conversion_produces_no_extension_markush_targets() -> None:
    rows = [_ocsr_unified_row("o0", "CCO"), _ocsr_unified_row("o1", "c1ccccc1")]
    converted, skipped = convert_ocsr_rows_to_no_extension_markush(rows)
    assert skipped == []
    assert len(converted) == 2
    for row in converted:
        assert row["task"] == "markush"
        assert row["mode"] == "molecule_ocsr_as_markush_no_extension"
        target = row["messages"][-1]["content"]
        assert target.startswith("<markush><cxsmi>")
        assert target.endswith("<stable></stable></markush>")
        assert "<r>" not in target.lower()


def test_decontamination_removes_eval_colliding_rows() -> None:
    converted, _ = convert_ocsr_rows_to_no_extension_markush(
        [_ocsr_unified_row("o0", "CCO"), _ocsr_unified_row("o1", "c1ccccc1")]
    )
    # "OCC" canonicalizes to the same molecule as "CCO"; both should be removed.
    eval_set = eval_canonical_smiles_set(_eval_jsonl_with_smiles(["OCC"]))
    kept, removed = decontaminate_ocsr_rows(converted, eval_canonical_smiles=eval_set)
    removed_ids = {entry["id"] for entry in removed}
    assert "o0" in removed_ids
    assert [row["id"] for row in kept] == ["o1"]


def _eval_jsonl_with_smiles(smiles: list[str]) -> list[Path]:
    import tempfile

    directory = Path(tempfile.mkdtemp())
    path = directory / "eval.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, smi in enumerate(smiles):
            handle.write(json.dumps({"id": f"b{index}", "smiles": smi}) + "\n")
    return [path]


def test_count_rows_matches_headline_5pct_split() -> None:
    markush, ocsr = count_rows(
        total_samples=None,
        markush_available=250_000,
        ocsr_fraction=0.05,
        ocsr_count=None,
        markush_count=250_000,
    )
    assert (markush, ocsr) == (250_000, 13_158)


# --- assembled multitask mix -------------------------------------------------


def _markush_base_row(index: int) -> dict:
    target = f"<markush><cxsmi>CC{'C' * (index % 3)}</cxsmi><stable></stable></markush>"
    return {
        "id": f"m{index}",
        "source": "uspto_mol_m",
        "source_id": f"m{index}",
        "mode": "structure_only_markush",
        "target_format": "canonicalized cxsmiles_opt",
        "messages": [
            {"role": "user", "content": "<image>\nExtract."},
            {"role": "assistant", "content": target},
        ],
        "images": [f"/tmp/glyphdata/x/images/mg2_structure_only/train/{index:06d}_m{index}.png"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_assemble_multitask_mix_composition_ratio_and_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    base = tmp_path / "base.jsonl"
    _write_jsonl(base, [_markush_base_row(i) for i in range(20)])
    ocsr = tmp_path / "molscribe_unified.jsonl"
    _write_jsonl(
        ocsr,
        [_ocsr_unified_row(f"o{i}", smi) for i, smi in enumerate(["CCO", "c1ccccc1", "CC(=O)O"])],
    )

    out_dir = root / "datasets" / "markush" / "prepared" / "multitask_5pct_seed74"
    manifest = assemble_multitask_mix(
        base_jsonl=base,
        ocsr_source_jsonl=ocsr,
        out_dir=out_dir,
        data_root=root,
        benchmark_eval_jsonls=[],
        seed=74,
        ocsr_fraction=0.05,
    )

    # round(20 * 0.05 / 0.95) == 1
    assert manifest["base_count"] == 20
    assert manifest["ocsr_count"] == 1
    assert manifest["written_rows"] == 21
    assert manifest["source_counts"] == {"uspto_mol_m": 20, "molscribe": 1}
    assert manifest["unresolved_image_rows"] == 0

    rows = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()]
    # Every image ref is $GLYPH_DATA_ROOT-relative (no leaked absolute cluster path).
    all_images = [img for row in rows for img in row.get("images", [])]
    assert all_images and all(not str(img).startswith("/") for img in all_images)
    assert any(img.startswith("images/mg2_structure_only/") for img in all_images)
    # partition provenance carries no absolute path either.
    sources = {row["stable_scaling_mix"]["partition_source_jsonl"] for row in rows}
    assert all(not value.startswith("/") for value in sources)


def test_assemble_multitask_mix_is_deterministic_for_same_output_dir(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    base = tmp_path / "base.jsonl"
    _write_jsonl(base, [_markush_base_row(i) for i in range(30)])
    ocsr = tmp_path / "molscribe_unified.jsonl"
    _write_jsonl(ocsr, [_ocsr_unified_row(f"o{i}", "CCO") for i in range(5)])
    out_dir = root / "mix"

    assemble_multitask_mix(
        base_jsonl=base, ocsr_source_jsonl=ocsr, out_dir=out_dir, data_root=root, seed=74
    )
    first = (out_dir / "train.jsonl").read_text()
    assemble_multitask_mix(
        base_jsonl=base, ocsr_source_jsonl=ocsr, out_dir=out_dir, data_root=root, seed=74
    )
    assert (out_dir / "train.jsonl").read_text() == first


def test_build_mix_relativize_mode_rewrites_cluster_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    source = tmp_path / "part.jsonl"
    _write_jsonl(
        source,
        [
            {
                "id": "a",
                "source": "uspto_mol_m",
                "source_id": "a",
                "mode": "structure_only_markush",
                "images": ["/tmp/glyphdata/x/images/mg2_structure_only/train/000001_a.png"],
            }
        ],
    )
    build_mix(
        partitions=[parse_partition(f"base=1:{source}")],
        output_jsonl=tmp_path / "train.jsonl",
        manifest_json=tmp_path / "manifest.json",
        report_md=None,
        image_roots={},
        seed=74,
        oversample=True,
        shuffle=False,
        require_absolute=False,
        allow_eval_sources=False,
        argv=[],
        relativize_to_root=root,
    )
    row = json.loads((tmp_path / "train.jsonl").read_text())
    assert row["images"] == ["images/mg2_structure_only/train/000001_a.png"]


# --- benchmark eval JSONL ----------------------------------------------------


def test_prepare_benchmark_emits_relative_image_and_contract_shape(tmp_path: Path) -> None:
    from glyph.data.markush_eval_jsonl import prepare_benchmark

    root = tmp_path / "data"
    root.mkdir()
    unified = tmp_path / "ip5m" / "unified.jsonl"
    _write_jsonl(
        unified,
        [
            {
                "id": "EP_1",
                "source": "ip5-markush",
                "split": "test",
                "task": "markush",
                "image": "images/ip5m/test/EP_1.png",
                "cxsmiles_opt": "NC(<r>R</r>)=NC1=CC=CC=C1",
            }
        ],
    )
    out = tmp_path / "eval_ip5m.jsonl"
    stats = prepare_benchmark(
        source_jsonl=unified, output_jsonl=out, benchmark="ip5m", data_root=root
    )
    assert stats.written_rows == 1
    assert stats.absolute_image_rows == 0
    row = json.loads(out.read_text())
    assert row["benchmark"] == "ip5m"
    assert row["task"] == "markush"
    assert row["image"] == "images/ip5m/test/EP_1.png"
    assert row["images"] == ["images/ip5m/test/EP_1.png"]
    assert row["annotation"].startswith("<markush><cxsmi>")
    assert row["messages"][0]["content"].startswith("<image>\n")


# --- orchestration -----------------------------------------------------------


def test_prepare_markush_dry_run_writes_nothing(tmp_path: Path) -> None:
    # The default (sources-only) plans just the four training image sources.
    report = prepare_markush(data_root=tmp_path, dry_run=True)
    assert report.family == "markush"
    assert report.data_root == tmp_path
    assert {step.status for step in report.steps} == {"planned"}
    assert not (tmp_path / "datasets").exists()
    assert not (tmp_path / "benchmarks").exists()
    # Only the four training sources are planned -- no benchmark steps.
    names = {step.name for step in report.steps}
    assert names == set(layout.MARKUSH_DATASET_NAMES)
    assert not any(name.startswith("benchmark_") for name in names)


def test_prepare_markush_refuses_large_sources_without_allow_or_limit(tmp_path: Path) -> None:
    # The multi-GB training sources are size-guarded; the small eval benchmarks are
    # NOT. Scope this to the sources stage so it asserts the
    # guard without triggering the (small, network) benchmark downloads.
    report = prepare_markush(data_root=tmp_path, allow_large=False, only="sources")
    statuses = {step.name: step.status for step in report.steps}
    assert statuses["mg1_synthetic"] == "refused"
    assert statuses["mg2_structure_only"] == "refused"
    assert statuses["molparser"] == "refused"
    assert statuses["ordinary_5pct"] == "refused"
    # Only the source steps ran -- no benchmark steps.
    assert not any(name.startswith("benchmark_") for name in statuses)
    assert report.ok
    assert not (tmp_path / "datasets" / "markush" / "mg1_synthetic" / "unified.jsonl").exists()


# --- keep-set (index-subset) materialization --------------------------------


def _write_index(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a tiny flat training index (``source`` + ``source_id`` per row)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for source, source_id in rows:
            handle.write(
                json.dumps(
                    {
                        "source": source,
                        "source_id": source_id,
                        "images": [f"images/{source}/train/{source_id}.png"],
                    }
                )
                + "\n"
            )


def test_load_markush_keep_ids_groups_source_ids_by_source(tmp_path: Path) -> None:
    from glyph.data.markush_sources import load_markush_keep_ids

    index = tmp_path / "train.jsonl"
    _write_index(
        index,
        [
            ("mg1_synthetic", "a"),
            ("mg1_synthetic", "b"),
            ("uspto_mol_m", "u1"),
            ("molparser", "p9"),
            ("molscribe", "s-x"),
        ],
    )

    keep = load_markush_keep_ids(str(index))
    assert keep == {
        "mg1_synthetic": {"a", "b"},
        "uspto_mol_m": {"u1"},
        "molparser": {"p9"},
        "molscribe": {"s-x"},
    }
    # An unresolvable explicit path yields None (full-source fallback), no network.
    assert load_markush_keep_ids(str(tmp_path / "missing.jsonl")) is None


def test_prepare_markush_sources_subsets_to_index_source_ids(tmp_path: Path, monkeypatch) -> None:
    # A small frozen index resolvable offline through the trainer's mirror env.
    mirror = tmp_path / "mirror"
    _write_index(
        mirror / "markush" / "train.jsonl",
        [
            ("mg1_synthetic", "m-a"),
            ("mg1_synthetic", "m-b"),
            ("uspto_mol_m", "u-1"),
            ("molparser", "p-9"),
            ("molscribe", "s-x"),
        ],
    )
    monkeypatch.setenv("GLYPH_MARKUSH_TRAIN_ROOT", str(mirror))

    import glyph.data.markush_sources as sources_mod
    from glyph.data.ocsr_sources import FetchOutcome

    captured: dict[str, set[str] | None] = {}

    def fake_import(dataset_name, out_root, *, keep_ids=None, **_kwargs):
        captured[dataset_name] = keep_ids
        return FetchOutcome(
            "materialized",
            "stub",
            path=out_root / dataset_name / "unified.jsonl",
            counts={"rows": 0},
        )

    monkeypatch.setattr(sources_mod, "import_markush_source", fake_import)

    report = prepare_markush(data_root=tmp_path / "data", allow_large=True, only="sources")
    assert report.ok
    # Each §3 dataset receives exactly its own source's index-referenced ids,
    # resolving the mg2->uspto_mol_m and ordinary_5pct->molscribe key divergence.
    assert captured["mg1_synthetic"] == {"m-a", "m-b"}
    assert captured["mg2_structure_only"] == {"u-1"}
    assert captured["molparser"] == {"p-9"}
    assert captured["ordinary_5pct"] == {"s-x"}


def test_prepare_markush_sources_full_when_index_unresolvable(tmp_path: Path, monkeypatch) -> None:
    import glyph.data.markush_sources as sources_mod
    from glyph.data.ocsr_sources import FetchOutcome

    captured: dict[str, set[str] | None] = {}

    def fake_import(dataset_name, out_root, *, keep_ids=None, **_kwargs):
        captured[dataset_name] = keep_ids
        return FetchOutcome(
            "materialized",
            "stub",
            path=out_root / dataset_name / "unified.jsonl",
            counts={"rows": 0},
        )

    monkeypatch.setattr(sources_mod, "import_markush_source", fake_import)

    # An explicit, nonexistent index resolves to nothing (no network): the run
    # falls back to full-source materialization (keep_ids stays None everywhere).
    report = prepare_markush(
        data_root=tmp_path / "data",
        allow_large=True,
        only="sources",
        index_path=str(tmp_path / "nope.jsonl"),
    )
    assert report.ok
    assert set(captured) == set(layout.MARKUSH_DATASET_NAMES)
    assert all(value is None for value in captured.values())


# --- benchmark-only selector ------------------------------------------------


def test_cli_enums_match_library_constants() -> None:
    from glyph.data.cli import MarkushBenchmark, MarkushStage
    from glyph.data.markush import MARKUSH_STAGES

    assert {b.value for b in MarkushBenchmark} == set(layout.MARKUSH_BENCHMARK_NAMES)
    assert {s.value for s in MarkushStage} == set(MARKUSH_STAGES)


def test_benchmark_import_is_not_size_guarded() -> None:
    from glyph.data.markush_sources import _should_refuse_large

    # A full pull of a size-guarded training source is refused without allow_large.
    assert _should_refuse_large(limit=None, allow_large=False, guard_large=True) is True
    assert _should_refuse_large(limit=None, allow_large=True, guard_large=True) is False
    assert _should_refuse_large(limit=10, allow_large=False, guard_large=True) is False
    # The eval benchmarks (guard_large=False) are never size-refused.
    assert _should_refuse_large(limit=None, allow_large=False, guard_large=False) is False


def test_prepare_markush_benchmarks_only_plans_only_benchmarks(tmp_path: Path) -> None:
    report = prepare_markush(data_root=tmp_path, dry_run=True, only="benchmarks")
    names = {step.name for step in report.steps}
    assert names == {"benchmark_ip5m", "benchmark_m2s", "benchmark_uspto_markush"}
    assert {step.status for step in report.steps} == {"planned"}
    # No multi-GB source, base, or mix steps -- and nothing written.
    assert not (tmp_path / "datasets").exists()
    assert not (tmp_path / "benchmarks").exists()


def test_prepare_markush_single_benchmark_selector(tmp_path: Path) -> None:
    report = prepare_markush(
        data_root=tmp_path, dry_run=True, only="benchmarks", benchmarks=["ip5m"]
    )
    names = {step.name for step in report.steps}
    assert names == {"benchmark_ip5m"}


def test_prepare_markush_rejects_unknown_stage_or_benchmark(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        prepare_markush(data_root=tmp_path, dry_run=True, only="nope")
    with pytest.raises(ValueError):
        prepare_markush(data_root=tmp_path, dry_run=True, benchmarks=["not_a_benchmark"])


def test_prepare_markush_benchmark_builds_without_allow_large_and_emits_relative(
    tmp_path: Path,
) -> None:
    # Pre-seed the imported benchmark source so no network import is needed; this
    # exercises the (un-gated) benchmark orchestration path AND the relative-ref
    # emission with an absolute-under-root image path.
    from glyph.data.markush_sources import unified_jsonl_path

    source_root = layout.markush_benchmarks_dir("_source", root=tmp_path)
    unified = unified_jsonl_path(source_root, "ip5m")
    abs_image = tmp_path / "images" / "ip5m" / "test" / "EP_1.png"
    _write_jsonl(
        unified,
        [
            {
                "id": "EP_1",
                "source": "ip5-markush",
                "split": "test",
                "task": "markush",
                "image": str(abs_image),  # absolute, but under $GLYPH_DATA_ROOT
                "cxsmiles_opt": "NC(<r>R</r>)=NC1=CC=CC=C1",
            }
        ],
    )

    report = prepare_markush(
        data_root=tmp_path, only="benchmarks", benchmarks=["ip5m"], allow_large=False
    )
    step = next(s for s in report.steps if s.name == "benchmark_ip5m")
    # Materialized without --allow-large and without touching the network.
    assert step.status == "materialized"

    out = layout.markush_benchmark_dir("ip5m", root=tmp_path) / "eval_ip5m.jsonl"
    row = json.loads(out.read_text())
    # The absolute-under-root image was rewritten $GLYPH_DATA_ROOT-relative.
    assert row["image"] == "images/ip5m/test/EP_1.png"
    assert row["images"] == ["images/ip5m/test/EP_1.png"]
    assert not row["image"].startswith("/")
