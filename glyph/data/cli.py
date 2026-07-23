"""Typer interface for ``glyph data prepare`` (thin wrapper over the library)."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Prepare release datasets and benchmarks.")


class MarkushStage(StrEnum):
    """Which Markush stage ``glyph data prepare --markush`` should build."""

    sources = "sources"
    benchmarks = "benchmarks"


class MarkushBenchmark(StrEnum):
    """The three DS4SD Markush eval benchmarks selectable via ``--markush-benchmark``."""

    ip5m = "ip5m"
    m2s = "m2s"
    uspto_markush = "uspto_markush"


def _print_report(report, *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
        return
    typer.echo(f"[{report.family}] data root: {report.data_root}")
    for step in report.steps:
        line = f"  {step.status:16s} {step.name}"
        if step.detail:
            line += f" — {step.detail}"
        typer.echo(line)
        if step.path is not None and step.status in {"materialized", "skipped_exists"}:
            typer.echo(f"                   -> {step.path}")


@app.command("prepare")
def prepare(
    ocsr: Annotated[
        bool, typer.Option("--ocsr", help="Prepare the OCSR datasets + 5719 benchmark.")
    ] = False,
    markush: Annotated[
        bool,
        typer.Option(
            "--markush",
            help="Prepare the Markush training sources (add '--markush-only benchmarks' "
            "for the eval JSONLs).",
        ),
    ] = False,
    all_families: Annotated[
        bool, typer.Option("--all", help="Prepare every implemented family (OCSR and Markush).")
    ] = False,
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Override $GLYPH_DATA_ROOT (default env or ./data)."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Sample this many rows/images per source where supported (small validation runs).",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Plan every step without network or disk writes.")
    ] = False,
    allow_large: Annotated[
        bool,
        typer.Option(
            "--allow-large",
            help="Permit the multi-GB source downloads (USPTO-680K, Stereo-200K, full PubChem).",
        ),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Re-materialize even if the artifact already exists.")
    ] = False,
    benchmark_parquet: Annotated[
        Path | None,
        typer.Option(
            "--benchmark-parquet",
            exists=True,
            dir_okay=False,
            help="Local hheiden/USPTO_OCSR_benchmark snapshot (else auto-resolved).",
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-j",
            min=1,
            help="Parallel worker processes for the OCSR white-crop + adjacent-ring mine "
            "and the Markush source-image materialization "
            "(default min(cpu_count, 8)). Use -j 1 for serial.",
        ),
    ] = None,
    markush_only: Annotated[
        MarkushStage,
        typer.Option(
            "--markush-only",
            help="Restrict --markush to one stage: 'sources' (default) materializes the "
            "training image sources; 'benchmarks' builds ONLY the small DS4SD eval JSONLs "
            "(no multi-GB sources, no --allow-large).",
        ),
    ] = MarkushStage.sources,
    markush_benchmark: Annotated[
        list[MarkushBenchmark] | None,
        typer.Option(
            "--markush-benchmark",
            help="Build only the named eval benchmark(s) (repeatable). Default: all three "
            "(ip5m/m2s/uspto_markush). E.g. '--markush-benchmark ip5m' for IP5-M (n=878).",
        ),
    ] = None,
    markush_index: Annotated[
        str | None,
        typer.Option(
            "--markush-index",
            help="Frozen flat training index used to subset '--markush-only sources' to only "
            "the referenced source_ids (a path or a 'hub:' value, like the trainer's "
            "--data-path). Default: resolve 'hub:markush/train.jsonl'. If unresolvable, the "
            "full upstream sources are materialized.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit the report as JSON.")] = False,
) -> None:
    """Pull, preprocess, and lay out release datasets under $GLYPH_DATA_ROOT."""

    do_ocsr = ocsr or all_families
    do_markush = markush or all_families
    if not (do_ocsr or do_markush):
        raise typer.BadParameter("Select at least one of --ocsr, --markush, or --all.")

    if do_ocsr:
        from glyph.data.ocsr import prepare_ocsr

        report = prepare_ocsr(
            data_root=data_root,
            limit=limit,
            dry_run=dry_run,
            allow_large=allow_large,
            force=force,
            benchmark_parquet=benchmark_parquet,
            workers=workers,
        )
        _print_report(report, json_output=json_output)
        if not report.ok:
            raise typer.Exit(code=1)

    if do_markush:
        from glyph.data.markush import prepare_markush

        selected_benchmarks = (
            [bench.value for bench in markush_benchmark] if markush_benchmark else None
        )
        report = prepare_markush(
            data_root=data_root,
            limit=limit,
            dry_run=dry_run,
            allow_large=allow_large,
            force=force,
            only=markush_only.value,
            benchmarks=selected_benchmarks,
            workers=workers,
            index_path=markush_index,
        )
        _print_report(report, json_output=json_output)
        if not report.ok:
            raise typer.Exit(code=1)


def main() -> None:
    """Run the data CLI."""

    app()


if __name__ == "__main__":
    main()
