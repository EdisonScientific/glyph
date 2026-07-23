"""Typer entry points for checkpoint evaluation utilities."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from glyph.markush.eval.checkpoint import main as checkpoint_main
from glyph.markush.eval.common import DEFAULT_MAX_NEW_TOKENS
from glyph.markush.eval.sample_validation import main as sample_validation_main

app = typer.Typer(
    no_args_is_help=True,
    help="Evaluate Markush checkpoints and inspect sample generations.",
)

#: Frozen greedy decode recipe that reproduces the published MarkushGlyph
#: headline (ip5m greedy 511/878) given the published weights. Baked into
#: ``benchmark`` so the turnkey predict step matches the paper without
#: hand-tuning knobs. Note the checkpoint/base are resolved from the Hugging
#: Face Hub at ``main`` (no revision pin here), so reproduction assumes the
#: currently-published weights.
_GREEDY_SEED = 20260625
_GREEDY_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
_GREEDY_PROMPT_POLICY = "visible_sg"


def _forward_to_entrypoint(command: str, args: list[str], main: Callable[[], None]) -> None:
    """Run an existing library entry point behind the package Typer CLI."""

    original_argv = sys.argv
    sys.argv = [command, *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@app.command(
    "checkpoint",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def checkpoint(ctx: typer.Context) -> None:
    """Run benchmark-style MCS evaluation for a LoRA checkpoint."""

    _forward_to_entrypoint("markush eval checkpoint", list(ctx.args), checkpoint_main)


@app.command(
    "samples",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def samples(ctx: typer.Context) -> None:
    """Print target/prediction pairs for small Markush and OCSR samples."""

    _forward_to_entrypoint("markush eval samples", list(ctx.args), sample_validation_main)


@app.command("benchmark")
def benchmark(
    benchmark: Annotated[
        str | None,
        typer.Option("--benchmark", help="Benchmark name: ip5m, m2s, or uspto_markush."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Argument(help="Benchmark name (alternative to --benchmark)."),
    ] = None,
    checkpoint: Annotated[
        str,
        typer.Option(
            "--checkpoint",
            help=(
                "LoRA adapter path or HF repo id (default: EdisonScientific/MarkushGlyph). "
                "Security: loaded with trust_remote_code=True / torch unpickling and may "
                "execute code from the referenced repo — only use trusted sources."
            ),
        ),
    ] = "EdisonScientific/MarkushGlyph",
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model",
            help=(
                "Base model path or HF repo id (default: Qwen/Qwen3.5-2B-Base). "
                "Security: loaded with trust_remote_code=True and may execute code from "
                "the referenced repo — only use trusted sources."
            ),
        ),
    ] = "Qwen/Qwen3.5-2B-Base",
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, help="Output predictions JSON path."),
    ] = Path("preds.json"),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", min=1)
    ] = _GREEDY_MAX_NEW_TOKENS,
    num_gpus: Annotated[int, typer.Option("--num-gpus", min=0)] = 1,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    benchmark_parquet: Annotated[
        Path | None,
        typer.Option(
            "--benchmark-parquet",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional local parquet override (default: the hosted, auto-downloaded parquet).",
        ),
    ] = None,
    dest_root: Annotated[
        Path | None,
        typer.Option(
            "--dest-root",
            help="Where to unpack the benchmark JSONL + images (default: under eval_out).",
        ),
    ] = None,
) -> None:
    """Greedy-decode a hosted benchmark end to end (STEP 1 of the two-step flow).

    Auto-downloads the self-contained ``<benchmark>`` parquet from
    ``EdisonScientific/glyph-datasets``, unpacks it to a fast_eval-ready
    JSONL + image tree, and runs the frozen greedy recipe. Score the resulting
    predictions with ``tools/mg2_scorer/score_benchmark.sh`` (STEP 2).
    """

    from glyph.markush.eval import fast_eval
    from glyph.markush.eval.benchmark_data import materialize_benchmark, normalize_benchmark_name

    chosen = benchmark or name
    if chosen is None:
        typer.echo("Error: pass a benchmark name via --benchmark or as an argument.", err=True)
        raise typer.Exit(code=2)
    if benchmark is not None and name is not None and benchmark != name:
        typer.echo("Error: benchmark name given twice with different values.", err=True)
        raise typer.Exit(code=2)

    try:
        canonical = normalize_benchmark_name(chosen)
        eval_jsonl, image_root = materialize_benchmark(
            canonical,
            dest_root,
            explicit_parquet=benchmark_parquet,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"[benchmark] {canonical}: {eval_jsonl}")

    argv = [
        "--checkpoint",
        checkpoint,
        "--base-model",
        base_model,
        "--eval-data",
        str(eval_jsonl),
        "--image-root",
        str(image_root),
        "--output",
        str(output),
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
        "--num-gpus",
        str(num_gpus),
        "--device",
        device,
        "--markush-prompt-policy",
        _GREEDY_PROMPT_POLICY,
        "--multi-image-mode",
        "single",
        "--seed",
        str(_GREEDY_SEED),
        "--skip-mcs",
        "--quiet",
    ]
    if limit is not None:
        argv += ["--limit", str(limit)]

    _forward_to_entrypoint("markush eval benchmark", argv, fast_eval.main)


@app.command("baseline")
def baseline(
    benchmark: Annotated[
        str,
        typer.Option("--benchmark", help="Benchmark name: ip5m, m2s, or uspto_markush."),
    ],
    model: Annotated[
        list[str],
        typer.Option(
            "--model",
            help=(
                "Frontier model codename or litellm 'provider/model' string. "
                "Repeat to run several models."
            ),
        ),
    ],
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help=(
                "Durable results root. Cells land under <out-dir>/<model>/<benchmark>/. "
                "Defaults to the GLYPH data root (eval_out/frontier_baselines)."
            ),
        ),
    ] = None,
    variant: Annotated[
        str,
        typer.Option(
            "--variant",
            help="Prompt variant: default (exact MarkushGlyph prompt), format, or both.",
        ),
    ] = "default",
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 8000,
    reasoning_effort: Annotated[str | None, typer.Option("--reasoning-effort")] = None,
    temperature: Annotated[float | None, typer.Option("--temperature")] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 8,
    num_retries: Annotated[int, typer.Option("--num-retries", min=0)] = 6,
    timeout: Annotated[float, typer.Option("--timeout", min=1.0)] = 180.0,
    score: Annotated[bool, typer.Option("--score/--no-score")] = True,
    ds4sd: Annotated[bool, typer.Option("--ds4sd/--no-ds4sd")] = True,
    probe: Annotated[
        bool,
        typer.Option("--probe", help="One-row reachability probe per model, then exit."),
    ] = False,
) -> None:
    """Establish a frontier-model baseline on a Markush benchmark, via litellm.

    Feeds the SAME benchmark image + exact ``visible_sg`` Markush prompt to each
    frontier model, parses the output with the SAME eval extraction, writes a
    ``predictions.json`` in the ``glyph markush eval benchmark`` schema, and
    scores it under BOTH the official DS4SD grader and the strict/ours grader.
    Predictions checkpoint per row (resumable) so long runs survive a crash.
    """

    from glyph.markush.eval import baselines
    from glyph.paths import data_path

    resolved_out_dir = out_dir or data_path("eval_out", "frontier_baselines")
    try:
        variants = list(baselines.iter_variants(variant))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if probe:
        summaries = []
        for one_model in model:
            probe_result = baselines.run_baseline(
                baselines.GenerationConfig(
                    model=one_model,
                    benchmark=benchmark,
                    variant=variants[0],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    num_retries=0,
                    concurrency=1,
                ),
                out_dir=resolved_out_dir,
                limit=1,
                score=False,
                progress=False,
            )
            summaries.append(probe_result)
        typer.echo(json.dumps(summaries, indent=2))
        return

    all_summaries = []
    for one_model in model:
        for one_variant in variants:
            summary = baselines.run_baseline(
                baselines.GenerationConfig(
                    model=one_model,
                    benchmark=benchmark,
                    variant=one_variant,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    num_retries=num_retries,
                    concurrency=concurrency,
                ),
                out_dir=resolved_out_dir,
                limit=limit,
                score=score,
                score_ds4sd_contract=ds4sd,
            )
            all_summaries.append(summary)
            typer.echo(json.dumps(summary, indent=2))

    roll_up = (
        Path(resolved_out_dir)
        / f"summary_{baselines.normalize_benchmark_name(benchmark)}_last_run.json"
    )
    roll_up.parent.mkdir(parents=True, exist_ok=True)
    roll_up.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    # Consolidated, non-clobbering table across every cell on disk (the roll-up
    # above is only this invocation's cells; this scans all models/variants).
    json_path, md_path = baselines.write_consolidated(resolved_out_dir)
    typer.echo(f"[baseline] consolidated: {json_path} + {md_path}")


@app.command("baseline-report")
def baseline_report(
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Frontier-baseline results root to consolidate."),
    ],
) -> None:
    """Rebuild the consolidated ``summary_all.json`` + ``summary_all.md`` table.

    Scans every ``<out-dir>/<model>/<benchmark>/<variant>.summary.json`` cell
    (never clobbered) into one table; safe to re-run as background runs finish.
    """

    from glyph.markush.eval import baselines

    json_path, md_path = baselines.write_consolidated(out_dir)
    typer.echo(f"[baseline-report] wrote {json_path}")
    typer.echo(f"[baseline-report] wrote {md_path}")
