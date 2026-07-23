"""Typer CLI commands for unified data import, preparation, and publishing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from glyph.markush.data.distribution import compare_profiles, load_profile, profile_paths
from glyph.markush.data.example_modes import DEFAULT_MODE_WEIGHTS
from glyph.markush.data.hf_publish import push_dataset_artifact
from glyph.markush.data.importers.legacy import convert_existing_files
from glyph.markush.data.importers.public_datasets import DATASET_SPECS, download_public_datasets
from glyph.markush.data.pipeline import (
    generate_failure_mode_decorated_samples,
    load_scaffold_pool,
    write_failure_mode_seed_outputs,
)
from glyph.markush.data.prepare_sft import MixWeights, prepare_sft_jsonl

app = typer.Typer(no_args_is_help=True, help="Unified Markush data commands.")
import_app = typer.Typer(no_args_is_help=True, help="Import datasets into the unified schema.")
generate_app = typer.Typer(
    no_args_is_help=True, help="Generate renderer-ready Markush seed sets."
)


def _print_json(value: dict) -> None:
    typer.echo(json.dumps(value, indent=2))


@import_app.command("public")
def import_public(
    output: Annotated[Path, typer.Option("--output")] = Path("data/unified"),
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            help=f"Dataset to import. Repeatable. Choices: all, {', '.join(DATASET_SPECS)}",
        ),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cache_dir: Annotated[str | None, typer.Option("--cache-dir")] = None,
    hf_split: Annotated[str | None, typer.Option("--hf-split")] = None,
    validate_configs_only: Annotated[bool, typer.Option("--validate-configs-only")] = False,
    trust_remote_code: Annotated[bool, typer.Option("--trust-remote-code")] = False,
    save_original: Annotated[bool, typer.Option("--save-original")] = False,
    save_images: Annotated[bool, typer.Option("--save-images/--no-save-images")] = True,
    save_unified: Annotated[bool, typer.Option("--save-unified")] = False,
    workers: Annotated[int, typer.Option("--workers")] = 1,
) -> None:
    """Import public MG2/MolParser/MolScribe datasets."""

    report = download_public_datasets(
        output=output,
        datasets=datasets or ["all"],
        limit=limit,
        cache_dir=cache_dir,
        hf_split=hf_split,
        validate_configs_only=validate_configs_only,
        trust_remote_code=trust_remote_code,
        save_original=save_original,
        save_images=save_images,
        save_unified=save_unified,
        workers=workers,
    )
    _print_json(report)


@import_app.command("legacy")
def import_legacy(
    inputs: Annotated[list[Path], typer.Option("--input")],
    output: Annotated[Path, typer.Option("--output")],
    source: Annotated[str, typer.Option("--source")] = "pipeline_generated",
    split: Annotated[str, typer.Option("--split")] = "train",
    task: Annotated[str, typer.Option("--task")] = "markush",
    image_has_text: Annotated[bool, typer.Option("--image-has-text/--no-image-has-text")] = True,
    save_hf: Annotated[Path | None, typer.Option("--save-hf")] = None,
) -> None:
    """Import existing local JSON/JSONL records into the unified schema."""

    report = convert_existing_files(
        inputs=inputs,
        output=output,
        source=source,
        split=split,
        task=task,
        image_has_text=image_has_text,
        save_hf=save_hf,
    )
    _print_json(report)


app.add_typer(import_app, name="import")
app.add_typer(generate_app, name="generate")


@app.command("profile")
def profile_data(
    inputs: Annotated[list[Path], typer.Option("--input")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    compare_to: Annotated[Path | None, typer.Option("--compare-to")] = None,
) -> None:
    """Profile Markush target features in JSON/JSONL data."""

    profile = profile_paths(inputs)
    if compare_to is not None:
        profile["comparison"] = compare_profiles(profile, load_profile(compare_to))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    _print_json(profile)


@generate_app.command("failure-seeds")
def generate_failure_seeds(
    scaffold_pool: Annotated[Path, typer.Option("--scaffold-pool")],
    output_jsonl: Annotated[Path, typer.Option("--output-jsonl")],
    output_csv: Annotated[Path | None, typer.Option("--output-csv")] = None,
    samples_per_focus: Annotated[int, typer.Option("--samples-per-focus")] = 5000,
    focus: Annotated[
        list[str] | None,
        typer.Option("--focus", help="Failure-mode focus to generate; repeatable."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    decorate_attempts_per_sample: Annotated[
        int, typer.Option("--decorate-attempts-per-sample")
    ] = 100,
    decorate_progress_interval: Annotated[int, typer.Option("--decorate-progress-interval")] = 0,
    id_prefix: Annotated[str, typer.Option("--id-prefix")] = "failure",
    stats_json: Annotated[Path | None, typer.Option("--stats-json")] = None,
) -> None:
    """Generate renderer-ready seeds balanced across failure-mode buckets."""

    try:
        scaffolds = load_scaffold_pool(scaffold_pool)
        focused_samples, stats = generate_failure_mode_decorated_samples(
            scaffolds,
            samples_per_focus=samples_per_focus,
            focuses=focus,
            seed=seed,
            max_attempts_per_sample=decorate_attempts_per_sample,
            progress_interval=decorate_progress_interval,
        )
        write_failure_mode_seed_outputs(
            focused_samples,
            output_jsonl=output_jsonl,
            output_csv=output_csv,
            id_prefix=id_prefix,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    stats["output_jsonl"] = str(output_jsonl)
    stats["output_csv"] = str(output_csv) if output_csv else None
    stats["focus_counts"] = dict(Counter(sample.focus for sample in focused_samples))
    if stats_json is not None:
        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    _print_json(stats)


@app.command("prepare-sft")
def prepare_sft(
    datasets: Annotated[list[Path], typer.Option("--dataset")],
    output: Annotated[Path, typer.Option("--output")],
    stats_json: Annotated[Path | None, typer.Option("--stats-json")] = None,
    total_samples: Annotated[int | None, typer.Option("--total-samples")] = None,
    text_dropout: Annotated[float, typer.Option("--text-dropout")] = 0.5,
    structure_only_markush_fraction: Annotated[
        float,
        typer.Option("--structure-only-markush-fraction"),
    ] = DEFAULT_MODE_WEIGHTS["structure_only_markush"],
    patent_image_markush_fraction: Annotated[
        float,
        typer.Option("--patent-image-markush-fraction"),
    ] = DEFAULT_MODE_WEIGHTS["patent_image_markush"],
    text_conditioned_markush_fraction: Annotated[
        float,
        typer.Option("--text-conditioned-markush-fraction"),
    ] = DEFAULT_MODE_WEIGHTS["text_conditioned_markush"],
    molecule_ocsr_fraction: Annotated[
        float,
        typer.Option("--molecule-ocsr-fraction"),
    ] = DEFAULT_MODE_WEIGHTS["molecule_ocsr"],
    molparser_converted_markush_fraction: Annotated[
        float,
        typer.Option(
            "--molparser-converted-markush-fraction",
            help="Opt-in fraction for validated MolParser E-SMILES converted to cxsmiles_opt.",
        ),
    ] = DEFAULT_MODE_WEIGHTS["molparser_converted_markush"],
    molparser_native_markush_fraction: Annotated[
        float,
        typer.Option(
            "--molparser-native-markush-fraction",
            help="Opt-in fraction for valid MolParser-Markush rows trained as native E-SMILES.",
        ),
    ] = DEFAULT_MODE_WEIGHTS["molparser_native_markush"],
    structured_transcript_markush_fraction: Annotated[
        float,
        typer.Option(
            "--structured-transcript-markush-fraction",
            help=(
                "Opt-in fraction for deterministic transcript-plus-final canonicalized "
                "cxsmiles_opt Markush rows."
            ),
        ),
    ] = DEFAULT_MODE_WEIGHTS["structured_transcript_markush"],
    seed: Annotated[int, typer.Option("--seed")] = 42,
    include_eval: Annotated[bool, typer.Option("--include-eval")] = False,
    oversample: Annotated[bool, typer.Option("--oversample/--no-oversample")] = True,
    exclude_files: Annotated[
        list[Path] | None,
        typer.Option("--exclude-file", help="Audit report/exclusion JSON or JSONL rows to omit"),
    ] = None,
    canonicalize_markush_targets: Annotated[
        bool,
        typer.Option(
            "--canonicalize-markush-targets",
            help="Prefer verified canonicalized cxsmiles_opt targets for Markush examples",
        ),
    ] = False,
    require_canonical_markush_targets: Annotated[
        bool,
        typer.Option(
            "--require-canonical-markush-targets",
            help="Skip Markush rows whose canonicalized target cannot be verified",
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help="Process workers for target validation/transcript generation during mixing",
        ),
    ] = 1,
) -> None:
    """Prepare mixed SFT chat JSONL from unified records."""

    stats = prepare_sft_jsonl(
        datasets=datasets,
        output=output,
        stats_json=stats_json,
        total_samples=total_samples,
        text_dropout=text_dropout,
        weights=MixWeights(
            structure_only_markush=structure_only_markush_fraction,
            patent_image_markush=patent_image_markush_fraction,
            text_conditioned_markush=text_conditioned_markush_fraction,
            molecule_ocsr=molecule_ocsr_fraction,
            molparser_native_markush=molparser_native_markush_fraction,
            molparser_converted_markush=molparser_converted_markush_fraction,
            structured_transcript_markush=structured_transcript_markush_fraction,
        ),
        seed=seed,
        include_eval=include_eval,
        oversample=oversample,
        exclude_files=exclude_files or [],
        canonicalize_markush_targets=canonicalize_markush_targets,
        require_canonical_markush_targets=require_canonical_markush_targets,
        workers=workers,
    )
    _print_json(stats)


@app.command("push-hf")
def push_hf(
    dataset: Annotated[Path, typer.Option("--dataset")],
    repo: Annotated[str, typer.Option("--repo")],
    config_name: Annotated[str | None, typer.Option("--config-name")] = None,
    private: Annotated[bool, typer.Option("--private")] = False,
    commit_message: Annotated[
        str, typer.Option("--commit-message")
    ] = "Upload unified Markush dataset",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Push a local unified dataset artifact to Hugging Face Hub."""

    report = push_dataset_artifact(
        dataset_path=dataset,
        repo=repo,
        config_name=config_name,
        private=private,
        commit_message=commit_message,
        dry_run=dry_run,
    )
    _print_json(report)
