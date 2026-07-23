# Frontier-model baselines (via litellm)

This is the pipeline behind the paper's frontier-model baselines: hosted models
evaluated on the Markush benchmarks through the same eval pipeline as
MarkushGlyph. Each model receives the same input MarkushGlyph sees (the
benchmark image plus the exact `visible_sg` Markush prompt) through
[litellm](https://github.com/BerriAI/litellm). The raw reply is parsed with the
same extraction the offline eval uses, written to a `predictions.json` in the
`glyph markush eval benchmark` schema, and scored under both scoring
contracts:

- the official DS4SD `get_scores` grader (`tools/mg2_scorer/score_benchmark.sh`), and
- the strict/ours graph grader (`glyph.markush.grading.grade_files`, `protocol="both"`).

The tool is `glyph/markush/eval/baselines.py` (library) + `glyph markush eval
baseline` (CLI). This directory carries only the isolated-environment setup and
a thin run wrapper; all logic lives in the library.

## 1. Environment (isolated, uv)

litellm is not a core runtime dependency. It is declared under the optional
`baseline` extra and imported lazily. Build a dedicated venv:

```bash
tools/frontier_baseline/setup_venv.sh            # CPU torch wheels
# or, for a CUDA box:
TORCH_BACKEND=cu128 tools/frontier_baseline/setup_venv.sh
source tools/frontier_baseline/.venv-baseline/bin/activate
```

Any environment that already has `glyph[markush]` installed can instead add just
litellm: `uv pip install 'glyph[baseline]'`.

## 2. Credentials

litellm reads standard provider keys from the environment:

- `ANTHROPIC_API_KEY` for `anthropic/...`
- `OPENAI_API_KEY` for `openai/...`

Confirm reachability before a full run with a one-row probe per model:

```bash
glyph markush eval baseline --benchmark uspto_markush --probe \
  --model claude-opus-4-8 --model gpt-5.6-sol --model claude-fable-5
```

Each probe prints a per-model summary; a non-empty `n_errors` with an `error`
string in the checkpoint means that model needs a key or workspace access.

Some hosted model ids are gated per organization or workspace (for example by a
data-retention requirement); the probe below confirms access before a full run.

## 3. Run

The benchmark inputs (images, prompts, targets) download automatically from
`EdisonScientific/glyph-datasets`; no data preparation is needed.

```bash
# One model, one benchmark (checkpoints per row; safe to re-run to resume):
glyph markush eval baseline --benchmark ip5m --model claude-opus-4-8 \
  --out-dir /path/to/frontier-baselines

# Several models at once, both prompt variants, custom decode budget:
glyph markush eval baseline --benchmark ip5m --variant both \
  --model claude-opus-4-8 --model gpt-5.6-sol \
  --max-tokens 8000 --concurrency 8 \
  --out-dir /path/to/frontier-baselines
```

Useful flags: `--variant {default,format,both}`, `--limit N` (first N rows),
`--reasoning-effort {low,medium,high}` (reasoning models), `--temperature`,
`--concurrency`, `--num-retries`, `--no-ds4sd` (skip the DS4SD contract),
`--no-score` (generate only).

The convenience wrapper `run_baseline.sh` forwards its arguments to the CLI:

```bash
tools/frontier_baseline/run_baseline.sh --benchmark ip5m --model claude-opus-4-8
```

## 4. Outputs (durable)

Under `<out-dir>/<model>/<benchmark>/`, per prompt variant:

| File | Purpose |
| --- | --- |
| `<variant>.checkpoint.jsonl` | Per-row predictions, appended as they complete. **Resumable**: re-running skips completed rows and retries failed ones. |
| `<variant>.predictions.json` | `glyph markush eval benchmark` schema; scored unchanged by the DS4SD grader. |
| `<variant>.grading_predictions.jsonl` / `<variant>.grading_gt.jsonl` | `{id, prediction}` / `{id, cxsmiles_opt}` consumed by `grade_files`. |
| `<variant>.ds4sd_scored.json` | Official DS4SD `get_scores` output. |
| `<variant>.strict_ours.json` | Strict/ours `grade_files(protocol="both")` output. |
| `<variant>.summary.json` | Per-cell roll-up (counts + both-contract scores). |

After a run, `glyph markush eval baseline` writes a per-invocation roll-up
`summary_<benchmark>_last_run.json` at the `--out-dir` root (this run's cells
only) and (re)writes the consolidated `summary_all.json` + `summary_all.md`
across every cell on disk. `glyph markush eval baseline-report --out-dir
<out-dir>` rebuilds `summary_all.{json,md}` on demand.

## Resuming

Re-run the exact same command. `load_checkpoint` reads the existing
`<variant>.checkpoint.jsonl`, treats rows with an empty `error` as done, and
only calls the API for the rest. Completed predictions are never lost, and
transient (rate-limit / transport) failures are retried on the next pass.

## Prompt variants

- `default` sends the exact MarkushGlyph `visible_sg` prompt, byte-for-byte.
- `format` additionally spells out the target `<markush><cxsmi>…</cxsmi><stable>…</stable></markush>`
  format and instructs the model to return only that. Use it when a model's
  free-form answers are largely unparseable; report both to see the lift.
