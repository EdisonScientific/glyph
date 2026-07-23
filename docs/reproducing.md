# Reproducing the paper numbers

The [README](../README.md#results) gives the short version: one command for the OCSR benchmark, two commands for the Markush benchmarks. This page holds the detail behind those commands: pinned dataset revisions, the exact scoring contracts, the manual step-by-step Markush scoring flow, and the provenance of the DS4SD ground-truth files.

Related documents:

- [`tools/mg2_scorer/MG2_BASELINE.md`](../tools/mg2_scorer/MG2_BASELINE.md) — regenerating the MarkushGrapher-2 baseline end to end (ChemicalOCR, MG2 beam search, both graders).
- [`tools/frontier_baseline/README.md`](../tools/frontier_baseline/README.md) — the frontier-model baselines (hosted models via litellm, same prompts and scoring contracts).

All commands assume the data root is set:

```bash
export GLYPH_DATA_ROOT=/path/to/data
```

## Pinned dataset revisions

Source preparation is deterministic against the revisions below (recorded in `glyph/data/layout.py`).

| Purpose | HF dataset | Config / split | Revision |
| --- | --- | --- | --- |
| Markush train (MG1) | `docling-project/MarkushGrapher-Datasets` | `markushgrapher-synthetic-training` / `train` | `a6cf981dd100198ce87e8099402f0645b44080ed` |
| Markush train (USPTO-MOL-M) and eval (IP5-M) | `docling-project/MarkushGrapher-2-Datasets` | `uspto-mol-m-54k` / `train`; `ip5-markush` / `test` | `e90868cdd3b62f1fe175d9ede9e0c65485c4a96c` (IP5-M fingerprint `15733712f9c574af`, 878 rows) |
| Markush train (MolParser) | `UniParser/MolParser-7M` | `sft_real` / `train` | `f37e164f179c7cd8470fc879422c29af2c47874c` |
| Markush train (MolScribe) | `yujieq/MolScribe` | `uspto_mol.zip` / `train_680k.csv` | `a0189776b7415b82795c7ee81eed311bf5c8724b` |
| OCSR benchmark | `hheiden/USPTO_OCSR_benchmark` | `train` | `32379850` |
| OCSR train (public raw) | `CYF200127/MolNexTR` | PubChem-1M + USPTO-680K | `9ac2da6f` |
| OCSR train (stereo) | `Robert-zwr/Stereo-200k` | `stereo.zip` | `f6cdec49` |

The MG1 and MolParser pins are the commits that were current on each repo's `main` when we imported them; they are not upstream release tags, and those repos may advance.

The model weights are a softer case. `EdisonScientific/OCSRGlyph`, `EdisonScientific/MarkushGlyph`, and the base `Qwen/Qwen3.5-2B-Base` are fetched from Hugging Face `main` without a revision pin, so they track upstream. If you need bit-exact weights, download a specific revision and pass the local path.

## 1. OCSRGlyph on USPTO-OCSR-5719

### The benchmark

The eval pulls one frozen, self-contained parquet (`ocsr/uspto_ocsr_benchmark.parquet` on `EdisonScientific/glyph-datasets`): the 5,719 public `hheiden/USPTO_OCSR_benchmark` images joined to complete SMILES labels. The upstream benchmark labels only 5,704 rows; of the remaining 15, 13 labels were recovered from the MolScribe `real/USPTO.csv` release and 2 stay unlabelled. The `source_of_label` column records per-row provenance. 5,711 rows are RDKit-scoreable; 8 are not.

The full 5,719 denominator is strict: parse failures and unlabelled rows count as wrong. The 5,704-row labelled subset is the robustness reference.

### Scoring contract

- **Decoding:** greedy (argmax to EOS), max decode length 256.
- **Postprocess (default `light`):** disconnected `[H]` / `[HH]` hydrogen fragments are stripped, then RDKit re-canonicalizes. `--postprocess none` disables it.
- **Conventions** (defined in `glyph/ocsr/eval.py`): A is RDKit isomeric canonical-SMILES equality; B strips `/` and `\` then canonicalizes with tetrahedral chirality kept (mirroring MolScribe's `canon_smiles`); C removes all stereochemistry and compares connectivity only.
- **Image loading:** resized to 384x384 (RGB, bilinear), normalized to `[-1, 1]` via `(x - 0.5) / 0.5`.

### Commands

The benchmark downloads automatically on first use; no data preparation is needed.

```bash
uv run glyph ocsr eval \
    --checkpoint EdisonScientific/OCSRGlyph \
    --batch-size 32 \
    --per-row-out "$GLYPH_DATA_ROOT/eval_out/ocsr/uspto5719_perrow.jsonl" \
    --output "$GLYPH_DATA_ROOT/eval_out/ocsr/uspto5719.json"
```

Expected numbers, over the full 5,719-row benchmark and over the 5,704 rows with upstream labels:

| Convention | Full 5,719 | Labelled 5,704 |
| --- | --- | --- |
| A: canonical (full stereo) | 5,362 = 93.8% | 5,360 = 93.97% |
| B: chirality kept | 5,372 = 93.9% | 5,370 = 94.14% |
| C: graph (connectivity only) | 5,499 = 96.2% | 5,496 = 96.35% |
| valid SMILES | — | 5,692 = 99.79% |

Lower `--batch-size` if you hit out-of-memory errors; decoding is greedy, so the numbers do not change. To score a custom benchmark, pass `--benchmark` (a complete-label CSV with `image_id, smiles, source_of_label` columns) or `--benchmark-parquet`.

Majority voting is opt-in and reported separately from the greedy headline:

```bash
uv run glyph ocsr eval \
    --checkpoint EdisonScientific/OCSRGlyph \
    --batch-size 32 --majority-vote --vote-k 8
```

## 2. MarkushGlyph training data

We release the exact 263,158-row training sequence as a flat JSONL index (`hub:markush/train.jsonl` on `EdisonScientific/glyph-datasets`). Do not regenerate, repartition, or reshuffle it: repeated rows are intentional and preserve the training order. Expected per-source counts:

| Source | Rows |
| --- | --- |
| `mg1_synthetic` | 152,620 |
| `uspto_mol_m` | 52,524 |
| `molparser` | 44,856 |
| `molscribe` (standard molecules, 5%) | 13,158 |

Each record holds only `source`, `source_id`, `images`, and `messages`. Image paths follow `images/<source>/train/<source_id>.png` and resolve under `$GLYPH_DATA_ROOT`; `glyph data prepare --markush --allow-large` materializes them (231,537 unique images, since repeated rows share images). Subsetting to the index applies only on a full run where the index resolves: a `--limit N` run materializes a prefix slice per source instead, and a `--dry-run` plans without subsetting, so it reports the full ~32 GB source tree.

The paper-faithful SFT run computes the LM loss with FLA's fused Triton cross-entropy, which avoids materializing the fp32 `[batch, seq, vocab]` logits copy. That kernel is part of the training-only `train` extra (`uv sync --extra all`, or `uv sync --extra markush --extra train`). Without it, the trainer warns once and falls back to standard cross-entropy: the loss is identical, but the memory use is higher, so a smaller batch size may be needed.

## 3. MarkushGlyph evaluation

Expected accuracy under the official contract, matching the paper:

| Benchmark | Greedy | MV@8 |
| --- | --- | --- |
| IP5-M | 58.2% (511/878) | 60.6% (532/878) |
| M2S | 61.2% | 62.1% |
| USPTO-Markush | 59.5% | 63.5% |

The MarkushGrapher-2 baseline re-scores to 53.2% (467/878) on IP5-M under the same contract. (Exact M2S and USPTO-Markush row counts are being reconfirmed against a fresh eval run.)

### Scoring contract

- **Denominator = 878** for IP5-M: every `test` row at the pinned revision. A prediction that cannot be scored counts as incorrect for every system. Do not report `467/870 = 53.7%`; the denominator is 878.
- **One pinned scorer for all systems:** DS4SD MarkushGrapher's public `markushgrapher.utils.ocsr.utils_evaluation.get_scores` (the MG2 `cxsmi_equality` grader), used unmodified.
- **8 unscorable ground-truth rows:** the DS4SD ground truth has 8 rows whose CXSMILES `get_scores` cannot parse. They stay in the denominator and count as incorrect for every system.
- **Prediction conversion:** MarkushGlyph output is converted to the official format by extracting `<cxsmi>`, removing spaces, decoding the inline form with `CXSMILESTokenizer(training_dataset="mdu", condense_labels=True).convert_opt_to_out`, then `fix_cxsmiles`. A prediction that fails to convert or parse becomes `None` (incorrect). MG2's own predictions are scored as supplied.

We provide two graders, and `score_benchmark.sh` reports both against the same ground truth:

- **MG2 `cxsmi_equality`** (pinned DS4SD code in `tools/mg2_scorer/`) is the headline grader. It reads the ground truth's `gt_cxsmiles` column.
- **Strict parsed-graph equality** (`glyph.markush.eval`, also available standalone via `glyph markush eval checkpoint`) is a VF2 graph-isomorphism metric graded against the model-format `gt_cxsmiles_opt` column. It is stricter than the MG2 contract and not a substitute for it. IP5-M greedy scores 52.1% here.

### Generation recipe

The two-command flow in the README (`glyph markush eval benchmark` + `score_benchmark.sh`) bakes in the frozen greedy recipe: `--markush-prompt-policy visible_sg`, `--max-new-tokens 1024`, `--seed 20260625`, single image view, greedy decode. The steps below run the same pipeline by hand, which you need for MV@8 or custom decoding knobs.

**Greedy predictions:**

```bash
uv run python -m glyph.markush.eval.fast_eval \
    --checkpoint EdisonScientific/MarkushGlyph \
    --base-model Qwen/Qwen3.5-2B-Base \
    --eval-data "$GLYPH_DATA_ROOT/benchmarks/markush/ip5m/eval_ip5m.jsonl" \
    --image-root "$GLYPH_DATA_ROOT" \
    --output "$GLYPH_DATA_ROOT/eval_out/markush/ip5m_greedy.json" \
    --batch-size 8 --max-new-tokens 1024 --num-gpus 1 --protocol ours \
    --prompt "Extract the Markush structure from this image." \
    --multi-image-mode single --skip-mcs --quiet \
    --markush-prompt-policy visible_sg --seed 20260625
```

**MV@8 predictions** (majority vote over 8 sampled candidates):

```bash
uv run python -m glyph.markush.eval.fast_eval \
    --checkpoint EdisonScientific/MarkushGlyph \
    --base-model Qwen/Qwen3.5-2B-Base \
    --eval-data "$GLYPH_DATA_ROOT/benchmarks/markush/ip5m/eval_ip5m.jsonl" \
    --image-root "$GLYPH_DATA_ROOT" \
    --output "$GLYPH_DATA_ROOT/eval_out/markush/ip5m_mv8.json" \
    --max-new-tokens 1024 --num-gpus 1 --protocol ours \
    --prompt "Extract the Markush structure from this image." \
    --multi-image-mode single --skip-mcs --quiet \
    --markush-prompt-policy visible_sg \
    --do-sample --temperature 0.7 --top-p 0.95 \
    --num-return-sequences 8 --majority-vote --batch-size 1 --seed 20260625
```

The `eval_ip5m.jsonl` input is written by `glyph data prepare --markush`; `--image-root` resolves its data-root-relative image references.

### Building the scorer environment

The official scorer needs its own environment, separate from the base `uv.lock`. `tools/mg2_scorer/requirements-public-mg2.txt` pins the load-bearing numerics (`rdkit==2024.3.5`, `numpy==1.24.4`, `torch==2.2.0`, `torchvision==0.17.0`); the remaining dependencies resolve to current releases.

```bash
tools/mg2_scorer/fetch_mg2_scorer.sh   # clones DS4SD repos at the pinned commits (idempotent)

uv venv --python 3.11 tools/mg2_scorer/.venv-mg2
uv pip install --python tools/mg2_scorer/.venv-mg2/bin/python \
    -r tools/mg2_scorer/requirements-public-mg2.txt
```

`./setup.sh` runs `fetch_mg2_scorer.sh` for you. The pinned commits are listed in `tools/mg2_scorer/VENDOR_COMMITS.md`.

### Rescoring under the official contract

The `official` contract is the default and reproduces the headline numbers. It scores against the DS4SD ground truth (next section), resolved automatically:

```bash
GT="$(uv run python -m glyph.markush.eval.benchmark_data ip5m --gt)"

PYTHONPATH="$PWD" tools/mg2_scorer/.venv-mg2/bin/python \
    tools/mg2_scorer/rescore_with_public_mg2.py \
    --contract official \
    --input-json "$GLYPH_DATA_ROOT/eval_out/markush/ip5m_greedy.json" \
    --gt-jsonl "$GT" \
    --output-json "$GLYPH_DATA_ROOT/eval_out/markush/ip5m_greedy_scored.json"
```

Run the same command with `ip5m_mv8.json` for the MV@8 numbers. The accuracy is the `public_mg2.exact_counts.cxsmi_equality_*` block in the output: 511/878 greedy, 532/878 MV@8.

### The DS4SD ground truth

The official contract scores against DS4SD's own ground truth, not the ground truth inside the prepared `eval_ip5m.jsonl` (which has zero `None` rows and cannot reproduce the 878-row contract). The file is `ds4sd_mg2_predictions_878.jsonl`: MarkushGrapher-2's own IP5-M evaluation output, where each row carries an `id`, the DS4SD ground-truth CXSMILES (`gt_cxsmiles`, with the 8 unparseable rows kept as `None`), and MG2's own prediction (`cxsmiles`). The rescorer joins it to your predictions by `id`.

Two ways to obtain it:

1. **Auto-resolve (recommended).** `uv run python -m glyph.markush.eval.benchmark_data ip5m --gt` prints the resolved path, downloading `markush-eval/ip5m/ds4sd_mg2_predictions_878.jsonl` from `EdisonScientific/glyph-datasets` on first use. This is what `score_benchmark.sh` uses.
2. **Regenerate it** by running DS4SD MarkushGrapher-2's official evaluation on the IP5-M `test` split at the pinned revision `e90868c...`. The vendored checkout fetched by `fetch_mg2_scorer.sh` provides that pipeline; it writes `evaluation/predictions_878.jsonl` with identical content.

The same file supplies the MarkushGrapher-2 baseline number (467/878): its stored `cxsmiles` predictions, scored through the same pinned `get_scores`.

**M2S and USPTO-Markush.** No DS4SD model run is published for these benchmarks, so we provide derived ground-truth bundles on `EdisonScientific/glyph-datasets` instead (`markush-eval/m2s/ds4sd_mg2_gt_103.jsonl`, `markush-eval/uspto_markush/ds4sd_mg2_gt_74.jsonl`). Each row is `id`, `gt_cxsmiles`, `gt_cxsmiles_opt`, where `gt_cxsmiles` is the pinned dataset annotation run through the same official conversion applied to predictions, so ground truth and prediction share one contract. `score_benchmark.sh --benchmark <name>` selects the right bundle automatically. Because these bundles carry no MG2 prediction column, there is no MarkushGrapher-2 baseline for those two benchmarks; the numbers are self-consistent under the official contract rather than compared against a published target.
