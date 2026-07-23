# `glyph data prepare` — release data pipeline

`glyph data prepare` pulls the public OCSR and Markush sources, preprocesses
them, and lays them out under `$GLYPH_DATA_ROOT` in the approved layout. The
library holds all logic; the Typer command (`glyph/data/cli.py`) is a thin
wrapper over `glyph.data.ocsr.prepare_ocsr` and `glyph.data.markush.prepare_markush`.

```
$GLYPH_DATA_ROOT/
  datasets/ocsr/{pubchem_1m, uspto_680k, stereo_200k_white, adjacent_ring}/
    <data files> + provenance.json
  datasets/markush/{mg1_synthetic, mg2_structure_only, molparser, ordinary_5pct}/
    unified.jsonl + provenance.json
  images/{mg1_synthetic, uspto_mol_m, molparser, molscribe}/train/<source_id>.png
  markush_train.jsonl
  benchmarks/ocsr/uspto_ocsr_5719_complete.csv (+ .provenance.json, provenance.json)
  benchmarks/markush/{ip5m, m2s, uspto_markush}/eval_*.jsonl + provenance.json
```

Every materialized dataset directory carries a `provenance.json` sidecar so a
reader can attribute the source without reading code. Attributions are
authoritative — external data is never labelled as an original upload.

## OCSR

```bash
export GLYPH_DATA_ROOT=/path/to/data

# Plan only — no network, no writes:
glyph data prepare --ocsr --dry-run

# Small validation slice (streams a PubChem head, mines adjacent_ring from it,
# builds the 5719 benchmark; multi-GB sources are still refused):
glyph data prepare --ocsr --limit 3000

# Full run (permits the multi-GB downloads):
glyph data prepare --ocsr --allow-large
```

The pipeline is idempotent: an already-materialized artifact is reported
`skipped_exists` and left untouched (`--force` re-materializes). Multi-GB source
downloads are **refused** unless `--allow-large` is passed (`--dry-run` plans
them, `--limit` samples what it can). `--data-root` overrides `$GLYPH_DATA_ROOT`.

The two CPU-bound steps — the Stereo-200K white-crop and the adjacent-ring
PubChem scan — run on a process pool. `--workers/-j N` sets the pool size
(default `min(cpu_count, 8)`); `-j 1` forces the serial path. Output is
byte-identical regardless of the worker count (same image bytes, same rewritten
CSV row order/content), and each step stays idempotent and resumable.

### Steps, sources, and expected full-run counts/sizes

| Artifact | Source (pinned) | Download | Rows | Attribution |
|---|---|---|---|---|
| `datasets/ocsr/pubchem_1m/pubchem_1m.csv` | `CYF200127/MolNexTR` `train_pubchem.csv` @ `9ac2da6f` | ~210 MB | 1,000,000 | External, MolNexTR public. Rendered on the fly (Indigo) at train time. |
| `datasets/ocsr/uspto_680k/` | `CYF200127/MolNexTR` `train_uspto.zip` @ `9ac2da6f` | ~2.1 GB | 680,220 | External, MolNexTR public. Images + CSV. |
| `datasets/ocsr/stereo_200k_white/` | `Robert-zwr/Stereo-200k` `stereo.zip` @ `f6cdec49` | ~4.3 GB | 200,000 | **External** (MolSight, cite `zhang2026molsight`). Only the white-crop preprocessing is ours; SMILES identical to source. Not our upload. |
| `datasets/ocsr/adjacent_ring/adjacent_ring.csv` | derived from `pubchem_1m` | 0 (local) | 64,752 | **This work.** Pure-logic adjacent-ring-stereo subset of PubChem-1M; not standalone. |
| `benchmarks/ocsr/uspto_ocsr_5719_complete.csv` | `hheiden/USPTO_OCSR_benchmark` @ `32379850` (images) | ~23 MB | 5,719 | Benchmark label sidecar (5704 upstream-labelled + 13 recovered + 2 retained unlabelled); `glyph ocsr eval` reads the frozen self-contained parquet directly (see note). |

Notes:

- **adjacent_ring** requires `pubchem_1m` to be present first (it mines
  the local CSV). At the defaults (`min_len=50`, `max_len=180`) the full scan
  keeps 64,752 / 1,000,000 rows (6.48%), matching the frozen enrichment CSV.
- **stereo_200k_white** downloads and extracts the external `stereo.zip` into a
  `_source/` subdirectory, then white-crops each image
  (`glyph.data.ocsr_white_crop.white_crop_image`) into `train/` and rewrites the
  CSV `file_path` column. The `_source/` tree is left in place (not deleted).
- **uspto_680k** downloads and extracts the upstream archive, then normalizes
  the nested `train_uspto/` tree to the documented layout: the CSV becomes
  `datasets/ocsr/uspto_680k/train.csv` and the image subtree moves under
  `datasets/ocsr/uspto_680k/images/` (CSV `file_path` values keep their
  `uspto_mol/` prefix, stripped at load time so `uspto_mol/<year>/…` resolves to
  `images/<year>/…`). This makes `scripts/train_ocsrglyph.sh` resolve
  `USPTO_CSV` / `USPTO_IMAGE_ROOT` / `USPTO_PATH_STRIP_PREFIX` from their
  defaults with no env overrides; the same normalization repairs an
  un-normalized extract in place (no re-download).
- **5719 benchmark (eval).** `glyph ocsr eval` downloads one frozen,
  self-contained parquet (`ocsr/uspto_ocsr_benchmark.parquet`) from
  `EdisonScientific/glyph-datasets` — the public `hheiden/USPTO_OCSR_benchmark`
  images joined to the complete SMILES labels, with the MolScribe null-label
  recovery already baked in (5704 upstream-labelled + 13 recovered + 2 retained
  unlabelled), so results include that recovery.
  Rebuilding the CSV with `glyph data prepare --ocsr --allow-large` fetches the
  MolScribe `real/USPTO.csv` recovery archive automatically; see
  [`docs/reproducing.md`](../../docs/reproducing.md).

### Library entry points

- `glyph.data.ocsr.prepare_ocsr(...)` — orchestrator returning a `PrepareReport`.
- `glyph.data.ocsr_sources` — network fetchers (with the large-download guard).
- `glyph.data.ocsr_white_crop` — Stereo-200K white-crop preprocessing.
- `glyph.data.ocsr_adjacent_ring` — adjacent-ring-stereo mining.
- `glyph.ocsr.benchmark.build_uspto5719_benchmark(...)` — the 5719 benchmark.
- `glyph.data.ocsr_eval_jsonl.materialize_ocsr_eval(...)` — task-conditioned OCSR
  eval JSONL (multitask flavour; not part of the core prepare path).

## Markush

The released-model data flow has two independent inputs: a frozen flat index
and four public image sources. Prepare the images first:

```bash
export GLYPH_DATA_ROOT=/path/to/data
glyph data prepare --markush --allow-large
```

Then download `markush/train.jsonl` from `EdisonScientific/glyph-datasets` as
`$GLYPH_DATA_ROOT/markush_train.jsonl`. Do not rebuild or reshuffle this file.
Its 263,158 rows preserve the exact training order and oversampled repeats, and
each row has only `source`, `source_id`, `images`, and `messages`.

Images use one uniform, source-id-keyed layout:

```
$GLYPH_DATA_ROOT/images/<source>/train/<source_id>.png
```

The four `source` values and row counts are:

| Source | Rows | Public source revision |
|---|---:|---|
| `mg1_synthetic` | 152,620 | `docling-project/MarkushGrapher-Datasets` @ `a6cf981dd100198ce87e8099402f0645b44080ed` |
| `uspto_mol_m` | 52,524 | `docling-project/MarkushGrapher-2-Datasets` @ `e90868cdd3b62f1fe175d9ede9e0c65485c4a96c` |
| `molparser` | 44,856 | `UniParser/MolParser-7M` @ `f37e164f179c7cd8470fc879422c29af2c47874c` |
| `molscribe` | 13,158 | `yujieq/MolScribe` @ `a0189776b7415b82795c7ee81eed311bf5c8724b` |

The MG1 and MolParser pins are current-main commits, recorded explicitly so
source preparation stays deterministic. MolScribe pins a specific archived
snapshot. MolScribe TIF files are decoded to RGB pixels and saved as PNG; the
PNG stores the same pixels the vision pipeline sees, in a different container.

### Only the referenced images are materialized

`glyph data prepare --markush --markush-only sources` materializes only the
images the flat index references — its 263,158 rows resolve to 231,537 unique
images, not the full upstream trees (which total roughly a million images --
e.g. MolScribe ships ~680K to supply the 13,158 rows the index uses). The index
is resolved the same way the
trainer resolves `--data-path` (`hub:markush/train.jsonl` by default: a local
mirror under `$GLYPH_MARKUSH_TRAIN_ROOT`, a `$GLYPH_DATA_ROOT` snapshot, then a
Hugging Face download). Pass `--markush-index <path|hub:...>` to point at a
specific index. The materialized PNG bytes and unified JSONL rows are identical
to a full pull for every kept image; only never-referenced images are skipped.
A `source_id` the index references but the upstream source lacks is surfaced
(warning + count), never silently dropped. If the index cannot be resolved
(e.g. offline with no snapshot), the run falls back to materializing the full
sources with a warning. `--limit N` keeps its prefix-slice validation behavior
(no index subsetting).

Use `--limit N` with the preparation command for a small validation pull. The
full source trees are about 32 GB and therefore require `--allow-large`.

### Running eval on the emitted benchmarks (`--image-root`)

Build the three small DS4SD eval JSONLs (no multi-GB sources, no `--allow-large`)
with the benchmark-only selector:

```bash
glyph data prepare --markush --markush-only benchmarks
```

The emitted `benchmarks/markush/{ip5m,m2s,uspto_markush}/eval_*.jsonl` carry
`$GLYPH_DATA_ROOT`-relative image refs (`images/<benchmark>/<split>/…`) — no
absolute paths. The eval loader must therefore
be pointed at `$GLYPH_DATA_ROOT` as its image root so those refs resolve:

```bash
python -m glyph.markush.eval.fast_eval \
    --eval-data "$GLYPH_DATA_ROOT/benchmarks/markush/ip5m/eval_ip5m.jsonl" \
    --image-root "$GLYPH_DATA_ROOT" \
    ...  # --checkpoint / --base-model / --output as usual
```

### Train from the flat index

```bash
python -m glyph.markush.train.sft \
  --config configs/ablations/ocsr-as-markush-fraction/sft_2b_markush_ocsr5pct.yaml \
  --data-path "$GLYPH_DATA_ROOT/markush_train.jsonl"
```

The config supplies the released hyperparameters. The `--data-path` override
selects the frozen flat index.

### Library entry points

- `glyph.data.markush.prepare_markush(...)` prepares the pinned sources.
- `glyph.data.markush_sources` contains the guarded source importers.
- `glyph.markush.train.sft.MarkushDataset` resolves flat-index image paths
  against `$GLYPH_DATA_ROOT`.
