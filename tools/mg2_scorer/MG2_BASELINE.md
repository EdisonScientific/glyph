# Reproducing the MarkushGrapher-2 Table 3 baseline

`reproduce_mg2_baseline.sh` runs the public MarkushGrapher-2 (MG2) baseline on
IP5-M, M2S, and USPTO-Markush without changing its OCR, beam-search inference,
or evaluation routines. It then grades each fresh prediction file with the two
release contracts and compares every MG2 output row with the cached cross-check
shipped here.

```bash
tools/mg2_scorer/reproduce_mg2_baseline.sh
```

The full run processes **878 + 103 + 74 = 1,055 images**. On an RTX 5090 it uses
approximately 12 GB for MG2 inference and up to 32 GB for vLLM/ChemicalOCR.
Expect roughly 45 minutes for inference after the initial environment and model
downloads. The command uses only the local GPU. Its checkout, environments,
weights, datasets, and results live under
`tools/mg2_scorer/local-mg2-reproduction/`, which is ignored by git.

To resume a completed checkout or re-grade existing predictions:

```bash
tools/mg2_scorer/reproduce_mg2_baseline.sh --skip-setup
tools/mg2_scorer/reproduce_mg2_baseline.sh \
  --skip-setup --skip-inference --work-dir /path/to/run
```

Use `--benchmark ip5m` (repeatable) for a subset. The final all-benchmark report
is `runs/summary.json`. Each benchmark directory contains the raw MG2
`predictions_<N>.jsonl`, `scores.json`, `scores_strict.json`, and
`comparison.json`.

## Published pipeline boundary

The launcher calls the following from the pinned DS4SD checkout:

1. `scripts/dataset/prepare_hf_dataset.py` to load the pinned benchmark;
2. `scripts/ocr/apply_ocr.py` and `Chemical_OCR.predict` for OCR;
3. `python -m markushgrapher.eval` with MG2's published beam-search config;
4. `tools/mg2_scorer/score_benchmark.sh`, which delegates official grading to
   DS4SD's unmodified `get_scores` and strict grading to the release grader.

`mg2_reproduction.py` is deliberately limited to orchestration seams. It
rewrites Hugging Face feature metadata between the two required `datasets`
versions, wraps MG2's optimized prediction in the input envelope expected by
MG2's own `convert_opt_to_out`, compares JSONL artifacts, and formats the final
counts. It does not implement OCR, decoding, CXSMILES conversion, or either
chemical metric.

## Pins

| Component | Pin used by the launcher |
|---|---|
| DS4SD/MarkushGrapher source | `35c71a0b4423983920b4703f52f46c5bca9ae631` |
| DS4SD/MarkushGenerator | `17580c20d266886ef9ae22a50aab51444e0898b2` |
| lucas-morin/transformers fork | `867ac6e128e2e71ecc12a9c1b65e48e451245a67` |
| lucas-morin/MolScribe source | `b15c96e10da5e2ec4444a877d1031d8d10190c4d` |
| `docling-project/MarkushGrapher-2-Datasets` | `e90868cdd3b62f1fe175d9ede9e0c65485c4a96c` |
| `docling-project/MarkushGrapher-2` weights | `2e1091d5acad0046910971439cedced0df659367` |
| `docling-project/ChemicalOCR` weights | `650c1c8e6ea7699ca1f1049d30490b89fab538f2` |
| `yujieq/MolScribe` checkpoint snapshot | `a0189776b7415b82795c7ee81eed311bf5c8724b` |

The source checkouts are verified before every inference run. Hugging Face
snapshots are requested by immutable revision. Model bytes and environments are
never committed.

## RTX 5090 / Blackwell compatibility

MG2's April 2026 setup script pins PyTorch 2.2 / CUDA 12.1, which has no `sm_120`
kernels. The local recipe keeps MG2's source and inference logic pinned while
updating the runtime only:

- ChemicalOCR: Python 3.10, PyTorch 2.11.0 CUDA 13.0, vLLM 0.25.1,
  transformers 5.14.1, tokenizers 0.22.2, datasets 5.0.0, NumPy 1.26.4.
- MG2: Python 3.10, PyTorch 2.13.0 CUDA 13.0, the pinned custom transformers
  fork (4.34.0.dev0), tokenizers 0.13.3, datasets 2.14.6, pyarrow 12.0.1,
  accelerate 0.21.0, OpenNMT-py 2.2.0, and NumPy 1.24.4.
- PyTorch 2.13 removed the private `torch._utils._accumulate` symbol imported by
  MG2. The setup adds an `itertools.accumulate` import fallback in the local,
  ignored checkout. This is the standard-library equivalent iterator and does
  not change model behavior.
- ChemicalOCR's datasets 5 output describes nested cells with `List` feature
  metadata; MG2's datasets 2.14 cannot deserialize that type. The helper copies
  the table to an independent DatasetDict and changes both `dataset_info.json`
  and every Arrow shard's embedded feature metadata to the equivalent legacy
  `Sequence` representation. Cell values, images, IDs, and annotations are
  preserved.

Both environments assert CUDA visibility and print their runtime version and
compute capability before a run.

## Local RTX 5090 results (2026-07-24)

Fresh ChemicalOCR and MG2 beam-search predictions were generated on one NVIDIA
RTX 5090 (32 GB, compute capability 12.0, driver 595.71.05). Full-denominator
results from `score_benchmark.sh` were:

| Benchmark | DS4SD official | Strict parsed graph | Fresh vs cached rows |
|---|---:|---:|---:|
| IP5-M (878) | 465 / 878 = **52.96%** | 313 / 878 = **35.65%** | 821 / 878 exact |
| M2S (103) | 55 / 103 = **53.40%** | 51 / 103 = **49.51%** | 93 / 103 exact |
| USPTO-Markush (74) | 32 / 74 = **43.24%** | 32 / 74 = **43.24%** | 69 / 74 exact |

The official paper reports 53.7%, 56.0%, and 55.0%. For IP5-M, its 53.7% uses
the 870 rows the official grader can parse; the fresh run is 465 / 870 = 53.45%
on that effective denominator. This release keeps all eight invalid GT rows in
the headline denominator, giving 52.96%. The released checkpoint closely but
not exactly reproduces M2S and substantially under-reproduces the published
USPTO-Markush scalar. No rows were dropped, substituted, or tuned. The cached
artifacts in `cached_predictions/` are earlier runs of the same released public
checkpoint and are retained to expose run-to-run/runtime drift, not to replace
fresh predictions.

All 1,055 fresh and cached IDs were in identical order, and all ground-truth
fields matched. Prediction-field agreement was: IP5-M 830 / 878 standard and
821 / 878 optimized CXSMILES; M2S 93 / 103 for both; USPTO-Markush 69 / 74 for
both. The differing predictions are enumerated by index, ID, and field in each
`comparison.json`.

The cached JSONL files contain only MG2's public output schema (`id`, predicted
standard/optimized CXSMILES, and corresponding GT fields). Their provenance is
the same pinned public source, weights, datasets, and published pipeline listed
above. `comparison.json` reports ID order, exact full-row matches, matches per
field, and every differing row index/field.
