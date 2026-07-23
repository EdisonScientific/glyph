# Cached MarkushGrapher-2 prediction cross-checks

These three JSONL files are earlier outputs of DS4SD's released
MarkushGrapher-2 checkpoint, retained so the local reproduction can perform a
row-for-row comparison rather than checking aggregate scores alone.

| File | Rows | Benchmark |
|---|---:|---|
| `mg2_ip5m_predictions_878.jsonl` | 878 | IP5-M |
| `mg2_m2s_predictions_103.jsonl` | 103 | M2S |
| `mg2_uspto-markush_predictions_74.jsonl` | 74 | USPTO-Markush |

Each row is MG2's own public evaluation schema: `id`, `cxsmiles`,
`cxsmiles_opt`, `gt_cxsmiles`, and `gt_cxsmiles_opt`. The files contain no
weights, images, environments, or private inputs. They were produced with the
public pipeline and immutable pins documented in
[`../MG2_BASELINE.md`](../MG2_BASELINE.md), on the same public benchmark IDs and
order used by the fresh run.

The predictions are cross-checks, not inference inputs. The reproduction always
runs ChemicalOCR and MG2 first, grades those fresh outputs, and only then invokes
`mg2_reproduction.py compare`. Runtime upgrades needed for new GPU
architectures can produce a small number of different decoded rows; the report
therefore records exact rows, per-field matches, and every differing row rather
than requiring byte identity.
