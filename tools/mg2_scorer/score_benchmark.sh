#!/usr/bin/env bash
# Both-contract MG2 scoring for one MarkushGlyph benchmark (STEP 2, turnkey).
#
# Reports the paper's two columns for a single (benchmark, predictions.json):
#   - "MarkushGrapher-2 score": the official DS4SD get_scores exact match, and
#   - "Strict parsed-graph equality": our stricter graph-isomorphism grader.
#
# Given a predictions JSON from STEP 1
# (`glyph markush eval benchmark --benchmark <name> ... --output preds.json`),
# this:
#   1. runs the idempotent fetch of the pinned DS4SD scorer sources
#      (fetch_mg2_scorer.sh) and verifies both checkouts are at the pinned
#      commits (a checkout at another commit is corrected, not accepted),
#   2. builds the pinned py3.11 scorer venv (tools/mg2_scorer/.venv-mg2) if missing,
#   3. auto-pulls the benchmark's ground truth from EdisonScientific/glyph-datasets
#      (via glyph.markush.eval.benchmark_data.resolve_benchmark_gt),
#   4. runs rescore_with_public_mg2.py --contract official against it (DS4SD
#      get_scores used UNMODIFIED; see VENDOR_COMMITS.md / docs/reproducing.md), and
#   5. runs the strict parsed-graph grader (glyph.markush.eval.strict_score,
#      reusing glyph.markush.grading, protocol="ours") against the SAME GT,
# then prints both headline counts side by side.
#
# The strict grader needs RDKit (the base uv `markush`/`ocsr` extra). If it is
# not importable the script still reports the official number and warns.
#
# Usage:
#   tools/mg2_scorer/score_benchmark.sh --benchmark <name> --input-json <preds.json> \
#       [--output-json <scored.json>] [--gt-jsonl <override.jsonl>]
#
#   <name> is ip5m, m2s, or uspto_markush.
#   --output-json defaults to <input-json stem>_scored.json next to the input;
#     the strict report is written next to it as <stem>_strict.json.
#   --gt-jsonl overrides the auto-pulled ground truth (offline / custom GT).
#   GLYPH_MARKUSH_EXTRA overrides the base uv extra used for the strict grader
#     (default: markush).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PY="${SCRIPT_DIR}/.venv-mg2/bin/python"

BENCH=""
INPUT_JSON=""
OUTPUT_JSON=""
GT_OVERRIDE=""

usage() {
  cat <<'EOF'
score_benchmark.sh - both-contract MG2 scoring for one MarkushGlyph benchmark.

Usage:
  tools/mg2_scorer/score_benchmark.sh --benchmark <name> --input-json <preds.json> \
      [--output-json <scored.json>] [--gt-jsonl <override.jsonl>]

  --benchmark    ip5m, m2s, or uspto_markush.
  --input-json   predictions JSON from `glyph markush eval benchmark` (STEP 1).
  --output-json  official scored output (default: <input-json stem>_scored.json).
                 The strict report is written as <stem>_strict.json alongside it.
  --gt-jsonl     override the auto-pulled ground truth (offline / custom GT).

Fetches the pinned DS4SD scorer, builds the pinned py3.11 scorer venv, auto-pulls
the benchmark ground truth from EdisonScientific/glyph-datasets, then reports both
paper columns against that one ground truth: the UNMODIFIED DS4SD get_scores exact
match and our strict parsed-graph equality. The strict grader needs RDKit (base uv
`markush`/`ocsr` extra); without it the official number is still reported.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)   BENCH="$2"; shift 2 ;;
    --input-json)  INPUT_JSON="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --gt-jsonl)    GT_OVERRIDE="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$BENCH" || -z "$INPUT_JSON" ]]; then
  echo "ERROR: --benchmark and --input-json are required." >&2
  echo >&2; usage >&2; exit 2
fi
if [[ ! -f "$INPUT_JSON" ]]; then
  echo "ERROR: predictions file not found: $INPUT_JSON" >&2; exit 2
fi
if [[ -z "$OUTPUT_JSON" ]]; then
  OUTPUT_JSON="${INPUT_JSON%.json}_scored.json"
fi

cd "$ROOT"

# 1. Pinned DS4SD scorer sources. Always run the idempotent fetch so an existing
#    checkout sitting at a *different* commit is corrected to the pinned commit
#    rather than silently accepted. The fetch skips the network when a checkout
#    is already at the pinned commit, then we verify both HEADs before scoring.
echo "[score] ensuring pinned DS4SD scorer sources (idempotent fetch)..."
"${SCRIPT_DIR}/fetch_mg2_scorer.sh"

# Verify both checkouts are exactly at the pinned commits (source of truth:
# tools/mg2_scorer/VENDOR_COMMITS.md) before scoring against them.
MARKUSHGRAPHER_COMMIT="35c71a0b4423983920b4703f52f46c5bca9ae631"
MARKUSHGENERATOR_COMMIT="17580c20d266886ef9ae22a50aab51444e0898b2"
verify_pinned_commit() {
  local name="$1" want="$2" dest="${SCRIPT_DIR}/vendor/$1" head
  head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo none)"
  if [[ "$head" != "$want" ]]; then
    echo "ERROR: ${name} vendor checkout is at ${head:0:12}, expected pinned ${want:0:12}." >&2
    echo "       Run 'tools/mg2_scorer/fetch_mg2_scorer.sh --force' to reset it." >&2
    exit 7
  fi
}
verify_pinned_commit "MarkushGrapher"   "$MARKUSHGRAPHER_COMMIT"
verify_pinned_commit "MarkushGenerator" "$MARKUSHGENERATOR_COMMIT"

# 2. Pinned scorer venv (older rdkit/torch stack, isolated from the base uv.lock).
if [[ ! -x "$VENV_PY" ]]; then
  echo "[score] building pinned scorer venv (${SCRIPT_DIR}/.venv-mg2)..."
  uv venv --python 3.11 "${SCRIPT_DIR}/.venv-mg2"
  uv pip install --python "$VENV_PY" -r "${SCRIPT_DIR}/requirements-public-mg2.txt"
fi

# 3. Auto-pull the ground truth for <name> (or use the explicit override).
if [[ -n "$GT_OVERRIDE" ]]; then
  GT="$GT_OVERRIDE"
else
  echo "[score] resolving ground truth for '${BENCH}' from EdisonScientific/glyph-datasets..."
  GT="$(uv run python -m glyph.markush.eval.benchmark_data "$BENCH" --gt)"
fi
echo "[score] ground truth: $GT"

# 4. Official DS4SD scoring (get_scores UNMODIFIED).
echo "[score] official DS4SD scoring $INPUT_JSON -> $OUTPUT_JSON"
PYTHONPATH="$ROOT" "$VENV_PY" "${SCRIPT_DIR}/rescore_with_public_mg2.py" \
    --contract official \
    --input-json "$INPUT_JSON" \
    --gt-jsonl "$GT" \
    --output-json "$OUTPUT_JSON"

# 5. Strict parsed-graph (ours) scoring against the SAME ground truth. Reuses
#    glyph.markush.grading via glyph.markush.eval.strict_score; needs RDKit from
#    the base uv `markush`/`ocsr` extra. Soft-fail: keep the official number.
STRICT_JSON="${OUTPUT_JSON%.json}_strict.json"
GLYPH_EXTRA="${GLYPH_MARKUSH_EXTRA:-markush}"
echo "[score] strict parsed-graph (ours) scoring $INPUT_JSON -> $STRICT_JSON"
STRICT_OK=1
if ! uv run --extra "$GLYPH_EXTRA" python -m glyph.markush.eval.strict_score \
    --benchmark "$BENCH" \
    --input-json "$INPUT_JSON" \
    --gt-jsonl "$GT" \
    --output-json "$STRICT_JSON"; then
  STRICT_OK=0
  echo "[score] WARNING: strict parsed-graph scoring failed (it needs RDKit;" >&2
  echo "        install the base extra, e.g. 'uv sync --extra markush'). Reporting" >&2
  echo "        the official DS4SD number only." >&2
fi

# 6. Combined headline: the paper's two columns for this benchmark.
uv run python - "$OUTPUT_JSON" "$STRICT_JSON" "$BENCH" "$STRICT_OK" <<'PY'
import json, sys

official_path, strict_path, bench, strict_ok = sys.argv[1:5]
ec = json.load(open(official_path))["public_mg2"]["exact_counts"]
off_count, off_size = ec["cxsmi_equality_count"], ec["size"]
off_pct = 100.0 * off_count / off_size if off_size else 0.0

line = "=" * 68
print()
print(line)
print(f"  {bench}: MarkushGlyph benchmark — both scoring contracts")
print(line)
print(
    f"  MarkushGrapher-2 score (DS4SD official):  "
    f"{off_count} / {off_size} = {off_pct:.2f}%"
)
if strict_ok == "1":
    st = json.load(open(strict_path))["strict"]
    print(
        f"  Strict parsed-graph equality (ours):      "
        f"{st['count']} / {st['total']} = {st['percent']:.2f}%"
    )
else:
    print("  Strict parsed-graph equality (ours):      unavailable (RDKit missing)")
print(line)
PY
