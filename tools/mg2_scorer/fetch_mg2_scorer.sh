#!/usr/bin/env bash
# Fetch the pinned DS4SD public MG2 scorer sources.
#
# The official MarkushGlyph IP5-M / M2S / USPTO-Markush scoring uses DS4SD's public
# `markushgrapher.utils.ocsr.utils_evaluation.get_scores` UNMODIFIED (see
# docs/reproducing.md "MarkushGlyph evaluation"). `tools/mg2_scorer/rescore_with_public_mg2.py`
# imports that scorer from two vendored checkouts:
#
#   tools/mg2_scorer/vendor/MarkushGrapher    (the get_scores contract)
#   tools/mg2_scorer/vendor/MarkushGenerator  (its cxsmiles tokenizer dependency)
#
# The vendor/ directory is git-ignored (heavy, third-party, local-only), so this
# script recreates it at the EXACT pinned commits. It is idempotent: a checkout
# already present at the right commit is left untouched.
#
# Pinned commits (source of truth: tools/mg2_scorer/VENDOR_COMMITS.md):
#   MarkushGrapher    35c71a0b4423983920b4703f52f46c5bca9ae631
#   MarkushGenerator  17580c20d266886ef9ae22a50aab51444e0898b2
#
# Usage:
#   tools/mg2_scorer/fetch_mg2_scorer.sh            # clone/verify both checkouts
#   tools/mg2_scorer/fetch_mg2_scorer.sh --force    # remove + re-clone from scratch
#   tools/mg2_scorer/fetch_mg2_scorer.sh --help
#
# After fetching, run the scorer as documented in docs/reproducing.md:
#   PYTHONPATH="$PWD" python tools/mg2_scorer/rescore_with_public_mg2.py ...

set -euo pipefail

# --- Pinned upstream sources (keep in sync with VENDOR_COMMITS.md) -----------
MARKUSHGRAPHER_URL="https://github.com/DS4SD/MarkushGrapher.git"
MARKUSHGRAPHER_COMMIT="35c71a0b4423983920b4703f52f46c5bca9ae631"

MARKUSHGENERATOR_URL="https://github.com/DS4SD/MarkushGenerator.git"
MARKUSHGENERATOR_COMMIT="17580c20d266886ef9ae22a50aab51444e0898b2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/vendor"

FORCE=0

usage() {
  cat <<EOF
fetch_mg2_scorer.sh - clone the pinned DS4SD public MG2 scorer sources.

Recreates the git-ignored vendor/ checkouts used by
tools/mg2_scorer/rescore_with_public_mg2.py, pinned to the exact upstream
commits recorded in tools/mg2_scorer/VENDOR_COMMITS.md:

  MarkushGrapher    ${MARKUSHGRAPHER_COMMIT}
  MarkushGenerator  ${MARKUSHGENERATOR_COMMIT}

Clone target:
  ${VENDOR_DIR}/MarkushGrapher
  ${VENDOR_DIR}/MarkushGenerator

Options:
  --force     Remove any existing checkout and re-clone from scratch.
  -h, --help  Show this help and exit.

The script is idempotent: a checkout already at the pinned commit is skipped.
Requires git and network access to github.com.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --force)   FORCE=1 ;;
    *) echo "ERROR: unknown argument: $arg" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

# --- Preconditions -----------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed or not on PATH. Install git and retry." >&2
  exit 3
fi

# --- Fetch one repo at a pinned commit (idempotent) --------------------------
# Args: <display-name> <clone-url> <commit-sha>
fetch_repo() {
  local name="$1" url="$2" commit="$3"
  local dest="${VENDOR_DIR}/${name}"

  if [[ "$FORCE" -eq 1 && -e "$dest" ]]; then
    echo "[${name}] --force: removing existing ${dest}"
    rm -rf "$dest"
  fi

  if [[ -d "${dest}/.git" ]]; then
    local head
    head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo none)"
    if [[ "$head" == "$commit" ]]; then
      echo "[${name}] already at pinned commit ${commit:0:12} - skipping."
      return 0
    fi
    echo "[${name}] present at ${head:0:12}, want ${commit:0:12} - fetching..."
    if ! git -C "$dest" fetch --quiet origin "$commit" 2>/dev/null; then
      # Some servers reject fetch-by-sha; fall back to a full fetch.
      git -C "$dest" fetch --quiet --tags origin \
        || { echo "ERROR: [${name}] git fetch failed (network/remote?)." >&2; exit 4; }
    fi
    git -C "$dest" checkout --quiet "$commit" \
      || { echo "ERROR: [${name}] cannot check out ${commit} (unknown commit?)." >&2; exit 5; }
    echo "[${name}] checked out ${commit:0:12}."
    return 0
  fi

  if [[ -e "$dest" ]]; then
    echo "ERROR: ${dest} exists but is not a git checkout. Move it aside or run --force." >&2
    exit 6
  fi

  echo "[${name}] cloning ${url} -> ${dest}"
  mkdir -p "$VENDOR_DIR"
  if ! git clone --quiet "$url" "$dest"; then
    echo "ERROR: [${name}] git clone failed. Check network access to github.com." >&2
    exit 4
  fi
  git -C "$dest" checkout --quiet "$commit" \
    || { echo "ERROR: [${name}] cannot check out pinned commit ${commit}." >&2; exit 5; }
  echo "[${name}] cloned and pinned at ${commit:0:12}."
}

echo "Fetching pinned DS4SD public MG2 scorer sources into ${VENDOR_DIR}"
fetch_repo "MarkushGrapher"   "$MARKUSHGRAPHER_URL"   "$MARKUSHGRAPHER_COMMIT"
fetch_repo "MarkushGenerator" "$MARKUSHGENERATOR_URL" "$MARKUSHGENERATOR_COMMIT"

echo
echo "Done. Vendored scorer sources:"
echo "  ${VENDOR_DIR}/MarkushGrapher    @ ${MARKUSHGRAPHER_COMMIT}"
echo "  ${VENDOR_DIR}/MarkushGenerator  @ ${MARKUSHGENERATOR_COMMIT}"
echo
echo "Score predictions with (see docs/reproducing.md):"
echo "  PYTHONPATH=\"\$PWD\" python tools/mg2_scorer/rescore_with_public_mg2.py \\"
echo "      --input-json <preds.json> --output-json <scored.json>"
