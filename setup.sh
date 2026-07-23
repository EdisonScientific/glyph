#!/usr/bin/env bash
# Glyph environment bootstrap.
#
# One idempotent entry point for the reproducible environment behind the paper
# "Chemical Structure Transcription: From Molecules to Markush Patents".
# It:
#   1. ensures `uv` is available (the project is uv-managed; uv.lock is tracked),
#   2. `uv sync`s the requested extra (core / ocsr / markush / all),
#   3. fetches the pinned DS4SD public MG2 scorer sources (for IP5-M scoring).
#
# Re-runnable: every step skips work that is already in place.
#
# Usage:
#   ./setup.sh                       # core deps + pinned MG2 scorer
#   ./setup.sh --extra all           # everything (torch/transformers/timm)
#   ./setup.sh --extra markush       # Markush VLM stack only
#   ./setup.sh --extra ocsr          # OCSRGlyph stack only
#   ./setup.sh --no-scorer           # skip the MG2 scorer fetch
#   ./setup.sh --help
#
# Environment overrides:
#   EXTRA=all ./setup.sh             # same as --extra all
#   SKIP_UV_INSTALL=1 ./setup.sh     # never auto-install uv (fail if missing)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

EXTRA="${EXTRA:-}"          # empty = core only
FETCH_SCORER=1

usage() {
  cat <<EOF
setup.sh - bootstrap the Glyph environment (idempotent).

Steps: ensure uv -> uv sync [--extra ...] -> fetch pinned MG2 scorer.

Options:
  --extra NAME   uv extra to install: ocsr | markush | all
                 (omit for the core package only). Also settable via EXTRA=NAME.
  --no-scorer    Skip fetching the DS4SD public MG2 scorer sources.
  -h, --help     Show this help and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extra)      EXTRA="${2:?--extra needs a value}"; shift 2 ;;
    --extra=*)    EXTRA="${1#*=}"; shift ;;
    --no-scorer)  FETCH_SCORER=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

step() { echo; echo "==> $*"; }

# --- 1. uv -------------------------------------------------------------------
step "1/3  Ensuring uv is installed"
if command -v uv >/dev/null 2>&1; then
  echo "    uv found: $(uv --version)"
else
  if [[ "${SKIP_UV_INSTALL:-0}" == "1" ]]; then
    echo "ERROR: uv not found and SKIP_UV_INSTALL=1. Install uv:" >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 3
  fi
  echo "    uv not found - installing via the official installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops uv in ~/.local/bin (or $XDG_BIN_HOME); make it visible now.
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv still not on PATH after install. Add ~/.local/bin to PATH." >&2
    exit 3
  }
  echo "    installed: $(uv --version)"
fi

# --- 2. uv sync --------------------------------------------------------------
step "2/3  Syncing dependencies from uv.lock"
if [[ -n "$EXTRA" ]]; then
  echo "    uv sync --extra ${EXTRA}"
  uv sync --extra "$EXTRA"
else
  echo "    uv sync   (core only; add --extra ocsr|markush|all for model stacks)"
  uv sync
fi
cat <<'EOF'
    Extras available:
      --extra ocsr     OCSRGlyph model  (torch, timm, rdkit, datasets)
      --extra markush  Markush VLM           (torch, transformers, peft, accelerate)
      --extra all      everything

    Blackwell / RTX 50-series (sm_120): the tracked uv.lock ALREADY pins torch to
    a cu128 wheel on Linux (torch 2.11+cu128 via [tool.uv.sources] -> the
    pytorch-cu128 index), so `uv sync` yields an sm_120-capable build. If you
    install torch outside the lock (plain pip, or a non-Linux host), pull it from
    the CUDA 12.8 index explicitly:
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    Pre-cu128 wheels ship kernels only up to sm_90 and fail on Blackwell with
    "no kernel image available".

    Optional inference-perf deps (NOT in the base lock): the Qwen3.5 Markush path
    can use flash-linear-attention + Triton kernels (the cluster's `.venv-fla-tl`).
    They are optional accelerators, not required for correctness; install
    separately if you want them and your GPU/Triton stack supports them.
EOF

# --- 3. Pinned MG2 scorer ----------------------------------------------------
step "3/3  Fetching the pinned DS4SD public MG2 scorer"
if [[ "$FETCH_SCORER" -eq 1 ]]; then
  bash "${REPO_ROOT}/tools/mg2_scorer/fetch_mg2_scorer.sh"
else
  echo "    skipped (--no-scorer). Run tools/mg2_scorer/fetch_mg2_scorer.sh before scoring."
fi
cat <<'EOF'

    NOTE: the official IP5-M rescore runs in a SEPARATE pinned env, NOT the base
    uv.lock. DS4SD's get_scores pins an older stack (rdkit 2024.3.5, numpy
    1.24.4, torch 2.2.0) and needs nltk/rouge-score/matplotlib, none of which
    are in the base lock. Build the isolated scorer venv once:
      uv venv --python 3.11 tools/mg2_scorer/.venv-mg2
      uv pip install --python tools/mg2_scorer/.venv-mg2/bin/python \
          -r tools/mg2_scorer/requirements-public-mg2.txt
    Then rescore with tools/mg2_scorer/.venv-mg2/bin/python (see
    docs/reproducing.md, "Building the scorer environment").
EOF

echo
echo "Setup complete."
