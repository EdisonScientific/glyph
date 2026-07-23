#!/usr/bin/env bash
# Build the isolated frontier-baseline environment with uv.
#
# litellm is intentionally NOT a core runtime dependency. This creates a
# dedicated venv (tools/frontier_baseline/.venv-baseline) with the Markush eval
# stack plus litellm, so frontier-model baselines never bloat the base install.
#
# Usage:
#   tools/frontier_baseline/setup_venv.sh            # CPU wheels (default)
#   TORCH_BACKEND=cu128 tools/frontier_baseline/setup_venv.sh
#
# Then export the venv and run the CLI, for example:
#   source tools/frontier_baseline/.venv-baseline/bin/activate
#   glyph markush eval baseline --benchmark ip5m --model claude-opus-4-8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv-baseline"
VENV_PY="${VENV_DIR}/bin/python"
TORCH_BACKEND="${TORCH_BACKEND:-cpu}"

cd "$ROOT"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[baseline] creating venv at ${VENV_DIR} (python 3.11)"
  uv venv --python 3.11 "$VENV_DIR"
fi

echo "[baseline] installing glyph[markush] + litellm (torch backend: ${TORCH_BACKEND})"
uv pip install --python "$VENV_PY" --no-sources --torch-backend "$TORCH_BACKEND" \
  --editable "${ROOT}[markush]" litellm

echo "[baseline] done. Example:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  glyph markush eval baseline --benchmark ip5m --model claude-opus-4-8 --model gpt-5.6-sol"
