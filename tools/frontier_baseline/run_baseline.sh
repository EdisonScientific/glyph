#!/usr/bin/env bash
# Thin wrapper: run the frontier-model baseline CLI from the isolated venv when
# present, else the active environment. All arguments forward to
# `glyph markush eval baseline`.
#
# Example:
#   tools/frontier_baseline/run_baseline.sh --benchmark ip5m \
#       --model claude-opus-4-8 --model gpt-5.6-sol --variant both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PY="${SCRIPT_DIR}/.venv-baseline/bin/python"

cd "$ROOT"

if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m glyph.cli markush eval baseline "$@"
fi

exec glyph markush eval baseline "$@"
