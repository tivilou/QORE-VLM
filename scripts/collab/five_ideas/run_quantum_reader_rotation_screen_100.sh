#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TASK_ROOT"

if [[ -n "${QORE_PYTHON_BIN:-}" && -x "${QORE_PYTHON_BIN}" ]]; then
  TASK_PYTHON="$QORE_PYTHON_BIN"
else
  TASK_PYTHON=""
  for candidate in "${CONDA_PREFIX:-}/bin/python" "${VIRTUAL_ENV:-}/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then TASK_PYTHON="$candidate"; break; fi
  done
  if [[ -z "$TASK_PYTHON" ]]; then echo "No usable Python found; activate the project environment first." >&2; exit 2; fi
fi

exec "$TASK_PYTHON" -X utf8 -u scripts/collab/five_ideas/run_quantum_reader_rotation_screen_100.py --upload "$@"
