#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TASK_ROOT"
TASK_PYTHON=""
for candidate in "${QORE_PYTHON_BIN:-}" "${CONDA_PREFIX:-}/bin/python" "${VIRTUAL_ENV:-}/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c 'import sys, torch, numpy; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
    TASK_PYTHON="$candidate"
    break
  fi
done
if [[ -z "$TASK_PYTHON" ]]; then echo "Activate an environment with Python>=3.10, Torch and NumPy." >&2; exit 2; fi
exec "$TASK_PYTHON" -X utf8 -u scripts/collab/five_ideas/run_quantum_reader_membership_screen_100.py --upload "$@"
