#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASK_PYTHON=""
for candidate in "${QORE_PYTHON_BIN:-}" "${CONDA_PREFIX:-}/bin/python" "${VIRTUAL_ENV:-}/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c 'import datasets, numpy, torch, transformers; import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
    TASK_PYTHON="$candidate"
    break
  fi
done
if [[ -z "$TASK_PYTHON" ]]; then
  echo "ERROR: activate the existing project environment with Python>=3.10, datasets, NumPy, Torch and Transformers" >&2
  exit 2
fi
if [[ -z "${QORE_EXCHANGE_TOKEN:-}" ]]; then
  echo "ERROR: QORE_EXCHANGE_TOKEN must be configured for pinned-input download and automatic result upload" >&2
  exit 2
fi

exec "$TASK_PYTHON" -X utf8 -u scripts/collab/five_ideas/prepare_reader_larger_untouched_cohort_256.py --upload "$@"
