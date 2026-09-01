#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates=("$PYTHON_BIN")
else
    candidates=("$(command -v python || true)" "$(command -v python3 || true)")
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        candidates+=("$CONDA_PREFIX/bin/python")
    fi
fi

PYTHON_BIN=""
declare -A seen=()
for candidate_path in "${candidates[@]}"; do
    [[ -z "$candidate_path" || -n "${seen[$candidate_path]:-}" ]] && continue
    seen[$candidate_path]=1
    if [[ -x "$candidate_path" ]] && "$candidate_path" -c 'import datasets' >/dev/null 2>&1; then
        PYTHON_BIN="$candidate_path"
        break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: activate an environment containing the datasets package" >&2
    exit 1
fi

exec "$PYTHON_BIN" scripts/collab/five_ideas/run_dpr_title_presence_check.py "$@"
