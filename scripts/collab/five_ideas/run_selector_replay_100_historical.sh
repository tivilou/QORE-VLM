#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    CANDIDATES=("$PYTHON_BIN")
else
    CANDIDATES=()
    command -v python >/dev/null 2>&1 && CANDIDATES+=("$(command -v python)")
    command -v python3 >/dev/null 2>&1 && CANDIDATES+=("$(command -v python3)")
    [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]] && CANDIDATES+=("$CONDA_PREFIX/bin/python")
    shopt -s nullglob
    CANDIDATES+=(
        /usr/local/miniconda3/envs/*/bin/python
        /opt/conda/envs/*/bin/python
        "$HOME"/miniconda3/envs/*/bin/python
        "$HOME"/anaconda3/envs/*/bin/python
    )
fi

EXPERIMENT_PYTHON=""
declare -A SEEN=()
for candidate in "${CANDIDATES[@]}"; do
    [[ -z "$candidate" || -n "${SEEN[$candidate]:-}" ]] && continue
    SEEN[$candidate]=1
    if [[ -x "$candidate" ]] && "$candidate" -c 'import datasets, numpy, torch, yaml, transformers' >/dev/null 2>&1; then
        EXPERIMENT_PYTHON="$candidate"
        break
    fi
done
if [[ -z "$EXPERIMENT_PYTHON" ]]; then
    echo "ERROR: no compatible Python found; activate the experiment environment" >&2
    exit 1
fi

exec "$EXPERIMENT_PYTHON" scripts/collab/five_ideas/replay_selector_benchmark_100_historical.py "$@"
