#!/usr/bin/env bash
# One-click collaborator wrapper for the Phase 9B retrieval-ceiling diagnostic.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! "$PYTHON_BIN" -c 'import numpy, torch, yaml, transformers' >/dev/null 2>&1; then
        echo "ERROR: PYTHON_BIN lacks numpy, torch, PyYAML, or transformers: $PYTHON_BIN" >&2
        exit 1
    fi
else
    PYTHON_BIN=""
    shopt -s nullglob
    candidates=(
        "$(command -v python || true)"
        "$(command -v python3 || true)"
        "${CONDA_PREFIX:-}/bin/python"
        /usr/local/miniconda3/envs/*/bin/python
        /opt/conda/envs/*/bin/python
        "$HOME"/miniconda3/envs/*/bin/python
        "$HOME"/anaconda3/envs/*/bin/python
    )
    declare -A seen=()
    for candidate_path in "${candidates[@]}"; do
        if [[ -z "$candidate_path" || -n "${seen[$candidate_path]:-}" ]]; then
            continue
        fi
        seen[$candidate_path]=1
        if [[ -x "$candidate_path" ]] && "$candidate_path" -c 'import numpy, torch, yaml, transformers' >/dev/null 2>&1; then
            PYTHON_BIN="$candidate_path"
            break
        fi
    done
    if [[ -z "$PYTHON_BIN" ]]; then
        echo "ERROR: no PATH Python has numpy, torch, PyYAML, and transformers; activate the experiment environment." >&2
        exit 1
    fi
fi

exec "$PYTHON_BIN" scripts/collab/five_ideas/run_phase9b_retrieval_ceiling.py "$@"
