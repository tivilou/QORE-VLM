#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates=("$PYTHON_BIN")
else
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
fi

PYTHON_BIN=""
declare -A seen=()
for candidate_path in "${candidates[@]}"; do
    [[ -z "$candidate_path" || -n "${seen[$candidate_path]:-}" ]] && continue
    seen[$candidate_path]=1
    if [[ -x "$candidate_path" ]] && "$candidate_path" -c 'import datasets, numpy, torch, yaml, transformers' >/dev/null 2>&1; then
        PYTHON_BIN="$candidate_path"
        break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: no compatible Python found; activate an environment with datasets, numpy, torch, PyYAML, and transformers" >&2
    exit 1
fi

exec "$PYTHON_BIN" scripts/collab/five_ideas/run_gold_evidence_alignment_audit.py "$@"
