#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export QORE_PROJECT_ROOT="$PROJECT_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! "$PYTHON_BIN" -c 'import accelerate, torch, transformers' >/dev/null 2>&1; then
        echo "ERROR: PYTHON_BIN lacks accelerate, torch, or transformers" >&2
        exit 1
    fi
else
    PYTHON_BIN=""
    user_home_dir="${HOME:-}"
    shopt -s nullglob
    candidates=(
        "$(command -v python || true)"
        "$(command -v python3 || true)"
        "${CONDA_PREFIX:-}/bin/python"
        /usr/local/miniconda3/envs/*/bin/python
        /opt/conda/envs/*/bin/python
        "$user_home_dir"/miniconda3/envs/*/bin/python
        "$user_home_dir"/anaconda3/envs/*/bin/python
    )
    declare -A seen=()
    for candidate_path in "${candidates[@]}"; do
        [[ -z "$candidate_path" || -n "${seen[$candidate_path]:-}" ]] && continue
        seen[$candidate_path]=1
        if [[ -x "$candidate_path" ]] && "$candidate_path" -c 'import accelerate, torch, transformers' >/dev/null 2>&1; then
            PYTHON_BIN="$candidate_path"
            break
        fi
    done
    if [[ -z "$PYTHON_BIN" ]]; then
        echo "ERROR: no compatible Python found; activate an environment with accelerate, torch, and transformers" >&2
        exit 1
    fi
fi

OUTPUT_PATH="${OUTPUT_PATH:-exchange/five_ideas/generator_native_evidence_anchor_transport_hook_preflight/report.json}"
exec "$PYTHON_BIN" scripts/collab/five_ideas/generator_native_evidence_anchor_transport_hook_preflight.py --output "$OUTPUT_PATH" "$@"
