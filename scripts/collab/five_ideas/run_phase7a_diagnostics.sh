#!/usr/bin/env bash
# One-click collaborator wrapper for the Phase 7A Wiki-DPR diagnostic pilot.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! "$PYTHON_BIN" -c 'import numpy, torch, yaml' >/dev/null 2>&1; then
        echo "ERROR: PYTHON_BIN lacks one of numpy, torch, or PyYAML: $PYTHON_BIN" >&2
        exit 1
    fi
else
    PYTHON_BIN=""
    for candidate in python python3; do
        candidate_path="$(command -v "$candidate" || true)"
        if [[ -n "$candidate_path" ]] && "$candidate_path" -c 'import numpy, torch, yaml' >/dev/null 2>&1; then
            PYTHON_BIN="$candidate_path"
            break
        fi
    done
    if [[ -z "$PYTHON_BIN" ]]; then
        echo "ERROR: no PATH Python has numpy, torch, and PyYAML; activate the experiment environment or set PYTHON_BIN." >&2
        exit 1
    fi
fi

exec "$PYTHON_BIN" scripts/collab/five_ideas/run_phase7a_diagnostics.py "$@"
