#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
else
    PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON" ]] || ! "$PYTHON" -c 'import numpy; import sys' >/dev/null 2>&1; then
    echo "ERROR: activate the project Python environment with NumPy installed" >&2
    exit 1
fi
if [[ -z "${QORE_EXCHANGE_TOKEN:-}" ]]; then
    echo "ERROR: QORE_EXCHANGE_TOKEN is required to fetch the pinned Reader trace and create the run directory" >&2
    exit 1
fi

echo "Using Python: $PYTHON"
exec "$PYTHON" "$SCRIPT_DIR/run_dpr_span_relevance_quantum_screen_100.py" "$@"
