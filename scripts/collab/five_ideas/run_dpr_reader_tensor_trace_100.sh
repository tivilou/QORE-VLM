#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$PYTHON_BIN"
else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c 'import sys' >/dev/null 2>&1; then
    echo "ERROR: no runnable Python found; activate the experiment environment" >&2
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_dpr_reader_tensor_trace_100.py" --upload --progress "$@"
