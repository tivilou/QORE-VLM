#!/usr/bin/env bash
# Backward-compatible entry point for the configuration-driven development gate.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "${PYTHON_BIN:-$(command -v python3 || command -v python)}" \
    scripts/collab/five_ideas/run_development_gate.py "$@"
