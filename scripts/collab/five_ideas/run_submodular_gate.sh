#!/usr/bin/env bash
# Run the Phase 5 Submodular gate with the caller's Python environment.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exchange/five_ideas/submodular_gate}"
exec "$PYTHON_BIN" scripts/collab/five_ideas/run_development_gate.py \
  --gate-config configs/experiments/five_idea_submodular_gate.yaml \
  --output-root "$OUTPUT_ROOT" \
  "$@"
