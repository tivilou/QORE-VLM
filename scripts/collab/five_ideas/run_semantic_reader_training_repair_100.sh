#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON=python
if ! command -v "$PYTHON" >/dev/null 2>&1; then PYTHON=python3; fi
exec "$PYTHON" "$PROJECT_ROOT/scripts/collab/five_ideas/run_semantic_reader_training_repair_100.py" --upload "$@"
