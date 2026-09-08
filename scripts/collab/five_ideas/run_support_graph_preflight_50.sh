#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
exec python "$PROJECT_ROOT/scripts/collab/five_ideas/run_support_graph_preflight_50.py" "$@"
