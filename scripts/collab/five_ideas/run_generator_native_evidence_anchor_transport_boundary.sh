#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    exec "$PYTHON_BIN" scripts/collab/five_ideas/run_generator_native_evidence_anchor_transport_boundary.py "$@"
fi

for candidate in "$(command -v python3 || true)" "$(command -v python || true)"; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import yaml' >/dev/null 2>&1; then
        exec "$candidate" scripts/collab/five_ideas/run_generator_native_evidence_anchor_transport_boundary.py "$@"
    fi
done

echo "ERROR: no compatible Python with PyYAML found" >&2
exit 1
