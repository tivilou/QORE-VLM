#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
# The Python runner discovers the collaborator's local environment, model cache,
# corpus, and project root.  It also uploads the exchange-only trace when the
# collaborator has configured QORE_EXCHANGE_TOKEN.
exec python "$PROJECT_ROOT/scripts/collab/five_ideas/run_qarcg_reader_screen_100.py" --upload "$@"
