#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if command -v python3.11 >/dev/null 2>&1; then
  exec python3.11 -B scripts/run_two_datasets.py "$@"
fi
exec python3 -B scripts/run_two_datasets.py "$@"
