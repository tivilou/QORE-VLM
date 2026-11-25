#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${QDUET_BOOTSTRAP_PYTHON:-python3}" -u -B scripts/run_local_born_smallset.py "$@"
