#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${QDUET_BOOTSTRAP_PYTHON:-python3}" -B scripts/run_triviaqa_smallset.py "$@"
