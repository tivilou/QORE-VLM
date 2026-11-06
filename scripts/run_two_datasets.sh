#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -n "${QDUET_BOOTSTRAP_PYTHON:-}" ]]; then
  exec "$QDUET_BOOTSTRAP_PYTHON" -B scripts/run_two_datasets.py "$@"
fi
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  exec "$CONDA_PREFIX/bin/python" -B scripts/run_two_datasets.py "$@"
fi
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  exec "$VIRTUAL_ENV/bin/python" -B scripts/run_two_datasets.py "$@"
fi
if command -v python3.11 >/dev/null 2>&1; then
  exec python3.11 -B scripts/run_two_datasets.py "$@"
fi
exec python3 -B scripts/run_two_datasets.py "$@"
