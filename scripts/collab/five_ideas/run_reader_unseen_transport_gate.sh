#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FEATURES="${ROOT}/.cache/reader-unseen-validation87-v1"
BUNDLE="${ROOT}/.cache/protected_boundary_reader_inputs/20261011-v1/frozen_support_bundle.zip"
SEQ="${ROOT}/exchange/five_ideas/reader_sequential_protected_updates/20261018T045707538341Z/sequential_values.npz"
if [[ ! -f "${FEATURES}/cache_index.json" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/collab/five_ideas/prepare_reader_unseen_validation_features.py" --bundle "${BUNDLE}" --output "${FEATURES}"
fi
"${PYTHON_BIN}" "${ROOT}/scripts/collab/five_ideas/run_reader_unseen_transport_gate.py" --features "${FEATURES}" --sequential-values "${SEQ}"
