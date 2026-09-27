#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/audit_qgsrr_feature_100.py" "$@"
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$SCRIPT_DIR/audit_qgsrr_feature_100.py" "$@"
fi
exec python "$SCRIPT_DIR/audit_qgsrr_feature_100.py" "$@"
