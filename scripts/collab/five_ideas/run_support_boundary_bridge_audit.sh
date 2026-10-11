#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
for candidate in python python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
    exec "$candidate" -B scripts/collab/five_ideas/compile_support_boundary_bridge.py "$@"
  fi
done
printf '%s\n' 'Python >=3.10 is required for this offline audit.' >&2
exit 1
