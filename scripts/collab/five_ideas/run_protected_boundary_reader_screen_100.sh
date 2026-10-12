#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TASK_ROOT"
TASK_PYTHON=""
for candidate in python python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys,torch,transformers,numpy; assert sys.version_info >= (3,10)' >/dev/null 2>&1; then
    TASK_PYTHON="$candidate"
    break
  fi
done
if [[ -z "$TASK_PYTHON" ]] && command -v conda >/dev/null 2>&1; then
  TASK_ENVS="$(conda env list --json)"
  while IFS= read -r candidate; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys,torch,transformers,numpy; assert sys.version_info >= (3,10)' >/dev/null 2>&1; then
      TASK_PYTHON="$candidate"
      break
    fi
  done < <(printf '%s' "$TASK_ENVS" | conda run --no-capture-output -n base python -X utf8 -c 'import json,sys,os; print("\n".join(os.path.join(p,"python.exe" if os.name=="nt" else "bin/python").replace("\\","/") for p in json.load(sys.stdin)["envs"]))')
fi
if [[ -z "$TASK_PYTHON" ]]; then
  printf '%s\n' 'No Python >=3.10 with torch, transformers and numpy found. Activate your experiment environment.' >&2
  exit 2
fi
exec "$TASK_PYTHON" -X utf8 -u scripts/collab/five_ideas/run_protected_boundary_reader_screen_100.py --upload "$@"
