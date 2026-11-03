#!/usr/bin/env bash
set -euo pipefail
target="${QDUET_RUN_ROOT:-$PWD/qduet-two-datasets}"
mkdir -p -- "$target"
target="$(cd -- "$target" && pwd -P)"
if [[ ! -d "$target/code/.git" ]]; then
  git clone --single-branch --branch codex/top8-dual-scorer \
    https://github.com/tivilou/QORE-VLM.git "$target/code"
fi
cd -- "$target/code"
test "$(git remote get-url origin)" = https://github.com/tivilou/QORE-VLM.git || { echo 'Unexpected existing repository' >&2; exit 2; }
test "$(git branch --show-current)" = codex/top8-dual-scorer || { echo 'Unexpected existing branch' >&2; exit 2; }
exec bash scripts/run_two_datasets.sh --workspace "$target/work" "$@"
