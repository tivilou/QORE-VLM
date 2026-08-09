#!/usr/bin/env bash
# Five-idea program: generate solver-faithful candidate diagnostics.
# Default is the 200-question development pilot; --smoke runs 20 questions.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MAX_SAMPLES=200
SEED=42
OUTPUT_ROOT="exchange/five_ideas/diagnostic_pilot"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/py310/bin/python}"
ALLOW_DIRTY=0
SKIP_GENERATION=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collab/five_ideas/run_diagnostics_pilot.sh [options]

Options:
  --smoke              Run 20 questions for output/schema validation.
  --max-samples N      Override the sample count (default: 200).
  --seed N             Evaluation seed (default: 42).
  --output-root PATH   Result root (default: exchange/five_ideas/diagnostic_pilot).
  --python-bin PATH    Python executable (default: py310 environment).
  --skip-generation    Skip generator; useful for diagnostics-only smoke tests.
  --allow-dirty        Allow tracked worktree changes for a debug/smoke run.
  -h|--help            Show this help.
EOF
}

while (($# > 0)); do
    case "$1" in
        --smoke) MAX_SAMPLES=20; shift ;;
        --max-samples) [[ $# -ge 2 ]] || { echo "Missing value for --max-samples" >&2; exit 2; }; MAX_SAMPLES="$2"; shift 2 ;;
        --seed) [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 2; }; SEED="$2"; shift 2 ;;
        --output-root) [[ $# -ge 2 ]] || { echo "Missing value for --output-root" >&2; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
        --python-bin) [[ $# -ge 2 ]] || { echo "Missing value for --python-bin" >&2; exit 2; }; PYTHON_BIN="$2"; shift 2 ;;
        --skip-generation) SKIP_GENERATION=1; shift ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { echo "--max-samples must be a non-negative integer" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found or not executable: $PYTHON_BIN" >&2; exit 1; }

if (( ALLOW_DIRTY == 0 )); then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Tracked changes detected. Use --allow-dirty only for a debug/smoke run." >&2
        git status --short --branch >&2
        exit 1
    fi
fi

TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/$TIMESTAMP"
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git status --short --branch > "$RUN_DIR/git_status.txt"
git diff --stat > "$RUN_DIR/git_diff_stat.txt"

COMMON_ARGS=(
    --dataset nq_open
    --split validation
    --max_samples "$MAX_SAMPLES"
    --corpus_mode wiki_dpr
    --K 5
    --lam 2.0
    --seed "$SEED"
    --dump_passages
    --output_file result.json
)
if (( SKIP_GENERATION == 1 )); then
    COMMON_ARGS+=(--skip_generation)
fi

run_config() {
    local name="$1"
    shift
    local output_dir="$RUN_DIR/$name"
    mkdir -p "$output_dir"
    local -a args=("${COMMON_ARGS[@]}" "$@" --output_dir "$output_dir")
    {
        printf '%q ' "$PYTHON_BIN" -m scripts.rag.eval.eval_rag_refactored
        printf '%q ' "${args[@]}"
        printf '\n'
    } > "$output_dir/command.txt"
    echo "==> $name"
    "$PYTHON_BIN" -m scripts.rag.eval.eval_rag_refactored \
        "${args[@]}" 2>&1 | tee "$output_dir/log.txt"
    "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); s=d.get("samples", []); q=sum(1 for x in s if x.get("qubo")); print(f"validated samples={len(s)} qubo_diagnostics={q}")' "$output_dir/result.json"
}

run_config qore_dpr --method qore --gamma 0.5
run_config qore_as_control --method qore --gamma 0.5 --delta 0.0 --use_answer_scorer
run_config qore_as_idea6 --method qore --gamma 0.5 --delta 0.1 --complementarity_method dpr --use_answer_scorer
run_config topk_as --method topk --use_answer_scorer
run_config mmr_as --method mmr --lambda_mmr 0.7 --use_answer_scorer

echo "Completed five-idea diagnostics pilot: $RUN_DIR"
