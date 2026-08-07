#!/usr/bin/env bash
# Run the matched attribution matrix for QORE + Answer Scorer + Idea 6.
# Default: full NQ validation split (3610 samples), one deterministic seed.
# Smoke test: bash scripts/collab/idea6_phase3/run_attribution_matrix.sh --smoke.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MAX_SAMPLES=0
SEED=42
ALLOW_DIRTY=0
OUTPUT_ROOT="exchange/idea6_attribution"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collab/idea6_phase3/run_attribution_matrix.sh [options]

Options:
  --smoke              Run 20 samples instead of the full validation split.
  --max-samples N      Override the sample count (0 means all 3610 samples).
  --seed N             Random seed (default: 42; current pipeline is deterministic).
  --output-root PATH   Result root (default: exchange/idea6_attribution).
  --allow-dirty        Allow tracked changes; not recommended for paper runs.
  -h, --help           Show this help.

Environment:
  PYTHON_BIN           Python executable from the project environment (default: python).
EOF
}

while (($# > 0)); do
    case "$1" in
        --smoke) MAX_SAMPLES=20; shift ;;
        --max-samples)
            [[ $# -ge 2 ]] || { echo "Missing value for --max-samples" >&2; exit 2; }
            MAX_SAMPLES="$2"; shift 2 ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 2; }
            SEED="$2"; shift 2 ;;
        --output-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --output-root" >&2; exit 2; }
            OUTPUT_ROOT="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { echo "--max-samples must be a non-negative integer" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
}

if (( ALLOW_DIRTY == 0 )); then
    dirty_tracked=0
    if ! git diff --quiet; then dirty_tracked=1; fi
    if ! git diff --cached --quiet; then dirty_tracked=1; fi
    conflicts="$(git diff --name-only --diff-filter=U; git diff --cached --name-only --diff-filter=U)"
    if (( dirty_tracked )) || [[ -n "$conflicts" ]]; then
        echo "Tracked changes or merge conflicts detected. Use a clean worktree." >&2
        git status --short --branch >&2
        echo "Use --allow-dirty only for a non-paper smoke/debug run." >&2
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
    --output_file result.json
)

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
}

# These five configs separate Answer Scorer, complementarity, and baseline effects.
run_config qore_dpr --method qore --gamma 0.5
run_config qore_as_control --method qore --gamma 0.5 --delta 0.0 --use_answer_scorer
run_config qore_as_idea6 --method qore --gamma 0.5 --delta 0.1 --complementarity_method dpr --use_answer_scorer
run_config topk_as --method topk --use_answer_scorer
run_config mmr_as --method mmr --lambda_mmr 0.7 --use_answer_scorer

"$PYTHON_BIN" scripts/collab/idea6_phase3/summarize_attribution_matrix.py "$RUN_DIR"
echo
echo "Completed attribution matrix."
echo "Results: $RUN_DIR"
echo "Read: $RUN_DIR/summary.md"
