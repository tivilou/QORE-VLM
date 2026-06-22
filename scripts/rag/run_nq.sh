#!/bin/bash
# Run RAG experiment on Natural Questions
# Usage: bash scripts/rag/run_nq.sh [model_variant]
#
# Environment variables:
#   MODEL_PATH  - path to model (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   K           - number of passages to select (default: 5)
#   SOLVER      - qore solver method (default: anneal)
#   MAX_SAMPLES - limit number of queries (default: all)
#   OUTPUT_DIR  - output directory (default: results/rag/nq)

set -e

MODEL_VARIANT=${1:-llama3}

if [ "$MODEL_VARIANT" = "mistral" ]; then
    MODEL_PATH=${MODEL_PATH:-"mistralai/Mistral-7B-Instruct-v0.3"}
else
    MODEL_PATH=${MODEL_PATH:-"meta-llama/Meta-Llama-3-8B-Instruct"}
fi

K=${K:-5}
SOLVER=${SOLVER:-anneal}
MAX_SAMPLES=${MAX_SAMPLES:-0}  # 0 = all
OUTPUT_DIR=${OUTPUT_DIR:-"results/rag/nq"}
DATASET=${DATASET:-"natural_questions"}
EMBED_MODEL=${EMBED_MODEL:-"facebook/dpr-question_encoder-single-nq-base"}

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  QORE-RAG Experiment: Natural Questions"
echo "=============================================="
echo "  Model:       $MODEL_PATH"
echo "  Embeddings:  $EMBED_MODEL"
echo "  Budget K:    $K"
echo "  Solver:      $SOLVER"
echo "  Max samples: $MAX_SAMPLES (0=all)"
echo "  Output:      $OUTPUT_DIR"
echo "=============================================="

# Run all methods
for METHOD in qore topk mmr; do
    echo ""
    echo ">>> Running method: $METHOD"
    python -m scripts.rag.eval_rag \
        --model_path "$MODEL_PATH" \
        --embed_model "$EMBED_MODEL" \
        --dataset "$DATASET" \
        --method "$METHOD" \
        --K "$K" \
        --solver "$SOLVER" \
        --max_samples "$MAX_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --output_file "${METHOD}.json"
done

# Generate comparison table
echo ""
echo ">>> Generating summary..."
python -m scripts.rag.summarize --input_dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/summary.csv"

echo ""
echo "Done! Results in $OUTPUT_DIR/"
