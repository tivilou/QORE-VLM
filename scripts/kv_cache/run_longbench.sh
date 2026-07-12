#!/bin/bash
# Run KV-Cache eviction experiment on LongBench
# Usage: bash scripts/kv_cache/run_longbench.sh [model_variant]
#
# Environment variables:
#   MODEL_PATH       - path to model (default: ungated NousResearch mirror)
#   MAX_CAPACITY     - KV cache capacity (default: 1024)
#   TRIGGER_EVERY    - eviction trigger interval (default: 128)
#   MAX_SAMPLES      - limit number of samples (default: all)
#   MAX_INPUT_LENGTH - truncate prompts to this many tokens (default: 7900).
#                      Attention capture materializes an [heads, S, S] matrix per
#                      layer during prefill, so peak memory grows with S^2. Lower
#                      this (e.g. 4000) if you OOM on a smaller/shared GPU.
#   OUTPUT_DIR       - output directory (default: results/kv_cache/longbench)

set -e

MODEL_VARIANT=${1:-llama3}

# Defaults use UNGATED mirrors so this runs without HuggingFace gated-access
# approval. They are byte-identical re-uploads of the official weights. If you
# have gated access and prefer the official repos, override e.g.:
#   MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct bash scripts/kv_cache/run_longbench.sh
if [ "$MODEL_VARIANT" = "mistral" ]; then
    MODEL_PATH=${MODEL_PATH:-"unsloth/mistral-7b-instruct-v0.3"}
else
    MODEL_PATH=${MODEL_PATH:-"NousResearch/Meta-Llama-3-8B-Instruct"}
fi

MAX_CAPACITY=${MAX_CAPACITY:-1024}
TRIGGER_EVERY=${TRIGGER_EVERY:-128}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-7900}
OUTPUT_DIR=${OUTPUT_DIR:-"results/kv_cache/longbench"}

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  QORE KV-Cache Experiment: LongBench"
echo "=============================================="
echo "  Model:         $MODEL_PATH"
echo "  Max capacity:  $MAX_CAPACITY"
echo "  Trigger every: $TRIGGER_EVERY"
echo "  Max samples:   $MAX_SAMPLES (0=all)"
echo "  Max input len: $MAX_INPUT_LENGTH"
echo "  Output:        $OUTPUT_DIR"
echo "=============================================="

# Run all cache policies (real SOTA baselines: H2O, SnapKV, PyramidKV)
for POLICY in qore h2o snapkv pyramidkv window random full; do
    echo ""
    echo ">>> Running policy: $POLICY"
    python -m scripts.kv_cache.eval_kv_cache \
        --model_path "$MODEL_PATH" \
        --dataset "longbench" \
        --policy "$POLICY" \
        --max_capacity "$MAX_CAPACITY" \
        --trigger_every "$TRIGGER_EVERY" \
        --max_samples "$MAX_SAMPLES" \
        --max_input_length "$MAX_INPUT_LENGTH" \
        --output_dir "$OUTPUT_DIR" \
        --output_file "${POLICY}.json"
done

# Generate comparison table
echo ""
echo ">>> Generating summary..."
python -m scripts.kv_cache.summarize --input_dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/summary.csv"

echo ""
echo "Done! Results in $OUTPUT_DIR/"
