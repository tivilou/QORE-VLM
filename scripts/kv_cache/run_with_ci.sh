#!/bin/bash
# Multi-run evaluation with bootstrap CI for statistically rigorous comparison.
# Usage: bash scripts/kv_cache/run_with_ci.sh

set -e

POLICIES="qore h2o snapkv"
SEEDS="42 43 44 45 46"
MAX_SAMPLES=30
MODEL="NousResearch/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="results/kv_cache/longbench_ci"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Running KV-cache eval with 5 seeds for bootstrap CI"
echo "============================================================"

for policy in $POLICIES; do
  echo ""
  echo ">>> Policy: $policy"
  for seed in $SEEDS; do
    echo "  Running seed $seed (log: $OUTPUT_DIR/${policy}_seed${seed}.log)..."
    python scripts/kv_cache/eval_kv_cache.py \
      --model_path "$MODEL" \
      --policy "$policy" \
      --seed "$seed" \
      --max_samples "$MAX_SAMPLES" \
      --max_capacity 1024 \
      --trigger_every 128 \
      --num_sink_tokens 4 \
      --output_dir "$OUTPUT_DIR" \
      --output_file "${policy}_seed${seed}.json" \
      > "$OUTPUT_DIR/${policy}_seed${seed}.log" 2>&1
    echo "    Completed."
  done

  echo "  Aggregating with bootstrap CI..."
  python scripts/kv_cache/bootstrap_ci.py \
    "$OUTPUT_DIR/${policy}_seed"*.json \
    --output "$OUTPUT_DIR/${policy}_summary.json"
done

echo ""
echo "============================================================"
echo "Done. Summaries in $OUTPUT_DIR/*_summary.json"
echo "============================================================"
