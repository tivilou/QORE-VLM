#!/bin/bash
# Run all KV-Cache experiments
set -e

echo "===== KV-Cache Full Experiment Suite ====="
echo ""

echo "[1/4] LongBench (LLaMA-3-8B)"
bash scripts/kv_cache/run_longbench.sh

echo ""
echo "[2/4] LongBench (Mistral-7B)"
OUTPUT_DIR=results/kv_cache/longbench_mistral bash scripts/kv_cache/run_longbench.sh mistral

echo ""
echo "[3/4] Capacity sweep (512, 1024, 2048, 4096)"
for CAP in 512 1024 2048 4096; do
    echo "  capacity=$CAP"
    MAX_CAPACITY=$CAP OUTPUT_DIR="results/kv_cache/sweep_cap/cap${CAP}" \
        bash scripts/kv_cache/run_longbench.sh
done

echo ""
echo "[4/4] Perplexity (PG-19)"
# Ungated mirror (see run_longbench.sh); override MODEL_PATH for the official repo.
python -m scripts.kv_cache.eval_kv_cache \
    --model_path "${MODEL_PATH:-NousResearch/Meta-Llama-3-8B-Instruct}" \
    --dataset pg19 \
    --policy qore \
    --output_dir results/kv_cache/perplexity \
    --output_file qore.json

echo ""
echo "===== All KV-Cache experiments complete ====="
