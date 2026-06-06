#!/bin/bash
# Run all ablation experiments
set -e

echo "===== Ablation Experiment Suite ====="

echo ""
echo "[1/4] Lambda sensitivity"
for LAM in 0.5 1.0 2.0 5.0 10.0; do
    echo "  lam=$LAM"
    python -m scripts.rag.eval_rag \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset natural_questions \
        --method qore \
        --K 5 \
        --lam $LAM \
        --max_samples 200 \
        --output_dir "results/ablations/lambda" \
        --output_file "lam_${LAM}.json"
done

echo ""
echo "[2/4] QAOA depth"
for P in 1 2 3 4; do
    echo "  p=$P"
    python -m scripts.rag.eval_rag \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset natural_questions \
        --method qore \
        --solver qaoa_tc \
        --K 5 \
        --max_samples 100 \
        --output_dir "results/ablations/qaoa_depth" \
        --output_file "p_${P}.json"
done

echo ""
echo "[3/4] Block size (KV-Cache)"
for BS in 16 32 48 64; do
    echo "  block_size=$BS"
    python -m scripts.kv_cache.eval_kv_cache \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset longbench \
        --policy qore \
        --max_capacity 1024 \
        --max_samples 100 \
        --output_dir "results/ablations/block_size" \
        --output_file "bs_${BS}.json"
done

echo ""
echo "[4/4] Trigger interval (KV-Cache)"
for T in 32 64 128 256; do
    echo "  trigger_every=$T"
    python -m scripts.kv_cache.eval_kv_cache \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset longbench \
        --policy qore \
        --max_capacity 1024 \
        --trigger_every $T \
        --max_samples 100 \
        --output_dir "results/ablations/trigger" \
        --output_file "T_${T}.json"
done

echo ""
echo "===== All ablation experiments complete ====="
