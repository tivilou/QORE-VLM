#!/bin/bash
# Run all RAG experiments
set -e

echo "===== RAG Full Experiment Suite ====="
echo ""

echo "[1/4] Natural Questions (LLaMA-3-8B)"
bash scripts/rag/run_nq.sh

echo ""
echo "[2/4] HotpotQA (LLaMA-3-8B)"
DATASET=hotpotqa OUTPUT_DIR=results/rag/hotpotqa bash scripts/rag/run_nq.sh

echo ""
echo "[3/4] Natural Questions (Mistral-7B)"
OUTPUT_DIR=results/rag/nq_mistral bash scripts/rag/run_nq.sh mistral

echo ""
echo "[4/4] Budget sweep K=3,5,8,10,15"
for K in 3 5 8 10 15; do
    echo "  K=$K"
    K=$K OUTPUT_DIR="results/rag/sweep_K/K${K}" bash scripts/rag/run_nq.sh
done

echo ""
echo "===== All RAG experiments complete ====="
