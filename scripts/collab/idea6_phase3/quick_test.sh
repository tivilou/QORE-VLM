#!/bin/bash
# Quick test script for Phase 3 configuration validation
# Runs experiments on a small sample (10 items) to verify setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║   Phase 3 Quick Test (10 samples)                               ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "目的: 验证配置文件和环境设置"
echo "样本数: 10 (测试用)"
echo "预计时间: 5-10 分钟"
echo ""

OUTPUT_DIR="exchange/p3_quick_test/$(date +%Y%m%dT%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

echo "开始快速测试..."
echo ""

# Test baseline
echo "→ 测试 Baseline..."
python -m scripts.rag.eval.eval_rag_refactored \
    --dataset nq_open \
    --split validation \
    --max_samples 10 \
    --method qore \
    --K 5 \
    --lam 2.0 \
    --gamma 0.5 \
    --corpus_mode wiki_dpr \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}/baseline" \
    2>&1 | tee "${OUTPUT_DIR}/baseline.log"

echo "  ✓ Baseline 完成"
echo ""

# Test Idea 6
echo "→ 测试 Idea 6..."
python -m scripts.rag.eval.eval_rag_refactored \
    --dataset nq_open \
    --split validation \
    --max_samples 10 \
    --method qore \
    --K 5 \
    --lam 2.0 \
    --gamma 0.5 \
    --delta 0.1 \
    --complementarity_method dpr \
    --corpus_mode wiki_dpr \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}/idea6" \
    2>&1 | tee "${OUTPUT_DIR}/idea6.log"

echo "  ✓ Idea 6 完成"
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║   ✅ 快速测试完成！                                               ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "结果保存在: ${OUTPUT_DIR}"
echo ""
echo "如果测试通过，可以运行完整实验:"
echo "  bash scripts/collab/idea6_phase3/run_p3_experiments.sh"
echo ""
