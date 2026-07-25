#!/bin/bash
# Phase 1 诊断 - 快速测试（10题）

set -e

echo "=== Phase 1 诊断 - 快速测试 ==="
echo "预计耗时: 5 分钟"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/../.."

python scripts/tuning/run_suite.py \
    --config scripts/tuning/config/quick_test.yaml

echo ""
echo "✅ 完成！查看结果："
echo "  cat scratch/research/quick_test/analysis/analysis.md"
