#!/bin/bash
# Phase 1 诊断 - 完整实验（200题）

set -e

echo "=== Phase 1 诊断 - 完整实验 ==="
echo "预计耗时: 1-2 小时"
echo ""

read -p "确认运行完整实验？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消运行"
    exit 0
fi

# 切换到项目根目录
cd "$(dirname "$0")/../.."

python scripts/tuning/run_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml

echo ""
echo "✅ 完成！查看结果："
echo "  cat scratch/research/P1_diagnosis/analysis/analysis.md"
echo ""
echo "📦 打包文件："
echo "  ls scratch/research/P1_diagnosis/package/"
