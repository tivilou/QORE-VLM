#!/bin/bash
# Phase 1 诊断 - 完整实验（200题）

set -e

echo "=== Phase 1 诊断 - 完整实验 ==="
echo "预计耗时: 1.5-2 小时（3 配置 × 200 题，含 LLM 生成）"
echo ""

read -p "确认运行完整实验？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消运行"
    exit 0
fi

# 切换到项目根目录
cd "$(dirname "$0")/../.."

python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml

echo ""
echo "✅ 实验完成！下一步跑 5 个诊断分析（约15分钟，仅需CPU）："
echo "  bash scripts/diagnosis/run_all_diagnosis.sh"
echo ""
echo "然后查看报告："
echo "  ls scratch/research/P1_diagnosis/analysis/"
echo ""
echo "📦 打包文件："
echo "  ls scratch/research/P1_diagnosis/package/"
