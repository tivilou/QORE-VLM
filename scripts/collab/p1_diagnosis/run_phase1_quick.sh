#!/bin/bash
# Phase 1 诊断 - 快速测试（10题）

set -e

echo "=== Phase 1 诊断 - 快速测试 ==="
echo "计算部分约 5 分钟，加上 wiki_dpr 索引和 answer scorer 的加载会更长"
echo "（单配置超时阀值 30 分钟）"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/../.."

python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/quick_test.yaml

echo ""
echo "✅ 完成！下一步必须检查字段（不要跳）："
echo "  python scripts/diagnosis/check_dump_fields.py --expect_samples 10"
echo ""
echo "⚠️ 10 题只能验证流程跑通，不足以支撑任何诊断结论"
