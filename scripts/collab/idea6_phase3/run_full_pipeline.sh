#!/bin/bash
# Idea 6 Phase 3 - 全流程自动化脚本
# 一键完成：实验运行 → 结果分析 → 结果打包

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
OUTPUT_BASE="exchange/p3_solver_idea6/${TIMESTAMP}"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║   Idea 6 Phase 3 - 全流程自动化                                  ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "这个脚本会自动完成以下步骤："
echo "  1. 运行所有实验 (3 configs × 3 seeds = 9 runs)"
echo "  2. 分析结果并生成报告"
echo "  3. 打包结果用于 GitHub 提交"
echo ""
echo "实验参数:"
echo "  - 数据集: nq_open validation (3610 samples)"
echo "  - Seeds: 42, 43, 44"
echo "  - 配置: baseline + idea6 (推荐) + idea6 (最佳)"
echo "  - 输出: ${OUTPUT_BASE}"
echo ""
echo "预计总时间: 4.5-9 小时"
echo ""
read -p "按 Enter 开始，或 Ctrl+C 取消..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 1/3: 运行实验"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

mkdir -p "${OUTPUT_BASE}"

# 记录 Git 状态
echo "记录 Git 状态..."
git rev-parse HEAD > "${OUTPUT_BASE}/git_commit.txt"
git status --porcelain > "${OUTPUT_BASE}/git_status.txt" || true
git diff > "${OUTPUT_BASE}/git_diff.patch" || true
echo ""

# Seeds to test
SEEDS=(42 43 44)

# Configurations to test
CONFIGS=(
    "baseline_phase3:baseline"
    "idea6_phase3_recommended:idea6_recommended"
    "idea6_phase3_best:idea6_best"
)

# 运行所有实验
TOTAL_RUNS=$((${#SEEDS[@]} * ${#CONFIGS[@]}))
CURRENT_RUN=0

for seed in "${SEEDS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Seed: ${seed}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    for config_entry in "${CONFIGS[@]}"; do
        IFS=':' read -r config_file config_name <<< "$config_entry"

        ((CURRENT_RUN++))
        echo "[${CURRENT_RUN}/${TOTAL_RUNS}] 运行: ${config_name} (seed=${seed})"

        output_dir="${OUTPUT_BASE}/seed_${seed}/${config_name}"
        mkdir -p "${output_dir}"

        python -m scripts.rag.eval.eval_rag_refactored \
            --config "configs/experiments/${config_file}.yaml" \
            --seed "${seed}" \
            --output_dir "${output_dir}" \
            2>&1 | tee "${output_dir}/log.txt"

        echo "  ✓ 完成 [${CURRENT_RUN}/${TOTAL_RUNS}]"
        echo ""
    done

    echo ""
done

# 计算实验用时
END_TIME=$(date +%s)
EXPERIMENT_DURATION=$((END_TIME - START_TIME))
HOURS=$((EXPERIMENT_DURATION / 3600))
MINUTES=$(((EXPERIMENT_DURATION % 3600) / 60))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 2/3: 分析结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "实验完成！用时: ${HOURS}h ${MINUTES}m"
echo ""

# 分析结果
echo "分析结果..."
python scripts/collab/idea6_phase3/analyze_p3_results.py "${OUTPUT_BASE}" | tee "${OUTPUT_BASE}/analysis.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 3/3: 打包结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 打包结果
echo "打包结果..."
python scripts/collab/idea6_phase3/package_p3_results.py "${OUTPUT_BASE}"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║   ✅ 全流程完成！                                                 ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "结果保存在: ${OUTPUT_BASE}"
echo "实验用时: ${HOURS}h ${MINUTES}m"
echo ""
echo "生成的文件："
echo "  - ${OUTPUT_BASE}/README.md          (汇总报告)"
echo "  - ${OUTPUT_BASE}/analysis.txt       (详细分析)"
echo "  - ${OUTPUT_BASE}/results.zip        (压缩结果)"
echo "  - ${OUTPUT_BASE}/seed_XX/CONFIG/    (详细结果)"
echo ""
echo "下一步："
echo "  1. 查看分析: cat ${OUTPUT_BASE}/analysis.txt"
echo "  2. 查看报告: cat ${OUTPUT_BASE}/README.md"
echo "  3. 提交到 GitHub: git add ${OUTPUT_BASE} && git commit"
echo ""
echo "🎉 大功告成！"
echo ""
