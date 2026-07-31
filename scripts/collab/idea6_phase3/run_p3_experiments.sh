#!/bin/bash
# Idea 6 Phase 3 全量实验运行脚本
# 在完整 validation 集 (3610 samples) 上验证 Idea 6 效果
# 使用 3 个随机种子确保结果稳定性

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
OUTPUT_BASE="exchange/p3_solver_idea6/${TIMESTAMP}"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║   Idea 6 Phase 3 - 全量验证实验                                  ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "实验参数:"
echo "  - 数据集: nq_open validation (3610 samples)"
echo "  - Seeds: 42, 43, 44"
echo "  - 配置: baseline + idea6 (推荐) + idea6 (最佳)"
echo "  - 输出: ${OUTPUT_BASE}"
echo ""

mkdir -p "${OUTPUT_BASE}"

# Record git status
echo "记录 Git 状态..."
git rev-parse HEAD > "${OUTPUT_BASE}/git_commit.txt"
git status --porcelain > "${OUTPUT_BASE}/git_status.txt" || true
git diff > "${OUTPUT_BASE}/git_diff.patch" || true

# Seeds to test
SEEDS=(42 43 44)

# Configurations to test
CONFIGS=(
    "baseline_phase3:baseline"
    "idea6_phase3_recommended:idea6_recommended"
    "idea6_phase3_best:idea6_best"
)

echo "开始实验..."
echo ""

for seed in "${SEEDS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Seed: ${seed}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    for config_entry in "${CONFIGS[@]}"; do
        IFS=':' read -r config_file config_name <<< "$config_entry"

        echo "→ 运行: ${config_name} (seed=${seed})"

        output_dir="${OUTPUT_BASE}/seed_${seed}/${config_name}"
        mkdir -p "${output_dir}"

        # 根据配置名称设置参数
        case "$config_name" in
            "baseline")
                # Baseline: gamma=0.5, delta=0.0 (no complementarity)
                python -m scripts.rag.eval.eval_rag_refactored \
                    --dataset nq_open \
                    --split validation \
                    --max_samples 0 \
                    --corpus_mode aligned \
                    --n_distractors 36000 \
                    --method qore \
                    --K 5 \
                    --lam 2.0 \
                    --gamma 0.5 \
                    --skip_generation \
                    --seed "${seed}" \
                    --output_dir "${output_dir}" \
                    2>&1 | tee "${output_dir}/log.txt"
                ;;
            "idea6_recommended")
                # Idea 6 推荐: gamma=0.5, delta=0.1
                python -m scripts.rag.eval.eval_rag_refactored \
                    --dataset nq_open \
                    --split validation \
                    --max_samples 0 \
                    --corpus_mode aligned \
                    --n_distractors 36000 \
                    --method qore \
                    --K 5 \
                    --lam 2.0 \
                    --gamma 0.5 \
                    --delta 0.1 \
                    --complementarity_method dpr \
                    --use_answer_scorer \
                    --skip_generation \
                    --seed "${seed}" \
                    --output_dir "${output_dir}" \
                    2>&1 | tee "${output_dir}/log.txt"
                ;;
            "idea6_best")
                # Idea 6 最佳: gamma=0.3, delta=0.1
                python -m scripts.rag.eval.eval_rag_refactored \
                    --dataset nq_open \
                    --split validation \
                    --max_samples 0 \
                    --corpus_mode aligned \
                    --n_distractors 36000 \
                    --method qore \
                    --K 5 \
                    --lam 2.0 \
                    --gamma 0.3 \
                    --delta 0.1 \
                    --complementarity_method dpr \
                    --use_answer_scorer \
                    --skip_generation \
                    --seed "${seed}" \
                    --output_dir "${output_dir}" \
                    2>&1 | tee "${output_dir}/log.txt"
                ;;
            *)
                echo "  ✗ 未知配置: ${config_name}"
                exit 1
                ;;
        esac

        echo "  ✓ 完成: ${config_name}"
        echo ""
    done

    echo ""
done

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║   ✅ 所有实验完成！                                               ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "结果保存在: ${OUTPUT_BASE}"
echo ""
echo "下一步："
echo "1. 查看结果: ls -la ${OUTPUT_BASE}/seed_*/"
echo "2. 分析结果: python scripts/collab/idea6_phase3/analyze_p3_results.py ${OUTPUT_BASE}"
echo "3. 打包结果: python scripts/collab/idea6_phase3/package_p3_results.py ${OUTPUT_BASE}"
echo ""
