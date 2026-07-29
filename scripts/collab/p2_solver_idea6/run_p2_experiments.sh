#!/bin/bash
# 自动运行 Phase 2 实验：solver fix + idea 6 调参网格
# 用法: bash scripts/collab/run_p2_experiments.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# 设置 PYTHONPATH 使 Python 能找到 scripts 模块
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# 配置
CORPUS_MODE="wiki_dpr"  # 使用 wiki_dpr 模式（已验证可用，更真实）
DATASET="nq_open"
MAX_SAMPLES=200
METHOD="qore"
K=5
LAM=2.0
SEED=42
OUTPUT_BASE="$REPO_ROOT/exchange/p2_solver_idea6"
WIKI_DPR_CONFIG="psgs_w100.nq.compressed"
WIKI_DPR_CACHE_DIR="/root/.cache/huggingface/datasets"

# 时间戳（北京时间）
TIMESTAMP=$(TZ='Asia/Shanghai' date +%Y%m%dT%H%M%S)
echo "=========================================="
echo "Phase 2 实验批次: $TIMESTAMP"
echo "=========================================="
echo

# 检查依赖
echo ">>> 检查 eval_rag_refactored.py 参数支持..."
if ! python -m scripts.rag.eval_rag_refactored --help | grep -q "\-\-delta"; then
    echo "ERROR: eval_rag_refactored.py 不支持 --delta 参数，请先添加"
    exit 1
fi
if ! python -m scripts.rag.eval_rag_refactored --help | grep -q "\-\-complementarity_method"; then
    echo "ERROR: eval_rag_refactored.py 不支持 --complementarity_method 参数，请先添加"
    exit 1
fi
echo "✓ 参数支持检查通过"
echo

# 实验计数
TOTAL_RUNS=10
CURRENT_RUN=0

# 函数：运行单个实验
run_experiment() {
    local gamma=$1
    local delta=$2
    local use_complementarity=$3

    CURRENT_RUN=$((CURRENT_RUN + 1))

    local config_name="gamma${gamma}_delta${delta}"
    local output_dir="${OUTPUT_BASE}/${TIMESTAMP}/${config_name}"

    echo "=========================================="
    echo "[$CURRENT_RUN/$TOTAL_RUNS] 运行配置: $config_name"
    echo "  gamma=$gamma, delta=$delta"
    if [ "$use_complementarity" = "true" ]; then
        echo "  complementarity=dpr, use_answer_scorer=true"
    else
        echo "  complementarity=none (solver fix baseline)"
    fi
    echo "=========================================="

    mkdir -p "$output_dir"

    # 构建命令
    cmd="python -m scripts.rag.eval_rag_refactored \
        --corpus_mode $CORPUS_MODE \
        --dataset $DATASET \
        --max_samples $MAX_SAMPLES \
        --method $METHOD \
        --K $K \
        --lam $LAM \
        --gamma $gamma \
        --delta $delta \
        --seed $SEED \
        --output_dir $output_dir \
        --output_file result.json"

    # wiki_dpr 模式需要额外参数
    if [ "$CORPUS_MODE" = "wiki_dpr" ]; then
        cmd="$cmd --wiki_dpr_config $WIKI_DPR_CONFIG"
        cmd="$cmd --wiki_dpr_cache_dir $WIKI_DPR_CACHE_DIR"
    fi

    if [ "$use_complementarity" = "true" ]; then
        cmd="$cmd --complementarity_method dpr --use_answer_scorer"
    fi

    echo ">>> 命令: $cmd"
    echo

    # 运行实验
    eval "$cmd"

    if [ $? -eq 0 ]; then
        echo "✓ 实验完成: $config_name"
    else
        echo "✗ 实验失败: $config_name"
        exit 1
    fi
    echo
}

# ============================================
# 实验 1: Baseline (solver fix only, delta=0.0)
# ============================================
run_experiment 0.5 0.0 false

# ============================================
# 实验 2-10: Idea 6 调参网格
# ============================================
for gamma in 0.3 0.5 0.7; do
    for delta in 0.1 0.3 0.5; do
        run_experiment $gamma $delta true
    done
done

echo "=========================================="
echo "✓ 所有实验完成！"
echo "批次时间戳: $TIMESTAMP"
echo "结果目录: $OUTPUT_BASE/$TIMESTAMP/"
echo ""
echo "正在汇总结果..."
echo "=========================================="
echo

# 自动运行汇总脚本
python "$SCRIPT_DIR/collect_p2_results.py" "$TIMESTAMP"

if [ $? -eq 0 ]; then
    echo
    echo "=========================================="
    echo "✓ 实验和汇总全部完成！"
    echo ""
    echo "下一步："
    echo "  1. 查看结果: cat $OUTPUT_BASE/$TIMESTAMP/README.md"
    echo "  2. git add exchange/p2_solver_idea6/$TIMESTAMP/"
    echo "  3. git commit -m 'experiment(p2): solver+idea6 results $TIMESTAMP'"
    echo "  4. git push"
    echo "=========================================="
else
    echo
    echo "=========================================="
    echo "✗ 汇总失败，请检查错误信息"
    echo "可手动重试: python scripts/collab/p2_solver_idea6/collect_p2_results.py $TIMESTAMP"
    echo "=========================================="
    exit 1
fi
