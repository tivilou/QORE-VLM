#!/bin/bash
# Idea 7 重新评估 - 在 Idea 6 实现后测量新的 oracle gap

set -e

# 使用 git 找到仓库根目录并切换到根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# 设置 PYTHONPATH 使 Python 能找到 scripts 模块
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# 实验配置
TIMESTAMP=$(TZ='Asia/Shanghai' date +%Y%m%dT%H%M%S)
OUTPUT_DIR="$REPO_ROOT/exchange/p2_idea7_diagnosis/${TIMESTAMP}"
GAMMA=0.5
DELTA=0.1
SAMPLES=200
SEED=42
LAM=2.0
CORPUS_MODE="wiki_dpr"
DATASET="nq_open"
METHOD="qore"
K=5

echo "=========================================="
echo "Idea 7 诊断实验: $TIMESTAMP"
echo "=========================================="
echo "配置: γ=${GAMMA}, δ=${DELTA} (Idea 6 推荐配置)"
echo "样本数: ${SAMPLES}"
echo "输出: ${OUTPUT_DIR}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行评估（带 dump_passages 以获取 QUBO 数据）
echo "步骤 1/3: 运行 RAG 评估（含 QUBO 数据）..."
echo ">>> 命令: python -m scripts.rag.eval_rag_refactored ..."
echo ""

python -m scripts.rag.eval_rag_refactored \
    --corpus_mode ${CORPUS_MODE} \
    --dataset ${DATASET} \
    --max_samples ${SAMPLES} \
    --method ${METHOD} \
    --selector_k ${K} \
    --qore_prefilter_size 15 \
    --qore_lambda ${LAM} \
    --qore_gamma ${GAMMA} \
    --delta ${DELTA} \
    --complementarity_method dpr \
    --use_answer_scorer \
    --seed ${SEED} \
    --dump_passages \
    --output "${OUTPUT_DIR}/result.json"

if [ $? -ne 0 ]; then
    echo "✗ 评估失败"
    exit 1
fi

echo ""
echo "✓ 评估完成"
echo ""
echo "步骤 2/3: 运行 QUBO 目标诊断..."
python "$REPO_ROOT/scripts/diagnosis/qubo_objective_diagnosis.py" \
    --results "${OUTPUT_DIR}/result.json" \
    --output "${OUTPUT_DIR}/qubo_diagnosis.md" \
    --gamma ${GAMMA} \
    --lam ${LAM}

if [ $? -ne 0 ]; then
    echo "✗ 诊断失败"
    exit 1
fi

echo ""
echo "✓ 诊断完成"
echo ""
echo "步骤 3/3: 生成对比报告..."
cat > "${OUTPUT_DIR}/README.md" <<EOF
# Idea 7 重新评估 - ${TIMESTAMP}

## 背景

在 Phase 1 诊断中，Idea 7 (QUBO 代理目标) 显示：
- QUBO 最优解平均 Recall: 0.5909
- Oracle 最优解平均 Recall: 0.8913
- **Gap**: 0.3004 (命中率 34.7%)

**问题**: Idea 6 和 Idea 7 之前共享同一个 oracle。现在 Idea 6 已实现并取得显著效果：
- Recall@5: 0.3196 → 0.4454 (+37.9%)
- F1: 0.4406 → 0.5092 (+13.6%)

本实验目的：测量实现 Idea 6 后，Idea 7 的 oracle gap 是否仍然显著。

## 实验配置

- **γ**: ${GAMMA} (Idea 6 推荐值)
- **δ**: ${DELTA} (Idea 6 推荐值)
- **λ**: 2.0
- **样本数**: ${SAMPLES}
- **种子**: ${SEED}
- **Complementarity**: DPR answer scorer

## 结果

详见 \`qubo_diagnosis.md\`

## 对比分析

| 指标 | Phase 1 (无 Idea 6) | Phase 2 (含 Idea 6) | 变化 |
|------|---------------------|---------------------|------|
| QUBO 最优 Recall | 0.5909 | TODO | TODO |
| Oracle Recall | 0.8913 | TODO | TODO |
| Gap | 0.3004 | TODO | TODO |
| 命中率 | 34.7% | TODO | TODO |

## 决策建议

- [ ] **Gap 仍显著 (>0.1)**: 考虑实现 Idea 7
- [ ] **Gap 已缩小 (<0.1)**: Idea 6 已充分优化，暂不做 Idea 7
- [ ] **需要更多数据**: 扩大样本量或调整参数

---

*生成于 ${TIMESTAMP}*
EOF

echo ""
echo "=========================================="
echo "✓ Idea 7 诊断完成！"
echo "=========================================="
echo ""
echo "结果目录: ${OUTPUT_DIR}"
echo ""
echo "请查看:"
echo "  - ${OUTPUT_DIR}/qubo_diagnosis.md (诊断报告)"
echo "  - ${OUTPUT_DIR}/README.md (对比分析)"
echo ""
echo "下一步："
echo "  1. 查看诊断报告: cat ${OUTPUT_DIR}/qubo_diagnosis.md"
echo "  2. git add exchange/p2_idea7_diagnosis/${TIMESTAMP}/"
echo "  3. git commit -m 'experiment(idea7): re-evaluation after idea6 implementation'"
echo "  4. git push"
echo "=========================================="

