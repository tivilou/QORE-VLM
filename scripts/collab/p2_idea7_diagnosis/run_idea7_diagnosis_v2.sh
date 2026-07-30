#!/bin/bash
# Idea 7 重新评估 - 增强版，带详细错误检查和日志

set -e  # 任何命令失败立即退出

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

LOG_FILE="${OUTPUT_DIR}/run.log"

echo "=========================================="
echo "Idea 7 诊断实验: $TIMESTAMP"
echo "=========================================="
echo "配置: γ=${GAMMA}, δ=${DELTA} (Idea 6 推荐配置)"
echo "样本数: ${SAMPLES}"
echo "输出: ${OUTPUT_DIR}"
echo "日志: ${LOG_FILE}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 初始化日志
cat > "${LOG_FILE}" <<EOF
Idea 7 诊断实验日志
时间: ${TIMESTAMP}
配置: γ=${GAMMA}, δ=${DELTA}, samples=${SAMPLES}
========================================

EOF

# 步骤 1: 运行评估
echo "步骤 1/3: 运行 RAG 评估（含 QUBO 数据）..."
echo ">>> 开始时间: $(date +%H:%M:%S)"
echo ""

echo "步骤 1 开始: $(date)" >> "${LOG_FILE}"

python -m scripts.rag.eval_rag_refactored \
    --corpus_mode ${CORPUS_MODE} \
    --dataset ${DATASET} \
    --max_samples ${SAMPLES} \
    --method ${METHOD} \
    --K ${K} \
    --qore_prefilter_size 15 \
    --lam ${LAM} \
    --gamma ${GAMMA} \
    --delta ${DELTA} \
    --complementarity_method dpr \
    --use_answer_scorer \
    --seed ${SEED} \
    --dump_passages \
    --output_file result.json \
    --output_dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "✗ 评估失败！退出码: $EVAL_EXIT_CODE" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    echo "请查看日志: ${LOG_FILE}"
    exit 1
fi

echo "" | tee -a "${LOG_FILE}"
echo "步骤 1 完成: $(date)" >> "${LOG_FILE}"

# 验证 result.json 存在且有效
RESULT_FILE="${OUTPUT_DIR}/result.json"
if [ ! -f "${RESULT_FILE}" ]; then
    echo "✗ 错误: result.json 文件不存在！" | tee -a "${LOG_FILE}"
    echo "预期路径: ${RESULT_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi

RESULT_SIZE=$(stat -f%z "${RESULT_FILE}" 2>/dev/null || stat -c%s "${RESULT_FILE}" 2>/dev/null)
if [ "$RESULT_SIZE" -lt 1000 ]; then
    echo "✗ 错误: result.json 文件太小 (${RESULT_SIZE} bytes)，可能是空文件！" | tee -a "${LOG_FILE}"
    exit 1
fi

# 检查样本数
ACTUAL_SAMPLES=$(python3 -c "import json; data=json.load(open('${RESULT_FILE}')); print(len(data.get('samples', [])))" 2>/dev/null || echo "0")
echo "✓ result.json 已生成 (${RESULT_SIZE} bytes, ${ACTUAL_SAMPLES} 样本)" | tee -a "${LOG_FILE}"

if [ "$ACTUAL_SAMPLES" != "$SAMPLES" ]; then
    echo "⚠️  警告: 实际样本数 (${ACTUAL_SAMPLES}) 与配置 (${SAMPLES}) 不符" | tee -a "${LOG_FILE}"
fi

echo ""
echo "✓ 评估完成"
echo ">>> 完成时间: $(date +%H:%M:%S)"
echo ""

# 步骤 2: 运行诊断
echo "步骤 2/3: 运行 QUBO 目标诊断..."
echo ">>> 开始时间: $(date +%H:%M:%S)"
echo ""

echo "步骤 2 开始: $(date)" >> "${LOG_FILE}"

python "$REPO_ROOT/scripts/diagnosis/qubo_objective_diagnosis.py" \
    --results "${RESULT_FILE}" \
    --output "${OUTPUT_DIR}/qubo_diagnosis.md" \
    --gamma ${GAMMA} \
    --lam ${LAM} 2>&1 | tee -a "${LOG_FILE}"

DIAG_EXIT_CODE=${PIPESTATUS[0]}

if [ $DIAG_EXIT_CODE -ne 0 ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "✗ 诊断失败！退出码: $DIAG_EXIT_CODE" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    echo "请查看日志: ${LOG_FILE}"
    exit 1
fi

echo "" | tee -a "${LOG_FILE}"
echo "步骤 2 完成: $(date)" >> "${LOG_FILE}"

# 验证诊断报告生成
if [ ! -f "${OUTPUT_DIR}/qubo_diagnosis.md" ]; then
    echo "✗ 错误: qubo_diagnosis.md 文件不存在！" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "✓ 诊断完成"
echo ">>> 完成时间: $(date +%H:%M:%S)"
echo ""

# 步骤 3: 生成对比报告
echo "步骤 3/3: 生成对比报告..."

# 从诊断报告中提取关键指标
QUBO_RECALL=$(grep "QUBO 能量最低的子集.*平均 Recall" "${OUTPUT_DIR}/qubo_diagnosis.md" | grep -oE "[0-9]+\.[0-9]+")
ORACLE_RECALL=$(grep "枚举出的最优子集.*平均 Recall" "${OUTPUT_DIR}/qubo_diagnosis.md" | grep -oE "[0-9]+\.[0-9]+")
GAP=$(grep "平均差距.*:" "${OUTPUT_DIR}/qubo_diagnosis.md" | grep -oE "[0-9]+\.[0-9]+")
HIT_RATE=$(grep "QUBO 命中最优的题数" "${OUTPUT_DIR}/qubo_diagnosis.md" | grep -oE "[0-9]+\.[0-9]+%")

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
- **λ**: ${LAM}
- **样本数**: ${ACTUAL_SAMPLES}
- **种子**: ${SEED}
- **Complementarity**: DPR answer scorer

## 结果

详见 \`qubo_diagnosis.md\`

## 对比分析

| 指标 | Phase 1 (无 Idea 6) | Phase 2 (含 Idea 6) | 变化 |
|------|---------------------|---------------------|------|
| QUBO 最优 Recall | 0.5909 | ${QUBO_RECALL:-N/A} | TODO |
| Oracle Recall | 0.8913 | ${ORACLE_RECALL:-N/A} | TODO |
| Gap | 0.3004 | ${GAP:-N/A} | TODO |
| 命中率 | 34.7% | ${HIT_RATE:-N/A} | TODO |

## 决策建议

- [ ] **Gap 仍显著 (>0.15)**: 强烈建议实现 Idea 7
- [ ] **Gap 中等 (0.08-0.15)**: 考虑实现 Idea 7（权衡成本）
- [ ] **Gap 较小 (<0.08)**: Idea 6 已充分优化，暂不做 Idea 7

## 文件清单

- \`result.json\` - RAG 评估完整结果 (${RESULT_SIZE} bytes, ${ACTUAL_SAMPLES} 样本)
- \`qubo_diagnosis.md\` - QUBO 诊断报告
- \`run.log\` - 运行日志

---

*生成于 ${TIMESTAMP}*
EOF

echo "✓ 对比报告生成完成"
echo ""

echo "=========================================="
echo "✓ Idea 7 诊断完成！"
echo "=========================================="
echo ""
echo "结果目录: ${OUTPUT_DIR}"
echo ""
echo "关键结果:"
echo "  - QUBO 最优 Recall: ${QUBO_RECALL:-N/A}"
echo "  - Oracle Recall: ${ORACLE_RECALL:-N/A}"
echo "  - Gap: ${GAP:-N/A}"
echo "  - 命中率: ${HIT_RATE:-N/A}"
echo ""
echo "文件:"
echo "  - ${OUTPUT_DIR}/qubo_diagnosis.md (诊断报告)"
echo "  - ${OUTPUT_DIR}/README.md (对比分析)"
echo "  - ${OUTPUT_DIR}/result.json (${RESULT_SIZE} bytes)"
echo "  - ${OUTPUT_DIR}/run.log (运行日志)"
echo ""
echo "下一步："
echo "  1. 查看诊断报告: cat ${OUTPUT_DIR}/qubo_diagnosis.md"
echo "  2. git add exchange/p2_idea7_diagnosis/${TIMESTAMP}/"
echo "  3. git commit -m 'experiment(idea7): re-evaluation after idea6 (corrected)'"
echo "  4. git push"
echo "=========================================="
