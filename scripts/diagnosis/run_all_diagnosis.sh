#!/bin/bash
"""
Phase 1 完整诊断脚本

运行所有诊断分析，验证改进方案的假设

诊断内容:
1. γ sweep - 验证两阶段 QUBO 假设
2. 答案多样性分析 - 验证答案多样性约束假设
3. 查询类型分析 - 验证 Query-adaptive 假设
4. 上下文依赖分析 - 验证上下文完整性建模假设
5. QUBO 目标分析 - 验证 Soft QUBO 假设
"""

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║             Phase 1 完整诊断 - 假设验证                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 配置
RESULTS_DIR="scratch/research/P1_diagnosis/experiments"
ANALYSIS_DIR="scratch/research/P1_diagnosis/analysis"
GAMMA_0_5_RESULT="$RESULTS_DIR/gamma_0.5/result.json"

# 创建输出目录
mkdir -p "$ANALYSIS_DIR"

# ============================================================================
# 1. γ Sweep (已由 tuning framework 运行)
# ============================================================================
echo "[1/5] ✅ γ Sweep"
echo "      已由 run_tuning_suite.py 完成"
echo "      结果: $RESULTS_DIR/gamma_*/"
echo ""

# ============================================================================
# 2. 答案多样性分析
# ============================================================================
echo "[2/5] 📊 答案多样性分析"
echo "      验证假设: 文本多样但答案重复"

if [ -f "$GAMMA_0_5_RESULT" ]; then
    python scripts/diagnosis/answer_diversity_diagnosis.py \
        --results "$GAMMA_0_5_RESULT" \
        --output "$ANALYSIS_DIR/answer_diversity.md"

    echo "      ✅ 完成"
else
    echo "      ⚠️  结果文件不存在: $GAMMA_0_5_RESULT"
    echo "      请先运行 Phase 1 实验"
fi
echo ""

# ============================================================================
# 3. 查询类型分析
# ============================================================================
echo "[3/5] 🔍 查询类型分析"
echo "      验证假设: 不同查询需要不同多样性策略"

if [ -d "$RESULTS_DIR" ]; then
    python scripts/diagnosis/query_type_diagnosis.py \
        --results_dir "$RESULTS_DIR" \
        --output "$ANALYSIS_DIR/query_type.md"

    echo "      ✅ 完成"
else
    echo "      ⚠️  结果目录不存在: $RESULTS_DIR"
fi
echo ""

# ============================================================================
# 4. 上下文依赖分析
# ============================================================================
echo "[4/5] 🔗 上下文依赖分析"
echo "      验证假设: 独立选择破坏段落间依赖"

if [ -f "$GAMMA_0_5_RESULT" ]; then
    python scripts/diagnosis/context_dependency_diagnosis.py \
        --results "$GAMMA_0_5_RESULT" \
        --output "$ANALYSIS_DIR/context_dependency.md"

    echo "      ✅ 完成"
else
    echo "      ⚠️  结果文件不存在: $GAMMA_0_5_RESULT"
fi
echo ""

# ============================================================================
# 5. QUBO 目标分析
# ============================================================================
echo "[5/5] 🎯 QUBO 目标分析"
echo "      验证假设: QUBO 目标与 F1 不一致"

if [ -f "$GAMMA_0_5_RESULT" ]; then
    python scripts/diagnosis/qubo_objective_diagnosis.py \
        --results "$GAMMA_0_5_RESULT" \
        --output "$ANALYSIS_DIR/qubo_objective.md" \
        --gamma 0.5 \
        --lam 2.0

    echo "      ✅ 完成"
else
    echo "      ⚠️  结果文件不存在: $GAMMA_0_5_RESULT"
fi
echo ""

# ============================================================================
# 生成综合报告
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      诊断完成                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 所有分析报告位于: $ANALYSIS_DIR/"
echo ""
echo "   1. answer_diversity.md    - 答案多样性分析"
echo "   2. query_type.md          - 查询类型分析"
echo "   3. context_dependency.md  - 上下文依赖分析"
echo "   4. qubo_objective.md      - QUBO 目标分析"
echo ""
echo "下一步:"
echo "  1. 查看各诊断报告"
echo "  2. 根据假设验证结果决定 Phase 2-3 实施顺序"
echo "  3. 见 improvement_roadmap.md"
echo ""
