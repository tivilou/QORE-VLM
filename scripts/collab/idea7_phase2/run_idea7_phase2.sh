#!/bin/bash
# Idea 7 Phase 2: 真实数据训练一键脚本
#
# 用途：在 GPU 机器上训练 Soft QUBO，验证 Recall 提升
# 预计时间：GPU ~1 小时，CPU ~3 小时
#
# 使用方法：
#   cd ~/QORE-VLM
#   bash scripts/collab/idea7_phase2/run_idea7_phase2.sh

set -e  # 遇到错误立即退出

# ============================================================================
# 配置
# ============================================================================

# 时间戳（Asia/Shanghai）
export TZ=Asia/Shanghai
TIMESTAMP=$(date +%Y%m%dT%H%M%S)

# 输出目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
OUTPUT_DIR="$REPO_ROOT/exchange/idea7_phase2/${TIMESTAMP}"

# 实验参数
MAX_SAMPLES=200          # 训练样本数
EPOCHS=100              # 训练轮数
LEARNING_RATE=0.01      # 学习率
K=5                     # 选择 K 个 passages
SEED=42                 # 随机种子

# Phase 2 baseline 配置（用于生成训练数据）
GAMMA=0.5
DELTA=0.1
METHOD="qore"

echo "========================================"
echo "Idea 7 Phase 2: 真实数据训练"
echo "========================================"
echo "时间戳: ${TIMESTAMP}"
echo "输出目录: ${OUTPUT_DIR}"
echo "样本数: ${MAX_SAMPLES}"
echo "训练轮数: ${EPOCHS}"
echo ""

# ============================================================================
# Step 1: 准备环境
# ============================================================================

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/data_prep"

echo "[1/4] 准备环境..."
echo "  工作目录: $REPO_ROOT"
echo "  Python 路径: $PYTHONPATH"
echo ""

# ============================================================================
# Step 2: 生成训练数据（运行 eval_rag_refactored 获取 retrieved passages）
# ============================================================================

echo "[2/4] 生成训练数据（运行 RAG 评估获取候选 passages）..."
echo "  这可能需要 10-20 分钟..."

RESULT_JSON="$OUTPUT_DIR/data_prep/result.json"

python -m scripts.rag.eval.eval_rag_refactored \
    --dataset nq_open \
    --split validation \
    --max_samples ${MAX_SAMPLES} \
    --corpus_mode wiki_dpr \
    --corpus_output_dir data/nq_corpus \
    --method ${METHOD} \
    --K ${K} \
    --gamma ${GAMMA} \
    --delta ${DELTA} \
    --use_answer_scorer \
    --complementarity_method dpr \
    --skip_generation \
    --output_file "$RESULT_JSON" \
    --output_dir "$OUTPUT_DIR/data_prep" \
    2>&1 | tee "$OUTPUT_DIR/data_prep/eval.log"

if [ ! -f "$RESULT_JSON" ]; then
    echo "❌ 错误：result.json 未生成！"
    echo "   检查日志: $OUTPUT_DIR/data_prep/eval.log"
    exit 1
fi

RESULT_SIZE=$(stat -f%z "$RESULT_JSON" 2>/dev/null || stat -c%s "$RESULT_JSON" 2>/dev/null || echo 0)
echo "✅ result.json 已生成 ($(numfmt --to=iec-i --suffix=B $RESULT_SIZE 2>/dev/null || echo "${RESULT_SIZE} bytes"))"

# 检查有效样本数
VALID_SAMPLES=$(python -c "
import json
with open('$RESULT_JSON') as f:
    data = json.load(f)
# 统计有 retrieved 且有 gold 的样本
valid = 0
for item in data:
    if 'retrieved' in item and len(item['retrieved']) > 0:
        has_gold = any(p.get('is_gold', False) for p in item['retrieved'])
        if has_gold:
            valid += 1
print(valid)
")

echo "  有效训练样本: ${VALID_SAMPLES}/${MAX_SAMPLES}"

if [ "$VALID_SAMPLES" -lt 10 ]; then
    echo "❌ 错误：有效样本太少（< 10）！"
    exit 1
fi

echo ""

# ============================================================================
# Step 3: 训练 Soft QUBO
# ============================================================================

echo "[3/4] 训练 Soft QUBO..."
echo "  模型: LearnableQUBO"
echo "  学习率: ${LEARNING_RATE}"
echo "  训练轮数: ${EPOCHS}"
echo "  这可能需要 30-180 分钟（取决于 CPU/GPU）..."
echo ""

TRAIN_OUTPUT_DIR="$OUTPUT_DIR/training"

python -m scripts.rag.train.train_soft_qubo_simple \
    --result_json "$RESULT_JSON" \
    --max_samples 0 \
    --K ${K} \
    --model_type learnable \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --temperature_init 1.0 \
    --temperature_final 0.3 \
    --temperature_anneal_epochs 60 \
    --seed ${SEED} \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --save_every 20 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

if [ ! -f "$TRAIN_OUTPUT_DIR/best_model.pt" ]; then
    echo "❌ 错误：训练失败，best_model.pt 未生成！"
    echo "   检查日志: $OUTPUT_DIR/training.log"
    exit 1
fi

echo "✅ 训练完成！"
echo ""

# ============================================================================
# Step 4: 生成结果报告
# ============================================================================

echo "[4/4] 生成结果报告..."

python - <<EOF
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
history_file = output_dir / "training" / "history.json"
config_file = output_dir / "training" / "config.json"

with open(history_file) as f:
    history = json.load(f)

with open(config_file) as f:
    config = json.load(f)

# 提取关键指标
final_epoch = history[-1]
best_val_recall = max(ep['val_recall'] for ep in history)
best_train_recall = max(ep['train_recall'] for ep in history)
final_w_a = final_epoch.get('w_a', 'N/A')
final_w_b = final_epoch.get('w_b', 'N/A')

# Phase 2 baseline
baseline_recall = 0.4454
baseline_f1 = 0.5092

# 计算改进
recall_improvement = best_val_recall - baseline_recall
recall_improvement_pct = (recall_improvement / baseline_recall) * 100

# 生成报告
report = f"""# Idea 7 Phase 2 训练结果

**时间戳**: ${TIMESTAMP}
**样本数**: ${VALID_SAMPLES} (有效) / ${MAX_SAMPLES} (总)
**训练轮数**: ${EPOCHS}
**模型**: LearnableQUBO

---

## 核心结果

| 指标 | Phase 2 Baseline | Idea 7 Phase 2 | 改进 |
|------|-----------------|---------------|------|
| **Recall@5** | {baseline_recall:.4f} | **{best_val_recall:.4f}** | **{recall_improvement:+.4f} ({recall_improvement_pct:+.1f}%)** |
| Train Recall | - | {best_train_recall:.4f} | - |

### 学到的权重

- **w_a (quality weight)**: {final_w_a:.3f}
- **w_b (redundancy weight)**: {final_w_b:.3f}

**解释**:
- w_a > 1.0 表示模型学会更重视质量信号
- w_b ≈ 1.0 表示冗余权重保持平衡

---

## 训练曲线

### 最后 10 轮

| Epoch | Train Loss | Train Recall | Val Recall | w_a | w_b |
|-------|-----------|-------------|------------|-----|-----|
"""

for ep in history[-10:]:
    w_a_str = f"{ep.get('w_a', 0):.3f}" if 'w_a' in ep else "N/A"
    w_b_str = f"{ep.get('w_b', 0):.3f}" if 'w_b' in ep else "N/A"
    report += f"| {ep['epoch']} | {ep['train_loss']:.4f} | {ep['train_recall']:.4f} | {ep['val_recall']:.4f} | {w_a_str} | {w_b_str} |\n"

report += f"""
---

## 决策

"""

if best_val_recall > baseline_recall + 0.05:
    decision = "✅ **成功！继续 Phase 3 完整实验**"
    reason = f"Recall 提升 {recall_improvement_pct:.1f}% (≥ 5% 目标)"
elif best_val_recall > baseline_recall:
    decision = "⚠️ **部分成功，需进一步分析**"
    reason = f"Recall 提升 {recall_improvement_pct:.1f}%，但未达到 5% 目标"
else:
    decision = "❌ **失败，考虑 Pivot 到 Idea 2**"
    reason = f"Recall 无改进或下降 ({recall_improvement_pct:.1f}%)"

report += f"""
### {decision}

**原因**: {reason}

### 下一步

"""

if best_val_recall > baseline_recall + 0.05:
    report += """
1. **Phase 3 完整实验** (4 种配置对比)
   - Baseline (Topk)
   - Idea 6 (γ=0.5, δ=0.1)
   - Idea 7 (Soft QUBO)
   - Idea 6+7 (组合)

2. **QUBO gap 诊断**
   - 测量 Idea 7 的 QUBO gap 是否缩小
   - 目标: gap < 0.15（Phase 2 gap = 0.3004）

3. **论文写作**
   - 整理实验结果
   - 分析学到的权重含义
"""
else:
    report += """
1. **调试分析**
   - 检查训练曲线是否过拟合
   - 尝试不同超参数（lr, temperature）
   - 增加训练样本（200 → 500）

2. **备选方案**
   - Idea 2: 简化版（只学习 γ，不用 Soft QUBO）
   - Idea 4: 其他优化方法
"""

report += f"""
---

## 文件清单

```
{output_dir}/
├── data_prep/
│   ├── result.json              # 评估结果（训练数据源）
│   └── eval.log                 # 评估日志
├── training/
│   ├── best_model.pt            # 最佳模型
│   ├── checkpoint_epoch*.pt     # 检查点
│   ├── config.json              # 训练配置
│   └── history.json             # 训练历史
├── training.log                 # 训练日志
└── RESULTS.md                   # 本报告
```

---

## 配置

```json
{json.dumps(config, indent=2)}
```
"""

# 保存报告
report_file = output_dir / "RESULTS.md"
with open(report_file, 'w') as f:
    f.write(report)

print(f"✅ 报告已生成: {report_file}")

# 输出关键结果到控制台
print()
print("=" * 60)
print("核心结果")
print("=" * 60)
print(f"Baseline Recall@5:     {baseline_recall:.4f}")
print(f"Idea 7 Val Recall@5:   {best_val_recall:.4f}")
print(f"改进:                  {recall_improvement:+.4f} ({recall_improvement_pct:+.1f}%)")
print(f"学到的权重:            w_a={final_w_a:.3f}, w_b={final_w_b:.3f}")
print("=" * 60)
print()
if best_val_recall > baseline_recall + 0.05:
    print("✅ 成功！Recall 提升显著，建议继续 Phase 3")
elif best_val_recall > baseline_recall:
    print("⚠️ 部分成功，Recall 有提升但不够显著")
else:
    print("❌ Recall 无改进，需要调试或 Pivot")
print()

EOF

echo ""
echo "========================================"
echo "✅ Idea 7 Phase 2 实验完成！"
echo "========================================"
echo "结果目录: ${OUTPUT_DIR}"
echo "查看报告: ${OUTPUT_DIR}/RESULTS.md"
echo ""
echo "下一步："
echo "  1. 查看报告决定是否继续 Phase 3"
echo "  2. 将结果目录打包发送给导师"
echo "  3. 提交代码: cd $REPO_ROOT && git add exchange/idea7_phase2/${TIMESTAMP} && git commit -m 'results: Idea 7 Phase 2 training results' && git push"
echo ""
