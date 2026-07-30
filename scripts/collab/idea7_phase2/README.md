# Idea 7 Phase 2: 真实数据训练

师弟在 GPU 机器上运行的一键实验脚本。

## 快速开始

```bash
cd ~/QORE-VLM
git pull
bash scripts/collab/idea7_phase2/run_idea7_phase2.sh
```

**预计时间**:
- GPU: ~1 小时
- CPU: ~3 小时

## 实验流程

脚本会自动执行 4 个步骤：

1. **准备环境** (1 分钟)
   - 检查工作目录
   - 设置 Python 路径

2. **生成训练数据** (10-20 分钟)
   - 运行 RAG 评估获取 200 个样本
   - 使用 Phase 2 最优配置 (γ=0.5, δ=0.1)
   - 保存 retrieved passages 和 gold 标注

3. **训练 Soft QUBO** (30-180 分钟)
   - LearnableQUBO 模型
   - 100 epochs
   - 学习最优 w_a 和 w_b 权重

4. **生成结果报告** (1 分钟)
   - 对比 Phase 2 baseline
   - 计算 Recall 改进
   - 自动判断成功/失败

## 输出

所有结果保存在 `exchange/idea7_phase2/<timestamp>/`:

```
exchange/idea7_phase2/20260730T120000/
├── data_prep/
│   ├── result.json              # 训练数据（RAG 评估结果）
│   └── eval.log                 # 评估日志
├── training/
│   ├── best_model.pt            # 最佳模型
│   ├── checkpoint_epoch*.pt     # 检查点（每 20 轮）
│   ├── config.json              # 训练配置
│   └── history.json             # 训练历史
├── training.log                 # 完整训练日志
└── RESULTS.md                   # 自动生成的结果报告 ⭐
```

## 成功标准

| 指标 | Phase 2 Baseline | Idea 7 目标 | 判定 |
|------|-----------------|------------|------|
| Recall@5 | 0.4454 | **> 0.467** (+5%) | ✅ 成功 |
| Recall@5 | 0.4454 | 0.445-0.467 | ⚠️ 部分成功 |
| Recall@5 | 0.4454 | **≤ 0.445** | ❌ 失败 |

## 检查结果

### 1. 查看自动生成的报告

```bash
cat exchange/idea7_phase2/<timestamp>/RESULTS.md
```

报告会显示：
- Recall 改进百分比
- 学到的权重 (w_a, w_b)
- 训练曲线（最后 10 轮）
- **决策建议**（继续 Phase 3 / 调试 / Pivot）

### 2. 检查训练曲线

```bash
# 查看完整训练历史
python -c "
import json
with open('exchange/idea7_phase2/<timestamp>/training/history.json') as f:
    history = json.load(f)
for ep in history[-10:]:
    print(f\"Epoch {ep['epoch']:3d}: Train={ep['train_recall']:.4f}, Val={ep['val_recall']:.4f}, w_a={ep.get('w_a', 0):.3f}\")
"
```

期望看到：
- Train/Val Recall 稳步提升
- Val Recall > 0.467 (比 baseline 高 5%)
- w_a 在 1.5-3.0 之间稳定
- w_b 在 0.5-2.0 之间

### 3. 检查是否过拟合

```bash
# 对比最后 10 轮的 train vs val
tail -20 exchange/idea7_phase2/<timestamp>/training/history.json
```

**正常**: train ≈ val 或 train 略高于 val (< 10% 差距)  
**过拟合**: train >> val (差距 > 20%)

## 故障排查

### 问题 1: result.json 未生成

```bash
# 检查评估日志
cat exchange/idea7_phase2/<timestamp>/data_prep/eval.log
```

可能原因：
- 语料库未缓存（首次运行需要 10 分钟下载）
- 内存不足（需要 ~8GB RAM）

解决方案：
```bash
# 手动运行评估并观察错误
python -m scripts.rag.eval_rag_refactored \
    --max_samples 10 \
    --corpus_mode aligned \
    --skip_generation \
    --output_file test_result.json
```

### 问题 2: 训练失败或 NaN

检查训练日志：
```bash
grep -i "nan\|inf\|error" exchange/idea7_phase2/<timestamp>/training.log
```

可能原因：
- 学习率过高（降低到 0.001）
- 数据问题（embeddings 有 NaN）

解决方案：
```bash
# 使用更保守的超参数重新训练
python -m scripts.idea7.train_soft_qubo_simple \
    --result_json exchange/idea7_phase2/<timestamp>/data_prep/result.json \
    --epochs 100 \
    --lr 0.001 \
    --output_dir exchange/idea7_phase2_retry
```

### 问题 3: Recall 无改进

可能原因：
1. **数据太少**：增加到 500 样本
2. **训练不足**：增加到 200 epochs
3. **超参数不优**：调整 lr / temperature

尝试：
```bash
# 增加样本数和训练轮数
bash scripts/collab/idea7_phase2/run_idea7_phase2.sh
# 然后手动编辑脚本：MAX_SAMPLES=500, EPOCHS=200
```

## 提交结果

如果实验成功，提交到 Git：

```bash
cd ~/QORE-VLM
git add exchange/idea7_phase2/<timestamp>
git commit -m "results(idea7): Phase 2 training results on 200 samples

Recall@5: <your_result> (baseline: 0.4454)
Improvement: +X.XX%
Learned weights: w_a=X.XX, w_b=X.XX"

git push
```

然后通知导师查看结果。

## 常见问题

**Q: 需要 GPU 吗？**  
A: 不是必需，但强烈推荐。CPU 训练需要 3 小时，GPU 只需 30-60 分钟。

**Q: 可以中断后继续吗？**  
A: 可以。检查点保存在 `training/checkpoint_epoch*.pt`，但目前脚本不支持自动恢复。需要手动加载检查点。

**Q: 内存需要多少？**  
A: ~8GB RAM（数据准备），~4GB GPU memory（训练）。

**Q: 如何调整超参数？**  
A: 编辑脚本顶部的配置部分：
```bash
EPOCHS=100              # 训练轮数
LEARNING_RATE=0.01      # 学习率
MAX_SAMPLES=200         # 样本数
```
