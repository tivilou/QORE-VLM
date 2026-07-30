# Idea 7 Quick Start

快速验证 Idea 7 技术可行性（10 样本，5-10 分钟）。

## 一行命令

```bash
cd ~/QORE-VLM && python -m scripts.idea7.train_soft_qubo --max_samples 10 --epochs 50 --output_dir exchange/idea7_mvp
```

## 检查结果

### 1. 训练是否收敛？

```bash
# 查看最后几轮的 Recall
tail -20 exchange/idea7_mvp/history.json
```

期望看到 Recall 从 0.3-0.4 提升到 0.5-0.6。

### 2. 学到的权重是否合理？

```bash
# 查看 w_a (quality weight) 和 w_b (redundancy weight)
grep "w_a\|w_b" exchange/idea7_mvp/history.json | tail -5
```

期望：
- `w_a > 0` (质量权重为正)
- `w_b > 0` (冗余权重为正)
- `w_a / w_b` 在 0.5-5.0 范围内（平衡）

### 3. 梯度是否反向传播？

检查训练日志中是否有：
- Loss 逐步下降 ✅
- Recall 逐步上升 ✅
- 没有 NaN/Inf ✅

## 成功标准（Day 3 决策点）

- ✅ **技术可行**：训练收敛，Recall 提升，无梯度消失/爆炸
- ✅ **继续 Day 4-7**：完整训练（200 样本）+ 评估

- ❌ **技术不可行**：梯度问题、权重崩溃、无改进
- ❌ **Pivot**：尝试 Idea 2 或简化版本（只学 γ，不用 Gumbel-Softmax）

## 下一步

如果 MVP 成功，运行完整训练：

```bash
python -m scripts.idea7.train_soft_qubo \
    --max_samples 200 \
    --epochs 100 \
    --model_type learnable \
    --output_dir exchange/idea7_full \
    --save_every 20
```

预计用时：2-3 小时（CPU）或 30-60 分钟（GPU）。
