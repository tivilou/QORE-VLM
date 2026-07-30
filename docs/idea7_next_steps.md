# Idea 7 重新评估后的下一步行动方案

## 📊 当前结论

✅ **实验成功完成，结论明确**：
- QUBO vs Oracle gap = 0.3004（未变）
- 命中率 = 34.7%（未变）
- **强烈建议实现 Idea 7**

## 🎯 我的推荐：立即实现 Idea 7 ⭐⭐⭐⭐⭐

### 为什么优先做 Idea 7？

1. **证据最充分** - gap 显著（0.3004 > 0.15 阈值），65.3% 的题有优化空间
2. **独立价值** - 与 Idea 6 正交，可能产生叠加效果
3. **理论基础扎实** - 直接优化任务目标，避免代理目标偏差
4. **论文核心创新** - 端到端优化 vs 启发式目标函数

### 预期收益

- **理论上限**：Recall +0.3004（达到 Oracle 水平）
- **保守估计**：Recall +0.10~0.15（缩小 gap 一半）
- **最终 F1**：可能达到 0.55-0.58（从 0.5092）

---

## 🚀 实施计划（2 周）

### Week 1: 设计 + MVP + 验证

#### Day 1-2: 文献调研 + 方案设计

**调研关键词**：
- "differentiable combinatorial optimization"
- "Gumbel-Softmax for discrete selection"
- "learned to rank with task loss"
- "end-to-end neural passage retrieval"

**设计可微分 QUBO**：
```python
# qore/soft_qubo.py
class SoftQUBO(nn.Module):
    """可微分的 QUBO 选择器
    
    核心思路：
    1. 用 Gumbel-Softmax 或 Top-K 梯度近似替代硬选择
    2. 保持 QUBO 能量函数的结构
    3. 允许端到端训练
    """
    
    def __init__(self, temperature=1.0, method='gumbel'):
        self.temperature = temperature
        self.method = method  # 'gumbel', 'topk_gradient', 'sparsemax'
    
    def forward(self, a, b, K):
        """
        Args:
            a: [N] 质量分
            b: [N, N] 冗余度矩阵
            K: int 选择数量
        
        Returns:
            soft_selection: [N] 软选择概率分布（和为 K）
        """
        # 计算 QUBO 能量对每个变量的梯度
        # 用软选择替代硬选择
        # 返回可微分的选择分布
        pass
```

**训练流程**：
```python
# 输入：候选池 passages, 质量分 a, 相似度矩阵 b
# 输出：软选择分布 soft_selection [N]
soft_selection = soft_qubo(a, b, K)

# 用软选择加权段落，送入 LLM
selected_passages = passages * soft_selection.unsqueeze(-1)  # 加权
answer = llm.generate(query, selected_passages)

# 计算任务损失（F1 或 Recall）
loss = -f1_score(answer, gold_answer)

# 反向传播，更新 QUBO 权重
loss.backward()
optimizer.step()
```

#### Day 3-4: 最小验证实验

**目标**：验证技术可行性

```bash
# 在 10 个题上测试
python scripts/experiments/idea7_mvp.py \
    --samples 10 \
    --method soft_qubo \
    --loss_type recall  # 用 Recall 代理 F1（可计算）
```

**验证点**：
- ✅ 梯度可以反向传播
- ✅ 损失函数可以下降
- ✅ 收敛速度合理（< 100 steps）

**决策点 1**：如果验证失败 → 调整方法或放弃

#### Day 5-7: 完整实现

1. 实现完整的训练循环
2. 加入正则化（避免过拟合）
3. 记录训练曲线

```python
# qore/trainer.py
class Idea7Trainer:
    def train(self, train_samples, epochs=10):
        for epoch in range(epochs):
            for sample in train_samples:
                # 前向传播
                soft_selection = self.soft_qubo(sample)
                recall = self.compute_recall(soft_selection, sample)
                
                # 反向传播
                loss = -recall
                loss.backward()
                self.optimizer.step()
                
                # 记录
                self.log(loss, recall)
```

### Week 2: Phase 3 实验 + 分析

#### Day 8-10: 完整实验

```bash
# Phase 3: Idea 7 vs Baseline vs Idea 6
python scripts/rag/eval_rag_refactored.py \
    --method qore_idea7 \
    --gamma 0.5 \
    --delta 0.1 \
    --max_samples 200 \
    --seed 42
```

**对比配置**：
1. **Baseline**: γ=0.5, δ=0.0（无互补性）
2. **Idea 6**: γ=0.5, δ=0.1（互补性）
3. **Idea 7**: 端到端优化的 QUBO
4. **Idea 6+7**: 互补性 + 端到端优化

#### Day 11-12: 结果分析

**关键指标**：
- QUBO vs Oracle gap 是否缩小？
- 最终 F1 是否提升？
- Idea 6 和 Idea 7 是否有协同效应？

**分析维度**：
- 按题目难度分层分析
- 可视化典型案例
- 训练曲线分析

#### Day 13-14: 文档和报告

- 更新 README 和实验文档
- 准备论文草稿
- 提交代码和结果

---

## 🔬 备选方案：如果实现困难

### Plan B1: 简化版 - 只优化权重

不改变 QUBO 结构，只学习 γ 和 λ 的最优值：

```python
# 参数化 gamma 和 lambda
gamma = nn.Parameter(torch.tensor(0.5))
lam = nn.Parameter(torch.tensor(2.0))

# 在每个样本上优化
for sample in train_set:
    selection = qubo_solve(a, b, K, gamma, lam)
    recall = compute_recall(selection, sample)
    loss = -recall
    loss.backward()  # 更新 gamma 和 lam
```

**优点**：实现简单，风险低  
**缺点**：改进空间有限

### Plan B2: 两阶段优化

第一阶段：优化 QUBO 权重（如 Plan B1）  
第二阶段：端到端微调（如原计划）

**优点**：渐进式，容易调试  
**缺点**：需要更多时间

---

## 📊 其他 Ideas 的优先级

如果 Idea 7 遇到不可克服的困难，或者有多余时间：

### Idea 2: 答案多样性 ⭐⭐⭐

**现状**：Phase 1 诊断发现 NQ 上不可测（单答案）

**改进方案**：
1. 换数据集 → **AmbigQA**（明确标注多个合理答案）
2. 重新运行诊断
3. 如果假设成立，实现多样性优化

**时间成本**：3-5 天（主要是数据集适配）

### Idea 4: 上下文完整性 ⭐⭐

**现状**：Phase 1 实现偏离设计

**改进方案**：
1. 集成更好的共指工具（如 Neuralcoref, SpanBERT）
2. 重新实现 context integrity 检测
3. 在 QUBO 中加入完整性项

**时间成本**：5-7 天

### Idea 3: Query-adaptive γ ⭐

**现状**：分类器无效（simple 类只有 2 题）

**改进方案**：
1. 改进题目复杂度分类器
2. 或者放弃这个方向

**时间成本**：未知（分类器设计困难）

**优先级排序**：
1. **Idea 7** ⭐⭐⭐⭐⭐
2. **Idea 2** ⭐⭐⭐（如果换数据集）
3. **Idea 4** ⭐⭐（需要好的工具）
4. **Idea 3** ⭐（放弃）

---

## ⚠️ 风险与缓解

### 风险 1: 可微分组合优化困难

**症状**：梯度消失/爆炸，训练不稳定

**缓解**：
- 尝试多种梯度近似（Gumbel-Softmax, Top-K, Sparsemax）
- 调整 temperature 超参数
- 使用梯度裁剪

### 风险 2: 训练成本高

**症状**：每个 epoch 需要运行 200 次 LLM 生成

**缓解**：
- **用 Recall 代理 F1**（只需要检查 gold 是否被选中）
- 缓存 LLM 生成结果
- 在小数据集上训练（50-100 题）

### 风险 3: 收益低于预期

**症状**：gap 缩小了，但最终 F1 提升 < 2%

**分析**：
- Recall 提升是否转化为 F1 提升？
- 是否被其他瓶颈限制（如 LLM 生成质量）？

**决策**：如果收益 < 2%，仍然值得发表（方法创新）

---

## 🎯 立即可做的第一步

### 今天（1-2 小时）

1. **文献调研** - 搜索可微分组合优化的方法
2. **阅读代码** - 理解当前 QUBO 实现（`qore/qubo.py`）
3. **草图设计** - 画出 Soft QUBO 的前向和反向传播流程

### 明天（半天）

1. **代码骨架** - 创建 `qore/soft_qubo.py`
2. **单元测试** - 在 toy example 上测试梯度
3. **最小实验** - 在 1 个题上跑通整个流程

### 本周末（1-2 天）

1. **MVP 实现** - 完整的训练循环
2. **10 题验证** - 确认技术可行性
3. **决策** - Go / No-go 决定

---

## 📝 总结

**推荐方案**：**立即实现 Idea 7**

**理由**：
- ✅ 证据充分（gap = 0.3004）
- ✅ 独立价值（与 Idea 6 正交）
- ✅ 论文创新点
- ✅ 风险可控（有 Plan B）

**时间线**：2 周完成 MVP + Phase 3 实验

**决策点**：
- Day 3: 技术可行性验证
- Day 10: 收益评估
- Day 14: Go / No-go 最终决定

如果成功，这将是论文的核心贡献！🚀
