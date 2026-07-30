# Idea 7 重新评估实验

## 背景

**Idea 7 (QUBO 代理目标优化)** 的假设：当前 QUBO 能量函数是人工设计的代理目标，与真实的 F1/Recall 目标存在系统性偏离。

### Phase 1 诊断结果（2026-07-27）

在 150 个样本上测试（γ=0.5, δ=0.0, 无互补性）：

| 指标 | 值 |
|------|-----|
| QUBO 最优解平均 Recall | 0.5909 |
| Oracle 最优解平均 Recall | 0.8913 |
| **Gap** | **0.3004** |
| QUBO 命中最优的题数 | 52/150 (34.7%) |

**结论**：假设成立，存在显著的 gap。

### 为什么需要重新评估？

**关键变化**：Idea 6（互补性打分）已实现，Phase 2 结果显示显著提升：

| 指标 | Baseline (δ=0.0) | Idea 6 (δ=0.1) | 提升 |
|------|------------------|----------------|------|
| Recall@5 | 0.3196 | 0.4454 | +37.9% |
| F1 | 0.4406 | 0.5092 | +13.6% |
| Redundancy | 0.8090 | 0.7876 | -2.6% |

**问题**：
- Phase 1 时，Idea 6 和 Idea 7 共享同一个 oracle（都是基于 Recall 的最优子集）
- 现在 Idea 6 已实现，QUBO 最优解的质量可能已经提升
- Oracle 最优解的质量也可能提升
- 需要测量**新的 gap** 是否仍然显著到值得实现 Idea 7

## 实验目标

测量在 Idea 6 实现后：
1. QUBO 最优解的平均 Recall
2. Oracle 最优解的平均 Recall  
3. 新的 gap 大小
4. 决定是否需要实现 Idea 7

## 实验配置

```bash
数据集: nq_open
样本数: 200
方法: qore
K: 5
γ: 0.5  # Idea 6 推荐值
δ: 0.1  # Idea 6 推荐值
λ: 2.0
complementarity: dpr
种子: 42
```

**关键参数**：必须加 `--dump_passages` 以记录 QUBO 的 a/b 矩阵。

## 运行步骤

### 1. 拉取最新代码

```bash
cd ~/QORE-VLM
git pull origin main
```

### 2. 运行诊断脚本

```bash
cd scripts/collab/p2_idea7_diagnosis
bash run_idea7_diagnosis.sh
```

脚本会自动完成三个步骤：
- 步骤 1: 运行 RAG 评估（含 QUBO 数据）
- 步骤 2: 运行 QUBO 目标诊断
- 步骤 3: 生成对比报告

### 3. 检查结果

运行完成后，查看生成的目录：
```bash
exchange/p2_idea7_diagnosis/<timestamp>/
├── README.md              # 实验说明和对比分析
├── result.json            # RAG 评估完整结果
└── qubo_diagnosis.md      # QUBO 诊断报告（核心）
```

**关键文件**：`qubo_diagnosis.md` 包含：
- QUBO 最优解平均 Recall
- Oracle 最优解平均 Recall
- Gap 大小和命中率
- 假设验证结论

### 4. 提交结果

```bash
cd ~/QORE-VLM
git add exchange/p2_idea7_diagnosis/<timestamp>/
git commit -m "experiment(idea7): re-evaluation after idea6 implementation"
git push
```

**注意**：`result.json` 文件较大（~4MB），需要提交以供后续分析。

## 预期时间

- 评估阶段：~30-60 分钟（200 样本 × DPR 打分 + 生成）
- 诊断阶段：~5-10 分钟（枚举子集，C(12,5)=792 × 200）
- **总计**：~40-70 分钟

## 决策标准

根据新的 gap 大小决定后续行动：

| Gap | 命中率 | 决策 |
|-----|--------|------|
| > 0.15 | < 50% | ✅ **强烈建议实现 Idea 7** - gap 仍然显著 |
| 0.08 - 0.15 | 50-70% | ⚠️ **考虑实现 Idea 7** - 有提升空间但需权衡成本 |
| < 0.08 | > 70% | ❌ **暂不实现** - Idea 6 已充分优化，边际收益有限 |

对比 Phase 1 的 gap (0.3004, 命中率 34.7%)，如果新 gap 缩小 > 50%，说明 Idea 6 已经显著改善了选择质量。

## 故障排查

### 常见问题

1. **缺少模型文件**
   ```bash
   # 确认 DPR answer scorer 已下载
   ls models/dpr/
   ```

2. **内存不足**
   ```bash
   # 减少样本数（编辑 run_idea7_diagnosis.sh）
   SAMPLES=100  # 改为 100
   ```

3. **CUDA 错误**
   ```bash
   # 检查 GPU 可用性
   nvidia-smi
   ```

4. **找不到 dump_passages 参数**
   ```bash
   # 确认代码是最新的
   python -m scripts.rag.eval_rag_refactored --help | grep dump_passages
   ```

## 相关文档

- Phase 1 诊断报告：`exchange/p1_diagnosis/20260727T201307/analysis/qubo_objective.md`
- Phase 2 Idea 6 结果：`exchange/p2_solver_idea6/20260729T131723/README.md`
- 诊断脚本实现：`scripts/diagnosis/qubo_objective_diagnosis.py`
- 诊断方法说明：`.ai-progress/workstreams/rag-selector/refs/idea-diagnosis-gap-20260726.md`

---

**有问题随时联系！**
