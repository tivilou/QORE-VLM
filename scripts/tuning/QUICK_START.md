# 给师弟：实验自动化框架快速开始

**目标**: 一键运行 Phase 1 诊断实验，自动分析和打包结果  
**耗时**: 1-2 小时（3 个实验 × 200 题）

---

## 🚀 快速开始（3 步）

### Step 1: 验证框架（可选，5 分钟）

先用 10 题测试框架是否正常工作：

```bash
cd /home/Q-DUET-VLM/QORE-VLM

python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/quick_test.yaml
```

**预期输出**:
```
==================================================================
环境检查
==================================================================
✅ GPU 可用
✅ 磁盘空间充足: XX GB 可用
✅ torch 已安装
✅ transformers 已安装
✅ 环境检查通过

==================================================================
开始实验: Quick Test (10 samples)
==================================================================
实验数量: 3
输出目录: scratch/research/quick_test

[1/3] 运行: gamma_0.0
  描述: 纯质量
  预计耗时: 2 分钟
  ✅ 成功 (耗时: 118s)
     Recall: 0.4500, F1: 0.4200, 冗余: 0.8650

[2/3] 运行: gamma_0.5
  ...

[3/3] 运行: gamma_1.0
  ...

==================================================================
后处理
==================================================================

📊 分析结果...
  ✅ 摘要保存: scratch/research/quick_test/run_summary.json
  ✅ 报告生成: scratch/research/quick_test/analysis/quick_summary.md
  ✅ 分析完成: scratch/research/quick_test/analysis/gamma_sweep.md

📦 打包结果...
  ✅ 打包完成: scratch/research/quick_test/package/quick_test_20260724_143052.zip (1.2 MB)
     包含 5 个文件

==================================================================
实验完成
==================================================================
名称: Quick Test (10 samples)
总耗时: 0:06:24
输出目录: scratch/research/quick_test

实验统计:
  总数: 3
  成功: 3
  失败: 0

📊 分析报告: scratch/research/quick_test/analysis/gamma_sweep.md
📦 打包文件: scratch/research/quick_test/package/quick_test_20260724_143052.zip
```

如果看到 ✅ 全部成功，说明框架工作正常，可以进入 Step 2。

---

### Step 2: 运行 Phase 1 诊断（1-2 小时）

运行完整的 Phase 1 诊断实验（200 题 × 3 个配置）：

```bash
cd /home/Q-DUET-VLM/QORE-VLM

python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml
```

**这个命令会自动**:
1. ✅ 检查环境（GPU、磁盘、依赖）
2. ✅ 运行 3 个实验（gamma=0.0, 0.5, 1.0）
3. ✅ 收集所有结果
4. ✅ 自动分析（调用 `analyze_gamma_diagnosis.py`）
5. ✅ 生成报告
6. ✅ 打包成 zip

你只需要等待即可，中途可以看到进度。

**如果中途失败**: 框架会自动重试 2 次，如果还是失败会记录错误并继续下一个实验。

---

### Step 3: 查看结果并决策（5 分钟）

实验完成后，查看分析报告：

```bash
cat scratch/research/P1_diagnosis/analysis/gamma_sweep.md
```

**报告会告诉你**:
- ✅ 哪个 γ 配置最优（Recall 和 F1）
- ✅ 多样性是否有用
- ✅ 下一步建议（实现两阶段 QUBO 或其他）

**然后把结果发给我**:
```bash
# 查看打包文件
ls scratch/research/P1_diagnosis/package/

# 发送这个 zip 文件和 analysis.md
```

---

## 📁 输出位置

所有结果都在 `scratch/research/P1_diagnosis/`:

```
P1_diagnosis/
├── experiments/              # 各实验的详细结果
│   ├── gamma_0.0/
│   │   ├── result.json      # ⭐ 实验数据
│   │   ├── stdout.log       # 标准输出
│   │   ├── stderr.log       # 错误日志
│   │   └── status.json      # 状态信息
│   ├── gamma_0.5/
│   └── gamma_1.0/
│
├── analysis/                 # ⭐ 分析结果（重点查看）
│   ├── analysis.md          # 详细分析报告
│   └── quick_summary.md     # 快速摘要
│
├── package/                  # ⭐ 打包文件（发给我）
│   └── P1_diagnosis_YYYYMMDD_HHMMSS.zip
│
├── run_summary.json         # ⭐ 运行摘要（重点查看）
└── run.log                  # 完整日志
```

**重点查看**:
1. `analysis/gamma_sweep.md` - 分析报告
2. `run_summary.json` - 运行摘要
3. `package/*.zip` - 打包文件（发给我）

---

## ⚠️ 常见问题

### Q1: 环境检查失败怎么办？

```
❌ GPU 不可用
```

**解决**: 检查 GPU
```bash
nvidia-smi
```

如果看到 GPU 信息就没问题，可能是临时问题，重试即可。

---

### Q2: 某个实验失败怎么办？

```
[2/3] ❌ gamma_0.5: Command failed
```

**框架会自动重试**，如果 2 次重试都失败：

1. 查看错误日志:
```bash
cat scratch/research/P1_diagnosis/experiments/gamma_0.5/stderr.log
```

2. 查看状态:
```bash
cat scratch/research/P1_diagnosis/experiments/gamma_0.5/status.json
```

3. 如果是代码问题，发给我看

4. 如果是临时问题（如网络），可以手动重跑单个实验:
```bash
python -m scripts.rag.eval_rag_refactored \
    --corpus_mode wiki_dpr --dataset nq_open --max_samples 200 \
    --method qore --gamma 0.5 --use_answer_scorer --skip_generation \
    --output_dir scratch/research/P1_diagnosis/experiments/gamma_0.5 \
    --output_file result.json
```

---

### Q3: 想快速测试，不想跑 200 题？

```bash
# 只跑 50 题（快速验证）
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml \
    --override max_samples=50
```

这样只需要 15-20 分钟。

---

### Q4: 想中途停止怎么办？

按 `Ctrl+C`，框架会安全停止。

已完成的实验结果会保留，可以查看。

---

## 📊 理解结果

### run_summary.json 示例

```json
{
  "total_experiments": 3,
  "success": 3,
  "failed": 0,
  "metrics": {
    "gamma_0.0": {"recall": 0.485, "f1": 0.478, "redundancy": 0.862},
    "gamma_0.5": {"recall": 0.471, "f1": 0.470, "redundancy": 0.786},
    "gamma_1.0": {"recall": 0.445, "f1": 0.448, "redundancy": 0.732}
  },
  "best_recall_config": "gamma_0.0",
  "best_f1_config": "gamma_0.0"
}
```

**解读**:
- `gamma_0.0` Recall 最高 → 纯质量最优
- 多样性（γ=1.0）损害 Recall
- **下一步**: 可能不需要两阶段 QUBO，直接用 Top-K on Answer Scorer

---

### analysis.md 示例

```markdown
# Phase 1 诊断分析报告

## 关键发现

### 1. Recall 变化趋势
- γ: 0.0 → 0.5: -1.4%
- γ: 0.0 → 1.0: -4.0%

**结论**: 多样性显著损害 Recall（>3%）

### 2. 最优配置
- Recall 最高: gamma_0.0 (0.485)

## 下一步建议

纯质量（γ=0.0）显著最优。
建议：不实现两阶段 QUBO，直接用 Top-K on Answer Scorer...
```

---

## 🎯 完成后交付

运行完成后，发给我：

1. **分析报告**: `scratch/research/P1_diagnosis/analysis/gamma_sweep.md`
2. **打包文件**: `scratch/research/P1_diagnosis/package/*.zip`
3. **你的决策**: 根据报告，你认为下一步应该做什么？

---

## 📚 更多信息

- **详细文档**: `scripts/tuning/README.md`
- **配置文件**: `scripts/tuning/config/phase1_diagnosis.yaml`
- **框架设计**: 看代码注释

---

**有问题随时联系！祝实验顺利！** 🚀
