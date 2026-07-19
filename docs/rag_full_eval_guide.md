# RAG 全量评测实验指引

## 更新（2026-07-19）

**修复三个问题**（commit `2fc455f`）：

1. **DPR 检索分数丢失** ✅ 已修复
   - 之前：Top-K 用余弦相似度重新排序，不是真正的 DPR Top-K
   - 现在：所有方法使用 DPR 原始 inner product scores
   - 影响：基线定义正确，Top-K/MMR/QORE 的相关性信号与检索器一致

2. **QORE 参数 `lam` 过高** ✅ 已修复
   - 之前：`lam=2.0` 过度偏向多样性，导致 Recall@5 低（0.218 vs MMR 0.317）
   - 现在：`lam=1.0` 平衡 relevance 和 diversity
   - 预期：提高证据覆盖率，同时保持去冗余优势

3. **生成答案过长** ✅ 已修复
   - 之前：`max_new_tokens=128`，prompt 不够明确，导致 EM=0
   - 现在：`max_new_tokens=32`，prompt 强调"1-5 词简短答案"
   - 预期：EM 指标从 0 提升

**参考**： 200 样本实验报告（`/home/Q-DUET-VLM/analysis.md`）指出的问题。

---

## 概述

RAG 模块代码已完成（模块化重构 + 内存优化），现在需要在大机器上执行全量 NQ 评测。

**机器要求**：
- RAM: 32GB+（推荐 64GB）
- GPU: 24GB 显存（用于生成答案）
- 磁盘: 100GB+ 可用空间（HuggingFace 缓存）
- 网络: 能访问 HuggingFace（首次会下载 ~80GB wiki_dpr 数据集）

---

## 实验步骤

### 1. 拉取最新代码

```bash
cd /path/to/QORE-VLM
git pull origin main
```

### 2. 运行全量评测

在完整的 21M passage 语料库上评测 3 个方法（QORE / MMR / Top-K），单个 seed。

**建议在 tmux 中运行**（避免 SSH 断开中断）：
```bash
tmux new -s rag_eval
```

**执行评测命令**：
```bash
# 在 QORE-VLM 根目录下运行
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 0 \
  --methods qore mmr topk \
  --seeds 42 \
  --K 5 \
  --output_dir results/rag/full_eval_21M \
  --num_workers 1
```

**参数说明**：
- `--max_samples 0`：使用全部样本（NQ validation 有 ~3600 个问题）
- `--methods qore mmr topk`：三个选择方法
- `--seeds 42`：单个随机种子
- `--num_workers 1`：串行运行（如果机器资源充足可以改为 2-3 并行）

**Tmux 操作**：
```bash
# Ctrl+B, D  分离 tmux（实验继续后台运行）
# tmux attach -t rag_eval  # 重新连接查看进度
```

### 3. 监控进度

脚本会显示进度：
```
Evaluating 3610 questions...
  Progress: 100/3610
  Progress: 200/3610
  ...
```

**首次运行**会下载 facebook/wiki_dpr 数据集（~80GB）。

### 4. 检查输出

运行结束后，`results/rag/full_eval_21M/` 目录下应该有：
```
qore_K5_seed42.json
mmr_K5_seed42.json
topk_K5_seed42.json
summary.json          # 自动生成的汇总结果
```

**✅ 检查点**：
- [ ] 3 个 JSON 文件都生成了
- [ ] summary.json 包含统计显著性分析（p-value）
- [ ] QORE 的 Redundancy 显著低于 MMR/Top-K

---

## 交付方式

**将结果提交到 GitHub**：

```bash
git add results/rag/full_eval_21M/*.json
git commit -m "实验结果:RAG全量评测(3 methods × seed 42)"
git push origin main
```

**关键指标**（在 summary.json 中）：
1. **Recall@5**：QORE vs MMR vs Top-K（越高越好）
2. **Redundancy**：QORE 应该显著最低（p < 0.05）
3. **EM / F1**：最终答案质量
4. **统计显著性**：paired t-test 结果
