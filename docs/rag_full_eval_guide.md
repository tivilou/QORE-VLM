# RAG 全量评测实验指引

## 概述

RAG 模块代码已完成（模块化重构 + 内存优化），现在需要在大机器上执行全量 NQ 评测。
你只需要按照下面的步骤执行命令，遇到报错及时反馈，最后交付 summary.json 结果文件。

**预计时间**：2-3 天（大部分是后台运行）

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
git log --oneline -1  # 确认最新 commit 是 b6933e0 或更新
```

### 2. 运行全量评测

在完整的 21M passage 语料库上评测 3 个方法（QORE / MMR / Top-K），各跑 3 个 seed，共 9 组实验。

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
  --seeds 42 123 456 \
  --K 5 \
  --output_dir results/rag/full_eval_21M \
  --num_workers 1
```

**参数说明**：
- `--max_samples 0`：使用全部样本（NQ validation 有 ~3600 个问题）
- `--methods qore mmr topk`：三个选择方法
- `--seeds 42 123 456`：三个随机种子
- `--num_workers 1`：串行运行（如果机器资源充足可以改为 2-3 并行）

**预计时间**：
- 单个 run（1 method × 1 seed × 3600 questions）：约 4-8 小时
- 总共 9 个 runs：串行约 36-72 小时，并行可缩短

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

**首次运行**会下载 facebook/wiki_dpr 数据集（~80GB），显示：
```
Loading facebook/wiki_dpr [psgs_w100.nq.compressed]...
(This downloads the dataset + prebuilt FAISS index on first run)
Downloading...
```
耐心等待下载完成（可能需要几小时到一天，取决于网络速度）。

### 4. 检查输出

运行结束后，`results/rag/full_eval_21M/` 目录下应该有：
```
qore_K5_seed42.json
qore_K5_seed123.json
qore_K5_seed456.json
mmr_K5_seed42.json
mmr_K5_seed123.json
mmr_K5_seed456.json
topk_K5_seed42.json
topk_K5_seed123.json
topk_K5_seed456.json
summary.json          # 自动生成的汇总结果
```

**✅ 检查点**：
- [ ] 9 个 JSON 文件都生成了
- [ ] summary.json 包含统计显著性分析（p-value）
- [ ] QORE 的 Redundancy 显著低于 MMR/Top-K

---

## 交付物

**请把以下文件打包发给我**：

```bash
cd results/rag/full_eval_21M
tar -czf rag_full_eval_results.tar.gz *.json
# 把 rag_full_eval_results.tar.gz 发给我
```

**我需要查看的关键指标**：
1. **Recall@5**：QORE vs MMR vs Top-K（越高越好）
2. **Redundancy**：QORE 应该显著最低（p < 0.05）
3. **EM / F1**：最终答案质量
4. **统计显著性**：summary.json 中的 paired t-test 结果

---

## 常见问题

**Q1: 下载 wiki_dpr 数据集太慢怎么办？**

A: 首次下载需要 ~80GB，可能需要几小时到一天。如果网络不稳定：
- 让程序继续跑，HuggingFace 支持断点续传
- 或者联系我使用 faiss+mmap 备选方案

**Q2: 运行中途崩溃了怎么办？**

A: 
1. **立即**截图/复制完整的报错信息发给我
2. 查看 `results/rag/full_eval_21M/` 已经生成了哪些文件
3. 可以单独重跑失败的 run：
```bash
python -m scripts.rag.eval_rag_refactored \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --method qore \
  --seed 123 \
  --K 5 \
  --max_samples 0 \
  --output_dir results/rag/full_eval_21M
```

**Q3: GPU 显存不足怎么办？**

A: 加 `--skip_generation` 跳过答案生成，只评测选择质量（Recall/Redundancy）：
```bash
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 0 \
  --methods qore mmr topk \
  --seeds 42 123 456 \
  --skip_generation \
  --output_dir results/rag/full_eval_21M
```

**Q4: 想加快实验速度怎么办？**

A: 如果机器资源充足，可以并行运行多个实验：
```bash
# 改 --num_workers 2 或 3（同时跑2-3个实验）
--num_workers 2
```

---

## 联系方式

- **有任何问题立即联系我**
- 报错时请提供：完整报错信息 + 运行的命令 + 机器配置
- 预期完成时间：2-3 天（大部分是后台运行）

加油！🚀
