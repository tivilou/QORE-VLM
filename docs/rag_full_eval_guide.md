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

## 阶段 1: 冒烟测试（半天）

### 目的
验证代码在你的机器上能正常运行，先用小样本测试。

### 步骤

**1.1 拉取最新代码**
```bash
cd /path/to/QORE-VLM
git pull origin main
git log --oneline -1  # 确认最新 commit 是 0943f8d (内存优化)
```

**1.2 测试 HotpotQA (precomputed 模式)**
```bash
# 50 样本，约 5-10 分钟
python -m scripts.rag.eval_rag_refactored \
  --corpus_mode precomputed \
  --dataset hotpotqa_distractor \
  --method qore \
  --K 5 \
  --max_samples 50 \
  --seed 42 \
  --output_dir results/rag/stage1_smoke
```

**预期结果**：
- 生成 `results/rag/stage1_smoke/qore_K5_seed42.json`
- EM 约 0.35-0.45，F1 约 0.50-0.60
- 无报错

**1.3 测试 wiki_dpr 模式（小样本）**
```bash
# 首次运行会下载数据集（~80GB，需要时间）
# 50 样本，约 10-20 分钟（不含下载时间）
python -m scripts.rag.eval_rag_refactored \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --method qore \
  --K 5 \
  --max_samples 50 \
  --seed 42 \
  --output_dir results/rag/stage1_smoke
```

**预期结果**：
- 第一次运行会显示 "Downloading facebook/wiki_dpr..."，耐心等待
- 生成 `results/rag/stage1_smoke/qore_K5_seed42.json`
- Recall@5、Redundancy、EM、F1 都有数值（不是 0 或 NaN）
- 无崩溃

**✅ 阶段 1 检查点**：
- [ ] HotpotQA 测试通过
- [ ] wiki_dpr 数据集下载完成
- [ ] wiki_dpr 50 样本测试通过
- [ ] 两个 JSON 文件都生成了

如果有任何报错，**立即截图/复制报错信息发给我**。

---

## 阶段 2: 全量语料库准备（已省略）

**好消息**：wiki_dpr 模式不需要单独构建语料库！

在阶段 1 下载完 facebook/wiki_dpr 数据集后，阶段 3 可以直接使用，无需额外准备。

---

## 阶段 3: 全量评测（1-2 天后台运行）

### 目的
在完整的 21M passage 语料库上评测 3 个方法（QORE / MMR / Top-K），各跑 3 个 seed，共 9 组实验。

### 步骤

**3.1 使用 eval_suite 批量运行**

创建配置文件 `configs/rag_full_eval.yaml`（或直接用命令行）：

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

**3.2 监控进度**

脚本会显示进度：
```
Evaluating 3610 questions...
  Progress: 100/3610
  Progress: 200/3610
  ...
```

建议在 `tmux` 或 `screen` 中运行，避免 SSH 断开中断：
```bash
tmux new -s rag_eval
python -m scripts.rag.eval_suite ...
# Ctrl+B, D 分离 tmux
# tmux attach -t rag_eval  # 重新连接
```

**3.3 检查输出**

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

**✅ 阶段 3 检查点**：
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

A: 首次下载需要 ~80GB，可能需要几小时到一天。如果网络不稳定，可以考虑：
1. 使用 HuggingFace 镜像（设置 `HF_ENDPOINT` 环境变量）
2. 或者改用 faiss+mmap 备选方案（需要先运行 `build_faiss_corpus.py`，见附录）

**Q2: 运行中途崩溃了怎么办？**

A: 
1. 截图/复制完整的报错信息发给我
2. 查看 `results/rag/full_eval_21M/` 已经生成了哪些文件
3. 可以单独重跑失败的 run：
```bash
python -m scripts.rag.eval_rag_refactored \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --method qore \
  --seed 123 \
  --output_dir results/rag/full_eval_21M
```

**Q3: GPU 显存不足怎么办？**

A: 生成阶段需要加载 LLM（Meta-Llama-3-8B），需要 ~24GB 显存。如果显存不够：
1. 加 `--skip_generation` 跳过生成，只评测选择质量（Recall/Redundancy）
2. 或者换更小的模型（改 `--model_path`）

**Q4: 内存不足怎么办？**

A: wiki_dpr 模式内存占用很小（<10GB）。如果还是不够：
1. 检查是否有其他进程占用内存
2. 确认使用的是 `--corpus_mode wiki_dpr`（不是 faiss）

---

## 附录：faiss+mmap 备选方案（仅在 wiki_dpr 不可用时使用）

如果 wiki_dpr 下载失败或网络问题，可以改用 faiss+mmap 模式：

**A1. 构建 faiss 语料库**
```bash
# 一次性操作，生成 embeddings.npy (~65GB) 和 passages.pkl
python -m scripts.rag.build_faiss_corpus \
  --corpus_size 0 \
  --out data/wiki_dpr_full
# 预计时间：6-12 小时
```

**A2. 运行评测（使用 mmap）**
```bash
python -m scripts.rag.eval_suite \
  --corpus_mode faiss \
  --faiss_embeddings_path data/wiki_dpr_full/embeddings.npy \
  --faiss_passages_path data/wiki_dpr_full/passages.pkl \
  --faiss_mmap \
  --dataset nq_open \
  --max_samples 0 \
  --methods qore mmr topk \
  --seeds 42 123 456 \
  --output_dir results/rag/full_eval_21M_faiss
```

---

## 联系方式

- 有任何问题立即联系我
- 报错时请提供：完整报错信息 + 运行的命令 + 机器配置
- 预期完成时间：阶段1（半天）+ 阶段3（2天后台）= 2.5天

加油！🚀
