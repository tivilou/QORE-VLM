# RAG 全量评测实验指引

## 更新（2026-07-20）

**gamma 参数修复**（commit `f6672f2`），本轮命令必须显式加 `--gamma 0.5`：

1. **新增 `--gamma`（多样性权重，本轮的关键参数）**
   - 根因：冗余惩罚权重此前固定 1.0，K=5 时 10 个成对冗余项累加压过质量项，
     导致 QORE 丢失含答案段落（上轮 Recall@5 比基线低约 11 个百分点）
   - 50k 子集网格搜索（200 题）：**gamma=0.5 最优** — Recall 20.23%（gamma=1.0 仅
     12.69%），冗余度 0.675 仍显著低于基线
2. **纠正 2026-07-19 版第 2 条**：`lam` 不是质量-多样性权衡参数，它只控制"恰好选
   K 段"的基数约束，对选哪 K 段没有影响；默认值已回滚为 `lam=2.0`，无需调整
3. 上轮修复保持有效（commit `2fc455f`）：DPR 检索分数传递、简短答案 prompt
   （EM 已从 0 恢复到 ~25%）

**评测改进**（commit `df972e7`），本轮自动生效：

4. **Gold 判定改进**：从简单字符串包含改为 token 边界匹配，避免误命中（如 "2012"
   匹配 "20122"），Recall/Precision 数值更准确
5. **检索失败统计修复**（commit `6b99cec`，重要）：之前版本的
   `recall_at_retrieved=1.0` 和 `n_retrieval_failure=0` **是错误指标**，已修复：
   - 现在每题记录 `answer_hit_at_retrieved`（Top-50 中是否命中 gold）
   - `n_with_gold`：检索命中的题数；`n_retrieval_failure`：Top-50 未含答案的题数
6. **QORE prefilter 修复**（commit `6b99cec`）：`direct_solve_max_n` 从 64 降为 20，
   N=50 现在正确走 relevance-first 预过滤，`--qore_prefilter_size` 参数生效
7. **可选**：新增 `--qore_prefilter_size` 参数（默认 max(K*3,15)），若达标困难可
   尝试 `--qore_prefilter_size 15` 进一步限制候选池

**⚠️ 注意**：2026-07-20 的 400 题实验用的是 `gamma=None`（=1.0），**不是** gamma=0.5；
那轮 QORE 结果（Recall 28.1%，F1 44.5%）是 gamma=1.0 的表现，不能作为 gamma=0.5 的参考。

**参考**：200 样本（`/home/Q-DUET-VLM/analysis(1).md`）和 400 样本（`analysis(2).md`）实验报告。

---

## 概述

RAG 模块代码已完成（模块化重构 + 内存优化 + gamma 参数 + 评测修复）。本轮分两步执行：
先跑 400 题验证 gamma=0.5 的效果，达标后再跑全量 NQ 评测。

**机器要求**：
- RAM: 32GB+（推荐 64GB）
- GPU: 24GB 显存（用于生成答案）
- 磁盘: 100GB+ 可用空间（HuggingFace 缓存）
- 网络: 能访问 HuggingFace（wiki_dpr 应已缓存，直接命中）

---

## 实验步骤

### 1. 拉取最新代码

```bash
cd /path/to/QORE-VLM
git pull origin main
```

### 2. 第一步：400 题验证（gamma=0.5）

与上轮 400 题实验的唯一区别：加上 `--gamma 0.5`。

```bash
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 400 \
  --methods qore,mmr,topk \
  --seeds 42 \
  --K 5 \
  --lam 2.0 \
  --gamma 0.5 \
  --output_dir results/rag/gamma05_400
```

**注意**：
- `--methods` 用**逗号分隔**（用空格会报 unrecognized arguments）
- 本地已有模型的话加 `--model_path models/llama3-8b`（按实际路径）
- QORE 的 prefilter 现在默认生效（direct_solve_max_n=20），选择耗时应显著低于上轮 476 ms

**预期输出样例**：
```
Retrieval: 311/400 hit gold in Top-50 | 89 retrieval failures
Recall@5:  0.XXXX (conditional on retrieval hit)
Redundancy: 0.XXXX
EM: 0.XXXX
F1: 0.XXXX
```

**通过标准**（对比上轮 gamma=1.0 的 400 题）：
- [ ] QORE Recall@5 ≥ 34%（上轮 28.1%；基线 Top-K 38.3%）
- [ ] QORE F1 ≥ 46%（上轮 44.5%；基线 MMR 47.0%）
- [ ] QORE 冗余度 < 0.80（上轮 0.766，基线 MMR 0.835）

三项达标 → 执行第二步全量评测；不达标 → 按交付方式推送 JSON 等分析。

### 3. 第二步：全量评测

**建议在 tmux 中运行**：
```bash
tmux new -s rag_eval
```

```bash
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 0 \
  --methods qore,mmr,topk \
  --seeds 42 \
  --K 5 \
  --lam 2.0 \
  --gamma 0.5 \
  --output_dir results/rag/full_eval_21M
```

Ctrl+B, D 分离 tmux；`tmux attach -t rag_eval` 重新连接查看进度。

### 4. 检查输出

```
qore_K5_seed42.json
mmr_K5_seed42.json
topk_K5_seed42.json
summary.json
```

**检查点**：
- [ ] 3 个 JSON 文件都生成了
- [ ] `n_with_gold` 和 `n_retrieval_failure` 有合理数值（非 0/1.0）
- [ ] QORE 的 Redundancy 明显低于 MMR/Top-K
- 注：单 seed 下 summary.json 不含 t-test

---

## 交付方式

```bash
git add -f results/rag/gamma05_400/*.json
git commit -m "实验结果:RAG 400题验证(gamma=0.5)"
git push origin main
```

全量评测同样操作（目录换成 `results/rag/full_eval_21M`）。

**关键指标**：
1. **Recall@5**：QORE vs MMR vs Top-K（越高越好）
2. **Redundancy**：QORE 应该最低
3. **EM / F1**：最终答案质量
