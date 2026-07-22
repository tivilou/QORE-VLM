# Answer Scorer 优化使用指南

## 概述

**优化目标**：用答案抽取置信度替代 DPR inner product 作为 QUBO 质量信号，让 QORE 直接优化"在含答案的段落中选择多样子集"。

**实现状态**：代码已完成（commit `d3de3bb`），待真机验证。

**预期效果**：Recall@5 提升 3-4 个百分点，接近 MMR/Top-K 水平。

---

## 背景

### 当前问题

**QORE 的 Recall@5 落后基线 ~2.5-3.2 个百分点**（3-seed 全量评测）：
- QORE: 33.76%±0.10%
- MMR: 36.24%
- Top-K: 36.92%

**根因分析**：
- QUBO 质量项 `a_i` 当前用 DPR 检索分数（query ⊙ passage embedding）
- 这衡量的是**语义相关性**，不是**答案存在性**
- 段落可以"很相关"但恰好不含答案字符串
- QORE 为了多样性放弃的恰好是"语义相似但含答案"的段落

### 解决方案

用 **DPR Reader 的 relevance_logits** 替代 DPR 检索分数：
- `relevance_logits`：DPR reader 对每个段落输出的"是否含答案"二分类分数
- 直接预测答案存在性，与最终 QA 目标对齐
- QORE 的优化目标从"相关且多样"变成"含答案且多样"

---

## 使用方法

### 基本用法

在评测命令中添加 `--use_answer_scorer`：

```bash
python -m scripts.rag.eval_rag_refactored \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 200 \
  --method qore \
  --K 5 \
  --gamma 0.5 \
  --direct_solve_max_n 20 \
  --seed 42 \
  --use_answer_scorer \
  --output_dir results/rag/answer_scorer_test
```

### 选择后端

默认使用 DPR reader，也可以切换到 cross-encoder：

```bash
# 使用 DPR reader（推荐，最准确）
--use_answer_scorer --answer_scorer_backend dpr

# 使用 cross-encoder（更快，略不准确）
--use_answer_scorer --answer_scorer_backend cross_encoder
```

---

## 对比实验方案

### Phase 1：200题快速验证（推荐先跑）

**Baseline**（当前最佳配置）：
```bash
python -m scripts.rag.eval_rag_refactored \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 200 \
  --method qore \
  --K 5 \
  --lam 2.0 \
  --gamma 0.5 \
  --direct_solve_max_n 20 \
  --seed 42 \
  --skip_generation \
  --output_dir results/rag/answer_scorer_200 \
  --output_file baseline_200.json
```

**Answer Scorer**（新优化）：
```bash
python -m scripts.rag.eval_rag_refactored \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 200 \
  --method qore \
  --K 5 \
  --lam 2.0 \
  --gamma 0.5 \
  --direct_solve_max_n 20 \
  --seed 42 \
  --skip_generation \
  --use_answer_scorer \
  --output_dir results/rag/answer_scorer_200 \
  --output_file answer_scorer_200.json
```

**预期耗时**：
- Baseline: ~5 分钟（200 题，skip_generation）
- Answer Scorer: ~20-30 分钟（DPR reader 推理慢）

**通过标准**：
- Recall@5 ≥ 35%（接近 MMR 的 ~35.5%）
- 冗余度保持 < 0.82
- 选择耗时增加可接受（目标 < 200ms/题）

---

### Phase 2：全量评测（如果 Phase 1 有效）

```bash
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 0 \
  --methods qore \
  --seeds 42,123,456 \
  --K 5 \
  --lam 2.0 \
  --gamma 0.5 \
  --direct_solve_max_n 20 \
  --qore_use_answer_scorer \
  --output_dir results/rag/answer_scorer_full_3seeds
```

**注意**：eval_suite.py 需要添加 `--qore_use_answer_scorer` 参数支持（当前未实现）。

---

## 技术细节

### Answer Scorer 后端对比

| 后端 | 模型 | 速度 | 准确性 | 说明 |
|------|------|------|--------|------|
| **dpr** | `facebook/dpr-reader-single-nq-base` | 慢 (~150ms/query) | 高 | 输出 relevance_logits，直接预测答案存在性 |
| **cross_encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 快 (~50ms/query) | 中 | 输出语义匹配分数，不是真正的答案似然 |

**推荐**：先用 `dpr` 验证效果，如果速度成为瓶颈再尝试 `cross_encoder`。

### 实现架构

```python
# 1. 检索（保持不变）
retrieved_idx, retrieval_scores = corpus_manager.retrieve(query_emb, top_k=50)

# 2. Answer Scoring（新增）
if use_answer_scorer:
    retrieval_scores = answer_scorer.score_passages(question, retrieved_texts)
    # retrieval_scores 从 DPR inner product 变成 answer likelihood

# 3. 选择（QUBO 输入变化）
selected = select_passages(
    query_emb, retrieved_embs, K=5,
    relevance_scores=retrieval_scores,  # ← 现在是 answer likelihood
    gamma=0.5
)
```

**关键变化**：QUBO 质量项 `a_i` 的语义从"检索相关性"变成"答案存在概率"。

---

## 预期结果

### 200题验证（skip_generation）

| 配置 | Recall@5 | Precision@5 | 冗余度 | 选择耗时 |
|------|----------|-------------|--------|---------|
| Baseline (DPR scores) | ~32% | ~24% | ~0.81 | ~76ms |
| **Answer Scorer** | **~35-36%** | **~26-27%** | **~0.81** | **~150-200ms** |

**预期改善**：
- ✅ Recall@5 提升 3-4 个百分点（接近 MMR）
- ✅ Precision@5 同步提升
- ✅ 冗余度保持不变（多样性目标未变）
- ⚠️ 选择耗时增加 2-3 倍（仍在可接受范围）

### 全量+生成（如果 200 题验证通过）

| 配置 | Recall@5 | F1 | 冗余度 | EM |
|------|----------|----|---------|----|
| 当前最佳 (gamma=0.5, 3-seed) | 33.76%±0.10% | 43.46%±0.21% | 0.8103±0.0005 | 26.76% |
| **Answer Scorer（预期）** | **~36%** | **~45-46%** | **~0.81** | **~28%** |

如果达到这个水平，QORE 在所有指标上都与 MMR/Top-K 持平或更优。

---

## 风险与备选

### 风险 1：Reader 推理太慢

**现象**：DPR reader 对 50 个候选推理耗时 > 500ms/题，影响端到端性能。

**缓解**：
- 切换到 `--answer_scorer_backend cross_encoder`（更快但略不准确）
- 批量推理优化（已在代码中实现，batch_size=16）
- FP16 推理（需要修改 `answer_scorer.py` 添加 `torch.float16`）

### 风险 2：Reader 分数仍然不够准确

**现象**：Recall@5 提升 < 2 个百分点，不如预期。

**缓解**：
- 尝试更大的 reader 模型（如 `facebook/dpr-reader-multiset-base`）
- 微调 reader：用 NQ train set 上的 passage-level 标注微调
- 组合信号：`final_score = α * dpr_score + (1-α) * answer_score`

### 风险 3：与生成阶段不匹配

**现象**：skip_generation 时 Recall 提升，但端到端 F1 没提升。

**原因**：选出的段落虽然含答案，但不适合 LLM 生成（如格式不友好、上下文缺失）。

**缓解**：
- 同时跑端到端实验（不 skip_generation）
- 分析 F1 未提升的 case，调整 reader 或 prompt

---

## 后续改进方向

如果 Answer Scorer 有效，可以进一步：

1. **联合训练**：端到端训练 reader + QORE，直接优化 QA F1
2. **多信号融合**：`score = α*dpr + β*reader + γ*bm25`
3. **查询自适应**：简单问题用纯 relevance，复杂问题用 answer likelihood

---

## 常见问题

**Q1: 为什么不直接用 Top-K on answer scores？**

A: 那就退化成纯答案覆盖，丢失多样性。QORE 的优势是在答案覆盖的基础上**加上**多样性约束，两者兼得。

**Q2: Answer scorer 会影响 MMR/Top-K 吗？**

A: 当前实现只在 `method=qore` 时生效。如果要对比，需要单独跑 MMR/Top-K with answer scorer。

**Q3: 可以用其他 reader 吗？**

A: 可以。修改 `answer_scorer.py` 的 `model_name` 参数，换成任何 HuggingFace 的 DPR reader 或 QA 模型。

---

## 交付检查清单

运行 200 题验证后，检查：
- [ ] Recall@5 是否提升 ≥ 2 个百分点
- [ ] 冗余度是否保持 < 0.82
- [ ] 选择耗时是否 < 250ms/题
- [ ] 代码运行无错误

如果全部通过，继续全量评测；否则分析瓶颈，尝试备选方案。
