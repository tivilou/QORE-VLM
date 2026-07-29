# RAG 模块重构使用指南

## 概述

RAG 评测代码已完成模块化重构,主要改进:

1. **三种语料库模式**:
   - `aligned`: 构建包含 gold passages 的对齐语料库(推荐)
   - `precomputed`: 使用数据集自带的候选集(如 HotpotQA distractor)
   - `faiss`: 完整 21M 语料库 + FAISS 索引(最真实)

2. **模块化架构**:
   - `applications/rag/data/`: 语料库管理和数据加载
   - `applications/rag/retrieval/`: DPR/Sentence 编码器
   - `applications/rag/evaluation/`: 评估指标(EM/F1/Recall/Redundancy)
   - `applications/rag/generation/`: LLM 生成器
   - `applications/rag/selector.py`: Selection 方法(QORE/MMR/TopK)

3. **自动化批量评测**:
   - `eval_suite.py`: 一键运行多 seed × 多 method
   - 自动汇总统计(均值、标准差、显著性检验)

---

## 快速开始

### 1. 测试重构模块(5 分钟)

验证所有模块正常工作:

```bash
cd /home/Q-DUET-VLM/QORE-VLM
python -m scripts.rag.test_refactored
```

**预期输出**:
```
Testing CorpusManager...
  ✓ Aligned corpus works
Testing Evaluator...
  ✓ Evaluator works
...
✓ All tests passed
```

---

### 2. 单次评测(HotpotQA,10 分钟)

使用 `precomputed` 模式(最简单,不需要构建语料库):

```bash
python -m scripts.rag.eval_rag_refactored \
    --dataset hotpotqa_distractor \
    --corpus_mode precomputed \
    --method qore \
    --K 5 \
    --max_samples 10 \
    --skip_generation \
    --output_dir results/rag/test
```

**检查结果**:
```bash
cat results/rag/test/qore_K5_seed42.json | head -30
```

应该看到 `mean_recall > 0` (说明 gold 对齐成功)。

---

### 3. 批量评测(HotpotQA 100 样本,30 分钟)

一键运行 3 methods × 3 seeds:

```bash
python -m scripts.rag.eval_suite \
    --dataset hotpotqa_distractor \
    --corpus_mode precomputed \
    --methods qore,mmr,topk \
    --seeds 42,123,456 \
    --K 5 \
    --max_samples 100 \
    --skip_generation \
    --output_dir results/rag/hotpotqa_100
```

**查看汇总结果**:
```bash
cat results/rag/hotpotqa_100/summary.json
```

---

## 完整 NQ 评测(目标:3610 样本)

### 阶段 1: 构建对齐语料库(4-6 小时)

**问题**: NQ 数据集不自带 gold passages,需要从 wiki_dpr 匹配。

**临时方案**(快速验证):
使用 HotpotQA fullwiki 模式,绕过这个问题:

```bash
python -m scripts.rag.eval_rag_refactored \
    --dataset hotpotqa_fullwiki \
    --corpus_mode aligned \
    --corpus_output_dir data/hotpotqa_corpus \
    --method qore \
    --K 5 \
    --max_samples 50 \
    --skip_generation
```

**完整方案**(需要实现 NQ gold 匹配):

1. 修改 `build_corpus.py` 添加 NQ gold 提取逻辑
2. 或使用 DPR 检索每个问题的 top-100,人工标注 gold
3. 或从 NQ 原始数据集提取 gold context

**这是一个复杂步骤,需要讨论具体实现**。

---

### 阶段 2: 运行全量评测(27 小时串行或 9 小时并行)

构建好语料库后:

```bash
python -m scripts.rag.eval_suite \
    --dataset nq_open \
    --corpus_mode aligned \
    --corpus_output_dir data/nq_aligned_corpus \
    --methods qore,mmr,topk \
    --seeds 42,123,456 \
    --K 5 \
    --max_samples 0 \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir results/rag/nq_full
```

**监控进度**:
```bash
# 查看已完成的文件
ls -lh results/rag/nq_full/*.json

# 查看最新结果
tail -f results/rag/nq_full/qore_K5_seed42.json
```

---

## 代码结构

### 新增模块

```
applications/rag/
├── data/
│   ├── corpus_manager.py       # 统一语料库接口
│   ├── aligned.py              # 对齐语料库(gold + distractors)
│   ├── precomputed.py          # 数据集自带候选
│   ├── faiss_corpus.py         # 完整语料库 + FAISS
│   └── dataset_loader.py       # 数据集加载器
│
├── retrieval/
│   └── encoder.py              # DPR/Sentence 编码器
│
├── evaluation/
│   └── metrics.py              # 评估指标
│
└── generation/
    └── generator.py            # LLM 生成器
```

### 脚本

```
scripts/rag/
├── eval_rag_refactored.py      # 单次评测(替代原 eval_rag.py)
├── eval_suite.py               # 批量评测(新)
├── build_corpus.py             # 构建语料库(新)
└── test_refactored.py          # 模块测试(新)
```

---

## 参数说明

### 语料库模式参数

| 模式 | 参数 | 说明 |
|------|------|------|
| `aligned` | `--corpus_output_dir` | 缓存目录 |
| | `--n_distractors` | 干扰 passages 数量(默认 36000) |
| `precomputed` | (无) | 使用数据集自带候选 |
| `faiss` | `--faiss_embeddings_path` | 完整 embeddings .npy 文件 |
| | `--faiss_passages_path` | 完整 passages .pkl 文件 |

### Selection 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | qore / mmr / topk | qore |
| `--K` | 选择数量 | 5 |
| `--num_reads` | QORE SA 读取次数 | 100 |
| `--lam` | QORE 惩罚权重 | 2.0 |
| `--lambda_mmr` | MMR lambda 参数 | 0.7 |

### 批量评测参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--methods` | 逗号分隔的方法 | `qore,mmr,topk` |
| `--seeds` | 逗号分隔的种子 | `42,123,456` |
| `--resume` | 跳过已完成的运行 | (flag) |

---

## 常见问题

### Q1: `test_refactored.py` 报错

**错误**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**解决**:
```bash
pip install sentence-transformers
```

---

### Q2: Aligned 模式构建失败

**错误**: `ValueError: Some gold passages have no embedding`

**原因**: 数据集的 `gold_passages` 字段缺失或格式不对。

**解决**: 
1. 检查数据集是否有 `gold_passages` 字段
2. 或使用 `precomputed` 模式(HotpotQA distractor)

---

### Q3: 生成阶段 OOM

**错误**: `CUDA out of memory`

**解决**:
1. 加 `--skip_generation` 跳过生成(只测 selection)
2. 或降低 `--max_new_tokens` (默认 128)
3. 或使用更小的模型

---

### Q4: 如何并行运行多个 seed?

**方法 1**: 手动并行(如果 GPU 显存够)
```bash
# 终端 1
python -m scripts.rag.eval_rag_refactored ... --seed 42 &

# 终端 2
python -m scripts.rag.eval_rag_refactored ... --seed 123 &

# 终端 3
python -m scripts.rag.eval_rag_refactored ... --seed 456 &
```

**方法 2**: 使用 `eval_suite.py` 的串行模式(更稳定)

---

## 向后兼容

**原有脚本仍然可用**:
- `scripts/rag/eval_rag.py` (原版,未修改)
- `applications/rag/selector.py` (未修改)
- `applications/rag/baselines/` (未修改)

**新脚本是独立的**:
- `eval_rag_refactored.py` 不影响原有代码
- 可以逐步迁移,或两套并行使用

---

## 下一步

### 立即可做(今天)
1. ✅ 运行 `test_refactored.py` 验证模块
2. ✅ 用 HotpotQA distractor 跑 100 样本(precomputed 模式)
3. ✅ 验证结果格式和指标正确

### 短期(1-2 天)
4. 实现 NQ gold passage 提取逻辑
5. 构建 NQ 对齐语料库
6. 小规模测试(50 样本)

### 中期(3-5 天)
7. NQ 全量评测(3610 样本 × 3 seed × 3 method)
8. 结果分析和可视化
9. 准备论文素材

---

## 获取帮助

**遇到问题时**:
1. 查看错误信息(通常很明确)
2. 检查本文档的"常见问题"部分
3. 运行 `test_refactored.py` 验证模块状态
4. 联系我们讨论

**反馈改进建议**:
- 哪些参数不清楚
- 哪些错误提示不够友好
- 需要什么新功能

---

**文档版本**: 2024-07-16
**代码版本**: refactor-v1
