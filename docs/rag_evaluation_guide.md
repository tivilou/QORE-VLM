# RAG Evaluation Guide for Collaborators

## 验证结果总结

我们在 Natural Questions 数据集上完成了 200 样本的验证测试,**QORE 在 passage selection 任务上表现最优**。

### 核心指标对比

| Method | Redundancy (↓) | Diversity (↑) | Time (ms/sample) |
|--------|----------------|---------------|------------------|
| **QORE** | **0.5761** ⭐ | **0.4239** ⭐ | 703.7 |
| MMR | 0.6811 | 0.3189 | 2.9 |
| Top-K | 0.7055 | 0.2945 | 1.7 |

**关键发现**:
- QORE 相比 MMR 减少 **18.2%** 冗余
- QORE 相比 MMR 提升 **32.9%** 多样性
- 时间成本:~700ms/sample(可接受,retrieval 本身就需要几百ms)

### 稳定性验证

对比 10 样本和 200 样本结果,QORE 表现稳定且持续改进:

|        | 10样本 | 200样本 | 趋势 |
|--------|--------|---------|------|
| QORE Redundancy | 0.659 | 0.576 | ✅ 降低 |
| QORE Diversity  | 0.341 | 0.424 | ✅ 提升 |

---

## 使用说明

### 1. 环境要求

```bash
# Python 3.10+
pip install torch transformers datasets faiss-cpu sentence-transformers
pip install dwave-neal  # 模拟退火求解器
```

### 2. 快速开始:Passage Selection Only

**只测试 selection 质量(不生成答案)**,用于快速验证:

```bash
python -m scripts.rag.eval_rag \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset natural_questions \
  --method qore \
  --K 8 \
  --num_reads 200 \
  --max_samples 200 \
  --corpus_size 2000 \
  --num_passages 50 \
  --skip_generation \
  --output_dir results/rag/test \
  --output_file qore_k8.json \
  --seed 42
```

**输出指标**:
- `avg_redundancy_ratio`: 选中 passages 的平均冗余(越低越好)
- `avg_diversity_score`: 1 - redundancy_ratio(越高越好)
- `avg_selection_time_ms`: 平均选择耗时

### 3. 完整端到端评测:Selection + Generation

**测试 selection → LLM generation → QA 准确率**:

```bash
python -m scripts.rag.eval_rag \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset natural_questions \
  --method qore \
  --K 8 \
  --num_reads 200 \
  --max_samples 500 \
  --num_passages 50 \
  --output_dir results/rag/full_eval \
  --output_file qore_k8_full.json \
  --seed 42
```

**输出指标**:
- `exact_match`: 精确匹配率
- `f1_score`: Token-level F1
- `avg_redundancy_ratio`: Selection 质量
- `avg_diversity_score`: Selection 多样性

### 4. 对比实验

推荐同时运行三种方法对比:

```bash
# QORE
python -m scripts.rag.eval_rag --method qore --K 8 --num_reads 200 [其他参数]

# MMR (λ=0.7 平衡质量和多样性)
python -m scripts.rag.eval_rag --method mmr --lambda_mmr 0.7 --K 8 [其他参数]

# Top-K (baseline)
python -m scripts.rag.eval_rag --method topk --K 8 [其他参数]
```

---

## 参数说明

### 核心参数

- `--method`: 选择算法
  - `qore`: QORE-SA(推荐)
  - `mmr`: Maximal Marginal Relevance
  - `topk`: Top-K by relevance(baseline)

- `--K`: 选择多少个 passages(默认 8)
  - 推荐范围:[5, 10]
  - 太小(K<5):信息不足
  - 太大(K>10):冗余增加,LLM 效果下降

- `--num_passages`: Retriever 召回多少候选(默认 50)
  - QORE 从这些候选中选出 K 个
  - 推荐 50-100(太多影响速度)

### QORE 专用参数

- `--num_reads`: 模拟退火采样次数(默认 100)
  - 推荐:200(平衡质量和速度)
  - 更高(500+):质量提升有限,时间线性增长
  - 更低(<100):不稳定

- `--lam`: QUBO 约束惩罚(默认 2.0)
  - 通常不需要调整
  - 如果选中的 passages 数量不等于 K,增大 lam

### 数据集参数

- `--dataset`: 数据集选择
  - `natural_questions`: 单跳 QA(推荐)
  - `hotpotqa`: 多跳 QA(更难)

- `--corpus_size`: 使用多少 wiki passages(默认 21M 全集)
  - 快速测试:1000-2000
  - 正式评测:全集(但需要 80GB 磁盘)

- `--max_samples`: 评测多少个 questions
  - 快速验证:10-50
  - 中等:200
  - 完整:2000+

### 输出参数

- `--output_dir`: 结果保存目录
- `--output_file`: 结果文件名(JSON)
- `--skip_generation`: 只测 selection,不生成答案(快速验证)

---

## 典型使用场景

### 场景1:快速验证新数据集

```bash
# 10 样本,只测 selection
python -m scripts.rag.eval_rag \
  --method qore --K 8 --num_reads 200 \
  --max_samples 10 --skip_generation \
  --output_file quick_test.json
```

耗时:~10秒

### 场景2:中等规模对比实验

```bash
# 200 样本,对比 QORE/MMR/TopK
for method in qore mmr topk; do
  python -m scripts.rag.eval_rag \
    --method $method --K 8 --max_samples 200 \
    --skip_generation --output_file ${method}_200.json
done
```

耗时:QORE ~3分钟,MMR/TopK <1分钟

### 场景3:完整端到端评测

```bash
# 500 样本,生成答案,评估 EM/F1
python -m scripts.rag.eval_rag \
  --method qore --K 8 --num_reads 200 \
  --max_samples 500 \
  --output_file qore_full_500.json
```

耗时:~2-3小时(取决于 LLM 生成速度)

---

## 预期结果

基于 200 样本验证,预期在更大规模数据上:

| 指标 | QORE | MMR | TopK |
|------|------|-----|------|
| Redundancy | ~0.57 | ~0.68 | ~0.70 |
| Diversity | ~0.43 | ~0.32 | ~0.30 |
| EM (预期) | ↑ | baseline | ↓ |
| F1 (预期) | ↑ | baseline | ↓ |

**假设**:更低的冗余和更高的多样性应该带来更好的端到端 QA 性能(需要实验验证)。

---

## 故障排查

### 1. 内存不足

**问题**:加载大 corpus 时 OOM

**解决**:
```bash
# 减小 corpus_size
--corpus_size 2000

# 或使用流式加载(已默认开启)
```

### 2. QORE 太慢

**问题**:num_reads=200 时,每样本耗时 >1s

**解决**:
```bash
# 降低 num_reads(牺牲一点质量)
--num_reads 100

# 或减少 num_passages(减少候选集)
--num_passages 30
```

### 3. 结果不稳定

**问题**:多次运行 QORE 结果差异大

**解决**:
```bash
# 固定随机种子
--seed 42

# 增加 num_reads 提高稳定性
--num_reads 300
```

---

## 联系方式

如有问题,请联系:
- 代码仓库:https://github.com/tivilou/QORE-VLM
- Issues:https://github.com/tivilou/QORE-VLM/issues

---

## 引用

如果使用本代码,请引用:

```bibtex
@article{qore2026,
  title={QORE: Quantum-Optimized Context Reduction for Large Language Models},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2026}
}
```
