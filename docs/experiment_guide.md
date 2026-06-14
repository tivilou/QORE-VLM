# Experiment Guide for Bootrear

> 本文档是给 Bootrear 的实验指南。包含环境配置、实验清单、运行命令和结果提交规范。

---

## 1. 环境配置

### 硬件要求

主力实验用 LLaMA-3-8B（fp16 权重约 16GB），不同实验显存需求不同：

- **RAG 实验**：24GB 显存够用（RTX 4090 / A5000 / A6000）。context 短，只放 K 个段落。
- **KV-Cache 实验**：建议 40GB（A100 40GB / A6000）。LongBench 输入长（8K+ token），prefill 阶段 KV cache 峰值较高。注意 QORE 本身就是压 cache 的，跑起来之后峰值会降。
- **70B 模型（可选）**：才需要 80GB。主力实验用不到。

其他：RAM ≥ 32GB；存储 ≥ 100GB（模型权重 + 数据集，其中 `wiki_dpr` 语料约 35GB）。

### 软件安装

```bash
# 1. Clone 仓库
git clone https://github.com/tivilou/QORE-VLM.git
cd QORE-VLM

# 2. 创建环境
conda create -n qore python=3.10 -y
conda activate qore

# 3. 安装核心依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.44 datasets accelerate

# 4. 安装 QORE（含量子库）
pip install dwave-neal dimod
pip install pennylane tensorcircuit
pip install qiskit qiskit-algorithms qiskit-optimization

# 5. 安装 RAG 相关
pip install sentence-transformers faiss-gpu

# 6. 验证安装
python -m pytest qore/tests/ applications/ -q
# 应该看到 "78 passed"
```

### 模型下载

```bash
# LLaMA-3-8B-Instruct (需要 HF token)
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir models/llama3-8b

# Mistral-7B-Instruct (备选)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/mistral-7b

# Embedding model (RAG 用)
huggingface-cli download BAAI/bge-base-en-v1.5 --local-dir models/bge-base
```

### 数据集下载

```bash
# 放在 datasets/ 目录下（已被 .gitignore 排除）
mkdir -p datasets

# Natural Questions (RAG)
python scripts/rag/download_nq.py

# HotpotQA (RAG)
python scripts/rag/download_hotpotqa.py

# LongBench (KV-Cache)
python scripts/kv_cache/download_longbench.py

# RULER (KV-Cache)
python scripts/kv_cache/download_ruler.py
```

---

## 2. 实验清单

### 2.1 RAG 实验（优先级高）

| ID | 实验 | 模型 | 数据集 | 命令 |
|----|------|------|--------|------|
| RG-1 | QORE vs top-K vs MMR | LLaMA-3-8B | Natural Questions | `bash scripts/rag/run_nq.sh` |
| RG-2 | QORE vs top-K vs MMR | LLaMA-3-8B | HotpotQA | `bash scripts/rag/run_hotpotqa.sh` |
| RG-3 | 模型泛化 | Mistral-7B | NQ + HotpotQA | `bash scripts/rag/run_nq.sh mistral` |
| RG-4 | Budget sweep K=3,5,8,10,15 | LLaMA-3-8B | NQ | `bash scripts/rag/sweep_budget.sh` |
| RG-5 | Solver 对比 (SA vs QAOA×3) | LLaMA-3-8B | NQ (subset) | `bash scripts/rag/compare_solvers.sh` |
| RG-6 | Kernel 对比 (cosine vs quantum×3) | LLaMA-3-8B | NQ (subset) | `bash scripts/rag/compare_kernels.sh` |

### 2.2 KV-Cache 实验

| ID | 实验 | 模型 | 数据集 | 命令 |
|----|------|------|--------|------|
| KC-1 | QORE vs H2O vs Window vs Random | LLaMA-3-8B | LongBench | `bash scripts/kv_cache/run_longbench.sh` |
| KC-2 | 同上 | LLaMA-3-8B | RULER | `bash scripts/kv_cache/run_ruler.sh` |
| KC-3 | Perplexity 对比 | LLaMA-3-8B | PG-19 | `bash scripts/kv_cache/run_perplexity.sh` |
| KC-4 | 模型泛化 | Mistral-7B | LongBench | `bash scripts/kv_cache/run_longbench.sh mistral` |
| KC-5 | Cache capacity sweep | LLaMA-3-8B | LongBench | `bash scripts/kv_cache/sweep_capacity.sh` |

### 2.3 Ablation 实验

| ID | 实验 | 命令 |
|----|------|------|
| AB-1 | λ sensitivity (0.5, 1, 2, 5, 10) | `bash scripts/ablations/sweep_lambda.sh` |
| AB-2 | Block size (16, 32, 48, 64) | `bash scripts/ablations/sweep_block_size.sh` |
| AB-3 | Trigger interval T (32, 64, 128, 256) | `bash scripts/ablations/sweep_trigger.sh` |
| AB-4 | QAOA depth p (1, 2, 3, 4) | `bash scripts/ablations/sweep_qaoa_depth.sh` |

### 2.4 Overhead 实验

| ID | 实验 | 命令 |
|----|------|------|
| OH-1 | Per-token latency | `bash scripts/overhead/measure_latency.sh` |
| OH-2 | Peak memory | `bash scripts/overhead/measure_memory.sh` |

---

## 3. 运行说明

### 单个实验示例

```bash
# RAG: Natural Questions
bash scripts/rag/run_nq.sh

# 输出在 results/rag/nq/ 目录下:
#   results/rag/nq/qore_sa.json
#   results/rag/nq/topk.json
#   results/rag/nq/mmr.json
#   results/rag/nq/summary.csv    ← 对比表格
```

### 参数覆盖

所有脚本支持环境变量覆盖默认参数：

```bash
# 换模型
MODEL_PATH=models/mistral-7b bash scripts/rag/run_nq.sh

# 换 budget
K=10 bash scripts/rag/run_nq.sh

# 换 solver
SOLVER=qaoa_tc bash scripts/rag/run_nq.sh

# 限制测试数量（调试用）
MAX_SAMPLES=50 bash scripts/rag/run_nq.sh
```

### 全量运行

```bash
# 跑所有 RAG 实验
bash scripts/rag/run_all.sh

# 跑所有 KV-Cache 实验
bash scripts/kv_cache/run_all.sh

# 跑所有 ablation
bash scripts/ablations/run_all.sh
```

---

## 4. 结果格式

每个实验输出两种文件：

### JSON 详细结果

```json
{
  "experiment": "RG-1",
  "method": "qore_sa",
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "dataset": "natural_questions",
  "K": 5,
  "metrics": {
    "accuracy": 0.423,
    "f1": 0.512,
    "recall_at_K": 0.67,
    "redundancy_ratio": 0.23,
    "diversity_score": 0.77,
    "avg_time_ms": 12.3
  },
  "config": {
    "solver": "anneal",
    "num_reads": 50,
    "lam": 2.0,
    "redundancy_method": "cosine"
  },
  "timestamp": "2026-07-15T14:30:00",
  "hardware": "1xA100-80GB",
  "num_samples": 3610
}
```

### CSV 对比表格

```csv
method,accuracy,f1,recall_at_K,redundancy_ratio,diversity_score,time_ms
qore_sa,0.423,0.512,0.67,0.23,0.77,12.3
topk,0.389,0.471,0.52,0.45,0.55,0.1
mmr_0.5,0.401,0.489,0.61,0.31,0.69,5.2
mmr_0.7,0.412,0.503,0.58,0.38,0.62,4.8
```

---

## 5. 结果提交

按 `reproduction/_TEMPLATE/` 格式提交：

```bash
cd QORE-VLM

# 创建你的结果目录
cp -r reproduction/_TEMPLATE reproduction/Bootrear_2026-07-XX

# 把结果放进去
cp -r results/ reproduction/Bootrear_2026-07-XX/results/
# 填写 README.md, env.md

# 提交
git add reproduction/Bootrear_2026-07-XX/
git commit -m "repro: Bootrear — RAG + KV-Cache full benchmark"
git push origin main
```

---

## 6. 优先级建议

1. **先跑 RG-1 和 RG-2**（RAG NQ + HotpotQA）——这是论文最强的 demo 场景
2. 再跑 KC-1（KV-Cache LongBench）——第二个应用
3. 然后 ablation（RG-5, RG-6, AB-1~4）——补充实验
4. 最后 overhead（OH-1, OH-2）——效率分析

如果时间有限，RG-1 + RG-2 + KC-1 三个实验就够撑起论文的 experiments section。

---

## 7. 常见问题

**Q: CUDA OOM 怎么办？**
A: 减小 batch_size 或用 `--load-in-4bit`。KV-Cache 实验测的是长序列，prefill 峰值显存高，建议 40GB 卡。如果只有 24GB，把 `MAX_SAMPLES` 调小、或限制输入长度也能跑。

**Q: QAOA 太慢？**
A: 正常。QAOA 实验只在 subset (MAX_SAMPLES=100) 上跑，用于 scaling 分析。主实验用 SA。

**Q: 结果和论文预期不一致？**
A: 记录下来，在 notes.md 里说明。负面结果也是有价值的。

---

*文档版本: 2026-06-06*
