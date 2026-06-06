# QORE 项目说明

你之前跑过 DUET-VLM 的代码，对"token 太多要压缩"这事应该有概念了。QORE 这个项目跟 DUET 没有代码上的继承关系，是全新写的，但核心想法有关联——都是"从一堆东西里选一个子集出来"。区别在于我们换了应用场景，也换了选择的方法。

下面把项目情况跟你讲清楚。

---

## 这个项目在干嘛

简单来说：LLM 推理的时候，上下文信息太多装不下（KV cache 爆了，或者 RAG 检索回来太多段落塞不进 context window）。你得从 N 个候选里选 K 个留下来。

现在所有方法（H2O、SnapKV、MMR 之类的）都是贪心选——按分数排序取 top-K。问题是：两个分数都很高但内容几乎一样的东西，贪心会把它俩都选上，浪费了一个名额。

我们的做法是把这个选择问题写成一个 QUBO（二次无约束二值优化），用模拟退火/QAOA来解。QUBO 的好处是能同时考虑"单个项的质量"和"两两之间的冗余"，全局最优而不是贪心。

---

## 和 DUET-VLM 的关系

| | DUET-VLM | QORE |
|---|---------|------|
| 解决的问题 | VLM 里视觉 token 太多 | LLM 里上下文信息太多 |
| 选择什么 | 图像 patch token | KV-Cache entries / RAG 段落 |
| 怎么选 | 贪心 top-K | QUBO 优化（质量 + 冗余一起考虑）|
| 代码关系 | — | 完全独立，零继承 |

说白了，数学上是同一个问题的不同实例，但工程实现和应用场景完全不一样。论文里也不会提 DUET。

---

## 数学公式

就一个目标函数：

```
min E(x) = -Σ aᵢxᵢ + γ·Σ bᵢⱼxᵢxⱼ + λ(Σxᵢ - K)²

xᵢ ∈ {0,1}  选还是不选
aᵢ           质量分（越大越该留）
bᵢⱼ          i和j的冗余度（越大越不该同时留）
K            预算
λ            约束惩罚权重
γ            冗余项权重（代码里自动调的）
```

这是标准 QUBO 形式，能直接扔给量子退火器或 QAOA 线路跑。当然我们主实验用经典模拟退火就够了，不需要真量子硬件。

---

## 两个应用场景

**RAG 段落选择**：检索回来 100 个段落，context window 只能放 5-10 个。用 QORE 选出来的比 top-K 和 MMR 覆盖面更广（不会把名额浪费在说同一件事的重复段落上）。

**KV-Cache 驱逐**：长文本生成的时候 cache 满了，要决定丢哪些旧 token。现有方法按"被关注程度"排序丢，我们额外考虑 key 向量的相似度——两个 key 很像的 entry 只需要留一个。

---

## 量子的部分

QAOA 和 quantum kernel 在代码里都实现了（Qiskit、PennyLane、TensorCircuit 三个框架各一版），但你跑主实验的时候用不到——主实验全用模拟退火。量子部分只在 ablation 里出场，用来对比"QAOA 跟 SA 解的质量差多少"以及"量子 kernel 比 cosine similarity 好多少"。

不需要量子计算机，不需要 IBM Quantum 账号，在 GPU 上用模拟器跑就行。

---

## 代码结构

```
QORE-VLM/
├── qore/                    ← 核心框架，你不用动
│   ├── qubo.py              # QUBO 矩阵构建
│   ├── signals.py           # 冗余度计算
│   ├── solvers/             # SA / QAOA×3 / greedy / brute
│   └── kernels/             # quantum kernel×3
│
├── applications/
│   ├── rag/                 ← 跑 RAG 实验看这个
│   │   ├── selector.py      # select_passages(method="qore"|"topk"|"mmr")
│   │   └── baselines/       # top-K, MMR
│   └── kv_cache/            ← 跑 KV-Cache 实验看这个
│       ├── qore_cache.py    # QORECache，直接当 past_key_values 传
│       └── baselines/       # H2O, Random, Window
│
├── scripts/                 ← 你跑实验直接 bash 这里的脚本
│   ├── rag/run_nq.sh
│   ├── kv_cache/run_longbench.sh
│   └── ablations/run_all.sh
│
├── docs/
│   └── experiment_guide.md  ← 你的操作手册，详细的步骤都在这
│
└── reproduction/            ← 结果提交到这个目录
    └── _TEMPLATE/           # 复制一份改个名，按格式填
```

---

## 你要做的事

### 1. 配环境

```bash
git clone https://github.com/tivilou/QORE-VLM.git
cd QORE-VLM
pip install dwave-neal dimod
pip install pennylane tensorcircuit qiskit qiskit-algorithms qiskit-optimization
pip install torch transformers datasets accelerate sentence-transformers faiss-gpu

# 跑一下测试确认装对了
python -m pytest qore/tests/ applications/ -q
# 应该看到 "78 passed"
```

### 2. 跑实验

优先级：

1. `bash scripts/rag/run_nq.sh` — Natural Questions 上的 RAG 实验
2. `DATASET=hotpotqa bash scripts/rag/run_nq.sh` — HotpotQA
3. `bash scripts/kv_cache/run_longbench.sh` — KV-Cache 在 LongBench 上
4. `bash scripts/ablations/run_all.sh` — 消融

前三个跑完就够写论文 experiments section 了。具体的模型下载、数据准备、参数怎么覆盖，都写在 `docs/experiment_guide.md` 里了。

### 3. 提交结果

```bash
cp -r reproduction/_TEMPLATE reproduction/Bootrear_2026-07-XX
# 把跑出来的 json/csv 放进 results/ 子目录
# 填一下 README.md 和 env.md（硬件、软件版本）
git add reproduction/Bootrear_2026-07-XX/
git commit -m "repro: Bootrear — RAG + KV-Cache results"
git push origin main
```

---

## 一些你可能会问的

**和 DUET 有代码重叠吗？** 没有，从零写的，场景也不一样。

**需要量子计算机吗？** 不需要。主实验用模拟退火跑在普通 GPU 上。QAOA 用模拟器，只出现在 ablation 里。

**QORECache 怎么用？** 它继承了 HuggingFace 的 DynamicCache，直接传给 generate 就行：
```python
from applications.kv_cache import QORECache
cache = QORECache(max_capacity=1024, num_layers=32)
output = model.generate(input_ids, past_key_values=cache, max_new_tokens=500)
```

**RAG 实验需要额外准备什么？** 需要段落语料库的 embedding。推荐 `sentence-transformers` + `faiss-gpu` 做检索，然后 QORE 负责从检索结果里选子集。

**跑出来的数和预期对不上怎么办？** 正常，记下来就行。如果某个场景 QORE 没赢 baseline，大概率是那个场景冗余度不高——我们的方法本来就是在冗余高的时候才有优势。

**论文分工？** 你负责 Experiments section（跑数字、画图表、写实验设置和分析），我写 Method、Related Work 和 Introduction，Discussion 一起写。

---

## 论文信息

标题：QORE: Quantum-Optimized Context Reduction for Large Language Models

投稿目标：ICLR 2027 或 AAAI 2027

故事线：
- 现在的 LLM 上下文选择方法都是贪心的，忽略了候选项之间的冗余
- 我们把它形式化成 QUBO，用组合优化（而非贪心）求解
- 两个场景（RAG + KV-Cache）验证通用性
- 三个量子框架实现证明形式化的 backend-agnostic 特性
- 模拟退火就能赢 baseline，QAOA 提供量子演进路径

有问题随时问我。仓库地址：https://github.com/tivilou/QORE-VLM
