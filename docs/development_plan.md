# QORE Development Plan

> 分工：tivilou 负责代码开发 + 简单验证测试，Bootrear 负责正式实验。

---

## 角色分工

| 角色 | 负责人 | 职责范围 |
|------|--------|---------|
| 开发 (Dev) | tivilou | 代码实现、单元测试、小规模验证、文档 |
| 实验 (Exp) | Bootrear | 正式 benchmark 实验、大模型评测、结果收集 |

**交接点**：每个 Phase 结束时，tivilou 提供可运行的代码 + 简单验证结果 + 使用说明；
Bootrear 拉取代码，按说明在正式环境跑全量实验，结果提交到 `reproduction/`。

---

## Phase 1: Core QUBO Framework（Dev: tivilou）

**目标**：实现 `qore/` 核心模块，能在合成数据上验证"QUBO选择 > greedy top-K"。

**产出文件**：
```
qore/
├── __init__.py
├── qubo.py              # 从 quality + redundancy 构建 Q 矩阵
├── signals.py           # 构造 aᵢ 和 bᵢⱼ 的通用接口
├── block_decompose.py   # 将大问题分解为小块 QUBO
├── solvers/
│   ├── __init__.py
│   ├── brute.py         # 精确枚举 (N ≤ 20)
│   ├── anneal.py        # 模拟退火 (dwave-neal)
│   └── greedy.py        # greedy top-K baseline (对照组)
└── tests/
    ├── test_qubo.py     # Q 矩阵构建正确性
    ├── test_solvers.py  # 各 solver 在小规模问题上的一致性
    └── test_synthetic.py # 合成数据: QUBO-SA vs greedy 信息保留对比
```

**验证标准**：
- 合成实验（N=50, K=10, 人工注入冗余 token 对）中，SA-QUBO 选出的子集
  信息保留（重构误差）显著优于 greedy top-K
- 所有 solver 在 N≤16 时与 brute-force 结果一致

**预计耗时**：1–2 周

---

## Phase 2: KV-Cache Eviction Integration（Dev: tivilou）

**目标**：将 QORE 接入 HuggingFace LLaMA 推理流程，替换默认 KV-cache 管理。

**产出文件**：
```
applications/kv_cache/
├── __init__.py
├── eviction.py          # QORE eviction policy (hook into HF generate)
├── signals_kv.py        # 从 attention pattern 提取 aᵢ, bᵢⱼ
├── config.py            # 超参配置 (K, T, block_size, solver, α, λ)
├── baselines/
│   ├── h2o.py           # H2O reimplementation
│   ├── snapkv.py        # SnapKV reimplementation
│   └── random_evict.py  # 随机驱逐 baseline
└── tests/
    ├── test_eviction.py # 单元测试: eviction 逻辑正确性
    └── demo_longtext.py # 小规模验证: 1 条长文本, 对比各方法 perplexity
```

**验证标准**：
- `demo_longtext.py` 在单条 PG-19 长文本上跑通，输出各方法的 perplexity 对比
- QORE-SA 的 perplexity 不差于 H2O（验证集成正确性）
- overhead < 5%（计时）

**预计耗时**：2–3 周

---

## Phase 3: RAG Context Selection Integration（Dev: tivilou）

**目标**：将 QORE 接入 RAG pipeline，替换 top-K / MMR 段落选择。

**产出文件**：
```
applications/rag/
├── __init__.py
├── selector.py          # QORE passage selector
├── signals_rag.py       # 从 embeddings 提取 aᵢ, bᵢⱼ
├── config.py            # 超参配置
├── baselines/
│   ├── topk.py          # 纯 top-K by retriever score
│   ├── mmr.py           # MMR implementation
│   └── dpp_greedy.py    # Greedy DPP MAP
└── tests/
    ├── test_selector.py # 单元测试
    └── demo_nq.py       # 小规模验证: Natural Questions 前 50 条
```

**验证标准**：
- `demo_nq.py` 在 NQ 的前 50 条 query 上跑通
- QORE-SA 的 answer accuracy ≥ MMR（验证方法有效性）
- 输出清晰的对比表格

**预计耗时**：2 周

---

## Phase 4: Quantum Components（Dev: tivilou）

**目标**：实现量子核 (B2) 和 QAOA 求解器，作为 SA 的可选升级。

**产出文件**：
```
qore/
├── kernels.py           # quantum kernel + classical baselines (cosine, RBF)
└── solvers/
    └── qaoa.py          # QAOA solver (qiskit / pennylane)
```

**验证标准**：
- quantum kernel 在合成数据上能正确计算 bᵢⱼ，与 RBF 在高相似区域趋势一致
- QAOA (p=2) 在 N≤24 的 sub-QUBO 上达到 SA 解质量的 95%+
- 生成 scaling plot: approximation ratio vs circuit depth p

**预计耗时**：2 周

---

## Phase 5: Experiment Scripts & Handoff（Dev: tivilou → Exp: Bootrear）

**目标**：提供"一键跑实验"的脚本和文档，交给 Bootrear 跑正式评测。

**产出文件**：
```
scripts/
├── kv_cache/
│   ├── run_longbench.sh
│   ├── run_ruler.sh
│   ├── run_needle.sh
│   └── run_perplexity.sh    # PG-19 / GovReport
├── rag/
│   ├── run_nq.sh
│   ├── run_hotpotqa.sh
│   └── run_multihop_rag.sh
└── ablations/
    ├── sweep_budget_K.sh
    ├── sweep_block_size.sh
    ├── sweep_lambda.sh
    └── compare_solvers.sh

docs/
└── experiment_guide.md      # Bootrear 专用: 环境配置 + 运行指南 + 结果格式
```

**交接清单（给 Bootrear）**：
- [ ] 环境配置文档 (GPU 要求、依赖安装)
- [ ] 各 benchmark 的数据下载脚本
- [ ] 每个实验的运行命令 + 预期输出格式
- [ ] 结果上传规范 (→ `reproduction/Bootrear_<date>/`)

**预计耗时**：1 周

---

## Phase 6: Formal Experiments（Exp: Bootrear）

**目标**：在正式环境跑全量实验，产出论文所需的所有数字和图。

**实验清单**：

### KV-Cache 实验
| ID | 内容 | 模型 | Benchmark |
|----|------|------|-----------|
| KC-1 | QORE-SA vs H2O vs SnapKV vs PyramidKV | LLaMA-3-8B | LongBench |
| KC-2 | 同上 | LLaMA-3-8B | RULER |
| KC-3 | 同上 | LLaMA-3-8B | Needle-in-a-Haystack |
| KC-4 | Perplexity 对比 | LLaMA-3-8B | PG-19, GovReport |
| KC-5 | 模型泛化 | Mistral-7B | LongBench |
| KC-6 | Budget sweep K/N = 0.1, 0.25, 0.5 | LLaMA-3-8B | LongBench |

### RAG 实验
| ID | 内容 | 模型 | Benchmark |
|----|------|------|-----------|
| RG-1 | QORE-SA vs top-K vs MMR vs DPP-greedy | LLaMA-3-8B | Natural Questions |
| RG-2 | 同上 | LLaMA-3-8B | HotpotQA |
| RG-3 | 同上 | LLaMA-3-8B | MultiHop-RAG |
| RG-4 | 模型泛化 | Mistral-7B | NQ + HotpotQA |
| RG-5 | Budget sweep K = 3, 5, 8, 10, 15 | LLaMA-3-8B | NQ |

### Ablation 实验
| ID | 内容 | 应用 |
|----|------|------|
| AB-1 | bᵢⱼ 方法: cosine vs RBF vs quantum kernel | Both |
| AB-2 | Block size: n = 16, 32, 48, 64 | KV-Cache |
| AB-3 | λ sensitivity: 0.5, 1, 2, 5, 10 | Both |
| AB-4 | Trigger frequency T = 32, 64, 128, 256 | KV-Cache |
| AB-5 | Solver: greedy vs SA vs QAOA (p=1,2,3) | Both |
| AB-6 | DPP sampling vs QORE-SA | Both |

### Overhead 实验
| ID | 内容 |
|----|------|
| OH-1 | Wall-clock: QORE vs no-eviction vs greedy, per token latency |
| OH-2 | Peak memory: KV-Cache 各方法对比 |
| OH-3 | QAOA circuit depth vs solve time vs solution quality |

**预计耗时**：3–4 周（取决于 GPU 资源）

---

## Phase 7: Paper Writing（Dev: tivilou, Exp: Bootrear 提供数据）

**目标**：完成论文初稿。

**分工**：
- tivilou: Method (Section 3–5), Related Work, Introduction, Abstract
- Bootrear: Experiments (Section 6), 表格/图, Appendix
- 共同: Discussion, Conclusion

**预计耗时**：3–4 周

---

## 整体时间线

```
Week 1-2:   Phase 1 — Core QUBO framework
Week 3-5:   Phase 2 — KV-Cache integration
Week 5-7:   Phase 3 — RAG integration
Week 7-9:   Phase 4 — Quantum components (QAOA + kernel)
Week 9-10:  Phase 5 — Experiment scripts + handoff
Week 10-14: Phase 6 — Bootrear runs formal experiments
Week 12-16: Phase 7 — Paper writing (overlaps with Phase 6)
```

**Target submission**: ~4 months from now (Oct 2026)
**Candidate venue**: ICLR 2027 (deadline ~Sep 2026) or AAAI 2027 (deadline ~Aug 2026)

---

## 沟通与协作规范

- **代码交接**：每个 Phase 完成后打 git tag（`v0.1-core`, `v0.2-kvcache`, ...）
- **Issue tracking**：用 GitHub Issues 管理 bug / 实验问题
- **结果提交**：Bootrear 按 `reproduction/_TEMPLATE/` 格式提交到 `reproduction/`
- **同步频率**：建议每周至少一次进度同步

---

*Plan version: 2026-06-05*
