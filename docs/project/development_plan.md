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

## 进度总览

| Phase | 状态 | 完成日期 | 测试数 |
|-------|------|---------|--------|
| Phase 1: Core QUBO Framework | ✅ 完成 | 2026-06-06 | 30 |
| Phase 2: RAG Context Selection | ✅ 完成 | 2026-06-06 | 19 |
| Phase 2.5: HF Cache API 调研 | ✅ 完成 | 2026-06-06 | — |
| Phase 3: KV-Cache Eviction | ✅ 完成 | 2026-06-06 | 17 |
| Phase 4: Quantum Components | ✅ 完成 | 2026-06-06 | 12 |
| Phase 5: 实验脚本 + 交接 | ✅ 完成 | 2026-06-06 | — |
| Phase 6: 正式实验 (Bootrear) | ⏳ 待开始 | — | — |
| Phase 7: 论文写作 | ⏳ 待开始 | — | — |

**总计 78 个测试全部通过。**

---

## Phase 1: Core QUBO Framework ✅

**产出**：`qore/` 核心模块

```
qore/
├── __init__.py
├── qubo.py              # QUBO 矩阵构建 (含 gamma 自动调参)
├── signals.py           # cosine/RBF redundancy + quality 归一化
├── block_decompose.py   # 大问题按比例分块
└── solvers/
    ├── __init__.py      # 统一 dispatch (method 参数切换)
    ├── brute.py         # 精确枚举 (N ≤ 20)
    ├── anneal.py        # 模拟退火 (dwave-neal)
    └── greedy.py        # greedy top-K (baseline)
```

**验证结果**：
- 合成实验中，高冗余场景下 SA-QUBO 能量改善 +31（vs greedy），cluster coverage 从 1/5 → 5/5
- 低冗余场景下，SA-QUBO 与 greedy 持平（不伤害）
- SA solve time < 50ms for N=32 (num_reads=30)

---

## Phase 2: RAG Context Selection ✅

**产出**：`applications/rag/`

```
applications/rag/
├── selector.py          # 统一 API: select_passages(method="qore"|"topk"|"mmr")
├── signals_rag.py       # 从 embedding 构建 quality/redundancy
├── baselines/
│   ├── topk.py          # Top-K by relevance
│   └── mmr.py           # Maximal Marginal Relevance
├── demo_synthetic.py    # 合成 RAG 场景实验
└── tests/
    └── test_selector.py # 19 tests
```

**设计决策**：
- 两阶段选择：先 quality 预筛 (top-3K)，再 QUBO 做 diversity-aware 选择
- gamma 自动调参：基于 top-K 候选项内的实际冗余度

**验证结果** (K=5, 5 gold among 100 passages)：
- QORE-SA: 60% gold recall (best), 4/5 groups covered
- MMR (λ=0.7): 40% gold recall, 5/5 groups covered
- Top-K: 20% gold recall, 1/5 groups covered

---

## Phase 2.5: HF Cache API 调研 ✅

**产出**：`docs/phase2_5_hf_cache_api.md`

**核心结论**：子类化 `DynamicCache.update()` 即可实现自定义 eviction。
不需要复制 HF 的 generation_utils.py（DUET-VLM 复制了 4673 行来做这事）。

---

## Phase 3: KV-Cache Eviction ✅

**产出**：`applications/kv_cache/`

```
applications/kv_cache/
├── qore_cache.py        # QORECache(DynamicCache) — drop-in 替换
├── signals_kv.py        # key-norm quality + pairwise key similarity
├── baselines/
│   ├── h2o_cache.py     # Heavy Hitter (greedy top-K by attention)
│   ├── random_cache.py  # Random eviction
│   └── window_cache.py  # Sliding window
└── tests/
    └── test_cache.py    # 17 tests
```

**设计决策**：
- Eviction 在 `update()` 中触发（last layer 完成后），同步应用到所有层
- Sink tokens 始终保留（attention sink 现象）
- Block decomposition 处理大 cache (N > 64)
- Key-norm 作为 attention 的 cheap proxy（不需要 hook）

---

## Phase 4: Quantum Components ✅

**产出**：3 个 QAOA solver + 3 个 quantum kernel 实现

```
qore/solvers/
├── qaoa_qiskit.py          # via qiskit-optimization
├── qaoa_pennylane.py       # via PennyLane (autograd)
└── qaoa_tensorcircuit.py   # via TensorCircuit (tensor contraction)

qore/kernels/
└── __init__.py             # quantum_kernel(features, backend=...) 统一接口
                            # 支持 "pennylane" / "tensorcircuit" / "qiskit"
```

**设计决策**：
- 三个框架覆盖主流量子生态（IBM/Xanadu/腾讯）
- Lazy import：只在调用特定 backend 时才加载对应库
- Cross-backend consistency test 通过（三个 kernel 产出一致的矩阵）
- 统一 dispatch：`method="qaoa_qk"/"qaoa_pl"/"qaoa_tc"` 一个参数切换

---

## Phase 5: 实验脚本 + 交接 ✅

**产出**：

```
docs/experiment_guide.md     # Bootrear 完整实验指南
scripts/
├── rag/
│   ├── run_nq.sh            # NQ 主实验
│   ├── run_all.sh           # 全量 RAG 实验
│   ├── eval_rag.py          # 评测框架
│   └── summarize.py         # 结果汇总
├── kv_cache/
│   ├── run_longbench.sh     # LongBench 主实验
│   ├── run_all.sh           # 全量 KV-Cache 实验
│   ├── eval_kv_cache.py     # 评测框架
│   └── summarize.py         # 结果汇总
└── ablations/
    └── run_all.sh           # lambda/depth/block/trigger sweeps
```

**所有脚本支持环境变量覆盖**（MODEL_PATH, K, SOLVER, MAX_SAMPLES 等）。

---

## Phase 6: 正式实验 ⏳（Bootrear）

**优先级**：
1. RG-1: QORE vs baselines on Natural Questions
2. RG-2: QORE vs baselines on HotpotQA
3. KC-1: QORE vs baselines on LongBench
4. Ablations: solver/kernel/lambda/depth sweeps
5. Overhead: latency + memory

**详细清单和运行说明**：见 `docs/experiment_guide.md`

---

## Phase 7: 论文写作 ⏳

**分工**：
- tivilou: Method (Section 3–5), Related Work, Introduction, Abstract
- Bootrear: Experiments (Section 6), 表格/图, Appendix
- 共同: Discussion, Conclusion

**Target**: ICLR 2027 (deadline ~Sep 2026) 或 AAAI 2027 (deadline ~Aug 2026)

---

## 技术栈总结

| 组件 | 依赖 | 角色 |
|------|------|------|
| dwave-neal + dimod | 核心 | 模拟退火 SA (主力 solver) |
| PennyLane | optional | QAOA + quantum kernel (backend 1) |
| TensorCircuit | optional | QAOA + quantum kernel (backend 2) |
| Qiskit | optional | QAOA + quantum kernel (backend 3) |
| numpy + scipy | 核心 | 数值计算 |
| torch | 核心 | KV-Cache tensor 操作 |
| transformers | Phase 6 | 模型加载 + 推理 |
| sentence-transformers | Phase 6 | RAG embedding |
| datasets | Phase 6 | benchmark 数据加载 |

核心（SA + greedy + brute）无需任何量子库即可运行。

---

*Plan updated: 2026-06-06. All dev phases (1–5) complete.*
