<div align="center">
  <h1>QORE</h1>
  <p><b>Q</b>uantum-<b>O</b>ptimized Context <b>RE</b>duction for Large Language Models</p>
  <p><i>(read as "Q-core")</i></p>

  <a href="https://github.com/tivilou/QORE-VLM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/status-research%20WIP-orange" alt="status">
</div>

---

QORE is a research project that brings **quantum optimization** to context management
in Large Language Models. It provides a unified QUBO (Quadratic Unconstrained Binary
Optimization) framework for deciding **what information an LLM should attend to** under
a finite context/memory budget — and solves this selection problem with quantum and
quantum-inspired optimizers.

We demonstrate the framework on two complementary applications:

- **KV-Cache Eviction** (fine-grained, dynamic): which token-level KV entries to
  retain during autoregressive generation
- **RAG Context Selection** (coarse-grained, static): which retrieved passages to
  include in the prompt before generation

Both are instances of the same mathematical problem — coupled subset selection under
a cardinality budget — and share a single solver stack.

## Motivation

LLMs operate under finite context budgets. Whether it's a KV cache hitting its
physical memory limit during long-sequence generation, or a RAG pipeline that must
choose which retrieved passages fit within the context window, the core question is
the same:

> **From N candidates, select K to keep, maximizing information while minimizing
> redundancy.**

Current approaches in both settings rely on **greedy, score-based heuristics**:
rank items by a scalar importance score, keep the top-K. This ignores **pairwise
interactions** — two items can both score high individually yet be mutually redundant
(e.g., two KV entries encoding the same fact, or two passages covering the same
evidence). Greedy selection wastes budget on redundancy.

QORE reformulates context selection as a **QUBO** and solves it with combinatorial
optimizers that respect pairwise coupling — including quantum solvers that are
theoretically suited to this problem structure.

## Unified Formulation

Define binary decision variables `x ∈ {0,1}^N`, where `xᵢ = 1` means item i is kept.

**Objective (minimization):**

```
E(x) = - Σᵢ aᵢ xᵢ  +  Σᵢ<ⱼ bᵢⱼ xᵢ xⱼ  +  λ (Σᵢ xᵢ - K)²
         ─────────      ──────────────────      ──────────────────
          quality        redundancy penalty      budget constraint
```

| Symbol | Meaning | KV-Cache instance | RAG instance |
|--------|---------|-------------------|--------------|
| N | candidate pool size | KV entries in cache | retrieved passages |
| K | budget | cache capacity | max passages in context |
| aᵢ | individual quality | cumulative attention score | query-passage relevance |
| bᵢⱼ | pairwise redundancy | key vector similarity | passage content overlap |
| λ | constraint weight | tuned per model | tuned per retriever |

The QUBO matrix Q is constructed as:
```
Qᵢᵢ = -aᵢ + λ(1 - 2K)
Qᵢⱼ = bᵢⱼ + 2λ         (i ≠ j)
```

Minimizing `x^T Q x` over binary x gives the optimal subset.

## Two Applications, One Framework

### Application A: KV-Cache Eviction

During long-sequence generation, the KV cache grows until it exceeds memory. QORE
replaces greedy eviction (H2O, SnapKV, PyramidKV) with QUBO-optimal selection:

- **Quality signal** `aᵢ`: cumulative attention received by each KV entry from
  subsequent tokens (the "heavy hitter" score from H2O, or windowed attention from
  SnapKV)
- **Redundancy signal** `bᵢⱼ`: cosine similarity between key vectors of entries i
  and j — high similarity means keeping both wastes budget
- **Timing**: solve every T generation steps (amortized overhead)
- **Scaling**: block decomposition by attention head or position window

### Application B: RAG Context Selection

After retrieval, a typical pipeline has 50–200 candidate passages but can only fit
5–15 in context. QORE replaces greedy re-ranking and MMR with QUBO-optimal selection:

- **Quality signal** `aᵢ`: retriever relevance score (embedding similarity to query)
- **Redundancy signal** `bᵢⱼ`: passage-pair semantic similarity (embedding cosine)
  or lexical overlap (n-gram Jaccard)
- **Timing**: once before generation (offline)
- **Scaling**: N is naturally small (50–200), no decomposition needed — direct solve

## Why Quantum

- **Coupling-aware optimization.** The `bᵢⱼ` terms make the objective non-separable.
  Greedy top-K cannot handle quadratic coupling. QUBO solvers optimize the full
  coupled objective.
- **Natural problem structure.** Subset selection with pairwise penalties maps directly
  onto Ising / QUBO — the native problem class for quantum annealers and QAOA.
- **Graceful degradation.** The same QUBO runs on classical simulated annealing
  (always available), quantum-inspired solvers, gate-based QAOA, or quantum annealing
  hardware. No quantum hardware is required to reproduce results.
- **Feasible scale.** KV-cache blocks have 24–64 variables per sub-QUBO; RAG selection
  has 50–200 variables total. Both are within near-term quantum device capacity.

## Theoretical Grounding: DPP & GBS

The selection problem has a deep connection to **Determinantal Point Processes (DPP)**:
selecting a diverse, high-quality subset is exactly k-DPP MAP inference (NP-hard in
general). This connects to **Gaussian Boson Sampling (GBS)**, which naturally samples
subsets weighted by matrix permanents / hafnians — providing a principled quantum
sampling approach complementary to QUBO optimization.

See `docs/technical_roadmap.md` for the full theoretical treatment.

## Solver Backends

| Backend | Library | Role |
|---------|---------|------|
| Simulated annealing | `dwave-neal` / `dimod` | Classical baseline, always available |
| QAOA (gate model) | `qiskit` / `pennylane` | Variational quantum optimizer |
| Quantum annealing | `dwave-system` | Hardware sampler (optional) |
| Quantum kernel | `pennylane` | Richer `bᵢⱼ` via Hilbert-space overlap |
| Brute force | `numpy` | Exact reference for small N |

## Project Structure

```
QORE/
├── qore/                    # core QUBO framework
│   ├── qubo.py              # build Q matrix from quality + redundancy signals
│   ├── solvers/             # SA / QAOA / brute-force / quantum annealing
│   ├── kernels.py           # classical + quantum kernels for bᵢⱼ
│   └── block_decompose.py   # spatial / head-wise block partitioning
├── applications/
│   ├── kv_cache/            # KV-cache eviction integration
│   │   ├── eviction.py      # hook into HuggingFace KV cache
│   │   └── baselines/       # H2O, SnapKV, PyramidKV reimplementations
│   └── rag/                 # RAG context selection integration
│       ├── selector.py      # post-retrieval QUBO selection
│       └── baselines/       # MMR, DPP-greedy, top-K reranker
├── experiments/             # configs, scripts, result logs
├── reproduction/            # collaborator-submitted reproduction results
├── docs/                    # technical roadmap, paper drafts
└── scripts/                 # evaluation & benchmarking scripts
```

## Status

This repository is an early-stage research work-in-progress. Code is under active
development. Nothing here is a released artifact yet.

## License

Released under the [Apache License 2.0](LICENSE).
