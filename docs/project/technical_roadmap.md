# QORE: Technical Roadmap & Method Design

> **Quantum-Optimized Context Reduction for Large Language Models**
>
> A unified QUBO framework for LLM context budget allocation, demonstrated on
> KV-Cache eviction and RAG passage selection.

---

## 1. Problem Statement

Large Language Models operate under finite context budgets at multiple granularities:

| Scenario | What's being selected | Budget source | Current SOTA |
|----------|----------------------|---------------|--------------|
| KV-Cache management | Token-level KV entries | Physical memory / latency | H2O, SnapKV, PyramidKV |
| RAG context curation | Passage-level chunks | Context window length | MMR, reranking, DPP-greedy |

In both cases, the system must answer: **from N candidates, which K should the model
attend to?** Current methods rank candidates by a scalar score and keep top-K — a
greedy heuristic that ignores pairwise interactions between candidates.

**The gap**: two candidates can both score high yet be mutually redundant (encoding the
same information). Greedy selection wastes budget. The optimal selection requires
solving a **coupled subset selection** problem — choosing items that are individually
strong AND collectively diverse.

This is a combinatorial optimization problem with quadratic structure: exactly a QUBO.

---

## 2. Unified QUBO Formulation

### 2.1 Decision Variables

Binary vector `x ∈ {0,1}^N`: `xᵢ = 1` means candidate i is retained.

### 2.2 Objective Function

```
E(x) = - Σᵢ aᵢ xᵢ  +  Σᵢ<ⱼ bᵢⱼ xᵢ xⱼ  +  λ (Σᵢ xᵢ - K)²
```

**Term 1 — Quality** (`-Σ aᵢ xᵢ`):
Rewards selecting high-quality candidates. Minimizing E → maximize total quality.

**Term 2 — Redundancy** (`Σ bᵢⱼ xᵢ xⱼ`):
Penalizes co-selecting redundant pairs. Only active when both i and j are kept.

**Term 3 — Budget** (`λ(Σxᵢ - K)²`):
Enforces exactly K items selected. Expands into linear + quadratic terms.

### 2.3 QUBO Matrix Construction

The objective rewrites as `min x^T Q x` with:

```
Qᵢᵢ = -aᵢ + λ(1 - 2K)
Qᵢⱼ = bᵢⱼ/2 + λ          (i ≠ j, upper triangle)
```

This Q matrix is the input to all solvers (SA, QAOA, quantum annealing).

### 2.4 Ising Mapping

For gate-model quantum solvers, convert to Ising form via `xᵢ = (1 - zᵢ)/2`:

```
H = Σᵢ hᵢ σᵢᶻ + Σᵢ<ⱼ Jᵢⱼ σᵢᶻ σⱼᶻ + const
```

Standard mapping; handled automatically by `qiskit.optimization` or `dimod`.

---

## 3. Application A: KV-Cache Eviction

### 3.1 Background

During autoregressive generation with sequence length L, the KV cache stores L
key-value pairs per layer per head. For long sequences (L > 4096), cache memory
becomes the dominant bottleneck. Eviction policies decide which entries to discard
when the cache is full.

**Current methods and their limitations:**

| Method | Selection strategy | Limitation |
|--------|-------------------|------------|
| H2O (NeurIPS'23) | Keep "heavy hitters" by cumulative attention | Greedy; ignores redundancy between kept entries |
| SnapKV (2024) | Windowed attention pattern voting | Greedy; no pairwise coupling |
| PyramidKV (2024) | Layer-wise budget allocation + attention rank | Greedy per layer |
| ScissorHands (2024) | Importance by retrieval frequency | Greedy; single score per entry |

**All are greedy top-K on a scalar score.** None consider that two high-scoring
entries might encode nearly identical information.

### 3.2 QORE Instantiation

**Quality signal `aᵢ`:**

Option 1 — H2O-style cumulative attention:
```python
aᵢ = Σ_{t > i} attention(t, i)   # total attention received from later tokens
```

Option 2 — SnapKV-style windowed voting:
```python
aᵢ = Σ_{t ∈ window} attention(t, i)   # attention from recent window only
```

Option 3 — Hybrid (our default):
```python
aᵢ = β · cumulative_attn(i) + (1-β) · recent_window_attn(i)
```

**Redundancy signal `bᵢⱼ`:**

Option 1 — Key cosine similarity (fast, default):
```python
bᵢⱼ = max(0, cos(kᵢ, kⱼ))   # kᵢ = key vector of entry i
```

Option 2 — RBF kernel:
```python
bᵢⱼ = exp(-‖kᵢ - kⱼ‖² / 2σ²)
```

Option 3 — Quantum kernel (B2 component, see Section 5):
```python
bᵢⱼ = |⟨0| U(kᵢ)† U(kⱼ) |0⟩|²
```

**Budget K:** Target cache size (e.g., keep 1024 out of 4096 entries).

**Trigger frequency:** Solve QUBO every T=64 or T=128 generation steps.
Between triggers, use a lightweight fallback (e.g., don't evict, or greedy evict
the single lowest-score entry). This amortizes the solve cost.

### 3.3 Block Decomposition for Scale

A single attention head at sequence length 4096 has N=4096 candidates — too large
for direct QUBO. Decomposition strategy:

**Option A — Per-head solve (recommended):**
Each attention head is an independent QUBO. With 32 heads, we solve 32 parallel
sub-problems, each of size N/num_layers_sharing.

**Option B — Positional windowing:**
Partition KV entries into positional blocks of size n=32–64. Allocate budget
proportionally to each block's total quality. Solve B independent QUBOs.

**Option C — Hybrid (head × position):**
Each (head, position_block) is a tiny QUBO with n=16–32 variables.
Maximally parallelizable; each sub-QUBO needs only 16–32 qubits.

### 3.4 Baselines for Comparison

| Baseline | Implementation |
|----------|---------------|
| Full cache (no eviction) | Upper bound — memory-unlimited |
| Random eviction | Lower bound |
| H2O | Reimplemented from paper |
| SnapKV | Reimplemented from paper |
| PyramidKV | Reimplemented from paper |
| Greedy top-K (our `aᵢ` only) | Ablation: quality-only, no redundancy |
| QORE-SA | Our method with simulated annealing |
| QORE-QAOA | Our method with QAOA |

### 3.5 Benchmarks

| Benchmark | Task | Why |
|-----------|------|-----|
| LongBench | Multi-task long-context | Standard for KV-cache papers |
| RULER | Synthetic retrieval in long context | Controlled difficulty |
| Needle-in-a-Haystack | Single-fact retrieval | Stress test for eviction |
| ∞Bench | Very long sequences (100K+) | Extreme compression regime |

### 3.6 Metrics

- Accuracy / F1 on each benchmark (primary)
- Cache size K / reduction ratio
- Perplexity on held-out long text (PG-19, GovReport)
- Wall-clock latency per generation step (must show overhead is acceptable)
- Peak memory usage

---

## 4. Application B: RAG Context Selection

### 4.1 Background

Retrieval-Augmented Generation retrieves top-N passages (typically N=50–200) from
a corpus, then must select K (typically 5–15) to fit in the LLM's context window.

**Current methods and their limitations:**

| Method | Strategy | Limitation |
|--------|----------|------------|
| Top-K by retriever score | Rank by embedding similarity, take top-K | Ignores inter-passage redundancy |
| MMR (Maximal Marginal Relevance) | Greedy: iteratively pick max(relevance - λ·max_sim_to_selected) | Greedy approximation; order-dependent |
| DPP sampling | Sample from det(L_S) distribution | Approximate MAP; randomized |
| Reranker + top-K | Cross-encoder reranks, then top-K | Still greedy on scalar scores |

**MMR is the closest competitor**: it does consider redundancy, but greedily. Once
a passage is selected, it's never reconsidered. QORE finds the globally optimal
subset.

### 4.2 QORE Instantiation

**Quality signal `aᵢ`:**

```python
aᵢ = retriever_score(passageᵢ, query)   # e.g., cosine(embed(passage), embed(query))
```

Optionally enhanced by a cross-encoder reranker score.

**Redundancy signal `bᵢⱼ`:**

Option 1 — Embedding cosine (fast, default):
```python
bᵢⱼ = max(0, cos(embed(passageᵢ), embed(passageⱼ)))
```

Option 2 — Lexical overlap:
```python
bᵢⱼ = jaccard(ngrams(passageᵢ), ngrams(passageⱼ))
```

Option 3 — Hybrid:
```python
bᵢⱼ = γ · cosine_sim + (1-γ) · jaccard_overlap
```

Option 4 — Quantum kernel (see Section 5)

**Budget K:** Number of passages that fit in context (depends on passage length
and model's context window).

### 4.3 Scale Considerations

N = 50–200 is small enough to solve directly without decomposition. This makes
RAG the "clean" demonstration of QORE: no approximations, no blocking, pure
QUBO optimality. The contrast with KV-Cache (which needs decomposition) shows
the framework's adaptability.

For very large retrieval pools (N > 500), a two-stage approach:
1. Pre-filter to top-200 by retriever score (cheap)
2. QUBO select K from 200 (our method)

### 4.4 Baselines for Comparison

| Baseline | Implementation |
|----------|---------------|
| Top-K retriever | Just take highest-scoring K passages |
| MMR | λ-tuned Maximal Marginal Relevance |
| DPP-greedy | Greedy MAP inference on L-ensemble DPP |
| DPP-sampling | Sample from k-DPP, take highest-det sample |
| Greedy top-K (our `aᵢ` only) | Ablation: quality-only |
| QORE-SA | Our method with simulated annealing |
| QORE-QAOA | Our method with QAOA |

### 4.5 Benchmarks

| Benchmark | Task | Why |
|-----------|------|-----|
| Natural Questions (open) | Single-hop factoid QA | Standard RAG benchmark |
| HotpotQA | Multi-hop reasoning | Needs diverse evidence passages |
| MultiHop-RAG | Multi-hop with distractors | Tests redundancy handling |
| MS MARCO | Passage ranking | Large-scale retrieval |
| ASQA | Ambiguous questions | Needs diverse perspectives |

### 4.6 Metrics

- Answer accuracy / F1 (primary)
- Recall@K of gold passages in selected set
- Context utilization: fraction of selected passages that contribute to answer
- Redundancy ratio: average pairwise similarity in selected set (lower = better)
- Selection time (must be negligible vs LLM generation time)

---

## 5. Quantum Kernel for Redundancy (Component B2)

### 5.1 Motivation

The redundancy matrix `bᵢⱼ` determines the quality of QORE's selection. Classical
kernels (cosine, RBF) operate in finite-dimensional feature spaces. Quantum kernels
map data into exponentially large Hilbert spaces, potentially capturing richer
similarity structure that classical kernels miss.

### 5.2 Circuit Design

For each pair of items (hᵢ, hⱼ) with feature vectors h:

1. **Dimensionality reduction**: PCA project from high-d (4096 for KV keys, 768 for
   passage embeddings) to d' = 6–10 dimensions.
2. **Feature map** U(h): data re-uploading circuit with L=2 layers on d' qubits.
   Each layer applies `R_Y(h_k)` rotations followed by entangling CNOT ladder.
3. **Fidelity kernel**:
   ```
   bᵢⱼ = |⟨0| U(hᵢ)† U(hⱼ) |0⟩|²
   ```
4. **Projected quantum kernel** (alternative): measure each qubit, use classical
   post-processing of measurement statistics. Often lower variance.

### 5.3 Practical Cost

- KV-Cache: within a block of n=32 entries → n(n-1)/2 = 496 circuit evaluations.
  Batched on simulator: milliseconds.
- RAG: N=100 passages → ~5000 circuit evaluations. Still fast on simulator.

### 5.4 Role in Paper

Quantum kernel is an **optional upgrade** to the redundancy signal. The paper's main
results use classical cosine similarity (fast, reproducible). Quantum kernel results
are presented as an ablation showing when richer similarity captures help.

---

## 6. Theoretical Grounding: DPP & GBS (Component A2)

### 6.1 DPP Connection

A Determinantal Point Process with L-ensemble kernel L selects subset S with:
```
P(S) ∝ det(L_S)
```

**Our mapping**: define L as `Lᵢⱼ = √(aᵢ) · K(hᵢ,hⱼ) · √(aⱼ)` where K is a
positive-definite kernel. Then `det(L_S)` is maximized when S contains items that
are both high-quality (large `aᵢ`) and diverse (large determinant → low pairwise
correlation).

k-DPP MAP inference (finding the size-K subset maximizing `det(L_S)`) is NP-hard
in general — motivating our QUBO approach as an efficient optimization strategy.

### 6.2 GBS Connection

Gaussian Boson Sampling produces photon-detection patterns S with probability
proportional to `|Haf(A_S)|²`. For appropriately constructed A:
- GBS preferentially samples dense, diverse subgraphs
- This connects to DPP sampling via hafnian ↔ permanent relationships
- Literature: Arrazola et al. (2018), Jahangiri et al. (2020)

### 6.3 Role in Paper

The DPP/GBS section provides:
- **Theoretical motivation**: why subset selection is hard (NP-hard k-DPP MAP)
- **Alternative perspective**: QUBO optimization ≈ approximate k-DPP MAP
- **Quantum advantage narrative**: GBS gives a known quantum advantage pathway
  for sampling from this exact problem structure
- **Experimental addition**: classical DPP sampling as another baseline, showing
  QORE-SA matches or exceeds DPP-sampling quality

---

## 7. Experiment Design

### 7.1 Models

| Model | Parameters | Use case |
|-------|-----------|----------|
| LLaMA-3-8B-Instruct | 8B | Primary model for both applications |
| Mistral-7B-Instruct | 7B | Secondary (generalization check) |
| LLaMA-3-70B-Instruct | 70B | Scale test (KV-cache pressure is higher) |

### 7.2 Experiment Matrix

| Experiment | Application | Purpose |
|-----------|-------------|---------|
| E1: QORE-SA vs greedy baselines | Both | Main result |
| E2: QORE-QAOA vs QORE-SA | Both | Quantum solver comparison |
| E3: Quantum kernel vs cosine vs RBF | Both | Redundancy signal ablation |
| E4: Block size sweep | KV-Cache | n = 16, 32, 48, 64 |
| E5: Budget K sweep | Both | K/N = 0.1, 0.25, 0.5, 0.75 |
| E6: λ sensitivity | Both | Constraint weight tuning |
| E7: Trigger frequency T | KV-Cache | T = 32, 64, 128, 256 |
| E8: DPP sampling comparison | Both | Alternative formulation |
| E9: Runtime overhead | Both | Wall-clock and memory |
| E10: Scaling analysis | Both | QAOA depth p vs approximation ratio |

### 7.3 Key Hypotheses

1. **QORE-SA > all greedy baselines** at the same budget K, especially at aggressive
   compression (K/N < 0.25), because it avoids redundant selections.
2. **Gains are larger when redundancy is high** (repetitive text, similar passages,
   homogeneous KV patterns) — controlled experiments with synthetic redundancy.
3. **QAOA matches SA quality** on small blocks (n ≤ 32) at depth p ≥ 2.
4. **Quantum kernel improves selection** when items have complex, nonlinear
   similarity structure (measured by ablation).
5. **Overhead is negligible**: QUBO solve time << LLM forward pass time.

---

## 8. Paper Structure

```
Title: QORE: Quantum-Optimized Context Reduction for Large Language Models

Abstract (250 words)

1. Introduction
   - LLM context budget problem (memory + window limits)
   - Greedy selection ignores coupling → suboptimal
   - Our contribution: unified QUBO framework + quantum solvers + two applications

2. Related Work
   2.1 KV-Cache compression (H2O, SnapKV, PyramidKV, ScissorHands)
   2.2 RAG passage selection (MMR, DPP, reranking)
   2.3 Quantum optimization (QAOA, quantum annealing, QUBO applications)
   2.4 Quantum kernels and DPP/GBS

3. Method: QORE Framework
   3.1 Unified QUBO formulation
   3.2 Quality and redundancy signal construction
   3.3 Block decomposition for scalability
   3.4 Solver stack (SA, QAOA, quantum kernel)
   3.5 Theoretical connection to DPP / GBS

4. Application A: KV-Cache Eviction
   4.1 Integration with transformer inference
   4.2 Signal construction from attention patterns
   4.3 Amortized solving strategy

5. Application B: RAG Context Selection
   5.1 Integration with retrieval pipeline
   5.2 Signal construction from embeddings
   5.3 Comparison with MMR

6. Experiments
   6.1 Setup (models, benchmarks, baselines, hardware)
   6.2 KV-Cache results
   6.3 RAG results
   6.4 Ablations (shared: kernel, budget, solver comparison)
   6.5 Quantum scaling analysis
   6.6 Runtime and overhead analysis

7. Discussion
   - When does QORE help most?
   - Limitations and future work
   - Path toward quantum advantage

8. Conclusion

Appendix:
   A. Full benchmark tables
   B. QUBO construction details and proofs
   C. Quantum circuit diagrams
   D. Hyperparameter sensitivity
```

---

## 9. Reviewer Defense Strategy

| Anticipated critique | Defense |
|---------------------|---------|
| "No quantum speedup on current hardware" | Main claim is formulation quality: coupled optimization > greedy (proven with classical SA). QAOA is forward-looking scaling analysis, clearly labeled as such. |
| "Overhead of QUBO solving negates savings" | Block decomposition → each sub-QUBO has n≤32 vars → SA solves in <0.1ms. Total overhead < 1% of LLM forward pass. Reported with wall-clock numbers. |
| "Why not ILP / branch-and-bound?" | We compare against ILP as a classical exact solver. QUBO is preferred because (1) maps to quantum hardware, (2) annealing-based solvers scale better than ILP for dense Q matrices. |
| "Quantum kernel is expensive" | Cosine similarity is our default. Quantum kernel is an optional upgrade tested in ablation. We show exactly when it helps and when it doesn't. |
| "DPP/GBS section is purely theoretical" | We implement classical k-DPP sampling and compare empirically. GBS is theoretical motivation connecting to known quantum advantage results. |
| "Two applications make the paper scattered" | Unified formulation is the contribution. Two instantiations prove generality. Each instantiation has full experimental treatment. |
| "SA already gives good solutions, why quantum?" | Exactly — SA validates the QUBO formulation works. QAOA/hardware results show the path forward. The contribution is the formulation + proof it beats greedy, not claiming quantum speedup today. |

---

## 10. Contrast with Existing Work

### Why this is NOT "just MMR for KV-cache" or "just DPP for RAG"

| Aspect | MMR | DPP | QORE |
|--------|-----|-----|------|
| Optimization | Greedy sequential | Approximate MAP or sampling | Global QUBO (exact or near-optimal) |
| Pairwise handling | One-at-a-time marginal | Full determinant (but intractable MAP) | Full quadratic objective (tractable via SA/QAOA) |
| Tunability | Single λ tradeoff | Kernel choice only | Separate α (quality blend), kernel, λ (budget), solver |
| Quantum path | None | GBS sampling (theoretical) | QAOA + quantum annealing + quantum kernel (concrete) |
| Unified across tasks | No — each task has own heuristic | Possible but not demonstrated | **Explicitly demonstrated on two tasks** |

---

## 11. Timeline & Milestones

| Phase | Deliverable | Duration |
|-------|-------------|----------|
| 1. Core framework | `qore/` module: QUBO builder + SA solver + unit tests | 1–2 weeks |
| 2. KV-Cache integration | Hook into HuggingFace LLaMA, run LongBench/RULER | 2–3 weeks |
| 3. RAG integration | Post-retrieval selector, run NQ/HotpotQA | 2 weeks |
| 4. Quantum kernel | Pennylane kernel, ablation experiments | 1–2 weeks |
| 5. QAOA | Qiskit QAOA on block-decomposed QUBOs, scaling plots | 2 weeks |
| 6. DPP baseline | Classical k-DPP implementation + experiments | 1 week |
| 7. Full evaluation | All benchmarks, all ablations, figures | 2–3 weeks |
| 8. Paper writing | Draft → internal review → camera-ready | 3–4 weeks |

Total: ~4 months from start to submission-ready manuscript.

---

## 12. Target Venues

| Venue | Fit | Typical deadline |
|-------|-----|-----------------|
| NeurIPS 2027 | Main conference — ML + optimization | May 2027 |
| ICML 2027 | Main conference — ML theory + applications | Jan 2027 |
| ICLR 2027 | Main conference — representation learning | Sep 2026 |
| AAAI 2027 | Broad AI — applied quantum + NLP | Aug 2026 |
| ACL 2027 | NLP venue — RAG + efficiency | Jan 2027 |
| IEEE QCE 2027 | Quantum computing — applied quantum | ~Apr 2027 |
| NeurIPS QML Workshop | Workshop — accepts WIP | Sep 2026 |

---

*Document version: 2026-06-05. Living document — updated as research progresses.*
