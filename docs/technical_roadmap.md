# QORE-VLM: Technical Roadmap & Method Design

> **Quantum-Optimized token REduction for Vision-Language Models**
>
> This document lays out the complete research idea — from motivation through
> formulation to experiment plan — for a paper anchored on three synergistic
> contributions: (A1) QUBO token selection, (A2) DPP / GBS theoretical grounding,
> and (B2) quantum kernel redundancy estimation.

---

## 1. Problem Statement

Modern VLMs (LLaVA, Qwen2.5-VL, etc.) encode a single high-resolution image into
N = 576–2800+ visual tokens. Attention cost is O(N²), making token reduction essential.

DUET-VLM reduces tokens in two stages:

| Stage | Where | Mechanism | Code |
|-------|-------|-----------|------|
| Stage 1 (V2V) | Vision encoder (CLIP) | Keep "dominant" tokens by self-attention score + cluster remaining into "contextual" tokens | `visionzip/` |
| Stage 2 (T2V) | LLM backbone (LLaMA) | Use salient text tokens to rank visual tokens, drop lowest layer-by-layer (PyramidDrop) | `modeling_llama_pdrop.py` |

**The gap we exploit**: both stages use **greedy, score-based heuristics** — rank
tokens by a scalar importance score, keep the top-K. This ignores **pairwise
interactions**: two tokens can both score high individually yet be mutually redundant
(e.g., two patches of sky). Greedy selection wastes budget on redundancy.

Token selection under a cardinality budget is a **combinatorial optimization** problem.
When pairwise interactions matter, it becomes quadratic — exactly the structure that
QUBO / Ising models capture, and that quantum optimizers are designed to solve.

---

## 2. Core Contribution: A1 — QUBO Token Selection

### 2.1 Formulation

Given N candidate visual tokens (after Stage 1 pre-filtering), we want to select
exactly K tokens that maximize task-relevant information while minimizing redundancy.

Define binary decision variables: `x ∈ {0,1}^N`, where `xᵢ = 1` means token i is kept.

**Objective (minimization form):**

```
E(x) = - Σᵢ aᵢ xᵢ  +  Σᵢ<ⱼ bᵢⱼ xᵢ xⱼ  +  λ (Σᵢ xᵢ - K)²
         ─────────      ──────────────────      ──────────────────
         salience       redundancy penalty      budget constraint
```

Where:
- `aᵢ` = per-token salience (higher → more important to keep)
- `bᵢⱼ` = pairwise redundancy (higher → keeping both i,j wastes budget)
- `λ` = penalty weight enforcing exactly K tokens are selected
- `K` = target token budget

### 2.2 Constructing `aᵢ` — Token Salience

We reuse DUET's existing signals (no new computation needed):

1. **Vision self-attention score** (from Stage 1 / VisionZip):
   - Already computed in `visionzip/utils.py:96` — the `raw_key_states.mean(1)` metric
   - Aggregated across last encoder layer's attention heads
   - Captures "which patches attract attention from others" (dominant tokens)

2. **Text-to-vision relevance** (from Stage 2 / PyramidDrop):
   - Cross-attention between salient text tokens and visual tokens
   - `salient_token_finder.py` extracts question words, nouns, verbs
   - Their attention to visual tokens reveals "which visual evidence supports the question"

3. **Combined salience**:
   ```
   aᵢ = α · norm(attn_scoreᵢ) + (1-α) · norm(text_relevanceᵢ)
   ```
   α is a hyperparameter (default 0.5); both terms are min-max normalized to [0,1].

### 2.3 Constructing `bᵢⱼ` — Redundancy Matrix

This is where the quantum kernel (B2) comes in. Three options for `bᵢⱼ`:

| Method | Definition | Pros | Cons |
|--------|-----------|------|------|
| Cosine similarity | `bᵢⱼ = max(0, cos(hᵢ, hⱼ))` | Fast, no params | Linear features only |
| RBF kernel | `bᵢⱼ = exp(-‖hᵢ-hⱼ‖²/2σ²)` | Captures nonlinear similarity | σ tuning needed |
| **Quantum kernel** (B2) | `bᵢⱼ = |⟨φ(hᵢ)|φ(hⱼ)⟩|²` | Exponential feature space, theoretically richer | N² circuit evals |

Where `hᵢ` is the hidden-state vector of token i (dimension d=4096 for LLaMA-7B,
reduced via PCA to manageable dimensionality for the quantum kernel).

### 2.4 Budget Constraint

The quadratic penalty `λ(Σxᵢ - K)²` expands to linear + quadratic terms and folds
directly into the QUBO matrix Q:

```
Qᵢᵢ = -aᵢ + λ(1 - 2K)
Qᵢⱼ = bᵢⱼ + 2λ         (for i ≠ j)
```

The full QUBO is: `min x^T Q x` subject to `x ∈ {0,1}^N`.

### 2.5 Scaling Strategy: Block Decomposition

N = 576 qubits is beyond NISQ. Our solution: **spatial block decomposition**.

1. VisionZip already groups tokens into spatial clusters (cluster_width parameter).
   After Stage 1, we have ~192 candidate tokens.
2. Partition these into B blocks of size n = N/B (e.g., B=8 blocks of n=24 tokens).
   Partition criterion: spatial proximity in the original image grid.
3. Allocate budget proportionally: block b gets Kᵦ = K × (salience_sum_b / total_salience).
4. Solve B independent small QUBOs, each with ~24 variables (→ 24 qubits, NISQ-feasible).
5. Optional refinement: one final global pass swapping tokens between blocks.

This gives us a **hierarchical QUBO** that's both principled and hardware-compatible.

### 2.6 Solvers (Planned Backends)

| Backend | Library | When to use |
|---------|---------|-------------|
| Brute force | numpy | n ≤ 20, ground-truth reference |
| Simulated annealing | `dwave-neal` | Classical baseline, always available |
| QAOA (p=1,2,3) | `qiskit` or `pennylane` | Gate-model quantum, scaling analysis |
| Quantum annealing | `dwave-system` | Hardware demo (if access available) |

The paper's main experiments use **simulated annealing** (reproducible, no hardware
needed) and compare against greedy top-K. QAOA results are presented as scaling
analysis / proof-of-concept for the quantum pathway.

---

## 3. Theoretical Grounding: A2 — DPP / GBS Connection

### 3.1 DPP Interpretation

A Determinantal Point Process with kernel L selects subset S with probability:

```
P(S) ∝ det(L_S)
```

where L_S is the submatrix of L indexed by S. DPPs naturally balance quality (diagonal
of L) against diversity (off-diagonal repulsion from the determinant).

**Our mapping:**
- L = diag(a) + redundancy structure → specifically, define L as a positive
  semidefinite kernel: `Lᵢⱼ = √(aᵢ) · K(hᵢ, hⱼ) · √(aⱼ)` where K is a similarity
  kernel. Then `det(L_S)` is large when tokens in S are both salient (large aᵢ) and
  diverse (low pairwise similarity → large determinant).
- Maximizing det(L_S) subject to |S|=K is the MAP inference of a k-DPP — which is
  NP-hard in general, connecting back to why we need combinatorial optimizers.

### 3.2 GBS Connection

Gaussian Boson Sampling with a symmetric matrix A produces photon-detection patterns
S with probability proportional to:

```
P(S) ∝ |Haf(A_S)|²
```

For adjacency matrices, this preferentially samples dense subgraphs. The connection
to DPPs:
- When A encodes the kernel structure, GBS samples approximate k-DPP MAP solutions.
- Literature: Arrazola et al. (2018) "Using Gaussian Boson Sampling to Find Dense
  Subgraphs"; Jahangiri et al. (2020) "Point processes with GBS".

**In QORE-VLM**: we frame token selection as finding the densest (most informative)
subgraph of size K in the token-salience graph → GBS is a natural quantum sampler for
this. Even classically simulating GBS (via hafnian computation on small matrices)
gives us a principled sampling baseline that's theoretically motivated by quantum
advantage results.

### 3.3 What This Adds to the Paper

- **Theoretical depth**: elevates the paper from "we tried QUBO on tokens" to
  "token selection is a DPP/GBS problem with known quantum-advantage connections."
- **Alternative solver**: GBS-inspired sampling as a second approach alongside QAOA,
  showing the problem has multiple quantum attack vectors.
- **Connects to photonic QC**: broadens relevance beyond gate-model / annealing.

---

## 4. Component: B2 — Quantum Kernel for Redundancy

### 4.1 Motivation

The redundancy matrix `bᵢⱼ` should capture "how much information token j adds given
we already have token i." Classical kernels (cosine, RBF) operate in fixed-dimensional
feature spaces. Quantum kernels map data into exponentially large Hilbert spaces,
potentially capturing richer similarity structure.

### 4.2 Circuit Design

For each pair (hᵢ, hⱼ):

1. **Dimensionality reduction**: PCA project the d=4096 hidden states to d'=8
   dimensions (matching available qubits).
2. **Feature map** U(h): data re-uploading circuit with L=2 layers on d' qubits.
   Each layer: `R_Y(h) → R_Z(h) → entangling CNOT ladder`.
3. **Fidelity kernel**:
   ```
   bᵢⱼ = |⟨0| U(hᵢ)† U(hⱼ) |0⟩|²
   ```
   This equals the state overlap — high when hᵢ ≈ hⱼ (redundant), low otherwise.

### 4.3 Practical Considerations

- **Cost**: N² circuit evaluations. For N=192 tokens, that's ~18K circuits.
  Batched on a simulator this takes seconds; on real hardware, minutes.
- **Classical surrogate**: we always compare against RBF kernel as the classical
  baseline for bᵢⱼ. The quantum kernel is an upgrade, not a requirement.
- **Projected quantum kernel** (Huang et al. 2021): measure individual qubits and
  use classical post-processing → reduces variance, often outperforms fidelity kernel.

---

## 5. Integration into DUET-VLM Codebase

### 5.1 Where QORE Plugs In

```
Original DUET pipeline:
  image → ViT+VisionZip → [N tokens] → greedy_topK → [K tokens] → LLM+PyramidDrop → output

QORE pipeline:
  image → ViT+VisionZip → [N tokens] → QUBO_select(a, b, K) → [K tokens] → LLM+PyramidDrop → output
                                              ↑           ↑
                                         salience a    redundancy b
                                        (from DUET)   (quantum kernel or classical)
```

### 5.2 Code Changes Required

| File | Change |
|------|--------|
| `qore/qubo.py` (new) | Build QUBO matrix from salience + redundancy |
| `qore/solvers/anneal.py` (new) | Simulated annealing solver |
| `qore/solvers/qaoa.py` (new) | QAOA solver (qiskit/pennylane) |
| `qore/solvers/brute.py` (new) | Exact solver for small N |
| `qore/kernels.py` (new) | Quantum kernel + classical baselines |
| `qore/integration.py` (new) | Hook into DUET Stage 1/2 decision points |
| `visionzip/utils.py` | Expose raw salience scores (minor mod) |
| `llava/model/modeling_llama_pdrop.py` | Replace greedy selection with QORE call |

### 5.3 Inference Flow (Pseudocode)

```python
def qore_select(hidden_states, text_salience, vision_salience, K, solver="anneal"):
    """Replace greedy top-K with QUBO-optimized selection."""
    N = hidden_states.shape[1]

    # 1. Build salience vector a
    a = alpha * normalize(vision_salience) + (1-alpha) * normalize(text_salience)

    # 2. Build redundancy matrix b
    b = compute_redundancy(hidden_states, method="rbf")  # or "quantum_kernel"

    # 3. Construct QUBO
    Q = build_qubo_matrix(a, b, K, lambda_penalty)

    # 4. Solve
    if solver == "anneal":
        solution = simulated_annealing(Q, num_reads=100)
    elif solver == "qaoa":
        solution = qaoa_solve(Q, p=2)
    elif solver == "brute":
        solution = brute_force(Q)

    # 5. Select tokens
    keep_indices = np.where(solution == 1)[0]
    return hidden_states[:, keep_indices, :]
```

---

## 6. Experiment Plan

### 6.1 Models & Benchmarks

| Model | Benchmarks | Baseline |
|-------|-----------|----------|
| LLaVA-1.5-7B | TextVQA, POPE, GQA, MMBench, ScienceQA | DUET-VLM greedy |
| Video-LLaVA-7B | MSVD-QA, MSRVTT-QA, ActivityNet-QA | DUET-VLM greedy |
| Qwen2.5-VL-7B | TextVQA, POPE | DUET-VLM greedy |

### 6.2 Ablations

| Experiment | Purpose |
|-----------|---------|
| Greedy top-K vs SA-QUBO vs QAOA | Main result: coupled optimization > greedy |
| bᵢⱼ = cosine vs RBF vs quantum kernel | Does quantum kernel improve selection? |
| Block size n = 12, 24, 48 | Scaling vs quality tradeoff |
| λ sensitivity | How tight must the budget constraint be? |
| DPP sampling vs QUBO | Alternative formulation comparison |
| K = 64, 128, 192, 256 | Token budget sweep |

### 6.3 Metrics

- **Primary**: accuracy on each benchmark (match DUET-VLM's tables)
- **Secondary**: average tokens kept, reduction %, information retention (measured
  by reconstruction error of dropped tokens from kept ones)
- **Quantum-specific**: QUBO energy gap (optimal vs greedy solution), number of
  qubits needed, circuit depth for QAOA, wall-clock time per image

### 6.4 What We Expect to Show

1. **SA-QUBO > greedy** by 0.5–2% accuracy at the same token budget (especially at
   aggressive reduction like 88%+), because it avoids redundant selections.
2. **Quantum kernel bᵢⱼ > RBF bᵢⱼ** by a smaller margin, showing richer similarity
   capture.
3. **QAOA matches SA** on small blocks, with a clear path to quantum advantage as
   hardware scales (log the approximation ratio vs circuit depth p).
4. **DPP/GBS sampling** provides a second theoretically-motivated approach that
   converges to similar solutions, validating the formulation from both optimization
   and sampling perspectives.

---

## 7. Paper Outline (Target: Quantum ML or CV+AI venue)

```
Title: QORE-VLM: Quantum-Optimized Token Reduction for Vision-Language Models

Abstract

1. Introduction
   - VLM token explosion problem
   - Limitation of greedy token selection
   - Our contribution: combinatorial optimization formulation + quantum solvers

2. Related Work
   - Token reduction in VLMs (FastV, VisionZip, PyramidDrop, DUET)
   - Quantum optimization (QAOA, quantum annealing, QUBO)
   - Quantum kernels and DPP

3. Method
   3.1 Problem formulation: token selection as QUBO
   3.2 Salience and redundancy construction
   3.3 Quantum kernel for redundancy (B2)
   3.4 Solver backends (SA, QAOA, brute force)
   3.5 Block decomposition for scalability
   3.6 Theoretical connection to DPP / GBS (A2)

4. Experiments
   4.1 Setup (models, benchmarks, baselines)
   4.2 Main results: QORE vs greedy vs random
   4.3 Ablations (kernel type, block size, budget sweep)
   4.4 Quantum scaling analysis (QAOA depth, qubit count)
   4.5 DPP/GBS sampling comparison

5. Discussion
   - When does QORE help most? (high reduction ratios, redundant scenes)
   - Limitations: runtime overhead, hardware requirements
   - Path to quantum advantage

6. Conclusion
```

---

## 8. Honest Limitations & Reviewer Defenses

| Likely critique | Our defense |
|----------------|-------------|
| "No quantum advantage on current hardware" | We claim formulation quality + quantum-inspired gain (SA-QUBO > greedy). QAOA results are forward-looking scaling analysis, clearly labeled. |
| "Runtime overhead of QUBO solving" | Block decomposition keeps each QUBO at n≤24 vars → SA solves in <1ms. Total overhead is negligible vs LLM forward pass. We report wall-clock. |
| "Why not just use ILP / branch-and-bound?" | Valid classical alternative; we compare against it. Our point is the natural QUBO structure enables quantum hardware path. |
| "Quantum kernel is expensive (N² circuits)" | We provide RBF as cheap classical default. Quantum kernel is an optional upgrade; experiments show when it helps. |
| "DPP/GBS section is theoretical only" | We implement classical DPP sampling + hafnian-based simulation. The GBS connection is theoretical motivation for the photonic QC community, not a hardware claim. |

---

## 9. Timeline & Milestones

| Phase | Deliverable | Duration |
|-------|-------------|----------|
| 1. Prototype | `qore/` module: QUBO builder + SA solver + synthetic-data demo | 1-2 weeks |
| 2. Integration | Hook into LLaVA-1.5 pipeline, run TextVQA/POPE | 2-3 weeks |
| 3. Quantum kernel | Implement B2, compare against RBF baseline | 1-2 weeks |
| 4. QAOA | Qiskit QAOA on block-decomposed QUBOs, scaling plots | 2 weeks |
| 5. DPP/GBS | Theoretical section + classical DPP sampling experiments | 1-2 weeks |
| 6. Full eval | All benchmarks, all models, ablations | 2-3 weeks |
| 7. Paper writing | Draft, figures, camera-ready | 3-4 weeks |

---

## 10. Target Venues

| Venue | Fit | Deadline (approx.) |
|-------|-----|-------------------|
| NeurIPS (QML workshop) | Quantum + ML, accepts WIP | Sep 2026 |
| AAAI 2027 | Broad AI, VLM optimization angle | Aug 2026 |
| ICLR 2027 | Strong ML theory + experiments | Sep 2026 |
| Quantum Science and Technology | Physics venue, values quantum formulation | Rolling |
| IEEE QCE (Quantum Computing and Engineering) | Applied quantum, industry-friendly | Apr 2026 (next: 2027) |

---

*Document version: 2026-06-05. Living document — will be updated as experiments proceed.*
