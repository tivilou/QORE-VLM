<div align="center">
  <h1>QORE-VLM</h1>
  <p><b>Q</b>uantum-<b>O</b>ptimized token <b>RE</b>duction for Vision-Language Models</p>
  <p><i>(read as "Q-core")</i></p>

  <a href="https://github.com/tivilou/QORE-VLM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/status-research%20WIP-orange" alt="status">
</div>

---

QORE-VLM is a research project that brings **quantum optimization** to visual-token
reduction in Vision-Language Models (VLMs). It builds on the dual-stage token
reduction framework [DUET-VLM](https://github.com/AMD-AGI/DUET-VLM) and replaces the
heuristic token-selection step with a principled combinatorial optimizer that can be
solved on quantum (or quantum-inspired) hardware.

## Motivation

Modern VLMs emit thousands of visual tokens per high-resolution image, and attention
cost grows quadratically with sequence length. Dual-stage reduction frameworks such as
DUET-VLM cut this cost by (1) merging redundant tokens inside the vision encoder and
(2) pruning visual tokens layer-by-layer inside the language model. Both stages rely on
**greedy, score-based heuristics** to decide which tokens to keep.

Token selection — choosing the subset of `K` visual tokens (out of `N`) that preserves
the most task-relevant information under a budget — is fundamentally a **combinatorial
optimization** problem. Greedy selection ignores interactions between tokens (redundancy,
complementarity). QORE-VLM reformulates the keep/drop decision as a **QUBO** (Quadratic
Unconstrained Binary Optimization) problem and solves it with quantum and quantum-inspired
optimizers, aiming for selections that better trade off coverage against budget.

## Core Idea

```
                 ┌─────────────────────────────────────────────┐
   image ─▶ ViT ─┤ Stage 1: V2V merge (encoder)                 │
                 │   token importance + pairwise redundancy      │
                 │            │                                  │
                 │            ▼                                  │
                 │   ┌───────────────────────┐                  │
                 │   │  QUBO token selection  │ ◀── QORE module  │
                 │   │  (QAOA / annealing /   │                  │
                 │   │   VQE / simulated)     │                  │
                 │   └───────────────────────┘                  │
                 │            │                                  │
   text ────────▶│ Stage 2: T2V prune (LLM, text-guided)        │
                 └─────────────────────────────────────────────┘
                              │
                              ▼
                       reduced token set ─▶ LLM decoder
```

The keep/drop vector `x ∈ {0,1}^N` is chosen to minimize an energy
`E(x) = -Σ aᵢ xᵢ + Σ bᵢⱼ xᵢ xⱼ + λ(Σ xᵢ - K)²`, where `aᵢ` is per-token salience
(vision self-attention and text-to-vision relevance, reused from DUET's two stages),
`bᵢⱼ` penalizes keeping mutually redundant tokens, and the last term enforces the
budget `K`. This maps directly onto a QUBO / Ising model.

## Why Quantum

- **Coupling-aware selection.** The `bᵢⱼ` redundancy terms make the objective
  non-separable; greedy top-K ignores them. QUBO solvers optimize the full coupled
  objective.
- **Hardware path.** The same QUBO runs unchanged on a classical simulated-annealing
  baseline, a quantum-inspired solver (e.g. digital annealer / tensor methods), and
  gate-based QAOA / quantum annealing — so the method degrades gracefully and is
  reproducible without quantum hardware.
- **Small, well-scoped subproblems.** Selection happens per-image / per-layer over a
  bounded candidate pool, keeping qubit counts in a near-term-feasible range.

## Planned Backends

| Backend | Library | Role |
|---|---|---|
| Simulated annealing | `dwave-neal` / `dimod` | Classical baseline, always available |
| QAOA (gate model) | `qiskit` / `pennylane` | Variational quantum optimizer |
| Quantum annealing | `dwave-system` | Hardware sampler (optional, needs access) |
| Brute force | numpy | Exact reference for tiny `N` (sanity check) |

## Status

This repository is an early-stage research work-in-progress. The README captures the
intended design; code is being ported and extended from DUET-VLM. Nothing here is a
released artifact yet.

Planned layout:

```
QORE-VLM/
├── qore/                # quantum token-selection module (QUBO build + solvers)
│   ├── qubo.py          # build E(x) from salience + redundancy
│   ├── solvers/         # annealing / QAOA / brute-force backends
│   └── integration.py   # hooks into DUET Stage 1 / Stage 2
├── llava/ videollava/ qwen2_5_vl/   # VLM backbones (from DUET-VLM)
├── visionzip/           # Stage 1 token merging (from DUET-VLM)
├── scripts/             # training + evaluation
└── experiments/         # configs and result logs for the paper
```

## Acknowledgement

QORE-VLM builds directly on [DUET-VLM](https://github.com/AMD-AGI/DUET-VLM) and, through
it, on [LLaVA](https://github.com/haotian-liu/LLaVA),
[Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA),
[VisionZip](https://github.com/dvlab-research/VisionZip),
[PyramidDrop](https://github.com/Cooperx521/PyramidDrop), and
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). The quantum-optimization layer is the
new contribution of this project.

## License

Released under the [Apache License 2.0](LICENSE).
