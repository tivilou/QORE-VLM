# Phase 2 Implementation Plan: RAG Context Selection

## Goal
Build `applications/rag/` module that integrates QORE into a RAG passage selection
pipeline, demonstrating QORE-SA > MMR > top-K on passage diversity and gold recall.

## Environment Constraints
- No GPU, ~200MB free RAM on dev machine
- Verification uses synthetic data (controlled, reproducible)
- Full NQ/HotpotQA evaluation deferred to Bootrear (Phase 6)

## File Structure
```
applications/
├── __init__.py
└── rag/
    ├── __init__.py
    ├── selector.py          # QORE passage selector (main entry point)
    ├── signals_rag.py       # Build a_i (relevance) and b_ij (overlap) from embeddings
    ├── baselines/
    │   ├── __init__.py
    │   ├── topk.py          # Top-K by retriever score
    │   └── mmr.py           # Maximal Marginal Relevance
    ├── tests/
    │   ├── __init__.py
    │   ├── test_selector.py # Unit tests
    │   └── test_baselines.py # Baseline correctness
    └── demo_synthetic.py    # Synthetic RAG scenario demonstrating QORE advantage
```

## Module Designs

### selector.py
```python
def select_passages(
    query_embedding,       # (d,) query vector
    passage_embeddings,    # (N, d) candidate passages
    K,                     # budget: how many to select
    method="qore",         # "qore" | "topk" | "mmr"
    redundancy_method="cosine",  # "cosine" | "rbf"
    lam=2.0,
    num_reads=50,
    **kwargs,
) -> np.ndarray:
    """Select K passages from N candidates. Returns indices of selected passages."""
```

### signals_rag.py
```python
def passage_relevance(query_embedding, passage_embeddings) -> np.ndarray:
    """Compute a_i = cosine(query, passage_i). Shape (N,)."""

def passage_redundancy(passage_embeddings, method="cosine") -> np.ndarray:
    """Compute b_ij = pairwise similarity. Shape (N, N)."""
```

### baselines/mmr.py
```python
def select(query_embedding, passage_embeddings, K, lambda_mmr=0.5) -> np.ndarray:
    """MMR: iteratively select max(lambda*relevance - (1-lambda)*max_sim_to_selected)."""
```

### demo_synthetic.py — Controlled Scenario
Simulates a realistic RAG setting:
- 1 query
- N=100 candidate passages:
  - 5 gold passages (relevant, diverse aspects of the answer)
  - 20 near-duplicates of gold passages (high relevance, high redundancy)
  - 75 distractors (low relevance)
- K=8 selection budget
- Expected:
  - top-K: picks gold + near-duplicates (wastes budget on redundancy)
  - MMR: better diversity but greedy → misses some gold passages
  - QORE-SA: finds all 5 gold passages + 3 diverse distractors

Metrics:
- Gold recall@K (how many gold passages in selection)
- Redundancy ratio (avg pairwise sim in selection — lower = better)
- Coverage score (semantic spread of selected set)

## Verification Criteria
1. All unit tests pass
2. QORE-SA achieves higher gold recall than top-K and MMR on synthetic demo
3. QORE-SA has lower redundancy ratio than top-K
4. demo_synthetic.py prints clear comparison table
