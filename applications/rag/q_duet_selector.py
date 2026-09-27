"""DUET-inspired two-stage quantum-ready passage selector.

The selector adapts the DUET-VLM decomposition to fixed Top-50 -> Top-5 RAG:

1. Dominant selection uses retrieval structure and passage centrality.
2. Residual selection uses question-conditioned answer support plus coverage.
3. A second QUBO selects the final K passages from the mixed intermediate set.

This module is selector-only. It never consumes Silver labels, generated
answers, or evaluator outputs. The QUBO objective is solver-agnostic and can
be sent to simulated annealing, exact brute force, or an available QAOA
backend through the existing qore solver modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qore.qubo import build_qubo_matrix_from_w, energy
from qore.signals import normalize
from applications.rag.signals_rag import passage_redundancy, passage_relevance


def _as_scores(values: np.ndarray | None, fallback: np.ndarray, n: int) -> np.ndarray:
    if values is None:
        raw = np.asarray(fallback, dtype=np.float64)
    else:
        raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (n,):
        raise ValueError(f"score vector must have shape {(n,)}, got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("score vector must contain finite values")
    return normalize(raw)


def _cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normed = values / np.maximum(norms, 1.0e-12)
    similarity = normed @ normed.T
    np.clip(similarity, 0.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def _solve_qubo(
    quality: np.ndarray,
    interactions: np.ndarray,
    k: int,
    *,
    solver: str,
    lam: float,
    gamma: float,
    num_reads: int,
    seed: int,
    qaoa_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve one fixed-cardinality QUBO and return a traceable result."""

    quality = np.asarray(quality, dtype=np.float64)
    interactions = np.asarray(interactions, dtype=np.float64)
    n = len(quality)
    if not 1 <= k < n:
        raise ValueError(f"QUBO budget must satisfy 1 <= k < n; got k={k}, n={n}")
    matrix = build_qubo_matrix_from_w(quality, gamma * interactions, k, lam=lam)
    requested = solver.lower()
    resolved = requested
    kwargs: dict[str, Any] = {}
    if requested == "auto":
        resolved = "brute" if n <= 20 else "anneal"
    if resolved == "brute":
        from qore.solvers.brute import solve
        vector = solve(matrix, k)
    elif resolved == "anneal":
        from qore.solvers.anneal import solve
        vector = solve(matrix, k, num_reads=num_reads, seed=seed)
    elif resolved in {"qaoa_qk", "qaoa_pl", "qaoa_tc"}:
        from qore.solvers import solve
        kwargs.update(qaoa_kwargs or {})
        vector = solve(
            quality,
            interactions,
            k,
            lam=lam,
            gamma=gamma,
            method=resolved,
            **kwargs,
        )
    else:
        raise ValueError(
            "unknown Q-DUET solver; choose auto, brute, anneal, "
            "qaoa_qk, qaoa_pl, or qaoa_tc"
        )
    vector = np.asarray(vector, dtype=np.int32).reshape(-1)
    if vector.shape != (n,) or int(vector.sum()) != k:
        raise ValueError(f"solver returned invalid cardinality {vector.sum()} for k={k}")
    return vector, {
        "solver_requested": requested,
        "solver_resolved": resolved,
        "budget": int(k),
        "pool_size": int(n),
        "energy": float(energy(vector, matrix)),
        "quality": [round(float(value), 8) for value in quality],
    }


def select_passages(
    query_embedding: np.ndarray,
    passage_embeddings: np.ndarray,
    K: int,
    *,
    relevance_scores: np.ndarray | None = None,
    retrieval_scores: np.ndarray | None = None,
    stage1_budget: int = 12,
    dominant_budget: int = 3,
    stage1_solver: str = "auto",
    stage2_solver: str = "auto",
    stage1_lam: float = 2.0,
    stage1_gamma: float = 0.35,
    stage2_lam: float = 2.0,
    stage2_gamma: float = 0.50,
    cross_complementarity: float = 0.08,
    num_reads: int = 100,
    seed: int = 42,
    qaoa_kwargs: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    """Select K passages through a two-stage DUET-inspired QUBO.

    ``relevance_scores`` are question-conditioned answer-support scores. The
    optional ``retrieval_scores`` are used only in the first, passage-side
    dominant view. Silver evidence and generated answers are intentionally not
    accepted as inputs.
    """

    embeddings = np.asarray(passage_embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("passage_embeddings must be a non-empty 2-D array")
    n = len(embeddings)
    if K < 1 or K > n:
        raise ValueError(f"K must be in [1, {n}], got {K}")
    if stage1_budget < K or stage1_budget >= n:
        raise ValueError("stage1_budget must satisfy K <= stage1_budget < candidate count")
    if not 1 <= dominant_budget < stage1_budget:
        raise ValueError("dominant_budget must be smaller than stage1_budget")

    query = np.asarray(query_embedding, dtype=np.float64)
    if query.shape != (embeddings.shape[1],):
        raise ValueError("query_embedding dimensionality does not match passages")
    query_relevance = passage_relevance(query, embeddings)
    answer = _as_scores(relevance_scores, query_relevance, n)
    retrieval = _as_scores(retrieval_scores, query_relevance, n)
    similarity = _cosine_similarity(embeddings)
    redundancy = passage_redundancy(embeddings, method="cosine")

    # Passage-side view: centrality is deliberately separate from answer
    # support, echoing DUET's dominant-vs-residual decomposition.
    centrality = similarity.sum(axis=1) / max(n - 1, 1)
    centrality = normalize(centrality)
    dominant_quality = 0.75 * retrieval + 0.25 * centrality
    dominant_vector, dominant_trace = _solve_qubo(
        dominant_quality,
        redundancy,
        dominant_budget,
        solver=stage1_solver,
        lam=stage1_lam,
        gamma=stage1_gamma,
        num_reads=num_reads,
        seed=seed,
        qaoa_kwargs=qaoa_kwargs,
    )
    dominant_indices = np.flatnonzero(dominant_vector)
    residual_indices = np.asarray(
        [index for index in range(n) if index not in set(dominant_indices)],
        dtype=np.int64,
    )
    residual_budget = stage1_budget - dominant_budget

    # Residual view: select representatives that have answer support while
    # covering regions not already represented by the dominant subset.
    if len(dominant_indices):
        dominant_similarity = similarity[:, dominant_indices]
        coverage = 1.0 - dominant_similarity.max(axis=1)
    else:
        coverage = np.ones(n, dtype=np.float64)
    coverage = normalize(coverage)
    residual_quality = normalize(0.65 * answer + 0.20 * retrieval + 0.15 * coverage)
    residual_quality_pool = residual_quality[residual_indices]
    residual_redundancy = redundancy[np.ix_(residual_indices, residual_indices)]
    residual_vector, residual_trace = _solve_qubo(
        residual_quality_pool,
        residual_redundancy,
        residual_budget,
        solver=stage1_solver,
        lam=stage1_lam,
        gamma=stage1_gamma,
        num_reads=num_reads,
        seed=seed + 1,
        qaoa_kwargs=qaoa_kwargs,
    )
    contextual_indices = residual_indices[np.flatnonzero(residual_vector)]
    stage1_indices = np.concatenate([dominant_indices, contextual_indices])
    if len(stage1_indices) != stage1_budget:
        raise AssertionError("stage1 cardinality invariant failed")

    # Text-conditioned final view. Cross-role reward is only applied when a
    # dominant and contextual passage are dissimilar, encouraging complementary
    # evidence without changing the fixed K budget.
    stage1_coverage = coverage[stage1_indices]
    final_quality = normalize(
        0.75 * answer[stage1_indices]
        + 0.15 * retrieval[stage1_indices]
        + 0.10 * stage1_coverage
    )
    final_redundancy = redundancy[np.ix_(stage1_indices, stage1_indices)].copy()
    role_dominant = np.isin(stage1_indices, dominant_indices)
    for left in range(stage1_budget):
        for right in range(left + 1, stage1_budget):
            if bool(role_dominant[left]) != bool(role_dominant[right]):
                complement = 1.0 - final_redundancy[left, right]
                final_redundancy[left, right] -= cross_complementarity * complement
                final_redundancy[right, left] = final_redundancy[left, right]
    final_vector, final_trace = _solve_qubo(
        final_quality,
        final_redundancy,
        K,
        solver=stage2_solver,
        lam=stage2_lam,
        gamma=stage2_gamma,
        num_reads=num_reads,
        seed=seed + 2,
        qaoa_kwargs=qaoa_kwargs,
    )
    selected = stage1_indices[np.flatnonzero(final_vector)]
    selected = selected[np.argsort(final_quality[final_vector == 1])[::-1]]

    if diagnostics is not None:
        diagnostics.update({
            "selector_id": "q_duet_rag",
            "version": "0.1",
            "stage1_budget": int(stage1_budget),
            "dominant_budget": int(dominant_budget),
            "selected_count": int(K),
            "dominant_indices": [int(value) for value in dominant_indices],
            "contextual_indices": [int(value) for value in contextual_indices],
            "stage1_indices": [int(value) for value in stage1_indices],
            "selected_indices": [int(value) for value in selected],
            "stage1": {
                "dominant": dominant_trace,
                "residual": residual_trace,
            },
            "stage2": final_trace,
            "parameters": {
                "stage1_gamma": float(stage1_gamma),
                "stage2_gamma": float(stage2_gamma),
                "cross_complementarity": float(cross_complementarity),
                "seed": int(seed),
            },
        })
    return selected.astype(np.int64)


__all__ = ["select_passages"]
