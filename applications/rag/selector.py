"""QORE passage selector: unified interface for RAG context selection."""

import numpy as np

from qore import solve as qore_solve
from qore.signals import normalize
from .signals_rag import passage_relevance, passage_redundancy
from .baselines import topk, mmr


def select_passages(
    query_embedding: np.ndarray,
    passage_embeddings: np.ndarray,
    K: int,
    method: str = "qore",
    relevance_scores: np.ndarray | None = None,
    redundancy_method: str = "cosine",
    lam: float = 2.0,
    num_reads: int = 50,
    lambda_mmr: float = 0.5,
    seed: int | None = None,
    direct_solve_max_n: int = 64,
    vqc_encoder=None,
    vqc_backend: str = "tensorcircuit",
) -> np.ndarray:
    """
    Select K passages from N candidates for inclusion in the LLM context.

    This is the main entry point for QORE-RAG. It supports four methods:
    - "qore": QUBO-optimized selection (classical signals, SA solver)
    - "vqc": Quantum-native pipeline (VQC encoder for signals + QUBO solve)
    - "topk": Greedy top-K by relevance (baseline)
    - "mmr": Maximal Marginal Relevance (greedy diversity-aware baseline)

    Args:
        query_embedding: (d,) query vector.
        passage_embeddings: (N, d) candidate passage embeddings.
        K: Number of passages to select (context budget).
        method: Selection method. One of "qore", "vqc", "topk", "mmr".
        relevance_scores: Optional (N,) pre-computed relevance scores from the
            retriever. If None, cosine(query, passage) is used.
        redundancy_method: How to compute b_ij. "cosine" (default) or "rbf".
        lam: QUBO penalty weight (only for method="qore"/"vqc").
        num_reads: SA reads (only for method="qore"/"vqc").
        lambda_mmr: MMR trade-off (only for method="mmr").
        seed: Random seed for reproducibility.
        direct_solve_max_n: If N <= this, solve the full QUBO directly with no
            top-M prefilter (roadmap §4.3 "pure" demonstration). Above it, fall
            back to prefilter + QUBO for tractability. Applies to "qore"/"vqc".
        vqc_encoder: Pre-trained VQCEncoder instance (only for method="vqc").
            If None, a fresh (untrained) encoder is created.
        vqc_backend: Quantum backend for VQC (only for method="vqc").

    Returns:
        indices: (K,) integer array of selected passage indices.
    """
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    passage_embeddings = np.asarray(passage_embeddings, dtype=np.float64)
    N = len(passage_embeddings)
    K = min(K, N)

    # Compute relevance (quality signal)
    if relevance_scores is not None:
        a = normalize(np.asarray(relevance_scores, dtype=np.float64))
    else:
        raw_rel = passage_relevance(query_embedding, passage_embeddings)
        a = normalize(raw_rel)

    if method == "topk":
        return topk.select(a, K)

    elif method == "mmr":
        return mmr.select(
            query_embedding,
            passage_embeddings,
            K,
            lambda_mmr=lambda_mmr,
            relevance_scores=a,
        )

    elif method == "qore":
        kwargs = {"num_reads": num_reads}
        if seed is not None:
            kwargs["seed"] = seed

        if N <= direct_solve_max_n:
            # PURE path (roadmap §4.3): N is small enough to solve the full QUBO
            # directly — no prefilter, no approximation. This is RAG's "clean"
            # demonstration that global QUBO selection beats greedy top-K/MMR.
            b = passage_redundancy(passage_embeddings, method=redundancy_method)
            x = qore_solve(a, b, K, lam=lam, method="anneal", **kwargs)
            indices = np.where(x == 1)[0]
            return indices[np.argsort(a[indices])[::-1]]

        # Large N: two-stage. Pre-filter to top-M by quality, then QUBO on the
        # pool (redundancy is the differentiator within already-relevant items).
        M = min(N, max(K * 3, 15))
        prefilter_idx = np.argsort(a)[-M:]

        a_filtered = a[prefilter_idx]
        embeddings_filtered = passage_embeddings[prefilter_idx]
        b = passage_redundancy(embeddings_filtered, method=redundancy_method)

        x = qore_solve(a_filtered, b, K, lam=lam, method="anneal", **kwargs)

        selected_in_pool = np.where(x == 1)[0]
        indices = prefilter_idx[selected_in_pool]
        return indices[np.argsort(a[indices])[::-1]]

    elif method == "vqc":
        # Fully quantum pipeline: VQC encoder produces both signals
        from qore.vqc.scorer import vqc_select_passages

        indices = vqc_select_passages(
            query_embedding=query_embedding,
            passage_embeddings=passage_embeddings,
            K=K,
            backend=vqc_backend,
            encoder=vqc_encoder,
            seed=seed,
            direct_solve_max_n=direct_solve_max_n,
        )
        return indices

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'qore', 'vqc', 'topk', 'mmr'."
        )


def evaluate_selection(
    selected_indices: np.ndarray,
    gold_indices: np.ndarray | set,
    passage_embeddings: np.ndarray,
) -> dict:
    """
    Evaluate a passage selection against gold-standard passages.

    Args:
        selected_indices: (K,) indices of selected passages.
        gold_indices: Indices of gold (relevant) passages.
        passage_embeddings: (N, d) all passage embeddings.

    Returns:
        Dictionary with metrics:
        - recall: fraction of gold passages in selection
        - redundancy_ratio: average pairwise cosine similarity among selected
        - diversity_score: 1 - redundancy_ratio
    """
    selected = set(int(i) for i in selected_indices)
    gold = set(int(i) for i in gold_indices)

    # Recall
    hits = len(selected & gold)
    recall = hits / len(gold) if len(gold) > 0 else 0.0

    # Redundancy ratio: avg pairwise cosine sim among selected
    sel_embeddings = passage_embeddings[list(selected_indices)]
    norms = np.linalg.norm(sel_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = sel_embeddings / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, 0.0)

    K = len(selected_indices)
    if K > 1:
        redundancy = sim.sum() / (K * (K - 1))
    else:
        redundancy = 0.0

    return {
        "recall": recall,
        "gold_hits": hits,
        "gold_total": len(gold),
        "redundancy_ratio": float(redundancy),
        "diversity_score": 1.0 - float(redundancy),
        "K": K,
    }
