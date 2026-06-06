"""Maximal Marginal Relevance (MMR) baseline for diverse passage selection."""

import numpy as np
from numpy.linalg import norm


def select(
    query_embedding: np.ndarray,
    passage_embeddings: np.ndarray,
    K: int,
    lambda_mmr: float = 0.5,
    relevance_scores: np.ndarray | None = None,
) -> np.ndarray:
    """
    Select K passages using Maximal Marginal Relevance.

    MMR iteratively selects the passage that maximizes:
        score = lambda * relevance(q, p) - (1-lambda) * max_sim(p, selected)

    This balances relevance to the query against diversity from already-selected
    passages. It is a greedy sequential algorithm — once a passage is selected,
    it is never reconsidered.

    Args:
        query_embedding: (d,) query vector.
        passage_embeddings: (N, d) passage vectors.
        K: Number of passages to select.
        lambda_mmr: Trade-off parameter in [0, 1].
            1.0 = pure relevance (equivalent to top-K).
            0.0 = pure diversity.
            0.5 = balanced (default).
        relevance_scores: Optional (N,) pre-computed relevance scores.
            If None, uses cosine(query, passage) as relevance.

    Returns:
        indices: (K,) indices of selected passages in selection order.
    """
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    passage_embeddings = np.asarray(passage_embeddings, dtype=np.float64)
    N = len(passage_embeddings)
    K = min(K, N)

    # Compute relevance scores
    if relevance_scores is not None:
        rel = np.asarray(relevance_scores, dtype=np.float64)
    else:
        q_norm = norm(query_embedding)
        q = query_embedding / max(q_norm, 1e-12)
        p_norms = norm(passage_embeddings, axis=1, keepdims=True)
        p_norms = np.maximum(p_norms, 1e-12)
        p_normed = passage_embeddings / p_norms
        rel = p_normed @ q

    # Normalize passage embeddings for similarity computation
    p_norms = norm(passage_embeddings, axis=1, keepdims=True)
    p_norms = np.maximum(p_norms, 1e-12)
    p_normed = passage_embeddings / p_norms

    # Precompute full pairwise similarity matrix
    sim_matrix = p_normed @ p_normed.T
    np.clip(sim_matrix, 0.0, 1.0, out=sim_matrix)

    # Greedy MMR selection
    selected = []
    candidates = set(range(N))

    for _ in range(K):
        best_idx = None
        best_score = -np.inf

        for idx in candidates:
            relevance_term = lambda_mmr * rel[idx]

            if selected:
                max_sim_to_selected = max(sim_matrix[idx, s] for s in selected)
                diversity_term = (1 - lambda_mmr) * max_sim_to_selected
            else:
                diversity_term = 0.0

            score = relevance_term - diversity_term

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        candidates.remove(best_idx)

    return np.array(selected, dtype=np.int64)
