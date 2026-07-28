"""Signal construction for RAG passage selection: relevance and redundancy."""

import numpy as np
from numpy.linalg import norm


def passage_relevance(
    query_embedding: np.ndarray,
    passage_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute per-passage relevance to the query (quality signal a_i).

    Uses cosine similarity between query and each passage embedding.

    Args:
        query_embedding: (d,) query vector.
        passage_embeddings: (N, d) candidate passage vectors.

    Returns:
        a: (N,) relevance scores in [-1, 1]. Higher = more relevant.
    """
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    passage_embeddings = np.asarray(passage_embeddings, dtype=np.float64)

    q_norm = norm(query_embedding)
    if q_norm < 1e-12:
        return np.zeros(len(passage_embeddings))

    q = query_embedding / q_norm
    p_norms = norm(passage_embeddings, axis=1, keepdims=True)
    p_norms = np.maximum(p_norms, 1e-12)
    p = passage_embeddings / p_norms

    return (p @ q).astype(np.float64)


def passage_redundancy(
    passage_embeddings: np.ndarray,
    method: str = "cosine",
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute pairwise passage redundancy matrix (b_ij).

    Two passages with high b_ij are redundant — selecting both wastes budget.

    Args:
        passage_embeddings: (N, d) passage vectors.
        method: "cosine" (default) or "rbf".
        sigma: Bandwidth for RBF kernel (only used when method="rbf").

    Returns:
        b: (N, N) symmetric matrix with zero diagonal. Values in [0, 1].
    """
    passage_embeddings = np.asarray(passage_embeddings, dtype=np.float64)

    if method == "cosine":
        norms = norm(passage_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = passage_embeddings / norms
        sim = normed @ normed.T
        np.clip(sim, 0.0, 1.0, out=sim)
        np.fill_diagonal(sim, 0.0)
        return sim

    elif method == "rbf":
        sq_norms = (passage_embeddings**2).sum(axis=1)
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * passage_embeddings @ passage_embeddings.T
        np.maximum(dist_sq, 0.0, out=dist_sq)
        b = np.exp(-dist_sq / (2 * sigma**2))
        np.fill_diagonal(b, 0.0)
        return b

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'cosine' or 'rbf'.")


def retriever_scores_to_quality(
    retriever_scores: np.ndarray,
    query_embedding: np.ndarray | None = None,
    passage_embeddings: np.ndarray | None = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Combine retriever scores with optional embedding-based relevance.

    Args:
        retriever_scores: (N,) raw scores from the retrieval system (e.g., BM25, dense).
        query_embedding: Optional (d,) vector for embedding-based relevance.
        passage_embeddings: Optional (N, d) passage vectors.
        alpha: Weight for retriever scores vs embedding relevance.
            1.0 = only retriever scores, 0.0 = only embedding relevance.

    Returns:
        a: (N,) combined quality signal, min-max normalized to [0, 1].
    """
    from qore.signals import normalize

    a = normalize(np.asarray(retriever_scores, dtype=np.float64))

    if query_embedding is not None and passage_embeddings is not None and alpha < 1.0:
        embed_rel = passage_relevance(query_embedding, passage_embeddings)
        embed_rel = normalize(embed_rel)
        a = alpha * a + (1 - alpha) * embed_rel

    return normalize(a)


def passage_complementarity_dpr(
    question: str,
    passages: list[str],
    answer_scorer,
) -> np.ndarray:
    """
    Compute pairwise complementarity using DPR answer scorer.

    Complementarity measures whether two passages together provide more
    answer support than either alone. High complementarity means the pair
    covers different aspects of the answer.

    Args:
        question: The query string.
        passages: List of N passage texts.
        answer_scorer: Instance with score_passages(question, texts) method.

    Returns:
        c: (N, N) symmetric complementarity matrix with zero diagonal.
           c_ij = s(q, [p_i, p_j]) - max(s(q, p_i), s(q, p_j))
           Positive c_ij means the pair is complementary (selecting both helps).
           Negative c_ij means redundancy (one subsumes the other).
    """
    N = len(passages)
    c = np.zeros((N, N), dtype=np.float64)

    # Single-passage scores
    single_scores = answer_scorer.score_passages(question, passages)

    # Pairwise scores
    for i in range(N):
        for j in range(i + 1, N):
            # Concatenate passages (order shouldn't matter much, but use consistent order)
            pair_text = passages[i] + " " + passages[j]
            pair_score = answer_scorer.score_passages(question, [pair_text])[0]

            # Complementarity = joint score - max individual score
            c_ij = pair_score - max(single_scores[i], single_scores[j])
            c[i, j] = c_ij
            c[j, i] = c_ij  # Symmetric

    return c
