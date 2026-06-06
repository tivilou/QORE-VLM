"""Top-K baseline: select passages by retriever relevance score only."""

import numpy as np


def select(
    relevance_scores: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Select the K passages with highest relevance scores.

    This is the simplest baseline. It ignores inter-passage redundancy.

    Args:
        relevance_scores: (N,) relevance to query. Higher = more relevant.
        K: Number of passages to select.

    Returns:
        indices: (K,) indices of selected passages, sorted by score (descending).
    """
    relevance_scores = np.asarray(relevance_scores, dtype=np.float64)
    N = len(relevance_scores)
    K = min(K, N)

    top_indices = np.argsort(relevance_scores)[::-1][:K]
    return top_indices
