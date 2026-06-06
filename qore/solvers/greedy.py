"""Greedy top-K solver: selects items by quality score only (ignores redundancy)."""

import numpy as np


def solve(a: np.ndarray, K: int) -> np.ndarray:
    """
    Select the K items with highest quality scores.

    This is the baseline that QORE aims to beat. It ignores pairwise
    redundancy — two highly-scored but near-identical items will both be kept.

    Args:
        a: (N,) quality scores. Higher = more important.
        K: Number of items to select.

    Returns:
        x: (N,) binary vector with exactly K ones.
    """
    a = np.asarray(a, dtype=np.float64)
    N = len(a)

    if K >= N:
        return np.ones(N, dtype=np.int32)
    if K <= 0:
        return np.zeros(N, dtype=np.int32)

    top_indices = np.argsort(a)[-K:]
    x = np.zeros(N, dtype=np.int32)
    x[top_indices] = 1
    return x
