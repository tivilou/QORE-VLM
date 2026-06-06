"""Brute-force exact solver for small QUBO instances (N ≤ 20)."""

import numpy as np
from itertools import combinations


def solve(Q: np.ndarray, K: int) -> np.ndarray:
    """
    Enumerate all C(N, K) subsets and return the one with minimum energy.

    Only feasible for small N (≤ 20). Used as ground-truth reference.

    Args:
        Q: (N, N) upper-triangular QUBO matrix.
        K: Number of items to select.

    Returns:
        x: (N,) binary vector of the optimal solution.

    Raises:
        ValueError: If N > 20 (too expensive to enumerate).
    """
    N = Q.shape[0]
    if N > 20:
        raise ValueError(
            f"Brute-force is only feasible for N ≤ 20, got N={N}. "
            f"C({N},{K}) = too many subsets."
        )

    best_energy = np.inf
    best_x = None

    for subset in combinations(range(N), K):
        x = np.zeros(N, dtype=np.float64)
        x[list(subset)] = 1.0
        e = float(x @ Q @ x)
        if e < best_energy:
            best_energy = e
            best_x = x.copy()

    return best_x.astype(np.int32)
