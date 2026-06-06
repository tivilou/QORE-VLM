"""Unified solver interface for QORE QUBO problems."""

import numpy as np

from . import brute, anneal, greedy
from ..qubo import build_qubo_matrix


def solve(
    a: np.ndarray,
    b: np.ndarray,
    K: int,
    lam: float = 2.0,
    method: str = "anneal",
    **kwargs,
) -> np.ndarray:
    """
    Solve a subset selection problem: pick K items maximizing quality,
    minimizing redundancy.

    This is the main entry point for QORE optimization.

    Args:
        a: (N,) quality scores.
        b: (N, N) redundancy matrix (symmetric, zero diagonal).
        K: Budget — number of items to select.
        lam: Penalty weight for the cardinality constraint.
        method: Solver backend.
            - "anneal": Simulated annealing (default, recommended).
            - "brute": Exact enumeration (N ≤ 20 only).
            - "greedy": Top-K by quality (baseline, ignores b).
        **kwargs: Additional arguments passed to the solver.
            - anneal: num_reads (int), seed (int|None)
            - brute: (none)
            - greedy: (none)

    Returns:
        x: (N,) binary vector with K ones indicating selected items.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if method == "greedy":
        # Greedy ignores redundancy — only uses quality scores
        return greedy.solve(a, K)

    Q = build_qubo_matrix(a, b, K, lam=lam)

    if method == "brute":
        return brute.solve(Q, K)
    elif method == "anneal":
        return anneal.solve(Q, K, **kwargs)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'anneal', 'brute', 'greedy'."
        )
