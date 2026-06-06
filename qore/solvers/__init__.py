"""Unified solver interface for QORE QUBO problems."""

import numpy as np

from . import brute, anneal, greedy
from ..qubo import build_qubo_matrix


def solve(
    a: np.ndarray,
    b: np.ndarray,
    K: int,
    lam: float = 2.0,
    gamma: float | None = None,
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
        gamma: Weight for redundancy relative to quality.
            None = auto-tune (recommended). 1.0 = equal weight.
        method: Solver backend.
            - "anneal": Simulated annealing (default, recommended).
            - "brute": Exact enumeration (N ≤ 20 only).
            - "greedy": Top-K by quality (baseline, ignores b).
            - "qaoa_qk": QAOA via Qiskit.
            - "qaoa_pl": QAOA via PennyLane.
            - "qaoa_tc": QAOA via TensorCircuit.
        **kwargs: Additional arguments passed to the solver.
            - anneal: num_reads (int), seed (int|None)
            - qaoa_*: p (int, circuit depth), maxiter (int), seed (int|None)
            - brute: (none)
            - greedy: (none)

    Returns:
        x: (N,) binary vector with K ones indicating selected items.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    N = len(a)

    # Edge case: if K >= N, select everything
    if K >= N:
        return np.ones(N, dtype=np.int32)

    if method == "greedy":
        # Greedy ignores redundancy — only uses quality scores
        return greedy.solve(a, K)

    Q = build_qubo_matrix(a, b, K, lam=lam, gamma=gamma)

    if method == "brute":
        return brute.solve(Q, K)
    elif method == "anneal":
        return anneal.solve(Q, K, **kwargs)
    elif method == "qaoa_qk":
        from . import qaoa_qiskit
        return qaoa_qiskit.solve(Q, K, **kwargs)
    elif method == "qaoa_pl":
        from . import qaoa_pennylane
        return qaoa_pennylane.solve(Q, K, **kwargs)
    elif method == "qaoa_tc":
        from . import qaoa_tensorcircuit
        return qaoa_tensorcircuit.solve(Q, K, **kwargs)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            f"'anneal', 'brute', 'greedy', 'qaoa_qk', 'qaoa_pl', 'qaoa_tc'."
        )
