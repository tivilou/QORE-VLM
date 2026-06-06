"""QUBO matrix construction from quality and redundancy signals."""

import numpy as np


def build_qubo_matrix(
    a: np.ndarray,
    b: np.ndarray,
    K: int,
    lam: float = 2.0,
    gamma: float | None = None,
) -> np.ndarray:
    """
    Build a QUBO matrix Q such that argmin x^T Q x gives the optimal K-subset.

    The objective being encoded:
        E(x) = -sum_i a_i x_i + gamma * sum_{i<j} b_ij x_i x_j + lam*(sum_i x_i - K)^2

    Expanding the penalty term and combining:
        Q_ii = -a_i + lam*(1 - 2K)
        Q_ij = gamma * b_ij + 2*lam        (for i < j)

    Args:
        a: (N,) quality scores. Higher means more important to keep.
        b: (N, N) symmetric redundancy matrix with zero diagonal.
            Higher b_ij means items i and j are more redundant.
        K: Budget — number of items to select.
        lam: Penalty weight for the cardinality constraint.
            Should be large enough to enforce |S|=K but not so large
            that it drowns out the quality/redundancy signals.
            Rule of thumb: lam >= max(a) + max(b).
        gamma: Weight for the redundancy term relative to quality.
            If None, auto-tuned so that the expected quality contribution
            and redundancy contribution are balanced for a typical K-selection.
            Explicitly set to 1.0 to disable auto-tuning.

    Returns:
        Q: (N, N) upper-triangular QUBO matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    N = len(a)

    if b.shape != (N, N):
        raise ValueError(f"b must be (N, N)={(N, N)}, got {b.shape}")
    if K < 1 or K > N - 1:
        raise ValueError(f"K must be in [1, N-1], got K={K}, N={N}")

    # Auto-tune gamma if not provided
    if gamma is None:
        # Heuristic: balance quality and redundancy contributions
        # for the items most likely to be selected (top-K by quality).
        # Look at redundancy AMONG top-K candidates specifically,
        # since those are the items competing for selection.
        top_k_idx = np.argsort(a)[-K:]
        quality_scale = a[top_k_idx].mean() if K > 0 else 1.0

        b_topk = b[np.ix_(top_k_idx, top_k_idx)]
        b_topk_vals = b_topk[np.triu_indices(K, k=1)]
        if len(b_topk_vals) > 0 and b_topk_vals.mean() > 1e-6:
            # Expected redundancy per selected item: avg_b * (K-1)/2
            redundancy_scale = b_topk_vals.mean() * (K - 1) / 2
            gamma = quality_scale / max(redundancy_scale, 1e-12)
            gamma = np.clip(gamma, 0.01, 10.0)
        else:
            gamma = 1.0

    Q = np.zeros((N, N), dtype=np.float64)

    # Diagonal terms
    Q[np.diag_indices(N)] = -a + lam * (1 - 2 * K)

    # Off-diagonal terms (upper triangle only)
    upper_mask = np.triu_indices(N, k=1)
    Q[upper_mask] = gamma * b[upper_mask] + 2 * lam

    return Q


def energy(x: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute the QUBO energy for a binary solution vector.

    Args:
        x: (N,) binary vector {0, 1}^N
        Q: (N, N) upper-triangular QUBO matrix

    Returns:
        Scalar energy value x^T Q x.
    """
    x = np.asarray(x, dtype=np.float64)
    return float(x @ Q @ x)


def energy_decomposed(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    K: int,
    lam: float = 2.0,
    gamma: float = 1.0,
) -> dict:
    """
    Compute the energy broken into its three components for analysis.

    The QUBO matrix drops a constant offset -lam*K^2 (doesn't affect optimization).
    We include it here so that 'total' matches energy(x, Q) exactly.

    Returns dict with keys: 'quality', 'redundancy', 'penalty', 'constant', 'total'.
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    quality = -float(a @ x)
    redundancy = gamma * float(0.5 * x @ b @ x)  # b is symmetric; sum_{i<j} = 0.5 * x^T b x
    penalty = lam * (float(x.sum()) - K) ** 2
    constant = -lam * K**2  # absorbed into Q diagonal but missing from x^T Q x
    total = quality + redundancy + penalty + constant

    return {
        "quality": quality,
        "redundancy": redundancy,
        "penalty": penalty,
        "constant": constant,
        "total": total,
    }
