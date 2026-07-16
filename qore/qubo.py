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
        # Heuristic: balance quality and redundancy contributions.
        #
        # OLD approach: use top-K mean redundancy.
        # PROBLEM: When top-K contains clusters of near-duplicates (high relevance
        # + high intra-cluster redundancy), the mean redundancy is inflated,
        # causing gamma to be too low, which under-penalizes redundancy.
        #
        # NEW approach: use top-M candidates (M = 3*K) and take the MEDIAN
        # redundancy. This is more robust to outliers (near-duplicate clusters)
        # while still focusing on competitive candidates.

        quality_scale = a.mean() if N > 0 else 1.0

        # Use top-M candidates (M = 3*K) for a larger, more robust sample
        M = min(3 * K, N)
        top_m_idx = np.argsort(a)[-M:]
        b_topm = b[np.ix_(top_m_idx, top_m_idx)]
        b_topm_vals = b_topm[np.triu_indices(M, k=1)]

        if len(b_topm_vals) > 0 and np.median(b_topm_vals) > 1e-6:
            # Use median redundancy (robust to outliers like near-duplicate clusters)
            redundancy_median = np.median(b_topm_vals)
            # Expected redundancy contribution per selected item
            redundancy_scale = redundancy_median * (K - 1) / 2
            gamma = quality_scale / max(redundancy_scale, 1e-12)
            # Allow wider range for gamma (up to 50 instead of 10)
            gamma = np.clip(gamma, 0.1, 50.0)
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
