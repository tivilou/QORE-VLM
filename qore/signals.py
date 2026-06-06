"""Signal construction utilities: quality normalization and redundancy matrices."""

import numpy as np
from numpy.linalg import norm


def normalize(a: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a quality vector to [0, 1].

    If all values are equal, returns uniform 0.5.
    """
    a = np.asarray(a, dtype=np.float64)
    a_min, a_max = a.min(), a.max()
    if a_max - a_min < 1e-12:
        return np.full_like(a, 0.5)
    return (a - a_min) / (a_max - a_min)


def cosine_redundancy(features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix, clipped to [0, 1].

    Negative similarities are clipped to 0 (items pointing in opposite
    directions are not considered redundant — they carry complementary info).

    Args:
        features: (N, d) feature matrix.

    Returns:
        b: (N, N) symmetric matrix with zero diagonal. b_ij in [0, 1].
    """
    features = np.asarray(features, dtype=np.float64)
    norms = norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    normed = features / norms

    sim = normed @ normed.T
    np.clip(sim, 0.0, 1.0, out=sim)
    np.fill_diagonal(sim, 0.0)
    return sim


def rbf_redundancy(features: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute pairwise RBF (Gaussian) kernel similarity matrix.

    b_ij = exp(-||h_i - h_j||^2 / (2 * sigma^2))

    Args:
        features: (N, d) feature matrix.
        sigma: Bandwidth parameter. Smaller sigma = faster decay = less redundancy
            assigned to distant pairs.

    Returns:
        b: (N, N) symmetric matrix with zero diagonal. b_ij in (0, 1).
    """
    features = np.asarray(features, dtype=np.float64)
    # Compute squared pairwise distances efficiently
    sq_norms = (features**2).sum(axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * features @ features.T
    np.maximum(dist_sq, 0.0, out=dist_sq)  # numerical safety

    b = np.exp(-dist_sq / (2 * sigma**2))
    np.fill_diagonal(b, 0.0)
    return b


def combined_quality(
    *signals: np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    Combine multiple quality signals into one via weighted average.

    Each signal is normalized to [0,1] before combining.

    Args:
        *signals: Variable number of (N,) quality arrays.
        weights: Mixing weights (must sum to 1). If None, uniform weights.

    Returns:
        a: (N,) combined quality vector in [0, 1].
    """
    if not signals:
        raise ValueError("At least one signal required")
    n_signals = len(signals)
    if weights is None:
        weights = [1.0 / n_signals] * n_signals
    if len(weights) != n_signals:
        raise ValueError(f"Got {n_signals} signals but {len(weights)} weights")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1, got {sum(weights)}")

    normed = [normalize(np.asarray(s, dtype=np.float64)) for s in signals]
    combined = sum(w * s for w, s in zip(weights, normed))
    return combined
