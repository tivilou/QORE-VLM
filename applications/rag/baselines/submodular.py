"""Saturating submodular passage selection.

The objective is a concave quality utility minus a pairwise redundancy cost:

    F(S) = sum(u_i for i in S) - lambda * sum(b_ij for i<j, i,j in S)

where ``u_i = 1 - exp(-alpha * quality_i)``.  Its greedy marginal gain is
cheap to update and avoids the pairwise Answer Scorer calls used by Mobius.
"""

from __future__ import annotations

import numpy as np


def select(
    quality_scores: np.ndarray,
    redundancy: np.ndarray,
    K: int,
    *,
    saturation_alpha: float = 1.0,
    lambda_redundancy: float = 0.5,
) -> np.ndarray:
    """Greedily select exactly ``min(K, N)`` passages.

    Ties are resolved by the lowest candidate index.  ``quality_scores`` and
    ``redundancy`` are expected to be normalized to ``[0, 1]``; the selector
    validates this contract so compact replays cannot silently use a different
    scale than formal evaluation.
    """
    quality = np.asarray(quality_scores, dtype=np.float64)
    pairwise = np.asarray(redundancy, dtype=np.float64)
    if quality.ndim != 1:
        raise ValueError("quality_scores must be one-dimensional")
    if pairwise.shape != (len(quality), len(quality)):
        raise ValueError("redundancy must have shape (N, N)")
    if not np.all(np.isfinite(quality)) or not np.all(np.isfinite(pairwise)):
        raise ValueError("quality_scores and redundancy must be finite")
    if np.any(quality < -1e-12) or np.any(quality > 1.0 + 1e-12):
        raise ValueError("quality_scores must be normalized to [0, 1]")
    if np.any(pairwise < -1e-12) or np.any(pairwise > 1.0 + 1e-12):
        raise ValueError("redundancy must be normalized to [0, 1]")
    if not np.allclose(pairwise, pairwise.T, atol=1e-10):
        raise ValueError("redundancy must be symmetric")
    if not np.allclose(np.diag(pairwise), 0.0, atol=1e-10):
        raise ValueError("redundancy diagonal must be zero")
    if saturation_alpha < 0.0 or not np.isfinite(saturation_alpha):
        raise ValueError("saturation_alpha must be finite and non-negative")
    if lambda_redundancy < 0.0 or not np.isfinite(lambda_redundancy):
        raise ValueError("lambda_redundancy must be finite and non-negative")
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K must be a non-negative integer")

    N = len(quality)
    target = min(int(K), N)
    if target == 0:
        return np.empty(0, dtype=np.int64)

    # alpha=0 is the linear-quality limit; it is useful as an explicit ablation.
    if saturation_alpha == 0.0:
        utility = quality.copy()
    else:
        utility = -np.expm1(-saturation_alpha * quality)

    selected = np.empty(target, dtype=np.int64)
    available = np.ones(N, dtype=bool)
    redundancy_cost = np.zeros(N, dtype=np.float64)
    for step in range(target):
        marginal = utility - lambda_redundancy * redundancy_cost
        marginal[~available] = -np.inf
        # argmax scans in ascending index order, giving deterministic ties.
        chosen = int(np.argmax(marginal))
        selected[step] = chosen
        available[chosen] = False
        redundancy_cost += pairwise[:, chosen]

    return selected
