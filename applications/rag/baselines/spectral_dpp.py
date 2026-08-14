"""Query-conditioned Spectral/DPP passage selection.

The selector builds an L-ensemble kernel

    L = diag(exp(beta * quality / 2)) S diag(exp(beta * quality / 2))

where ``S`` is the PSD passage-embedding Gram matrix. Greedy MAP selection
maximizes the determinant of the selected principal submatrix, balancing the
query-conditioned quality signal against diversity without Answer Scorer pair
calls or QUBO composition.
"""

from __future__ import annotations

import numpy as np


PLUGIN_SPEC = {
    "id": "spectral_dpp",
    "version": "1.0.0",
    "type": "terminal_strategy",
    "insertion_point": "rag.selector.selection_strategy",
    "hypothesis": "A quality-weighted PSD DPP kernel improves context diversity without sacrificing answer utility.",
    "input_schema": "quality_scores[N] in [0,1], passage_embeddings[N,D], K",
    "output_schema": "int64 indices[min(K,N)]",
    "requires": [],
    "conflicts": ["qore", "vqc", "topk", "mmr", "submodular"],
    "deterministic": True,
    "resource_estimate": "O(N*K^4) greedy MAP with O(N^2) kernel memory",
    "failure_signatures": ["non-finite input", "non-PSD kernel after repair", "invalid K"],
}


def _validate_quality(quality_scores: np.ndarray) -> np.ndarray:
    quality = np.asarray(quality_scores, dtype=np.float64)
    if quality.ndim != 1:
        raise ValueError("quality_scores must be one-dimensional")
    if not np.all(np.isfinite(quality)):
        raise ValueError("quality_scores must be finite")
    if np.any(quality < -1e-12) or np.any(quality > 1.0 + 1e-12):
        raise ValueError("quality_scores must be normalized to [0, 1]")
    return np.clip(quality, 0.0, 1.0)


def _validate_kernel(similarity: np.ndarray, N: int) -> np.ndarray:
    kernel = np.asarray(similarity, dtype=np.float64)
    if kernel.shape != (N, N):
        raise ValueError("similarity must have shape (N, N)")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("similarity must be finite")
    if not np.allclose(kernel, kernel.T, atol=1e-10):
        raise ValueError("similarity must be symmetric")
    # Eigendecomposition gives a deterministic PSD projection and removes tiny
    # negative eigenvalues caused by floating-point arithmetic or compact replay.
    eigenvalues, eigenvectors = np.linalg.eigh((kernel + kernel.T) / 2.0)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (repaired + repaired.T) / 2.0


def _embedding_kernel(passage_embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(passage_embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("passage_embeddings must be two-dimensional")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("passage_embeddings must be finite")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-12)
    return _validate_kernel(normed @ normed.T, len(embeddings))


def _greedy_map(
    quality: np.ndarray,
    similarity: np.ndarray,
    K: int,
    quality_scale: float,
    jitter: float,
) -> np.ndarray:
    N = len(quality)
    target = min(int(K), N)
    if target == 0:
        return np.empty(0, dtype=np.int64)
    if quality_scale < 0.0 or not np.isfinite(quality_scale):
        raise ValueError("quality_scale must be finite and non-negative")
    if jitter <= 0.0 or not np.isfinite(jitter):
        raise ValueError("jitter must be finite and positive")

    # A common rescaling of all L entries does not change fixed-cardinality
    # MAP selection, so subtract the largest log-weight before exponentiating.
    # This keeps very large exploratory quality scales finite.
    log_weights = 0.5 * quality_scale * quality
    weights = np.exp(log_weights - np.max(log_weights))
    kernel = np.asarray(similarity, dtype=np.float64) * np.outer(weights, weights)
    kernel = kernel + jitter * np.eye(N, dtype=np.float64)

    selected: list[int] = []
    available = np.ones(N, dtype=bool)
    for _ in range(target):
        best_index = -1
        best_logdet = -np.inf
        for candidate in np.flatnonzero(available):
            subset = selected + [int(candidate)]
            sign, logdet = np.linalg.slogdet(kernel[np.ix_(subset, subset)])
            if sign <= 0.0 or not np.isfinite(logdet):
                continue
            # Explicit tolerance makes ties stable across BLAS implementations.
            if logdet > best_logdet + 1e-12:
                best_logdet = float(logdet)
                best_index = int(candidate)
        if best_index < 0:
            raise np.linalg.LinAlgError("DPP kernel has no positive selected principal minor")
        selected.append(best_index)
        available[best_index] = False

    return np.asarray(selected, dtype=np.int64)


def select(
    quality_scores: np.ndarray,
    passage_embeddings: np.ndarray,
    K: int,
    *,
    quality_scale: float = 2.0,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Select exactly ``min(K, N)`` passages using greedy DPP MAP."""
    quality = _validate_quality(quality_scores)
    embeddings = np.asarray(passage_embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) != len(quality):
        raise ValueError("passage_embeddings must have shape (N, D)")
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K must be a non-negative integer")
    return _greedy_map(
        quality,
        _embedding_kernel(embeddings),
        K,
        quality_scale,
        jitter,
    )


def select_from_similarity(
    quality_scores: np.ndarray,
    similarity: np.ndarray,
    K: int,
    *,
    quality_scale: float = 2.0,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Replay helper using a passage-free similarity/Gram matrix."""
    quality = _validate_quality(quality_scores)
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K must be a non-negative integer")
    kernel = _validate_kernel(similarity, len(quality))
    return _greedy_map(quality, kernel, K, quality_scale, jitter)
