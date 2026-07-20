"""
Unified VQC scoring interface: encode features → build QUBO → solve.

This is the high-level API that combines the VQC encoder with QUBO solving
into a single call. It replaces the classical pipeline of
"compute a_i with heuristics + compute b_ij with cosine + solve with SA"
with a fully quantum pipeline:
"encode with VQC → measure a_i + fidelity b_ij → solve with QAOA/SA"
"""

import numpy as np
from typing import Optional

from .encoder import VQCEncoder
from ..qubo import build_qubo_matrix
from ..solvers import solve as qore_solve
from ..signals import normalize


def vqc_score(
    features: np.ndarray,
    K: int,
    n_qubits: int = 6,
    n_layers: int = 2,
    backend: str = "tensorcircuit",
    solver: str = "anneal",
    lam: float = 2.0,
    gamma: float | None = None,
    encoder: Optional[VQCEncoder] = None,
    num_reads: int = 50,
    seed: Optional[int] = None,
    return_signals: bool = False,
) -> np.ndarray | tuple:
    """
    Full quantum pipeline: VQC encode → QUBO → solve.

    Uses a single VQC encoder to produce both quality (a_i) and redundancy (b_ij)
    signals, then solves the resulting QUBO.

    Args:
        features: (N, d) token/passage feature matrix.
        K: Budget — number of items to select.
        n_qubits: Number of qubits for the encoder.
        n_layers: Variational circuit depth.
        backend: Quantum backend ("tensorcircuit", "pennylane", "qiskit").
        solver: QUBO solver ("anneal", "qaoa_tc", "qaoa_pl", "qaoa_qk", "brute").
        lam: QUBO penalty weight.
        gamma: Redundancy weight (None = auto-tune).
        encoder: Pre-built VQCEncoder instance (reuse across calls).
            If None, a new encoder is created.
        num_reads: SA reads (if solver="anneal").
        seed: Random seed.
        return_signals: If True, also return the quality/redundancy signals.

    Returns:
        x: (N,) binary selection vector.
        If return_signals=True: (x, {"quality": a, "redundancy": b, "states": ...})
    """
    features = np.asarray(features, dtype=np.float64)
    N = len(features)

    # Create or reuse encoder
    if encoder is None:
        encoder = VQCEncoder(
            n_qubits=n_qubits,
            n_layers=n_layers,
            backend=backend,
            seed=seed,
        )

    # Encode: get both signals from one circuit
    signals = encoder.encode_and_measure(features)
    a = normalize(signals["quality"])
    b = signals["redundancy"]

    # Ensure b has zero diagonal and is in [0, 1]
    np.fill_diagonal(b, 0.0)
    np.clip(b, 0.0, 1.0, out=b)

    # Solve QUBO
    solver_kwargs = {}
    if "anneal" in solver:
        solver_kwargs["num_reads"] = num_reads
    if seed is not None:
        solver_kwargs["seed"] = seed

    x = qore_solve(a, b, K, lam=lam, gamma=gamma, method=solver, **solver_kwargs)

    if return_signals:
        return x, signals
    return x


def vqc_select_passages(
    query_embedding: np.ndarray,
    passage_embeddings: np.ndarray,
    K: int,
    n_qubits: int = 6,
    n_layers: int = 2,
    backend: str = "tensorcircuit",
    solver: str = "anneal",
    encoder: Optional[VQCEncoder] = None,
    seed: Optional[int] = None,
    direct_solve_max_n: int = 64,
) -> np.ndarray:
    """
    Quantum-native RAG passage selection.

    Combines classical query-passage relevance with VQC-derived quality and
    redundancy for a hybrid quantum-classical selection.

    Args:
        query_embedding: (d,) query vector.
        passage_embeddings: (N, d) passage embeddings.
        K: Number of passages to select.
        n_qubits: VQC qubits.
        n_layers: VQC depth.
        backend: Quantum backend.
        solver: QUBO solver.
        encoder: Reusable VQCEncoder.
        seed: Random seed.

    Returns:
        indices: (K,) selected passage indices.
    """
    N = len(passage_embeddings)

    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    passage_embeddings = np.asarray(passage_embeddings, dtype=np.float64)

    # Classical relevance (used for pre-filtering and/or presentation order)
    q_norm = np.linalg.norm(query_embedding)
    if q_norm > 1e-12:
        q = query_embedding / q_norm
    else:
        q = query_embedding
    p_norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
    p_norms = np.maximum(p_norms, 1e-12)
    p_normed = passage_embeddings / p_norms
    relevance = p_normed @ q

    # Small N: solve directly without pre-filtering
    if N <= direct_solve_max_n:
        x = vqc_score(
            features=passage_embeddings, K=K, n_qubits=n_qubits, n_layers=n_layers,
            backend=backend, solver=solver, encoder=encoder, seed=seed,
        )
        indices = np.where(x == 1)[0]
        return indices[np.argsort(relevance[indices])[::-1]]

    # Large N: pre-filter to top-M by relevance to keep the VQC problem tractable.
    M = min(N, max(K * 3, 15))
    prefilter_idx = np.argsort(relevance)[-M:]
    filtered_embeddings = passage_embeddings[prefilter_idx]

    x = vqc_score(
        features=filtered_embeddings, K=K, n_qubits=n_qubits, n_layers=n_layers,
        backend=backend, solver=solver, encoder=encoder, seed=seed,
    )

    # Map back to original indices
    selected_in_pool = np.where(x == 1)[0]
    indices = prefilter_idx[selected_in_pool]

    # Sort by classical relevance for presentation
    indices = indices[np.argsort(relevance[indices])[::-1]]
    return indices
