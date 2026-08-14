"""QORE passage selector: unified interface for RAG context selection.

REFACTORED: Now supports pluggable enhancers for clean idea composition.
"""

import numpy as np

from qore import solve as qore_solve
from qore.qubo import build_qubo_matrix, build_qubo_matrix_from_w, energy, energy_decomposed
from qore.signals import normalize
from .signals_rag import passage_relevance, passage_redundancy
from .baselines import topk, mmr, submodular, spectral_dpp


def _record_qubo_diagnostics(diagnostics, a, b, x, K, lam, gamma, prefiltered,
                             n_candidates, pool_ranks, w=None):
    """Record the QUBO the solver actually saw.

    Exists because the energy cannot be reconstructed downstream from a result
    JSON: `a` is min-max normalized over ALL N candidates (only K are dumped),
    `b` is embedding cosine (not text overlap), and for large N the QUBO is
    solved on a prefiltered pool of M < N. A diagnosis that recomputes the
    objective from the dumped K passages measures a different function — with
    answer-scorer scores the quality term came out ~42x off.
    """
    gamma_eff = 1.0 if gamma is None else float(gamma)

    # build_qubo_matrix requires 1 <= K <= N-1. qore.solve short-circuits
    # K >= N (selects everything) before building Q, so this path can legally
    # be reached with K == N — recording diagnostics must not crash there.
    if not (1 <= K <= len(a) - 1):
        diagnostics.update({
            "skipped": f"K={K} out of range for pool of {len(a)}; "
                       "QUBO was not built (degenerate selection)",
            "K": int(K),
            "pool_size": int(len(a)),
            "n_candidates": int(n_candidates),
            "prefiltered": bool(prefiltered),
        })
        return

    Q = build_qubo_matrix(a, b, K, lam=lam, gamma=gamma_eff)
    terms = energy_decomposed(x, a, b, K, lam=lam, gamma=gamma_eff)
    diagnostics.update({
        "energy": float(energy(x, Q)),
        "terms": {k: float(v) for k, v in terms.items()},
        "gamma_effective": gamma_eff,
        "lam": float(lam),
        "K": int(K),
        "n_candidates": int(n_candidates),
        "prefiltered": bool(prefiltered),
        "pool_size": int(len(a)),
        "quality_min": float(a.min()),
        "quality_max": float(a.max()),
        "redundancy_mean": float(b[np.triu_indices(len(b), k=1)].mean()) if len(b) > 1 else 0.0,
        # The pool's actual a and b, so downstream can enumerate subsets against
        # the SAME objective the solver used. Without these, any reconstruction
        # needs a stand-in for b; a score-proximity stand-in was measured to
        # agree with true cosine on only 0.5% of subsets.
        "a": [float(v) for v in a],
        "b": [[float(v) for v in row] for row in b],
        "x": [int(v) for v in np.asarray(x).reshape(-1)],
        # Original candidate index for each pool position. Required to map
        # is_gold (recorded per retrieved_rank) onto a/b, whose indices are
        # pool-local after prefiltering.
        "pool_ranks": [int(v) for v in pool_ranks],
    })

    # Record actual w matrix if using enhancers
    if w is not None:
        diagnostics["w"] = [[float(v) for v in row] for row in w]


def select_passages(
    query_embedding: np.ndarray,
    passage_embeddings: np.ndarray,
    K: int,
    method: str = "qore",
    relevance_scores: np.ndarray | None = None,
    redundancy_method: str = "cosine",
    lam: float = 2.0,
    # Legacy parameters (for backward compatibility)
    gamma: float | None = None,
    delta: float = 0.0,
    complementarity_method: str | None = None,
    # New pluggable enhancer parameters
    enhancers: list[str] | None = None,
    enhancer_configs: dict[str, dict] | None = None,
    # Other parameters
    answer_scorer=None,
    passage_texts: list[str] | None = None,
    question: str | None = None,
    passages_meta: list[dict] | None = None,
    num_reads: int = 50,
    lambda_mmr: float = 0.5,
    saturation_alpha: float = 1.0,
    lambda_submodular: float = 0.5,
    dpp_quality_scale: float = 2.0,
    dpp_jitter: float = 1e-8,
    seed: int | None = None,
    direct_solve_max_n: int = 20,
    qore_prefilter_size: int | None = None,
    vqc_encoder=None,
    vqc_backend: str = "tensorcircuit",
    diagnostics: dict | None = None,
) -> np.ndarray:
    """
    Select K passages from N candidates for inclusion in the LLM context.

    This is the main entry point for QORE-RAG. It supports four methods:
    - "qore": QUBO-optimized selection (classical signals, SA solver)
    - "vqc": Quantum-native pipeline (VQC encoder for signals + QUBO solve)
    - "topk": Greedy top-K by relevance (baseline)
    - "mmr": Maximal Marginal Relevance (greedy diversity-aware baseline)
    - "submodular": Saturating quality with pairwise redundancy penalty
    - "spectral_dpp": Quality-weighted Spectral/DPP MAP strategy

    REFACTORED: Now supports pluggable enhancers for clean idea composition.

    Args:
        query_embedding: (d,) query vector.
        passage_embeddings: (N, d) candidate passage embeddings.
        K: Number of passages to select (context budget).
        method: Selection method. One of "qore", "vqc", "topk", "mmr", "submodular", "spectral_dpp".
        relevance_scores: Optional (N,) pre-computed relevance scores from the
            retriever. If None, cosine(query, passage) is used.
        redundancy_method: How to compute b_ij. "cosine" (default) or "rbf".
        lam: QUBO penalty weight (only for method="qore"/"vqc").

        --- Legacy parameters (for backward compatibility) ---
        gamma: QUBO redundancy weight. If None, auto-tunes to 1.0.
            DEPRECATED: Use enhancers=["baseline"] with enhancer_configs={"baseline": {"gamma": x}}
        delta: Complementarity weight.
            DEPRECATED: Use enhancers=["idea6"] with enhancer_configs={"idea6": {"delta": x}}
        complementarity_method: How to compute complementarity c_ij.
            DEPRECATED: Use enhancers=["idea6"]

        --- New pluggable enhancer parameters ---
        enhancers: List of enhancer names to apply in sequence.
            Examples:
                None (default) - inferred from legacy parameters
                ["baseline"] - standard QUBO with gamma * b
                ["idea6"] - Idea 6 complementarity
                ["baseline", "idea4"] - combine baseline + Idea 4
        enhancer_configs: Config dict for each enhancer.
            Example: {"baseline": {"gamma": 0.5}, "idea6": {"delta": 0.1}}

        --- Other parameters ---
        answer_scorer: Instance with score_passages(question, texts) method.
        passage_texts: List of N passage strings.
        question: Query string.
        passages_meta: List of N passage metadata dicts (for idea4, etc.).
        num_reads: SA reads (only for method="qore"/"vqc").
        lambda_mmr: MMR trade-off (only for method="mmr").
        saturation_alpha: Concavity parameter for method="submodular".
        lambda_submodular: Pairwise redundancy penalty for method="submodular".
        dpp_quality_scale: Quality weight in the Spectral/DPP L-ensemble.
        dpp_jitter: Positive numerical regularizer for the DPP kernel.
        seed: Random seed for reproducibility.
        direct_solve_max_n: If N <= this, solve the full QUBO directly with no
            top-M prefilter. Default is 20.
        qore_prefilter_size: Relevance-first candidate pool size for large N.
        vqc_encoder: Pre-trained VQCEncoder instance (only for method="vqc").
        vqc_backend: Quantum backend for VQC (only for method="vqc").
        diagnostics: Optional dict, filled in-place with solver diagnostics.

    Returns:
        indices: (K,) integer array of selected passage indices.
    """
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    passage_embeddings = np.asarray(passage_embeddings, dtype=np.float64)
    N = len(passage_embeddings)
    K = min(K, N)

    # Compute relevance (quality signal)
    if relevance_scores is not None:
        a = normalize(np.asarray(relevance_scores, dtype=np.float64))
    else:
        raw_rel = passage_relevance(query_embedding, passage_embeddings)
        a = normalize(raw_rel)

    if method == "topk":
        return topk.select(a, K)

    elif method == "mmr":
        return mmr.select(
            query_embedding,
            passage_embeddings,
            K,
            lambda_mmr=lambda_mmr,
            relevance_scores=a,
        )

    elif method == "submodular":
        b = passage_redundancy(passage_embeddings, method=redundancy_method)
        return submodular.select(
            a,
            b,
            K,
            saturation_alpha=saturation_alpha,
            lambda_redundancy=lambda_submodular,
        )

    elif method == "spectral_dpp":
        return spectral_dpp.select(
            a,
            passage_embeddings,
            K,
            quality_scale=dpp_quality_scale,
            jitter=dpp_jitter,
        )

    elif method == "qore":
        kwargs = {"num_reads": num_reads}
        if seed is not None:
            kwargs["seed"] = seed

        # Infer enhancer configuration from legacy parameters if needed
        if enhancers is None:
            enhancers, enhancer_configs = _infer_enhancers_from_legacy(
                gamma, delta, complementarity_method
            )

        # Build enhancer pipeline
        from qore.enhancers import create_pipeline

        if N <= direct_solve_max_n:
            # PURE path: N is small enough to solve the full QUBO directly
            b = passage_redundancy(passage_embeddings, method=redundancy_method)

            # Prepare context for enhancers
            context = {
                "embeddings": passage_embeddings,
                "query_embedding": query_embedding,
                "passages": passage_texts,
                "question": question,
                "answer_scorer": answer_scorer,
                "passages_meta": passages_meta,
                "selection_K": K,
            }

            # Apply enhancers to build w matrix
            pipeline = create_pipeline(enhancers, enhancer_configs)
            if diagnostics is None:
                w = pipeline.enhance(a, b, context)
                enhancer_trace = None
            else:
                w, enhancer_trace = pipeline.enhance_with_diagnostics(a, b, context)

            # Build QUBO and solve
            Q = build_qubo_matrix_from_w(a, w, K, lam=lam)
            from qore.solvers.brute import solve as brute_solve
            x = brute_solve(Q, K)

            if diagnostics is not None:
                diagnostics["enhancers"] = pipeline.describe()
                diagnostics["enhancer_trace"] = enhancer_trace
                _record_qubo_diagnostics(
                    diagnostics, a=a, b=b, x=x, K=K, lam=lam, gamma=gamma,
                    n_candidates=N, prefiltered=False,
                    pool_ranks=list(range(N)),
                    w=w,
                )
            indices = np.where(x == 1)[0]
            return indices[np.argsort(a[indices])[::-1]]

        # Large N: two-stage with prefilter
        if qore_prefilter_size is not None:
            M = min(N, qore_prefilter_size)
        else:
            M = min(N, max(K * 3, 15))
        prefilter_idx = np.argsort(a)[-M:]

        a_filtered = a[prefilter_idx]
        embeddings_filtered = passage_embeddings[prefilter_idx]
        b = passage_redundancy(embeddings_filtered, method=redundancy_method)

        # Prepare filtered context for enhancers
        context = {
            "embeddings": embeddings_filtered,
            "query_embedding": query_embedding,
            "passages": [passage_texts[i] for i in prefilter_idx] if passage_texts else None,
            "question": question,
            "answer_scorer": answer_scorer,
            "passages_meta": [passages_meta[i] for i in prefilter_idx] if passages_meta else None,
            "selection_K": K,
        }

        # Apply enhancers to build w matrix
        pipeline = create_pipeline(enhancers, enhancer_configs)
        if diagnostics is None:
            w = pipeline.enhance(a_filtered, b, context)
            enhancer_trace = None
        else:
            w, enhancer_trace = pipeline.enhance_with_diagnostics(
                a_filtered, b, context
            )

        # Build QUBO and solve
        Q = build_qubo_matrix_from_w(a_filtered, w, K, lam=lam)

        # Guard: brute solver caps at N≤20; if M>20, fall back to anneal
        M_actual = len(a_filtered)
        if M_actual <= 20:
            from qore.solvers.brute import solve as brute_solve
            x = brute_solve(Q, K)
        else:
            from qore.solvers.anneal import solve as anneal_solve
            x = anneal_solve(Q, K, **kwargs)

        if diagnostics is not None:
            diagnostics["enhancers"] = pipeline.describe()
            diagnostics["enhancer_trace"] = enhancer_trace
            _record_qubo_diagnostics(
                diagnostics, a=a_filtered, b=b, x=x, K=K, lam=lam, gamma=gamma,
                n_candidates=N, prefiltered=True,
                pool_ranks=prefilter_idx,
                w=w,
            )

        selected_in_pool = np.where(x == 1)[0]
        indices = prefilter_idx[selected_in_pool]
        return indices[np.argsort(a[indices])[::-1]]

    elif method == "vqc":
        # Fully quantum pipeline: VQC encoder produces both signals
        from qore.vqc.scorer import vqc_select_passages

        indices = vqc_select_passages(
            query_embedding=query_embedding,
            passage_embeddings=passage_embeddings,
            K=K,
            backend=vqc_backend,
            encoder=vqc_encoder,
            seed=seed,
            direct_solve_max_n=direct_solve_max_n,
        )
        return indices

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'qore', 'vqc', 'topk', 'mmr', 'submodular', 'spectral_dpp'."
        )


def _infer_enhancers_from_legacy(
    gamma: float | None,
    delta: float,
    complementarity_method: str | None,
) -> tuple[list[str], dict[str, dict]]:
    """
    Infer enhancer configuration from legacy parameters.

    This provides backward compatibility for existing code using gamma/delta/complementarity_method.

    Args:
        gamma: Legacy gamma parameter.
        delta: Legacy delta parameter.
        complementarity_method: Legacy complementarity method.

    Returns:
        (enhancers, enhancer_configs) tuple.
    """
    # If complementarity is used, infer idea6
    if complementarity_method is not None and delta != 0.0:
        enhancers = ["idea6"]
        enhancer_configs = {
            "idea6": {
                "gamma": gamma if gamma is not None else 1.0,
                "delta": delta,
                "method": complementarity_method,
            }
        }
    else:
        # Default to baseline
        enhancers = ["baseline"]
        enhancer_configs = {
            "baseline": {
                "gamma": gamma if gamma is not None else 1.0,
            }
        }

    return enhancers, enhancer_configs


def evaluate_selection(
    selected_indices: np.ndarray,
    gold_indices: np.ndarray | set,
    passage_embeddings: np.ndarray,
) -> dict:
    """
    Evaluate a passage selection against gold-standard passages.

    Args:
        selected_indices: (K,) indices of selected passages.
        gold_indices: Indices of gold (relevant) passages.
        passage_embeddings: (N, d) all passage embeddings.

    Returns:
        Dictionary with metrics:
        - recall: fraction of gold passages in selection
        - redundancy_ratio: average pairwise cosine similarity among selected
        - diversity_score: 1 - redundancy_ratio
    """
    selected = set(int(i) for i in selected_indices)
    gold = set(int(i) for i in gold_indices)

    # Recall
    hits = len(selected & gold)
    recall = hits / len(gold) if len(gold) > 0 else 0.0

    # Redundancy ratio: avg pairwise cosine sim among selected
    sel_embeddings = passage_embeddings[list(selected_indices)]
    norms = np.linalg.norm(sel_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = sel_embeddings / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, 0.0)

    K = len(selected_indices)
    if K > 1:
        redundancy = sim.sum() / (K * (K - 1))
    else:
        redundancy = 0.0

    return {
        "recall": recall,
        "gold_hits": hits,
        "gold_total": len(gold),
        "redundancy_ratio": float(redundancy),
        "diversity_score": 1.0 - float(redundancy),
        "K": K,
    }
