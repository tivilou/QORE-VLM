"""QORE passage selector: unified interface for RAG context selection."""

import numpy as np

from qore import solve as qore_solve
from qore.qubo import build_qubo_matrix, energy, energy_decomposed
from qore.signals import normalize
from .signals_rag import passage_relevance, passage_redundancy
from .baselines import topk, mmr


def _record_qubo_diagnostics(diagnostics, a, b, x, K, lam, gamma, prefiltered,
                             n_candidates, pool_ranks):
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


def select_passages(
    query_embedding: np.ndarray,
    passage_embeddings: np.ndarray,
    K: int,
    method: str = "qore",
    relevance_scores: np.ndarray | None = None,
    redundancy_method: str = "cosine",
    lam: float = 2.0,
    gamma: float | None = None,
    delta: float = 0.0,
    complementarity_method: str | None = None,
    answer_scorer=None,
    passage_texts: list[str] | None = None,
    question: str | None = None,
    num_reads: int = 50,
    lambda_mmr: float = 0.5,
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

    Args:
        query_embedding: (d,) query vector.
        passage_embeddings: (N, d) candidate passage embeddings.
        K: Number of passages to select (context budget).
        method: Selection method. One of "qore", "vqc", "topk", "mmr".
        relevance_scores: Optional (N,) pre-computed relevance scores from the
            retriever. If None, cosine(query, passage) is used.
        redundancy_method: How to compute b_ij. "cosine" (default) or "rbf".
        lam: QUBO penalty weight (only for method="qore"/"vqc").
        gamma: QUBO redundancy weight (only for method="qore"/"vqc"). If None,
            auto-tunes to 1.0.
        delta: Complementarity weight (only for method="qore" with complementarity_method set).
            Positive delta rewards selecting complementary passage pairs.
        complementarity_method: How to compute complementarity c_ij (only for method="qore").
            None (default): no complementarity term, w = gamma * b (standard QUBO).
            "dpr": use DPR answer scorer pairwise signals (requires answer_scorer, passage_texts, question).
        answer_scorer: Instance with score_passages(question, texts) method (required when complementarity_method="dpr").
        passage_texts: List of N passage strings (required when complementarity_method="dpr").
        question: Query string (required when complementarity_method="dpr").
        num_reads: SA reads (only for method="qore"/"vqc").
        lambda_mmr: MMR trade-off (only for method="mmr").
        seed: Random seed for reproducibility.
        direct_solve_max_n: If N <= this, solve the full QUBO directly with no
            top-M prefilter. Default is 20, which ensures N=50 (typical RAG
            Top-K retrieval) goes through the relevance-first prefilter path.
            Increase only for small-N demos where global QUBO is intended.
        qore_prefilter_size: Relevance-first candidate pool size for large N
            (only for method="qore"/"vqc"). If None, defaults to max(K*3, 15).
            Smaller values (e.g., 15-20) reduce the risk of QORE selecting
            low-relevance passages. Larger values give more diversity headroom.
        vqc_encoder: Pre-trained VQCEncoder instance (only for method="vqc").
            If None, a fresh (untrained) encoder is created.
        vqc_backend: Quantum backend for VQC (only for method="vqc").

        diagnostics: Optional dict, filled in-place with what the solver
            actually optimized (method="qore" only): the normalized quality
            vector, the cosine redundancy matrix, and the decomposed QUBO
            energy of the returned solution.

            This exists because the energy is NOT reconstructable downstream:
            `a` is min-max normalized over all N candidates (so the selected
            K alone don't determine it), `b` is embedding cosine (not text
            overlap), and for N > direct_solve_max_n the QUBO is solved on a
            top-M pool rather than all N. Recomputing from a result JSON gave
            a quality term ~42x off when relevance_scores come from the answer
            scorer. Pass a dict here to get the real numbers instead.

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

    elif method == "qore":
        kwargs = {"num_reads": num_reads}
        if seed is not None:
            kwargs["seed"] = seed

        # Build pairwise interaction matrix w
        # w = gamma * b (redundancy penalty) - delta * c (complementarity reward)
        use_complementarity = complementarity_method is not None and delta != 0.0

        if N <= direct_solve_max_n:
            # PURE path (roadmap §4.3): N is small enough to solve the full QUBO
            # directly with brute-force enumeration — no prefilter, exact solution.
            # This is RAG's "clean" demonstration that global QUBO selection beats
            # greedy top-K/MMR.
            b = passage_redundancy(passage_embeddings, method=redundancy_method)

            if use_complementarity:
                if complementarity_method == "dpr":
                    if answer_scorer is None or passage_texts is None or question is None:
                        raise ValueError("complementarity_method='dpr' requires answer_scorer, passage_texts, and question")
                    from .signals_rag import passage_complementarity_dpr
                    c = passage_complementarity_dpr(question, passage_texts, answer_scorer)
                else:
                    raise ValueError(f"Unknown complementarity_method: {complementarity_method}")

                # w = gamma * b - delta * c
                gamma_eff = 1.0 if gamma is None else gamma
                w = gamma_eff * b - delta * c

                from qore.qubo import build_qubo_matrix_from_w
                from qore.solvers.brute import solve as brute_solve
                Q = build_qubo_matrix_from_w(a, w, K, lam=lam)
                x = brute_solve(Q, K)
            else:
                x = qore_solve(a, b, K, lam=lam, gamma=gamma, method="brute", **kwargs)

            if diagnostics is not None:
                _record_qubo_diagnostics(
                    diagnostics, a=a, b=b, x=x, K=K, lam=lam, gamma=gamma,
                    n_candidates=N, prefiltered=False,
                    pool_ranks=list(range(N)),   # 直接求解：池子就是全部候选
                )
            indices = np.where(x == 1)[0]
            return indices[np.argsort(a[indices])[::-1]]

        # Large N: two-stage. Pre-filter to top-M by quality, then QUBO on the
        # pool (redundancy is the differentiator within already-relevant items).
        # Smaller M (e.g., 15-20) reduces risk of low-relevance selections.
        if qore_prefilter_size is not None:
            M = min(N, qore_prefilter_size)
        else:
            M = min(N, max(K * 3, 15))
        prefilter_idx = np.argsort(a)[-M:]

        a_filtered = a[prefilter_idx]
        embeddings_filtered = passage_embeddings[prefilter_idx]
        b = passage_redundancy(embeddings_filtered, method=redundancy_method)

        if use_complementarity:
            if complementarity_method == "dpr":
                if answer_scorer is None or passage_texts is None or question is None:
                    raise ValueError("complementarity_method='dpr' requires answer_scorer, passage_texts, and question")
                from .signals_rag import passage_complementarity_dpr
                # Only compute complementarity for filtered passages
                texts_filtered = [passage_texts[i] for i in prefilter_idx]
                c = passage_complementarity_dpr(question, texts_filtered, answer_scorer)
            else:
                raise ValueError(f"Unknown complementarity_method: {complementarity_method}")

            gamma_eff = 1.0 if gamma is None else gamma
            w = gamma_eff * b - delta * c

            from qore.qubo import build_qubo_matrix_from_w
            Q = build_qubo_matrix_from_w(a_filtered, w, K, lam=lam)

            # Guard: brute solver caps at N≤20; if M>20, fall back to anneal
            M_actual = len(a_filtered)
            if M_actual <= 20:
                from qore.solvers.brute import solve as brute_solve
                x = brute_solve(Q, K)
            else:
                from qore.solvers.anneal import solve as anneal_solve
                x = anneal_solve(Q, K, **kwargs)
        else:
            # Guard: brute solver caps at N≤20; if M>20, fall back to anneal
            M_actual = len(a_filtered)
            solver_method = "brute" if M_actual <= 20 else "anneal"
            x = qore_solve(a_filtered, b, K, lam=lam, gamma=gamma, method=solver_method, **kwargs)

        if diagnostics is not None:
            # Note: the QUBO was solved on the M-item pool, not all N candidates.
            # Energy is only meaningful against a_filtered/b, so record those.
            _record_qubo_diagnostics(
                diagnostics, a=a_filtered, b=b, x=x, K=K, lam=lam, gamma=gamma,
                n_candidates=N, prefiltered=True,
                pool_ranks=prefilter_idx,
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
            f"Unknown method '{method}'. Choose from: 'qore', 'vqc', 'topk', 'mmr'."
        )


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
