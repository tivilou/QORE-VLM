import numpy as np

from applications.rag.q_duet_selector import select_passages


def test_q_duet_preserves_two_stage_cardinality_and_roles():
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(16, 8))
    query = rng.normal(size=8)
    diagnostics = {}

    selected = select_passages(
        query,
        embeddings,
        5,
        relevance_scores=np.linspace(0.1, 1.0, 16),
        retrieval_scores=np.linspace(1.0, 0.1, 16),
        stage1_budget=10,
        dominant_budget=3,
        stage1_solver="brute",
        stage2_solver="brute",
        diagnostics=diagnostics,
    )

    assert selected.shape == (5,)
    assert len(set(selected.tolist())) == 5
    assert len(diagnostics["dominant_indices"]) == 3
    assert len(diagnostics["contextual_indices"]) == 7
    assert len(diagnostics["stage1_indices"]) == 10
    assert len(diagnostics["selected_indices"]) == 5


def test_q_duet_does_not_accept_oracle_fields():
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(10, 6))
    query = rng.normal(size=6)
    selected = select_passages(
        query,
        embeddings,
        4,
        relevance_scores=np.arange(10, dtype=float),
        retrieval_scores=np.arange(10, dtype=float)[::-1],
        stage1_budget=7,
        dominant_budget=2,
        stage1_solver="brute",
        stage2_solver="brute",
    )
    assert len(selected) == 4
