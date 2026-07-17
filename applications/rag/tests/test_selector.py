"""Unit tests for RAG selector and baselines."""

import numpy as np
import pytest

from applications.rag.selector import select_passages, evaluate_selection
from applications.rag.signals_rag import passage_relevance, passage_redundancy
from applications.rag.baselines import topk, mmr


class TestSignals:
    """Test signal construction."""

    def test_relevance_shape(self):
        q = np.random.randn(64)
        passages = np.random.randn(20, 64)
        a = passage_relevance(q, passages)
        assert a.shape == (20,)

    def test_relevance_range(self):
        """Cosine similarity is in [-1, 1]."""
        q = np.random.randn(64)
        passages = np.random.randn(50, 64)
        a = passage_relevance(q, passages)
        assert a.min() >= -1.0 - 1e-10
        assert a.max() <= 1.0 + 1e-10

    def test_relevance_identical(self):
        """Identical vectors should have similarity 1."""
        q = np.random.randn(64)
        passages = np.stack([q, q, np.random.randn(64)])
        a = passage_relevance(q, passages)
        assert np.isclose(a[0], 1.0, atol=1e-10)
        assert np.isclose(a[1], 1.0, atol=1e-10)

    def test_redundancy_shape(self):
        passages = np.random.randn(20, 64)
        b = passage_redundancy(passages)
        assert b.shape == (20, 20)

    def test_redundancy_symmetric_zero_diag(self):
        passages = np.random.randn(15, 64)
        b = passage_redundancy(passages)
        assert np.allclose(b, b.T)
        assert np.allclose(np.diag(b), 0.0)

    def test_redundancy_methods(self):
        passages = np.random.randn(10, 32)
        b_cos = passage_redundancy(passages, method="cosine")
        b_rbf = passage_redundancy(passages, method="rbf", sigma=1.0)
        # Both should be symmetric with zero diagonal
        assert np.allclose(b_cos, b_cos.T)
        assert np.allclose(b_rbf, b_rbf.T)
        assert np.allclose(np.diag(b_cos), 0)
        assert np.allclose(np.diag(b_rbf), 0)


class TestTopK:
    """Test top-K baseline."""

    def test_selects_highest(self):
        scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        indices = topk.select(scores, K=2)
        assert set(indices) == {1, 3}

    def test_respects_K(self):
        scores = np.random.rand(50)
        indices = topk.select(scores, K=10)
        assert len(indices) == 10


class TestMMR:
    """Test MMR baseline."""

    def test_output_size(self):
        q = np.random.randn(64)
        passages = np.random.randn(30, 64)
        indices = mmr.select(q, passages, K=5)
        assert len(indices) == 5

    def test_no_duplicates(self):
        q = np.random.randn(64)
        passages = np.random.randn(20, 64)
        indices = mmr.select(q, passages, K=8)
        assert len(set(indices)) == 8

    def test_lambda_1_equals_topk(self):
        """lambda=1 means pure relevance → should match top-K."""
        q = np.random.randn(64)
        passages = np.random.randn(20, 64)
        indices_mmr = mmr.select(q, passages, K=5, lambda_mmr=1.0)
        # With lambda=1.0, MMR reduces to top-K by relevance
        rel = passage_relevance(q, passages)
        indices_topk = topk.select(rel, K=5)
        assert set(indices_mmr) == set(indices_topk)


class TestQORESelector:
    """Test the unified QORE selector."""

    @pytest.fixture
    def simple_scenario(self):
        """Small scenario with clear structure."""
        rng = np.random.default_rng(42)
        d = 32
        q = rng.standard_normal(d)

        # 5 relevant passages (close to query)
        relevant = q + 0.1 * rng.standard_normal((5, d))
        # 15 distractors (random)
        distractors = rng.standard_normal((15, d))
        passages = np.vstack([relevant, distractors])

        return {"query": q, "passages": passages, "gold": np.arange(5)}

    def test_qore_returns_K(self, simple_scenario):
        indices = select_passages(
            simple_scenario["query"],
            simple_scenario["passages"],
            K=5,
            method="qore",
            num_reads=30,
            seed=42,
        )
        assert len(indices) == 5

    def test_topk_returns_K(self, simple_scenario):
        indices = select_passages(
            simple_scenario["query"],
            simple_scenario["passages"],
            K=5,
            method="topk",
        )
        assert len(indices) == 5

    def test_mmr_returns_K(self, simple_scenario):
        indices = select_passages(
            simple_scenario["query"],
            simple_scenario["passages"],
            K=5,
            method="mmr",
        )
        assert len(indices) == 5

    def test_all_methods_valid_indices(self, simple_scenario):
        N = len(simple_scenario["passages"])
        for method in ["qore", "topk", "mmr"]:
            kwargs = {"num_reads": 30, "seed": 42} if method == "qore" else {}
            indices = select_passages(
                simple_scenario["query"],
                simple_scenario["passages"],
                K=5,
                method=method,
                **kwargs,
            )
            assert all(0 <= i < N for i in indices)


class TestEvaluateSelection:
    """Test evaluation metrics."""

    def test_perfect_recall(self):
        passages = np.random.randn(20, 32)
        gold = np.array([0, 1, 2])
        selected = np.array([0, 1, 2, 5, 10])
        metrics = evaluate_selection(selected, gold, passages)
        assert metrics["recall"] == 1.0
        assert metrics["gold_hits"] == 3

    def test_zero_recall(self):
        passages = np.random.randn(20, 32)
        gold = np.array([0, 1, 2])
        selected = np.array([5, 6, 7, 8, 9])
        metrics = evaluate_selection(selected, gold, passages)
        assert metrics["recall"] == 0.0

    def test_redundancy_identical_passages(self):
        """Identical passages should have maximum redundancy."""
        base = np.random.randn(32)
        passages = np.tile(base, (10, 1))
        selected = np.array([0, 1, 2, 3, 4])
        metrics = evaluate_selection(selected, set(), passages)
        assert metrics["redundancy_ratio"] > 0.99

    def test_redundancy_diverse_passages(self):
        """Orthogonal passages should have low redundancy."""
        passages = np.eye(10, 32)  # orthogonal rows
        selected = np.array([0, 1, 2, 3, 4])
        metrics = evaluate_selection(selected, set(), passages)
        assert metrics["redundancy_ratio"] < 0.01


class TestDirectSolve:
    """QORE RAG pure-solve path (roadmap §4.3): small N solves full QUBO,
    no prefilter; large N falls back to prefilter. Both return valid K-sets."""

    def test_small_n_direct_solve_returns_K(self):
        rng = np.random.default_rng(0)
        q = rng.standard_normal(32)
        emb = rng.standard_normal((20, 32))
        idx = select_passages(q, emb, K=5, method="qore", seed=1,
                              direct_solve_max_n=64)
        assert len(idx) == 5
        assert len(set(idx.tolist())) == 5  # no duplicates

    def test_direct_solve_avoids_redundant_pair(self):
        # Two near-duplicate high-relevance passages: pure QUBO should not spend
        # two of its K slots on both when diverse alternatives exist.
        rng = np.random.default_rng(3)
        d = 32
        q = rng.standard_normal(d)
        emb = q + 0.05 * rng.standard_normal((12, d))  # all relevant
        emb[1] = emb[0] + 1e-4 * rng.standard_normal(d)  # 0 and 1 near-identical
        idx = set(select_passages(q, emb, K=4, method="qore", seed=2,
                                  direct_solve_max_n=64).tolist())
        assert not (0 in idx and 1 in idx), "QUBO kept both near-duplicates"

    def test_large_n_prefilter_returns_K(self):
        rng = np.random.default_rng(1)
        q = rng.standard_normal(32)
        emb = rng.standard_normal((200, 32))
        idx = select_passages(q, emb, K=5, method="qore", seed=1,
                              direct_solve_max_n=64)
        assert len(idx) == 5
        assert len(set(idx.tolist())) == 5
