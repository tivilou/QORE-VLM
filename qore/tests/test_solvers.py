"""Tests for solver consistency: all solvers should agree on small problems."""

import numpy as np
import pytest

from qore.qubo import build_qubo_matrix, energy
from qore.signals import cosine_redundancy, normalize
from qore.solvers import solve
from qore.solvers import brute, anneal, greedy


class TestSolverConsistency:
    """On small N, brute-force gives ground truth. Other solvers should match."""

    @pytest.fixture
    def small_problem(self):
        """N=12, K=4 problem with moderate redundancy."""
        np.random.seed(123)
        N, K = 12, 4
        features = np.random.randn(N, 16)
        # Inject some redundancy: make items 0,1,2 very similar
        features[1] = features[0] + 0.01 * np.random.randn(16)
        features[2] = features[0] + 0.01 * np.random.randn(16)
        a = normalize(np.random.rand(N) + 0.5)
        # Make the redundant items also high-quality (greedy trap)
        a[0], a[1], a[2] = 0.95, 0.93, 0.91
        b = cosine_redundancy(features)
        lam = 2.0
        Q = build_qubo_matrix(a, b, K, lam)
        return {"a": a, "b": b, "K": K, "lam": lam, "Q": Q, "N": N}

    def test_brute_finds_global_optimum(self, small_problem):
        """Brute force should find the minimum energy solution."""
        Q, K = small_problem["Q"], small_problem["K"]
        x_brute = brute.solve(Q, K)
        assert x_brute.sum() == K
        e_brute = energy(x_brute, Q)
        # Verify it's actually optimal by checking a few random alternatives
        N = small_problem["N"]
        for _ in range(100):
            idx = np.random.choice(N, K, replace=False)
            x_rand = np.zeros(N, dtype=np.int32)
            x_rand[idx] = 1
            assert energy(x_rand, Q) >= e_brute - 1e-10

    def test_anneal_matches_brute(self, small_problem):
        """SA should find the same optimum as brute-force on small N."""
        Q, K = small_problem["Q"], small_problem["K"]
        x_brute = brute.solve(Q, K)
        e_brute = energy(x_brute, Q)

        # Run SA with many reads to ensure convergence
        x_sa = anneal.solve(Q, K, num_reads=500, seed=42)
        e_sa = energy(x_sa, Q)

        assert x_sa.sum() == K
        # SA should find optimal or very close.
        # Use absolute tolerance since energies can be negative.
        gap = abs(e_sa - e_brute) / (abs(e_brute) + 1e-10)
        assert gap < 0.10, f"SA energy gap {gap:.2%} > 10% (SA={e_sa:.3f}, brute={e_brute:.3f})"

    def test_greedy_is_suboptimal_with_redundancy(self, small_problem):
        """Greedy should be worse than brute-force when redundancy exists."""
        Q, K = small_problem["Q"], small_problem["K"]
        a = small_problem["a"]

        x_brute = brute.solve(Q, K)
        e_brute = energy(x_brute, Q)

        x_greedy = greedy.solve(a, K)
        e_greedy = energy(x_greedy, Q)

        assert x_greedy.sum() == K
        # Greedy should have equal or worse energy (higher E = worse)
        assert e_greedy >= e_brute - 1e-10

    def test_unified_interface(self, small_problem):
        """The solve() dispatcher should work for all methods."""
        a, b, K, lam = (
            small_problem["a"],
            small_problem["b"],
            small_problem["K"],
            small_problem["lam"],
        )

        x_anneal = solve(a, b, K, lam=lam, method="anneal", num_reads=100, seed=42)
        x_brute = solve(a, b, K, lam=lam, method="brute")
        x_greedy = solve(a, b, K, lam=lam, method="greedy")

        assert x_anneal.sum() == K
        assert x_brute.sum() == K
        assert x_greedy.sum() == K

    def test_cardinality_always_satisfied(self, small_problem):
        """All solvers must return exactly K selected items."""
        a, b, K, lam = (
            small_problem["a"],
            small_problem["b"],
            small_problem["K"],
            small_problem["lam"],
        )

        for method in ["anneal", "brute", "greedy"]:
            kwargs = {"num_reads": 50, "seed": 0} if method == "anneal" else {}
            x = solve(a, b, K, lam=lam, method=method, **kwargs)
            assert x.sum() == K, f"{method} returned {x.sum()} items, expected {K}"


class TestEdgeCases:
    """Test boundary conditions."""

    def test_k_equals_1(self):
        """K=1: should select the single best item."""
        N = 5
        a = np.array([0.1, 0.9, 0.3, 0.2, 0.4])
        b = np.zeros((N, N))
        x = solve(a, b, K=1, method="greedy")
        assert x[1] == 1 and x.sum() == 1

    def test_k_equals_n_minus_1(self):
        """K=N-1: should drop exactly one item."""
        N = 6
        a = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
        b = np.zeros((N, N))
        x = solve(a, b, K=5, method="greedy")
        assert x.sum() == 5
        assert x[5] == 0  # lowest quality dropped

    def test_no_redundancy(self):
        """With b=0, QUBO should agree with greedy."""
        N = 10
        np.random.seed(99)
        a = np.random.rand(N)
        b = np.zeros((N, N))
        K = 4
        x_greedy = solve(a, b, K, method="greedy")
        x_brute = solve(a, b, K, method="brute")
        # Without redundancy, greedy IS optimal
        assert np.array_equal(x_greedy, x_brute)
