"""Tests for QUBO matrix construction."""

import numpy as np
import pytest

from qore.qubo import build_qubo_matrix, energy, energy_decomposed


class TestBuildQuboMatrix:
    """Verify Q matrix construction is mathematically correct."""

    def test_shape(self):
        N = 10
        a = np.random.rand(N)
        b = np.random.rand(N, N)
        b = (b + b.T) / 2
        np.fill_diagonal(b, 0)
        Q = build_qubo_matrix(a, b, K=3)
        assert Q.shape == (N, N)

    def test_upper_triangular(self):
        """Q should have zeros below diagonal (upper-triangular format)."""
        N = 8
        a = np.random.rand(N)
        b = np.random.rand(N, N)
        b = (b + b.T) / 2
        np.fill_diagonal(b, 0)
        Q = build_qubo_matrix(a, b, K=3)
        assert np.allclose(Q, np.triu(Q))

    def test_diagonal_values(self):
        """Q_ii = -a_i + lam*(1 - 2K)."""
        N = 5
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.zeros((N, N))
        K = 2
        lam = 3.0
        Q = build_qubo_matrix(a, b, K=K, lam=lam)
        expected_diag = -a + lam * (1 - 2 * K)
        assert np.allclose(np.diag(Q), expected_diag)

    def test_offdiag_values(self):
        """Q_ij (i<j) = b_ij + 2*lam."""
        N = 4
        a = np.ones(N)
        b = np.array([
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.4, 0.5],
            [0.2, 0.4, 0, 0.6],
            [0.3, 0.5, 0.6, 0],
        ])
        K = 2
        lam = 1.5
        Q = build_qubo_matrix(a, b, K=K, lam=lam)
        for i in range(N):
            for j in range(i + 1, N):
                assert np.isclose(Q[i, j], b[i, j] + 2 * lam)

    def test_invalid_K(self):
        N = 5
        a = np.ones(N)
        b = np.zeros((N, N))
        with pytest.raises(ValueError):
            build_qubo_matrix(a, b, K=0)
        with pytest.raises(ValueError):
            build_qubo_matrix(a, b, K=5)

    def test_invalid_b_shape(self):
        a = np.ones(5)
        b = np.zeros((4, 4))
        with pytest.raises(ValueError):
            build_qubo_matrix(a, b, K=2)


class TestEnergy:
    """Verify energy computation."""

    def test_manual(self):
        """Hand-computed example."""
        Q = np.array([[1.0, 2.0], [0.0, 3.0]])
        x = np.array([1, 1])
        # x^T Q x = 1*1 + 1*2*1 + 0 + 1*3 = 6
        assert np.isclose(energy(x, Q), 6.0)

    def test_zero_selection(self):
        Q = np.random.rand(5, 5)
        x = np.zeros(5)
        assert energy(x, Q) == 0.0

    def test_single_selection(self):
        N = 5
        Q = np.zeros((N, N))
        Q[2, 2] = -3.0
        x = np.zeros(N)
        x[2] = 1
        assert np.isclose(energy(x, Q), -3.0)


class TestEnergyDecomposed:
    """Verify decomposed energy matches total."""

    def test_consistency(self):
        """Decomposed terms should sum to total QUBO energy."""
        N = 8
        np.random.seed(42)
        a = np.random.rand(N)
        b = np.random.rand(N, N)
        b = (b + b.T) / 2
        np.fill_diagonal(b, 0)
        K = 3
        lam = 2.0

        Q = build_qubo_matrix(a, b, K, lam)
        x = np.zeros(N)
        x[[1, 3, 5]] = 1

        e_total = energy(x, Q)
        decomp = energy_decomposed(x, a, b, K, lam)

        assert np.isclose(decomp["total"], e_total, atol=1e-8)
        assert np.isclose(
            decomp["quality"] + decomp["redundancy"] + decomp["penalty"] + decomp["constant"],
            decomp["total"],
        )

    def test_feasible_no_penalty(self):
        """When |x| == K, penalty should be zero."""
        N = 6
        a = np.ones(N)
        b = np.zeros((N, N))
        K = 3
        x = np.array([1, 1, 1, 0, 0, 0])
        decomp = energy_decomposed(x, a, b, K, lam=5.0)
        assert decomp["penalty"] == 0.0
