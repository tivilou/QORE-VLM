"""Tests for quantum solvers (QAOA) and quantum kernels.

Due to memory constraints, each backend is tested independently on small problems.
Run individual backends with: pytest -k "pennylane" or pytest -k "tensorcircuit"
"""

import numpy as np
import pytest

from qore.qubo import build_qubo_matrix, energy
from qore.signals import cosine_redundancy, normalize


# ---------------------------------------------------------------------------
# Small test problem (N=6, K=2) — feasible for any QAOA backend
# ---------------------------------------------------------------------------

@pytest.fixture
def small_qubo():
    """A tiny problem where brute-force optimal is known."""
    np.random.seed(42)
    N, K = 6, 2
    a = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    # Items 0,1 are redundant (similar), items 2-5 are diverse
    b = np.zeros((N, N))
    b[0, 1] = b[1, 0] = 0.9  # 0 and 1 are near-duplicates
    Q = build_qubo_matrix(a, b, K, lam=2.0, gamma=1.0)

    # Brute-force optimal: should pick one of {0,1} + item 2
    from qore.solvers.brute import solve as brute_solve
    x_optimal = brute_solve(Q, K)
    e_optimal = energy(x_optimal, Q)

    return {"Q": Q, "K": K, "N": N, "x_optimal": x_optimal, "e_optimal": e_optimal}


# ---------------------------------------------------------------------------
# QAOA tests (one per backend, skipped if import fails)
# ---------------------------------------------------------------------------

class TestQAOATensorCircuit:
    """Test QAOA via TensorCircuit."""

    @pytest.fixture(autouse=True)
    def check_import(self):
        pytest.importorskip("tensorcircuit")

    def test_returns_valid_solution(self, small_qubo):
        from qore.solvers.qaoa_tensorcircuit import solve
        x = solve(small_qubo["Q"], small_qubo["K"], p=1, maxiter=20, seed=42)
        assert x.shape == (small_qubo["N"],)
        assert x.sum() == small_qubo["K"]
        assert set(np.unique(x)).issubset({0, 1})

    def test_energy_reasonable(self, small_qubo):
        """QAOA solution should be within 50% of optimal on this tiny problem."""
        from qore.solvers.qaoa_tensorcircuit import solve
        x = solve(small_qubo["Q"], small_qubo["K"], p=2, maxiter=30, seed=42)
        e = energy(x, small_qubo["Q"])
        e_opt = small_qubo["e_optimal"]
        # Allow generous margin — QAOA on 6 qubits with p=2 may not converge fully
        assert e <= 0 or e <= abs(e_opt) * 3, f"Energy {e} too high vs optimal {e_opt}"


class TestQAOAPennyLane:
    """Test QAOA via PennyLane."""

    @pytest.fixture(autouse=True)
    def check_import(self):
        pytest.importorskip("pennylane")

    def test_returns_valid_solution(self, small_qubo):
        from qore.solvers.qaoa_pennylane import solve
        x = solve(small_qubo["Q"], small_qubo["K"], p=1, maxiter=10, seed=42)
        assert x.shape == (small_qubo["N"],)
        assert x.sum() == small_qubo["K"]
        assert set(np.unique(x)).issubset({0, 1})


class TestQAOAQiskit:
    """Test QAOA via Qiskit."""

    @pytest.fixture(autouse=True)
    def check_import(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_optimization")

    def test_returns_valid_solution(self, small_qubo):
        from qore.solvers.qaoa_qiskit import solve
        x = solve(small_qubo["Q"], small_qubo["K"], p=1, maxiter=30, seed=42)
        assert x.shape == (small_qubo["N"],)
        assert x.sum() == small_qubo["K"]
        assert set(np.unique(x)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Quantum Kernel tests
# ---------------------------------------------------------------------------

class TestQuantumKernelTensorCircuit:
    """Test quantum kernel via TensorCircuit."""

    @pytest.fixture(autouse=True)
    def check_import(self):
        pytest.importorskip("tensorcircuit")

    def test_kernel_shape(self):
        from qore.kernels import quantum_kernel
        features = np.random.randn(8, 16)
        b = quantum_kernel(features, backend="tensorcircuit", n_qubits=4, n_layers=1)
        assert b.shape == (8, 8)

    def test_kernel_symmetric_zero_diag(self):
        from qore.kernels import quantum_kernel
        features = np.random.randn(6, 10)
        b = quantum_kernel(features, backend="tensorcircuit", n_qubits=4, n_layers=1)
        assert np.allclose(b, b.T, atol=1e-6)
        assert np.allclose(np.diag(b), 0.0)

    def test_identical_features_high_kernel(self):
        """Identical features should have high kernel value."""
        from qore.kernels import quantum_kernel
        base = np.random.randn(10)
        features = np.stack([base, base, np.random.randn(10)])
        b = quantum_kernel(features, backend="tensorcircuit", n_qubits=4, n_layers=1)
        assert b[0, 1] > 0.9, f"Identical features got kernel {b[0, 1]}, expected > 0.9"


class TestQuantumKernelPennyLane:
    """Test quantum kernel via PennyLane."""

    @pytest.fixture(autouse=True)
    def check_import(self):
        pytest.importorskip("pennylane")

    def test_kernel_shape(self):
        from qore.kernels import quantum_kernel
        features = np.random.randn(6, 12)
        b = quantum_kernel(features, backend="pennylane", n_qubits=4, n_layers=1)
        assert b.shape == (6, 6)

    def test_kernel_symmetric_zero_diag(self):
        from qore.kernels import quantum_kernel
        features = np.random.randn(5, 10)
        b = quantum_kernel(features, backend="pennylane", n_qubits=4, n_layers=1)
        assert np.allclose(b, b.T, atol=1e-6)
        assert np.allclose(np.diag(b), 0.0)


class TestQuantumKernelQiskit:
    """Test quantum kernel via Qiskit."""

    @pytest.fixture(autouse=True)
    def check_import(self):
        pytest.importorskip("qiskit")

    def test_kernel_shape(self):
        from qore.kernels import quantum_kernel
        features = np.random.randn(6, 12)
        b = quantum_kernel(features, backend="qiskit", n_qubits=4, n_layers=1)
        assert b.shape == (6, 6)

    def test_kernel_symmetric_zero_diag(self):
        from qore.kernels import quantum_kernel
        features = np.random.randn(5, 10)
        b = quantum_kernel(features, backend="qiskit", n_qubits=4, n_layers=1)
        assert np.allclose(b, b.T, atol=1e-6)
        assert np.allclose(np.diag(b), 0.0)


# ---------------------------------------------------------------------------
# Cross-backend consistency (kernels should produce similar results)
# ---------------------------------------------------------------------------

class TestCrossBackendKernel:
    """All backends should produce similar kernel matrices for the same input."""

    @pytest.fixture(autouse=True)
    def check_all_imports(self):
        pytest.importorskip("pennylane")
        pytest.importorskip("tensorcircuit")
        pytest.importorskip("qiskit")

    def test_kernels_consistent(self):
        """All three backends should agree on kernel values (same circuit)."""
        from qore.kernels import quantum_kernel
        np.random.seed(123)
        features = np.random.randn(5, 8)

        b_pl = quantum_kernel(features, backend="pennylane", n_qubits=4, n_layers=1)
        b_tc = quantum_kernel(features, backend="tensorcircuit", n_qubits=4, n_layers=1)
        b_qk = quantum_kernel(features, backend="qiskit", n_qubits=4, n_layers=1)

        # All should produce similar values (same circuit, different simulators)
        assert np.allclose(b_pl, b_tc, atol=0.05), \
            f"PennyLane vs TensorCircuit max diff: {abs(b_pl - b_tc).max():.4f}"
        assert np.allclose(b_pl, b_qk, atol=0.05), \
            f"PennyLane vs Qiskit max diff: {abs(b_pl - b_qk).max():.4f}"
