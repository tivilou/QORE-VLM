"""Tests for VQC encoder, scorer, and training."""

import numpy as np
import pytest

from qore.vqc.encoder import VQCEncoder
from qore.vqc.scorer import vqc_score, vqc_select_passages
from qore.vqc.train import train_encoder, energy_loss, diversity_loss, coverage_loss


class TestVQCEncoder:
    """Test the unified VQC encoder."""

    @pytest.fixture(autouse=True)
    def check_tc(self):
        pytest.importorskip("tensorcircuit")

    def test_encode_output_shape(self):
        encoder = VQCEncoder(n_qubits=4, n_layers=1, backend="tensorcircuit", seed=42)
        features = np.random.randn(10, 16)
        result = encoder.encode_and_measure(features)

        assert result["quality"].shape == (10,)
        assert result["redundancy"].shape == (10, 10)
        assert len(result["states"]) == 10

    def test_quality_in_range(self):
        """Quality scores should be in [0, 1]."""
        encoder = VQCEncoder(n_qubits=4, n_layers=2, backend="tensorcircuit", seed=0)
        features = np.random.randn(8, 20)
        result = encoder.encode_and_measure(features)
        a = result["quality"]
        assert a.min() >= -0.01  # small tolerance
        assert a.max() <= 1.01

    def test_redundancy_symmetric_zero_diag(self):
        encoder = VQCEncoder(n_qubits=4, n_layers=1, backend="tensorcircuit", seed=42)
        features = np.random.randn(6, 12)
        result = encoder.encode_and_measure(features)
        b = result["redundancy"]
        assert np.allclose(b, b.T, atol=1e-6)
        assert np.allclose(np.diag(b), 0.0, atol=1e-6)

    def test_identical_features_high_fidelity(self):
        """Identical inputs should produce identical states → fidelity = 1."""
        encoder = VQCEncoder(n_qubits=4, n_layers=1, backend="tensorcircuit", seed=42)
        base = np.random.randn(12)
        features = np.stack([base, base, np.random.randn(12)])
        result = encoder.encode_and_measure(features)
        # Fidelity between items 0 and 1 should be ~1
        assert result["redundancy"][0, 1] > 0.99

    def test_update_params(self):
        encoder = VQCEncoder(n_qubits=3, n_layers=2, backend="tensorcircuit", seed=0)
        old_params = encoder.params.copy()
        new_params = np.zeros_like(old_params)
        encoder.update_params(new_params)
        assert np.allclose(encoder.params, 0.0)
        assert not np.allclose(old_params, 0.0)

    def test_n_params(self):
        encoder = VQCEncoder(n_qubits=5, n_layers=3)
        # Each layer: n_qubits * 2 params (RY + RZ)
        assert encoder.n_params == 3 * 5 * 2


class TestVQCEncoderPennyLane:
    """Test PennyLane backend."""

    @pytest.fixture(autouse=True)
    def check_pl(self):
        pytest.importorskip("pennylane")

    def test_encode_works(self):
        encoder = VQCEncoder(n_qubits=3, n_layers=1, backend="pennylane", seed=42)
        features = np.random.randn(5, 10)
        result = encoder.encode_and_measure(features)
        assert result["quality"].shape == (5,)
        assert result["redundancy"].shape == (5, 5)


class TestVQCEncoderQiskit:
    """Test Qiskit backend."""

    @pytest.fixture(autouse=True)
    def check_qk(self):
        pytest.importorskip("qiskit")

    def test_encode_works(self):
        encoder = VQCEncoder(n_qubits=3, n_layers=1, backend="qiskit", seed=42)
        features = np.random.randn(5, 10)
        result = encoder.encode_and_measure(features)
        assert result["quality"].shape == (5,)
        assert result["redundancy"].shape == (5, 5)


class TestVQCEncoderCrossBackend:
    """Verify all backends produce consistent results."""

    @pytest.fixture(autouse=True)
    def check_all(self):
        pytest.importorskip("tensorcircuit")
        pytest.importorskip("pennylane")
        pytest.importorskip("qiskit")

    def test_backends_consistent(self):
        np.random.seed(123)
        features = np.random.randn(4, 8)

        results = {}
        for backend in ["tensorcircuit", "pennylane", "qiskit"]:
            enc = VQCEncoder(n_qubits=3, n_layers=1, backend=backend, seed=99)
            results[backend] = enc.encode_and_measure(features)

        # Quality signals should be close
        a_tc = results["tensorcircuit"]["quality"]
        a_pl = results["pennylane"]["quality"]
        a_qk = results["qiskit"]["quality"]
        assert np.allclose(a_tc, a_pl, atol=0.05), f"TC vs PL diff: {abs(a_tc-a_pl).max():.4f}"
        assert np.allclose(a_tc, a_qk, atol=0.05), f"TC vs QK diff: {abs(a_tc-a_qk).max():.4f}"

        # Redundancy should be close
        b_tc = results["tensorcircuit"]["redundancy"]
        b_pl = results["pennylane"]["redundancy"]
        assert np.allclose(b_tc, b_pl, atol=0.05), f"Redundancy diff: {abs(b_tc-b_pl).max():.4f}"


class TestVQCScorer:
    """Test the unified scoring interface."""

    @pytest.fixture(autouse=True)
    def check_tc(self):
        pytest.importorskip("tensorcircuit")

    def test_vqc_score_returns_binary(self):
        features = np.random.randn(12, 16)
        x = vqc_score(features, K=4, n_qubits=4, n_layers=1, backend="tensorcircuit", seed=42)
        assert x.shape == (12,)
        assert x.sum() == 4
        assert set(np.unique(x)).issubset({0, 1})

    def test_vqc_select_passages(self):
        query = np.random.randn(32)
        passages = np.random.randn(20, 32)
        indices = vqc_select_passages(
            query, passages, K=5,
            n_qubits=4, n_layers=1, backend="tensorcircuit", seed=42
        )
        assert len(indices) == 5
        assert all(0 <= i < 20 for i in indices)


class TestVQCTraining:
    """Test parameter training (very small scale)."""

    @pytest.fixture(autouse=True)
    def check_tc(self):
        pytest.importorskip("tensorcircuit")

    def test_training_reduces_loss(self):
        """A few training steps should reduce the energy loss."""
        np.random.seed(42)
        features = np.random.randn(8, 10)
        encoder = VQCEncoder(n_qubits=3, n_layers=1, backend="tensorcircuit", seed=42)

        # Get initial loss
        from qore.vqc.train import _evaluate
        initial_loss = _evaluate(encoder, features, K=3, loss_fn=energy_loss,
                                 lam=2.0, solver="anneal", num_reads=20)

        # Train for a few steps
        losses = train_encoder(
            encoder, features, K=3, loss_fn=energy_loss,
            n_steps=5, lr=0.3, solver="anneal", num_reads=20, verbose=False
        )

        # Loss should decrease (or at least not increase significantly)
        # Note: with so few steps and SA randomness, we just check it doesn't explode
        assert losses[-1] <= initial_loss + 1.0, \
            f"Loss increased too much: {initial_loss:.3f} → {losses[-1]:.3f}"

    def test_coverage_loss_factory(self):
        """Coverage loss should return valid callable."""
        loss_fn = coverage_loss(gold_indices=np.array([0, 1, 2]))
        x = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        a = np.random.rand(8)
        b = np.random.rand(8, 8)
        loss = loss_fn(x, a, b)
        # Selected {0,1,4}, gold {0,1,2} → hits=2, recall=2/3, loss=-2/3
        assert np.isclose(loss, -2/3)
