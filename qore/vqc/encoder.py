"""
Parameterized quantum encoder U(h; θ) for token feature encoding.

The encoder maps classical token features into quantum states. From these states:
- Measurement expectations → quality signal a_i
- State fidelities → redundancy signal b_ij

Supports PennyLane, TensorCircuit, and Qiskit backends.
"""

import numpy as np
from typing import Optional


class VQCEncoder:
    """
    Variational Quantum Circuit encoder that maps token features to quantum states.

    The same circuit produces both quality and redundancy signals:
    - quality: a_i = <ψ_i| O |ψ_i> (expectation of observable)
    - redundancy: b_ij = |<ψ_i|ψ_j>|² (state fidelity)

    Args:
        n_qubits: Number of qubits (features are PCA-reduced to this dim).
        n_layers: Number of variational layers in the circuit.
        backend: "pennylane", "tensorcircuit", or "qiskit".
        observable: Which observable to measure for quality.
            "Z0" = Z on qubit 0; "Z_mean" = average Z across all qubits.
        seed: Random seed for parameter initialization.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 2,
        backend: str = "tensorcircuit",
        observable: str = "Z_mean",
        seed: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.observable = observable

        # Initialize trainable parameters θ
        rng = np.random.default_rng(seed)
        # Each layer has: n_qubits rotation angles (RY) + n_qubits rotation angles (RZ)
        self.params = rng.uniform(
            -np.pi, np.pi, size=(n_layers, n_qubits, 2)
        )

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return self.params.size

    def encode_and_measure(self, features: np.ndarray) -> dict:
        """
        Encode all tokens and return both quality and redundancy signals.

        This is the unified interface: one call produces everything needed
        for QUBO construction.

        Args:
            features: (N, d) token feature matrix. Will be PCA-reduced to
                n_qubits dimensions internally.

        Returns:
            dict with:
                "quality": (N,) quality scores a_i
                "redundancy": (N, N) redundancy matrix b_ij
                "states": list of statevectors (for advanced use)
        """
        features = np.asarray(features, dtype=np.float64)
        N, d = features.shape

        # PCA reduce to n_qubits dimensions
        features_reduced = self._reduce_features(features)

        # Scale to [0, π] for angle encoding
        features_scaled = self._scale_features(features_reduced)

        # Dispatch to backend
        if self.backend == "tensorcircuit":
            return self._encode_tc(features_scaled)
        elif self.backend == "pennylane":
            return self._encode_pl(features_scaled)
        elif self.backend == "qiskit":
            return self._encode_qiskit(features_scaled)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _reduce_features(self, features: np.ndarray) -> np.ndarray:
        """PCA reduce features to n_qubits dimensions."""
        N, d = features.shape
        if d <= self.n_qubits:
            # Pad with zeros if features are fewer than qubits
            padded = np.zeros((N, self.n_qubits))
            padded[:, :d] = features
            return padded

        mean = features.mean(axis=0)
        centered = features - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        n_comp = min(self.n_qubits, N - 1, d)
        reduced = centered @ Vt[:n_comp].T

        if n_comp < self.n_qubits:
            padded = np.zeros((N, self.n_qubits))
            padded[:, :n_comp] = reduced
            return padded
        return reduced

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features to [0, π] for angle encoding."""
        f_min = features.min(axis=0)
        f_max = features.max(axis=0)
        scale = f_max - f_min
        scale[scale < 1e-12] = 1.0
        return (features - f_min) / scale * np.pi

    def _encode_tc(self, features: np.ndarray) -> dict:
        """TensorCircuit implementation."""
        import tensorcircuit as tc
        tc.set_backend("numpy")

        N = len(features)
        n_q = self.n_qubits
        params = self.params

        states = []
        qualities = []

        for i in range(N):
            c = tc.Circuit(n_q)
            x = features[i]

            # Data encoding + variational layers
            for layer in range(self.n_layers):
                # Data encoding: RY(x_k) on each qubit
                for k in range(n_q):
                    c.ry(k, theta=float(x[k]))

                # Variational: RY(θ) + RZ(θ) with trainable params
                for k in range(n_q):
                    c.ry(k, theta=float(params[layer, k, 0]))
                    c.rz(k, theta=float(params[layer, k, 1]))

                # Entangling: circular CNOT
                for k in range(n_q - 1):
                    c.cnot(k, k + 1)
                if n_q > 2:
                    c.cnot(n_q - 1, 0)

            # Get statevector
            state = np.array(c.state())
            states.append(state)

            # Measure quality: expectation of Z
            if self.observable == "Z0":
                # <Z> on qubit 0
                z_exp = float(np.real(c.expectation_ps(z=[0])))
            else:  # Z_mean
                z_exp = 0.0
                for k in range(n_q):
                    z_exp += float(np.real(c.expectation_ps(z=[k])))
                z_exp /= n_q

            # Transform from [-1, 1] to [0, 1] for quality
            qualities.append((z_exp + 1) / 2)

        # Compute redundancy: fidelity between all pairs
        quality = np.array(qualities)
        redundancy = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                fid = float(abs(np.conj(states[i]) @ states[j]) ** 2)
                redundancy[i, j] = fid
                redundancy[j, i] = fid

        return {"quality": quality, "redundancy": redundancy, "states": states}

    def _encode_pl(self, features: np.ndarray) -> dict:
        """PennyLane implementation."""
        import pennylane as qml

        N = len(features)
        n_q = self.n_qubits
        params = self.params

        dev = qml.device("default.qubit", wires=n_q)

        @qml.qnode(dev)
        def circuit_expval(x, p):
            """Circuit returning Z expectations."""
            for layer in range(self.n_layers):
                for k in range(n_q):
                    qml.RY(x[k], wires=k)
                for k in range(n_q):
                    qml.RY(p[layer, k, 0], wires=k)
                    qml.RZ(p[layer, k, 1], wires=k)
                for k in range(n_q - 1):
                    qml.CNOT(wires=[k, k + 1])
                if n_q > 2:
                    qml.CNOT(wires=[n_q - 1, 0])
            return [qml.expval(qml.PauliZ(k)) for k in range(n_q)]

        @qml.qnode(dev)
        def circuit_state(x, p):
            """Circuit returning statevector."""
            for layer in range(self.n_layers):
                for k in range(n_q):
                    qml.RY(x[k], wires=k)
                for k in range(n_q):
                    qml.RY(p[layer, k, 0], wires=k)
                    qml.RZ(p[layer, k, 1], wires=k)
                for k in range(n_q - 1):
                    qml.CNOT(wires=[k, k + 1])
                if n_q > 2:
                    qml.CNOT(wires=[n_q - 1, 0])
            return qml.state()

        states = []
        qualities = []

        for i in range(N):
            x = features[i]
            z_vals = circuit_expval(x, params)
            state = np.array(circuit_state(x, params))
            states.append(state)

            if self.observable == "Z0":
                z_exp = float(z_vals[0])
            else:
                z_exp = float(np.mean(z_vals))

            qualities.append((z_exp + 1) / 2)

        quality = np.array(qualities)
        redundancy = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                fid = float(abs(np.conj(states[i]) @ states[j]) ** 2)
                redundancy[i, j] = fid
                redundancy[j, i] = fid

        return {"quality": quality, "redundancy": redundancy, "states": states}

    def _encode_qiskit(self, features: np.ndarray) -> dict:
        """Qiskit implementation."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        N = len(features)
        n_q = self.n_qubits
        params = self.params

        states = []
        qualities = []

        for i in range(N):
            qc = QuantumCircuit(n_q)
            x = features[i]

            for layer in range(self.n_layers):
                for k in range(n_q):
                    qc.ry(float(x[k]), k)
                for k in range(n_q):
                    qc.ry(float(params[layer, k, 0]), k)
                    qc.rz(float(params[layer, k, 1]), k)
                for k in range(n_q - 1):
                    qc.cx(k, k + 1)
                if n_q > 2:
                    qc.cx(n_q - 1, 0)

            sv = Statevector.from_instruction(qc)
            state = np.array(sv.data)
            states.append(state)

            # Compute <Z_k> from state probabilities
            probs = np.abs(state) ** 2
            z_exps = []
            for k in range(n_q):
                # <Z_k> = P(qubit k = 0) - P(qubit k = 1)
                p0 = sum(probs[idx] for idx in range(2**n_q) if not (idx >> (n_q - 1 - k)) & 1)
                z_exps.append(2 * p0 - 1)

            if self.observable == "Z0":
                z_exp = z_exps[0]
            else:
                z_exp = np.mean(z_exps)

            qualities.append((z_exp + 1) / 2)

        quality = np.array(qualities)
        redundancy = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                fid = float(abs(np.conj(states[i]) @ states[j]) ** 2)
                redundancy[i, j] = fid
                redundancy[j, i] = fid

        return {"quality": quality, "redundancy": redundancy, "states": states}

    def update_params(self, new_params: np.ndarray):
        """Update trainable parameters (for training loop)."""
        expected_shape = (self.n_layers, self.n_qubits, 2)
        if new_params.shape != expected_shape:
            raise ValueError(f"Expected params shape {expected_shape}, got {new_params.shape}")
        self.params = new_params.copy()

    def save(self, path: str):
        """Save encoder config and trained parameters to a .npz file."""
        np.savez(
            path,
            params=self.params,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            backend=self.backend,
            observable=self.observable,
        )

    @classmethod
    def load(cls, path: str) -> "VQCEncoder":
        """Load a trained encoder from a .npz file."""
        data = np.load(path, allow_pickle=True)
        encoder = cls(
            n_qubits=int(data["n_qubits"]),
            n_layers=int(data["n_layers"]),
            backend=str(data["backend"]),
            observable=str(data["observable"]),
        )
        encoder.params = data["params"]
        return encoder
