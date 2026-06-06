"""Quantum and classical kernels for computing redundancy matrix b_ij."""

import numpy as np
from typing import Optional


def quantum_kernel(
    features: np.ndarray,
    backend: str = "pennylane",
    n_qubits: Optional[int] = None,
    n_layers: int = 2,
) -> np.ndarray:
    """
    Compute pairwise quantum kernel (fidelity) matrix.

    b_ij = |<0| U(h_i)† U(h_j) |0>|²

    High fidelity means items i,j are "redundant" in quantum feature space.

    Args:
        features: (N, d) feature matrix. Will be PCA-reduced to n_qubits dims.
        backend: "pennylane", "tensorcircuit", or "qiskit".
        n_qubits: Number of qubits (= reduced feature dimension).
            If None, uses min(d, 8).
        n_layers: Number of data re-uploading layers in the feature map.

    Returns:
        b: (N, N) symmetric kernel matrix with zero diagonal. Values in [0, 1].
    """
    features = np.asarray(features, dtype=np.float64)
    N, d = features.shape

    if n_qubits is None:
        n_qubits = min(d, 8)

    # PCA reduction to n_qubits dimensions
    if d > n_qubits:
        features_reduced = _pca_reduce(features, n_qubits)
    else:
        features_reduced = features

    # Actual number of qubits may be less if N or d < n_qubits
    actual_qubits = features_reduced.shape[1]
    if actual_qubits < n_qubits:
        n_qubits = actual_qubits

    # Scale features to [0, pi] for angle encoding
    f_min = features_reduced.min(axis=0)
    f_max = features_reduced.max(axis=0)
    scale = f_max - f_min
    scale[scale < 1e-12] = 1.0
    features_scaled = (features_reduced - f_min) / scale * np.pi

    # Dispatch to backend
    if backend == "pennylane":
        b = _kernel_pennylane(features_scaled, n_qubits, n_layers)
    elif backend == "tensorcircuit":
        b = _kernel_tensorcircuit(features_scaled, n_qubits, n_layers)
    elif backend == "qiskit":
        b = _kernel_qiskit(features_scaled, n_qubits, n_layers)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: pennylane, tensorcircuit, qiskit")

    # Ensure symmetric, zero diagonal, clipped to [0, 1]
    b = (b + b.T) / 2
    np.clip(b, 0.0, 1.0, out=b)
    np.fill_diagonal(b, 0.0)
    return b


def _pca_reduce(features: np.ndarray, n_components: int) -> np.ndarray:
    """Simple PCA via SVD. Returns at most min(N-1, d, n_components) dimensions."""
    N, d = features.shape
    max_components = min(N - 1, d, n_components)
    if max_components < 1:
        max_components = 1
    mean = features.mean(axis=0)
    centered = features - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:max_components].T


def _kernel_pennylane(features: np.ndarray, n_qubits: int, n_layers: int) -> np.ndarray:
    """Fidelity kernel via PennyLane."""
    import pennylane as qml

    N = len(features)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        # Encode x1
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(x1[i], wires=i)
                qml.RZ(x1[i], wires=i)
            # Entangling
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Adjoint of encoding x2
        qml.adjoint(qml.StronglyEntanglingLayers)(
            weights=_make_weights(x2, n_qubits, n_layers), wires=range(n_qubits)
        )
        return qml.probs(wires=range(n_qubits))

    # Simpler approach: use AngleEmbedding + adjoint
    @qml.qnode(dev)
    def fidelity_circuit(x1, x2):
        """Compute |<0|U†(x2)U(x1)|0>|²."""
        # U(x1)
        for layer in range(n_layers):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Z")
            if n_qubits > 1:
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

        # U†(x2)
        for layer in reversed(range(n_layers)):
            if n_qubits > 1:
                for i in reversed(range(n_qubits - 1)):
                    qml.CNOT(wires=[i, i + 1])
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Z")
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")

        return qml.probs(wires=range(n_qubits))

    # Compute kernel matrix
    b = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            probs = fidelity_circuit(features[i], features[j])
            b[i, j] = float(probs[0])  # |<0...0|U†U|0...0>|²
            b[j, i] = b[i, j]

    return b


def _kernel_tensorcircuit(features: np.ndarray, n_qubits: int, n_layers: int) -> np.ndarray:
    """Fidelity kernel via TensorCircuit."""
    import tensorcircuit as tc

    tc.set_backend("numpy")
    N = len(features)

    def fidelity(x1, x2):
        """Compute |<0|U†(x2)U(x1)|0>|²."""
        c = tc.Circuit(n_qubits)

        # U(x1): data re-uploading
        for layer in range(n_layers):
            for i in range(n_qubits):
                c.ry(i, theta=float(x1[i]))
                c.rz(i, theta=float(x1[i]))
            if n_qubits > 1:
                for i in range(n_qubits - 1):
                    c.cnot(i, i + 1)

        # U†(x2): adjoint of encoding
        for layer in range(n_layers):
            if n_qubits > 1:
                for i in reversed(range(n_qubits - 1)):
                    c.cnot(i, i + 1)
            for i in reversed(range(n_qubits)):
                c.rz(i, theta=-float(x2[i]))
                c.ry(i, theta=-float(x2[i]))

        # Probability of |0...0>
        state = c.state()
        prob_zero = float(abs(state[0]) ** 2)
        return prob_zero

    # Compute kernel matrix
    b = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            b[i, j] = fidelity(features[i], features[j])
            b[j, i] = b[i, j]

    return b


def _kernel_qiskit(features: np.ndarray, n_qubits: int, n_layers: int) -> np.ndarray:
    """Fidelity kernel via Qiskit."""
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import Statevector

    N = len(features)

    def build_feature_map(x):
        """Build a parameterized feature map circuit."""
        qc = QuantumCircuit(n_qubits)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qc.ry(float(x[i]), i)
                qc.rz(float(x[i]), i)
            if n_qubits > 1:
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
        return qc

    def fidelity(x1, x2):
        """Compute |<0|U†(x2)U(x1)|0>|²."""
        qc1 = build_feature_map(x1)
        qc2 = build_feature_map(x2)

        # Get statevectors
        sv1 = Statevector.from_instruction(qc1)
        sv2 = Statevector.from_instruction(qc2)

        # Fidelity = |<sv2|sv1>|²
        overlap = float(abs(sv1.inner(sv2)) ** 2)
        return overlap

    # Compute kernel matrix
    b = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            b[i, j] = fidelity(features[i], features[j])
            b[j, i] = b[i, j]

    return b


def _make_weights(x, n_qubits, n_layers):
    """Helper to create StronglyEntanglingLayers weights from data."""
    import pennylane as qml
    weights = np.zeros((n_layers, n_qubits, 3))
    for layer in range(n_layers):
        for i in range(n_qubits):
            weights[layer, i, 0] = x[i]
            weights[layer, i, 1] = x[i]
            weights[layer, i, 2] = 0.0
    return weights
