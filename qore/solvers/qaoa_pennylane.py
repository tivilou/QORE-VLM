"""QAOA solver via PennyLane."""

import numpy as np


def solve(
    Q: np.ndarray,
    K: int,
    p: int = 2,
    maxiter: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Solve a QUBO using QAOA via PennyLane.

    Constructs a cost Hamiltonian from the QUBO matrix and optimizes
    variational parameters (gamma, beta) for p layers.

    Args:
        Q: (N, N) upper-triangular QUBO matrix.
        K: Desired cardinality (for feasibility checking).
        p: QAOA circuit depth (number of layers).
        maxiter: Maximum optimizer iterations.
        seed: Random seed for reproducibility.

    Returns:
        x: (N,) binary vector.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    N = Q.shape[0]
    if seed is not None:
        np.random.seed(seed)

    # Convert QUBO to Ising Hamiltonian
    # x_i = (1 - Z_i) / 2, so x_i x_j = (1 - Z_i)(1 - Z_j)/4
    # Expand and collect ZZ, Z, and constant terms
    coeffs = []
    obs = []

    constant = 0.0
    z_coeffs = np.zeros(N)

    for i in range(N):
        # Diagonal: Q_ii * x_i = Q_ii * (1 - Z_i) / 2
        constant += Q[i, i] / 2
        z_coeffs[i] -= Q[i, i] / 2

    for i in range(N):
        for j in range(i + 1, N):
            if abs(Q[i, j]) < 1e-12:
                continue
            # Off-diagonal: Q_ij * x_i * x_j = Q_ij * (1-Z_i)(1-Z_j)/4
            #   = Q_ij/4 * (1 - Z_i - Z_j + Z_i Z_j)
            constant += Q[i, j] / 4
            z_coeffs[i] -= Q[i, j] / 4
            z_coeffs[j] -= Q[i, j] / 4
            # ZZ term
            coeffs.append(Q[i, j] / 4)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    # Add single-Z terms
    for i in range(N):
        if abs(z_coeffs[i]) > 1e-12:
            coeffs.append(z_coeffs[i])
            obs.append(qml.PauliZ(i))

    cost_hamiltonian = qml.Hamiltonian(coeffs, obs)

    # Mixer Hamiltonian: standard X mixer
    mixer_coeffs = [1.0] * N
    mixer_obs = [qml.PauliX(i) for i in range(N)]
    mixer_hamiltonian = qml.Hamiltonian(mixer_coeffs, mixer_obs)

    # Device
    dev = qml.device("default.qubit", wires=N)

    # QAOA circuit
    @qml.qnode(dev)
    def qaoa_circuit(gammas, betas):
        # Initial state: |+>^N
        for i in range(N):
            qml.Hadamard(wires=i)

        # p layers of QAOA
        for layer in range(p):
            # Cost layer
            qml.ApproxTimeEvolution(cost_hamiltonian, gammas[layer], 1)
            # Mixer layer
            qml.ApproxTimeEvolution(mixer_hamiltonian, betas[layer], 1)

        return qml.probs(wires=range(N))

    # Precompute energies for all bitstrings
    energies = []
    for idx in range(2**N):
        bitstring = np.array([int(b) for b in format(idx, f'0{N}b')], dtype=np.float64)
        energies.append(float(bitstring @ Q @ bitstring))

    # Optimize parameters
    gammas = pnp.random.uniform(0, 2 * np.pi, p, requires_grad=True)
    betas = pnp.random.uniform(0, np.pi, p, requires_grad=True)

    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    energies_arr = pnp.array(energies, requires_grad=False)

    def cost_fn(gammas, betas):
        probs = qaoa_circuit(gammas, betas)
        # Expected energy: sum over all bitstrings of prob * energy
        return pnp.dot(probs, energies_arr)

    for _ in range(maxiter):
        gammas, betas = opt.step(cost_fn, gammas, betas)

    # Sample the best bitstring
    probs = qaoa_circuit(gammas, betas)
    best_idx = int(np.argmax(probs))
    x = np.array([int(b) for b in format(best_idx, f'0{N}b')], dtype=np.int32)

    # Fix cardinality if needed
    if x.sum() != K:
        x = _fix_cardinality(x, Q, K)

    return x


def _fix_cardinality(x: np.ndarray, Q: np.ndarray, K: int) -> np.ndarray:
    """Greedily fix solution to have exactly K selected items."""
    x = x.copy()
    N = len(x)

    while x.sum() > K:
        selected = np.where(x == 1)[0]
        best_drop, best_e = None, np.inf
        for i in selected:
            x[i] = 0
            e = float(x @ Q @ x)
            if e < best_e:
                best_e, best_drop = e, i
            x[i] = 1
        x[best_drop] = 0

    while x.sum() < K:
        unselected = np.where(x == 0)[0]
        best_add, best_e = None, np.inf
        for i in unselected:
            x[i] = 1
            e = float(x @ Q @ x)
            if e < best_e:
                best_e, best_add = e, i
            x[i] = 0
        x[best_add] = 1

    return x
