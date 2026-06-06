"""QAOA solver via TensorCircuit."""

import numpy as np


def solve(
    Q: np.ndarray,
    K: int,
    p: int = 2,
    maxiter: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Solve a QUBO using QAOA via TensorCircuit.

    Uses TensorCircuit's native PyTorch/NumPy backend for fast parameterized
    circuit simulation with automatic differentiation.

    Args:
        Q: (N, N) upper-triangular QUBO matrix.
        K: Desired cardinality (for feasibility checking).
        p: QAOA circuit depth (number of layers).
        maxiter: Maximum optimizer iterations.
        seed: Random seed for reproducibility.

    Returns:
        x: (N,) binary vector.
    """
    import tensorcircuit as tc

    tc.set_backend("numpy")
    N = Q.shape[0]

    if seed is not None:
        np.random.seed(seed)

    # Precompute energies for all 2^N bitstrings (feasible for N ≤ ~20)
    # For larger N, use sampling-based estimation
    if N <= 20:
        all_energies = _precompute_energies(Q, N)
        use_exact = True
    else:
        use_exact = False

    # Convert QUBO to Ising coefficients for the cost unitary
    # x_i = (1 - z_i) / 2 where z_i ∈ {+1, -1}
    h = np.zeros(N)  # single-qubit Z coefficients
    J = np.zeros((N, N))  # two-qubit ZZ coefficients

    for i in range(N):
        h[i] -= Q[i, i] / 2

    for i in range(N):
        for j in range(i + 1, N):
            if abs(Q[i, j]) < 1e-12:
                continue
            h[i] -= Q[i, j] / 4
            h[j] -= Q[i, j] / 4
            J[i, j] = Q[i, j] / 4

    def qaoa_circuit(gammas, betas):
        """Build QAOA circuit and return probability distribution."""
        c = tc.Circuit(N)

        # Initial |+>^N state
        for i in range(N):
            c.H(i)

        # p QAOA layers
        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Cost layer: exp(-i * gamma * C)
            # ZZ interactions
            for i in range(N):
                for j in range(i + 1, N):
                    if abs(J[i, j]) > 1e-12:
                        c.rzz(i, j, theta=2 * gamma * J[i, j])

            # Single Z rotations
            for i in range(N):
                if abs(h[i]) > 1e-12:
                    c.rz(i, theta=2 * gamma * h[i])

            # Mixer layer: exp(-i * beta * B) where B = sum X_i
            for i in range(N):
                c.rx(i, theta=2 * beta)

        return c.probability()

    # Optimization loop
    best_x = None
    best_energy = np.inf

    # Multiple random initializations for robustness
    n_inits = 3
    for init in range(n_inits):
        gammas = np.random.uniform(0, 2 * np.pi, p)
        betas = np.random.uniform(0, np.pi, p)

        # Simple gradient-free optimization (COBYLA-style coordinate descent)
        step_size = 0.3
        for iteration in range(maxiter // n_inits):
            probs = np.array(qaoa_circuit(gammas, betas))

            if use_exact:
                current_energy = float(np.dot(probs, all_energies))
            else:
                # Sample top probabilities for energy estimation
                top_indices = np.argsort(probs)[-min(100, len(probs)):]
                current_energy = sum(
                    probs[idx] * _bitstring_energy(Q, N, idx)
                    for idx in top_indices
                )

            # Coordinate-wise parameter update (finite difference)
            for param_idx in range(p):
                # Gamma update
                gammas[param_idx] += step_size
                probs_plus = np.array(qaoa_circuit(gammas, betas))
                if use_exact:
                    e_plus = float(np.dot(probs_plus, all_energies))
                else:
                    top_idx = np.argsort(probs_plus)[-min(100, len(probs_plus)):]
                    e_plus = sum(probs_plus[i] * _bitstring_energy(Q, N, i) for i in top_idx)

                gammas[param_idx] -= 2 * step_size
                probs_minus = np.array(qaoa_circuit(gammas, betas))
                if use_exact:
                    e_minus = float(np.dot(probs_minus, all_energies))
                else:
                    top_idx = np.argsort(probs_minus)[-min(100, len(probs_minus)):]
                    e_minus = sum(probs_minus[i] * _bitstring_energy(Q, N, i) for i in top_idx)

                gammas[param_idx] += step_size  # reset
                grad = (e_plus - e_minus) / (2 * step_size)
                gammas[param_idx] -= 0.1 * grad

                # Beta update
                betas[param_idx] += step_size
                probs_plus = np.array(qaoa_circuit(gammas, betas))
                if use_exact:
                    e_plus = float(np.dot(probs_plus, all_energies))
                else:
                    top_idx = np.argsort(probs_plus)[-min(100, len(probs_plus)):]
                    e_plus = sum(probs_plus[i] * _bitstring_energy(Q, N, i) for i in top_idx)

                betas[param_idx] -= 2 * step_size
                probs_minus = np.array(qaoa_circuit(gammas, betas))
                if use_exact:
                    e_minus = float(np.dot(probs_minus, all_energies))
                else:
                    top_idx = np.argsort(probs_minus)[-min(100, len(probs_minus)):]
                    e_minus = sum(probs_minus[i] * _bitstring_energy(Q, N, i) for i in top_idx)

                betas[param_idx] += step_size  # reset
                grad = (e_plus - e_minus) / (2 * step_size)
                betas[param_idx] -= 0.1 * grad

            step_size *= 0.99  # decay

        # Get best bitstring from final distribution
        probs = np.array(qaoa_circuit(gammas, betas))
        best_idx_local = int(np.argmax(probs))
        x_local = np.array([int(b) for b in format(best_idx_local, f'0{N}b')], dtype=np.int32)
        e_local = float(x_local @ Q @ x_local)

        if e_local < best_energy:
            best_energy = e_local
            best_x = x_local

    # Fix cardinality if needed
    if best_x.sum() != K:
        best_x = _fix_cardinality(best_x, Q, K)

    return best_x


def _precompute_energies(Q: np.ndarray, N: int) -> np.ndarray:
    """Precompute QUBO energy for all 2^N bitstrings."""
    energies = np.zeros(2**N)
    for idx in range(2**N):
        x = np.array([int(b) for b in format(idx, f'0{N}b')], dtype=np.float64)
        energies[idx] = float(x @ Q @ x)
    return energies


def _bitstring_energy(Q: np.ndarray, N: int, idx: int) -> float:
    """Compute energy for a single bitstring index."""
    x = np.array([int(b) for b in format(idx, f'0{N}b')], dtype=np.float64)
    return float(x @ Q @ x)


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
