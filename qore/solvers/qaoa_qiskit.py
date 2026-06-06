"""QAOA solver via Qiskit (qiskit-optimization)."""

import numpy as np


def solve(
    Q: np.ndarray,
    K: int,
    p: int = 2,
    maxiter: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Solve a QUBO using QAOA via Qiskit.

    Uses qiskit-optimization's QuadraticProgram → MinimumEigenOptimizer pipeline.

    Args:
        Q: (N, N) upper-triangular QUBO matrix.
        K: Desired cardinality (for feasibility checking).
        p: QAOA circuit depth (number of layers). Higher p = better approximation.
        maxiter: Maximum optimizer iterations for variational parameters.
        seed: Random seed for reproducibility.

    Returns:
        x: (N,) binary vector.
    """
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import StatevectorSampler

    N = Q.shape[0]

    # Build QuadraticProgram from Q matrix
    qp = QuadraticProgram("qore_selection")
    for i in range(N):
        qp.binary_var(name=f"x{i}")

    # Objective: min x^T Q x
    linear = {f"x{i}": float(Q[i, i]) for i in range(N)}
    quadratic = {}
    for i in range(N):
        for j in range(i + 1, N):
            if abs(Q[i, j]) > 1e-12:
                quadratic[(f"x{i}", f"x{j}")] = float(Q[i, j])

    qp.minimize(linear=linear, quadratic=quadratic)

    # Set up QAOA
    optimizer = COBYLA(maxiter=maxiter)
    sampler = StatevectorSampler(seed=seed)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)

    # Solve
    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qp)

    # Extract solution
    x = np.array([int(result.variables_dict[f"x{i}"]) for i in range(N)], dtype=np.int32)

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
