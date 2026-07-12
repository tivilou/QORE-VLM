"""Simulated annealing solver via dwave-neal."""

import numpy as np
import dimod
import neal


def solve(
    Q: np.ndarray,
    K: int,
    num_reads: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Solve a QUBO using simulated annealing (dwave-neal).

    The solver samples multiple solutions and returns the lowest-energy one
    that satisfies the cardinality constraint |x| = K. If no feasible solution
    is found (rare for well-tuned lambda), returns the lowest-energy sample
    regardless of cardinality.

    Args:
        Q: (N, N) upper-triangular QUBO matrix.
        K: Desired number of selected items (for feasibility filtering).
        num_reads: Number of SA runs (more = better solution, slower).
        seed: Random seed for reproducibility.

    Returns:
        x: (N,) binary vector.
    """
    N = Q.shape[0]

    # Convert numpy Q matrix to dimod BQM
    # dimod expects a dict of {(i, j): value} for the QUBO
    qubo_dict = {}
    for i in range(N):
        if Q[i, i] != 0:
            qubo_dict[(i, i)] = float(Q[i, i])
        for j in range(i + 1, N):
            if Q[i, j] != 0:
                qubo_dict[(i, j)] = float(Q[i, j])

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)

    # Run simulated annealing
    sampler = neal.SimulatedAnnealingSampler()
    kwargs = {"num_reads": num_reads}
    if seed is not None:
        kwargs["seed"] = seed

    sampleset = sampler.sample(bqm, **kwargs)

    # Read solutions from the record arrays directly. sampleset.record.sample is
    # a (num_reads, num_vars) int array whose columns follow sampleset.variables
    # (the QUBO indices, not necessarily 0..N-1 or in order). We reindex each
    # row into a dense (N,) vector. Avoids the per-sample dict API, whose
    # `.get()` shape varies across dimod versions (SamplesArray has no `.get`).
    records = sampleset.record.sample  # (num_reads, num_vars)
    energies = sampleset.record.energy  # (num_reads,)
    variables = list(sampleset.variables)
    col_of = {v: c for c, v in enumerate(variables)}

    def to_dense(row):
        x = np.zeros(N, dtype=np.int32)
        for i in range(N):
            c = col_of.get(i)
            if c is not None:
                x[i] = row[c]
        return x

    # Find best feasible solution (|x| = K)
    best_energy = np.inf
    best_x = None
    for row, e in zip(records, energies):
        x = to_dense(row)
        if x.sum() == K and e < best_energy:
            best_energy = e
            best_x = x

    # Fallback: if no exact-K solution found, take lowest energy and fix
    if best_x is None:
        best_idx = int(np.argmin(energies))
        x = to_dense(records[best_idx])
        best_x = _fix_cardinality(x, Q, K)

    return best_x


def _fix_cardinality(x: np.ndarray, Q: np.ndarray, K: int) -> np.ndarray:
    """
    Greedily fix a solution to have exactly K selected items.

    If too many selected: drop the one whose removal decreases energy most.
    If too few: add the one whose addition decreases energy most.
    """
    x = x.copy()
    N = len(x)

    while x.sum() > K:
        # Find best item to remove
        selected = np.where(x == 1)[0]
        best_drop = None
        best_delta = np.inf
        for i in selected:
            x[i] = 0
            e = float(x @ Q @ x)
            if e < best_delta:
                best_delta = e
                best_drop = i
            x[i] = 1
        x[best_drop] = 0

    while x.sum() < K:
        # Find best item to add
        unselected = np.where(x == 0)[0]
        best_add = None
        best_delta = np.inf
        for i in unselected:
            x[i] = 1
            e = float(x @ Q @ x)
            if e < best_delta:
                best_delta = e
                best_add = i
            x[i] = 0
        x[best_add] = 1

    return x
