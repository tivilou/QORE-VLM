"""
Training loop for VQC encoder parameters using parameter-shift rule.

The encoder's parameters θ are optimized so that the VQC-derived signals
(quality + redundancy) produce QUBO solutions that maximize downstream
task performance (e.g., gold passage recall, information retention).
"""

import numpy as np
from typing import Callable, Optional
from .encoder import VQCEncoder
from ..qubo import build_qubo_matrix, energy
from ..solvers import solve as qore_solve
from ..signals import normalize


def parameter_shift_gradient(
    encoder: VQCEncoder,
    features: np.ndarray,
    K: int,
    loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    lam: float = 2.0,
    solver: str = "anneal",
    num_reads: int = 30,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """
    Compute gradient of loss w.r.t. encoder parameters using parameter-shift rule.

    For each parameter θ_k:
        ∂L/∂θ_k ≈ [L(θ_k + shift) - L(θ_k - shift)] / (2 sin(shift))

    Args:
        encoder: VQCEncoder with current parameters.
        features: (N, d) feature matrix.
        K: Selection budget.
        loss_fn: Function(x, quality, redundancy) → scalar loss.
            Takes the selection vector and signals, returns loss to minimize.
        lam: QUBO penalty weight.
        solver: Solver for QUBO within the loop.
        num_reads: SA reads.
        shift: Parameter shift amount (default π/2 for exact gradient).

    Returns:
        grad: Array with same shape as encoder.params, containing gradients.
    """
    params_flat = encoder.params.flatten()
    grad_flat = np.zeros_like(params_flat)
    original_params = encoder.params.copy()

    for idx in range(len(params_flat)):
        # Forward shift
        params_plus = params_flat.copy()
        params_plus[idx] += shift
        encoder.update_params(params_plus.reshape(original_params.shape))
        loss_plus = _evaluate(encoder, features, K, loss_fn, lam, solver, num_reads)

        # Backward shift
        params_minus = params_flat.copy()
        params_minus[idx] -= shift
        encoder.update_params(params_minus.reshape(original_params.shape))
        loss_minus = _evaluate(encoder, features, K, loss_fn, lam, solver, num_reads)

        # Gradient
        grad_flat[idx] = (loss_plus - loss_minus) / (2 * np.sin(shift))

    # Restore original params
    encoder.update_params(original_params)
    return grad_flat.reshape(original_params.shape)


def _evaluate(
    encoder: VQCEncoder,
    features: np.ndarray,
    K: int,
    loss_fn: Callable,
    lam: float,
    solver: str,
    num_reads: int,
) -> float:
    """Run the full pipeline and compute loss."""
    signals = encoder.encode_and_measure(features)
    a = normalize(signals["quality"])
    b = signals["redundancy"]
    np.fill_diagonal(b, 0.0)
    np.clip(b, 0.0, 1.0, out=b)

    x = qore_solve(a, b, K, lam=lam, method=solver, num_reads=num_reads)
    return loss_fn(x, a, b)


def train_encoder(
    encoder: VQCEncoder,
    features: np.ndarray,
    K: int,
    loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_steps: int = 50,
    lr: float = 0.1,
    lam: float = 2.0,
    solver: str = "anneal",
    num_reads: int = 30,
    verbose: bool = False,
) -> list:
    """
    Train the VQC encoder parameters to minimize a task-specific loss.

    Args:
        encoder: VQCEncoder to train (modified in-place).
        features: (N, d) training features.
        K: Selection budget.
        loss_fn: Function(x, quality, redundancy) → loss to minimize.
        n_steps: Number of gradient steps.
        lr: Learning rate.
        lam: QUBO penalty weight.
        solver: Solver backend.
        num_reads: SA reads per step.
        verbose: Print progress.

    Returns:
        losses: List of loss values per step.
    """
    losses = []

    for step in range(n_steps):
        # Compute current loss
        current_loss = _evaluate(encoder, features, K, loss_fn, lam, solver, num_reads)
        losses.append(current_loss)

        if verbose and step % 10 == 0:
            print(f"  Step {step:3d}: loss = {current_loss:.4f}")

        # Compute gradient
        grad = parameter_shift_gradient(
            encoder, features, K, loss_fn,
            lam=lam, solver=solver, num_reads=num_reads,
        )

        # Gradient descent update
        encoder.update_params(encoder.params - lr * grad)

        # Decay learning rate
        lr *= 0.99

    # Final loss
    final_loss = _evaluate(encoder, features, K, loss_fn, lam, solver, num_reads)
    losses.append(final_loss)
    if verbose:
        print(f"  Final:    loss = {final_loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Pre-built loss functions
# ---------------------------------------------------------------------------

def energy_loss(x: np.ndarray, quality: np.ndarray, redundancy: np.ndarray) -> float:
    """
    Loss = QUBO energy of the solution.

    Minimizing this trains the encoder to produce signals that lead to
    low-energy (high-quality, low-redundancy) selections.
    """
    Q = build_qubo_matrix(quality, redundancy, K=int(x.sum()), lam=2.0, gamma=1.0)
    return energy(x, Q)


def coverage_loss(
    gold_indices: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Factory: returns a loss function that penalizes missing gold items.

    Usage:
        loss_fn = coverage_loss(gold_indices=np.array([0, 1, 2, 3, 4]))
        train_encoder(encoder, features, K, loss_fn)
    """
    gold_set = set(int(i) for i in gold_indices)

    def loss(x: np.ndarray, quality: np.ndarray, redundancy: np.ndarray) -> float:
        selected = set(np.where(x == 1)[0])
        hits = len(selected & gold_set)
        # Negative recall as loss (lower = better)
        recall = hits / len(gold_set) if len(gold_set) > 0 else 0
        return -recall  # minimize → maximize recall

    return loss


def diversity_loss(x: np.ndarray, quality: np.ndarray, redundancy: np.ndarray) -> float:
    """
    Loss = average pairwise redundancy among selected items.

    Trains the encoder to identify which pairs are truly redundant.
    """
    selected = np.where(x == 1)[0]
    if len(selected) < 2:
        return 0.0
    b_sel = redundancy[np.ix_(selected, selected)]
    K = len(selected)
    return float(b_sel.sum()) / (K * (K - 1))
