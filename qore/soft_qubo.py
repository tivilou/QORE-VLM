"""Differentiable QUBO for end-to-end optimization (Idea 7).

The standard QUBO solver is non-differentiable: it uses simulated annealing
to find a discrete binary solution x ∈ {0,1}^N. This prevents end-to-end
training with task loss (Recall, F1).

Idea 7 replaces the hard binary selection with a soft continuous relaxation:
- Input: a (quality), b (redundancy), K (budget)
- Output: soft selection probabilities p ∈ [0,1]^N with sum(p) ≈ K
- Differentiable: gradients flow through p back to a, b, or upstream encoders

Two implementation approaches:
1. **Gumbel-Softmax**: Inject Gumbel noise + softmax with temperature annealing
2. **Top-K gradient approximation**: Straight-through estimator for discrete selections

This module provides both, with the Gumbel-Softmax approach as the default
(more theoretically grounded for combinatorial optimization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftQUBO(nn.Module):
    """Differentiable QUBO selector using Gumbel-Softmax relaxation.

    The QUBO objective (to minimize):
        E(x) = -sum_i a_i x_i + gamma * sum_{i<j} b_ij x_i x_j + lam*(sum_i x_i - K)^2

    In the soft version, we replace binary x with continuous p ∈ [0,1]^N:
        E_soft(p) = -a^T p + gamma * p^T B p + lam*(sum(p) - K)^2

    where B is the symmetric redundancy matrix (b_ij for all i,j).

    Gumbel-Softmax allows us to sample soft selections that:
    - Are differentiable w.r.t. the scores
    - Approach discrete {0,1} selections as temperature → 0
    - Preserve the constraint sum(p) ≈ K via a K-hot formulation
    """

    def __init__(
        self,
        temperature: float = 1.0,
        hard: bool = False,
        use_straight_through: bool = False,
    ):
        """
        Args:
            temperature: Gumbel-Softmax temperature. Lower = closer to discrete.
                Start with 1.0, anneal to 0.1-0.5 during training.
            hard: If True, use straight-through estimator (forward uses argmax,
                backward uses soft probabilities). Only applies if use_straight_through=True.
            use_straight_through: Use straight-through estimator instead of pure Gumbel-Softmax.
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.use_straight_through = use_straight_through

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        K: int,
        lam: float = 2.0,
        gamma: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Soft QUBO selection.

        Args:
            a: (N,) quality scores (higher = more relevant)
            b: (N, N) redundancy matrix (higher b_ij = more redundant)
            K: number of passages to select
            lam: cardinality penalty weight
            gamma: redundancy penalty weight

        Returns:
            p: (N,) soft selection probabilities, sum(p) ≈ K
            info: dict with energy breakdown and selection stats
        """
        N = len(a)
        device = a.device

        # Convert redundancy to symmetric matrix if needed
        if b.dim() == 1:
            # Upper-triangular form -> full symmetric
            B = torch.zeros(N, N, device=device, dtype=a.dtype)
            idx = torch.triu_indices(N, N, offset=1, device=device)
            B[idx[0], idx[1]] = b
            B = B + B.T
        else:
            B = b

        # Compute scores for each item (negative of QUBO diagonal term)
        # Higher score = more likely to be selected
        # NOTE: This is a simplification - the full QUBO has pairwise terms
        # that should ideally be considered. For now, we use quality as logits.
        logits = a.clone()

        if self.use_straight_through and self.hard:
            # Hard selection with straight-through gradient
            _, top_k_idx = torch.topk(logits, K)
            p = torch.zeros(N, device=device, dtype=a.dtype)
            p[top_k_idx] = 1.0

            # For backward pass, we compute soft probabilities
            # (straight-through: forward=hard, backward=soft)
            if self.training:
                soft_p = self._gumbel_softmax_k_hot(logits, K, self.temperature)
                p = p + (soft_p - soft_p.detach())  # ST trick
        else:
            # Pure soft selection via Gumbel-Softmax
            p = self._gumbel_softmax_k_hot(logits, K, self.temperature)

        # Compute energy terms
        quality_term = -(a * p).sum()
        redundancy_term = gamma * (p @ B @ p)
        penalty_term = lam * (p.sum() - K) ** 2
        energy = quality_term + redundancy_term + penalty_term

        info = {
            'energy': energy.item(),
            'quality': quality_term.item(),
            'redundancy': redundancy_term.item(),
            'penalty': penalty_term.item(),
            'sum_p': p.sum().item(),
            'max_p': p.max().item(),
            'min_p': p.min().item(),
        }

        return p, info

    def _gumbel_softmax_k_hot(
        self,
        logits: torch.Tensor,
        K: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        K-hot Gumbel-Softmax: sample K items with soft probabilities.

        Standard Gumbel-Softmax produces a distribution over N categories.
        For K-subset selection, we need K-hot output. Strategy:

        1. Sample K times with Gumbel noise
        2. Apply softmax with temperature
        3. Each sample gets probability mass ≈ 1/K from the K selected items
        4. Sum across samples to get total selection strength per item

        Alternative (simpler): Use continuous relaxation of top-K.
        We'll use sigmoid(logits/temp) and scale to sum = K.
        """
        if not self.training:
            # Evaluation mode: deterministic soft selection
            scores = torch.sigmoid(logits / temperature)
            # Scale to sum = K
            p = scores * (K / scores.sum())
            return p

        # Training mode: add Gumbel noise for exploration
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        noisy_logits = (logits + gumbel_noise) / temperature

        # Soft top-K via sigmoid
        scores = torch.sigmoid(noisy_logits)

        # Scale to sum ≈ K
        p = scores * (K / (scores.sum() + 1e-8))

        return p


class LearnableQUBO(nn.Module):
    """Learnable QUBO weights for end-to-end optimization.

    Instead of using fixed gamma for redundancy weight, this module learns
    optimal weights for both quality and redundancy terms from task loss.

    Architecture:
        a_weighted = w_a * a
        b_weighted = w_b * b
        E_soft(p) = -(a_weighted)^T p + (b_weighted)^T (p ⊗ p) + penalty

    where w_a, w_b are learned scalars (or per-dimension weights).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        init_w_a: float = 1.0,
        init_w_b: float = 1.0,
        learn_lam: bool = False,
    ):
        """
        Args:
            temperature: Gumbel-Softmax temperature
            init_w_a: Initial quality weight
            init_w_b: Initial redundancy weight
            learn_lam: If True, also learn the cardinality penalty weight
        """
        super().__init__()
        self.soft_qubo = SoftQUBO(temperature=temperature)

        # Learnable weights (log-space to ensure positivity)
        self.log_w_a = nn.Parameter(torch.tensor(np.log(init_w_a)))
        self.log_w_b = nn.Parameter(torch.tensor(np.log(init_w_b)))

        if learn_lam:
            self.log_lam = nn.Parameter(torch.tensor(np.log(2.0)))
        else:
            self.register_buffer('log_lam', torch.tensor(np.log(2.0)))

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        K: int,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with learned weights.

        Args:
            a: (N,) quality scores
            b: (N, N) redundancy matrix
            K: budget

        Returns:
            p: (N,) soft selection probabilities
            info: dict with energy breakdown and learned weights
        """
        w_a = torch.exp(self.log_w_a)
        w_b = torch.exp(self.log_w_b)
        lam = torch.exp(self.log_lam)

        # Apply learned weights
        a_weighted = w_a * a
        b_weighted = w_b * b

        # Soft QUBO selection
        p, info = self.soft_qubo(a_weighted, b_weighted, K, lam=lam.item(), gamma=1.0)

        # Add learned weights to info
        info['w_a'] = w_a.item()
        info['w_b'] = w_b.item()
        info['lam'] = lam.item()

        return p, info


def soft_select_to_hard(
    p: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Convert soft selection probabilities to hard K-subset.

    Args:
        p: (N,) soft selection probabilities
        K: number of items to select

    Returns:
        x: (N,) binary selection vector {0, 1}^N with sum(x) = K
    """
    _, top_k_idx = torch.topk(p, K)
    x = torch.zeros_like(p)
    x[top_k_idx] = 1.0
    return x


def compute_recall_loss(
    p: torch.Tensor,
    gold_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute recall loss for soft selection.

    Recall = (# selected gold items) / (# total gold items)
           = sum_i [i in gold] * p_i / |gold|

    Loss = 1 - Recall (to minimize)

    Args:
        p: (N,) soft selection probabilities
        gold_indices: (G,) indices of gold items

    Returns:
        loss: scalar recall loss (1 - recall)
    """
    N = len(p)
    gold_mask = torch.zeros(N, device=p.device, dtype=p.dtype)
    gold_mask[gold_indices] = 1.0

    recall = (p * gold_mask).sum() / (gold_mask.sum() + 1e-8)
    loss = 1.0 - recall

    return loss
