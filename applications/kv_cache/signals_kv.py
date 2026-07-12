"""Signal construction for KV-cache eviction: quality and redundancy from key states."""

import torch
import numpy as np


def key_norm_quality(keys: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token quality signal from key vector norms.

    Intuition: tokens with larger key norms tend to attract more attention
    (since attention = softmax(Q·K^T / sqrt(d)), larger K norms → larger
    dot products → more attention received). This is a cheap proxy for
    cumulative attention without needing hooks.

    Args:
        keys: (num_heads, seq_len, head_dim) key states from one layer.

    Returns:
        quality: (seq_len,) importance scores (higher = more important).
    """
    # L2 norm per token, averaged across heads
    # keys shape: [num_heads, seq_len, head_dim]
    norms = torch.norm(keys, dim=-1)  # [num_heads, seq_len]
    quality = norms.mean(dim=0)  # [seq_len] — average across heads
    return quality


def attention_score_quality(
    attention_weights: torch.Tensor,
    previous_scores: torch.Tensor | None = None,
    decay: float = 0.9,
) -> torch.Tensor:
    """
    Accumulate attention-based importance scores (H2O-style "heavy hitter").

    Each token's importance = how much attention it receives from later tokens.
    Scores are exponentially decayed to favor recent importance.

    Args:
        attention_weights: (num_heads, 1, seq_len) attention from the latest
            generated token to all cached tokens.
        previous_scores: (seq_len - 1,) accumulated scores from prior steps.
            None on first call.
        decay: Exponential decay for historical scores.

    Returns:
        scores: (seq_len,) updated cumulative importance scores.
    """
    # attention_weights: [num_heads, 1, seq_len] — attention from new token to all
    # Sum across heads, squeeze query dim
    new_attn = attention_weights.sum(dim=0).squeeze(0)  # [seq_len]

    if previous_scores is None:
        return new_attn

    # Decay previous scores and add new attention
    seq_len = new_attn.shape[0]
    if previous_scores.shape[0] < seq_len:
        # Pad previous scores for newly added tokens
        padded = torch.zeros(seq_len, device=previous_scores.device, dtype=previous_scores.dtype)
        padded[:previous_scores.shape[0]] = previous_scores
        previous_scores = padded

    scores = decay * previous_scores[:seq_len] + new_attn
    return scores


def pairwise_key_similarity(
    keys: torch.Tensor,
    method: str = "cosine",
    max_heads: int = 4,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute pairwise redundancy matrix from key vectors.

    Two KV entries with similar keys will attend to similar queries —
    keeping both is redundant.

    Args:
        keys: (num_heads, seq_len, head_dim) key states.
        method: "cosine" (default) or "rbf".
        max_heads: Use at most this many heads for similarity computation
            (for speed — full computation is num_heads × seq_len²).
        sigma: RBF bandwidth (only for method="rbf").

    Returns:
        sim: (seq_len, seq_len) symmetric similarity matrix, zero diagonal.
    """
    num_heads, seq_len, head_dim = keys.shape

    # Use a subset of heads for efficiency
    head_indices = torch.linspace(0, num_heads - 1, min(max_heads, num_heads)).long()
    keys_subset = keys[head_indices]  # [max_heads, seq_len, head_dim]

    # Average across selected heads for a single representation per position
    keys_avg = keys_subset.mean(dim=0)  # [seq_len, head_dim]

    if method == "cosine":
        # Mean-center before cosine to correct transformer KEY ANISOTROPY.
        # Raw key vectors cluster in a narrow cone around a shared dominant
        # direction, so raw cosine similarity is ~1 for EVERY pair (measured on
        # Qwen2.5-7B: mean 0.85, std 0.04 — no discrimination). That saturated
        # redundancy collapses the QUBO's auto-tuned gamma to its floor, which
        # neutralizes the redundancy term entirely and degrades QORE to a plain
        # per-block top-K on the (early-biased) attention signal — producing
        # scattered, low-quality keep-sets and garbage generation on larger
        # models. Subtracting the per-batch mean removes the common component
        # and exposes the real pairwise structure (post-centering: mean ~0,
        # std 0.15). This is the standard anisotropy correction and lives in
        # signal construction, not the solver.
        orig_norms = torch.norm(keys_avg, dim=-1, keepdim=True)
        keys_centered = keys_avg - keys_avg.mean(dim=0, keepdim=True)
        norms = torch.norm(keys_centered, dim=-1, keepdim=True)
        # A vector whose centered norm is negligible relative to its ORIGINAL
        # norm has no distinguishing structure — it was essentially just the
        # common component. Zero it out so it contributes 0 similarity, rather
        # than letting a tiny floating-point residual normalize into a unit
        # vector (those residuals all point the same way and would spuriously
        # read as cosine ~1). Comparing against the original norm (not the
        # centered max) correctly catches the case where EVERY key is near-equal:
        # fp32 centering then leaves ~1e-7 residuals that an absolute 1e-12 floor
        # would miss.
        valid = (norms >= 1e-4 * orig_norms.clamp(min=1e-12)).float()
        normed = (keys_centered / norms.clamp(min=1e-12)) * valid
        sim = torch.mm(normed, normed.t())  # [seq_len, seq_len]
        sim.clamp_(min=0.0, max=1.0)
        sim.fill_diagonal_(0.0)

    elif method == "rbf":
        # Pairwise squared distances
        sq_norms = (keys_avg ** 2).sum(dim=-1)
        dist_sq = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * torch.mm(keys_avg, keys_avg.t())
        dist_sq.clamp_(min=0.0)
        sim = torch.exp(-dist_sq / (2 * sigma ** 2))
        sim.fill_diagonal_(0.0)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'cosine' or 'rbf'.")

    return sim
