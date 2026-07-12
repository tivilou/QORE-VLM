"""
AttentionAccumulatorMixin: shared cumulative-attention bookkeeping for caches.

Caches that select tokens by "heavy hitter" attention (H2O, SnapKV, QORE with
real attention) mix this in. It maintains a per-position running sum of attention
received, fed by the forward hooks in ``attention_capture.py`` via
``add_attention``, and keeps that sum aligned with the KV tensors across
evictions via ``prune_attention``.
"""

import torch
from typing import Optional


class AttentionAccumulatorMixin:
    """Adds an attention accumulator to a DynamicCache subclass.

    Subclasses must call ``self._init_attention_state()`` in ``__init__`` and,
    inside ``_evict``, call ``self.prune_attention(keep_tensor)`` with the same
    index tensor applied to the KV cache.
    """

    def _init_attention_state(self):
        # Per-key cumulative attention, shape [seq_len]. None until first hook.
        self._attn_scores: Optional[torch.Tensor] = None
        # Set True once any real attention has been captured, so callers can
        # decide whether to fall back to a proxy (e.g. key norm).
        self._has_attention = False

    def add_attention(self, layer_idx: int, scores: torch.Tensor):
        """
        Accumulate per-key attention scores from one layer's forward pass.

        We aggregate across layers additively within and across steps: every
        layer's hook contributes, giving a robust importance estimate. Scores
        for newly appended positions extend the accumulator; existing positions
        are incremented.

        Args:
            layer_idx: source layer (unused for aggregation, kept for hooks/debug).
            scores: [kv_len] attention received per cached key this pass.
        """
        scores = scores.detach().to(torch.float32)
        n = scores.shape[0]
        if self._attn_scores is None:
            self._attn_scores = scores.clone()
        elif self._attn_scores.shape[0] == n:
            self._attn_scores += scores
        elif self._attn_scores.shape[0] < n:
            # New tokens were appended since last accumulation: grow then add.
            grown = torch.zeros(n, dtype=torch.float32, device=scores.device)
            grown[: self._attn_scores.shape[0]] = self._attn_scores.to(scores.device)
            grown += scores
            self._attn_scores = grown
        else:
            # Accumulator longer than scores (shouldn't happen post-prune, but be
            # safe): add onto the leading slice.
            self._attn_scores[:n] += scores
        self._has_attention = True

    def prune_attention(self, keep_tensor: torch.Tensor):
        """Keep only the accumulator entries at ``keep_tensor`` positions."""
        if self._attn_scores is None:
            return
        idx = keep_tensor.to(self._attn_scores.device).long()
        idx = idx[idx < self._attn_scores.shape[0]]
        self._attn_scores = self._attn_scores[idx]

    def has_attention(self) -> bool:
        return self._has_attention

    def attention_scores(self) -> Optional[torch.Tensor]:
        return self._attn_scores


def assert_single_batch(key_cache, cache_name: str):
    """Guard: eviction is computed from batch element 0 and applied to all rows.

    That is correct only for autoregressive generation (batch_size == 1). With
    batched generation of differing content, rows 1..B-1 would have the wrong
    tokens evicted. All eviction caches share this assumption; fail loudly
    rather than silently corrupt.
    """
    if key_cache and key_cache[0].shape[0] != 1:
        raise AssertionError(
            f"{cache_name} eviction assumes batch_size==1 (generation). "
            f"Got batch={key_cache[0].shape[0]}; batched eviction is not supported."
        )
