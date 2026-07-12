"""
H2O-style KV-cache eviction: keep "Heavy Hitter" tokens by cumulative attention.

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models" (NeurIPS 2023).
"""

import torch
import numpy as np
from transformers.cache_utils import DynamicCache
from typing import Optional, Dict, Any, Tuple

from ..attention_accumulator import AttentionAccumulatorMixin, assert_single_batch


class H2OCache(AttentionAccumulatorMixin, DynamicCache):
    """
    H2O Heavy-Hitter eviction: keep tokens with the most cumulative attention.

    This is the greedy top-K baseline QORE aims to beat — it ranks tokens by the
    cumulative attention they receive (the "heavy hitter" score of Zhang et al.)
    and keeps the top-K, ignoring pairwise redundancy between kept entries.

    Quality signal is the real captured attention (via forward hooks +
    AttentionAccumulatorMixin). If no attention was captured (hooks not
    installed), it falls back to the key-norm proxy so the cache still works.
    """

    def __init__(
        self,
        max_capacity: int = 1024,
        trigger_every: int = 128,
        num_sink_tokens: int = 4,
        num_layers: Optional[int] = None,
        score_layer: int = 0,
    ):
        super().__init__()
        self.max_capacity = max_capacity
        self.trigger_every = trigger_every
        self.num_sink_tokens = num_sink_tokens
        self.score_layer = score_layer
        self._init_attention_state()
        self._tokens_since_eviction = 0
        self._eviction_count = 0
        self._num_layers = num_layers  # None = auto-detect
        self._last_layer_idx = -1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)

        if layer_idx == 0:
            self._tokens_since_eviction += key_states.shape[-2]

        # Detect total layers: len(key_cache) is unreliable during prefill (it
        # equals layer_idx+1 while layers are still being built, so layer 0 would
        # look like the "last" layer and get evicted alone). Instead detect the
        # real layer count when layer_idx wraps back — same approach as QORECache.
        if self._num_layers is None:
            if layer_idx <= self._last_layer_idx and self._last_layer_idx > 0:
                self._num_layers = self._last_layer_idx + 1
        self._last_layer_idx = layer_idx

        # Only evict once all layers have been updated this pass, so eviction is
        # synchronized across every layer (not just layer 0). Also decode-only
        # (query length == 1): evicting mid-prefill would truncate KV while the
        # last layer's full-length-query attention still runs → shape mismatch.
        is_decode_step = key_states.shape[-2] == 1
        if (self._num_layers is not None and
                layer_idx == self._num_layers - 1 and
                is_decode_step and
                self.get_seq_length() > self.max_capacity and
                self._tokens_since_eviction >= self.trigger_every):
            # Return pre-eviction tensors for this layer's attention (the step's
            # causal mask was sized from the pre-eviction length); the truncation
            # takes effect on the next forward.
            keys_ret = self.key_cache[layer_idx]
            values_ret = self.value_cache[layer_idx]
            self._evict()
            self._tokens_since_eviction = 0
            self._eviction_count += 1
            return keys_ret, values_ret

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _evict(self):
        """Greedy eviction: keep top-K by cumulative attention (heavy hitter)."""
        assert_single_batch(self.key_cache, "H2OCache")
        seq_len = self.get_seq_length()
        n_sink = min(self.num_sink_tokens, seq_len)
        n_keep = self.max_capacity - n_sink

        score_layer = min(self.score_layer, len(self.key_cache) - 1)
        keys = self.key_cache[score_layer][0]  # [num_heads, seq_len, head_dim]

        # Quality = real cumulative attention received (H2O heavy-hitter).
        attn = self.attention_scores()
        if self.has_attention() and attn is not None and attn.shape[0] == seq_len:
            scores = attn[n_sink:].to(keys.device)
        else:
            # Fallback proxy when no attention was captured: key-vector norm.
            scores = torch.norm(keys[:, n_sink:, :], dim=-1).mean(dim=0)

        # Greedy top-K
        _, top_indices = torch.topk(scores, n_keep)
        keep_candidates = top_indices.sort().values + n_sink

        # Combine sinks + top-K
        keep_positions = torch.cat([
            torch.arange(n_sink, device=keys.device),
            keep_candidates,
        ])

        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_positions, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_positions, :]

        self.prune_attention(keep_positions)

    @property
    def eviction_count(self) -> int:
        return self._eviction_count
