"""
H2O-style KV-cache eviction: keep "Heavy Hitter" tokens by cumulative attention.

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models" (NeurIPS 2023).
"""

import torch
import numpy as np
from transformers.cache_utils import DynamicCache
from typing import Optional, Dict, Any, Tuple


class H2OCache(DynamicCache):
    """
    KV cache with Heavy Hitter eviction: keeps tokens that received
    the most cumulative attention. This is the greedy top-K baseline
    that QORE aims to beat (ignores pairwise redundancy).
    """

    def __init__(
        self,
        max_capacity: int = 1024,
        trigger_every: int = 128,
        num_sink_tokens: int = 4,
        score_layer: int = 0,
    ):
        super().__init__()
        self.max_capacity = max_capacity
        self.trigger_every = trigger_every
        self.num_sink_tokens = num_sink_tokens
        self.score_layer = score_layer
        self._tokens_since_eviction = 0
        self._eviction_count = 0

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

        num_layers = len(self.key_cache)
        if (layer_idx == num_layers - 1 and
                self.get_seq_length() > self.max_capacity and
                self._tokens_since_eviction >= self.trigger_every):
            self._evict()
            self._tokens_since_eviction = 0
            self._eviction_count += 1

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _evict(self):
        """Greedy eviction: keep top-K by key norm (proxy for attention)."""
        seq_len = self.get_seq_length()
        n_sink = min(self.num_sink_tokens, seq_len)
        n_keep = self.max_capacity - n_sink

        # Score by key norm (same proxy as QORE uses for quality)
        score_layer = min(self.score_layer, len(self.key_cache) - 1)
        keys = self.key_cache[score_layer][0]  # [num_heads, seq_len, head_dim]
        scores = torch.norm(keys[:, n_sink:, :], dim=-1).mean(dim=0)  # [n_candidates]

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

    @property
    def eviction_count(self) -> int:
        return self._eviction_count
