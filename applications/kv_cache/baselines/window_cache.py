"""Sliding window eviction baseline: keeps sink tokens + most recent tokens."""

import torch
from transformers.cache_utils import DynamicCache
from typing import Optional, Dict, Any, Tuple


class WindowCache(DynamicCache):
    """
    KV cache with sliding window eviction. Keeps the first `num_sink_tokens`
    and the most recent `max_capacity - num_sink_tokens` entries.
    Equivalent logic to HF's SinkCache but using our eviction interface.
    """

    def __init__(
        self,
        max_capacity: int = 1024,
        trigger_every: int = 128,
        num_sink_tokens: int = 4,
    ):
        super().__init__()
        self.max_capacity = max_capacity
        self.trigger_every = trigger_every
        self.num_sink_tokens = num_sink_tokens
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
        """Window eviction: keep sinks + most recent tokens."""
        seq_len = self.get_seq_length()
        n_sink = min(self.num_sink_tokens, seq_len)
        n_recent = self.max_capacity - n_sink

        # Keep first n_sink + last n_recent
        keep_positions = torch.cat([
            torch.arange(n_sink, device=self.key_cache[0].device),
            torch.arange(seq_len - n_recent, seq_len, device=self.key_cache[0].device),
        ])

        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_positions, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_positions, :]

    @property
    def eviction_count(self) -> int:
        return self._eviction_count
