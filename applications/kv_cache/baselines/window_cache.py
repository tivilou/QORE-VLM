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
        num_layers: Optional[int] = None,
    ):
        super().__init__()
        self.max_capacity = max_capacity
        self.trigger_every = trigger_every
        self.num_sink_tokens = num_sink_tokens
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
        # synchronized across every layer (not just layer 0). Decode-only
        # (query length == 1): evicting mid-prefill truncates KV while the last
        # layer's full-length-query attention still runs → shape mismatch.
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
