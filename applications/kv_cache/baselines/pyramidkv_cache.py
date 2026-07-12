"""
PyramidKV-style KV-cache eviction: layer-wise budget allocation.

Reference: Cai et al., "PyramidKV: Dynamic KV Cache Compression based on
Pyramidal Information Funneling" (2024).

Key idea: lower layers attend broadly and need more cache; higher layers focus
on a few tokens and need less. PyramidKV therefore gives each layer a *different*
budget — largest at the bottom, smallest at the top — and evicts each layer
INDEPENDENTLY to its own budget by that layer's own attention.

This intentionally breaks the "all layers same length" invariant that the
synchronized caches (QORE, H2O, Window, Random) maintain — different layers end
up with different sequence lengths by design. So this class keeps its own
per-layer attention accumulator instead of the shared mixin.
"""

import torch
from transformers.cache_utils import DynamicCache
from typing import Optional, Dict, Any, Tuple, List

from ..attention_accumulator import assert_single_batch


class PyramidKVCache(DynamicCache):
    """PyramidKV eviction with per-layer budgets and per-layer attention."""

    def __init__(
        self,
        max_capacity: int = 1024,
        trigger_every: int = 128,
        num_sink_tokens: int = 4,
        num_layers: Optional[int] = None,
        beta: float = 0.5,
    ):
        """
        Args:
            max_capacity: AVERAGE per-layer budget. Total budget is
                ``max_capacity * num_layers``; it is distributed as a pyramid so
                the mean per-layer budget equals ``max_capacity`` (comparable to
                the other policies at the same nominal capacity).
            beta: Bottom/top spread. Layer budgets vary linearly from
                ``(2-beta)*max_capacity`` at layer 0 down to ``beta*max_capacity``
                at the last layer (mean = max_capacity). beta in (0, 1].
        """
        super().__init__()
        # Layers evict to DIFFERENT budgets, so per-layer KV lengths diverge.
        # The decode loop uses this flag to enable per-layer RoPE re-basing
        # (PerLayerPositionPatch) instead of the shared-position path.
        self.per_layer_uneven = True
        self.max_capacity = max_capacity
        self.trigger_every = trigger_every
        self.num_sink_tokens = num_sink_tokens
        self.beta = beta
        self._num_layers = num_layers
        self._last_layer_idx = -1
        # Per-layer cumulative attention scores.
        self._attn_by_layer: Dict[int, torch.Tensor] = {}
        # Per-layer token counter since that layer last evicted.
        self._tokens_since: Dict[int, int] = {}
        self._eviction_count = 0

    def add_attention(self, layer_idx: int, scores: torch.Tensor):
        """Accumulate per-layer attention (called by the forward hooks)."""
        scores = scores.detach().to(torch.float32)
        prev = self._attn_by_layer.get(layer_idx)
        n = scores.shape[0]
        if prev is None:
            self._attn_by_layer[layer_idx] = scores.clone()
        elif prev.shape[0] == n:
            self._attn_by_layer[layer_idx] = prev + scores
        elif prev.shape[0] < n:
            grown = torch.zeros(n, dtype=torch.float32, device=scores.device)
            grown[: prev.shape[0]] = prev.to(scores.device)
            self._attn_by_layer[layer_idx] = grown + scores
        else:
            prev[:n] += scores

    def has_attention(self) -> bool:
        return len(self._attn_by_layer) > 0

    def layer_budget(self, layer_idx: int) -> int:
        """Pyramid budget for a layer: linear from (2-beta)C at 0 to beta*C at top."""
        L = self._num_layers
        if L is None or L <= 1:
            return self.max_capacity
        top = (2.0 - self.beta) * self.max_capacity
        bot = self.beta * self.max_capacity
        frac = layer_idx / (L - 1)
        return max(self.num_sink_tokens + 1, int(round(top - (top - bot) * frac)))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)

        self._tokens_since[layer_idx] = self._tokens_since.get(layer_idx, 0) + key_states.shape[-2]

        if self._num_layers is None:
            if layer_idx <= self._last_layer_idx and self._last_layer_idx > 0:
                self._num_layers = self._last_layer_idx + 1
        self._last_layer_idx = layer_idx

        # Per-layer independent eviction (no cross-layer synchronization).
        # Decode-only (query length == 1): during prefill this layer's attention
        # runs over its full-length query right after update() returns, so
        # truncating its KV here would mismatch (q=prompt_len, kv=budget).
        is_decode_step = key_states.shape[-2] == 1
        if self._num_layers is not None and is_decode_step:
            budget = self.layer_budget(layer_idx)
            layer_len = self.key_cache[layer_idx].shape[2]
            if layer_len > budget and self._tokens_since[layer_idx] >= self.trigger_every:
                # Return pre-eviction tensors for this layer's attention; the
                # truncation takes effect on the next forward.
                keys_ret = self.key_cache[layer_idx]
                values_ret = self.value_cache[layer_idx]
                self._evict_layer(layer_idx, budget)
                self._tokens_since[layer_idx] = 0
                self._eviction_count += 1
                return keys_ret, values_ret

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _evict_layer(self, layer_idx: int, budget: int):
        """Evict a single layer down to its pyramid budget by that layer's attention."""
        assert_single_batch(self.key_cache, "PyramidKVCache")
        layer_len = self.key_cache[layer_idx].shape[2]
        n_sink = min(self.num_sink_tokens, layer_len)
        n_keep = budget - n_sink
        if n_keep <= 0:
            return

        keys = self.key_cache[layer_idx][0]  # [heads, layer_len, head_dim]
        device = keys.device

        attn = self._attn_by_layer.get(layer_idx)
        if attn is not None and attn.shape[0] == layer_len:
            scores = attn[n_sink:].to(device)
        else:
            scores = torch.norm(keys[:, n_sink:, :], dim=-1).mean(dim=0)

        k = min(n_keep, scores.shape[0])
        _, top = torch.topk(scores, k)
        keep_positions = torch.cat([
            torch.arange(n_sink, device=device),
            top.sort().values + n_sink,
        ]).sort().values

        self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_positions, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_positions, :]
        if attn is not None:
            idx = keep_positions.to(attn.device).long()
            idx = idx[idx < attn.shape[0]]
            self._attn_by_layer[layer_idx] = attn[idx]

    @property
    def eviction_count(self) -> int:
        return self._eviction_count
