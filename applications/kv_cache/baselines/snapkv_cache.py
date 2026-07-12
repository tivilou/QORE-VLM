"""
SnapKV-style KV-cache eviction: keep tokens voted important by a recent
observation window, plus the window itself.

Reference: Li et al., "SnapKV: LLM Knows What You are Looking for Before
Generation" (2024).

Unlike H2O (cumulative attention over all queries), SnapKV scores prefix tokens
only by the attention they receive from the last `window` query positions — the
"observation window" — on the premise that the most recent context best predicts
which prefix tokens matter for the answer. The window tokens are always kept.
"""

import torch
from transformers.cache_utils import DynamicCache
from typing import Optional, Dict, Any, Tuple

from ..attention_accumulator import AttentionAccumulatorMixin, assert_single_batch


class SnapKVCache(AttentionAccumulatorMixin, DynamicCache):
    """
    SnapKV eviction. Greedy top-K like H2O, but the importance vote comes from a
    recent observation window rather than all past queries. Still ignores
    pairwise redundancy (that is QORE's differentiator).

    Note on the attention signal: the forward hooks accumulate attention summed
    over *all* current queries each pass. During the eviction pass the most
    recent `window` queries dominate the just-added scores, approximating the
    SnapKV observation window without a second attention read-out. The last
    `window` positions are always retained regardless of score.
    """

    def __init__(
        self,
        max_capacity: int = 1024,
        trigger_every: int = 128,
        num_sink_tokens: int = 4,
        num_layers: Optional[int] = None,
        window: int = 32,
        score_layer: int = 0,
    ):
        super().__init__()
        self.max_capacity = max_capacity
        self.trigger_every = trigger_every
        self.num_sink_tokens = num_sink_tokens
        self.window = window
        self.score_layer = score_layer
        self._init_attention_state()
        self._tokens_since_eviction = 0
        self._eviction_count = 0
        self._num_layers = num_layers
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

        # Detect real layer count on wrap-around (same as the other caches).
        if self._num_layers is None:
            if layer_idx <= self._last_layer_idx and self._last_layer_idx > 0:
                self._num_layers = self._last_layer_idx + 1
        self._last_layer_idx = layer_idx

        # Decode-only (query length == 1): evicting mid-prefill truncates KV
        # while the last layer's full-length-query attention still runs.
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
        """Keep sinks + observation window + top-scored prefix tokens."""
        assert_single_batch(self.key_cache, "SnapKVCache")
        seq_len = self.get_seq_length()
        n_sink = min(self.num_sink_tokens, seq_len)
        window = min(self.window, seq_len - n_sink)
        # Prefix candidates lie between the sinks and the observation window.
        prefix_end = seq_len - window
        n_keep = self.max_capacity - n_sink - window

        score_layer = min(self.score_layer, len(self.key_cache) - 1)
        keys = self.key_cache[score_layer][0]  # [heads, seq_len, head_dim]
        device = keys.device

        if n_keep <= 0:
            # Budget smaller than sinks+window: just keep sinks + most recent.
            keep_positions = torch.cat([
                torch.arange(n_sink, device=device),
                torch.arange(seq_len - (self.max_capacity - n_sink), seq_len, device=device),
            ])
        else:
            attn = self.attention_scores()
            if self.has_attention() and attn is not None and attn.shape[0] == seq_len:
                scores = attn[n_sink:prefix_end].to(device)
            else:
                scores = torch.norm(keys[:, n_sink:prefix_end, :], dim=-1).mean(dim=0)
            k = min(n_keep, scores.shape[0])
            _, top = torch.topk(scores, k)
            prefix_keep = top.sort().values + n_sink
            keep_positions = torch.cat([
                torch.arange(n_sink, device=device),
                prefix_keep,
                torch.arange(prefix_end, seq_len, device=device),  # window
            ])

        keep_positions = keep_positions.sort().values
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_positions, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_positions, :]

        self.prune_attention(keep_positions)

    @property
    def eviction_count(self) -> int:
        return self._eviction_count
