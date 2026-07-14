"""
Faithful ports of the official KV-cache compression clusters.

These are the PREFILL-TIME COMPRESSION paradigm baselines (H2O / SnapKV /
PyramidKV), ported line-by-line from the official implementations so our
comparison uses genuine reproductions rather than approximations:

  - SnapKV   : FasterDecoding/SnapKV  (snapkv/monkeypatch/snapkv_utils.py)
  - H2O      : as unified in Zefan-Cai/KVCache-Factory (pyramidkv/pyramidkv_utils.py)
  - PyramidKV: Zefan-Cai/KVCache-Factory (pyramidkv/pyramidkv_utils.py)

Reference copies of the sources live in reference/official_kv_sources/ (git-ignored).

Paradigm (all three share it, differs fundamentally from our decode-time eviction):
  1. Compression happens ONCE, at the end of prefill, inside the attention forward
     (it needs query_states to score keys — a plain cache.update() can't do this).
  2. `update_kv(key, query, value, ...)` scores prefix keys, keeps the top-K PER HEAD
     plus the ENTIRE recent `window_size` block, and returns the compressed K/V.
  3. During decode the cache just appends; no further eviction, no RoPE re-rotation
     (retained keys keep their original absolute-position phase, new query uses the
     logical growing position, so relative geometry stays correct).

Scoring is the only real difference between the three:
  - H2O      : attention from ALL queries, summed over query dim, no pooling.
  - SnapKV   : attention from the last `window_size` queries only, then pooled.
  - PyramidKV: SnapKV-style scoring but a per-layer pyramidal budget.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- GQA helpers (ported verbatim from pyramidkv_utils.py) -------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, kv_heads, seq, dim) -> (batch, kv_heads*n_rep, seq, dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _gqa_groups(query_states, key_states):
    """query-heads-per-kv-head; 1 for MHA / pre-repeated tensors."""
    num_query_heads = query_states.shape[1]
    num_kv_heads = key_states.shape[1]
    assert num_query_heads % num_kv_heads == 0, (
        f"query heads {num_query_heads} not divisible by kv heads {num_kv_heads}")
    return num_query_heads // num_kv_heads


def _reduce_group_scores(scores, groups, gqa_score_agg):
    """Reduce (bsz, num_query_heads, ...) scores to kv-head granularity."""
    if groups == 1:
        return scores
    bsz, num_heads = scores.shape[0], scores.shape[1]
    grouped = scores.reshape(bsz, num_heads // groups, groups, *scores.shape[2:])
    if gqa_score_agg == 'mean':
        return grouped.mean(dim=2)
    elif gqa_score_agg == 'max':
        return grouped.amax(dim=2)
    elif gqa_score_agg == 'sum':
        return grouped.sum(dim=2)
    raise ValueError(f"GQA score aggregation {gqa_score_agg!r} not supported")


def _grouped_window_attn_cache(query_states, key_states, groups, window_size,
                               kernel_size, pooling, gqa_score_agg):
    """SnapKV-style observation-window scores at kv-head granularity (GQA path)."""
    head_dim = query_states.shape[-1]
    key_rep = repeat_kv(key_states, groups)
    attn_weights = torch.matmul(
        query_states[..., -window_size:, :], key_rep.transpose(2, 3)
    ) / math.sqrt(head_dim)
    mask = torch.full((window_size, window_size),
                      torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    attn_weights[:, :, -window_size:, -window_size:] += mask[None, None, :, :]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim=-2)
    attn_weights_sum = _reduce_group_scores(attn_weights_sum, groups, gqa_score_agg)
    if pooling == 'avgpool':
        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=kernel_size,
                                  padding=kernel_size // 2, stride=1)
    elif pooling == 'maxpool':
        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=kernel_size,
                                  padding=kernel_size // 2, stride=1)
    else:
        raise ValueError('Pooling method not supported')
    return attn_cache


def _select_topk_kv(key_states, value_states, attn_cache, capacity, window_size):
    """Top-k select past tokens per head + keep the whole observation window."""
    head_dim = key_states.shape[-1]
    indices = attn_cache.topk(capacity, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_past = key_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    v_past = value_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    k_cur = key_states[:, :, -window_size:, :]
    v_cur = value_states[:, :, -window_size:, :]
    return torch.cat([k_past, k_cur], dim=2), torch.cat([v_past, v_cur], dim=2)


# --- SnapKV cluster (ported from FasterDecoding/SnapKV) ----------------------

class SnapKVCluster:
    """Official SnapKV: observation-window attention scoring + pooling."""

    def __init__(self, window_size=32, max_capacity_prompt=1024, kernel_size=5,
                 pooling='avgpool', gqa_score_agg='mean'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.gqa_score_agg = gqa_score_agg

    def update_kv(self, key_states, query_states, value_states,
                  attention_mask=None, num_key_value_groups=1):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        groups = _gqa_groups(query_states, key_states)
        if groups > 1:
            if q_len < self.max_capacity_prompt:
                return key_states, value_states
            attn_cache = _grouped_window_attn_cache(
                query_states, key_states, groups, self.window_size,
                self.kernel_size, self.pooling, self.gqa_score_agg)
            return _select_topk_kv(key_states, value_states, attn_cache,
                                   self.max_capacity_prompt - self.window_size,
                                   self.window_size)

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        # Score prefix keys by attention from the last window_size queries only.
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :], key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        mask = torch.full((self.window_size, self.window_size),
                          torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -self.window_size:, -self.window_size:] += mask[None, None, :, :]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                      padding=self.kernel_size // 2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                      padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return _select_topk_kv(key_states, value_states, attn_cache,
                               self.max_capacity_prompt - self.window_size,
                               self.window_size)


# --- H2O cluster (ported from KVCache-Factory) -------------------------------

class H2OKVCluster:
    """Official H2O: cumulative attention from ALL queries, no pooling."""

    def __init__(self, window_size=32, max_capacity_prompt=1024, gqa_score_agg='mean'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.gqa_score_agg = gqa_score_agg

    def update_kv(self, key_states, query_states, value_states,
                  attention_mask=None, num_key_value_groups=1):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        groups = _gqa_groups(query_states, key_states)
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        if groups > 1:
            key_rep = repeat_kv(key_states, groups)
        else:
            key_rep = key_states
        # Score every prefix key by attention summed over ALL queries (heavy hitter).
        attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights += mask[None, None, :, :]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim=-2)  # no pooling
        attn_cache = _reduce_group_scores(attn_weights_sum, groups, self.gqa_score_agg)
        return _select_topk_kv(key_states, value_states, attn_cache,
                               self.max_capacity_prompt - self.window_size,
                               self.window_size)


# --- PyramidKV cluster (ported from KVCache-Factory) -------------------------

class PyramidKVCluster:
    """Official PyramidKV: SnapKV-style scoring + per-layer pyramidal budget.

    Lower layers get a larger budget, higher layers smaller, controlled by beta.
    The per-layer capacity schedule is computed from layer_idx / num_hidden_layers.
    """

    def __init__(self, num_hidden_layers=32, window_size=32, max_capacity_prompt=1024,
                 kernel_size=5, pooling='avgpool', beta=20, layer_idx=None,
                 gqa_score_agg='mean'):
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.beta = beta
        self.layer_idx = layer_idx
        self.gqa_score_agg = gqa_score_agg

    def _layer_capacity(self, q_len):
        """Pyramidal per-layer capacity (ported from official update_kv)."""
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        return max_num - self.layer_idx * steps

    def update_kv(self, key_states, query_states, value_states,
                  attention_mask=None, num_key_value_groups=1):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        groups = _gqa_groups(query_states, key_states)

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        # Per-layer capacity: mid-range uses fixed (max_cap - window), large context
        # uses the pyramidal schedule (matches official branch structure).
        if q_len < (self.max_capacity_prompt - self.window_size) * 2:
            capacity = self.max_capacity_prompt - self.window_size
        else:
            capacity = self._layer_capacity(q_len)

        if groups > 1:
            attn_cache = _grouped_window_attn_cache(
                query_states, key_states, groups, self.window_size,
                self.kernel_size, self.pooling, self.gqa_score_agg)
            return _select_topk_kv(key_states, value_states, attn_cache,
                                   capacity, self.window_size)

        # MHA path: SnapKV-style window scoring.
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :], key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        mask = torch.full((self.window_size, self.window_size),
                          torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -self.window_size:, -self.window_size:] += mask[None, None, :, :]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                      padding=self.kernel_size // 2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                      padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return _select_topk_kv(key_states, value_states, attn_cache,
                               capacity, self.window_size)
