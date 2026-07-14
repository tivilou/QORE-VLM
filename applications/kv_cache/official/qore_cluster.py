"""
QOREKVCluster: QORE selection in the official prefill-compression paradigm.

Same `update_kv(key, query, value)` contract as the official H2O/SnapKV/PyramidKV
clusters, so QORE competes apples-to-apples under the identical protocol:
  - compress ONCE at end of prefill,
  - keep the whole recent `window_size` block,
  - select `max_capacity - window_size` prefix tokens.

The ONLY difference from the baselines is HOW those prefix tokens are chosen:
baselines take greedy top-K on an attention score; QORE solves a QUBO that
balances per-token quality (aᵢ) against pairwise key redundancy (bᵢⱼ), so it
can avoid keeping two near-duplicate keys. This isolates "selection algorithm"
as the single variable — the cleanest possible comparison for the paper.

Quality signal aᵢ: attention from the observation window (same window scoring as
SnapKV) so the quality basis matches the baselines; redundancy bᵢⱼ: cosine of
mean-centered key vectors (anisotropy-corrected), computed per block to avoid the
O(N²) matrix. Selection is per kv-head-averaged (single QUBO over positions),
then applied to all heads — consistent with our decode-time QORECache.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from qore import solve as qore_solve
from qore.signals import normalize
from ..signals_kv import pairwise_key_similarity
from .clusters import repeat_kv, _gqa_groups


class QOREKVCluster:
    """QORE (QUBO) selection under the official prefill-compression protocol."""

    def __init__(self, window_size=32, max_capacity_prompt=1024,
                 redundancy_method="cosine", solver_method="anneal",
                 num_reads=30, lam=2.0, block_size=32, seed=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.redundancy_method = redundancy_method
        self.solver_method = solver_method
        self.num_reads = num_reads
        self.lam = lam
        self.block_size = block_size
        self.seed = seed

    # --- signal construction -------------------------------------------------

    def _window_quality(self, query_states, key_states, groups):
        """Per-prefix-key quality = attention from the observation window.

        Mirrors SnapKV's window scoring (last window_size queries), reduced to a
        single score per prefix position (mean over heads). Returns a 1-D numpy
        array of length (kv_len - window_size).
        """
        head_dim = query_states.shape[-1]
        if groups > 1:
            key_rep = repeat_kv(key_states, groups)
        else:
            key_rep = key_states
        attn = torch.matmul(
            query_states[..., -self.window_size:, :], key_rep.transpose(2, 3)
        ) / math.sqrt(head_dim)
        w = self.window_size
        mask = torch.full((w, w), torch.finfo(attn.dtype).min, device=attn.device)
        mc = torch.arange(w, device=attn.device)
        mask.masked_fill_(mc < (mc + 1).view(w, 1), 0)
        attn[:, :, -w:, -w:] += mask[None, None, :, :]
        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32)
        # sum over window queries -> [b, heads, prefix]; mean over heads & batch0
        score = attn[:, :, -w:, :-w].sum(dim=-2)[0].mean(dim=0)  # [prefix]
        return normalize(score.cpu().numpy().astype(np.float64))

    def _redundancy(self, keys_2d):
        """bᵢⱼ over the given [heads, n, dim] key slice (anisotropy-corrected cosine)."""
        b = pairwise_key_similarity(keys_2d, method=self.redundancy_method, max_heads=4)
        return b.cpu().numpy().astype(np.float64)

    # --- QUBO selection over prefix positions --------------------------------

    def _select_prefix(self, a, keys_prefix, n_keep):
        """Pick n_keep prefix indices via QUBO (block-decomposed, lazy redundancy)."""
        from qore.block_decompose import decompose, recompose
        N = len(a)
        if n_keep >= N:
            return np.arange(N)
        num_blocks = max(2, -(-N // self.block_size))
        blocks = decompose(a, None, n_keep, num_blocks=num_blocks)
        sols, idxs = [], []
        for a_b, _b, k_b, bi in blocks:
            bn = len(bi)
            if k_b >= bn:
                sols.append(np.ones(bn, dtype=np.int32)); idxs.append(bi); continue
            b_b = self._redundancy(keys_prefix[:, bi, :])
            kw = {"num_reads": self.num_reads}
            if self.seed is not None:
                kw["seed"] = self.seed
            x = qore_solve(a_b, b_b, k_b, lam=self.lam, method=self.solver_method, **kw)
            sols.append(x); idxs.append(bi)
        x_global = recompose(sols, idxs, N)
        return np.where(x_global == 1)[0]

    # --- official-paradigm entry point ---------------------------------------

    def update_kv(self, key_states, query_states, value_states,
                  attention_mask=None, num_key_value_groups=1):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_q_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        groups = _gqa_groups(query_states, key_states)
        n_prefix = q_len - self.window_size
        n_keep = self.max_capacity_prompt - self.window_size

        # Quality over prefix positions (attention from the window).
        a = self._window_quality(query_states, key_states, groups)  # [n_prefix]
        # Key vectors for redundancy: mean over kv-heads' prefix keys.
        keys_prefix = key_states[0, :, :n_prefix, :]  # [kv_heads, n_prefix, dim]

        keep_prefix = self._select_prefix(a, keys_prefix, n_keep)  # indices into prefix
        keep_prefix = np.sort(keep_prefix)
        idx = torch.tensor(keep_prefix, dtype=torch.long, device=key_states.device)
        idx = idx.view(1, 1, -1, 1).expand(bsz, key_states.shape[1], -1, head_dim)

        k_past = key_states[:, :, :n_prefix, :].gather(dim=2, index=idx)
        v_past = value_states[:, :, :n_prefix, :].gather(dim=2, index=idx)
        k_cur = key_states[:, :, n_prefix:, :]
        v_cur = value_states[:, :, n_prefix:, :]
        return torch.cat([k_past, k_cur], dim=2), torch.cat([v_past, v_cur], dim=2)
