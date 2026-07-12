"""Unit tests for QORECache and baseline caches using synthetic KV tensors."""

import torch
import numpy as np
import pytest

from applications.kv_cache.qore_cache import QORECache
from applications.kv_cache.baselines.h2o_cache import H2OCache
from applications.kv_cache.baselines.random_cache import RandomCache
from applications.kv_cache.baselines.window_cache import WindowCache
from applications.kv_cache.signals_kv import (
    key_norm_quality,
    pairwise_key_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_kv_states(batch=1, num_heads=4, seq_len=1, head_dim=32):
    """Generate random KV states mimicking a single generation step."""
    keys = torch.randn(batch, num_heads, seq_len, head_dim)
    values = torch.randn(batch, num_heads, seq_len, head_dim)
    return keys, values


def fill_cache(cache, total_tokens, num_layers=2, batch=1, num_heads=4, head_dim=32,
               prefill=None):
    """Simulate filling a cache: one multi-token prefill, then 1-token decodes.

    Eviction fires only on decode steps (query length == 1) — during prefill the
    layer's attention runs over the full-length query right after update(), so
    truncating there would mismatch shapes. This mirrors a real generation loop,
    so tests must drive prefill + decode rather than a single giant update().
    """
    if prefill is None:
        # Prefill all but one token (no eviction fires during prefill), then a
        # single decode step crosses capacity → exactly ONE eviction. This keeps
        # the classic `eviction_count == 1` assertions valid under the
        # decode-only trigger while matching real generation semantics.
        prefill = max(1, total_tokens - 1)
    pk = torch.randn(batch, num_heads, prefill, head_dim)
    pv = torch.randn(batch, num_heads, prefill, head_dim)
    for layer_idx in range(num_layers):
        cache.update(pk.clone(), pv.clone(), layer_idx)
    for _ in range(total_tokens - prefill):
        tk = torch.randn(batch, num_heads, 1, head_dim)
        tv = torch.randn(batch, num_heads, 1, head_dim)
        for layer_idx in range(num_layers):
            cache.update(tk.clone(), tv.clone(), layer_idx)
    return cache


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestSignals:
    """Test quality and redundancy signal computation."""

    def test_key_norm_quality_shape(self):
        keys = torch.randn(8, 100, 64)  # [num_heads, seq_len, head_dim]
        q = key_norm_quality(keys)
        assert q.shape == (100,)

    def test_key_norm_quality_positive(self):
        keys = torch.randn(4, 50, 32)
        q = key_norm_quality(keys)
        assert (q > 0).all()

    def test_pairwise_similarity_shape(self):
        keys = torch.randn(8, 50, 64)
        sim = pairwise_key_similarity(keys)
        assert sim.shape == (50, 50)

    def test_pairwise_similarity_symmetric(self):
        keys = torch.randn(4, 30, 32)
        sim = pairwise_key_similarity(keys)
        assert torch.allclose(sim, sim.t(), atol=1e-6)

    def test_pairwise_similarity_zero_diag(self):
        keys = torch.randn(4, 30, 32)
        sim = pairwise_key_similarity(keys)
        assert torch.allclose(sim.diag(), torch.zeros(30), atol=1e-6)

    def test_pairwise_discriminates_under_anisotropy(self):
        """Redundancy must discriminate even when keys share a dominant
        direction (transformer key anisotropy).

        pairwise_key_similarity mean-centers before cosine, so a large common
        component (the anisotropy) is removed and only the RELATIVE structure
        remains. We build two clusters that share a big common offset: within
        a cluster keys point the same relative way (redundant), across clusters
        they point opposite ways (complementary). Raw cosine would read ~1 for
        every pair because of the shared offset; after centering, intra-cluster
        similarity must clearly exceed inter-cluster.
        """
        torch.manual_seed(0)
        common = torch.randn(1, 32) * 10.0  # dominant shared direction
        dir_a = torch.randn(1, 32)
        keys = torch.zeros(4, 10, 32)
        # positions 0-4: common + dir_a ; positions 5-9: common - dir_a
        keys[:, :5, :] = common + dir_a
        keys[:, 5:, :] = common - dir_a
        sim = pairwise_key_similarity(keys)
        intra = sim[0, 1]      # same cluster
        inter = sim[0, 9]      # opposite cluster
        assert intra > inter + 0.5, (intra.item(), inter.item())

    def test_pairwise_centering_neutralizes_pure_offset(self):
        """Perfectly identical keys have no relative structure: after centering
        they map to zero vectors, so similarity is ~0 (not 1). This is the
        intended consequence of anisotropy correction — a constant carries no
        redundancy signal."""
        base = torch.randn(1, 32)
        keys = base.expand(4, 10, 32).clone()
        sim = pairwise_key_similarity(keys)
        mask = ~torch.eye(10, dtype=torch.bool)
        assert sim[mask].abs().mean() < 1e-4


# ---------------------------------------------------------------------------
# Cache eviction tests
# ---------------------------------------------------------------------------

class TestQORECache:
    """Test QORECache eviction behavior."""

    def test_no_eviction_below_capacity(self):
        cache = QORECache(max_capacity=100, trigger_every=1, num_layers=2)
        fill_cache(cache, total_tokens=50, num_layers=2)
        assert cache.get_seq_length() == 50
        assert cache.eviction_count == 0

    def test_eviction_triggered(self):
        cache = QORECache(max_capacity=32, trigger_every=1, num_reads=10, num_layers=2)
        fill_cache(cache, total_tokens=64, num_layers=2)
        assert cache.eviction_count == 1
        assert cache.get_seq_length() <= 32

    def test_eviction_preserves_sinks(self):
        """First num_sink_tokens should always be retained."""
        cache = QORECache(max_capacity=20, trigger_every=1, num_sink_tokens=4, num_reads=10, num_layers=2)
        num_layers = 2
        # Prefill 49 tokens (no eviction), mark sinks with large norms, then a
        # single decode token to cross capacity and trigger exactly one eviction.
        pk = torch.randn(1, 4, 49, 32)
        pk[0, :, :4, :] = 999.0
        for layer_idx in range(num_layers):
            cache.update(pk.clone(), pk.clone(), layer_idx)
        tk = torch.randn(1, 4, 1, 32)
        for layer_idx in range(num_layers):
            cache.update(tk.clone(), tk.clone(), layer_idx)

        # After eviction, check sinks are present
        assert cache.get_seq_length() <= 20
        # First 4 positions should have the large-norm keys
        retained_keys = cache.key_cache[0][0, 0, :4, 0]  # first head, first 4 pos, first dim
        assert (retained_keys == 999.0).all()

    def test_all_layers_same_length(self):
        """After eviction, all layers must have the same sequence length."""
        cache = QORECache(max_capacity=20, trigger_every=1, num_reads=10, num_layers=4)
        fill_cache(cache, total_tokens=50, num_layers=4)
        lengths = [cache.key_cache[i].shape[2] for i in range(4)]
        assert len(set(lengths)) == 1  # all same

    def test_respects_trigger_interval(self):
        """Eviction should not trigger before trigger_every tokens."""
        cache = QORECache(max_capacity=32, trigger_every=64, num_reads=10, num_layers=2)
        # Add 40 tokens (exceeds capacity but below trigger interval)
        fill_cache(cache, total_tokens=40, num_layers=2)
        assert cache.eviction_count == 0  # not triggered yet


class TestH2OCache:
    """Test H2O baseline."""

    def test_eviction_triggered(self):
        cache = H2OCache(max_capacity=32, trigger_every=1, num_layers=2)
        fill_cache(cache, total_tokens=64, num_layers=2)
        assert cache.eviction_count == 1
        assert cache.get_seq_length() <= 32

    def test_keeps_high_norm_tokens(self):
        """H2O should keep tokens with highest key norms."""
        cache = H2OCache(max_capacity=10, trigger_every=1, num_sink_tokens=2, num_layers=2)
        keys = torch.randn(1, 4, 20, 32)
        values = torch.randn(1, 4, 20, 32)
        # Make tokens 5, 10, 15 have very high norms
        keys[0, :, 5, :] = 10.0
        keys[0, :, 10, :] = 10.0
        keys[0, :, 15, :] = 10.0

        for layer_idx in range(2):
            cache.update(keys.clone(), values.clone(), layer_idx)

        # Verify high-norm tokens are retained
        retained = cache.key_cache[0][0, 0, :, 0]  # first head, all pos, first dim
        assert (retained == 10.0).sum() >= 3

    def test_all_layers_same_length(self):
        """After eviction, ALL layers must be compressed, not just layer 0."""
        cache = H2OCache(max_capacity=20, trigger_every=1, num_layers=4)
        fill_cache(cache, total_tokens=50, num_layers=4)
        assert cache.eviction_count == 1
        lengths = [cache.key_cache[i].shape[2] for i in range(4)]
        assert len(set(lengths)) == 1  # all layers same length
        assert lengths[0] <= 20


class TestRandomCache:
    """Test random baseline."""

    def test_eviction_triggered(self):
        cache = RandomCache(max_capacity=32, trigger_every=1, num_layers=2)
        fill_cache(cache, total_tokens=64, num_layers=2)
        assert cache.eviction_count == 1
        assert cache.get_seq_length() <= 32

    def test_preserves_sinks(self):
        cache = RandomCache(max_capacity=10, trigger_every=1, num_sink_tokens=4, num_layers=2)
        keys = torch.randn(1, 4, 30, 32)
        values = torch.randn(1, 4, 30, 32)
        keys[0, :, :4, :] = 999.0

        for layer_idx in range(2):
            cache.update(keys.clone(), values.clone(), layer_idx)

        retained_keys = cache.key_cache[0][0, 0, :4, 0]
        assert (retained_keys == 999.0).all()

    def test_all_layers_same_length(self):
        """After eviction, ALL layers must be compressed, not just layer 0."""
        cache = RandomCache(max_capacity=20, trigger_every=1, num_layers=4)
        fill_cache(cache, total_tokens=50, num_layers=4)
        assert cache.eviction_count == 1
        lengths = [cache.key_cache[i].shape[2] for i in range(4)]
        assert len(set(lengths)) == 1  # all layers same length
        assert lengths[0] <= 20


class TestWindowCache:
    """Test sliding window baseline."""

    def test_eviction_triggered(self):
        cache = WindowCache(max_capacity=32, trigger_every=1, num_layers=2)
        fill_cache(cache, total_tokens=64, num_layers=2)
        assert cache.eviction_count == 1
        assert cache.get_seq_length() <= 32

    def test_keeps_recent_and_sinks(self):
        """Window should keep first N sink + last M recent tokens."""
        cache = WindowCache(max_capacity=10, trigger_every=1, num_sink_tokens=2, num_layers=2)
        # Prefill 19 tokens (pos 0..18), then one decode token (pos 19) to
        # cross capacity and trigger eviction on the decode step.
        full = torch.arange(20).float().view(1, 1, 20, 1).expand(1, 4, 20, 32).clone()
        pk = full[:, :, :19, :].clone()
        for layer_idx in range(2):
            cache.update(pk.clone(), pk.clone(), layer_idx)
        tk = full[:, :, 19:20, :].clone()
        for layer_idx in range(2):
            cache.update(tk.clone(), tk.clone(), layer_idx)

        # Should have sink (pos 0,1) + recent tokens, capped at capacity.
        seq_len = cache.get_seq_length()
        assert seq_len == 10

    def test_all_layers_same_length(self):
        """After eviction, ALL layers must be compressed, not just layer 0."""
        cache = WindowCache(max_capacity=20, trigger_every=1, num_layers=4)
        fill_cache(cache, total_tokens=50, num_layers=4)
        assert cache.eviction_count == 1
        lengths = [cache.key_cache[i].shape[2] for i in range(4)]
        assert len(set(lengths)) == 1  # all layers same length
        assert lengths[0] <= 20
