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


def fill_cache(cache, total_tokens, num_layers=2, batch=1, num_heads=4, head_dim=32):
    """Simulate filling a cache with total_tokens across num_layers."""
    # Simulate prefill: add all tokens at once to each layer
    keys = torch.randn(batch, num_heads, total_tokens, head_dim)
    values = torch.randn(batch, num_heads, total_tokens, head_dim)
    for layer_idx in range(num_layers):
        cache.update(keys.clone(), values.clone(), layer_idx)
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

    def test_pairwise_identical_keys_high_sim(self):
        """Identical keys should have similarity close to 1."""
        base = torch.randn(1, 32)
        keys = base.expand(4, 10, 32).clone()  # [4 heads, 10 identical, 32 dim]
        sim = pairwise_key_similarity(keys)
        # Off-diagonal should be ~1
        mask = ~torch.eye(10, dtype=torch.bool)
        assert sim[mask].mean() > 0.99


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
        # Fill with known values
        num_layers = 2
        keys = torch.randn(1, 4, 50, 32)
        values = torch.randn(1, 4, 50, 32)
        # Mark sink tokens with large norms so we can verify they're kept
        keys[0, :, :4, :] = 999.0

        for layer_idx in range(num_layers):
            cache.update(keys.clone(), values.clone(), layer_idx)

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
        cache = H2OCache(max_capacity=32, trigger_every=1)
        fill_cache(cache, total_tokens=64, num_layers=2)
        assert cache.eviction_count == 1
        assert cache.get_seq_length() <= 32

    def test_keeps_high_norm_tokens(self):
        """H2O should keep tokens with highest key norms."""
        cache = H2OCache(max_capacity=10, trigger_every=1, num_sink_tokens=2)
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


class TestRandomCache:
    """Test random baseline."""

    def test_eviction_triggered(self):
        cache = RandomCache(max_capacity=32, trigger_every=1)
        fill_cache(cache, total_tokens=64, num_layers=2)
        assert cache.eviction_count == 1
        assert cache.get_seq_length() <= 32

    def test_preserves_sinks(self):
        cache = RandomCache(max_capacity=10, trigger_every=1, num_sink_tokens=4)
        keys = torch.randn(1, 4, 30, 32)
        values = torch.randn(1, 4, 30, 32)
        keys[0, :, :4, :] = 999.0

        for layer_idx in range(2):
            cache.update(keys.clone(), values.clone(), layer_idx)

        retained_keys = cache.key_cache[0][0, 0, :4, 0]
        assert (retained_keys == 999.0).all()


class TestWindowCache:
    """Test sliding window baseline."""

    def test_eviction_triggered(self):
        cache = WindowCache(max_capacity=32, trigger_every=1)
        fill_cache(cache, total_tokens=64, num_layers=2)
        assert cache.eviction_count == 1
        assert cache.get_seq_length() <= 32

    def test_keeps_recent_and_sinks(self):
        """Window should keep first N sink + last M recent tokens."""
        cache = WindowCache(max_capacity=10, trigger_every=1, num_sink_tokens=2)
        keys = torch.arange(20).float().view(1, 1, 20, 1).expand(1, 4, 20, 32).clone()
        values = keys.clone()

        for layer_idx in range(2):
            cache.update(keys.clone(), values.clone(), layer_idx)

        # Should have sink (pos 0,1) + recent 8 (pos 12-19)
        seq_len = cache.get_seq_length()
        assert seq_len == 10
