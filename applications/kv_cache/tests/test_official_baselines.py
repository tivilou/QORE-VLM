"""
Tests for the official-paradigm KV-cache baselines (prefill-time compression):
faithful ports of H2O / SnapKV / PyramidKV.

Covers cluster compression (MHA + GQA), parity with plain generate when capacity
is not exceeded, and real compression via the PrefillCompressor monkeypatch.
"""

import torch
import pytest

from applications.kv_cache.official.clusters import (
    SnapKVCluster, H2OKVCluster, PyramidKVCluster,
)
from applications.kv_cache.official import PrefillCompressor, make_cluster_factory


def _qkv(bsz, hq, hkv, q_len, d):
    torch.manual_seed(0)
    q = torch.randn(bsz, hq, q_len, d)
    k = torch.randn(bsz, hkv, q_len, d)
    v = torch.randn(bsz, hkv, q_len, d)
    return q, k, v


class TestClustersMHA:
    """MHA (query heads == kv heads)."""

    @pytest.mark.parametrize("cluster", [
        SnapKVCluster(window_size=32, max_capacity_prompt=100),
        H2OKVCluster(window_size=32, max_capacity_prompt=100),
        PyramidKVCluster(num_hidden_layers=32, window_size=32,
                         max_capacity_prompt=100, layer_idx=10),
    ])
    def test_compresses_above_capacity(self, cluster):
        q, k, v = _qkv(1, 4, 4, 200, 64)
        kc, vc = cluster.update_kv(k, q, v)
        assert kc.shape == vc.shape
        assert kc.shape[1] == 4  # heads preserved
        assert kc.shape[2] < 200  # compressed

    def test_no_compress_below_capacity(self):
        q, k, v = _qkv(1, 4, 4, 80, 64)
        kc, vc = SnapKVCluster(window_size=32, max_capacity_prompt=1000).update_kv(k, q, v)
        assert kc.shape[2] == 80  # unchanged


class TestClustersGQA:
    """GQA (32 query heads, 8 kv heads, unrepeated K/V — Llama-3-8B layout)."""

    @pytest.mark.parametrize("cluster", [
        SnapKVCluster(window_size=32, max_capacity_prompt=100),
        H2OKVCluster(window_size=32, max_capacity_prompt=100),
        PyramidKVCluster(num_hidden_layers=32, window_size=32,
                         max_capacity_prompt=100, layer_idx=0),
    ])
    def test_keeps_kv_head_granularity(self, cluster):
        q, k, v = _qkv(1, 32, 8, 200, 64)
        kc, vc = cluster.update_kv(k, q, v)
        assert kc.shape[1] == 8, "must stay at unrepeated kv-head granularity"
        assert kc.shape == vc.shape
        assert kc.shape[2] < 200


def _tiny_llama(nkv=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=8,
                      num_key_value_heads=nkv, max_position_embeddings=1024,
                      attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval()


class TestPrefillCompressorEndToEnd:
    @pytest.mark.parametrize("method", ["snapkv", "h2o", "pyramidkv", "qore"])
    def test_parity_when_capacity_not_exceeded(self, method):
        # Huge capacity → compressor is inert → output must equal plain generate.
        from transformers import DynamicCache
        model = _tiny_llama()
        ids = torch.randint(0, 256, (1, 120))
        with torch.no_grad():
            ref = model.generate(ids, max_new_tokens=15, do_sample=False,
                                  use_cache=True)[0][120:].tolist()
        kw = dict(num_reads=12, seed=0) if method == "qore" else {}
        fac = make_cluster_factory(method, max_capacity=100000, window_size=16, **kw)
        with torch.no_grad(), PrefillCompressor(model, fac):
            out = model.generate(ids, max_new_tokens=15, do_sample=False,
                                  past_key_values=DynamicCache(), use_cache=True)[0][120:].tolist()
        assert out == ref

    @pytest.mark.parametrize("method", ["snapkv", "h2o", "pyramidkv", "qore"])
    def test_compresses_prefill(self, method):
        from transformers import DynamicCache
        model = _tiny_llama()
        ids = torch.randint(0, 256, (1, 150))
        kw = dict(num_reads=12, seed=0) if method == "qore" else {}
        fac = make_cluster_factory(method, max_capacity=64, window_size=16, **kw)
        cache = DynamicCache()
        with torch.no_grad(), PrefillCompressor(model, fac):
            model.generate(ids, max_new_tokens=15, do_sample=False,
                           past_key_values=cache, use_cache=True)
        lens = [cache.key_cache[i].shape[2] for i in range(4)]
        # Prefill (150) compressed well below prompt length + a few decode steps.
        assert all(L < 150 for L in lens), f"not compressed: {lens}"
        if method == "pyramidkv":
            # Pyramid: lower layers keep more than higher layers.
            assert lens[0] > lens[-1], f"expected pyramid, got {lens}"
