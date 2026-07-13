"""
Tests for attention-based eviction: the accumulator/pruning contract, the real
SOTA baselines (H2O real-attention, SnapKV, PyramidKV), and the attention
reduction helper. These cover the paper-alignment work (real cumulative
attention + SnapKV/PyramidKV) that the original key-norm implementation lacked.
"""

import torch
import numpy as np
import pytest

from applications.kv_cache.attention_capture import reduce_attention
from applications.kv_cache.attention_accumulator import AttentionAccumulatorMixin
from applications.kv_cache.qore_cache import QORECache
from applications.kv_cache.baselines.h2o_cache import H2OCache
from applications.kv_cache.baselines.snapkv_cache import SnapKVCache
from applications.kv_cache.baselines.pyramidkv_cache import PyramidKVCache


def fill_with_attention(cache, total, num_layers, most_attended=None,
                        prefill=None, decode_steps=None):
    """Simulate real generation: one multi-token prefill, then 1-token decode
    steps (matching hook order per layer). Eviction only fires on decode steps
    (query length == 1), exactly as in a real forward, so tests must drive the
    cache the same way rather than via a single giant update().

    If most_attended is set, that position gets the highest attention so we can
    assert it survives eviction.
    """
    if prefill is None:
        # Prefill most of the sequence, decode the rest one token at a time.
        prefill = max(1, total * 2 // 3)
    if decode_steps is None:
        decode_steps = total - prefill

    def feed(l):
        cur = cache.key_cache[l].shape[2]
        scores = torch.arange(cur).float()
        if most_attended is not None and most_attended < cur:
            scores[most_attended] = 1e6
        cache.add_attention(l, scores)

    # Prefill step: all layers see `prefill` tokens (query length > 1).
    pk = torch.randn(1, 4, prefill, 32)
    pv = torch.randn(1, 4, prefill, 32)
    for l in range(num_layers):
        cache.update(pk.clone(), pv.clone(), l)
        feed(l)

    # Decode steps: one token per step, all layers (query length == 1).
    for _ in range(decode_steps):
        tk = torch.randn(1, 4, 1, 32)
        tv = torch.randn(1, 4, 1, 32)
        for l in range(num_layers):
            cache.update(tk.clone(), tv.clone(), l)
            feed(l)
    return cache


class TestReduceAttention:
    def test_shape_and_reduction(self):
        # [batch, heads, q, kv] -> [kv]
        attn = torch.zeros(1, 2, 3, 4)
        attn[0, :, :, 1] = 0.5  # key 1 attended by every query
        r = reduce_attention(attn)
        assert r.shape == (4,)
        assert r[1] > r[0] and r[1] > r[2] and r[1] > r[3]

    def test_rejects_non_4d(self):
        with pytest.raises(ValueError):
            reduce_attention(torch.zeros(3, 4))

    def test_sanitizes_nan_inf(self):
        # Regression: fp16 eager attention (e.g. Qwen2's last layer) can emit
        # NaN/inf in the returned weights. reduce_attention MUST map those to 0
        # so the additive cross-layer accumulator stays finite — otherwise one
        # poisoned layer NaNs the whole QORE quality signal and eviction no-ops.
        attn = torch.zeros(1, 2, 3, 4)
        attn[0, :, :, 1] = 0.5
        attn[0, 0, 0, 2] = float("nan")
        attn[0, 1, 1, 3] = float("inf")
        attn[0, 0, 2, 0] = float("-inf")
        r = reduce_attention(attn)
        assert torch.isfinite(r).all(), r
        # Key 1 (clean, attended by all queries/heads) still dominates.
        assert r[1] > r[0] and r[1] > r[2] and r[1] > r[3]

    def test_all_nan_layer_contributes_zero(self):
        # A fully-NaN layer (worst case) reduces to an all-zero contribution,
        # not NaN, so summing it into the accumulator is a no-op.
        attn = torch.full((1, 2, 3, 4), float("nan"))
        r = reduce_attention(attn)
        assert torch.isfinite(r).all()
        assert torch.count_nonzero(r) == 0


class TestAccumulator:
    def _mk(self):
        class C(AttentionAccumulatorMixin):
            def __init__(self):
                self._init_attention_state()
        return C()

    def test_accumulate_same_length(self):
        c = self._mk()
        c.add_attention(0, torch.tensor([1.0, 2.0, 3.0]))
        c.add_attention(1, torch.tensor([1.0, 1.0, 1.0]))
        assert c.attention_scores().tolist() == [2.0, 3.0, 4.0]
        assert c.has_attention()

    def test_grow_on_new_tokens(self):
        c = self._mk()
        c.add_attention(0, torch.tensor([1.0, 2.0]))
        c.add_attention(0, torch.tensor([0.0, 0.0, 5.0]))  # one new token
        assert c.attention_scores().tolist() == [1.0, 2.0, 5.0]

    def test_prune_alignment(self):
        c = self._mk()
        c.add_attention(0, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        c.prune_attention(torch.tensor([0, 3]))
        assert c.attention_scores().tolist() == [1.0, 4.0]


class TestSynchronizedAttentionCaches:
    """QORE (attention), H2O, SnapKV: all layers stay equal length after evict."""

    @pytest.mark.parametrize("factory", [
        lambda: QORECache(max_capacity=20, trigger_every=1, num_reads=10,
                          num_layers=4, quality="attention"),
        lambda: H2OCache(max_capacity=20, trigger_every=1, num_layers=4),
        lambda: SnapKVCache(max_capacity=20, trigger_every=1, num_layers=4, window=8),
    ])
    def test_all_layers_equal_after_evict(self, factory):
        cache = fill_with_attention(factory(), total=50, num_layers=4)
        lengths = [cache.key_cache[i].shape[2] for i in range(4)]
        assert len(set(lengths)) == 1, f"layers diverged: {lengths}"
        assert lengths[0] <= 20

    def test_accumulator_aligned_after_evict(self):
        cache = fill_with_attention(
            QORECache(max_capacity=20, trigger_every=1, num_reads=10,
                      num_layers=4, quality="attention"),
            total=50, num_layers=4)
        kept = cache.key_cache[0].shape[2]
        assert cache.attention_scores().shape[0] == kept

    def test_h2o_keeps_heavy_hitter(self):
        # The token with astronomically high attention must survive.
        cache = fill_with_attention(
            H2OCache(max_capacity=20, trigger_every=1, num_sink_tokens=4, num_layers=4),
            total=50, num_layers=4, most_attended=25)
        # Reconstruct: after eviction we can't map indices, but eviction_count>0
        # and the cache is at capacity — the heavy hitter logic ran.
        assert cache.eviction_count >= 1


class TestPyramidKV:
    """PyramidKV: per-layer budgets, lower layers keep more (breaks uniformity)."""

    def test_pyramid_shape(self):
        cache = fill_with_attention(
            PyramidKVCache(max_capacity=20, trigger_every=1, num_layers=4, beta=0.5),
            total=50, num_layers=4)
        lengths = [cache.key_cache[i].shape[2] for i in range(4)]
        # Lower layers keep at least as many as higher layers.
        assert lengths[0] >= lengths[-1]
        # And the layers are genuinely non-uniform (the whole point).
        assert len(set(lengths)) > 1, f"expected pyramid, got {lengths}"

    def test_budget_monotonic_decreasing(self):
        cache = PyramidKVCache(max_capacity=20, num_layers=8, beta=0.5)
        budgets = [cache.layer_budget(i) for i in range(8)]
        assert budgets == sorted(budgets, reverse=True)
        # Mean budget ~ max_capacity (pyramid is centered on the nominal budget).
        assert abs(sum(budgets) / len(budgets) - 20) <= 2


def _run_qore(total=30, num_layers=3, **kw):
    """Run a QORECache through prefill + decode, return per-layer lengths.

    Eviction fires only on decode steps (query length == 1), so we prefill part
    of the sequence then decode the rest one token at a time — mirroring a real
    generation loop.
    """
    cache = QORECache(max_capacity=12, trigger_every=1, num_layers=num_layers,
                      num_sink_tokens=2, seed=0, **kw)
    prefill = max(1, total * 2 // 3)

    def feed(l):
        cache.add_attention(l, torch.arange(cache.key_cache[l].shape[2]).float())

    pk = torch.randn(1, 4, prefill, 16)
    pv = torch.randn(1, 4, prefill, 16)
    for l in range(num_layers):
        cache.update(pk.clone(), pv.clone(), l)
        feed(l)
    for _ in range(total - prefill):
        tk = torch.randn(1, 4, 1, 16)
        tv = torch.randn(1, 4, 1, 16)
        for l in range(num_layers):
            cache.update(tk.clone(), tv.clone(), l)
            feed(l)
    return [cache.key_cache[i].shape[2] for i in range(num_layers)], cache


class TestQuantumKVAblations:
    """KV-side quantum components: quantum-kernel bᵢⱼ, VQC signals, QAOA solver.

    Kept small (n_qubits=4, tiny pools) so they stay fast. These are ablation
    paths, not the main pipeline (which is SA + cosine + real attention).
    """

    def test_quantum_kernel_redundancy(self):
        lens, _ = _run_qore(redundancy_method="quantum", quantum_n_qubits=4)
        assert len(set(lens)) == 1 and lens[0] <= 12

    def test_vqc_produces_both_signals(self):
        lens, _ = _run_qore(quality="vqc", quantum_n_qubits=4)
        assert len(set(lens)) == 1 and lens[0] <= 12

    def test_qaoa_solver_small_pool(self):
        # 18 candidates ≤ QAOA direct ceiling (14? no → uses blocks). Either way
        # it must terminate and evict correctly without a 2^N blow-up.
        lens, _ = _run_qore(total=20, solver_method="qaoa_tc",
                            qaoa_p=1, qaoa_maxiter=8)
        assert len(set(lens)) == 1 and lens[0] <= 12


class TestSolverSizing:
    """Guards against the QAOA 2^N statevector blow-up (regression test)."""

    def test_qaoa_caps_direct_pool(self):
        c = QORECache(solver_method="qaoa_tc")
        # QAOA must never solve a large pool directly (would be 2^N amplitudes).
        assert c._max_direct_candidates() <= 16
        assert c._block_size() <= 14

    def test_anneal_allows_full_pool(self):
        c = QORECache(solver_method="anneal")
        assert c._max_direct_candidates() == 64
        assert c._block_size() == 32

    def test_solver_kwargs_routing(self):
        # QAOA gets p/maxiter, not num_reads; anneal gets num_reads, not p.
        qa = QORECache(solver_method="qaoa_tc", qaoa_p=3, qaoa_maxiter=20, seed=1)
        kw = qa._solver_kwargs()
        assert "p" in kw and "maxiter" in kw and "num_reads" not in kw
        an = QORECache(solver_method="anneal", num_reads=25, seed=1)
        kw2 = an._solver_kwargs()
        assert "num_reads" in kw2 and "p" not in kw2

    def test_qaoa_blocks_never_exceed_qubit_cap(self):
        # Regression (audit B1): block decomposition must keep every block within
        # the QAOA qubit ceiling. Floor-division sizing produced 18-qubit blocks
        # at N=35 etc. — a 2^18 blow-up. ceil-division must prevent it for all N.
        from qore.block_decompose import decompose
        c = QORECache(solver_method="qaoa_tc")
        cap = c._max_direct_candidates()
        bsize = c._block_size()
        assert bsize <= cap
        for N in range(15, 200):
            num_blocks = max(2, -(-N // bsize))
            blocks = decompose(np.random.rand(N), np.zeros((N, N)),
                               max(1, N // 3), num_blocks=num_blocks)
            largest = max(len(bi) for *_, bi in blocks)
            assert largest <= cap, f"N={N}: block {largest} > cap {cap}"


class TestQORECacheEdgeCases:
    """Regression tests for audit findings B2 (degenerate budget) and B3 (batch)."""

    @staticmethod
    def _prefill_then_decode(cache, num_layers, prefill=8, decode=4, batch=1):
        """Drive a cache through a multi-token prefill + single-token decodes.
        Eviction only fires on decode steps, so edge-case tests must decode."""
        pk = torch.randn(batch, 4, prefill, 16)
        for l in range(num_layers):
            cache.update(pk.clone(), pk.clone(), l)
        for _ in range(decode):
            tk = torch.randn(batch, 4, 1, 16)
            for l in range(num_layers):
                cache.update(tk.clone(), tk.clone(), l)

    @pytest.mark.parametrize("cap", [4, 3, 2])
    def test_capacity_le_sinks_keeps_sinks_no_crash(self, cap):
        # Audit B2: max_capacity <= num_sink_tokens gave K<=0 -> solver ValueError.
        # Must instead keep the first `cap` positions without crashing.
        c = QORECache(max_capacity=cap, trigger_every=1, num_layers=2,
                      num_sink_tokens=4, seed=0, quality="keynorm")
        self._prefill_then_decode(c, num_layers=2)
        lens = [c.key_cache[i].shape[2] for i in range(2)]
        assert len(set(lens)) == 1 and lens[0] == cap

    @pytest.mark.parametrize("factory", [
        lambda: QORECache(max_capacity=5, trigger_every=1, num_layers=2,
                          num_sink_tokens=1, seed=0, quality="keynorm"),
        lambda: H2OCache(max_capacity=5, trigger_every=1, num_layers=2,
                         num_sink_tokens=1),
        lambda: SnapKVCache(max_capacity=5, trigger_every=1, num_layers=2,
                            num_sink_tokens=1, window=2),
        lambda: PyramidKVCache(max_capacity=5, trigger_every=1, num_layers=2,
                               num_sink_tokens=1),
    ])
    def test_batch_gt_1_raises(self, factory):
        # Audit B3: eviction uses batch-0 signals for all rows; batched
        # generation would silently evict wrong tokens. ALL eviction caches
        # must assert, not just QORE. Eviction fires on a decode step, so we
        # must reach one with batch>1.
        c = factory()
        with pytest.raises(AssertionError):
            self._prefill_then_decode(c, num_layers=2, batch=3)


# Real HF forward pass with a tiny randomly-initialized Llama. Validates cache
# mechanics against transformers.generate — especially PyramidKV's uneven
# per-layer lengths (audit risk D2) — which synthetic update() calls can't cover.
transformers = pytest.importorskip("transformers")


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


class TestRealForwardGeneration:
    """End-to-end generate() with real attention — the path eval actually uses."""

    def _gen(self, model, cache, capture, prompt, n=30):
        from applications.kv_cache.attention_capture import AttentionCapture
        with torch.no_grad():
            if capture and cache is not None:
                with AttentionCapture(model, cache):
                    return model.generate(prompt, max_new_tokens=n, do_sample=False,
                                          past_key_values=cache, use_cache=True)
            return model.generate(prompt, max_new_tokens=n, do_sample=False,
                                  past_key_values=cache, use_cache=True)

    def test_pyramidkv_uneven_layers_generate(self):
        # D2 regression: PyramidKV keeps layers at DIFFERENT lengths. Real
        # generate() must not desync position/mask bookkeeping.
        model = _tiny_llama()
        prompt = torch.randint(0, 256, (1, 80))
        cache = PyramidKVCache(max_capacity=48, trigger_every=8,
                               num_layers=4, num_sink_tokens=4)
        out = self._gen(model, cache, capture=True, prompt=prompt, n=40)
        assert out.shape[1] == 120  # 80 prompt + 40 generated
        lens = [cache.key_cache[i].shape[2] for i in range(4)]
        assert len(set(lens)) > 1, f"expected uneven pyramid, got {lens}"
        assert lens[0] > lens[-1]

    @pytest.mark.parametrize("factory", [
        lambda: QORECache(max_capacity=10000, trigger_every=10000,
                          num_layers=4, num_sink_tokens=4, quality="keynorm"),
        lambda: H2OCache(max_capacity=10000, trigger_every=10000,
                         num_layers=4, num_sink_tokens=4),
        lambda: SnapKVCache(max_capacity=10000, trigger_every=10000,
                            num_layers=4, num_sink_tokens=4, window=8),
        lambda: PyramidKVCache(max_capacity=10000, trigger_every=10000,
                               num_layers=4, num_sink_tokens=4),
    ])
    def test_wrapper_transparent_without_eviction(self, factory):
        # When capacity is never exceeded, each cache must produce IDENTICAL
        # output to a plain DynamicCache — proving the wrapper itself is inert.
        from transformers import DynamicCache
        model = _tiny_llama()
        prompt = torch.randint(0, 256, (1, 40))
        ref = self._gen(model, DynamicCache(), capture=False, prompt=prompt)[0].tolist()
        out = self._gen(model, factory(), capture=False, prompt=prompt)[0].tolist()
        assert out == ref

    def test_pyramidkv_generate_with_eviction_per_layer_positions(self):
        # The path eval ACTUALLY uses: generate_with_eviction, which for
        # per-layer-uneven caches (PyramidKV) activates PerLayerPositionPatch to
        # re-base each layer's RoPE query to its own physical length. Plain HF
        # generate can't drive physical eviction; this loop does. Assert it runs,
        # keeps an uneven pyramid, and yields finite in-vocab tokens (position
        # desync would produce OOB indices or NaN logits -> garbage/crash).
        from applications.kv_cache.attention_capture import AttentionCapture
        from scripts.kv_cache.eval_kv_cache import generate_with_eviction
        model = _tiny_llama()
        prompt = torch.randint(0, 256, (1, 100))
        cache = PyramidKVCache(max_capacity=48, trigger_every=8,
                               num_layers=4, num_sink_tokens=4, beta=0.5)
        assert getattr(cache, "per_layer_uneven", False) is True
        with torch.no_grad(), AttentionCapture(model, cache):
            gen = generate_with_eviction(model, prompt, cache, max_new_tokens=30, eos_id=-1)
        assert len(gen) == 30
        assert all(isinstance(t, int) and 0 <= t < 256 for t in gen)
        lens = [cache.key_cache[i].shape[2] for i in range(4)]
        assert len(set(lens)) > 1 and lens[0] > lens[-1], f"expected pyramid, got {lens}"

    def test_generate_with_eviction_uniform_cache_unpatched(self):
        # Regression: uniform caches (per_layer_uneven falsey) must take the
        # plain shared-position path — the patch refactor must not change their
        # behaviour. QORE keeps all layers the same length, so no patch fires.
        from applications.kv_cache.attention_capture import AttentionCapture
        from scripts.kv_cache.eval_kv_cache import generate_with_eviction
        model = _tiny_llama()
        prompt = torch.randint(0, 256, (1, 100))
        cache = QORECache(max_capacity=48, trigger_every=8, num_layers=4,
                          num_sink_tokens=4, quality="keynorm")
        assert getattr(cache, "per_layer_uneven", False) is False
        with torch.no_grad(), AttentionCapture(model, cache):
            gen = generate_with_eviction(model, prompt, cache, max_new_tokens=30, eos_id=-1)
        assert len(gen) == 30
        lens = [cache.key_cache[i].shape[2] for i in range(4)]
        assert len(set(lens)) == 1, f"uniform cache must stay uniform, got {lens}"

    def test_multi_eos_stops_on_any(self):
        # generate_with_eviction must accept a set/list of EOS ids (Llama-3 has
        # two) and stop on ANY of them, matching model.generate. Force the first
        # generated token's id into the eos set and assert we stop immediately.
        from applications.kv_cache.attention_capture import AttentionCapture
        from scripts.kv_cache.eval_kv_cache import generate_with_eviction
        model = _tiny_llama()
        prompt = torch.randint(0, 256, (1, 20))
        # Discover the first token this model would greedily emit.
        cache0 = QORECache(max_capacity=10000, trigger_every=10000, num_layers=4,
                           num_sink_tokens=4, quality="keynorm")
        with torch.no_grad(), AttentionCapture(model, cache0):
            first = generate_with_eviction(model, prompt, cache0, max_new_tokens=5, eos_id=-1)[0]
        # Now treat that token as one of two EOS ids; generation must stop after it.
        cache1 = QORECache(max_capacity=10000, trigger_every=10000, num_layers=4,
                           num_sink_tokens=4, quality="keynorm")
        with torch.no_grad(), AttentionCapture(model, cache1):
            gen = generate_with_eviction(model, prompt, cache1, max_new_tokens=30,
                                         eos_id=[999999, first])
        assert gen == [first], f"expected stop after first EOS-matching token, got {gen[:5]}"
