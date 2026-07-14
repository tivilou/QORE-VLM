"""
Prefill-time KV compression via attention monkeypatch (transformers 4.44).

The official H2O/SnapKV/PyramidKV paradigm compresses the KV cache ONCE at the
end of prefill, inside the attention forward (it needs query_states to score
keys). We replicate that here without rewriting the whole attention forward:

Strategy — post-forward cache compression:
  A forward hook on each attention module fires AFTER the module has appended
  this step's K/V to the DynamicCache. On the PREFILL pass (query length > 1) we:
    1. read the just-appended K/V for this layer from the cache,
    2. recompute the query projection for scoring (cheap: one matmul on the
       already-available hidden states — we capture them via a pre-hook),
    3. call cluster.update_kv(K, Q, V) to get compressed K/V,
    4. overwrite the cache's layer tensors with the compressed versions.
  During decode (query length == 1) the hook is a no-op; the cache just grows.

This keeps retained keys at their ORIGINAL RoPE phase and lets the model use its
normal growing logical position for new queries — so relative geometry stays
correct with NO re-rotation (the official paradigm's key advantage). We rely on
transformers' standard generate loop for positions/masks.

Only Llama and Qwen2 (our eval models) are wired here.
"""

import functools
import torch


class PrefillCompressor:
    """Install prefill-time compression hooks driven by per-layer clusters.

    Usage:
        comp = PrefillCompressor(model, make_cluster)
        with comp:
            model.generate(...)   # compression happens on the prefill forward

    `make_cluster(layer_idx, num_layers)` returns a fresh cluster exposing
    `update_kv(key, query, value)` (see official.clusters).
    """

    def __init__(self, model, make_cluster):
        self.model = model
        self.make_cluster = make_cluster
        self._handles = []
        self._hidden = {}   # layer_idx -> hidden_states captured pre-forward
        self._clusters = {}
        self._done = False  # compression happens once (prefill); then dormant

    def _attn_modules(self):
        base = getattr(self.model, "model", self.model)
        return list(getattr(base, "layers"))

    def __enter__(self):
        base = getattr(self.model, "model", self.model)
        num_layers = len(base.layers)
        for idx, layer in enumerate(base.layers):
            attn = layer.self_attn
            self._clusters[idx] = self.make_cluster(idx, num_layers)
            self._handles.append(
                attn.register_forward_pre_hook(
                    functools.partial(self._pre, layer_idx=idx), with_kwargs=True))
            self._handles.append(
                attn.register_forward_hook(
                    functools.partial(self._post, layer_idx=idx), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._hidden.clear()
        return False

    def _pre(self, module, args, kwargs, layer_idx):
        # Capture hidden_states + position_embeddings so the post-hook can
        # recompute a RoPE'd query for scoring (cache keys are already RoPE'd).
        hs = kwargs.get("hidden_states")
        if hs is None and args:
            hs = args[0]
        pos_emb = kwargs.get("position_embeddings")
        pos_ids = kwargs.get("position_ids")
        self._hidden[layer_idx] = (hs, pos_emb, pos_ids)
        return None

    def _post(self, module, args, kwargs, output, layer_idx):
        # Only compress once, during the prefill pass (query length > 1).
        if self._done:
            return output
        hs_tuple = self._hidden.get(layer_idx)
        if hs_tuple is None:
            return output
        hs, pos_emb, pos_ids = hs_tuple
        if hs is None or hs.shape[1] <= 1:
            return output  # decode step — nothing to compress

        past = kwargs.get("past_key_value")
        if past is None or not hasattr(past, "key_cache"):
            return output
        if layer_idx >= len(past.key_cache):
            return output

        cluster = self._clusters[layer_idx]
        # Skip cheap: if prefix already <= capacity, cluster returns unchanged.
        cur_len = past.key_cache[layer_idx].shape[2]
        if cur_len <= getattr(cluster, "max_capacity_prompt", cur_len):
            self._mark_last_layer(layer_idx)
            return output

        # Recompute a RoPE'd query for scoring (same recipe as attention forward).
        q = self._project_query(module, hs, pos_emb, pos_ids)

        # Cache holds K/V already RoPE'd (K) and raw (V), at UNREPEATED GQA
        # granularity — exactly what the cluster's GQA path expects.
        k = past.key_cache[layer_idx]
        v = past.value_cache[layer_idx]
        k_c, v_c = cluster.update_kv(k, q, v)
        past.key_cache[layer_idx] = k_c
        past.value_cache[layer_idx] = v_c
        self._mark_last_layer(layer_idx)
        return output

    def _mark_last_layer(self, layer_idx):
        # After the final layer's prefill compression, go dormant.
        if layer_idx == len(self._clusters) - 1:
            self._done = True

    @staticmethod
    def _project_query(module, hidden_states, position_embeddings, position_ids):
        """Recompute RoPE'd query_states from hidden_states (Llama/Qwen2 recipe)."""
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        bsz, q_len, _ = hidden_states.shape
        q = module.q_proj(hidden_states)
        q = q.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
        # Need key projection too, only to satisfy apply_rotary_pos_emb's API shape;
        # we rotate q using cos/sin. Reuse position_embeddings if available.
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = module.rotary_emb(q, position_ids)
        # apply_rotary_pos_emb rotates (q, k); pass q as both, keep the rotated q.
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
        return q_rot
