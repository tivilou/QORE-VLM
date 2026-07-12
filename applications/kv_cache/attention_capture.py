"""
Attention capture for KV-cache eviction.

Real H2O-style eviction needs the *cumulative attention* each cached token has
received (the "heavy hitter" score) — but `DynamicCache.update()` only sees the
key/value states, never the attention weights. This module bridges that gap by
installing forward hooks on the model's attention modules.

Mechanism (per generation step / prefill):
  1. A forward *pre-hook* forces `output_attentions=True` on each attention
     module, so eager attention returns its weight matrix.
  2. A forward *post-hook* reads that matrix `[batch, heads, q_len, kv_len]`,
     reduces it on the fly to a per-key score `[kv_len]` (sum over queries,
     mean over heads), and feeds it to the cache's accumulator.

The reduction happens immediately per layer, so we never retain the full
`[heads, S, S]` attention tensor across layers (which would be hundreds of GB at
8K context). Requires the model to be loaded with `attn_implementation="eager"`.
"""

import functools
from typing import List, Optional

import torch


def find_attention_modules(model) -> List[torch.nn.Module]:
    """
    Return attention submodules in layer order.

    Works for LLaMA/Mistral-style models (``model.model.layers[i].self_attn``)
    and falls back to a name-based scan for other architectures.
    """
    # Fast path: standard decoder layout
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is not None:
        mods = [getattr(layer, "self_attn", None) for layer in layers]
        if all(m is not None for m in mods):
            return list(mods)

    # Fallback: any module whose class name ends with "Attention", in order
    return [m for _, m in model.named_modules()
            if m.__class__.__name__.endswith("Attention")]


def reduce_attention(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Reduce an attention weight matrix to a per-key cumulative score.

    Args:
        attn_weights: ``[batch, heads, q_len, kv_len]`` — attention probabilities
            (post-softmax) from one layer for one forward pass.

    Returns:
        ``[kv_len]`` tensor: total attention received by each cached key,
        summed over query positions and averaged over heads (batch 0).

    This is the H2O heavy-hitter signal: how much attention each key attracts
    from the queries in this pass. Accumulated across steps, it approximates
    cumulative attention received.
    """
    if attn_weights.dim() != 4:
        raise ValueError(f"expected 4D attention, got shape {tuple(attn_weights.shape)}")
    # Sanitize NaN/inf BEFORE reducing. Under fp16 eager attention with
    # output_attentions=True, some models (e.g. Qwen2's last layer) emit NaN/inf
    # in the returned weight matrix — a numerical instability of the fp16 path.
    # Because the accumulator sums this signal ADDITIVELY across all layers, a
    # single poisoned layer turns the whole cumulative-attention vector into NaN,
    # which then makes the QORE QUBO diagonal NaN and the solver keep everything
    # (eviction silently no-ops). A NaN/inf attention entry carries no usable
    # importance information, so mapping it to 0 (no contribution) is the correct
    # semantics and keeps the signal finite. fp32 (e.g. on CPU) never triggered
    # this, which is why it only surfaced on real fp16 GPU runs.
    attn_weights = torch.nan_to_num(
        attn_weights.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    # [batch, heads, q_len, kv_len] -> sum over queries -> [batch, heads, kv_len]
    per_key = attn_weights.sum(dim=2)
    # mean over heads, take batch 0 -> [kv_len]
    return per_key.mean(dim=1)[0]


class AttentionCapture:
    """
    Installs hooks that stream per-key attention scores into a cache object.

    The target cache must expose ``add_attention(layer_idx, scores)`` where
    ``scores`` is a ``[kv_len]`` tensor. Use as a context manager so hooks are
    always removed::

        with AttentionCapture(model, cache):
            model.generate(...)
    """

    def __init__(self, model, cache):
        self.model = model
        self.cache = cache
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self):
        modules = find_attention_modules(self.model)
        for layer_idx, module in enumerate(modules):
            self._handles.append(
                module.register_forward_pre_hook(self._force_attn, with_kwargs=True)
            )
            self._handles.append(
                module.register_forward_hook(
                    functools.partial(self._collect, layer_idx=layer_idx),
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    @staticmethod
    def _force_attn(module, args, kwargs):
        """Pre-hook: force this attention module to return its weight matrix."""
        kwargs["output_attentions"] = True
        return args, kwargs

    def _collect(self, module, args, kwargs, output, layer_idx: int):
        """Post-hook: pull attention weights from the output and accumulate them."""
        # Eager attention returns (hidden_states, attn_weights, ...) with
        # attn_weights = None unless output_attentions=True (forced above).
        attn = None
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
        if attn is None:
            return output  # SDPA/FlashAttn path — no weights available
        scores = reduce_attention(attn.detach())
        self.cache.add_attention(layer_idx, scores)
        return output

