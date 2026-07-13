"""
RoPE re-rotation for physical KV eviction.

Why this exists
---------------
In HuggingFace decoder attention, keys have RoPE applied *before* they are
written to the cache (``apply_rotary_pos_emb`` runs, THEN
``past_key_value.update``). So a cached key at original absolute position ``p``
carries the rotation R(p) baked in.

Physical eviction keeps an arbitrary subset of keys and compacts them into slots
``0, 1, 2, ...``. But their baked-in phase still corresponds to their ORIGINAL
positions. If the next query is placed at the compact position (= physical
length L), its phase no longer matches the retained keys' phases — the relative
angle RoPE encodes (query_pos - key_pos), which is the whole point of RoPE, is
broken. This produces scrambled/degenerate generation.

The fix (StreamingLLM-style): after eviction, RE-ROTATE each retained key from
its original position ``p_orig`` to its new compact position ``p_new``. RoPE is
a rotation, so composing R(p_new) R(p_orig)^{-1} = R(p_new - p_orig) moves the
key's phase to what it *would* have been had it always sat at ``p_new``. Then
physical slot == phase for every key, the query at position L is consistent, and
HF's RoPE-table / cache_position / mask bookkeeping all line up naturally.

RoPE recap (per 2-dim pair d): a vector component pair (x_d, y_d) at position p
is rotated by angle ``p * inv_freq[d]``. transformers stores this in the
half-split layout used by ``rotate_half``:
    out = x * cos(p) + rotate_half(x) * sin(p)
To rotate an already-encoded key from p_orig to p_new we apply the delta angle
``Δ = p_new - p_orig`` with exactly the same formula.
"""

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """transformers' rotate_half: split last dim in two, (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rerotate_keys(keys, inv_freq, orig_positions, new_positions):
    """
    Move RoPE phase of cached keys from ``orig_positions`` to ``new_positions``.

    Args:
        keys: [num_heads, seq, head_dim] cached (already RoPE-encoded) key states
            for one layer, one batch element.
        inv_freq: [head_dim/2] the RoPE inverse frequencies (from rotary_emb).
        orig_positions: [seq] original absolute positions baked into each key.
        new_positions: [seq] target (compact) positions.

    Returns:
        keys with each position's phase shifted by (new - orig). Done in float32
        for accuracy, cast back to the keys' dtype.
    """
    dtype = keys.dtype
    device = keys.device
    inv_freq = inv_freq.to(device=device, dtype=torch.float32)
    delta = (new_positions - orig_positions).to(device=device, dtype=torch.float32)  # [seq]
    # angle[s, d] = delta[s] * inv_freq[d]
    angles = torch.outer(delta, inv_freq)  # [seq, head_dim/2]
    emb = torch.cat((angles, angles), dim=-1)  # [seq, head_dim]
    cos = emb.cos()[None, :, :]  # [1, seq, head_dim]
    sin = emb.sin()[None, :, :]
    k = keys.to(torch.float32)
    out = k * cos + _rotate_half(k) * sin
    return out.to(dtype)


def get_inv_freq(model):
    """Best-effort fetch of RoPE inv_freq from a HF causal LM.

    Handles both layouts:
      - model-level rotary_emb (Llama >= 4.43): model.model.rotary_emb.inv_freq
      - per-attention rotary_emb (Qwen2 4.44): first layer's self_attn.rotary_emb
    Returns a 1-D tensor [head_dim/2], or None if it can't be found.
    """
    base = getattr(model, "model", model)
    rot = getattr(base, "rotary_emb", None)
    if rot is not None and hasattr(rot, "inv_freq"):
        return rot.inv_freq.detach()
    layers = getattr(base, "layers", None)
    if layers:
        attn = getattr(layers[0], "self_attn", None)
        rot = getattr(attn, "rotary_emb", None) if attn is not None else None
        if rot is not None and hasattr(rot, "inv_freq"):
            return rot.inv_freq.detach()
    return None
