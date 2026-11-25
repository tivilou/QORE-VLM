"""Token-conditioned common Q/K adapter; no I/O, training loop or scheduler.

Three-qubit exact classical simulation, not a quantum hardware advantage claim.
Wire0 is the least significant basis bit. Only the final CLS query is changed;
keys change for valid tokens. Native values, mask, tail and readout stay frozen.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('HEAD', 'LINEAR_QK', 'MLP_QK', 'PRODUCT_QK', 'QUANTUM_QK')


def circuit(phi, theta, entangle=True):
    """phi[...,6], theta[2,3,2] -> seven nonempty Z-parity expectations."""
    if phi.dtype != torch.float32 or theta.dtype != torch.float32:
        raise ValueError('declared FP32 circuit inputs required')
    shape = phi.shape[:-1]
    phi = phi.reshape(-1, 6)
    state = torch.zeros((len(phi), 8), dtype=torch.complex64, device=phi.device)
    state[:, 0] = 1

    def gate(s, angle, wire, axis):
        c, t = (angle / 2).cos(), (angle / 2).sin()
        if axis == 'y':
            g = torch.stack((c, -t, t, c), -1).to(torch.complex64)
        elif axis == 'x':
            g = torch.stack((c, -1j*t, -1j*t, c), -1)
        else:
            z = torch.zeros_like(c)
            g = torch.stack((c-1j*t, z, z, c+1j*t), -1)
        if g.ndim == 1:
            g = g.expand(len(s), 4)
        blocks = s.reshape(len(s), 2**(2-wire), 2, 2**wire)
        return torch.einsum('nab,nlbr->nlar', g.reshape(-1, 2, 2), blocks).reshape(-1, 8)

    for layer in range(2):
        for wire in range(3):
            state = gate(state, phi[:, wire], wire, 'y')
            state = gate(state, phi[:, wire+3], wire, 'z')
            state = gate(state, theta[layer, wire, 0], wire, 'y')
            state = gate(state, theta[layer, wire, 1], wire, 'x')
        if layer == 0 and entangle:
            for control, target in ((0, 1), (1, 2)):
                order = [j ^ (1 << target) if j & (1 << control) else j for j in range(8)]
                state = state[:, order]
    signs = phi.new_tensor([[(-1)**((j & mask).bit_count()) for j in range(8)]
                           for mask in range(1, 8)])
    prob = state.real.square() + state.imag.square()
    return (prob @ signs.T).reshape(*shape, 7)


class Adapter(nn.Module):
    def __init__(self, dimension, arm, seed):
        super().__init__()
        if arm not in ARMS or arm == 'HEAD':
            raise ValueError('upstream Q/K arm required')
        self.arm, self.dimension = arm, dimension
        gen = torch.Generator(device='cpu').manual_seed(seed)

        def parameter(shape, scale):
            return nn.Parameter(torch.randn(shape, generator=gen) * scale)

        self.projection = parameter((6, dimension), 1/math.sqrt(dimension))
        self.projection_bias = nn.Parameter(torch.zeros(6))
        # Draw shared upstream/downstream tensors before arm-specific critic draws.
        self.query_decoder = parameter((dimension, 7), .1)
        self.key_decoder = parameter((dimension, 7), .1)
        if arm in ('PRODUCT_QK', 'QUANTUM_QK'):
            self.theta = parameter((2, 3, 2), .1)
        elif arm == 'LINEAR_QK':
            self.weight = parameter((7, 6), 1/math.sqrt(6))
            self.bias = nn.Parameter(torch.zeros(7))
        else:
            self.weight1 = parameter((12, 6), 1/math.sqrt(6))
            self.bias1 = nn.Parameter(torch.zeros(12))
            self.weight2 = parameter((7, 12), 1/math.sqrt(12))
            self.bias2 = nn.Parameter(torch.zeros(7))
        # Independent reference includes the initial projection, not just critic.
        for name, value in list(self.named_parameters()):
            if 'decoder' not in name:
                self.register_buffer('reference_'+name, value.detach().clone())

    def features(self, hidden, reference=False):
        def p(name):
            return getattr(self, ('reference_' if reference else '')+name)
        h = F.layer_norm(hidden, (self.dimension,))
        phi = math.pi * F.linear(h, p('projection'), p('projection_bias')).tanh()
        if self.arm in ('PRODUCT_QK', 'QUANTUM_QK'):
            return circuit(phi, p('theta'), self.arm == 'QUANTUM_QK')
        if self.arm == 'LINEAR_QK':
            return F.linear(phi, p('weight'), p('bias')).tanh()
        h = F.linear(phi, p('weight1'), p('bias1')).tanh()
        return F.linear(h, p('weight2'), p('bias2')).tanh()

    def forward(self, hidden, valid, reference=None):
        if reference is None:
            reference = self.features(hidden, True)
        difference = self.features(hidden) - reference
        delta_q = F.linear(difference[:, :1], self.query_decoder) / math.sqrt(7)
        delta_k = F.linear(difference, self.key_decoder) / math.sqrt(7)
        delta_k = delta_k * valid[..., None]
        return delta_q, delta_k


class CoRead(nn.Module):
    """Explicit joint readout hypothesis, shared identically across QK controls."""
    def __init__(self, original_head, dimension, arm, seed):
        super().__init__()
        self.qk = Adapter(dimension, arm, seed)
        self.head = copy.deepcopy(original_head).cpu()
        self.head.weight.requires_grad_(True)
        self.head.bias.requires_grad_(False)


def cls_scores(layer, bridge, head, hidden, mask, valid, adapter=None, capture=None,
               precomputed=None, reference=None):
    """CLS-only exact dependency slice of the native final self-attention layer.

Full Q/K/V weights and mask values are original. Query sequence length is1,
so actual native-full/null equality must be measured on the prepared inputs.
No eager/math override, output_attentions, attention bias or value adaptation.
"""
    attn = layer.attention.self
    if layer.training or any(p.requires_grad for p in layer.parameters()):
        raise ValueError('native final layer must remain frozen/eval')
    if getattr(attn, 'position_embedding_type', 'absolute') != 'absolute':
        raise ValueError('relative-position attention needs a separate adapter')
    if precomputed is None:
        q = attn.query(hidden[:, :1])
        k, v = attn.key(hidden), attn.value(hidden)
    else:
        q, k, v = precomputed
    dq = dk = None
    if adapter is not None:
        dq, dk = adapter(hidden, valid, reference)
        q, k = q+dq, k+dk
    heads, size = attn.num_attention_heads, attn.attention_head_size
    def split(x):
        return x.reshape(len(x), -1, heads, size).transpose(1, 2)
    if mask is not None and mask.shape[-2] != 1:
        mask = mask[..., :1, :]
    context = F.scaled_dot_product_attention(split(q), split(k), split(v),
                                             attn_mask=mask, dropout_p=0., is_causal=False)
    context = context.transpose(1, 2).reshape(len(hidden), 1, heads*size)
    attention = layer.attention.output(context, hidden[:, :1])
    final = layer.output(layer.intermediate(attention), attention)[:, 0]
    logits = head(bridge(final)).squeeze(-1)
    if capture is not None:
        capture.update(delta_q=dq, delta_k=dk, cls=final, logits=logits)
    return logits
