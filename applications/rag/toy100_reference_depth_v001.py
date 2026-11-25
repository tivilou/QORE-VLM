"""114: homogeneous-reference amplitude encoding of three native depth views.

Ideal statevector / exact classical quadratic function, not a speedup claim.
Reference coherence supplies signed linear terms; all amplitude/readout costs
and unequal capacity controls are disclosed. No new model/data extraction.
"""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need, digest, checked_member
from applications.rag.toy100_native_feature_readout_v001 import NativeFeatureReadout
from applications.rag.toy100_amplitude_readout_v001 import cnot_orders, z_signs

KINDS = ('dpr', 'bge')
ARMS = ('HEAD', 'CONCAT', 'MLP', 'LOW_RANK', 'DIAGONAL', 'FROZEN_BASIS',
        'NO_CNOT', 'DEPHASE_REFERENCE', 'QUANTUM')
WIRES = 12
BANKS = {
    'dpr': '1cb2c31a645be9a9e1edae5d06f2b93c13d06cfbe1c79eab5ec593ff3872cfea',
    'bge': '24df1209a2f71c5c76b7a631384dca6b435094e257c0b020e3511dd4c1c2d565'}


def rotate(state, theta, orders=None):
    """Two RY layers, LSB wire0, directed CNOT chain between layers."""
    wires = theta.shape[-1]
    need(theta.shape == (2, wires) and state.shape[-1] == 2**wires,
         'exact amplitude/circuit shape')
    need(state.dtype == theta.dtype and state.dtype in (torch.float32, torch.float64),
         'real common circuit dtype')
    shape = state.shape; state = state.reshape(-1, 2**wires)
    for layer in range(2):
        for q in range(wires):
            a, b = state.reshape(-1, 2**(wires-q-1), 2, 2**q).unbind(-2)
            c, s = (theta[layer, q]/2).cos(), (theta[layer, q]/2).sin()
            state = torch.stack((c*a-s*b, s*a+c*b), -2).reshape(-1, 2**wires)
        if layer == 0 and orders is not None:
            for indices in orders: state = state[:, indices]
    return state.reshape(shape)


def reference_observation(data, lam, theta, measurement, signs, orders,
                          dephased=False, diagonal=False):
    """v^T U^T D(m) U v/(2 lambda), v=[lambda,data,zeros].

    This is ||v||^2/(2lambda) times the normalized-state expectation.
    Dephase ONLY reference/data coherence before the same U/measurement.
    No data-only arm: it differs from this dephasing by a candidate-constant.
    """
    need(lam.numel() == 1 and bool(lam > 0), 'positive TRAIN-only reference scale')
    size = signs.shape[-1]
    need(data.shape[-1]+1 <= size, 'complete signed input fits amplitude register')
    zero = torch.zeros_like(data[..., :1])
    value = F.pad(torch.cat((zero+lam, data), -1), (0, size-data.shape[-1]-1))
    observable = measurement @ signs
    def expectation(v):
        z = v if diagonal else rotate(v, theta, orders)
        return z.square() @ observable
    if not dephased: return expectation(value)/(2*lam)
    data_only = F.pad(torch.cat((zero, data), -1), (0, size-data.shape[-1]-1))
    ref = torch.zeros(size, dtype=data.dtype, device=data.device)
    ref[0] = lam
    return (expectation(data_only)+expectation(ref))/(2*lam)


class ReferenceDepthMap(nn.Module):
    def __init__(self, dimensions, scales, arm, seed):
        super().__init__(); need(arm in ARMS[1:], 'fixed reference-depth arm')
        self.arm = arm; self.dimensions = dimensions
        sizes = []
        for k in KINDS:
            d = dimensions[k]
            sizes.append(2*d if arm == 'CONCAT' else
                         12*d+8 if arm == 'MLP' else
                         12*d+8 if arm == 'LOW_RANK' else
                         78 if arm in ('DIAGONAL', 'FROZEN_BASIS') else 102)
        self.sizes = sizes
        g = torch.Generator(device='cpu').manual_seed(seed+11400)
        values = []
        for k, size in zip(KINDS, sizes):
            # Spectral deletion arms share full Q initialization, including
            # the second backbone; do not shift RNG by omitting 24 angles.
            value = torch.randn(102 if arm not in ('CONCAT','MLP','LOW_RANK') else size, generator=g)[:size]
            if arm == 'CONCAT': value.zero_()
            elif arm in ('MLP', 'LOW_RANK'):
                d = 3*dimensions[k]; value[:4*d] /= math.sqrt(d)
                value[4*d:] /= 2
                value[4*d:4*d+4].zero_()
            else:
                value[:78] /= math.sqrt(78)
                if size > 78: value[78:] *= .1
            values.append(value)
        self.theta = nn.Parameter(torch.cat(values))
        self.register_buffer('initial', self.theta.detach().clone())
        self.register_buffer('scales', torch.tensor([scales[k] for k in KINDS]))
        self.register_buffer('orders', cnot_orders(WIRES))
        self.register_buffer('signs', z_signs(WIRES))
        # Identical basis initialization for frozen versus trainable arms.
        basis = []
        basis_generator = torch.Generator(device='cpu').manual_seed(seed+11400)
        for _ in KINDS:
            raw = torch.randn(102, generator=basis_generator)
            basis.append(raw[78:].reshape(2, WIRES)*.1)
        self.register_buffer('basis', torch.stack(basis))

    def scalar(self, views, kind, reference=False):
        i = KINDS.index(kind); start = sum(self.sizes[:i])
        p = (self.initial if reference else self.theta)[start:start+self.sizes[i]]
        if self.arm == 'CONCAT':
            return (views[..., :2, :]*p.reshape(2, -1)).sum((-1, -2))
        data = views.flatten(-2); lam = self.scales[i]
        if self.arm in ('MLP', 'LOW_RANK'):
            d = data.shape[-1]; x = data*math.sqrt(d)/lam
            hidden = F.linear(x, p[:4*d].reshape(4, d))
            hidden = hidden+p[4*d:4*d+4]
            hidden = hidden.tanh() if self.arm == 'MLP' else hidden.square()
            coefficients = p[4*d+4:]
            return hidden @ coefficients
        theta = (None if self.arm == 'DIAGONAL' else self.basis[i]
                 if self.arm == 'FROZEN_BASIS' else p[78:].reshape(2, WIRES))
        return reference_observation(data, lam, theta, p[:78], self.signs,
                                     None if self.arm == 'NO_CNOT' else self.orders,
                                     dephased=self.arm == 'DEPHASE_REFERENCE',
                                     diagonal=self.arm == 'DIAGONAL')


class ReferenceDepthReadout(NativeFeatureReadout):
    explicit_context = True

    def __init__(self, packed, scales, arm, seed):
        super().__init__(packed, 'HEAD', seed); self.arm = arm
        self.shared = None if arm == 'HEAD' else ReferenceDepthMap(
            {k:p['original_head']['weight'].shape[-1] for k,p in packed.items()}, scales, arm, seed)
        for k,p in packed.items():self.register_buffer('initial_head_'+k,p['original_head']['weight'].clone())

    def references(self, coords, native, features=None):
        with torch.no_grad():
            return {k:torch.zeros_like(native[k]) if self.shared is None else
                    self.shared.scalar(coords[k], k, True) for k in KINDS}

    def gradient_probe_parameter(self):return self.shared.theta
    def activity_gradient_names(self):return () if self.shared is None else ('shared.theta',)

    def one(self, kind, features, native, coords, reference):
        value = native+F.linear(features, self.heads[kind].weight-getattr(self,'initial_head_'+kind)).squeeze(-1)
        return value if self.shared is None else value+(self.shared.scalar(coords,kind)-reference)

    def forward(self, features, native, originals, coords, references):
        return {k:self.one(k,features[k],native[k],coords[k],references[k]) for k in KINDS}

    def forward_one(self, kind, features, native, original, coords):
        with torch.no_grad():ref = torch.zeros_like(native) if self.shared is None else self.shared.scalar(coords,kind,True)
        return self.one(kind,features,native,coords,ref)

    def mechanism_diagnostics(self):
        return dict(information='Same full signed three native depths, final frozen bridge, TRAIN-only RMS reference amplitude',
                    encoding='[lambda,h_early,h_middle,h_final,zeros],12idealqubits; norm-restored Hermitian expectation',
                    claim='Reference coherence creates signed linear cross terms; exact classical augmented quadratic oracle. Quantum superiority/firstness/hardware advantage UNDEMONSTRATED; same architecture with private backbone parameters.')


def prepare(packed, targets):
    root = Path(__file__).resolve().parents[3]
    bank = root/'depth-order-run-v002/depth-bank'
    coords = {}; scales = {}; records = {}
    for kind in KINDS:
        path = bank/(kind+'-depths.bin')
        need(digest(path) == BANKS[kind], 'independent pinned full three-depth bank SHA')
        value = torch.load(path, weights_only=True)
        d = packed[kind]['original_head']['weight'].shape[-1]
        need(value.shape == (100,50,3,d) and bool(torch.isfinite(value).all()), 'complete finite original ordered three-depth views')
        final = torch.cat((packed[kind]['features']['train'],packed[kind]['features']['holdout']))
        need(torch.equal(value[:,:,-1],final), 'all100 exact original native final-feature identities')
        # TRAIN covariates only: no held-out features, scores or targets in scale.
        scale = value[:80].flatten(-2).square().sum(-1).mean().sqrt()
        need(bool(torch.isfinite(scale)) and bool(scale > 0), 'finite positive TRAIN RMS norm')
        coords[kind] = value.cuda(); scales[kind] = float(scale)
        records[kind] = dict(bank_sha256=BANKS[kind],source_cache_sha256=packed[kind]['cache_sha256'],scale=scales[kind])
    mapping = dict(candidate=114,records=records,order='original80TRAINthen20held',scales=scales)
    def restored(path):
        saved = torch.load(path,weights_only=True);need(saved == mapping, 'exact compact reference-depth mapping replay')
        for k in KINDS:
            need(digest(bank/(k+'-depths.bin')) == BANKS[k], 'same full depth-bank bytes at replay')
    return dict(coords=coords,mapping=mapping,restore_mapping=restored,arms=ARMS,
        factory=lambda arm,seed:ReferenceDepthReadout(packed,scales,arm,seed),
        protocol=dict(candidate='114_REFERENCE_COHERENT_DEPTH_OBSERVABLE',
          mechanism='Learn a Hermitian task observable on full3depth signed homogeneous amplitude; complete frozen initial reference subtraction; native current-head adaptation retained.',
          controls='HEAD1792; CONCAT+3584, MLP+21520, augmented LOW_RANK+21520, DIAGONAL/FROZEN_BASIS+156, NO_CNOT/DEPHASE_REFERENCE/QUANTUM+204nominal. NO_CNOT24angles/model combine to at most12effective. Unequal capacity disclosed. Same input information/loss/seeds/200steps. Exact augmented quadratic is a correctness oracle.',
          claim='Source preparation from113TRAIN readout evidence plus old fullnative amplitude/trajectory precedents; no generic novelty, computational advantage or independent confirmation.'))
