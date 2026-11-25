"""Shared DPR/BGE readout on train-aligned coordinates; no I/O or fitting loop.

Exact three-qubit classical simulation. The mechanism under test is a shared
nonlinear quantum feature/measurement adapter, not computational advantage.
Both native linear heads remain separate. Inference needs only one backbone.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.toy100_density_kernel_v001 import prepared
from applications.rag.qvalue_portable_v001 import need

ARMS = ('HEAD', 'LINEAR_SHARED', 'MLP_STATE', 'PRODUCT', 'DEPHASED', 'QUANTUM')


def aligned_coordinates(packed):
    maps = {k:prepared(p['features']['train'].cuda(), p['features']['holdout'].cuda())
            for k,p in packed.items()}
    d, b = maps['dpr']['coordinates'], maps['bge']['coordinates']
    u, singular, vh = torch.linalg.svd(b[:4000].T@d[:4000], full_matrices=False)
    rotation = u@vh  # Row-vector Procrustes: B R approximates D, train pairs only.
    need(float((rotation.T@rotation-torch.eye(12, device=b.device)).abs().max()) < 1e-5,
         'orthogonal train-only paired alignment')
    values = dict(dpr=d.reshape(100,50,12), bge=(b@rotation).reshape(100,50,12))
    metadata = dict(fit_pairs=4000, fit_questions=80, dimensions=12,
                    rotation=rotation.cpu(), singular_values=singular.cpu(),
                    maps={k:{n:v.cpu() if isinstance(v,torch.Tensor) else v
                             for n,v in m.items() if n not in ('coordinates','unit')}
                          for k,m in maps.items()},
                    before_squared_error=float((b[:4000]-d[:4000]).square().mean()),
                    after_squared_error=float((values['bge'][:80].flatten(0,1)-d[:4000]).square().mean()))
    # Use a canonical one-question matrix shape for stored and standalone maps.
    # A flattened100question GEMM may round differently from a20question GEMM,
    # which can flip almost tied Top5 positions despite close score values.
    values={k:transform_readouts(torch.cat((p['features']['train'],p['features']['holdout'])).cuda(),k,metadata)
            for k,p in packed.items()}
    return values, metadata


def transform_readouts(features, kind, mapping):
    """Single-backbone inference transform; no paired inference inputs."""
    m=mapping['maps'][kind];device=features.device
    if features.ndim==3:
        return torch.stack([transform_readouts(row,kind,mapping) for row in features])
    centered=features-features.mean(-2,keepdim=True)
    coords=((centered-m['mean'].to(device))@m['basis'].to(device))/m['scale'].to(device)
    return coords if kind=='dpr' else coords@mapping['rotation'].to(device)


def statevector(phi, theta, entangle=True, initial_state=None):
    """RY/RZ input reupload, trainable RY/RX; only inter-block CNOT ring.

    LSB wire0. There is no terminal entangler that could cancel in a fidelity.
    This is the state-valued counterpart of the earlier tested QK circuit.
    """
    need(phi.dtype == theta.dtype == torch.float32, 'declared FP32 circuit')
    original = phi.shape[:-1];phi=phi.reshape(-1,6)
    # Optional local-circuit batching. Preserve the historical global-theta
    # branch exactly; local parameters broadcast over token/question axes.
    if theta.shape[-3:] != (2,3,2):raise ValueError('two layers/three wires/two rotations')
    local_theta=(torch.broadcast_to(theta,(*original,2,3,2)).reshape(-1,2,3,2) if theta.ndim>3 else None)
    if initial_state is None:
        state=torch.zeros(len(phi),8,dtype=torch.complex64,device=phi.device);state[:,0]=1
    else:
        need(initial_state.shape==(*original,8) and initial_state.dtype==torch.complex64 and initial_state.device==phi.device,'explicit recurrent quantum state')
        state=initial_state.reshape(-1,8)
    def gate(s, angle, wire, axis):
        c,t=(angle/2).cos(),(angle/2).sin()
        if axis=='y':g=torch.stack((c,-t,t,c),-1).to(torch.complex64)
        elif axis=='x':g=torch.stack((c,-1j*t,-1j*t,c),-1)
        else:
            z=torch.zeros_like(c);g=torch.stack((c-1j*t,z,z,c+1j*t),-1)
        if g.ndim==1:g=g.expand(len(s),4)
        blocks=s.reshape(len(s),2**(2-wire),2,2**wire)
        return torch.einsum('nab,nlbr->nlar',g.reshape(-1,2,2),blocks).reshape(-1,8)
    for layer in range(2):
        for wire in range(3):
            for angle,axis in ((phi[:,wire],'y'),(phi[:,wire+3],'z'),
                               (theta[layer,wire,0] if local_theta is None else local_theta[:,layer,wire,0],'y'),
                               (theta[layer,wire,1] if local_theta is None else local_theta[:,layer,wire,1],'x')):
                state=gate(state,angle,wire,axis)
        if layer==0 and entangle:
            for control,target in ((0,1),(1,2)):
                order=[j^(1<<target) if j&(1<<control) else j for j in range(8)]
                state=state[:,order]
    return state.reshape(*original,8)


def pauli_basis():
    matrices=[torch.eye(2,dtype=torch.complex64),
              torch.tensor([[0,1],[1,0]],dtype=torch.complex64),
              torch.tensor([[0,-1j],[1j,0]],dtype=torch.complex64),
              torch.tensor([[1,0],[0,-1]],dtype=torch.complex64)]
    return torch.stack([torch.kron(torch.kron(matrices[(i//16)%4],matrices[(i//4)%4]),matrices[i%4])
                        for i in range(1,64)])


class SharedMap(nn.Module):
    def __init__(self, arm, seed, private_entrances=False):
        super().__init__();need(arm in ARMS[1:], 'shared non-HEAD arm');self.arm=arm
        self.private_entrances=private_entrances
        g=torch.Generator(device='cpu').manual_seed(seed)
        def parameter(shape,scale):return nn.Parameter(torch.randn(shape,generator=g)*scale)
        if arm=='LINEAR_SHARED':
            self.linear=parameter((12,),1/math.sqrt(12))
        else:
            self.projection=parameter((6,12),1/math.sqrt(12))
            self.projection_bias=nn.Parameter(torch.zeros(6))
            self.measurement=parameter((63,),1/math.sqrt(63))
            if arm=='MLP_STATE':
                self.weight1=parameter((12,6),1/math.sqrt(6));self.bias1=nn.Parameter(torch.zeros(12))
                self.weight2=parameter((16,12),1/math.sqrt(12));self.bias2=nn.Parameter(torch.zeros(16))
            else:self.theta=parameter((2,3,2),.1)
            self.register_buffer('paulis',pauli_basis())
            self.register_buffer('diagonal_selector',torch.tensor([
                all((i//(4**q))%4 in (0,3) for q in range(3)) for i in range(1,64)]))
        if private_entrances:
            # Identical initial values: only the learned coupling changes.
            for name in (('linear',) if arm=='LINEAR_SHARED' else ('projection','projection_bias')):
                value=getattr(self,name).detach()
                setattr(self,name,nn.Parameter(value[None].repeat(2,*([1]*value.ndim))))
        self.domain_parameters=('linear','projection','projection_bias') if private_entrances else ()
        for n,p in list(self.named_parameters()):self.register_buffer('reference_'+n,p.detach().clone())

    def scalar(self, coords, reference=False, capture=None, domain=None):
        def p(n):
            value=getattr(self,('reference_' if reference else '')+n)
            if n in self.domain_parameters:
                need(domain in (0,1),'explicit own-model entrance')
                value=value[domain]
            return value
        if self.arm=='LINEAR_SHARED':return coords@p('linear')
        phi=math.pi*F.linear(coords,p('projection'),p('projection_bias')).tanh()
        if self.arm=='MLP_STATE':
            raw=F.linear(F.linear(phi,p('weight1'),p('bias1')).tanh(),p('weight2'),p('bias2'))
            state=torch.complex(raw[...,:8],raw[...,8:])
            state=state/torch.linalg.vector_norm(state,dim=-1,keepdim=True).clamp_min(1e-12)
        else:state=statevector(phi,p('theta'),self.arm!='PRODUCT')
        moments=torch.einsum('...i,pij,...j->...p',state.conj(),self.paulis,state).real
        if self.arm=='DEPHASED':moments=moments*self.diagonal_selector
        if capture is not None:capture.update(state=state,moments=moments,phi=phi)
        return moments@p('measurement')

    def forward(self, coords, reference=None, domain=None):
        if reference is None:
            with torch.no_grad():reference=self.scalar(coords,True,domain=domain)
        return self.scalar(coords,domain=domain)-reference


class PairedReadout(nn.Module):
    def __init__(self, packed, arm, seed, private_entrances=False):
        super().__init__();self.arm=arm;self.heads=nn.ModuleDict()
        for kind,p in packed.items():
            weight,bias=p['original_head']['weight'],p['original_head']['bias']
            head=nn.Linear(weight.shape[-1],1)
            head.load_state_dict(dict(weight=weight,bias=bias));head.bias.requires_grad_(False)
            self.heads[kind]=head
        self.shared=None if arm=='HEAD' else SharedMap(arm,seed,private_entrances)

    def references(self, coords, native, features=None):
        with torch.no_grad():
            return {k:torch.zeros_like(native[k]) if self.shared is None else self.shared.scalar(coords[k],True,domain=i)
                    for i,k in enumerate(('dpr','bge'))}

    def gradient_probe_parameter(self):
        if self.arm=='LINEAR_SHARED':return self.shared.linear
        return self.shared.measurement if self.shared.private_entrances else self.shared.projection

    def activity_gradient_names(self):
        if self.shared is None:return ()
        return ('shared.linear',) if self.arm=='LINEAR_SHARED' else ('shared.projection','shared.measurement')

    def forward_one(self, kind, features, native, original, coords):
        """Use one frozen backbone and its stored map/head; no other model input."""
        delta=0 if self.shared is None else self.shared(coords,domain=0 if kind=='dpr' else 1)
        return native+self.heads[kind](features).squeeze(-1)-original+delta

    def forward(self, features, native, originals, coords, references):
        counts=[coords[k].numel()//12 for k in ('dpr','bge')]
        delta=None
        if self.shared is not None:
            if self.shared.private_entrances:
                delta=[self.shared(coords[k],references[k],domain=i) for i,k in enumerate(('dpr','bge'))]
            else:
                joined=torch.cat([coords[k].reshape(-1,12) for k in ('dpr','bge')])
                ref=torch.cat([references[k].reshape(-1) for k in ('dpr','bge')])
                delta=self.shared(joined,ref).split(counts)
        return {k:native[k]+self.heads[k](features[k]).squeeze(-1)-originals[k]+
                (0 if delta is None else delta[i].reshape_as(native[k]))
                for i,k in enumerate(('dpr','bge'))}
