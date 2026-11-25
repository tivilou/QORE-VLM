"""Full-native signed amplitude encoding with a shallow quadratic observable.

Ten ideal simulator qubits, no state-preparation/hardware speedup claim. Any arm
is a classical differentiable function; quantum arms impose a structured basis.
Removing CNOTs does not make the amplitude-encoded input a product state.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.toy100_native_feature_readout_v001 import NativeFeatureReadout
from applications.rag.qvalue_portable_v001 import need

AMPLITUDE_ARMS=('HEAD','MLP','LOW_RANK','RANDOM_QUADRATIC','DIAGONAL','FROZEN_BASIS','LOCAL_ROTATIONS','QUANTUM')


def cnot_orders(wires):
    return torch.tensor([[j^(1<<(q+1)) if j&(1<<q) else j for j in range(2**wires)]
                         for q in range(wires-1)])


def rotate_amplitudes(amplitude,theta,orders=None):
    """LSB wire0, two RY layers, optional inter-layer directed CNOT chain."""
    need(amplitude.dtype==theta.dtype==torch.float32,'declared FP32 amplitude circuit')
    wires=theta.shape[-1];need(amplitude.shape[-1]==2**wires and theta.shape==(2,wires),'exact state/circuit dimensions')
    original=amplitude.shape;state=amplitude.reshape(-1,2**wires)
    for layer in range(2):
        for q in range(wires):
            a,b=state.reshape(-1,2**(wires-q-1),2,2**q).unbind(-2)
            c,s=(theta[layer,q]/2).cos(),(theta[layer,q]/2).sin()
            state=torch.stack((c*a-s*b,s*a+c*b),-2).reshape(-1,2**wires)
        if layer==0 and orders is not None:
            for order in orders:state=state[:,order]
    return state.reshape(original)


def z_signs(wires):
    masks=[1<<q for q in range(wires)]+[(1<<a)|(1<<b) for a in range(wires) for b in range(a+1,wires)]
    return torch.tensor([[1. if (j&mask).bit_count()%2==0 else -1. for j in range(2**wires)] for mask in masks])


def encode(features):
    need(features.shape[-1] in (768,1024),'pinned native CLS dimension')
    z=F.layer_norm(features,(features.shape[-1],))
    z=F.pad(z,(0,1024-z.shape[-1]))
    norm=torch.linalg.vector_norm(z,dim=-1,keepdim=True)
    need(bool((norm>1e-8).all()),'nondegenerate full native amplitude')
    return z/norm


class AmplitudeMap(nn.Module):
    def __init__(self,arm,seed):
        super().__init__();need(arm in AMPLITUDE_ARMS[1:],'allowlisted amplitude arm');self.arm=arm
        g=torch.Generator(device='cpu').manual_seed(seed)
        def parameter(shape,scale):
            value=torch.randn(shape,generator=g)*scale
            return nn.Parameter(value[None].repeat(2,*([1]*value.ndim)))
        width=4 if arm in ('MLP','LOW_RANK') else 55
        self.measurement=parameter((width,),1/math.sqrt(width))
        if arm in ('MLP','LOW_RANK'):
            self.weight=parameter((4,1024),1/math.sqrt(1024))
            if arm=='MLP':self.bias=nn.Parameter(torch.zeros(2,4))
        elif arm=='RANDOM_QUADRATIC':
            value=torch.randn(55,1024,generator=g);value=value/torch.linalg.vector_norm(value,dim=-1,keepdim=True)
            self.register_buffer('random_directions',value)
        elif arm!='DIAGONAL':
            self.theta=parameter((2,10),.1)
            if arm=='FROZEN_BASIS':self.theta.requires_grad_(False)
        self.register_buffer('orders',cnot_orders(10));self.register_buffer('signs',z_signs(10))
        for name,p in list(self.named_parameters()):self.register_buffer('reference_'+name,p.detach().clone())

    def scalar(self,features,kind,reference=False):
        domain=0 if kind=='dpr' else 1
        def p(name):return getattr(self,('reference_' if reference else '')+name)[domain]
        amplitude=encode(features)
        if self.arm in ('MLP','LOW_RANK'):
            # Same information. sqrt1024 restores unit input RMS for classical
            # fan-in initialization; it adds no magnitude or target information.
            hidden=F.linear(amplitude*32,p('weight'))
            moments=(hidden+p('bias')).tanh() if self.arm=='MLP' else hidden.square()
        elif self.arm=='RANDOM_QUADRATIC':moments=F.linear(amplitude*32,self.random_directions).square()-1
        else:
            state=amplitude if self.arm=='DIAGONAL' else rotate_amplitudes(amplitude,p('theta'),None if self.arm=='LOCAL_ROTATIONS' else self.orders)
            moments=state.square()@self.signs.T
        return moments@p('measurement')

    def forward(self,features,kind,reference=None):
        if reference is None:
            with torch.no_grad():reference=self.scalar(features,kind,True)
        return self.scalar(features,kind)-reference


class AmplitudeReadout(NativeFeatureReadout):
    def __init__(self,packed,arm,seed):
        need(arm in AMPLITUDE_ARMS,'allowlisted amplitude readout')
        super().__init__(packed,'HEAD',seed);self.arm=arm
        if arm!='HEAD':self.shared=AmplitudeMap(arm,seed)

    def activity_gradient_names(self):
        if self.shared is None:return ()
        names=['shared.measurement']
        for name in ('weight','theta'):
            p=getattr(self.shared,name,None)
            if isinstance(p,nn.Parameter) and p.requires_grad:names.append('shared.'+name)
        return tuple(names)
