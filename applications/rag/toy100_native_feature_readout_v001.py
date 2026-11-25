"""Full native readout to learned angles; shared small quantum/classical critic.

Reuses the audited exact statevector/Pauli implementation. No input PCA enters the
scientific function; no loader, optimizer, scheduler or evaluator in this plugin.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.toy100_shared_readout_v001 import statevector,pauli_basis
from applications.rag.qvalue_portable_v001 import need

NATIVE_FEATURE_ARMS=('HEAD','MLP_STATE','TRIG','PRODUCT','DEPHASED','QUANTUM')


class NativeFeatureMap(nn.Module):
    def __init__(self,dimensions,arm,seed):
        super().__init__();self.arm=arm
        g=torch.Generator(device='cpu').manual_seed(seed)
        def parameter(shape,scale):return nn.Parameter(torch.randn(shape,generator=g)*scale)
        for kind,dimension in dimensions.items():
            setattr(self,'projection_'+kind,parameter((6,dimension),1/math.sqrt(dimension)))
            setattr(self,'bias_'+kind,nn.Parameter(torch.zeros(6)))
        self.measurement=parameter((63,),1/math.sqrt(63))
        if arm=='MLP_STATE':
            self.weight1=parameter((12,6),1/math.sqrt(6));self.bias1=nn.Parameter(torch.zeros(12))
            self.weight2=parameter((16,12),1/math.sqrt(12));self.bias2=nn.Parameter(torch.zeros(16))
        elif arm!='TRIG':self.theta=parameter((2,3,2),.1)
        self.register_buffer('paulis',pauli_basis())
        self.register_buffer('diagonal_selector',torch.tensor([all((i//4**q)%4 in (0,3) for q in range(3)) for i in range(1,64)]))
        for n,p in list(self.named_parameters()):self.register_buffer('reference_'+n,p.detach().clone())

    def scalar(self,features,kind,reference=False):
        def p(n):return getattr(self,('reference_' if reference else '')+n)
        hidden=F.layer_norm(features,(features.shape[-1],))
        phi=math.pi*F.linear(hidden,p('projection_'+kind),p('bias_'+kind)).tanh()
        if self.arm=='TRIG':
            # Explicit independent classical63term multiaffine trig replacement.
            atom=torch.cat((phi[...,:3].sin(),phi[...,3:].cos()),-1)
            moments=torch.stack([torch.stack([atom[...,q] for q in range(6) if mask&(1<<q)],-1).prod(-1)
                                 for mask in range(1,64)],-1)
        else:
            if self.arm=='MLP_STATE':
                raw=F.linear(F.linear(phi,p('weight1'),p('bias1')).tanh(),p('weight2'),p('bias2'))
                state=torch.complex(raw[...,:8],raw[...,8:])
                norm=torch.linalg.vector_norm(state,dim=-1,keepdim=True)
                need(bool((norm>1e-8).all()),'nondegenerate learned complexstate')
                state=state/norm
            else:state=statevector(phi,p('theta'),self.arm!='PRODUCT')
            moments=torch.einsum('...i,pij,...j->...p',state.conj(),self.paulis,state).real
            if self.arm=='DEPHASED':moments=moments*self.diagonal_selector
        return moments@p('measurement')

    def forward(self,features,kind,reference=None):
        if reference is None:
            with torch.no_grad():reference=self.scalar(features,kind,True)
        return self.scalar(features,kind)-reference


class NativeFeatureReadout(nn.Module):
    batch_local_references=True

    def __init__(self,packed,arm,seed):
        super().__init__();need(arm in NATIVE_FEATURE_ARMS,'allowlisted fullfeaturearm');self.arm=arm;self.heads=nn.ModuleDict()
        dimensions={}
        for kind,p in packed.items():
            weight,bias=p['original_head']['weight'],p['original_head']['bias'];dimensions[kind]=weight.shape[-1]
            head=nn.Linear(weight.shape[-1],1);head.load_state_dict(dict(weight=weight,bias=bias));head.bias.requires_grad_(False)
            self.heads[kind]=head
        self.shared=None if arm=='HEAD' else NativeFeatureMap(dimensions,arm,seed)

    def references(self,coords,native,features=None):
        need(features is not None,'explicit actual own nativefeatures')
        with torch.no_grad():
            return {k:torch.zeros_like(native[k]) if self.shared is None else self.shared.scalar(features[k],k,True) for k in ('dpr','bge')}

    def gradient_probe_parameter(self):return self.shared.measurement

    def activity_gradient_names(self):
        return () if self.shared is None else ('shared.projection_dpr','shared.projection_bge','shared.measurement')

    def forward_one(self,kind,features,native,original,coords):
        return native+self.heads[kind](features).squeeze(-1)-original+(0 if self.shared is None else self.shared(features,kind))

    def forward(self,features,native,originals,coords,references):
        return {k:native[k]+self.heads[k](features[k]).squeeze(-1)-originals[k]+
                (0 if self.shared is None else self.shared(features[k],k,references[k])) for k in ('dpr','bge')}
