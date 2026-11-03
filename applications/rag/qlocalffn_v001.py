"""Shared local two-qubit Born channels inside native FFN; admitted science only."""
import copy,math
import torch
from torch.nn import functional as F
from applications.rag.qffn_activation_v001 import Scorer as Base,Activation,gate,fixed_probe
from applications.rag.qfusion_v001 import need
ARMS=('HEAD','LORA','MLP','QUADRATIC','PRODUCT','REAL','QUANTUM')
FROZEN_FFN={'output_weight','output_bias','norm_weight','norm_bias'}

def operations(kind):
    result=[]
    for layer in range(2):
        result.extend((6*layer+w,w,'Y',None) for w in range(2))
        result.extend((6*layer+2+w,w,'Y' if kind=='REAL' else 'Z',None) for w in range(2))
        result.extend((6*layer+4+w,1-w,'Y' if kind=='REAL' else 'X',None if kind=='PRODUCT' else w) for w in range(2))
    return result

def circuit(s,p,kind,independent=False):
    bits=torch.arange(4,device=s.device)
    for j,w,axis,control in operations(kind):
        if not independent:value=gate(s,p[j],w,axis)
        else:
            stride=1<<w;shape=s.shape;x=s.reshape(*shape[:-1],4//(2*stride),2,stride)
            u,v=x[...,0,:],x[...,1,:];c,t=torch.cos(p[j]/2),torch.sin(p[j]/2)
            if axis=='Y':r0,r1=c*u-t*v,t*u+c*v
            elif axis=='Z':r0,r1=torch.exp(-.5j*p[j])*u,torch.exp(.5j*p[j])*v
            else:r0,r1=c*u-1j*t*v,c*v-1j*t*u
            value=torch.stack((r0,r1),-2).reshape(shape)
        s=value if control is None else torch.where(((bits>>control)&1).bool(),value,s)
    return s

class LocalActivation(torch.nn.Module):
    def __init__(self,kind,seed,device):
        super().__init__();self.kind=kind;g=torch.Generator(device='cpu').manual_seed(seed)
        if kind=='MLP':
            self.left=torch.nn.Parameter((torch.randn(4,8,generator=g)/2).to(device));self.right=torch.nn.Parameter((.1*torch.randn(8,4,generator=g)).to(device))
        elif kind=='QUADRATIC':self.matrix=torch.nn.Parameter((.1*torch.randn(4,4,4,generator=g)).to(device))
        else:self.p=torch.nn.Parameter((.1*torch.randn(12,generator=g)).to(device))
    def forward(self,z):
        shape=z.shape;x=z.reshape(*shape[:-1],768,4)
        if self.kind=='MLP':return (torch.tanh(x@self.left)@self.right).reshape(shape)
        n=x.norm(dim=-1,keepdim=True);psi=x/n.clamp_min(1e-12)
        if self.kind=='QUADRATIC':
            m=(self.matrix+self.matrix.transpose(-1,-2))/2
            value=torch.einsum('...i,oij,...j->...o',psi,m,psi)
        else:
            dtype=torch.complex128 if z.dtype==torch.float64 else torch.complex64
            value=circuit(psi.to(dtype),self.p,self.kind).abs().square()
        return (4*n*value).reshape(shape)
    def physical_probe(self,z):
        if self.kind in ('MLP','QUADRATIC'):return dict(applicable=False),None
        with torch.no_grad():
            x=z.reshape(-1,4);n=x.norm(dim=-1,keepdim=True);psi=(x/n.clamp_min(1e-12)).to(torch.complex64)
            actual=circuit(psi,self.p,self.kind);oracle=circuit(psi,self.p,self.kind,True)
            err=float((actual-oracle).abs().max());norm=float((actual.abs().square().sum(-1)-(n.squeeze(-1)>1e-12).float()).abs().max())
            diff=float((circuit(psi,self.p,'QUANTUM')-circuit(psi,self.p,'PRODUCT')).abs().max())
            need(err<1e-5 and norm<2e-5 and diff>1e-8,'all local4 amplitudes independent oracle/norm/controlled intervention')
            return dict(applicable=True,qubits_per_block=2,blocks=768,shared_angles=12,full_state_oracle_error=err,norm_error=norm,controlled_vs_product_difference=diff),dict(input=psi.cpu(),state=actual.cpu(),oracle=oracle.cpu(),parameters=self.p.detach().cpu())

class Scorer(Base):
    def __init__(self,kind,weight,bias,seed,scale,ffn):
        super().__init__('LORA' if kind=='QUADRATIC' else kind,weight,bias,seed,scale,ffn);self.kind=kind
        if kind!='HEAD':
            w=self.weight.detach().clone();del self.weight;self.weight=torch.nn.Parameter(w)
            if kind!='LORA':self.encoder=LocalActivation(kind,seed,weight.device)
            self.reference=copy.deepcopy(self.encoder)
            for p in self.reference.parameters():p.requires_grad_(False)

def compact_state(state):return {k:v for k,v in state.items() if k not in FROZEN_FFN}

def load_compact(model,state):
    result=model.load_state_dict(state,strict=False)
    need(set(result.missing_keys)==FROZEN_FFN and not result.unexpected_keys,'only source-bound frozenFFN buffers omitted')
