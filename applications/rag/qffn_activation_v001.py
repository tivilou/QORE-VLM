"""Quantum Born activation inside the native final CLS FFN; admitted science only."""
from __future__ import annotations
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qfusion_v001 import need
ARMS=('HEAD','LORA','MLP','PRODUCT','REAL','QUANTUM','JOINT_Q')


def gate(s,a,w,axis):
    basis=torch.arange(s.shape[-1],device=s.device);bit=((basis>>w)&1).bool();other=s.index_select(-1,basis^(1<<w))
    c,t=torch.cos(a/2),torch.sin(a/2)
    if axis=='Y':return c*s+torch.where(bit,t,-t)*other
    if axis=='Z':return s*torch.exp(torch.where(bit,.5j*a,-.5j*a))
    return c*s-1j*t*other


def circuit(z,p,kind):
    state=F.pad(F.normalize(z,dim=-1),(0,1024)).to(torch.complex128 if z.dtype==torch.float64 else torch.complex64)
    bits=torch.arange(4096,device=z.device)
    for w in range(12):state=gate(state,p[w],w,'Y')
    for w in range(12):state=gate(state,p[12+w],w,'Y' if kind=='REAL' else 'Z')
    for w in range(12):
        moved=gate(state,p[24+w],(w+1)%12,'Y' if kind=='REAL' else 'X')
        state=moved if kind=='PRODUCT' else torch.where(((bits>>w)&1).bool(),moved,state)
    return state


def oracle(z,p,kind):
    """Independent block-matrix gate application over all4096 amplitudes."""
    s=F.pad(F.normalize(z,dim=-1),(0,1024)).to(torch.complex128 if z.dtype==torch.float64 else torch.complex64)
    def apply(a,w,axis,control=None):
        nonlocal s
        stride=1<<w;shape=s.shape;x=s.reshape(*shape[:-1],4096//(2*stride),2,stride)
        u,v=x[...,0,:],x[...,1,:];c,t=torch.cos(a/2),torch.sin(a/2)
        if axis=='Y':r0,r1=c*u-t*v,t*u+c*v
        elif axis=='Z':r0,r1=torch.exp(-.5j*a)*u,torch.exp(.5j*a)*v
        else:r0,r1=c*u-1j*t*v,c*v-1j*t*u
        candidate=torch.stack((r0,r1),-2).reshape(shape)
        if control is None:s=candidate
        else:
            selector=torch.tensor([(i//(1<<control))%2 for i in range(4096)],device=s.device,dtype=torch.bool)
            s=torch.where(selector,candidate,s)
    for w in range(12):apply(p[w],w,'Y')
    for w in range(12):apply(p[12+w],w,'Y' if kind=='REAL' else 'Z')
    for w in range(12):apply(p[24+w],(w+1)%12,'Y' if kind=='REAL' else 'X',None if kind=='PRODUCT' else w)
    return s


class Activation(nn.Module):
    def __init__(self,kind,seed,device):
        super().__init__();self.kind=kind;g=torch.Generator(device='cpu').manual_seed(seed)
        if kind in ('MLP','LORA'):
            self.left=nn.Parameter((torch.randn(3072,4,generator=g)/math.sqrt(3072)).to(device))
            self.right=nn.Parameter((.1*torch.randn(4,3072,generator=g)).to(device))
        else:self.p=nn.Parameter((.1*torch.randn(36,generator=g)).to(device))
    def forward(self,z):
        if self.kind in ('MLP','LORA'):
            x=z@self.left
            return (torch.tanh(x) if self.kind=='MLP' else x)@self.right
        state=circuit(z,self.p,self.kind)
        return state.abs().square()[...,:3072]*(64*z.norm(dim=-1,keepdim=True))
    def physical_probe(self,z):
        if self.kind in ('MLP','LORA'):return dict(applicable=False),None
        with torch.no_grad():
            s=circuit(z,self.p,self.kind);o=oracle(z,self.p,self.kind)
            err=float((s-o).abs().max());norm=float((s.abs().square().sum(-1)-1).abs().max())
            need(err<1e-5 and norm<2e-5,'full4096 block-matrix unitary replay and Born norm')
            difference=float((circuit(z,self.p,'QUANTUM')-circuit(z,self.p,'PRODUCT')).abs().max())
            need(difference>1e-8,'nontrivial controlled circuit intervention')
            return dict(applicable=True,qubits=12,full_state_oracle_error=err,norm_error=norm,controlled_vs_product_difference=difference,cropped_probability_not_renormalized=True),dict(preactivation=z.cpu(),state=s.cpu(),oracle=o.cpu(),parameters=self.p.detach().cpu())


class Scorer(nn.Module):
    def __init__(self,kind,weight,bias,seed,scale,ffn):
        super().__init__();self.kind=kind
        w=weight.detach().reshape(-1).clone()
        if kind in ('HEAD','JOINT_Q'):self.weight=nn.Parameter(w)
        else:self.register_buffer('weight',w)
        self.register_buffer('bias',bias.detach().reshape(()).clone())
        for name in ('output_weight','output_bias','norm_weight','norm_bias'):self.register_buffer(name,ffn[name].to(weight.device).clone())
        self.eps=float(ffn['eps']);self.register_buffer('activation_scale',torch.ones((),device=weight.device))
        if kind!='HEAD':
            self.encoder=Activation('QUANTUM' if kind=='JOINT_Q' else kind,seed,weight.device)
            self.reference=copy.deepcopy(self.encoder)
            for p in self.reference.parameters():p.requires_grad_(False)
    def named_parameters(self,prefix='',recurse=True,remove_duplicate=True):
        for name,p in super().named_parameters(prefix=prefix,recurse=recurse,remove_duplicate=remove_duplicate):
            if p.requires_grad:yield name,p
    def forward(self,h,data):
        z,r=data[...,:3072],data[...,3072:];a=F.gelu(z)
        if self.kind!='HEAD':
            with torch.no_grad():ref=self.reference(z)
            a=a+self.activation_scale*(self.encoder(z)-ref)
        cls=F.layer_norm(F.linear(a,self.output_weight,self.output_bias)+r,(768,),self.norm_weight,self.norm_bias,self.eps)
        return F.linear(cls,self.weight[None],self.bias[None]).squeeze(-1)


def fixed_probe(model,h,tiers,data):
    from applications.rag.qmps_born_v001 import loss_for
    from applications.rag.qregion_numeric_v001 import numeric_probe
    loss_for(model,h,tiers,data)[0].backward();gradients={n:float(p.grad.abs().max()) for n,p in model.named_parameters()}
    need(all(v>0 for v in gradients.values()),'active native FFN quantum/classical derivatives')
    numeric=numeric_probe(model,h,tiers,data,loss_for);need(numeric['passed'],'same fixed FFN probe FP32/FP64 derivatives')
    physical,trace=(dict(applicable=False),None) if model.kind=='HEAD' else model.encoder.physical_probe(data[0,:2,:3072])
    model.zero_grad(set_to_none=True)
    return dict(initial_gradient_maxima=gradients,diagnostic_gradient_maxima=gradients,directional=numeric,circuit_probe=physical),trace
