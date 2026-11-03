"""Full-CLS normalized tensor-network Born prototypes; admitted science only.

Two bond-two complex MPS states define E+/-=|phi+/- ><phi+/-|/2.
The reject effect is positive because each projector has norm one. Exact
classical contractions are used; no hardware or computational advantage claim.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qfusion_v001 import need
ARMS=('HEAD','MLP','DENSE','REALTT','PRODUCT','QUANTUM','JOINT_Q')


class Prototype(nn.Module):
    def __init__(self,kind,seed,device):
        super().__init__();self.kind=kind
        gen=torch.Generator(device='cpu').manual_seed(seed)
        if kind=='MLP':
            self.p=nn.Parameter((.1*torch.randn(3081,generator=gen)).to(device))
        elif kind=='DENSE':
            self.p=nn.Parameter(torch.randn(2,1024,2,generator=gen).to(device))
        else:
            bond=1 if kind=='PRODUCT' else 4 if kind=='REALTT' else 2
            self.complex_cores=kind!='REALTT';width=2 if self.complex_cores else 1
            self.cores=nn.ParameterList()
            for k in range(10):
                a=1 if k==0 else bond;b=1 if k==9 else bond
                self.cores.append(nn.Parameter((torch.randn(2,a,2,b,width,generator=gen)/(2*a)**.5).to(device)))

    def core_values(self):
        return [torch.complex(v[...,0],v[...,1] if self.complex_cores else torch.zeros_like(v[...,0])) for v in self.cores]

    def states(self):
        if self.kind=='DENSE':value=torch.complex(self.p[...,0],self.p[...,1])
        else:
            cores=self.core_values();value=torch.ones((2,1,1),dtype=cores[0].dtype,device=cores[0].device)
            for core in cores:value=torch.einsum('psa,patb->pstb',value,core).reshape(2,-1,core.shape[-1])
            value=value[...,0]
        norm=value.abs().square().sum(-1,keepdim=True).sqrt()
        need(bool(torch.isfinite(norm).all()) and bool((norm>1e-10).all()),'nonzero normalized prototypes')
        return value/norm

    def forward(self,h):
        x=F.normalize(h,dim=-1)
        if self.kind=='MLP':
            p=self.p;return (torch.tanh(x@p[:3072].reshape(768,4)+p[3072:3076])*p[3076:3080]).sum(-1)+p[3080]
        psi=F.pad(x,(0,256));phi=self.states()
        amplitude=psi.to(phi.dtype)@phi.conj().T
        probability=amplitude.abs().square()
        return probability[...,0]-probability[...,1]

    def physical_probe(self,h):
        if self.kind=='MLP':return dict(applicable=False),None
        phi=self.states();psi=F.pad(F.normalize(h,dim=-1),(0,256)).to(phi.dtype)
        err=0.
        if self.kind!='DENSE':
            # Independent basis-by-basis matrix products, all 1024 bit strings.
            bits=torch.arange(1024,device=h.device);v=torch.ones((2,1024,1),dtype=phi.dtype,device=h.device)
            for k,core in enumerate(self.core_values()):
                selected=core[:,:,(bits>>(9-k))&1,:].permute(0,2,1,3)
                v=torch.einsum('pna,pnab->pnb',v,selected)
            explicit=v[...,0];explicit=explicit/explicit.abs().square().sum(-1,keepdim=True).sqrt()
            err=float((explicit-phi).abs().max());need(err<1e-5,'independent full basis MPS contraction')
        gram=phi@phi.conj().T
        nonzero_eigenvalues=torch.linalg.eigvalsh(gram*.5)
        probs=(psi@phi.conj().T).abs().square()*.5
        need(float((phi.abs().square().sum(-1)-1).abs().max())<1e-5,'prototype unit norms')
        need(float(nonzero_eigenvalues.max())<=1.00001 and bool((probs>=0).all()) and float(probs.sum(-1).max())<=1.00001,'positive complete three-effect POVM')
        return dict(applicable=True,full_basis_oracle_error=err,maximum_effect_sum_eigenvalue=float(nonzero_eigenvalues.max()),minimum_reject_probability=float((1-probs.sum(-1)).min()),normalized=True),dict(states=phi.detach().cpu(),input_amplitudes=psi.detach().cpu(),probabilities=probs.detach().cpu())


class Scorer(nn.Module):
    def __init__(self,kind,weight,bias,seed,scale):
        super().__init__();self.kind=kind;self.scale=float(scale)
        w=weight.detach().reshape(-1).clone()
        if kind in ('HEAD','JOINT_Q'):self.weight=nn.Parameter(w)
        else:self.register_buffer('weight',w)
        self.register_buffer('bias',bias.detach().reshape(()).clone())
        self.register_buffer('denominator',torch.ones((),device=weight.device))
        if kind!='HEAD':
            self.encoder=Prototype('QUANTUM' if kind=='JOINT_Q' else kind,seed,weight.device)
            self.reference=copy.deepcopy(self.encoder)
            for p in self.reference.parameters():p.requires_grad_(False)

    def forward(self,h,unused):
        base=F.linear(h,self.weight[None],self.bias[None]).squeeze(-1)
        if self.kind=='HEAD':return base
        with torch.no_grad():reference=self.reference(h)
        return base+self.scale*(self.encoder(h)-reference)/self.denominator


def loss_for(model,h,tiers,unused):
    score=model(h,unused);mask=(tiers[:,:,None]>tiers[:,None,:])&(tiers[:,:,None]>=0)&(tiers[:,None,:]>=0);count=mask.sum((-1,-2))
    return ((F.softplus(-(score[:,:,None]-score[:,None,:]))*mask).sum((-1,-2))/count.clamp_min(1)).mean(),count


def train_model(model,h,tiers,data,seed,check,progress):
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    initial=copy.deepcopy(model.state_dict());records=[];g=torch.Generator(device='cpu').manual_seed(seed)
    orders=[torch.randperm(512,generator=g) for _ in range(4)]
    for step in range(256):
        ids=orders[step//64][step%64*8:(step%64+1)*8].to(h.device)
        loss,counts=loss_for(model,h[ids],tiers[ids],data[ids]);need(bool(torch.isfinite(loss)),'finite MPS loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'finite active scorer gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step()
        records.append(dict(step=step+1,loss=float(loss.detach()),supervision_pairs=int(counts.sum()),empty_questions=int((counts==0).sum()),gradient_norm=float(norm)))
        if (step+1)%16==0:check('screen_training');progress(model.kind,seed,step+1)
    return model,initial,records


def fixed_probe(model,h,tiers,data):
    from applications.rag.qregion_numeric_v001 import numeric_probe
    # Numeric helper visits named_parameters; frozen reference buffers must not
    # be mistaken for trainable parameters. Supply a view excluding reference.
    class ActiveView(nn.Module):
        def __init__(self,source):
            super().__init__();self.kind=source.kind;self.scale=source.scale
            if source.kind in ('HEAD','JOINT_Q'):self.weight=source.weight
            else:self.register_buffer('weight',source.weight)
            self.register_buffer('bias',source.bias);self.register_buffer('denominator',source.denominator)
            if source.kind!='HEAD':
                self.encoder=source.encoder
                self.register_buffer('reference_value',source.reference(h).detach())
        def forward(self,x,unused):
            base=F.linear(x,self.weight[None],self.bias[None]).squeeze(-1)
            return base if self.kind=='HEAD' else base+self.scale*(self.encoder(x)-self.reference_value)/self.denominator
    view=ActiveView(model);loss_for(view,h,tiers,data)[0].backward()
    gradients={n:float(p.grad.abs().max()) for n,p in view.named_parameters()}
    need(all(v>0 for v in gradients.values()),'active initial quantum/classical score derivatives')
    numeric=numeric_probe(view,h,tiers,data,loss_for);need(numeric['passed'],'FP32/FP64 MPS directional derivatives')
    physical,trace=(dict(applicable=False),None) if model.kind=='HEAD' else model.encoder.physical_probe(h[0,:2])
    model.zero_grad(set_to_none=True)
    return dict(initial_gradient_maxima=gradients,diagnostic_gradient_maxima=gradients,directional=numeric,circuit_probe=physical),trace
