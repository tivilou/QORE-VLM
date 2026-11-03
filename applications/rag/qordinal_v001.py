"""Sequential ordinal measurement of frozen scorer features.

Exact real-amplitude classical simulation. No quantum hardware/speed claim.
Native head and same reviewed ordinal supervision are shared by all arms.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('HEAD','ORDINAL_LINEAR','ORDINAL_MLP','COMMUTING','DEPHASED','QUANTUM','INDEPENDENT')


def need(ok,message):
    if not ok:raise RuntimeError(message)


class OrdinalEffect(nn.Module):
    def __init__(self,kind,dimension,seed):
        super().__init__();self.kind=kind
        generator=torch.Generator(device='cpu').manual_seed(seed)
        if kind in ('ORDINAL_LINEAR','ORDINAL_MLP'):
            self.encoder=(nn.Linear(dimension,2) if kind=='ORDINAL_LINEAR' else
                nn.Sequential(nn.Linear(dimension,16),nn.Tanh(),nn.Linear(16,2)))
            with torch.no_grad():
                for p in self.encoder.parameters():p.copy_(torch.randn(p.shape,generator=generator)*.02)
        else:
            self.useful=nn.Parameter(torch.randn(dimension,8,generator=generator))
            if kind!='COMMUTING':self.direct=nn.Parameter(torch.randn(dimension,8,generator=generator))

    def forward(self,h,trace=False):
        if self.kind in ('ORDINAL_LINEAR','ORDINAL_MLP'):
            conditional=self.encoder(h).sigmoid();pu=conditional[...,0];pd=pu*conditional[...,1]
            details=dict(conditional=conditional)
        else:
            state=F.normalize(h,dim=-1,eps=1e-12)
            qu=torch.linalg.qr(self.useful,mode='reduced').Q
            qd=qu[:,:4] if self.kind=='COMMUTING' else torch.linalg.qr(self.direct,mode='reduced').Q
            useful_amplitudes=state@qu
            projected=useful_amplitudes@qu.T
            if self.kind=='DEPHASED':
                populations=state.square()
                pu=populations@qu.square().sum(-1)
                chained=qu@(qu.T@qd)
                pd=populations@chained.square().sum(-1)
            else:
                pu=useful_amplitudes.square().sum(-1)
                pd=((state@qd).square().sum(-1)*pu if self.kind=='INDEPENDENT'
                    else (projected@qd).square().sum(-1))
            details=dict(state=state,projected_useful_state=projected,
                useful_basis=qu,direct_basis=qd,
                useful_orthogonality_error=(qu.T@qu-torch.eye(8,device=h.device)).abs().max(),
                direct_orthogonality_error=(qd.T@qd-torch.eye(qd.shape[1],device=h.device)).abs().max())
        need(bool(torch.isfinite(pu).all() and torch.isfinite(pd).all()),'finite effect probabilities')
        need(bool((pu>=-1e-6).all() and (pu<=1+1e-6).all() and
                  (pd>=-1e-6).all() and (pd<=pu+1e-6).all()),'ordinal measurement probability order')
        probabilities=torch.stack((1-pu,pu-pd,pd),-1)
        need(bool((probabilities.sum(-1)-1).abs().max()<1e-6),'three-class probability conservation')
        quality=pu+pd
        if trace:return quality,dict(details,probabilities=probabilities,quality=quality)
        return quality,probabilities


class Scorer(nn.Module):
    def __init__(self,kind,weight,bias,seed):
        super().__init__();need(kind in ARMS,'arm identity');self.kind=kind
        self.weight=nn.Parameter(weight.detach().reshape(-1).clone())
        self.register_buffer('bias',bias.detach().reshape(()).clone())
        if kind!='HEAD':
            self.effect=OrdinalEffect(kind,self.weight.numel(),seed)
            self.reference=copy.deepcopy(self.effect).requires_grad_(False)

    def forward(self,h,trace=False):
        score=F.linear(h,self.weight[None],self.bias[None]).squeeze(-1)
        if self.kind=='HEAD':return (score,dict(measurement='not_applicable')) if trace else score
        quality,details=self.effect(h,trace=trace)
        with torch.no_grad():reference,_=self.reference(h)
        value=score+quality-reference
        need(bool(torch.isfinite(value).all()),'finite enhanced score')
        return (value,dict(details,reference_quality=reference,score_correction=quality-reference)) if trace else value


def loss_for(model,h,tiers):
    score=model(h)
    mask=(tiers[:,:,None]>tiers[:,None,:])&(tiers[:,:,None]>=0)&(tiers[:,None,:]>=0)
    counts=mask.sum((-1,-2));difference=score[:,:,None]-score[:,None,:]
    pairwise=((F.softplus(-difference)*mask).sum((-1,-2))/counts.clamp_min(1)).mean()
    if model.kind=='HEAD':return pairwise,counts
    _,probabilities=model.effect(h)
    reviewed=tiers>=0
    # Missing supervision remains excluded. Every question stays in the mean.
    chosen=probabilities.gather(-1,tiers.clamp_min(0)[...,None]).squeeze(-1)
    nll=-chosen.clamp_min(1e-7).log()
    calibration=((nll*reviewed).sum(-1)/reviewed.sum(-1).clamp_min(1)).mean()
    return pairwise+calibration,counts


def train(kind,h,tiers,weight,bias,seed,check,progress):
    model=Scorer(kind,weight.cpu(),bias.cpu(),seed).to(h.device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    initial=copy.deepcopy(model.state_dict());records=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    orders=[torch.randperm(512,generator=generator) for _ in range(4)]
    for step in range(256):
        batch=orders[step//64][(step%64)*8:(step%64+1)*8].to(h.device)
        loss,count=loss_for(model,h[batch],tiers[batch]);need(bool(torch.isfinite(loss)),'finite ordinal loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters),'finite gradient')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        records.append(dict(step=step+1,loss=float(loss.detach()),ordered_pairs=int(count.sum()),
            no_ordered_pairs_questions=int((count==0).sum()),gradient_norm=float(norm)))
        if (step+1)%32==0:check('screen_training');progress(kind,seed,step+1)
    return model,initial,records
