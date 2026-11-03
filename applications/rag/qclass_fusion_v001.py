"""Class-conditioned two-scorer calibration. Exact classical Born simulation."""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qfusion_v001 import circuit,matrix_oracle,need

ARMS=('HEAD','FULLCLASS','LINEAR','POLY','TRIG','MLP','PRODUCT','QUANTUM')


def basis(kind,z):
    if kind=='LINEAR':return z
    a,b=torch.atan(z).unbind(-1)
    if kind=='POLY':extra=(a*a,b*b,a*b,a*a*b,a*b*b,a**3,b**3,a*a*b*b)
    else:extra=(a.sin(),b.sin(),a.cos(),b.cos(),(a+b).sin(),(a-b).sin(),(a+b).cos(),(a-b).cos())
    return torch.cat((z,torch.stack(extra,-1)),-1)


class Scorer(nn.Module):
    def __init__(self,kind,weight,bias,mean,std,seed,scale,denominator=1.,priors=None):
        super().__init__();self.kind=kind;self.denominator=1.
        self.register_buffer('mean',mean.detach().clone());self.register_buffer('std',std.detach().clone())
        self.register_buffer('priors',priors.detach().clone())
        self.register_buffer('bias',bias.detach().reshape(()).clone())
        generator=torch.Generator(device='cpu').manual_seed(seed)
        def draw(*shape):return (.1*torch.randn(shape,generator=generator)).to(mean.device)
        if kind=='HEAD':self.weight=nn.Parameter(weight.detach().reshape(-1).clone())
        elif kind=='FULLCLASS':
            self.class_weight=nn.Parameter(draw(768,3)/768**.5);self.class_bias=nn.Parameter(torch.zeros(3,device=mean.device))
        elif kind in ('LINEAR','POLY','TRIG'):
            d=2 if kind=='LINEAR' else 10
            self.class_weight=nn.Parameter(draw(d,3)/d**.5);self.class_bias=nn.Parameter(torch.zeros(3,device=mean.device))
        elif kind=='MLP':
            self.hidden_weight=nn.Parameter(draw(2,2));self.hidden_bias=nn.Parameter(draw(2))
            self.class_weight=nn.Parameter(draw(2,3));self.class_bias=nn.Parameter(torch.zeros(3,device=mean.device))
        else:
            self.p=nn.Parameter(draw(2,2,2));self.gain=nn.Parameter(torch.ones(2,device=mean.device));self.offset=nn.Parameter(torch.zeros(2,device=mean.device))

    def angles(self,scores):
        # Two RY uploads then span (0,pi), preserving the sign of relevance.
        # Using atan(z) twice would make the unrotated Born probability even.
        return torch.pi/4+torch.atan(((scores-self.mean)/self.std)*self.gain+self.offset)/2

    def probabilities(self,h,scores):
        if self.kind in ('PRODUCT','QUANTUM'):
            _,state=circuit(self.angles(scores),self.p,self.kind=='QUANTUM')
            born=state.abs().square()
            classes=torch.stack((born[...,0],born[...,1],born[...,2]+born[...,3]),-1)
            # Correct measurement-subspace dimensions before train-prior odds.
            evidence=classes/classes.new_tensor([1.,1.,2.])
            weighted=evidence.clamp_min(1e-6)*self.priors
            return weighted/weighted.sum(-1,keepdim=True)
        z=(scores-self.mean)/self.std
        if self.kind=='FULLCLASS':features=h
        elif self.kind=='MLP':features=torch.tanh(z@self.hidden_weight+self.hidden_bias)
        else:features=basis(self.kind,z)
        logits=features@self.class_weight+self.class_bias+self.priors.log()
        return logits.softmax(-1)

    def forward(self,h,scores):
        if self.kind=='HEAD':return F.linear(h,self.weight[None],self.bias[None]).squeeze(-1)
        p=self.probabilities(h,scores)
        need(bool(torch.isfinite(p).all()),'finite class posterior')
        # Expected ORIGINAL ordered label, with no dev-tuned utility weights.
        return p[...,1]+2*p[...,2]


def loss_for(model,h,tiers,scores):
    if model.kind=='HEAD':
        value=model(h,scores);mask=(tiers[:,:,None]>tiers[:,None,:])&(tiers[:,:,None]>=0)&(tiers[:,None,:]>=0)
        count=mask.sum((-1,-2))
        return ((F.softplus(-(value[:,:,None]-value[:,None,:]))*mask).sum((-1,-2))/count.clamp_min(1)).mean(),count
    p=model.probabilities(h,scores)
    mask=tiers>=0;count=mask.sum(-1)
    selected=torch.gather(p,-1,tiers.clamp_min(0)[...,None]).squeeze(-1)
    loss=(-(selected.clamp_min(1e-8).log())*mask).sum(-1)/count.clamp_min(1)
    return loss.mean(),count


def train(model,h,tiers,scores,seed,check,progress):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    initial=copy.deepcopy(model.state_dict());records=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    orders=[torch.randperm(512,generator=generator) for _ in range(4)]
    for step in range(256):
        batch=orders[step//64][(step%64)*8:(step%64+1)*8].to(h.device)
        loss,count=loss_for(model,h[batch],tiers[batch],scores[batch])
        need(bool(torch.isfinite(loss)),'finite class-calibration loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'finite class-calibration gradients')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        records.append(dict(step=step+1,loss=float(loss.detach()),supervision_positions_or_head_pairs=int(count.sum()),empty_questions=int((count==0).sum()),gradient_norm=float(norm)))
        if (step+1)%32==0:check('screen_training');progress(model.kind,seed,step+1)
    return model,initial,records
