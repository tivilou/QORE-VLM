"""Two frozen cross-encoder scores, controlled coherent calibration.

Exact two-qubit classical simulation; this is a two-model ensemble, not
single-model transfer, quantum hardware, or a claimed computational advantage.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('HEAD','LINEAR','POLY','TRIG','MLP','PRODUCT','QUANTUM')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def gate(state,angle,wire,axis):
    """Big endian |q0 q1>; independent batched RY or RZ on one wire."""
    shape=state.shape
    x=state.reshape(*shape[:-1],2,2)
    if wire==0:x=x.transpose(-1,-2)
    c=torch.cos(angle/2);s=torch.sin(angle/2)
    a,b=x[...,0],x[...,1]
    if axis=='Y':out=torch.stack((c[...,None]*a-s[...,None]*b,s[...,None]*a+c[...,None]*b),-1)
    else:out=torch.stack((torch.exp(-.5j*angle)[...,None]*a,torch.exp(.5j*angle)[...,None]*b),-1)
    if wire==0:out=out.transpose(-1,-2)
    return out.reshape(shape)


def circuit(x,theta,entangle):
    state=torch.zeros((*x.shape[:-1],4),dtype=torch.complex64,device=x.device)
    state[...,0]=1
    for layer in range(2):
        for wire in range(2):state=gate(state,x[...,wire],wire,'Y')
        for wire in range(2):
            state=gate(state,theta[layer,wire,0].expand(x.shape[:-1]),wire,'Z')
            state=gate(state,theta[layer,wire,1].expand(x.shape[:-1]),wire,'Y')
        if entangle:state=state*state.new_tensor([1,1,1,-1])
    probabilities=state.abs().square()
    zz=(probabilities*probabilities.new_tensor([1,-1,-1,1])).sum(-1)
    xx=2*(state[...,0].conj()*state[...,3]+state[...,1].conj()*state[...,2]).real
    return (zz+xx)/2,state


def raw(kind,x,p):
    if kind in ('QUANTUM','PRODUCT'):return circuit(x,p.reshape(2,2,2),kind=='QUANTUM')[0]
    a,b=x.unbind(-1)
    if kind=='POLY':basis=torch.stack((a*a,b*b,a*b,a*a*b,a*b*b,a**3,b**3,a*a*b*b),-1)
    elif kind=='TRIG':basis=torch.stack((a.sin(),b.sin(),a.cos(),b.cos(),(a+b).sin(),(a-b).sin(),(a+b).cos(),(a-b).cos()),-1)
    elif kind=='MLP':return torch.tanh(x@p[:4].reshape(2,2)+p[4:6])@p[6:8]
    else:raise RuntimeError('nonlinear control identity')
    return basis@p


class Scorer(nn.Module):
    def __init__(self,kind,weight,bias,mean,std,seed,scale,denominator=1.):
        super().__init__();self.kind=kind;self.scale=float(scale);self.denominator=float(denominator)
        self.register_buffer('mean',mean.detach().clone());self.register_buffer('std',std.detach().clone())
        self.register_buffer('bias',bias.detach().reshape(()).clone())
        if kind=='HEAD':self.weight=nn.Parameter(weight.detach().reshape(-1).clone())
        else:
            self.linear=nn.Parameter(torch.tensor([float(std[0]),0.],device=mean.device))
            if kind!='LINEAR':
                generator=torch.Generator(device='cpu').manual_seed(seed)
                self.p=nn.Parameter((.1*torch.randn(8,generator=generator)).to(mean.device))
                self.register_buffer('reference',self.p.detach().clone())

    def features(self,scores):return (scores-self.mean)/self.std

    def forward(self,h,scores):
        if self.kind=='HEAD':return F.linear(h,self.weight[None],self.bias[None]).squeeze(-1)
        z=self.features(scores)
        # The intercept restores native DPR units exactly at initialization.
        value=z@self.linear+self.mean[0]
        if self.kind!='LINEAR':
            x=torch.atan(z)
            value=value+self.scale*(raw(self.kind,x,self.p)-raw(self.kind,x,self.reference))/self.denominator
        need(bool(torch.isfinite(value).all()),'finite fusion score')
        return value


def loss_for(model,h,tiers,scores):
    score=model(h,scores)
    mask=(tiers[:,:,None]>tiers[:,None,:])&(tiers[:,:,None]>=0)&(tiers[:,None,:]>=0)
    count=mask.sum((-1,-2))
    return ((F.softplus(-(score[:,:,None]-score[:,None,:]))*mask).sum((-1,-2))/count.clamp_min(1)).mean(),count


def train(model,h,tiers,scores,seed,check,progress):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    initial=copy.deepcopy(model.state_dict());records=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    orders=[torch.randperm(512,generator=generator) for _ in range(4)]
    for step in range(256):
        batch=orders[step//64][(step%64)*8:(step%64+1)*8].to(h.device)
        loss,count=loss_for(model,h[batch],tiers[batch],scores[batch])
        need(bool(torch.isfinite(loss)),'finite fusion loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'finite fusion gradients')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        records.append(dict(step=step+1,loss=float(loss.detach()),ordered_pairs=int(count.sum()),no_ordered_pairs_questions=int((count==0).sum()),gradient_norm=float(norm)))
        if (step+1)%32==0:check('screen_training');progress(model.kind,seed,step+1)
    return model,initial,records


def matrix_oracle(x,theta,entangle):
    """Separate full 4x4 construction for real admitted circuit verification."""
    eye=torch.eye(2,dtype=torch.complex64,device=x.device)
    state=eye.new_tensor([1,0,0,0])
    def rotation(a,axis):
        if axis=='Y':return torch.stack((torch.stack((torch.cos(a/2),-torch.sin(a/2))),torch.stack((torch.sin(a/2),torch.cos(a/2))))).to(torch.complex64)
        return torch.diag(torch.stack((torch.exp(-.5j*a),torch.exp(.5j*a))))
    for layer in range(2):
        for wire in range(2):
            r=rotation(x[wire],'Y');state=(torch.kron(r,eye) if wire==0 else torch.kron(eye,r))@state
        for wire in range(2):
            for axis,k in [('Z',0),('Y',1)]:
                r=rotation(theta[layer,wire,k],axis);state=(torch.kron(r,eye) if wire==0 else torch.kron(eye,r))@state
        if entangle:state=torch.diag(eye.new_tensor([1,1,1,-1]))@state
    zz=eye.new_tensor([1,-1,-1,1]);xx=2*(state[0].conj()*state[3]+state[1].conj()*state[2]).real
    return ((state.abs().square()*zz).sum()+xx)/2,state
