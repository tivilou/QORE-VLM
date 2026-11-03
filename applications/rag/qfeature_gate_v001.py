"""Native-coordinate multiplicative quantum feature gate, exact simulation.

Unlike a scalar score residual, this changes each original feature's weight
conditionally on the input. Classical gates have the same insertion and loss.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('HEAD','LINEAR','MLP','TRIG','PRODUCT','STOCHASTIC','ADDITIVE','QUANTUM')
CIRCUIT_NUMERICS=dict(calls=0,max_raw_mass_error=0.0)


def need(ok,message):
    if not ok:raise RuntimeError(message)


def circuit(angles,theta,kind,trace=False):
    shape=angles.shape[:-1];x=angles.reshape(-1,4)
    indices=torch.arange(16,device=x.device)
    population=kind=='STOCHASTIC'
    state=torch.zeros((len(x),16),dtype=torch.float32 if population else torch.complex64,device=x.device);state[:,0]=1
    def rotate(value,bit,angle,axis='y',control=None):
        zero=indices[(indices&(1<<bit))==0]
        if control is not None:zero=zero[(zero&(1<<control))!=0]
        one=zero|(1<<bit);a=angle/2;c=a.cos();s=a.sin();out=value.clone()
        if population:
            if axis=='y':
                out[:,zero]=c.square()*value[:,zero]+s.square()*value[:,one]
                out[:,one]=s.square()*value[:,zero]+c.square()*value[:,one]
        elif axis=='y':
            out[:,zero]=c*value[:,zero]-s*value[:,one]
            out[:,one]=s*value[:,zero]+c*value[:,one]
        else:
            out[:,zero]=torch.complex(c,-s)*value[:,zero]
            out[:,one]=torch.complex(c,s)*value[:,one]
        return out
    for bit in range(4):
        state=rotate(state,bit,x[:,bit:bit+1])
        state=rotate(state,bit,x[:,bit:bit+1],'z')
    for layer in range(2):
        for bit in range(4):
            state=rotate(state,bit,theta[layer,0,bit].reshape(1,1))
            state=rotate(state,bit,theta[layer,1,bit].reshape(1,1),'z')
        for control,target in ((0,1),(1,2),(2,3),(3,0)):
            state=rotate(state,target,theta[layer,2,control].reshape(1,1),
                control=None if kind=='PRODUCT' else control)
    raw_probabilities=state if population else state.abs().square()
    need(bool(torch.isfinite(raw_probabilities).all()),'finite circuit probabilities')
    raw_mass=raw_probabilities.sum(-1,keepdim=True)
    mass_error=float((raw_mass-1).abs().max())
    CIRCUIT_NUMERICS['calls']+=1
    CIRCUIT_NUMERICS['max_raw_mass_error']=max(CIRCUIT_NUMERICS['max_raw_mass_error'],mass_error)
    need(mass_error<1e-5,'circuit probability conservation')
    # Unit mass is an exact mathematical invariant; FP32 gate accumulation
    # need not preserve it bit for bit. Retain the strict raw-mass guard and
    # normalize differentiably before the stricter marginal bounds. This is
    # not clipping a bad marginal or accepting a larger conservation defect.
    state=state/raw_mass if population else state/raw_mass.sqrt()
    probabilities=state if population else state.abs().square()
    bits=torch.stack([((indices>>bit)&1).float() for bit in range(4)],-1)
    measured=probabilities@bits
    need(bool((measured>=-1e-6).all() and (measured<=1+1e-6).all()),'Born gate bounds')
    return measured.reshape(*shape,4),dict(basis_probabilities=probabilities.reshape(*shape,16),
        state=state.reshape(*shape,16),raw_probability_mass=raw_mass.reshape(*shape)) if trace else None


class FeatureGate(nn.Module):
    def __init__(self,kind,dimension,seed):
        super().__init__();self.kind=kind
        generator=torch.Generator(device='cpu').manual_seed(seed)
        self.encoder=nn.Linear(dimension,4);self.decoder=nn.Linear(4,dimension)
        with torch.no_grad():
            for module in (self.encoder,self.decoder):
                module.weight.copy_(torch.randn(module.weight.shape,generator=generator)*.02)
                module.bias.zero_()
        if kind=='MLP':
            self.measure=nn.Sequential(nn.Linear(4,4),nn.Tanh(),nn.Linear(4,4))
            with torch.no_grad():
                for p in self.measure.parameters():p.copy_(torch.randn(p.shape,generator=generator)*.02)
        if kind in ('PRODUCT','STOCHASTIC','ADDITIVE','QUANTUM'):
            self.theta=nn.Parameter(torch.randn((2,3,4),generator=generator)*.02)

    def forward(self,x,trace=False):
        angles=math.pi*torch.tanh(self.encoder(x));details={}
        if self.kind=='LINEAR':measured=(angles/math.pi+1)/2
        elif self.kind=='TRIG':measured=(angles.cos()+1)/2
        elif self.kind=='MLP':measured=self.measure(angles/math.pi).sigmoid()
        else:measured,details=circuit(angles,self.theta,self.kind,trace)
        gates=self.decoder(measured).sigmoid()
        need(bool(torch.isfinite(gates).all()),'finite feature gate')
        return (gates,dict(details or {},encoding_angles=angles,measurement=measured,gates=gates)) if trace else gates


class Scorer(nn.Module):
    def __init__(self,kind,weight,bias,seed,mean,std):
        super().__init__();need(kind in ARMS,'arm identity');self.kind=kind
        self.weight=nn.Parameter(weight.detach().reshape(-1).clone())
        self.register_buffer('initial_weight',weight.detach().reshape(-1).clone())
        self.register_buffer('bias',bias.detach().reshape(()).clone())
        self.register_buffer('input_mean',mean.detach().clone());self.register_buffer('input_std',std.detach().clone())
        if kind!='HEAD':
            self.effect=FeatureGate(kind,self.weight.numel(),seed)
            self.reference=copy.deepcopy(self.effect).requires_grad_(False)

    def forward(self,h,trace=False):
        if self.kind=='HEAD':
            score=F.linear(h,self.weight[None],self.bias[None]).squeeze(-1)
            return (score,dict(gating='not_applicable')) if trace else score
        x=(h-self.input_mean)/self.input_std
        if trace:gates,details=self.effect(x,True)
        else:gates=self.effect(x);details=None
        with torch.no_grad():reference=self.reference(x)
        factor=1+gates-reference
        need(bool((factor>=-1e-6).all() and (factor<=2+1e-6).all()),'anchored multiplicative range')
        modulated=h if self.kind=='ADDITIVE' else h*factor
        score=F.linear(modulated,self.weight[None],self.bias[None]).squeeze(-1)
        if self.kind=='ADDITIVE':score=score+F.linear(gates-reference,self.initial_weight[None]).squeeze(-1)
        need(bool(torch.isfinite(score).all()),'finite enhanced score')
        if trace:
            native=F.linear(h,self.weight[None],self.bias[None]).squeeze(-1)
            return score,dict(details,reference_gates=reference,feature_factors=factor,
                injection='additive_frozen_readout' if self.kind=='ADDITIVE' else 'multiplicative_native_coordinates',
                modulated_cls=modulated,score_correction=score-native)
        return score


def loss_for(model,h,tiers):
    score=model(h)
    mask=(tiers[:,:,None]>tiers[:,None,:])&(tiers[:,:,None]>=0)&(tiers[:,None,:]>=0)
    counts=mask.sum((-1,-2));difference=score[:,:,None]-score[:,None,:]
    return ((F.softplus(-difference)*mask).sum((-1,-2))/counts.clamp_min(1)).mean(),counts


def train(kind,h,tiers,weight,bias,seed,check,progress,statistics):
    model=Scorer(kind,weight.cpu(),bias.cpu(),seed,*statistics).to(h.device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    initial=copy.deepcopy(model.state_dict());records=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    orders=[torch.randperm(512,generator=generator) for _ in range(4)]
    for step in range(256):
        batch=orders[step//64][(step%64)*8:(step%64+1)*8].to(h.device)
        loss,count=loss_for(model,h[batch],tiers[batch]);need(bool(torch.isfinite(loss)),'finite featuregate loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters),'finite gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        records.append(dict(step=step+1,loss=float(loss.detach()),ordered_pairs=int(count.sum()),
            no_ordered_pairs_questions=int((count==0).sum()),gradient_norm=float(norm)))
        if (step+1)%32==0:check('screen_training');progress(kind,seed,step+1)
    return model,initial,records
