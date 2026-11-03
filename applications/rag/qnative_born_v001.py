"""Native-log-odds quantum readout; exploratory source, exact CPU/GPU simulation.

The native logit is encoded in qubit0, not added after the circuit. Zero
trainable rotation is the native scorer. Classical controls receive the same
native logit and train-input-normalized hidden features. No hardware benefit.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('HEAD','LINEAR','TRIG','MLP','LOGIT_MLP','PRODUCT','COMMUTING','QUANTUM')
CIRCUIT_NUMERICS={'calls':0,'max_raw_mass_error':0.0}


def need(ok,message):
    if not ok:raise RuntimeError(message)


def born_logit(population):
    need(bool(torch.isfinite(population).all() and (population>=0).all()),'physical readout')
    # Separate marginals avoid cancellation in 1-p near a confident native logit.
    zero=population[...,0::2].sum(-1);one=population[...,1::2].sum(-1)
    need(bool((zero>0).all() and (one>0).all()),'zero Born marginal: stop, no clipping rescue')
    need(float((zero+one-1).abs().max())<2e-5,'Born mass')
    return one.log()-zero.log()


def circuit(score,angles,theta,kind,trace=False):
    shape=score.shape;s=score.reshape(-1);z=angles.reshape(-1,3)
    indices=torch.arange(16,device=s.device)
    # P(q0=1)/P(q0=0)=exp(native logit), including large negative logits.
    amplitude=torch.stack((torch.sigmoid(-s).sqrt(),torch.sigmoid(s).sqrt()),-1)
    state=torch.zeros((len(s),16),dtype=torch.complex64,device=s.device)
    state[:,0]=amplitude[:,0];state[:,1]=amplitude[:,1]*complex(2**-.5,2**-.5)
    def rotate(value,bit,angle,axis='y',control=None):
        zero=indices[(indices&(1<<bit))==0]
        if control is not None:zero=zero[(zero&(1<<control))!=0]
        one=zero|(1<<bit);c=(angle/2).cos();sine=(angle/2).sin();out=value.clone()
        if axis=='y':
            out[:,zero]=c*value[:,zero]-sine*value[:,one]
            out[:,one]=sine*value[:,zero]+c*value[:,one]
        else:
            out[:,zero]=torch.complex(c,-sine)*value[:,zero]
            out[:,one]=torch.complex(c,sine)*value[:,one]
        return out
    for bit in range(1,4):
        state=rotate(state,bit,z[:,bit-1:bit])
        state=rotate(state,bit,z[:,bit-1:bit],'z')
    for layer in range(2):
        if kind=='COMMUTING':
            # All target-Y rotations controlled by fixed ancilla Z projectors
            # commute. Only four combinations are identifiable across both
            # layers; the unused Z parameters remain nominal, not capacity.
            for j in range(4):state=rotate(state,0,theta[layer,0,j].reshape(1,1))
            for j in range(4):state=rotate(state,0,theta[layer,2,j].reshape(1,1),control=1+j%3)
        else:
            for bit in range(4):
                state=rotate(state,bit,theta[layer,0,bit].reshape(1,1))
                state=rotate(state,bit,theta[layer,1,bit].reshape(1,1),'z')
            for control,target in ((0,1),(1,2),(2,3),(3,0)):
                state=rotate(state,target,theta[layer,2,control].reshape(1,1),
                    control=None if kind=='PRODUCT' else control)
    raw_mass=state.abs().square().sum(-1,keepdim=True)
    need(bool(torch.isfinite(raw_mass).all() and (raw_mass>0).all()),'finite positive Born mass')
    mass_error=float((raw_mass-1).abs().max())
    need(mass_error<2e-5,'raw Born mass')
    CIRCUIT_NUMERICS['calls']+=1
    CIRCUIT_NUMERICS['max_raw_mass_error']=max(CIRCUIT_NUMERICS['max_raw_mass_error'],mass_error)
    # Common normalization leaves the readout log-ratio invariant. The raw
    # mass guard remains strict; this never clips a zero marginal.
    state=state/raw_mass.sqrt()
    probabilities=state.abs().square();result=born_logit(probabilities)
    details=dict(state=state.reshape(*shape,16),basis_probabilities=probabilities.reshape(*shape,16),
                 angles=angles,log_odds=result.reshape(shape)) if trace else None
    return result.reshape(shape),details


def rotated_logit(score,angle):
    # Exact analytic real-Y rotation of the same phased native qubit.
    a=torch.sigmoid(-score).sqrt();b=torch.sigmoid(score).sqrt()
    c=(angle/2).cos();s=(angle/2).sin();cross=2*c*s*a*b*(2**-.5)
    p0=c.square()*a.square()+s.square()*b.square()-cross
    p1=s.square()*a.square()+c.square()*b.square()+cross
    need(bool((p0>0).all() and (p1>0).all()),'analytic rotation marginals')
    return p1.log()-p0.log()


def commuting_oracle(score,angles,theta):
    """Independent eight-branch analytic mixture for commuting target-Y gates."""
    marginals=[]
    a=torch.sigmoid(-score).sqrt();b=torch.sigmoid(score).sqrt()
    for branch in range(8):
        mass=torch.ones_like(score);rotation=theta[:,0,:].sum().expand_as(score)
        for bit in range(3):
            probability=(angles[...,bit]/2).sin().square()
            mass=mass*(probability if branch&(1<<bit) else 1-probability)
        for j in range(4):
            if branch&(1<<(j%3)):rotation=rotation+theta[:,2,j].sum()
        c=(rotation/2).cos();s=(rotation/2).sin()
        cross=2*c*s*a*b*(2**-.5)
        marginals.append((mass*(c.square()*a.square()+s.square()*b.square()-cross),
                          mass*(s.square()*a.square()+c.square()*b.square()+cross)))
    zero=sum(v[0] for v in marginals);one=sum(v[1] for v in marginals)
    need(bool((zero>0).all() and (one>0).all()),'commuting oracle marginal')
    return one.log()-zero.log()


class Scorer(nn.Module):
    def __init__(self,kind,weight,bias,seed,mean,std):
        super().__init__();need(kind in ARMS,'native Born arm')
        self.kind=kind;self.weight=nn.Parameter(weight.detach().reshape(-1).clone())
        self.register_buffer('bias',bias.detach().reshape(()).clone())
        self.register_buffer('input_mean',mean.detach().clone());self.register_buffer('input_std',std.detach().clone())
        if kind=='HEAD':return
        generator=torch.Generator(device='cpu').manual_seed(seed)
        self.encoder=nn.Linear(self.weight.numel(),3)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.randn(self.encoder.weight.shape,generator=generator)*.02)
            self.encoder.bias.zero_()
        if kind in ('PRODUCT','COMMUTING','QUANTUM'):
            self.theta=nn.Parameter(torch.zeros(2,3,4))
        else:
            if kind in ('MLP','LOGIT_MLP'):
                self.hidden=nn.Linear(3,4)
                with torch.no_grad():
                    self.hidden.weight.copy_(torch.randn(self.hidden.weight.shape,generator=generator)*.02)
                    self.hidden.bias.zero_()
                dimension=4
            elif kind=='TRIG':dimension=6
            else:dimension=3
            self.readout=nn.Linear(dimension,1)
            with torch.no_grad():self.readout.weight.zero_();self.readout.bias.zero_()

    def forward(self,h,trace=False):
        native=F.linear(h,self.weight[None],self.bias[None]).squeeze(-1)
        if self.kind=='HEAD':return (native,dict(native=native)) if trace else native
        angles=math.pi*torch.tanh(self.encoder((h-self.input_mean)/self.input_std))
        details={}
        if self.kind in ('PRODUCT','COMMUTING','QUANTUM'):
            score,details=circuit(native,angles,self.theta,self.kind,trace)
        else:
            if self.kind=='TRIG':features=torch.cat((angles.sin(),angles.cos()),-1)
            elif self.kind in ('MLP','LOGIT_MLP'):features=self.hidden(angles/math.pi).tanh()
            else:features=angles/math.pi
            rotation=self.readout(features).squeeze(-1)
            score=native+rotation if self.kind=='LOGIT_MLP' else rotated_logit(native,rotation)
            if trace:details=dict(angles=angles,rotation=rotation)
        need(bool(torch.isfinite(score).all()),'finite native Born score')
        return (score,dict(details or {},native=native,score_correction=score-native)) if trace else score


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
        loss,count=loss_for(model,h[batch],tiers[batch]);need(bool(torch.isfinite(loss)),'finite Born loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters),'finite Born gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        records.append(dict(step=step+1,loss=float(loss.detach()),ordered_pairs=int(count.sum()),
            no_ordered_pairs_questions=int((count==0).sum()),gradient_norm=float(norm)))
        if (step+1)%32==0:check('screen_training');progress(kind,seed,step+1)
    return model,initial,records
