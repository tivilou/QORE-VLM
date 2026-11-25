"""Training-reference quantum kernel scorer; exact classical simulation.

The quantum feature map is fixed. Only the native head and 128 reference
coefficients learn. No hardware, computational advantage or novelty claim.
All feature fitting uses training inputs only; no development labels enter.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('HEAD','COSINE','POLYNOMIAL','RBF_FULL','RBF_ANGLES','PRODUCT','DEPHASED','QUANTUM')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def states(angles,entangled):
    """Little-endian q0; RY(a), CZ ring, RY(b), CZ ring, RZ(c)."""
    size=angles.shape[:-1];a=angles.reshape(-1,12)
    state=torch.zeros((len(a),16),dtype=torch.complex64,device=a.device);state[:,0]=1
    indices=torch.arange(16,device=a.device)
    for layer in range(3):
        for bit in range(4):
            zero=indices[(indices&(1<<bit))==0];one=zero|(1<<bit)
            theta=a[:,4*layer+bit:4*layer+bit+1]/2
            c=theta.cos();s=theta.sin();old=state;state=state.clone()
            if layer<2:
                state[:,zero]=c*old[:,zero]-s*old[:,one]
                state[:,one]=s*old[:,zero]+c*old[:,one]
            else:
                state[:,zero]=torch.complex(c,-s)*old[:,zero]
                state[:,one]=torch.complex(c,s)*old[:,one]
        if entangled and layer<2:
            for left,right in ((0,1),(1,2),(2,3),(3,0)):
                sign=torch.where(((indices>>left)&1)*((indices>>right)&1)>0,-1.,1.)
                state=state*sign
    need(bool(torch.isfinite(state).all()),'finite quantum states')
    need(float((state.abs().square().sum(-1)-1).abs().max())<5e-6,'unitary state normalization')
    return state.reshape(*size,16)


def product_analytic(angles,prototypes):
    """Independent-qubit closed-form fidelity, separate from statevector code."""
    t=(angles[...,:4]+angles[...,4:8])/2
    u=(prototypes[...,:4]+prototypes[...,4:8])/2
    c=angles[...,8:12]/2;d=prototypes[...,8:12]/2
    first=t.cos()[...,None,:]*u.cos()*torch.exp(1j*(c[...,None,:]-d))
    second=t.sin()[...,None,:]*u.sin()*torch.exp(-1j*(c[...,None,:]-d))
    return (first+second).abs().square().prod(-1)


def raw_features(h,bank):
    flat=h.reshape(-1,h.shape[-1]);unit=F.normalize(flat,dim=-1,eps=1e-12)
    similarity=unit@bank['prototype_unit'].T
    angles=((flat-bank['input_mean'])/bank['input_std'])@bank['projection']
    angles=math.pi*torch.tanh(angles)
    q=states(angles,True);p=states(angles,False)
    values={
        'COSINE':similarity,
        'POLYNOMIAL':((similarity+1)/2).square(),
        'RBF_FULL':torch.exp(-(2-2*similarity).clamp_min(0)/bank['full_bandwidth']),
        'RBF_ANGLES':torch.exp(-torch.cdist(angles,bank['prototype_angles']).square()/bank['angle_bandwidth']),
        'PRODUCT':(p.conj()@bank['prototype_product'].T).abs().square(),
        'DEPHASED':q.abs().square()@bank['prototype_quantum'].abs().square().T,
        'QUANTUM':(q.conj()@bank['prototype_quantum'].T).abs().square(),
    }
    for v in values.values():need(bool(torch.isfinite(v).all()),'finite kernel')
    return values,dict(angles=angles,quantum_state=q,product_state=p)


def fit_bank(training,prototype_indices,seed):
    need(training.ndim==3 and training.shape[:2]==(512,50),'training input identity')
    need(len(set(prototype_indices))==128 and min(prototype_indices)>=0 and max(prototype_indices)<25600,'reference membership')
    flat=training.reshape(-1,training.shape[-1]);dimension=flat.shape[-1]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    projection=(torch.randn((dimension,12),generator=generator)/math.sqrt(dimension)).to(flat.device)
    bank=dict(seed=seed,input_mean=flat.mean(0),input_std=flat.std(0,unbiased=False).clamp_min(1e-6),
        projection=projection,prototype_indices=torch.tensor(prototype_indices,dtype=torch.int64))
    prototype=flat[prototype_indices]
    bank['prototype_unit']=F.normalize(prototype,dim=-1,eps=1e-12)
    bank['prototype_angles']=math.pi*torch.tanh(((prototype-bank['input_mean'])/bank['input_std'])@projection)
    bank['prototype_quantum']=states(bank['prototype_angles'],True)
    bank['prototype_product']=states(bank['prototype_angles'],False)
    full=(2-2*bank['prototype_unit']@bank['prototype_unit'].T).clamp_min(0)
    angle=torch.cdist(bank['prototype_angles'],bank['prototype_angles']).square()
    offdiag=~torch.eye(128,device=flat.device,dtype=torch.bool)
    bank['full_bandwidth']=full[offdiag].median().clamp_min(1e-6)
    bank['angle_bandwidth']=angle[offdiag].median().clamp_min(1e-6)
    values,trace=raw_features(training,bank)
    bank['kernel_mean']={k:v.mean(0) for k,v in values.items()}
    bank['kernel_std']={k:v.std(0,unbiased=False).clamp_min(1e-6) for k,v in values.items()}
    # Independent density-vector identity on the first two fixed training questions.
    q=trace['quantum_state'][:100];r=bank['prototype_quantum']
    rho=(q[:,:,None]*q.conj()[:,None,:]).flatten(1)
    sigma=(r[:,:,None]*r.conj()[:,None,:]).flatten(1)
    density=(rho.conj()@sigma.T).real
    density_error=float((density-values['QUANTUM'][:100]).abs().max())
    analytic_error=float((product_analytic(trace['angles'][:100],bank['prototype_angles'])-values['PRODUCT'][:100]).abs().max())
    need(density_error<5e-6 and analytic_error<5e-6,'independent kernel algebra replay')
    gram=(r.conj()@r.T).abs().square()
    mineigen=float(torch.linalg.eigvalsh((gram+gram.T)/2).min())
    need(mineigen>=-1e-4,'prototype quantum Gram PSD')
    difference=float((values['QUANTUM'][:100]-values['PRODUCT'][:100]).abs().max())
    need(difference>1e-5,'quantum map collapsed to product control')
    features={k:((v-bank['kernel_mean'][k])/bank['kernel_std'][k]).reshape(512,50,128) for k,v in values.items()}
    return bank,features,dict(density_identity_error=density_error,product_analytic_error=analytic_error,
        prototype_gram_min_eigenvalue=mineigen,quantum_product_max_difference=difference,
        references=128,fitting_population='training_inputs_only',encoding_trainable_parameters=0)


def transform(h,bank):
    values,_=raw_features(h,bank)
    return {k:((v-bank['kernel_mean'][k])/bank['kernel_std'][k]).reshape(*h.shape[:-1],128) for k,v in values.items()}


class Scorer(nn.Module):
    def __init__(self,kind,weight,bias):
        super().__init__();need(kind in ARMS,'arm identity');self.kind=kind
        self.weight=nn.Parameter(weight.detach().reshape(-1).clone())
        self.register_buffer('bias',bias.detach().reshape(()).clone())
        if kind!='HEAD':self.coefficients=nn.Parameter(torch.zeros(128,device=self.weight.device))

    def forward(self,h,features=None):
        value=F.linear(h,self.weight[None],self.bias[None]).squeeze(-1)
        if self.kind!='HEAD':
            need(features is not None and features.shape==(*h.shape[:-1],128),'kernel feature shape')
            value=value+features@self.coefficients/math.sqrt(128)
        need(bool(torch.isfinite(value).all()),'finite enhanced score')
        return value


def loss_for(model,h,tiers,features=None):
    score=model(h,features)
    mask=(tiers[:,:,None]>tiers[:,None,:])&(tiers[:,:,None]>=0)&(tiers[:,None,:]>=0)
    counts=mask.sum((-1,-2));difference=score[:,:,None]-score[:,None,:]
    return ((F.softplus(-difference)*mask).sum((-1,-2))/counts.clamp_min(1)).mean(),counts


def train(kind,h,tiers,features,weight,bias,seed,check,progress):
    model=Scorer(kind,weight,bias).to(h.device);parameters=list(model.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    initial=copy.deepcopy(model.state_dict());records=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    orders=[torch.randperm(512,generator=generator) for _ in range(4)]
    for step in range(256):
        batch=orders[step//64][(step%64)*8:(step%64+1)*8].to(h.device)
        loss,count=loss_for(model,h[batch],tiers[batch],None if features is None else features[batch])
        need(bool(torch.isfinite(loss)),'finite kernel rank loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'finite head/kernel gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        records.append(dict(step=step+1,loss=float(loss.detach()),ordered_pairs=int(count.sum()),
            no_ordered_pairs_questions=int((count==0).sum()),gradient_norm=float(norm)))
        if (step+1)%32==0:check('screen_training');progress(kind,seed,step+1)
    return model,initial,records
