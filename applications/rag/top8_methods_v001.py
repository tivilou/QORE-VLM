"""Dimension-aware versions of the selected eight mechanisms, admitted workers only.

Old DPR sources stay immutable. BGE retains its native frozen dense/tanh bridge.
Exact classical simulation; no speed or uniquely quantum attribution claim.
"""
from __future__ import annotations
import copy
import inspect
import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from applications.rag.qvalue_portable_v001 import need
from applications.rag.qvalue_geometry_v002 import NativeTail, pairs, generators

IDEAS = ('MIXED_DEPTH', 'FIDELITY', 'VALUE_GEOMETRY', 'LOCAL_REUPLOAD',
         'NATIVE_BORN', 'FEATURE_GATE', 'LOCAL_FFN', 'ORDINAL')
ARMS = {
    'MIXED_DEPTH': ('HEAD','CONCAT','COSINE','HS','DEPHASED','QDA','QUANTUM'),
    'FIDELITY': ('HEAD','MLP','GIVENS','PRODUCT','QUANTUM'),
    'VALUE_GEOMETRY': ('HEAD','CONCAT','MLP','DENSE','LORA','TANGENT','CAYLEY','LOCAL','QUANTUM'),
    'LOCAL_REUPLOAD': ('HEAD','LORA','MLP','TRIG','PRODUCT','REAL','QUANTUM'),
    'NATIVE_BORN': ('HEAD','LINEAR','MLP','TRIG','PRODUCT','COMMUTING','QUANTUM','LOGIT_MLP'),
    'FEATURE_GATE': ('HEAD','LINEAR','MLP','TRIG','PRODUCT','STOCHASTIC','ADDITIVE','QUANTUM'),
    'LOCAL_FFN': ('HEAD','LORA','MLP','QUADRATIC','PRODUCT','REAL','QUANTUM'),
    'ORDINAL': ('HEAD','ORDINAL_LINEAR','ORDINAL_MLP','COMMUTING','DEPHASED','QUANTUM','INDEPENDENT'),
}


class Head(nn.Module):
    def __init__(self, head, kind='HEAD'):
        super().__init__();self.kind=kind
        self.head=copy.deepcopy(head);self.head.weight.requires_grad_(True);self.head.bias.requires_grad_(False)
    def forward(self, row):return self.head(row['readout']).squeeze(-1)


class DepthHead(Head):
    def __init__(self,head,kind,seed):
        super().__init__(head,kind);d=head.in_features
        if kind!='HEAD':self.earlier=nn.Parameter(torch.zeros(2,d,device=head.weight.device))
        if kind=='MLP':
            g=torch.Generator().manual_seed(seed);self.left=nn.Parameter((torch.randn(3*d,4,generator=g)/math.sqrt(3*d)).to(head.weight.device))
            self.right=nn.Parameter(torch.zeros(4,device=head.weight.device))
    def forward(self,row):
        h=row['depths'];out=self.head(h[:,2]).squeeze(-1)
        if self.kind!='HEAD':out=out+(h[:,:2]*self.earlier).sum((1,2))
        if self.kind=='MLP':out=out+torch.tanh(h.flatten(1)@self.left)@self.right
        return out


class ValueGeometry(Head):
    def __init__(self,head,layer,bridge,kind,seed):
        super().__init__(head,kind);self.bridge=bridge;self.tail=NativeTail(layer)
        h=head.in_features//64;self.heads=h
        self.register_buffer('identity',torch.eye(64,device=head.weight.device).expand(h,64,64).clone())
        if kind=='DENSE':self.delta=nn.Parameter(torch.zeros_like(self.identity))
        elif kind=='LORA':
            g=torch.Generator().manual_seed(seed)
            self.left=nn.Parameter(((torch.rand(h,4,64,generator=g)*2-1)/8).to(head.weight.device));self.right=nn.Parameter(torch.zeros(h,64,4,device=head.weight.device))
        else:
            self.angles=nn.Parameter(torch.zeros(h,24,device=head.weight.device))
            if kind in ('TANGENT','CAYLEY'):self.register_buffer('generators',generators(head.weight.device,head.weight.dtype))
    def operator(self):
        if self.kind=='DENSE':return self.identity+self.delta
        if self.kind=='LORA':return self.identity+self.right@self.left
        if self.kind in ('TANGENT','CAYLEY'):
            a=torch.einsum('hk,kij->hij',self.angles,self.generators)
            return self.identity+a if self.kind=='TANGENT' else torch.linalg.solve(self.identity-a/2,self.identity+a/2)
        value=self.identity;j=0
        for layer in range(2):
            for controlled in (False,True):
                for wire in range(6):
                    l,r=pairs(wire,controlled and self.kind!='LOCAL');l=torch.tensor(l,device=value.device);r=torch.tensor(r,device=value.device)
                    a,b=value[:,l],value[:,r];angle=self.angles[:,j,None,None];c,s=(angle/2).cos(),(angle/2).sin()
                    value=value.index_copy(1,l,c*a-s*b).index_copy(1,r,s*a+c*b);j+=1
        return value
    def forward(self,row):
        c=row['context'].reshape(-1,self.heads,64);u=self.operator()
        moved=torch.einsum('hij,nhj->nhi',u,c).flatten(1)
        original=self.tail(row['context'],row['residual'])
        changed=self.tail(moved,row['residual'])
        # Anchor in raw CLS space, then preserve the native nonlinear BGE readout.
        return self.head(self.bridge(row['final']+changed-original)).squeeze(-1)


class LocalEffect(nn.Module):
    def __init__(self,kind,seed,device,width,reupload):
        super().__init__();self.kind=kind;self.reupload=reupload;g=torch.Generator().manual_seed(seed)
        if kind in ('MLP','LORA'):
            n=4 if kind=='MLP' else width
            self.left=nn.Parameter((torch.randn(n,8 if kind=='MLP' else 4,generator=g)/math.sqrt(n)).to(device))
            self.right=nn.Parameter((.1*torch.randn(8 if kind=='MLP' else 4,n,generator=g)).to(device))
        elif kind=='TRIG':self.matrix=nn.Parameter((torch.randn(32,4,generator=g)/math.sqrt(32)).to(device))
        elif kind=='QUADRATIC':self.matrix=nn.Parameter((.1*torch.randn(4,4,4,generator=g)).to(device))
        else:self.p=nn.Parameter((.1*torch.randn(12,generator=g)).to(device))
    def forward(self,z):
        if self.kind=='LORA':return z@self.left@self.right
        shape=z.shape;need(shape[-1]%4==0,'full native local blocks');x=z.reshape(*shape[:-1],shape[-1]//4,4)
        if self.kind=='MLP':return (torch.tanh(x@self.left)@self.right).reshape(shape)
        if self.kind=='QUADRATIC':
            psi=F.normalize(x,dim=-1);m=(self.matrix+self.matrix.transpose(-1,-2))/2
            return (4*x.norm(dim=-1,keepdim=True)*torch.einsum('...i,oij,...j->...o',psi,m,psi)).reshape(shape)
        if self.reupload:
            a=x.atan()
            if self.kind=='TRIG':
                fs=[a.sin(),a.cos()]
                for i in range(4):
                    for j in range(i):
                        fs.extend(((a[...,i:i+1]+a[...,j:j+1]).sin(),(a[...,i:i+1]+a[...,j:j+1]).cos(),(a[...,i:i+1]-a[...,j:j+1]).sin(),(a[...,i:i+1]-a[...,j:j+1]).cos()))
                value=torch.cat(fs,-1)@self.matrix
            else:
                from applications.rag.qreuploadffn_v001 import circuit
                value=circuit(a,self.p,self.kind).abs().square()
        else:
            from applications.rag.qlocalffn_v001 import circuit
            value=circuit(F.normalize(x,dim=-1).to(torch.complex64),self.p,self.kind).abs().square()
        return (4*x.norm(dim=-1,keepdim=True)*value).reshape(shape)


class LocalFFN(Head):
    def __init__(self,head,layer,bridge,kind,seed,reupload):
        super().__init__(head,kind);self.output=layer.output;self.activation=layer.intermediate.intermediate_act_fn
        self.bridge=bridge
        if kind!='HEAD':
            self.effect=LocalEffect(kind,seed,head.weight.device,layer.intermediate.dense.out_features,reupload)
            self.reference=copy.deepcopy(self.effect).requires_grad_(False)
    def forward(self,row):
        if self.kind=='HEAD':return super().forward(row)
        z=row['z'];a=self.activation(z)
        with torch.no_grad():reference=self.reference(z)
        cls=self.output(a+self.effect(z)-reference,row['ffn_residual'])
        return self.head(self.bridge(cls)).squeeze(-1)


class Fidelity(Head):
    def __init__(self,head,layer,bridge,kind,seed):
        super().__init__(head,kind);self.layer=layer;self.bridge=bridge
        if kind!='HEAD':
            from applications.rag.qfidelity_v002 import Fidelity as Kernel
            self.effect=Kernel(kind,seed,head.weight.device)
    def forward(self,row):
        if self.kind=='HEAD':return super().forward(row)
        hidden=row['hidden'];att=self.layer.attention.self
        need(type(att).__name__ in ('BertSdpaSelfAttention','XLMRobertaSelfAttention'),
             'native pinned BERT SDPA / XLM-R eager adapter identity')
        h=att.num_attention_heads;d=att.attention_head_size
        with torch.no_grad():
            q=att.query(hidden).reshape(len(hidden),-1,h,d).transpose(1,2)[:,:,0]
            k=att.key(hidden).reshape(len(hidden),-1,h,d).transpose(1,2)
        delta=self.effect(q,k);l=hidden.shape[1]
        selector=F.one_hot(torch.tensor(0,device=delta.device),l).to(delta.dtype).reshape(1,1,l,1)
        original=row['mask'];base=0. if original is None else original
        if original is not None and original.dtype==torch.bool:
            base=torch.zeros_like(original,dtype=delta.dtype).masked_fill(~original,torch.finfo(delta.dtype).min)
        bias=delta.unsqueeze(2)*selector
        if original is not None:bias=bias*(original if original.dtype==torch.bool else original==0).to(delta.dtype)
        mask=base+bias
        with sdpa_kernel(SDPBackend.MATH):
            cls=self.layer(hidden,attention_mask=mask,output_attentions=False)[0][:,0]
        return self.head(self.bridge(cls)).squeeze(-1)


class MixedScorer(DepthHead):
    def __init__(self,head,kind,seed):
        super().__init__(head,'HEAD' if kind=='HEAD' else 'CONCAT',seed);self.kind=kind
        if kind not in ('HEAD','CONCAT'):self.coefficients=nn.Parameter(torch.zeros(3,device=head.weight.device))
    def forward(self,row):
        score=super().forward(row)
        if self.kind not in ('HEAD','CONCAT'):score=score+row['mixed'][self.kind]@self.coefficients/math.sqrt(3)
        return score


def mixed_banks(features, metadata, check):
    from applications.rag.qmixed_depth_v001 import sym, psd_eigen
    train=[r for r in metadata if r['role']=='training'];folds={r['id']:int(r['component_id'],16)%5 for r in train}
    need(set(folds.values())==set(range(5)), 'all five component folds present')
    d=next(iter(features.values()))['depths'].shape[-1];eye=torch.eye(d,dtype=torch.float64,device='cuda')
    def states(h):
        x=h.double();n=x.norm(dim=-1,keepdim=True);need(bool((n>1e-8).all()), 'nonzero amplitude state');return x/n
    banks={}
    for excluded in list(range(5))+[None]:
        totals=torch.zeros(3,d,d,dtype=torch.float64,device='cuda');means=torch.zeros(3,d,dtype=torch.float64,device='cuda');counts=[0]*3
        for r in train:
            if folds[r['id']]==excluded:continue
            check();u=states(features[r['id']]['depths'].cuda());y=torch.tensor(r['targets'],device='cuda')
            for c in range(3):
                x=u[y==c].reshape(-1,d)
                if len(x):totals[c]+=x.T@x/len(x);means[c]+=x.mean(0);counts[c]+=1
        need(min(counts)>1,'training class support in every fold')
        bank=[]
        for c,n in enumerate(counts):
            sigma=sym((totals[c]+eye/d)/(n+1));mu=means[c]/(n+1);cov=sym(sigma-mu[:,None]*mu[None,:]);cov=sym((cov+eye*cov.trace()/d)/2)
            ev,v=torch.linalg.eigh(cov);need(float(ev.min())>1e-12,'positive QDA covariance')
            bank.append((sigma,mu,sym((v/ev)@v.T),ev.log().sum()))
        banks[excluded]=bank
    raw={};names=ARMS['MIXED_DEPTH'][2:]
    for r in metadata:
        check();u=states(features[r['id']]['depths'].cuda());a=u/math.sqrt(3);diag=u.square().mean(1)
        bank=banks[folds[r['id']]] if r['role']=='training' else banks[None];v={k:[] for k in names}
        for sigma,mu,inv,logdet in bank:
            gram=sym((a@sigma)@a.transpose(-1,-2));f=psd_eigen(gram).sqrt().sum(-1).square();hs=gram.diagonal(dim1=-2,dim2=-1).sum(-1)
            deph=(diag*sigma.diag()).sqrt().sum(-1).square();res=u-mu
            need(bool((f>0).all()) and bool((f<=1+1e-8).all()),'Uhlmann fidelity bounds')
            v['QUANTUM'].append(f.log());v['HS'].append(hs.log());v['DEPHASED'].append(deph.log())
            v['COSINE'].append((u@F.normalize(mu,dim=0)).mean(-1));v['QDA'].append(-.5*((res@inv*res).sum(-1).mean(-1)+logdet))
        raw[r['id']]={k:torch.stack(vals,-1) for k,vals in v.items()}
    stats={}
    for k in names:
        x=torch.cat([raw[r['id']][k] for r in train]);m=x.mean(0);s=x.std(0,unbiased=False);s=torch.where(s>1e-10,s,torch.ones_like(s));stats[k]={'mean':m.cpu(),'std':s.cpu()}
        for r in metadata:
            features[r['id']].setdefault('mixed',{})[k]=((raw[r['id']][k]-m)/s).float().cpu()
    return dict(banks=banks,statistics=stats,folds=folds,normalization='global training OOF moments; same disclosed weak global dependency as DPR')


def build(idea,arm,seed,head,layer,bridge,statistics):
    need(idea in IDEAS and arm in ARMS[idea], 'frozen idea/arm registry')
    if idea=='MIXED_DEPTH':return MixedScorer(head,arm,seed)
    if idea=='FIDELITY':return Fidelity(head,layer,bridge,arm,seed)
    if idea=='VALUE_GEOMETRY':
        return DepthHead(head,arm,seed) if arm in ('HEAD','CONCAT','MLP') else ValueGeometry(head,layer,bridge,arm,seed)
    if idea in ('LOCAL_REUPLOAD','LOCAL_FFN'):return LocalFFN(head,layer,bridge,arm,seed,idea=='LOCAL_REUPLOAD')
    cls={'NATIVE_BORN':('qnative_born_v001',True),'FEATURE_GATE':('qfeature_gate_v001',True),'ORDINAL':('qordinal_v001',False)}[idea]
    import importlib
    constructor=importlib.import_module('applications.rag.'+cls[0]).Scorer
    model=constructor(arm,head.weight,head.bias,seed,*statistics) if cls[1] else constructor(arm,head.weight,head.bias,seed)
    class Wrapped(nn.Module):
        def __init__(self):super().__init__();self.inner=model;self.kind=arm
        def forward(self,row):return self.inner(row['readout'])
    return Wrapped().to(head.weight.device)


def objective(model,idea,row,tiers):
    score=model(row);mask=tiers[:,None]>tiers[None,:];count=mask.sum()
    loss=(F.softplus(-(score[:,None]-score[None,:]))*mask).sum()/count.clamp_min(1)
    if idea=='ORDINAL' and model.kind!='HEAD':
        _,p=model.inner.effect(row['readout']);chosen=p.gather(-1,tiers[:,None]).squeeze(-1)
        loss=loss-chosen.clamp_min(1e-7).log().mean() if len(tiers) else loss
    return loss,count
