"""Dimension-aware versions of the selected eight mechanisms, admitted workers only.

Old DPR sources stay immutable. BGE retains its native frozen dense/tanh bridge.
Exact classical simulation; no speed or uniquely quantum attribution claim.
"""
from __future__ import annotations
import copy
from collections import OrderedDict
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


@contextmanager
def _frozen_projections(hidden, projections):
    """Reuse same-shape Linear outputs within one serial frozen-layer call.

    Keep nn.Module.__call__ and its hooks; restore instance forwards even when
    native attention raises. No projection weight, parameter or state key changes.
    """
    saved=[];missing=object()
    try:
        for module,value in projections:
            own=module.__dict__.get('forward',missing)
            saved.append((module,own))
            def forward(x, *, value=value):
                need(x is hidden, 'frozen projection must consume the identical native hidden tensor')
                return value
            module.forward=forward
        yield
    finally:
        for module,own in reversed(saved):
            if own is missing:module.__dict__.pop('forward',None)
            else:module.forward=own


class Fidelity(Head):
    # Device tensors, detached and cell-local; model + final replay <=64 MiB.
    REFERENCE_CACHE_BYTES=32*1024**2
    def __init__(self,head,layer,bridge,kind,seed):
        super().__init__(head,kind);self.layer=layer;self.bridge=bridge
        self.cls_only=False  # Selected by the checkpoint-owned execution route.
        self._references=OrderedDict();self._reference_bytes=0;self._reference_binding=None
        if kind!='HEAD':
            from applications.rag.qfidelity_v002 import Fidelity as Kernel
            self.effect=Kernel(kind,seed,head.weight.device)
    def clear_references(self):
        self._references.clear();self._reference_bytes=0;self._reference_binding=None
    def _apply(self,fn,recurse=True):
        self.clear_references()
        return super()._apply(fn,recurse=recurse)
    def _reference(self,row,q,k,cacheable):
        key=row.get('frozen_sequence_key') if cacheable else None
        if key is None:
            with torch.no_grad():return self.effect.kernel(q,k,True)
        # Tensor versions cover in-place edits and state restore; identity covers
        # replacement, dtype/device conversion and a different frozen backbone.
        tensors=list(self.layer.attention.self.query.parameters())+list(self.layer.attention.self.key.parameters())
        tensors += [v for n,v in self.effect.named_buffers() if n.startswith('reference_')]
        binding=tuple((id(v),v._version,str(v.device),str(v.dtype)) for v in tensors)
        if self._reference_binding!=binding:
            self.clear_references();self._reference_binding=binding
        key=(key,tuple(k.shape),str(k.device),str(k.dtype))
        if key in self._references:
            value=self._references.pop(key);self._references[key]=value
            return value
        with torch.no_grad():value=self.effect.kernel(q,k,True).detach()
        size=value.numel()*value.element_size()
        if size<=self.REFERENCE_CACHE_BYTES:
            while self._references and self._reference_bytes+size>self.REFERENCE_CACHE_BYTES:
                _,old=self._references.popitem(last=False)
                self._reference_bytes-=old.numel()*old.element_size()
            self._references[key]=value;self._reference_bytes+=size
        return value
    def forward(self,row):
        if self.kind=='HEAD':return super().forward(row)
        hidden=row['hidden'];att=self.layer.attention.self
        need(type(att).__name__ in ('BertSdpaSelfAttention','XLMRobertaSelfAttention'),
             'native pinned BERT SDPA / XLM-R eager adapter identity')
        h=att.num_attention_heads;d=att.attention_head_size
        with torch.no_grad():
            query=att.query(hidden);key=att.key(hidden)
            q=query.reshape(len(hidden),-1,h,d).transpose(1,2)[:,:,0]
            k=key.reshape(len(hidden),-1,h,d).transpose(1,2)
        frozen=not hidden.requires_grad and not any(p.requires_grad for m in (att.query,att.key) for p in m.parameters())
        # Instrumented/custom projections retain their original double-call path.
        reusable=frozen and all(type(m) is nn.Linear and not
            (m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or 'forward' in m.__dict__)
            for m in (att.query,att.key))
        reference=self._reference(row,q,k,reusable)
        delta=self.effect.kernel(q,k)-reference;l=hidden.shape[1]
        if self.cls_only:
            need(not self.layer.training and getattr(att,'position_embedding_type','absolute')=='absolute',
                 'CLS route requires frozen eval attention with absolute positions')
            value=att.value(hidden).reshape(len(hidden),-1,h,d).transpose(1,2)
            original=row['mask']
            if original is None:mask=delta.unsqueeze(2)
            else:
                original=original if original.shape[-2]==1 else original[...,:1,:]
                if original.dtype==torch.bool:
                    base=torch.zeros_like(original,dtype=delta.dtype).masked_fill(~original,torch.finfo(delta.dtype).min)
                    mask=base+delta.unsqueeze(2)*original.to(delta.dtype)
                else:mask=original+delta.unsqueeze(2)*(original==0).to(delta.dtype)
            if type(att).__name__=='BertSdpaSelfAttention':
                with sdpa_kernel(SDPBackend.MATH):
                    context=F.scaled_dot_product_attention(q.unsqueeze(2),k,value,attn_mask=mask,dropout_p=0.)
            else:
                probabilities=F.softmax(torch.matmul(q.unsqueeze(2),k.transpose(-1,-2))/math.sqrt(d)+mask,dim=-1)
                context=torch.matmul(probabilities,value)
            context=context.transpose(1,2).contiguous().reshape(len(hidden),1,h*d)
            attention=self.layer.attention.output(context,hidden[:,:1])
            cls=self.layer.output(self.layer.intermediate(attention),attention)[:,0]
            return self.head(self.bridge(cls)).squeeze(-1)
        selector=F.one_hot(torch.tensor(0,device=delta.device),l).to(delta.dtype).reshape(1,1,l,1)
        original=row['mask'];base=0. if original is None else original
        if original is not None and original.dtype==torch.bool:
            base=torch.zeros_like(original,dtype=delta.dtype).masked_fill(~original,torch.finfo(delta.dtype).min)
        bias=delta.unsqueeze(2)*selector
        if original is not None:bias=bias*(original if original.dtype==torch.bool else original==0).to(delta.dtype)
        mask=base+bias
        with sdpa_kernel(SDPBackend.MATH):
            if reusable:
                with _frozen_projections(hidden,((att.query,query),(att.key,key))):
                    cls=self.layer(hidden,attention_mask=mask,output_attentions=False)[0][:,0]
            else:cls=self.layer(hidden,attention_mask=mask,output_attentions=False)[0][:,0]
        return self.head(self.bridge(cls)).squeeze(-1)


class MixedScorer(DepthHead):
    def __init__(self,head,kind,seed,label_classes=3):
        super().__init__(head,'HEAD' if kind=='HEAD' else 'CONCAT',seed);self.kind=kind
        if kind not in ('HEAD','CONCAT'):self.coefficients=nn.Parameter(torch.zeros(label_classes,device=head.weight.device))
    def forward(self,row):
        score=super().forward(row)
        if self.kind not in ('HEAD','CONCAT'):score=score+row['mixed'][self.kind]@self.coefficients/math.sqrt(3)
        return score


def mixed_banks(features, metadata, check,label_classes=3):
    need(label_classes in (2,3),'explicit binary or tiered training semantics')
    from applications.rag.qmixed_depth_v001 import sym, psd_eigen
    train=[r for r in metadata if r['role']=='training'];folds={r['id']:int(r['component_id'],16)%5 for r in train}
    need(set(folds.values())==set(range(5)), 'all five component folds present')
    d=next(iter(features.values()))['depths'].shape[-1];eye=torch.eye(d,dtype=torch.float64,device='cuda')
    def states(h):
        x=h.double();n=x.norm(dim=-1,keepdim=True);need(bool((n>1e-8).all()), 'nonzero amplitude state');return x/n
    banks={}
    for excluded in list(range(5))+[None]:
        totals=torch.zeros(label_classes,d,d,dtype=torch.float64,device='cuda');means=torch.zeros(label_classes,d,dtype=torch.float64,device='cuda');counts=[0]*label_classes
        for r in train:
            if folds[r['id']]==excluded:continue
            check();u=states(features[r['id']]['depths'].cuda());y=torch.tensor(r['targets'],device='cuda')
            for c in range(label_classes):
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
            value=((raw[r['id']][k]-m)/s).float().cpu()
            if hasattr(features,'set_aux'):
                mixed=features.aux.get(r['id'],{}).get('mixed',{})
                mixed[k]=value;features.set_aux(r['id'],'mixed',mixed)
            else:features[r['id']].setdefault('mixed',{})[k]=value
    return dict(banks=banks,statistics=stats,folds=folds,normalization='global training OOF moments; same disclosed weak global dependency as DPR')


def build(idea,arm,seed,head,layer,bridge,statistics,label_classes=3):
    need(label_classes in (2,3),'explicit training label classes')
    need(idea in IDEAS and arm in ARMS[idea], 'frozen idea/arm registry')
    if idea=='MIXED_DEPTH':return MixedScorer(head,arm,seed,label_classes)
    if idea=='FIDELITY':return Fidelity(head,layer,bridge,arm,seed)
    if idea=='VALUE_GEOMETRY':
        return DepthHead(head,arm,seed) if arm in ('HEAD','CONCAT','MLP') else ValueGeometry(head,layer,bridge,arm,seed)
    if idea in ('LOCAL_REUPLOAD','LOCAL_FFN'):return LocalFFN(head,layer,bridge,arm,seed,idea=='LOCAL_REUPLOAD')
    cls={'NATIVE_BORN':('qnative_born_v001',True),'FEATURE_GATE':('qfeature_gate_v001',True),'ORDINAL':('qordinal_v001',False)}[idea]
    import importlib
    constructor=importlib.import_module('applications.rag.'+cls[0]).Scorer
    model=constructor(arm,head.weight,head.bias,seed,*statistics) if cls[1] else constructor(arm,head.weight,head.bias,seed)
    class Wrapped(nn.Module):
        def __init__(self):super().__init__();self.inner=model;self.kind=arm;self.label_classes=label_classes
        def forward(self,row):return self.inner(row['readout'])
    return Wrapped().to(head.weight.device)


def objective(model,idea,row,tiers):
    score=model(row);mask=tiers[:,None]>tiers[None,:];count=mask.sum()
    loss=(F.softplus(-(score[:,None]-score[None,:]))*mask).sum()/count.clamp_min(1)
    if idea=='ORDINAL' and model.kind!='HEAD':
        _,p=model.inner.effect(row['readout'])
        # Binary labels supervise usefulness only. The two useful latent
        # outcomes retain the old sequential mechanism; neither means direct.
        if model.label_classes==2:p=torch.stack((p[:,0],p[:,1:].sum(-1)),-1)
        chosen=p.gather(-1,tiers[:,None]).squeeze(-1)
        loss=loss-chosen.clamp_min(1e-7).log().mean() if len(tiers) else loss
    return loss,count
