"""Mixed native-depth Uhlmann fidelity. Exact classical matrix computation.

Admitted worker only. No quantum hardware, speedup or originality claim.
All supervised banks use training labels, with component-disjoint cross-fitting.
"""
from __future__ import annotations
import copy, math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qdepth_readout_v001 import DepthReadout

ARMS = ('HEAD','CONCAT','COSINE','HS','DEPHASED','QDA','QUANTUM')
EXTRAS = ARMS[2:]


def need(ok, text):
    if not ok: raise RuntimeError(text)


def states(h):
    need(h.ndim == 3 and h.shape[1:] == (3,768), 'three full native CLS depths')
    x = h.double(); n = x.norm(dim=-1,keepdim=True)
    need(bool(torch.isfinite(x).all()) and bool((n>1e-8).all()), 'nonzero amplitude inputs')
    return x/n


def sym(x): return (x+x.transpose(-1,-2))/2


def psd_eigen(x):
    ev = torch.linalg.eigvalsh(sym(x))
    need(bool(torch.isfinite(ev).all()) and float(ev.min())>=-1e-12, 'PSD roundoff tolerance')
    return ev.clamp_min(0)  # Only negative float64 roundoff, never a fitted floor.


def raw_features(h, bank, capture=None):
    u = states(h); a = u/math.sqrt(3); d = u.square().mean(1)
    values = {k:[] for k in EXTRAS}; spectral=[]
    for sigma,mu,inv,logdet in zip(bank['density'],bank['mean'],bank['inverse'],bank['logdet']):
        g = sym((a@sigma)@a.transpose(-1,-2))
        ev = psd_eigen(g); fidelity = ev.sqrt().sum(-1).square()
        hs = g.diagonal(dim1=-2,dim2=-1).sum(-1)
        diagonal = (d*sigma.diag()).sqrt().sum(-1).square()
        need(bool((fidelity>0).all()) and bool((fidelity<=1+1e-8).all()), 'squared Uhlmann fidelity range')
        need(bool((hs>0).all()) and bool((diagonal>0).all()), 'strict pseudo-observation positivity')
        residual=u-mu
        values['QDA'].append(-.5*((residual@inv*residual).sum(-1).mean(-1)+logdet))
        values['COSINE'].append((u@F.normalize(mu,dim=0)).mean(-1))
        values['HS'].append(hs.log()); values['DEPHASED'].append(diagonal.log()); values['QUANTUM'].append(fidelity.log())
        spectral.append(dict(gram=g,eigenvalues=ev,fidelity=fidelity,hilbert_schmidt=hs,mixed_spectral_increment=fidelity-hs))
    out={k:torch.stack(v,-1) for k,v in values.items()}
    need(all(bool(torch.isfinite(v).all()) for v in out.values()), 'finite full-bank features')
    if capture is not None: capture.update(normalized_depth_states=u,spectral=spectral,raw_features=out)
    return out


def fit_one(features, train, targets, folds, excluded, check):
    device=next(iter(features.values())).device; eye=torch.eye(768,device=device,dtype=torch.float64)
    totals=[torch.zeros_like(eye) for _ in range(3)]
    means=[torch.zeros(768,device=device,dtype=torch.float64) for _ in range(3)]; counts=[0]*3; members=[]
    for case in train:
        if folds[case.question_sha256]==excluded: continue
        check(); q=case.question_sha256; u=states(features[q]); ys=dict(targets.rows_by_question[q]); y=torch.tensor([ys[i] for i in sorted(ys)],device=device)
        need(len(y)==len(u), 'training reviewed membership')
        members.append(q)
        for c in range(3):
            x=u[y==c].reshape(-1,768)
            if not len(x): continue
            totals[c]+=x.T@x/len(x); means[c]+=x.mean(0); counts[c]+=1
    density=[]; mu=[]; inverse=[]; logdet=[]; diagnostics=[]
    for c,n in enumerate(counts):
        need(n>1, 'at least two training questions per class in every fold bank')
        sigma=sym((totals[c]+eye/768)/(n+1)); m=means[c]/(n+1)
        cov=sym(sigma-m[:,None]*m[None,:]); cov=sym((cov+eye*cov.trace()/768)/2)
        ev,basis=torch.linalg.eigh(cov)
        need(bool(torch.isfinite(ev).all()) and float(ev.min())>1e-12, 'positive centered half-spherical QDA covariance')
        need(abs(float(sigma.trace())-1)<1e-9, 'class density trace one')
        density.append(sigma); mu.append(m); inverse.append(sym((basis/ev)@basis.T)); logdet.append(ev.log().sum())
        diagnostics.append(dict(question_count=n,density_trace=float(sigma.trace()),covariance_minimum_eigenvalue=float(ev.min())))
    return dict(density=torch.stack(density),mean=torch.stack(mu),inverse=torch.stack(inverse),logdet=torch.stack(logdet),members=members,excluded_fold=excluded,class_statistics=diagnostics)


def fit_cross_fitted(features, train, dev, targets, identities, check):
    identity={v['question_sha256']:v for v in identities}
    # Component hash modulo5 is frozen before labels/outcomes, no balancing after inspection.
    folds={q:int(identity[q]['component_id'],16)%5 for q in (v.question_sha256 for v in train)}
    need(set(folds.values())==set(range(5)), 'all five component folds present')
    banks={str(f):fit_one(features,train,targets,folds,f,check) for f in range(5)}
    banks['all']=fit_one(features,train,targets,folds,None,check)
    raw={}; all_values={k:[] for k in EXTRAS}
    for case in train+dev:
        check(); q=case.question_sha256; bank=banks[str(folds[q])] if case.role=='training' else banks['all']
        need(case.role!='training' or q not in bank['members'], 'out-of-fold question excluded')
        raw[q]=raw_features(features[q],bank)
        if case.role=='training':
            for k in EXTRAS: all_values[k].append(raw[q][k])
    stats={}
    for k in EXTRAS:
        v=torch.cat(all_values[k]); mean=v.mean(0); std=v.std(0,unbiased=False)
        # Constant dimensions stay zero after centering, with fixed divisor1.
        stats[k]=dict(mean=mean,std=torch.where(std>1e-10,std,torch.ones_like(std)),constant=std<=1e-10)
    def pack(h,bank):
        r=raw_features(h,bank); extra=torch.cat([(r[k]-stats[k]['mean'])/stats[k]['std'] for k in EXTRAS],-1).float()
        return torch.cat((h,extra.reshape(len(h),3,5)),-1)
    packed={q:torch.cat((features[q],torch.cat([(raw[q][k]-stats[k]['mean'])/stats[k]['std'] for k in EXTRAS],-1).float().reshape(len(features[q]),3,5)),-1) for q in features}
    return packed,dict(banks=banks,statistics=stats,folds=folds,normalization='global training OOF item moments; not fully independent of own labels via other OOF banks',development_bank='all training members; distribution shift disclosed'),pack


def oracle_probe(h,bank):
    """Dense768 rank-three oracle and independent SVD, on fixed two paragraphs."""
    h=h[:2]; cap={}; actual=raw_features(h,bank,cap)['QUANTUM'].exp(); u=states(h)
    dense=[]; svd=[]; minimum=[]
    for sigma in bank['density']:
        ev,basis=torch.linalg.eigh(sigma); need(float(ev.min())>0,'positive class density oracle')
        root=(basis*ev.sqrt())@basis.T; vals=[]; svals=[]
        for row in u:
            a=row.T/math.sqrt(3); rho=a@a.T
            matrix=sym(root@rho@root); eigen=torch.linalg.eigvalsh(matrix)
            need(float(eigen.min())>=-1e-12,'dense oracle PSD')
            # Known rank<=3; do not sum square roots of765 numerical null-space modes.
            vals.append(eigen[-3:].clamp_min(0).sqrt().sum().square())
            svals.append(torch.linalg.svdvals(root@a).sum().square()); minimum.append(float(eigen.min()))
        dense.append(torch.stack(vals)); svd.append(torch.stack(svals))
    dense=torch.stack(dense,-1); svd=torch.stack(svd,-1)
    error=float((actual-dense).abs().max()); serr=float((actual-svd).abs().max())
    permutation=raw_features(h[:,[2,0,1]],bank)['QUANTUM'].exp()
    perr=float((actual-permutation).abs().max())
    need(error<5e-7 and serr<5e-8 and perr<5e-8,'mixed fidelity independent oracles/permutation')
    repeated=h[:,0:1].expand(-1,3,-1); rank1={}; raw_features(repeated,bank,rank1)
    rank1_error=max(float((r['fidelity']-r['hilbert_schmidt']).abs().max()) for r in rank1['spectral'])
    need(rank1_error<5e-8,'rank-one fidelity equals overlap')
    increment=max(float(r['mixed_spectral_increment'].abs().max()) for r in cap['spectral'])
    need(increment>1e-6,'actual mixed-state nonlinear spectral term distinct from HS')
    return dict(dense768_error=error,independent_svd_error=serr,depth_permutation_error=perr,rank1_limit_error=rank1_error,max_mixed_increment=increment,dense_minimum_eigenvalue=min(minimum),rank3_nullspace_excluded_by_identity=True),cap


class Scorer(nn.Module):
    def __init__(self, original_head, kind, seed):
        super().__init__(); need(kind in ARMS,'mixed state arm'); self.kind=kind
        self.base=DepthReadout(original_head,'HEAD' if kind=='HEAD' else 'CONCAT')
        if kind in EXTRAS: self.coefficients=nn.Parameter(torch.zeros(3,device=original_head.weight.device))

    def forward(self,h,capture=None):
        need(h.shape[1:]==(3,773),'native states plus five bank-feature triples')
        score=self.base(h[...,:768])
        if self.kind in EXTRAS:
            extra=h[...,768:].reshape(len(h),15)[:,EXTRAS.index(self.kind)*3:EXTRAS.index(self.kind)*3+3]
            delta=extra@self.coefficients/math.sqrt(3); score=score+delta
            if capture is not None: capture.update(standardized_bank_features=extra,coefficients=self.coefficients,delta=delta)
        return score

    def trainable(self): return [p for p in self.parameters() if p.requires_grad]
    def reference_digest_values(self): return {}
    def check_frozen(self,before): need(not before,'no variational reference or trainable bank')
