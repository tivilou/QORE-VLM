"""116: whole native depth factors and a three-party Born measurement.

Accepted exact MPS contractions; new semantic input-factorization hypothesis.
Not a new MPS family or a quantum hardware/computational advantage claim.
"""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need,digest
from applications.rag.toy100_native_feature_readout_v001 import NativeFeatureReadout
from applications.rag.toy100_reference_depth_v001 import BANKS,KINDS
from applications.rag.toy100_product_input_mps_v001 import unpack_mps,cp_cores,born_log_evidence,signed_amplitude

ARMS=('HEAD','CONCAT','MLP8','QUADRATIC8','CP4_BORN','PRODUCT','DEPHASED','TT_AMPLITUDE','ADDITIVE_BORN','QUANTUM')
DEPTHS=3


def encode(views,mean,scale):
    z=(views.double()-mean)/scale
    v=torch.cat((torch.ones_like(z[...,:1]),z),-1)
    need(bool(torch.isfinite(v).all()),'finite complete3depth anchored inputs')
    return v/v.norm(dim=-1,keepdim=True)


def coordinates(psi):
    need(bool((psi[...,0]>0).all()),'positive anchor retains view radius')
    return (psi[...,1:]/psi[...,:1]).flatten(-2)


def additive_evidence(psi,raw):
    local=raw.reshape(DEPTHS,psi.shape[-1]);norm=local.square().sum(-1)
    need(bool(torch.isfinite(norm).all() and (norm>0).all()),'nonzero complete separate depth states')
    probability=(psi*local).sum(-1).square()/norm
    need(bool((probability<=1+1e-10).all()),'normalized separate depth Born probabilities')
    return (.95*psi.shape[-1]*probability+.05).log().mean(-1)


class DepthProductMeasurement(nn.Module):
    def __init__(self,widths,arm,seed):
        super().__init__();self.arm=arm;self.widths=widths;self.layout={};values=[];offset=0
        g=torch.Generator(device='cpu').manual_seed(seed+11600)
        for k in KINDS:
            d=widths[k];physical=d+1
            if arm=='CONCAT':value=torch.zeros(DEPTHS*d,dtype=torch.float64)
            elif arm in ('MLP8','QUADRATIC8'):
                value=torch.randn(8*DEPTHS*d+16,generator=g,dtype=torch.float64)/math.sqrt(DEPTHS*d)
                value[8*DEPTHS*d:8*DEPTHS*d+8]=0
            elif arm in ('PRODUCT','CP4_BORN','ADDITIVE_BORN'):
                rank=4 if arm=='CP4_BORN' else 1
                value=.02/math.sqrt(physical)*torch.randn(2,DEPTHS,rank,physical,generator=g,dtype=torch.float64)
                value[:,:,:,0]+=1;value=value.flatten()
            else:
                pieces=[]
                for label in range(2):
                    for site in range(DEPTHS):
                        left=1 if site==0 else 2;right=1 if site==DEPTHS-1 else 2
                        core=.02/math.sqrt(physical)*torch.randn(left,physical,right,generator=g,dtype=torch.float64)
                        for i in range(min(left,right)):core[i,0,i]+=1
                        pieces.append(core.flatten())
                value=torch.cat(pieces)
            self.layout[k]=(offset,len(value));offset+=len(value);values.append(value)
        self.weights=nn.Parameter(torch.cat(values));self.register_buffer('initial',self.weights.detach().clone())

    def raw(self,k,reference=False):
        start,size=self.layout[k]
        return (self.initial if reference else self.weights)[start:start+size]

    def states(self,k,reference=False):
        raw=self.raw(k,reference).reshape(2,-1);physical=self.widths[k]+1
        if self.arm=='CP4_BORN':return [cp_cores(v,DEPTHS,physical) for v in raw]
        rank=1 if self.arm=='PRODUCT' else 2
        return [unpack_mps(v,DEPTHS,physical,rank) for v in raw]

    def scalar(self,psi,k,reference=False):
        raw=self.raw(k,reference);d=self.widths[k]
        if self.arm=='CONCAT':return (coordinates(psi)*raw).sum(-1).float()
        if self.arm in ('MLP8','QUADRATIC8'):
            z=coordinates(psi);hidden=F.linear(z,raw[:8*DEPTHS*d].reshape(8,DEPTHS*d),raw[8*DEPTHS*d:8*DEPTHS*d+8])
            return ((hidden.tanh() if self.arm=='MLP8' else hidden.square())@raw[-8:]).float()
        if self.arm=='ADDITIVE_BORN':v=[additive_evidence(psi,p) for p in raw.reshape(2,-1)]
        else:
            cores=self.states(k,reference)
            v=[signed_amplitude(psi,c) if self.arm=='TT_AMPLITUDE' else born_log_evidence(psi,c,self.arm=='DEPHASED') for c in cores]
        return (v[1]-v[0]).float()


class DepthProductReadout(NativeFeatureReadout):
    explicit_context=True
    def __init__(self,packed,arm,seed):
        super().__init__(packed,'HEAD',seed);need(arm in ARMS,'declared three-depth product arm');self.arm=arm
        self.shared=None if arm=='HEAD' else DepthProductMeasurement({k:p['original_head']['weight'].shape[-1] for k,p in packed.items()},arm,seed)
        for k,p in packed.items():
            self.register_buffer('initial_head_'+k,p['original_head']['weight'].clone())
            scale=p['native']['train'].std(correction=0)
            need(bool(torch.isfinite(scale)) and float(scale)>0,'positive fixedTRAIN native score scale')
            self.register_buffer('scale_'+k,scale)
    def references(self,coords,native,features=None):
        with torch.no_grad():return {k:torch.zeros_like(native[k]) if self.shared is None else self.shared.scalar(coords[k],k,True) for k in KINDS}
    def gradient_probe_parameter(self):return self.shared.weights
    def activity_gradient_names(self):return () if self.shared is None else ('shared.weights',)
    def one(self,k,x,native,psi,reference):
        value=native+F.linear(x,self.heads[k].weight-getattr(self,'initial_head_'+k)).squeeze(-1)
        return value if self.shared is None else value+getattr(self,'scale_'+k)*(self.shared.scalar(psi,k)-reference)
    def forward(self,features,native,originals,coords,references):return {k:self.one(k,features[k],native[k],coords[k],references[k]) for k in KINDS}
    def forward_one(self,k,x,native,original,coords):
        with torch.no_grad():reference=torch.zeros_like(native) if self.shared is None else self.shared.scalar(coords,k,True)
        return self.one(k,x,native,coords,reference)
    def mechanism_diagnostics(self):
        return dict(information='Complete three native semantic depth factors; no arbitrary coordinate blocking/PCA or collapsed three-vector span.',parameters=0 if self.shared is None else self.shared.weights.numel(),mechanism='Normalized bond2 three-party class measurement, Born log evidence with fixed isotropic floor and complete initial subtraction.',limits='Exact classical tensor contraction; CP4 same degree and more nominal parameters, not strict capacitymatch. No quantumexclusive function/efficiency/novelty/entanglement-necessity or independentconfirmation claim.')


def prepare(packed,targets):
    root=Path(__file__).resolve().parents[3];bank=root/'depth-order-run-v002/depth-bank';coords={};mapping={}
    for k in KINDS:
        path=bank/(k+'-depths.bin');need(digest(path)==BANKS[k],'same complete pinned native depth bank')
        value=torch.load(path,weights_only=True);d=packed[k]['original_head']['weight'].shape[-1]
        need(value.shape==(100,50,DEPTHS,d) and bool(torch.isfinite(value).all()),'complete finite ordered three semantic depths')
        need(torch.equal(value[:,:,-1],torch.cat((packed[k]['features']['train'],packed[k]['features']['holdout']))),'full100 native finalfeature identities')
        train=value[:80].double();mean=train.mean((0,1));scale=(train-mean).square().sum(-1).mean((0,1)).sqrt()[:,None]
        need(bool(torch.isfinite(scale).all() and (scale>0).all()),'positive completeTRAIN view contrast scale')
        encoded=encode(value.cuda(),mean.cuda(),scale.cuda());coords[k]=encoded
        rebuilt=encoded[...,-1,1:]/encoded[...,-1,:1]*scale[-1].cuda()+mean[-1].cuda()
        need(torch.allclose(rebuilt,value[:,:,-1].cuda().double(),atol=1e-11,rtol=1e-11),'anchoredencoding retains complete finalview direction and radius')
        mapping[k]=dict(mean=mean,scale=scale,bank_sha256=BANKS[k],source_cache_sha256=packed[k]['cache_sha256'],depths=DEPTHS)
    def restore_mapping(path):
        saved=torch.load(path,weights_only=True)
        for k in KINDS:
            need(saved[k]['bank_sha256']==BANKS[k] and saved[k]['source_cache_sha256']==packed[k]['cache_sha256'] and saved[k]['depths']==DEPTHS,'saved semanticdepth source binding')
            need(torch.equal(saved[k]['mean'],mapping[k]['mean']) and torch.equal(saved[k]['scale'],mapping[k]['scale']),'complete TRAIN-only contrast statistics replay')
    return dict(coords=coords,mapping=mapping,restore_mapping=restore_mapping,arms=ARMS,factory=lambda arm,seed:DepthProductReadout(packed,arm,seed),protocol=dict(candidate='116_SEMANTIC_DEPTH_PRODUCT_BORN_MEASUREMENT',mechanism='Three complete centered anchored native depth factors; tensorproduct input; normalized bond2 classmeasurement and Born logratio. TRAINonly inputcontrast statistics; allcontrols have same threefactors.',controls='All+1792nativeheadweights. Perbackbone widthD: CONCAT3D; MLP8/QUADRATIC8 24D+16; CP4 24(D+1); PRODUCT/ADDITIVE6(D+1); Q/DEPHASED/TT_AMPLITUDE16(D+1). Modelprivateparameters, sameprocedure. Virtualgauge and CPcapacity differences disclosed.',claim='New configuration/input-factorization hypothesis, not a new MPS family. 83single-amplitude squared-affine;105arbitrary coordinateblocks;113/115single-view-span density. Here complete semanticdepth tensorproduct with crossdepth input interactions. Exact efficient classical tensor contractions; no quantumhardware/novelty/complexityclaim.'))
