"""Occupied-space quantum weight preparation; admitted queue execution only.

This is an orthogonal-parameterization hypothesis, not quantum acceleration.
The native scoring function remains linear in all three frozen CLS depths.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qcapacity_depth_v001 import Scorer as CapacityScorer

ARMS=('HEAD','CONCAT','MLP','NORM','TANGENT','LOCAL','QUANTUM')


def need(value,message):
    if not value: raise RuntimeError(message)


def gate_specs(local_only=False):
    """Pairs of occupied indices; conceptual 12-qubit unused support is identity.

    Index = depth*1024 + sector*256 + eight-bit word in the physical embedding.
    Compact storage removes the unused fourth sector at both ternary axes.
    No standard high-bit rotation is used across a ternary axis.
    """
    out=[]
    for layer in range(2):
        for depth in range(3):
            for controlled in (False,True):
                for wire in range(8):
                    left=[];right=[]
                    for sector in range(3):
                        for word in range(256):
                            if word&(1<<wire):continue
                            control=(wire+1)%8
                            if controlled and not local_only and not word&(1<<control):continue
                            left.append(depth*768+sector*256+word)
                            right.append(depth*768+sector*256+word+(1<<wire))
                    out.append(dict(layer=layer,depth=depth,axis='low',wire=wire,controlled=controlled,left=left,right=right))
            for first,second in ((0,1),(1,2)):
                out.append(dict(layer=layer,depth=depth,axis='sector',first=first,second=second,left=[depth*768+first*256+i for i in range(256)],right=[depth*768+second*256+i for i in range(256)]))
        for first,second in ((0,2),(1,2)):
            out.append(dict(layer=layer,depth=None,axis='depth',first=first,second=second,left=[first*768+i for i in range(768)],right=[second*768+i for i in range(768)]))
    need(len(out)==112,'two fixed112-rotation layers')
    return out


def tangent_columns(w0,specs):
    columns=[]
    for spec in specs:
        left=torch.tensor(spec['left'],device=w0.device)
        right=torch.tensor(spec['right'],device=w0.device)
        # Independent exact derivative of a half-angle plane rotation at zero.
        col=torch.zeros_like(w0)
        col[left]=-w0[right]/2;col[right]=w0[left]/2
        columns.append(col)
    return torch.stack(columns,1)


class Scorer(nn.Module):
    def __init__(self,original_head,kind,seed):
        super().__init__();need(kind in ARMS,'declared weight-manifold arm');self.kind=kind
        if kind in ('HEAD','CONCAT','MLP'):
            self.standard=CapacityScorer(original_head,kind,seed)
            return
        w=original_head.weight.detach().reshape(-1)
        need(w.numel()==768,'pinned DPR head dimension')
        self.register_buffer('w0',torch.cat((torch.zeros_like(w),torch.zeros_like(w),w)).clone())
        self.register_buffer('bias',original_head.bias.detach().reshape(()).clone())
        self.register_buffer('norm0',torch.linalg.vector_norm(self.w0))
        need(bool(self.norm0>0),'nonzero native weight state')
        if kind=='NORM':self.weight=nn.Parameter(self.w0.clone())
        else:
            self.angles=nn.Parameter(torch.zeros(112,device=w.device,dtype=w.dtype))
            self.specs=gate_specs(local_only=kind=='LOCAL')
            if kind=='TANGENT':
                self.register_buffer('j0',tangent_columns(self.w0,self.specs))
            else:
                for j,spec in enumerate(self.specs):
                    self.register_buffer(f'left_{j}',torch.tensor(spec['left'],device=w.device))
                    self.register_buffer(f'right_{j}',torch.tensor(spec['right'],device=w.device))

    def prepare(self,capture=None):
        if hasattr(self,'standard'):return None
        if self.kind=='NORM':
            raw=self.weight;norm=torch.linalg.vector_norm(raw)
            need(bool(torch.isfinite(norm)) and bool(norm>0),'finite nonzero norm parameter')
            weight=raw*(self.norm0/norm)
        elif self.kind=='TANGENT':
            raw=self.w0+self.j0@self.angles;norm=torch.linalg.vector_norm(raw)
            need(bool(torch.isfinite(norm)) and bool(norm>0),'finite tangent retraction')
            weight=raw*(self.norm0/norm)
        else:
            weight=self.w0;states=[]
            for j,angle in enumerate(self.angles):
                left=getattr(self,f'left_{j}');right=getattr(self,f'right_{j}')
                a=weight[left];b=weight[right];c=torch.cos(angle/2);s=torch.sin(angle/2)
                weight=weight.index_copy(0,left,c*a-s*b).index_copy(0,right,s*a+c*b)
                if capture is not None:states.append(weight)
            if capture is not None:capture['ordered_weight_states']=states
        need(bool(torch.isfinite(weight).all()),'finite scoring weights')
        if capture is not None:
            capture.update(prepared_weight=weight,native_weight=self.w0,norm=torch.linalg.vector_norm(weight),norm0=self.norm0,conceptual_unused_mass=0,unused_action='identity')
        return weight

    def forward(self,h,capture=None,prepared=None):
        need(h.shape[1:]==(3,768) and not h.requires_grad,'frozen complete native three-depth feature')
        if hasattr(self,'standard'):return self.standard(h,capture)
        weight=self.prepare(capture) if prepared is None else prepared
        # Native-anchor evaluation makes the zero-angle path numerically exact.
        # Algebraically this is the same linear readout w.h+b, not a residual MLP.
        base=F.linear(h[:,2],self.w0[1536:].reshape(1,768),self.bias.reshape(1)).squeeze(-1)
        return base+h.reshape(len(h),2304)@(weight-self.w0)

    def trainable(self):return [p for p in self.parameters() if p.requires_grad]
    def reference_digest_values(self):return {}
    def check_frozen(self,before):need(not before,'no trainable backbone or reference bank')


def structural_probe(model):
    """Actual occupied norm, zero tangent, independent gate and embedding oracles."""
    if model.kind not in ('QUANTUM','LOCAL','TANGENT','NORM'):return {'status':'N/A standard readout'}
    with torch.no_grad():
        weight=model.prepare();need(torch.equal(weight,model.w0),'exact initial weight identity')
        if model.kind=='NORM':return dict(initial_null=True,occupied_norm=float(model.norm0),unused_mass=0)
        specs=gate_specs(local_only=model.kind=='LOCAL')
        j0=tangent_columns(model.w0,specs)
        sv=torch.linalg.svdvals(j0.double());rank=int((sv>sv.max()*1e-8).sum())
        need(rank>0 and float((model.w0@j0).abs().max())<1e-6,'nontrivial tangent orthogonal to native state')
        if model.kind=='TANGENT':need(torch.equal(j0,model.j0),'exact shared quantum tangent')
        # Full physical4096 embedding, not just compact norm assertion.
        physical=torch.zeros((4,4,256),device=weight.device,dtype=weight.dtype)
        physical[:3,:3]=weight.reshape(3,3,256)
        need(bool((physical[3]==0).all()) and bool((physical[:,3]==0).all()),'occupied support embedding')
        if model.kind in ('QUANTUM','LOCAL'):
            # Independent 12-bit physical-index oracle, including nonzero unused input.
            basis=torch.arange(4096,device=weight.device)
            depth=basis>>10;sector=(basis>>8)&3;word=basis&255
            occupied=(depth<3)&(sector<3)
            physical_input=torch.cos(basis.double()*.173)/64
            x=physical_input[occupied];angle=.173;c=math.cos(angle/2);s=math.sin(angle/2);errors=[]
            for j,spec in enumerate(specs):
                left=getattr(model,f'left_{j}');right=getattr(model,f'right_{j}')
                actual=x.index_copy(0,left,c*x[left]-s*x[right]).index_copy(0,right,s*x[left]+c*x[right])
                if spec['axis']=='low':
                    select=occupied&(depth==spec['depth'])&((word&(1<<spec['wire']))==0)
                    if spec['controlled'] and model.kind!='LOCAL':select=select&((word&(1<<((spec['wire']+1)%8)))!=0)
                    pl=basis[select];pr=pl+(1<<spec['wire'])
                elif spec['axis']=='sector':
                    pl=basis[occupied&(depth==spec['depth'])&(sector==spec['first'])]
                    pr=pl+(spec['second']-spec['first'])*256
                else:
                    pl=basis[occupied&(depth==spec['first'])]
                    pr=pl+(spec['second']-spec['first'])*1024
                expected=physical_input.clone()
                expected[pl]=c*physical_input[pl]-s*physical_input[pr]
                expected[pr]=s*physical_input[pl]+c*physical_input[pr]
                error=float((actual-expected[occupied]).abs().max());errors.append(error)
                need(error<1e-10 and torch.equal(expected[~occupied],physical_input[~occupied]),'independent4096 generator and unused identity')
                need(abs(float(torch.linalg.vector_norm(actual))-float(torch.linalg.vector_norm(x)))<1e-10,'generator norm')
            oracle=max(errors)
        else:oracle=None
    return dict(nominal_angles=112,initial_tangent_rank=rank,tangent_column_norm_min=float(torch.linalg.vector_norm(j0,dim=0).min()),tangent_column_norm_max=float(torch.linalg.vector_norm(j0,dim=0).max()),zero_tangent_columns=int((torch.linalg.vector_norm(j0,dim=0)==0).sum()),full_occupied_oracle_max_error=oracle,unused_mass=0,unused_action='identity',fixed_norm_does_not_reduce_all_linear_rankings=True,initial_null=True)


def finite_difference(model,h,y,loss_for):
    """Fixed cases, largest actual gradient coordinate, independent FP64 loss FD."""
    duplicate=copy.deepcopy(model).double();hd=h.double()
    score=duplicate(hd);loss,_=loss_for(score,y);loss.backward()
    rows=[]
    for name,param in duplicate.named_parameters():
        if not param.requires_grad:continue
        need(param.grad is not None and bool(torch.isfinite(param.grad).all()),'finite double parameter gradient')
        flat=param.reshape(-1);g=param.grad.reshape(-1);index=int(g.abs().argmax());analytic=float(g[index])
        old=float(flat[index]);values=[]
        try:
            with torch.no_grad():
                for sign in (1,-1):
                    flat[index]=old+sign*1e-5
                    values.append(float(loss_for(duplicate(hd),y)[0]))
        finally:
            with torch.no_grad():flat[index]=old
        numeric=(values[0]-values[1])/2e-5;error=abs(numeric-analytic)
        need(error<=1e-7+1e-4*abs(analytic),'independent FP64 loss directional coordinate')
        rows.append(dict(parameter=name,coordinate=index,analytic=analytic,central=numeric,absolute_error=error))
    return rows
