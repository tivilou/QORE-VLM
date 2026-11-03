"""Full native Value geometry before frozen DPR residual/LN/FFN.

Exact classical simulation of a block-orthogonal quantum parameterization.
No hardware, computational acceleration, or entanglement attribution claim.
"""
from __future__ import annotations
import copy,math
import torch
from torch import nn
from applications.rag.qcapacity_depth_v001 import Scorer as CapacityScorer
from applications.rag.qweight_manifold_v001 import finite_difference

ARMS=('HEAD','CONCAT','MLP','DENSE','LORA','TANGENT','CAYLEY','LOCAL','QUANTUM')

def need(ok,message):
    if not ok:raise RuntimeError(message)

def pairs(wire,controlled):
    left=[i for i in range(64) if not i&(1<<wire) and (not controlled or i&(1<<((wire+1)%6)))]
    return left,[i+(1<<wire) for i in left]

def generators(device,dtype,local=False):
    rows=[]
    for layer in range(2):
        for controlled in (False,True):
            for wire in range(6):
                left,right=pairs(wire,controlled and not local)
                g=torch.zeros(64,64,device=device,dtype=dtype)
                g[left,right]=-.5;g[right,left]=.5;rows.append(g)
    return torch.stack(rows)

def circuit(angles,local=False,capture=None):
    need(angles.shape==(12,24),'all12heads and24logicalangles')
    value=torch.eye(64,device=angles.device,dtype=angles.dtype).expand(12,64,64).clone()
    states=[];j=0
    for layer in range(2):
        for controlled in (False,True):
            for wire in range(6):
                left,right=pairs(wire,controlled and not local)
                left=torch.tensor(left,device=angles.device);right=torch.tensor(right,device=angles.device)
                a=value[:,left];b=value[:,right];angle=angles[:,j,None,None];c=torch.cos(angle/2);s=torch.sin(angle/2)
                value=value.index_copy(1,left,c*a-s*b).index_copy(1,right,s*a+c*b)
                if capture is not None:states.append(value)
                j+=1
    if capture is not None:capture['ordered_operators']=states
    return value

class NativeTail(nn.Module):
    def __init__(self,layer):
        super().__init__()
        self.attention_output=layer.attention.output
        self.intermediate=layer.intermediate
        self.output=layer.output
        self.eval()
        need(all(not p.requires_grad for p in self.parameters()),'original frozen tail')
    def check(self):
        need(all(not m.training for m in self.modules()) and all(not p.requires_grad and p.grad is None for p in self.parameters()),'native tail recursively eval and frozen')
    def forward(self,context,residual,capture=None):
        self.check()
        attention=self.attention_output(context,residual)
        intermediate=self.intermediate(attention)
        result=self.output(intermediate,attention)
        if capture is not None:capture.update(attention_output=attention,intermediate=intermediate,tail_output=result)
        return result

class Scorer(nn.Module):
    def __init__(self,original_head,tail,kind,seed):
        super().__init__();need(kind in ARMS,'prospective native Value arm');self.kind=kind;self.tail=tail
        if kind in ('HEAD','CONCAT','MLP'):
            self.standard=CapacityScorer(original_head,kind,seed)
            return
        self.head=copy.deepcopy(original_head);self.head.weight.requires_grad_(True);self.head.bias.requires_grad_(False)
        device=original_head.weight.device;dtype=original_head.weight.dtype
        self.register_buffer('identity',torch.eye(64,device=device,dtype=dtype).expand(12,64,64).clone())
        if kind=='DENSE':self.delta=nn.Parameter(torch.zeros_like(self.identity))
        elif kind=='LORA':
            gen=torch.Generator(device='cpu').manual_seed(seed)
            self.low_a=nn.Parameter(((torch.rand(12,4,64,generator=gen)*2-1)/8).to(device=device,dtype=dtype))
            self.low_b=nn.Parameter(torch.zeros(12,64,4,device=device,dtype=dtype))
        else:
            self.angles=nn.Parameter(torch.zeros(12,24,device=device,dtype=dtype))
            if kind in ('TANGENT','CAYLEY'):self.register_buffer('generators',generators(device,dtype))

    def prepare(self,capture=None):
        if hasattr(self,'standard'):return None
        if self.kind=='DENSE':value=self.identity+self.delta
        elif self.kind=='LORA':value=self.identity+self.low_b@self.low_a
        elif self.kind in ('TANGENT','CAYLEY'):
            a=torch.einsum('hk,kij->hij',self.angles,self.generators)
            value=self.identity+a if self.kind=='TANGENT' else torch.linalg.solve(self.identity-a/2,self.identity+a/2)
        else:value=circuit(self.angles,self.kind=='LOCAL',capture)
        need(bool(torch.isfinite(value).all()),'finite full64 operators')
        if capture is not None:capture['operator']=value
        return value

    def forward(self,h,capture=None,prepared=None):
        need(h.shape[1:]==(4,768) and not h.requires_grad,'frozen native context,residual,block10,block12')
        if hasattr(self,'standard'):return self.standard(h[:,[2,1,3]],capture)
        operator=self.prepare(capture) if prepared is None else prepared
        context=h[:,0].reshape(-1,12,64)
        transformed=torch.einsum('hij,bhj->bhi',operator,context).reshape(-1,768)
        with torch.no_grad():reference=self.tail(h[:,0],h[:,1])
        parts={} if capture is not None else None
        modified=self.tail(transformed,h[:,1],parts)
        # FP32 anchor; in exact arithmetic native_cls==reference. All arms share it.
        hidden=h[:,3]+(modified-reference)
        score=self.head(hidden).squeeze(-1)
        if capture is not None:capture.update(context=context,transformed_context=transformed,reference_tail=reference,modified_tail=parts,anchored_hidden=hidden,scores=score)
        return score

    def owned_state_dict(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('tail.')}
    def load_owned_state(self,state):
        expected=self.owned_state_dict();need(set(state)==set(expected),'exact owned checkpoint keys')
        full=self.state_dict();full.update(state);self.load_state_dict(full,strict=True)
    def trainable(self):return [p for p in self.parameters() if p.requires_grad]
    def reference_digest_values(self):return {}
    def check_frozen(self,before):need(not before,'no trainable reference');self.tail.check()

def structural_probe(model):
    if model.kind in ('HEAD','CONCAT','MLP'):return {'scope':'standard native-depth readout'}
    with torch.no_grad():
        matrix=model.prepare();need(torch.equal(matrix,model.identity),'exact initial identity operator')
        if model.kind in ('DENSE','LORA'):return dict(initial_identity=True,nominal_parameters=sum(p.numel() for p in model.trainable()))
        g=generators(matrix.device,matrix.dtype,local=model.kind=='LOCAL')
        singular=torch.linalg.svdvals(g.reshape(24,-1).double());rank=int((singular>singular.max()*1e-8).sum())
        expected=6 if model.kind=='LOCAL' else 12;need(rank==expected,'declared independent generator rank')
        test=torch.sin(torch.arange(288,device=matrix.device,dtype=torch.float64)+1).reshape(12,24)*.17
        actual=circuit(test,model.kind=='LOCAL');independent=torch.eye(64,dtype=torch.float64,device=matrix.device).expand(12,64,64).clone()
        basis=torch.arange(64,device=matrix.device);j=0
        for layer in range(2):
            for controlled in (False,True):
                for wire in range(6):
                    select=((basis>>wire)&1)==0
                    if controlled and model.kind!='LOCAL':select=select&(((basis>>((wire+1)%6))&1)==1)
                    left=basis[select];right=left^(1<<wire)
                    dense=torch.eye(64,dtype=torch.float64,device=matrix.device).expand(12,64,64).clone()
                    co=torch.cos(test[:,j]/2)[:,None].expand(12,len(left)).contiguous();si=torch.sin(test[:,j]/2)[:,None].expand(12,len(left)).contiguous()
                    dense[:,left,left]=co;dense[:,right,right]=co;dense[:,left,right]=-si;dense[:,right,left]=si
                    independent=dense@independent;j+=1
        error=float((actual-independent).abs().max());orthogonal=float((actual@actual.transpose(-1,-2)-model.identity.double()).abs().max())
        need(error<1e-12 and orthogonal<1e-12,'full64 ordered dense oracle and unitarity')
        # Circuit and Cayley derivatives at zero share all24 generator columns.
        for index in range(24):
            t=torch.zeros_like(test);t[:,index]=1e-5
            derivative=(circuit(t,model.kind=='LOCAL')-circuit(-t,model.kind=='LOCAL'))/2e-5
            need(float((derivative-g[index].double()).abs().max())<1e-8,'exact quantum/local tangent generator')
        if model.kind=='CAYLEY':
            a=torch.einsum('hk,kij->hij',test,g.double())
            identity=model.identity.double()
            cayley=torch.linalg.solve(identity-a/2,identity+a/2)
            need(float((cayley@cayley.transpose(-1,-2)-identity).abs().max())<1e-12,'actual nonzero Cayley orthogonality')
    return dict(nominal_angles=288,initial_operator_tangent_rank_per_head=rank,initial_rank_upperbound_all_heads=rank*12,task_jacobian_rank_not_certified=True,dense_oracle_error=error,orthogonality_error=orthogonal,initial_identity=True,local_not_unentangled_control=True)
