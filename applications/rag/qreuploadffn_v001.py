"""Local angle-reuploading quantum FFN nonlinearity; admitted science only."""
import copy,math
import torch
from applications.rag.qlocalffn_v001 import Scorer as LocalBase,compact_state,load_compact
from applications.rag.qffn_activation_v001 import gate,fixed_probe
from applications.rag.qfusion_v001 import need
ARMS=('HEAD','LORA','MLP','TRIG','PRODUCT','REAL','QUANTUM')

def step(s,a,w,axis,control=None,independent=False):
    if not independent:value=gate(s,a.unsqueeze(-1) if a.ndim else a,w,axis)
    else:
        stride=1<<w;shape=s.shape;x=s.reshape(*shape[:-1],4//(2*stride),2,stride)
        u,v=x[...,0,:],x[...,1,:];c,t=torch.cos(a/2),torch.sin(a/2)
        while c.ndim<u.ndim:c=c.unsqueeze(-1);t=t.unsqueeze(-1)
        if axis=='Y':r0,r1=c*u-t*v,t*u+c*v
        elif axis=='Z':r0,r1=(c-1j*t)*u,(c+1j*t)*v
        else:r0,r1=c*u-1j*t*v,c*v-1j*t*u
        value=torch.stack((r0,r1),-2).reshape(shape)
    if control is None:return value
    bits=torch.arange(4,device=s.device)
    return torch.where(((bits>>control)&1).bool(),value,s)

def circuit(angles,p,kind,independent=False):
    dtype=torch.complex128 if angles.dtype==torch.float64 else torch.complex64
    s=torch.zeros(*angles.shape[:-1],4,dtype=dtype,device=angles.device);s[...,0]=1
    for layer in range(2):
        for w in range(2):
            s=step(s,angles[...,w],w,'Y',independent=independent)
            s=step(s,angles[...,2+w],w,'Z',independent=independent)
        for w in range(2):s=step(s,p[6*layer+w],w,'Y',independent=independent)
        for w in range(2):s=step(s,p[6*layer+2+w],w,'Y' if kind=='REAL' else 'Z',independent=independent)
        for w in range(2):s=step(s,p[6*layer+4+w],1-w,'Y' if kind=='REAL' else 'X',None if kind=='PRODUCT' else w,independent)
    return s

class ReuploadActivation(torch.nn.Module):
    def __init__(self,kind,seed,device):
        super().__init__();self.kind=kind;g=torch.Generator(device='cpu').manual_seed(seed)
        if kind=='TRIG':self.matrix=torch.nn.Parameter((torch.randn(32,4,generator=g)/math.sqrt(32)).to(device))
        else:self.p=torch.nn.Parameter((.1*torch.randn(12,generator=g)).to(device))
    def forward(self,z):
        shape=z.shape;x=z.reshape(*shape[:-1],768,4);angles=torch.atan(x)
        if self.kind=='TRIG':
            fs=[angles.sin(),angles.cos()]
            for i in range(4):
                for j in range(i):
                    fs.extend(((angles[...,i:i+1]+angles[...,j:j+1]).sin(),(angles[...,i:i+1]+angles[...,j:j+1]).cos(),(angles[...,i:i+1]-angles[...,j:j+1]).sin(),(angles[...,i:i+1]-angles[...,j:j+1]).cos()))
            value=torch.cat(fs,-1)@self.matrix
        else:value=circuit(angles,self.p,self.kind).abs().square()
        return (4*x.norm(dim=-1,keepdim=True)*value).reshape(shape)
    def physical_probe(self,z):
        if self.kind=='TRIG':return dict(applicable=False),None
        with torch.no_grad():
            angles=torch.atan(z.reshape(-1,4));actual=circuit(angles,self.p,self.kind);oracle=circuit(angles,self.p,self.kind,True)
            err=float((actual-oracle).abs().max());norm=float((actual.abs().square().sum(-1)-1).abs().max())
            diff=float((circuit(angles,self.p,'QUANTUM')-circuit(angles,self.p,'PRODUCT')).abs().max())
            need(err<1e-5 and norm<2e-5 and diff>1e-8,'all local angle-state oracle/norm/controlled intervention')
            return dict(applicable=True,qubits_per_block=2,blocks=768,shared_angles=12,data_uploads=2,full_state_oracle_error=err,norm_error=norm,controlled_vs_product_difference=diff),dict(angles=angles.cpu(),state=actual.cpu(),oracle=oracle.cpu(),parameters=self.p.detach().cpu())

class Scorer(LocalBase):
    def __init__(self,kind,weight,bias,seed,scale,ffn):
        super().__init__('QUADRATIC' if kind=='TRIG' else kind,weight,bias,seed,scale,ffn);self.kind=kind
        if kind not in ('HEAD','LORA','MLP'):
            self.encoder=ReuploadActivation(kind,seed,weight.device);self.reference=copy.deepcopy(self.encoder)
            for p in self.reference.parameters():p.requires_grad_(False)
