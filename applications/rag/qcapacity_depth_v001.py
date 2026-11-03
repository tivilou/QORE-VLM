"""Classical capacity diagnostic on frozen full native depth features.

No quantum implementation or advantage claim. Admitted queue only.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qdepth_readout_v001 import DepthReadout

ARMS=('HEAD','CONCAT','MLP','QUADRATIC','CROSSDEPTH')


def need(ok,text):
    if not ok: raise RuntimeError(text)


class Scorer(nn.Module):
    def __init__(self,original_head,kind,seed):
        super().__init__();need(kind in ARMS,'capacity diagnostic arm');self.kind=kind
        self.base=DepthReadout(original_head,'HEAD' if kind=='HEAD' else 'CONCAT')
        if kind not in ('HEAD','CONCAT'):
            # Same initialized projection and parameter count in nonlinear arms.
            generator=torch.Generator(device='cpu').manual_seed(seed)
            w=(torch.rand((3,8,768),generator=generator)*2-1)/math.sqrt(2304)
            self.projection=nn.Parameter(w.to(original_head.weight.device))
            self.offset=nn.Parameter(torch.zeros(8,device=original_head.weight.device))
            self.coefficients=nn.Parameter(torch.zeros(8,device=original_head.weight.device))

    def forward(self,h,capture=None):
        need(h.shape[1:]==(3,768) and not h.requires_grad,'three complete frozen native depths')
        score=self.base(h)
        if self.kind not in ('HEAD','CONCAT'):
            z=F.layer_norm(h,(768,))
            per_depth=torch.einsum('bdh,drh->bdr',z,self.projection)
            if self.kind=='MLP': f=torch.tanh(per_depth.sum(1)+self.offset)
            elif self.kind=='QUADRATIC': f=(per_depth.sum(1)+self.offset).square()
            else:
                # Three depth-specific affine factors; offsets shared and identifiable.
                p=per_depth+self.offset/math.sqrt(3)
                f=(p[:,0]*p[:,1]+p[:,0]*p[:,2]+p[:,1]*p[:,2])/math.sqrt(3)
            delta=f@self.coefficients/math.sqrt(8);score=score+delta
            if capture is not None:capture.update(normalized_full_depths=z,projected_depths=per_depth,nonlinear_features=f,coefficients=self.coefficients,delta=delta)
        return score

    def trainable(self):return [p for p in self.parameters() if p.requires_grad]
    def reference_digest_values(self):return {}
    def check_frozen(self,before):need(not before,'no trainable backbone or bank')
