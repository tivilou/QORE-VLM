"""147: prospective attribution of146's local Born positive branch.

No new entangling circuit or retrospective L3. Input dephasing removes the
fixed-frame off-diagonal terms of exactly the same net-angle local unitary.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need,digest
from applications.rag.toy100_global_ffn_v001 import GlobalActivation,circuit,born_activation
from applications.rag.toy100_output_frame_v001 import OutputFrameReadout
from applications.rag.toy100_depth_product_born_v001 import DepthProductReadout

ARMS=('HEAD','SIGNED_LOCAL','GELU_LOCAL','ABS_LOCAL','DEPHASED','RANDOM_FRAME','CLASSICAL_MATCHING','MLP8','QUANTUM')


def intensity_transfer(intensity,angle,wire):
    n=intensity.shape[-1];bit=1<<wire
    x=intensity.reshape(*intensity.shape[:-1],n//(2*bit),2,bit)
    a,b=x[...,0,:],x[...,1,:];c,s=(angle/2).cos().square(),(angle/2).sin().square()
    return torch.stack((c*a+s*b,s*a+c*b),-2).reshape_as(intensity)


def input_dephased_intensity(padded,angles):
    need(angles.shape[0]==2 and 1<<angles.shape[1]==padded.shape[-1],'complete local net-angle register')
    result=padded.square()
    for q in range(angles.shape[1]):result=intensity_transfer(result,angles[:,q].sum(),q)
    need(bool(torch.isfinite(result).all()) and bool((result>=0).all()) and
         torch.allclose(result.sum(-1),padded.square().sum(-1),atol=1e-10,rtol=1e-10),'finite positive mass-preserving exact input dephasing')
    return result


class LocalActivation(GlobalActivation):
    def transform(self,padded,angles):return circuit(padded,angles,False)

    def activation(self,z,kind,reference=False):
        d=z.shape[-1];flat=z.double().reshape(-1,d);p=self.parameters_for(kind,reference).reshape(2,self.wires)
        need(d==self.widths[kind] and bool(torch.isfinite(flat).all()),'finite complete local mode input')
        parts=[]
        for x in flat.split(self.chunk):
            padded=F.pad(x,(0,(1<<self.wires)-d));r=padded.square().mean(-1,keepdim=True).sqrt()
            need(bool(torch.isfinite(r).all()) and bool((r>torch.finfo(torch.float32).tiny).all()),'positive resolved native mode radius; no floor')
            if self.arm=='DEPHASED':value=input_dephased_intensity(padded,p)/r-r
            else:
                transformed=self.transform(padded,p)
                need(torch.allclose(transformed.square().sum(-1),padded.square().sum(-1),atol=1e-10,rtol=1e-10),'complete local coherent norm')
                if self.arm=='SIGNED_LOCAL':value=transformed
                elif self.arm=='GELU_LOCAL':value=F.gelu(transformed)
                elif self.arm=='ABS_LOCAL':value=transformed.abs()
                else:value=born_activation(transformed,r)
            need(bool(torch.isfinite(value).all()),'finite local decoder without clipping')
            parts.append(value[...,:d].to(z.dtype))
        return torch.cat(parts).reshape_as(z)

    def padding_probe(self,z,kind,reference=False):
        with torch.no_grad():
            p=self.parameters_for(kind,reference).reshape(2,self.wires);n=1<<self.wires;values=[]
            for x in z.double().reshape(-1,z.shape[-1]).split(self.chunk):
                padded=F.pad(x,(0,n-x.shape[-1]))
                intensity=(input_dephased_intensity(padded,p) if self.arm=='DEPHASED' else self.transform(padded,p).square())
                values.append(intensity[...,x.shape[-1]:].sum(-1)/padded.square().sum(-1))
            a=torch.cat(values)
        return dict(applicable=True,candidates=len(a),minimum=float(a.min()),mean=float(a.mean()),maximum=float(a.max()))


class LocalReadout(OutputFrameReadout):
    def __init__(self,packed,arm,seed,wires=10,chunk=64):
        super().__init__(packed,'RANDOM_FRAME' if arm=='RANDOM_FRAME' else 'NO_CNOT',seed,wires,chunk);self.arm=arm
        self.shared=LocalActivation({k:p['ffn_dense']['weight'].shape[0] for k,p in packed.items()},arm,seed+100,wires,chunk)

    def mechanism_diagnostics(self):
        out=super().mechanism_diagnostics()
        out.update(mechanism='Local U=tensorRY(theta1+theta2), noCNOT. Same native output-mode frame and fullreference/native-tail/currenthead. DEPHASED=inputcomputationaldephase then SAMEU thenmeasurement; notbetweenlayerdephasing.',
            limits=out['limits']+' Local20nominal/10effective/model; inputcoherence deletion isolatesoff-diagonalterms inthisfixedframe, notentanglementnecessity/quantumexclusivity. 146postselection/adaptive20held andnegativefullCNOT retained. SIGNED/GELU/ABS sameU butnotamplitudematched.')
        return out


def factory(packed,arm,seed,wires=10,chunk=64):
    need(arm in ARMS,'all declared local attribution controls')
    if arm=='HEAD':return DepthProductReadout(packed,'HEAD',seed)
    if arm=='QUANTUM':return OutputFrameReadout(packed,'NO_CNOT',seed,wires,chunk)
    if arm in ('CLASSICAL_MATCHING','MLP8'):return OutputFrameReadout(packed,arm,seed,wires,chunk)
    return LocalReadout(packed,arm,seed,wires,chunk)


def prepare(packed,targets):
    root=Path(__file__).resolve().parents[3]/'output-frame-run-v001/ffn-readouts'
    need(digest(root/'result.json')=='55b148ee6ead0f5f0622166d4406fe4fe477484172fe0970801842b525e81231','exact accepted146 frame manifest')
    for k,sha in [('dpr','5ce41aecce2eafe8fc280b98e9582971931f86eccb95396ad25d260314d89632'),('bge','aff49b19238daa21244d41db7aa2df6d0790128b837b8dcae4476f3e769ff34e')]:
        need(digest(root/(k+'-readouts.bin'))==sha,'exact accepted complete146 frame member')
    coords={k:torch.cat([torch.cat((p[n]['train'],p[n]['holdout'])).cuda() for n in ('ffn_z','ffn_residual')],-1) for k,p in packed.items()}
    return dict(coords=coords,mapping=dict(kind='unchanged146 fullnativeFFN/output-mode frames',source={k:p['ffn_frame_source'] for k,p in packed.items()}),
        arms=ARMS,factory=lambda arm,seed:factory(packed,arm,seed),protocol=dict(candidate='147_LOCAL_NATIVE_MODE_BORN_ATTRIBUTION',
        mechanism='Exact146NO_CNOT localBorn candidate. U=tensorRY(theta1+theta2),20nominal10effective/model. SameSVD y/padding/normalizedsquare/completefrozenreference/originalnative-tail/head; same3seeds200/allpairloss/8pairedqueries.',
        controls='HEAD; sameU SIGNED/GELU/ABS; exactinput-DEPHASED net-angle stochastictransfer; RANDOM_FRAME sameNO_CNOT; exact146matching20/MLP8. Allindependentreplacement/deletioncontrasts requiredperbackbone. FullCNOT146 remainscomparativecontext, notretroactivelyregraded. Mathematicalsamefunctionsimulator correctnessoracle only.',
        limits='Selectedafter146exposedheldresults, prospectiveattribution notindependentconfirmation. No entangling-gate necessity/hardware/exclusivity/newprimitive/fullsmallsetclaim. Fixedframeinputoff-diagonalterms only; all146rank/radius/zero-sum/cost/nonunique-frame limitations retained. No gain/seed/epoch/widthretune.'))
