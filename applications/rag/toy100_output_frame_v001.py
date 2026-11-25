"""146: known Born FFN nonlinearity in pinned native output-mode coordinates.

The unconstrained rowspace representation lemma is not a reachability claim for
the actual circuit. In particular BGE's full-register Born difference sums zero.
"""
import torch
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need
from applications.rag.toy100_ffn_activation_v001 import FFNReadout
from applications.rag.toy100_global_ffn_v001 import GlobalActivation,GlobalFFNReadout,compact_frozen_tails
from applications.rag.toy100_depth_product_born_v001 import DepthProductReadout

ARMS=('HEAD','OLD_RAW','NO_CNOT','SIGNED_U','GELU_U','RANDOM_FRAME','CLASSICAL_MATCHING','MLP8','QUANTUM')


def output_frames(weight,seed=14677):
    """All native output modes, no rank truncation or singular-value floor."""
    need(weight.dtype==torch.float32 and weight.ndim==2 and bool(torch.isfinite(weight).all()),'finite original FP32 output matrix')
    left,singular,right=torch.linalg.svd(weight.double(),full_matrices=False)
    h,d=weight.shape;threshold=max(h,d)*torch.finfo(weight.dtype).eps*singular[0]
    need(len(singular)==h and bool(singular[-1]>threshold),'full native row rank at frozen original-dtype threshold; no rank drop')
    signs=right[torch.arange(h,device=weight.device),right.abs().argmax(-1)].sign()
    need(bool((signs.abs()==1).all()),'nonzero deterministic SVD sign anchors')
    right=right*signs[:,None];left=left*signs[None,:];basis=right.T.contiguous()
    eye=torch.eye(h,dtype=torch.float64,device=weight.device)
    orth=float((basis.T@basis-eye).abs().max())
    reconstruction=float(((left*singular)@right-weight.double()).abs().max())
    projection=float((weight.double()@basis@basis.T-weight.double()).abs().max())
    need(orth<=2e-10 and reconstruction<=2e-10 and projection<=2e-10,'thin SVD orthogonality/reconstruction/fullrowspace')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    q,r=torch.linalg.qr(torch.randn(h,h,generator=generator,dtype=torch.float64).to(weight.device))
    q=q*r.diagonal().sign()[None,:]
    random=basis@q
    need(float((random.T@random-eye).abs().max())<=2e-10,'paired random orthogonal frame')
    return dict(basis=basis,random_basis=random,diagnostics=dict(output=h,intermediate=d,
        minimum_singular=float(singular[-1]),maximum_singular=float(singular[0]),rank_threshold=float(threshold),
        orthogonality_maximum=orth,reconstruction_maximum=reconstruction,row_projection_maximum=projection,
        frame_unique=False,random_frame_seed=seed,rank_truncated=False))


class OutputFrameReadout(GlobalFFNReadout):
    def __init__(self,packed,arm,seed,wires=10,chunk=64):
        FFNReadout.__init__(self,packed,'HEAD',seed);self.arm=arm
        widths={k:p['ffn_dense']['weight'].shape[0] for k,p in packed.items()}
        chosen='QUANTUM' if arm=='RANDOM_FRAME' else arm
        self.shared=GlobalActivation(widths,chosen,seed+100,wires,chunk)
        for k,p in packed.items():
            frame=p['ffn_frames']['random_basis' if arm=='RANDOM_FRAME' else 'basis']
            need(frame.dtype==torch.float64 and tuple(frame.shape)==(self.tails[k].intermediate,widths[k]),'source-bound complete native output frame')
            self.register_buffer('frame_'+k,frame.clone(),persistent=False)
            self.register_buffer('initial_head_'+k,p['original_head']['weight'].clone())
            self.register_buffer('probe_'+k,p['ffn_z']['train'][0].double()@frame,persistent=False)
        compact_frozen_tails(self)

    def project(self,k,z):return z.double()@getattr(self,'frame_'+k)

    def references(self,coords,native,features=None):
        with torch.no_grad():return {k:self.shared.activation(self.project(k,v[...,:self.tails[k].intermediate]),k,True) for k,v in coords.items()}

    def read(self,k,coords,reference=None):
        d=self.tails[k].intermediate;z,residual=coords[...,:d],coords[...,d:]
        y=self.project(k,z)
        if reference is None:
            with torch.no_grad():reference=self.shared.activation(y,k,True)
        difference=self.shared.activation(y,k)-reference
        activation=F.gelu(z)+(difference@getattr(self,'frame_'+k).T).to(z.dtype)
        return self.tails[k](activation,residual)

    def mechanism_diagnostics(self):
        probes={k:{n:self.shared.padding_probe(getattr(self,'probe_'+k),k,ref)
            for n,ref in [('initial',True),('final',False)]} for k in ('dpr','bge')}
        return dict(critic_parameters=sum(p.numel() for p in self.shared.parameters()),padding_mass_probe=probes,
            probe_scope='first TRAIN query/all50, not fullpopulation/trajectory',
            mechanism='Pinned full native output-rowspace SVD modes. y=V^Tz; pad768/1024 to1024. OriginalGELU+V*(Aθ(y)-Ainit(y)); unchangednative tail/currenthead. Randomorientation uses pairedencode/decode.',
            limits='KnownexactclassicalstructuredU/square, no newprimitive/hardware/exclusivity. Unconstrainedrowspacelemma notactualcircuitcoverage: BGEdeltaA sums0/excludesonepreLNoutputdirection. Randomframe also changesexcluded direction andgate/singularalignment, notpuresemanticcausality. V^Tz canvanish forvalidnativeinput; fataldoesnotmakeinputinvalid. FullprobabilityandO(Dh)framecosts; no amplitude/conditioningmatch, repeatedsingularframesnonunique. No_CNOT20nominal10effective, inputmayalreadyentangle. Frozenframe/tail reconstructedbyhashedfactory.')


def factory(packed,arm,seed):
    need(arm in ARMS,'all declared output-mode controls')
    if arm=='HEAD':return DepthProductReadout(packed,'HEAD',seed)
    if arm=='OLD_RAW':return GlobalFFNReadout(packed,'QUANTUM',seed)
    return OutputFrameReadout(packed,arm,seed)


def prepare(packed,targets):
    coords={k:torch.cat([torch.cat((p[n]['train'],p[n]['holdout'])).cuda() for n in ('ffn_z','ffn_residual')],-1) for k,p in packed.items()}
    return dict(coords=coords,mapping=dict(kind='complete original native FFN plus pinned fullrowspace output modes',
        source={k:p['ffn_frame_source'] for k,p in packed.items()}),arms=ARMS,
        factory=lambda arm,seed:factory(packed,arm,seed),protocol=dict(candidate='146_NATIVE_OUTPUT_MODE_FRAME_BORN_FFN',
        mechanism='Fullrank thinSVD W=LΣVᵀ, fixedactualV. y=Vᵀz, N1024,10qubit RY/CNOTchain/RY20angles/model. A=u²/r-r, originalGELU+V*(A-Afrozeninitial), originaltail/currenthead. Fixed3seeds200/allpairloss/8querybatch.',
        controls='HEADexact; OLD_RAWexact145Q; sameframe NO_CNOT/SIGNED_U/GELU_U/randommatchingBorn/MLP8; RANDOM_FRAME_Q samefullrowspace differentorientation; Q. MLP8has17h+8parameters/model vsQ20. Randomframe isorientationcontrol, notnoquantum. ExactQclassicalimplementation onlycorrectnessoracle.',
        limits='KnownBorn/unitarynewrepresentationconfiguration, no firstness/hardware/exclusivity/independentconfirmation/fullsmallsetclaim. Fullrowspaceunconstrainedrepresentation notactualreachability; BGEBornzero-sum excludesonepreLNoutputdirection. Sameunitsnotoutputscalematch; actualframecosts included; exactbasis/SVDnonuniqueness pinned. No145gain/seed/epoch/widthretune.'))
