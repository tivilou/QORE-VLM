"""105: full-native product encoding and normalized Born-MPS evidence.

Exact small-rank classical contractions, not a computational quantum advantage.
Input/optimizer/evaluation lifecycle belongs to the existing shared runner.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need

ARMS=('HEAD','MLP8','QUADRATIC8','CP4_BORN','PRODUCT','DEPHASED','TT_AMPLITUDE','QUANTUM')
KINDS=('dpr','bge')
BLOCK=32


def product_inputs(z,block=BLOCK):
    need(z.shape[-1]%block==0,'complete fixed contiguous native blocks')
    blocks=z.reshape(*z.shape[:-1],-1,block)/math.sqrt(block)
    x=torch.cat((torch.ones_like(blocks[...,:1]),blocks),-1)
    return x/torch.linalg.vector_norm(x,dim=-1,keepdim=True)


def scaled_product(matrices):
    """Balanced exact product up to positive Frobenius scales.

    Zero matrices divide by one, retaining their ordinary linear graph. This is
    essential for the signed TT amplitude's nonzero derivative at a zero node.
    """
    value=matrices
    scale=torch.zeros(value.shape[:-2],dtype=value.dtype,device=value.device)
    while True:
        norm=torch.linalg.vector_norm(value,dim=(-2,-1))
        divisor=torch.where(norm>0,norm,torch.ones_like(norm))
        value=value/divisor[...,None,None];scale=scale+divisor.log()
        if value.shape[-3]==1:return value[...,0,0,0],scale[...,0]
        length=value.shape[-3];pairs=length//2
        joined=value[...,:2*pairs:2,:,:]@value[...,1:2*pairs:2,:,:]
        logs=scale[...,:2*pairs:2]+scale[...,1:2*pairs:2]
        if length%2:
            joined=torch.cat((joined,value[...,-1:,:,:]),-3)
            logs=torch.cat((logs,scale[...,-1:]),-1)
        value,scale=joined,logs


def overlap_statistics(psi,cores,diagonal=False):
    rank=cores.shape[-1]
    transfer=torch.einsum('ldab,ldce->lacbe',cores,cores).reshape(cores.shape[0],rank*rank,rank*rank)
    entry,scale=scaled_product(transfer)
    need(bool(torch.isfinite(entry).all() and (entry>0).all()),'strictly positive finite full MPS norm')
    lognorm=entry.log()+scale
    if diagonal:
        weighted=torch.einsum('...ld,ldab,ldce->...lacbe',psi.square(),cores,cores)
        entry,scale=scaled_product(weighted.reshape(*psi.shape[:-2],psi.shape[-2],rank*rank,rank*rank))
        need(bool(torch.isfinite(entry).all() and (entry>=0).all()),'nonnegative actual dephased mass; no abs rescue')
    else:
        entry,scale=scaled_product(torch.einsum('...ld,ldab->...lab',psi,cores))
    return entry,scale,lognorm


def born_log_evidence(psi,cores,diagonal=False):
    entry,scale,lognorm=overlap_statistics(psi,cores,diagonal)
    nonzero=entry!=0
    safe=torch.where(nonzero,entry.abs(),torch.ones_like(entry))
    logmass=(safe.log()+scale)*(1 if diagonal else 2)-lognorm
    logmass=torch.where(nonzero,logmass,torch.full_like(logmass,-math.inf))
    # Multiplies both class probabilities by 33**L; the common factor cancels.
    return torch.logaddexp(logmass+math.log(.95)+psi.shape[-2]*math.log(psi.shape[-1]),
                          torch.full_like(logmass,math.log(.05)))


def signed_amplitude(psi,cores):
    entry,scale,lognorm=overlap_statistics(psi,cores)
    # Keep entry itself in the graph: sign/logabs would erase its zero gradient.
    return entry*torch.exp(scale-.5*lognorm)


def unpack_mps(raw,length,physical,rank):
    sizes=[physical*rank]+[(rank*physical*rank)]*(length-2)+[rank*physical]
    pieces=list(raw.split(sizes));cores=[]
    for site,piece in enumerate(pieces):
        left=1 if site==0 else rank;right=1 if site==length-1 else rank
        core=piece.reshape(left,physical,right).permute(1,0,2)
        cores.append(F.pad(core,(0,rank-right,0,rank-left)))
    return torch.stack(cores)


def cp_cores(raw,length,physical,rank=4):
    local=raw.reshape(length,rank,physical);out=[]
    for site in range(length):
        v=local[site].T
        if site==0:core=F.pad(v[:,None,:],(0,0,0,rank-1))
        elif site==length-1:core=F.pad(v[:,:,None],(0,rank-1))
        else:core=torch.diag_embed(v)
        out.append(core)
    return torch.stack(out)


class ProductMeasurement(nn.Module):
    def __init__(self,widths,arm,seed):
        super().__init__();self.arm=arm;self.widths=widths;self.layout={};draw=[];offset=0
        generator=torch.Generator(device='cpu').manual_seed(seed+10500)
        for kind in KINDS:
            width=widths[kind];length=width//BLOCK;physical=BLOCK+1
            if arm in ('MLP8','QUADRATIC8'):
                size=8*(width+1)+8
                value=torch.randn(size,generator=generator,dtype=torch.float64)/math.sqrt(width)
                value[8*width:8*(width+1)]=0
            elif arm=='CP4_BORN':
                size=2*length*4*physical
                value=.02/math.sqrt(physical)*torch.randn((2,length,4,physical),generator=generator,dtype=torch.float64)
                value[:,:,:,0]+=1;value=value.flatten()
            else:
                rank=1 if arm=='PRODUCT' else 2
                per=2*physical*rank+(length-2)*rank*physical*rank;size=2*per
                chunks=[]
                for label in range(2):
                    for site in range(length):
                        left=1 if site==0 else rank;right=1 if site==length-1 else rank
                        core=.02/math.sqrt(physical)*torch.randn((left,physical,right),generator=generator,dtype=torch.float64)
                        for i in range(min(left,right)):core[i,0,i]+=1
                        chunks.append(core.flatten())
                value=torch.cat(chunks)
            self.layout[kind]=(offset,size);draw.append(value);offset+=size
        self.weights=nn.Parameter(torch.cat(draw))
        self.register_buffer('initial_weights',self.weights.detach().clone())

    def states(self,kind,reference=False):
        start,size=self.layout[kind];weights=self.initial_weights if reference else self.weights
        raw=weights[start:start+size];length=self.widths[kind]//BLOCK;physical=BLOCK+1
        if self.arm=='CP4_BORN':return [cp_cores(v,length,physical) for v in raw.reshape(2,-1)]
        rank=1 if self.arm=='PRODUCT' else 2
        return [unpack_mps(v,length,physical,rank) for v in raw.reshape(2,-1)]

    def scalar(self,z,kind,reference=False):
        if self.arm in ('MLP8','QUADRATIC8'):
            start,size=self.layout[kind];raw=(self.initial_weights if reference else self.weights)[start:start+size]
            width=self.widths[kind];h=F.linear(z,raw[:8*width].reshape(8,width),raw[8*width:8*(width+1)])
            return (h.tanh() if self.arm=='MLP8' else h.square())@raw[-8:]
        psi=product_inputs(z);cores=self.states(kind,reference)
        values=[signed_amplitude(psi,c) if self.arm=='TT_AMPLITUDE' else
                born_log_evidence(psi,c,self.arm=='DEPHASED') for c in cores]
        return values[1]-values[0]


class ProductReadout(nn.Module):
    explicit_context=True
    def __init__(self,packed,mapping,arm,seed):
        super().__init__();need(arm in ARMS,'product-input arm');self.arm=arm;self.heads=nn.ModuleDict()
        widths={k:p['original_head']['weight'].shape[-1] for k,p in packed.items()}
        for kind,p in packed.items():
            original=p['original_head'];head=nn.Linear(widths[kind],1)
            head.load_state_dict(original);head.bias.requires_grad_(False);self.heads[kind]=head
            self.register_buffer('initial_head_'+kind,head.weight.detach().clone())
            self.register_buffer('mean_'+kind,mapping[kind]['mean'])
            self.register_buffer('std_'+kind,mapping[kind]['std'])
            self.register_buffer('native_scale_'+kind,p['native']['train'].double().std(correction=0))
        self.shared=None if arm=='HEAD' else ProductMeasurement(widths,arm,seed)
        if self.shared is not None:
            with torch.no_grad():
                for kind,p in packed.items():
                    z=(p['features']['train'].double()-getattr(self,'mean_'+kind))/getattr(self,'std_'+kind)
                    initial=self.shared.scalar(z,kind,True);std=initial.std(correction=0)
                    need(bool(torch.isfinite(std) and std>0),'finite nonconstant initial TRAIN4000 critic; no scale floor')
                    self.register_buffer('critic_std_'+kind,std)

    def references(self,coords,native,features=None):
        with torch.no_grad():return {k:torch.zeros_like(native[k]) if self.shared is None else self.shared.scalar(coords[k],k,True) for k in KINDS}

    def gradient_probe_parameter(self):return self.shared.weights
    def activity_gradient_names(self):return () if self.shared is None else ('shared.weights',)

    def score(self,kind,x,native,z,reference):
        value=native+F.linear(x,self.heads[kind].weight-getattr(self,'initial_head_'+kind)).squeeze(-1)
        if self.shared is not None:
            correction=(self.shared.scalar(z,kind)-reference)/getattr(self,'critic_std_'+kind)
            value=value+(getattr(self,'native_scale_'+kind)*correction).float()
        return value

    def forward(self,features,native,originals,coords,references):
        return {k:self.score(k,features[k],native[k],coords[k],references[k]) for k in KINDS}

    def forward_one(self,kind,x,native,original,coords):
        actual=(x.double()-getattr(self,'mean_'+kind))/getattr(self,'std_'+kind)
        need(torch.equal(actual,coords),'actual single-model full-native product input replay')
        with torch.no_grad():reference=None if self.shared is None else self.shared.scalar(actual,kind,True)
        return self.score(kind,x,native,actual,reference)

    def mechanism_diagnostics(self):
        return dict(arm=self.arm,extra_parameters=0 if self.shared is None else self.shared.weights.numel(),
            head_parameters=1792,critic_initial_std={} if self.shared is None else {k:float(getattr(self,'critic_std_'+k)) for k in KINDS},
            limits='Classically exact lowbond Born tensor model; common architecture, independent model parameters. Calibration matches initial scalar variance, not gradients/capacity. No computational/hardware or entanglement-exclusive attribution.')


def prepare(packed,targets):
    coords={};mapping={}
    with torch.no_grad():
        for kind,p in packed.items():
            train=p['features']['train'].double();mean=train.mean((0,1));std=train.std((0,1),correction=0)
            zero=std==0;std=torch.where(zero,torch.ones_like(std),std)
            need(bool(torch.isfinite(mean).all() and torch.isfinite(std).all()),'finite TRAIN-only channel statistics')
            mapping[kind]=dict(mean=mean,std=std,zero_std_channels=int(zero.sum()),block=BLOCK)
            coords[kind]=(torch.cat((p['features']['train'],p['features']['holdout'])).double().cuda()-mean.cuda())/std.cuda()
    def restore_mapping(path):
        saved=torch.load(path,weights_only=True)
        for kind in KINDS:
            need(torch.equal(saved[kind]['mean'],mapping[kind]['mean']) and torch.equal(saved[kind]['std'],mapping[kind]['std']),
                 'exact TRAIN-only native channel mapping replay')
    return dict(coords=coords,mapping=mapping,restore_mapping=restore_mapping,arms=ARMS,
        factory=lambda arm,seed:ProductReadout(packed,mapping,arm,seed),
        protocol=dict(candidate='PRODUCT_INPUT_BORN_MPS_v001',candidate_id=105,
            mechanism='TRAIN channel standardization; contiguous32native blocks with constant reference coordinate, product33-dimensional states. Two normalized real bond2 classMPS; log(.95 Born probability+.05/33^L) ratio. Full frozeninitial subtraction/initialTRAINcriticSTD*ownTRAINnativeSTD; equallylive nativehead. Class evidence, not jointPOVM/data likelihood.',
            controls='HEAD;MLP8 and QUADRATIC8 both14368;CP4_BORN14784;PRODUCT3696;DEPHASED,TT_AMPLITUDE,QUANTUM14256 each, all plus1792 heads. Same raw native input;DEPHASED deletes coordinate signs via psi.square(), so it is a sign-erasing deletion control. TT amplitude uses same product/MPS norm/parameterization, deletes Born-square/log. Capacities/Jacobians not matched;CP4 can also entangle.',
            paired_mapping='No paired alignment or PCA. Own fullnative TRAINmean/std and fixedcontiguous32channel product blocks;zeroTRAINstd divisor1.',
            shared='Same product-input/MPS architecture; disjoint ownmodel parameter slices trained jointly, independent native heads. No sharedweights or crossmodel inference.',
            distinction='83 amplitude-encodes whole native vector once, hence two squared affine features;105 tensor-product encodes all24DPR/32BGE blocks and learns multilinear MPS amplitudes beforeBorn square. No newphysics/firstness claim.',
            qualification='All3 fixedseeds200 finalcheckpoint; EACHbackbone total strictly exceeds native and EVERY classical/deletion control;jointseal beforeheldlabels.',
            limits='Adaptive label-stratified80TRAIN/20held only. Fixednative channelorder is an unverified lowrankprior. Nearproduct initialstate may suppress highorder gradients;Born nodes can have sharp gradients. Exact classical contraction/no hardware or complexity advantage. Same architecture, not shared weights or transfer.'))
