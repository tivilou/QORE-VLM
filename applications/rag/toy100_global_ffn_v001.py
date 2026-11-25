"""145: full native FFN amplitude interference, exact classical simulation.

Known structured orthogonal mixing plus a variance-normalized square map.
State preparation and all-coordinate probability readout are not free.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need
from applications.rag.toy100_ffn_activation_v001 import FFNReadout
from applications.rag.toy100_block_activation_v001 import BlockActivation
from applications.rag.toy100_depth_product_born_v001 import DepthProductReadout

ARMS=('HEAD','OLD_LOCAL','NO_CNOT','SIGNED_U','GELU_U','CLASSICAL_MATCHING','MLP8','QUANTUM')
WIRES=12
CHUNK=64


def ry(state,angle,wire):
    n=state.shape[-1];bit=1<<wire
    x=state.reshape(*state.shape[:-1],n//(2*bit),2,bit)
    a,b=x[...,0,:],x[...,1,:];c,s=(angle/2).cos(),(angle/2).sin()
    return torch.stack((c*a-s*b,s*a+c*b),-2).reshape_as(state)


def cnot_indices(n,control,target,device):
    i=torch.arange(n,device=device)
    return i^(((i>>control)&1)<<target)


def circuit(state,angles,entangle=True):
    wires=int(math.log2(state.shape[-1]))
    need(1<<wires==state.shape[-1] and angles.shape==(2,wires),'complete real amplitude register')
    value=state
    for wire in range(wires):value=ry(value,angles[0,wire],wire)
    if entangle:
        for wire in range(wires-1):value=value[...,cnot_indices(value.shape[-1],wire,wire+1,value.device)]
    for wire in range(wires):value=ry(value,angles[1,wire],wire)
    return value


def matching_circuit(state,angles,permutations,inverses=None):
    value=state
    for index,(angle,permutation) in enumerate(zip(angles.flatten(),permutations)):
        pairs=value[...,permutation].reshape(*value.shape[:-1],-1,2)
        a,b=pairs[...,0],pairs[...,1];c,s=(angle/2).cos(),(angle/2).sin()
        mixed=torch.stack((c*a-s*b,s*a+c*b),-1).flatten(-2)
        inverse=torch.argsort(permutation) if inverses is None else inverses[index]
        value=mixed[...,inverse]
    return value


def born_activation(value,radius):
    # Equivalent to r*(N*p-1); no renormalization after DPR projection.
    return value.square()/radius-radius


class GlobalActivation(nn.Module):
    def __init__(self,widths,arm,seed,wires=WIRES,chunk=CHUNK):
        super().__init__();self.arm=arm;self.widths=widths;self.wires=wires;self.chunk=chunk
        self.layout={};values=[];offset=0;n=1<<wires
        g=torch.Generator(device='cpu').manual_seed(seed+14500)
        for k,d in widths.items():
            need(d<=n,'zero padding only, no native neuron discarded')
            if arm=='MLP8':
                v=torch.cat((torch.randn(8*d,generator=g,dtype=torch.float64)/math.sqrt(d),
                             torch.zeros(8+8*d+d,dtype=torch.float64)))
            else:v=.1*torch.randn(2*wires,generator=g,dtype=torch.float64)
            self.layout[k]=(offset,len(v));offset+=len(v);values.append(v)
        self.weights=nn.Parameter(torch.cat(values));self.register_buffer('initial',self.weights.detach().clone())
        if arm=='CLASSICAL_MATCHING':
            g=torch.Generator(device='cpu').manual_seed(14577)
            permutations=torch.stack([torch.randperm(n,generator=g) for _ in range(2*wires)])
            self.register_buffer('permutations',permutations,persistent=False)
            self.register_buffer('inverse_permutations',permutations.argsort(-1),persistent=False)

    def parameters_for(self,k,reference=False):
        start,size=self.layout[k]
        return (self.initial if reference else self.weights)[start:start+size]

    def transform(self,padded,angles):
        if self.arm=='CLASSICAL_MATCHING':return matching_circuit(padded,angles,self.permutations,self.inverse_permutations)
        return circuit(padded,angles,self.arm!='NO_CNOT')

    def activation(self,z,kind,reference=False):
        dimension=z.shape[-1];flat=z.double().reshape(-1,dimension)
        need(dimension==self.widths[kind] and bool(torch.isfinite(flat).all()),'finite actual full native FFN channels')
        p=self.parameters_for(kind,reference);parts=[]
        for x in flat.split(self.chunk):
            if self.arm=='MLP8':
                hidden=F.linear(x,p[:8*dimension].reshape(8,dimension),p[8*dimension:8*dimension+8]).tanh()
                value=F.linear(hidden,p[8*dimension+8:-dimension].reshape(dimension,8),p[-dimension:])
            else:
                padded=F.pad(x,(0,(1<<self.wires)-dimension));radius=padded.square().mean(-1,keepdim=True).sqrt()
                need(bool(torch.isfinite(radius).all()) and bool((radius>torch.finfo(torch.float32).tiny).all()),
                     'finite resolved positive native radius, no floor')
                transformed=self.transform(padded,p.reshape(2,self.wires))
                need(torch.allclose(transformed.square().sum(-1),padded.square().sum(-1),atol=1e-10,rtol=1e-10),
                     'actual complete orthogonal amplitude norm')
                if self.arm=='SIGNED_U':value=transformed
                elif self.arm=='GELU_U':value=F.gelu(transformed)
                else:value=born_activation(transformed,radius)
                value=value[...,:dimension]
            need(bool(torch.isfinite(value).all()),'finite un-clipped full FFN activation')
            parts.append(value.to(z.dtype))
        return torch.cat(parts).reshape_as(z)

    def padding_probe(self,z,kind,reference=False):
        p=self.parameters_for(kind,reference)
        if self.arm=='MLP8':return dict(applicable=False)
        with torch.no_grad():
            n=1<<self.wires;flat=z.double().reshape(-1,z.shape[-1]);mass=[]
            for x in flat.split(self.chunk):
                padded=F.pad(x,(0,n-x.shape[-1]));value=self.transform(padded,p.reshape(2,self.wires))
                mass.append(value[...,x.shape[-1]:].square().sum(-1)/padded.square().sum(-1))
            a=torch.cat(mass)
        return dict(applicable=True,candidates=len(a),minimum=float(a.min()),mean=float(a.mean()),maximum=float(a.max()))


def compact_frozen_tails(module):
    # These exact immutable buffers are reconstructed by the pinned factory.
    # Trainable state, optimizer, initial critic reference and RNG remain saved.
    for tail in module.tails.values():
        need(not list(tail.parameters()),'only source-bound immutable native tail buffers')
        for name,value in list(tail.named_buffers(recurse=False)):
            tail.register_buffer(name,value,persistent=False)


class GlobalFFNReadout(FFNReadout):
    batch_local_references=False

    def __init__(self,packed,arm,seed,wires=WIRES,chunk=CHUNK):
        super().__init__(packed,'HEAD',seed);self.arm=arm
        if arm=='OLD_LOCAL':self.shared=BlockActivation('QUANTUM',seed)
        else:self.shared=GlobalActivation({k:p['ffn_z']['train'].shape[-1] for k,p in packed.items()},arm,seed,wires,chunk)
        for k,p in packed.items():
            self.register_buffer('initial_head_'+k,p['original_head']['weight'].clone())
            self.register_buffer('probe_'+k,p['ffn_z']['train'][0].clone(),persistent=False)
        compact_frozen_tails(self)

    def gradient_probe_parameter(self):return self.shared.theta if self.arm=='OLD_LOCAL' else self.shared.weights

    def read(self,kind,coords,reference=None):
        d=self.tails[kind].intermediate;z,residual=coords[...,:d],coords[...,d:]
        if reference is None:
            with torch.no_grad():reference=self.shared.activation(z,kind,True)
        difference=self.shared.activation(z,kind)-reference
        if self.arm=='OLD_LOCAL':difference=getattr(self,'scale_'+kind)*difference
        return self.tails[kind](F.gelu(z)+difference,residual)

    def one(self,k,features,native,coords,reference=None):
        d=self.tails[k].intermediate
        with torch.no_grad():frozen=self.tails[k](F.gelu(coords[...,:d]),coords[...,d:])
        base=native+F.linear(features,self.heads[k].weight-getattr(self,'initial_head_'+k)).squeeze(-1)
        return base+F.linear(self.read(k,coords,reference)-frozen,self.heads[k].weight).squeeze(-1)

    def forward(self,features,native,originals,coords,references):
        return {k:self.one(k,features[k],native[k],coords[k],references[k]) for k in ('dpr','bge')}

    def forward_one(self,k,features,native,original,coords):return self.one(k,features,native,coords)

    def mechanism_diagnostics(self):
        probes={}
        if self.arm!='OLD_LOCAL':
            probes={k:{name:self.shared.padding_probe(getattr(self,'probe_'+k),k,ref)
                       for name,ref in [('initial',True),('final',False)]} for k in ('dpr','bge')}
        return dict(critic_parameters=sum(p.numel() for p in self.shared.parameters()),padding_mass_probe=probes,
            probe_scope='first TRAIN query/all50 only, not a population or trajectory bound',
            mechanism='AllnativeFFNchannels paddedto4096. NativeGELU plus completeinitialdifference of structuredorthogonalmixing/normalizedquadraticBorn map, then originalfrozenoutputdense/residual/LN/BGEbridge/currenthead.',
            limits='Exactclassicalpermutation/Givens/square map, knownBornactivation/newconfiguration. No_CNOT24nominal/12effective/model; inputmayalreadyentangle. Randommatchingdifferentprior notsuperset. Sameunitsnotamplitudematch: |deltaA_i|<=N*r. DPRpaddingparticipates, no truncatedprobabilityrenormalization. Fullprobabilityreadoutcostnotfree. OLD_LOCALalsochangesnormalizationgeometry. Frozen tail reconstructed from hashedreadouts, not duplicated in CP. Adaptive20held/nohardware/exclusivity/firstness/fullsmallsetclaim.')


def factory(packed,arm,seed):
    need(arm in ARMS,'all declared global FFN controls')
    if arm=='HEAD':return DepthProductReadout(packed,'HEAD',seed)
    return GlobalFFNReadout(packed,arm,seed)


def prepare(packed,targets):
    coords={k:torch.cat([torch.cat((p[n]['train'],p[n]['holdout'])).cuda() for n in ('ffn_z','ffn_residual')],-1)
            for k,p in packed.items()}
    return dict(coords=coords,mapping=dict(kind='source-bound fullnative finalCLS preGELU/residual',wires=12,chunk=64,
        source={k:p['global_ffn_source'] for k,p in packed.items()}),arms=ARMS,
        factory=lambda arm,seed:factory(packed,arm,seed),protocol=dict(candidate='145_GLOBAL_NATIVE_FFN_BORN_INTERFERENCE',
        mechanism='r=||padz||/sqrt4096; Aθ=(Uθpadz)^2/r−r. NativeGELU+Aθ−Afrozeninitial, exactoriginalnative tail. 12RY/CNOTchain/12RY,24angles/model, fullnative dimensions andownhead; fixedoriginal3seeds200/AdamW/8queryallpairloss.',
        controls='HEAD; OLD_LOCALoriginal6channel3qubitactivation+TRAINperneuronstd; NO_CNOT24nominal12effective; SIGNED_U; GELU_U; CLASSICAL_MATCHING24seededglobalperfectmatchings+sameBornmap; MLP8moreparameters/sameinsertion; Q. Sameunitsnotcalibratedequalamplitudes. ExactclassicalQidentity is oracle not independentcontrol requiredtolose.',
        lifecycle='Frozen tail/probe buffers rebuilt fromSHA-boundFFNreadouts atfactory, not CPduplicated. Complete trainable/initialreference/optimizer/RNG retained; actualnull/fullstatecontinuation/checkpoint/ownbackbonereplay required. Source/file/role/token/native/fulltailparityexport before study.',
        limits='Knownunitary/Bornmechanisms, nohardware/computationalexclusivity/independentconfirmation/fullsmallsetclaim. AllDPRpaddingmassretainedintheunitary; onlyzero-extendedoriginalWout discardsoutputcoordinates. No probabilityrenormalization or gain/epoch/seedsearch.'))
