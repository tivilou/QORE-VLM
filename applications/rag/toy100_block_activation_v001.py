"""Full-dimensional native readout adapter built from local quantum activations.

All native channels are retained: deterministic six-channel blocks, no learned
projection or global scalar bottleneck. Blocks share one circuit per backbone.
This is a structural feature-map test, not a hardware/computational claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.toy100_shared_readout_v001 import statevector
from applications.rag.toy100_native_feature_readout_v001 import NativeFeatureReadout
from applications.rag.qvalue_portable_v001 import need

BLOCK_ARMS=('HEAD','MLP','TRIG','PRODUCT','MARGINAL_PRODUCT','QUANTUM')


def six_moments(state,marginal=False):
    indices=torch.arange(8,device=state.device)
    signs=torch.stack([1-2*((indices>>wire)&1) for wire in range(3)]).to(state.real.dtype)
    z=state.abs().square()@signs.T
    if marginal:
        x=torch.stack([(state.conj()*state[...,indices^(1<<wire)]).sum(-1).real for wire in range(3)],-1)
        pair=torch.stack((x[...,0]*x[...,1],x[...,1]*x[...,2],x[...,2]*x[...,0]),-1)
    else:pair=torch.stack([(state.conj()*state[...,indices^(1<<a)^(1<<b)]).sum(-1).real for a,b in ((0,1),(1,2),(2,0))],-1)
    return torch.cat((z,pair),-1)


class BlockActivation(nn.Module):
    def __init__(self,arm,seed):
        super().__init__();self.arm=arm;g=torch.Generator(device='cpu').manual_seed(seed)
        def parameter(shape,scale):
            x=torch.randn(shape,generator=g)*scale
            return nn.Parameter(x[None].repeat(2,*([1]*x.ndim)))
        if arm in ('QUANTUM','PRODUCT','MARGINAL_PRODUCT'):self.theta=parameter((2,3,2),.1)
        else:
            width=6 if arm=='MLP' else 12;self.weight=parameter((6,width),1/math.sqrt(width));self.bias=nn.Parameter(torch.zeros(2,6))
        for name,value in list(self.named_parameters()):self.register_buffer('reference_'+name,value.detach().clone())

    def activation(self,features,kind,reference=False):
        domain=0 if kind=='dpr' else 1
        def par(name):return getattr(self,('reference_' if reference else '')+name)[domain]
        dimension=features.shape[-1];x=F.layer_norm(features,(dimension,));padding=(-dimension)%6
        x=F.pad(x,(0,padding)).reshape(*x.shape[:-1],-1,6);phi=math.pi*(x/2).tanh()
        if self.arm in ('MLP','TRIG'):
            value=phi if self.arm=='MLP' else torch.cat((phi.sin(),phi.cos()),-1)
            output=F.linear(value,par('weight'),par('bias')).tanh()
        else:output=six_moments(statevector(phi,par('theta'),self.arm!='PRODUCT'),self.arm=='MARGINAL_PRODUCT')
        return output.flatten(-2)[...,:dimension]


class BlockActivationReadout(NativeFeatureReadout):
    explicit_context=True

    def __init__(self,packed,arm,seed):
        super().__init__(packed,'HEAD',seed);self.arm=arm
        for kind,p in packed.items():self.register_buffer('scale_'+kind,p['features']['train'].std((0,1),unbiased=False))
        if arm!='HEAD':self.shared=BlockActivation(arm,seed)

    def references(self,coords,native,features=None):
        with torch.no_grad():return {k:torch.zeros_like(coords[k]) if self.shared is None else self.shared.activation(coords[k],k,True) for k in ('dpr','bge')}

    def gradient_probe_parameter(self):return self.shared.theta if hasattr(self.shared,'theta') else self.shared.weight

    def activity_gradient_names(self):return () if self.shared is None else ('shared.theta' if hasattr(self.shared,'theta') else 'shared.weight',)

    def adapt(self,kind,features,reference=None):
        if self.shared is None:return features
        if reference is None:
            with torch.no_grad():reference=self.shared.activation(features,kind,True)
        return features+getattr(self,'scale_'+kind)*(self.shared.activation(features,kind)-reference)

    def forward_one(self,kind,features,native,original,coords):
        return native+self.heads[kind](self.adapt(kind,features)).squeeze(-1)-original

    def forward(self,features,native,originals,coords,references):
        return {k:native[k]+self.heads[k](self.adapt(k,features[k],references[k])).squeeze(-1)-originals[k] for k in ('dpr','bge')}


def prepare(packed,targets):
    coords={k:torch.cat((p['features']['train'],p['features']['holdout'])).cuda() for k,p in packed.items()}
    return dict(coords=coords,mapping=dict(kind='full_native_classifier_inputs',blocks=6,padding='Only BGE two rightzeros before blockfunction; truncate outputs to original1024'),
                arms=BLOCK_ARMS,factory=lambda arm,seed:BlockActivationReadout(packed,arm,seed),
                protocol=dict(candidate='FULL_NATIVE_BLOCK_QUANTUM_ACTIVATION_v001',
                 paired_mapping='Every original native finalclassifierinput channel retained (DPR768, BGE1024 postnative dense+tanh). LayerNorm before rightpadding into sixchannelblocks. No randomprojection/PCA or pairedinference.',
                 circuit='Threequbit2reupload RY/RZ/trainRYRX, CNOT0to1/1to2 betweenblocks. Output Z0/Z1/Z2/X0X1/X1X2/X2X0, sixoutputs map back to same sixchannelpositions. One12theta circuit perbackbone shared across all its blocks.',
                 shared='Add TRAIN-only perchannelstd x fullinitial-reference activation difference to originalnativefeatures, then equallytrained originalhead. Fixedgain1/20epochs, originalfeature path andnativeanchor retained.',
                 controls='MLP6to6tanh42params, TRIG12sincos to6tanh78params; PRODUCT removesCNOT same12theta; MARGINAL_PRODUCT sameentangledstate replacesXX by<Xa><Xb> removingconnectedXX, notphysicaldephasing. Commonlayout/injection and supervision.'))
