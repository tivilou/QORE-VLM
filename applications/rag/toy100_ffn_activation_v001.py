"""Shared local quantum activation inside each native last-layer FFN.

Keep every original FFN neuron, frozen output projection/residual/LayerNorm and
BGE dense+tanh. Only train a small shared activation module and native heads.
"""
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.toy100_block_activation_v001 import BlockActivation,BLOCK_ARMS
from applications.rag.toy100_native_feature_readout_v001 import NativeFeatureReadout


class FrozenTail(nn.Module):
    def __init__(self,p):
        super().__init__();self.eps=p['ffn_eps'];self.intermediate=p['ffn_dense']['weight'].shape[1]
        for prefix,key in (('dense','ffn_dense'),('norm','ffn_normalization'),('bridge','ffn_bridge')):
            for n,v in (p[key] or {}).items():self.register_buffer(prefix+'_'+n,v.clone())

    def forward(self,activation,residual):
        y=F.linear(activation,self.dense_weight,self.dense_bias)
        y=F.layer_norm(y+residual,(residual.shape[-1],),self.norm_weight,self.norm_bias,self.eps)
        return F.linear(y,self.bridge_weight,self.bridge_bias).tanh() if hasattr(self,'bridge_weight') else y


class FFNReadout(NativeFeatureReadout):
    explicit_context=True

    def __init__(self,packed,arm,seed):
        super().__init__(packed,'HEAD',seed);self.arm=arm;self.tails=nn.ModuleDict({k:FrozenTail(p) for k,p in packed.items()})
        for k,p in packed.items():self.register_buffer('scale_'+k,F.gelu(p['ffn_z']['train']).std((0,1),unbiased=False))
        if arm!='HEAD':self.shared=BlockActivation(arm,seed)

    def gradient_probe_parameter(self):return self.shared.theta if hasattr(self.shared,'theta') else self.shared.weight

    def activity_gradient_names(self):return () if self.shared is None else tuple('shared.'+n for n,_ in self.shared.named_parameters())

    def references(self,coords,native,features=None):
        with torch.no_grad():return {k:torch.zeros_like(v[...,:self.tails[k].intermediate]) if self.shared is None else self.shared.activation(v[...,:self.tails[k].intermediate],k,True) for k,v in coords.items()}

    def read(self,kind,coords,reference=None):
        d=self.tails[kind].intermediate;z,residual=coords[...,:d],coords[...,d:];activation=F.gelu(z)
        if self.shared is not None:
            if reference is None:
                with torch.no_grad():reference=self.shared.activation(z,kind,True)
            activation=activation+getattr(self,'scale_'+kind)*(self.shared.activation(z,kind)-reference)
        return self.tails[kind](activation,residual)

    def forward_one(self,kind,features,native,original,coords):
        # Recomputed currenthead on frozen unmodified nativeFFN cancels the
        # numerical export offset; separately retain head learning on originalx.
        with torch.no_grad():
            d=self.tails[kind].intermediate;frozen=self.tails[kind](F.gelu(coords[...,:d]),coords[...,d:])
        delta=self.heads[kind](self.read(kind,coords)).squeeze(-1)-self.heads[kind](frozen).squeeze(-1)
        return native+self.heads[kind](features).squeeze(-1)-original+delta

    def forward(self,features,native,originals,coords,references):
        result={}
        for k in ('dpr','bge'):
            d=self.tails[k].intermediate
            with torch.no_grad():frozen=self.tails[k](F.gelu(coords[k][...,:d]),coords[k][...,d:])
            delta=self.heads[k](self.read(k,coords[k],references[k])).squeeze(-1)-self.heads[k](frozen).squeeze(-1)
            result[k]=native[k]+self.heads[k](features[k]).squeeze(-1)-originals[k]+delta
        return result


def prepare(packed,targets):
    coords={k:torch.cat([torch.cat((p[n]['train'],p[n]['holdout'])).cuda() for n in ('ffn_z','ffn_residual')],-1) for k,p in packed.items()}
    return dict(coords=coords,mapping=dict(kind='original_native_finalCLS_FFN_preGELU_plus_attentionresidual'),arms=BLOCK_ARMS,
        factory=lambda arm,seed:FFNReadout(packed,arm,seed),protocol=dict(candidate='NATIVE_FFN_QUANTUM_ACTIVATION_v001',
        paired_mapping='All3072/4096 nativeFFN preGELU channels plus nativeattentionresidual; noinputcompression. Shared6channelactivation perbackbone,12thetaeach, fullnativeoutputdense/LayerNorm/BGE dense+tanh frozen.',
        circuit='Same threequbit2reupload6moments aspriorfeatureactivation, nowinside nativeFFN beforeoutputprojection/residual/LayerNorm/bridge. OriginalGELU plus TRAIN-onlyperneuronGELUstd timescompleteinitialactivationreference difference.',
        shared='24addedquantumparameters total; disjointD/Brows, no sharedweighttransfer. Original1792headweights equallytrained. Fixednativeanchor+headupdate+(currentFFNcurrenthead-frozenFFNcurrenthead), cancels nativeexport rounding at null.',
        controls='All sameFFNinsertion: MLP84params/TRIG156, PRODUCT24 noCNOT, MARGINAL24 XXto<Xa><Xb>, QUANTUM24, HEAD. Same200stepsallpairloss/all3seeds; parameterandcompute differencesdisclosed.'))
