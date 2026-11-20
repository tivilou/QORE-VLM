"""Random tiny CPU layers only: exact full-layer FIDELITY optimization parity."""
import copy
import hashlib
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from applications.rag.top8_methods_v001 import Fidelity,_frozen_projections
from applications.rag.tests.test_top8_cache_numerics import NativeFixture
from pathlib import Path


def original_forward(model,row):
    """Pre-optimization full-sequence path; independent reference evaluation."""
    hidden=row['hidden'];att=model.layer.attention.self
    h=att.num_attention_heads;d=att.attention_head_size
    with torch.no_grad():
        q=att.query(hidden).reshape(len(hidden),-1,h,d).transpose(1,2)[:,:,0]
        k=att.key(hidden).reshape(len(hidden),-1,h,d).transpose(1,2)
    delta=model.effect(q,k);length=hidden.shape[1]
    selector=F.one_hot(torch.tensor(0,device=delta.device),length).to(delta.dtype).reshape(1,1,length,1)
    original=row['mask'];base=0. if original is None else original
    if original is not None and original.dtype==torch.bool:
        base=torch.zeros_like(original,dtype=delta.dtype).masked_fill(~original,torch.finfo(delta.dtype).min)
    bias=delta.unsqueeze(2)*selector
    if original is not None:bias=bias*(original if original.dtype==torch.bool else original==0).to(delta.dtype)
    with sdpa_kernel(SDPBackend.MATH):
        cls=model.layer(hidden,attention_mask=base+bias,output_attentions=False)[0][:,0]
    return model.head(model.bridge(cls)).squeeze(-1)


class FidelitySpeedTests(unittest.TestCase):
    def make(self,backbone='dpr',arm='QUANTUM',heads=1):
        native=NativeFixture(backbone,Path('.'))
        if heads!=1:
            from transformers import BertConfig,XLMRobertaConfig
            from transformers.models.bert.modeling_bert import BertLayer
            from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLayer
            config=(BertConfig if backbone=='dpr' else XLMRobertaConfig)(hidden_size=64*heads,
                num_attention_heads=heads,intermediate_size=128,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
            config._attn_implementation='sdpa' if backbone=='dpr' else 'eager'
            native.last=(BertLayer if backbone=='dpr' else XLMRobertaLayer)(config).eval().requires_grad_(False)
            native.head=nn.Linear(64*heads,1).eval().requires_grad_(False)
            from applications.rag.top8_backbone_v001 import ReadoutBridge
            native.bridge=ReadoutBridge(nn.Linear(64*heads,64*heads) if backbone=='bge' else None)
        model=Fidelity(native.head,native.last,native.bridge,arm,23)
        g=torch.Generator().manual_seed(37)
        hidden=torch.randn(3,7,64*heads,generator=g)
        return model,dict(hidden=hidden,mask=None,frozen_sequence_key=hashlib.sha256(b'chunk-a').hexdigest())

    def assert_states(self,a,b):
        self.assertEqual(set(a),set(b))
        for k in a:
            if isinstance(a[k],torch.Tensor):self.assertTrue(torch.equal(a[k],b[k]),str(k))
            elif isinstance(a[k],dict):self.assert_states(a[k],b[k])
            elif isinstance(a[k],list):
                self.assertEqual(a[k],b[k])
            else:self.assertEqual(a[k],b[k])

    def test_all_active_arms_native_backbones_masks_five_adam_updates_bit_equal(self):
        for backbone in ('dpr','bge'):
            for arm in ('MLP','GIVENS','PRODUCT','QUANTUM'):
                with self.subTest(backbone=backbone,arm=arm):
                    a,row=self.make(backbone,arm,heads=2);b=copy.deepcopy(a)
                    parameters_a=[p for p in a.parameters() if p.requires_grad]
                    parameters_b=[p for p in b.parameters() if p.requires_grad]
                    oa=torch.optim.AdamW(parameters_a,lr=.001,weight_decay=0)
                    ob=torch.optim.AdamW(parameters_b,lr=.001,weight_decay=0)
                    for step in range(5):
                        # Keep cache identity tied to the exact original mask.
                        mask=torch.tensor([[[[True]*5+[False]*2]]]*3) if step%2 else None
                        current=dict(row,mask=mask,frozen_sequence_key=row['frozen_sequence_key']+str(step%2))
                        oa.zero_grad();ob.zero_grad();sa=original_forward(a,current);sb=b(current)
                        self.assertTrue(torch.equal(sa,sb))
                        self.assertTrue(torch.equal(torch.argsort(sa),torch.argsort(sb)))
                        loss_a=F.softplus(-(sa[0]-sa[2]));loss_b=F.softplus(-(sb[0]-sb[2]))
                        loss_a.backward();loss_b.backward()
                        for p,q in zip(parameters_a,parameters_b):self.assertTrue(torch.equal(p.grad,q.grad))
                        torch.nn.utils.clip_grad_norm_(parameters_a,1.);torch.nn.utils.clip_grad_norm_(parameters_b,1.)
                        oa.step();ob.step();self.assert_states(a.state_dict(),b.state_dict())
                        self.assert_states(oa.state_dict(),ob.state_dict())
                        if step==1:
                            # Reconstruct with a cold memo and restore AdamW,
                            # matching an old checkpoint continued by new code.
                            state=copy.deepcopy(b.state_dict());optimizer=copy.deepcopy(ob.state_dict())
                            rng=torch.get_rng_state()
                            b,_=self.make(backbone,arm,heads=2);b.load_state_dict(state)
                            torch.set_rng_state(rng)
                            parameters_b=[p for p in b.parameters() if p.requires_grad]
                            ob=torch.optim.AdamW(parameters_b,lr=.001,weight_decay=0);ob.load_state_dict(optimizer)
                    restored,_=self.make(backbone,arm,heads=2);restored.load_state_dict(b.state_dict())
                    with torch.no_grad():self.assertTrue(torch.equal(b(current),restored(current)))

    def test_projection_work_halved_and_references_hit_without_rng_or_state_change(self):
        model,row=self.make();weights=(model.layer.attention.self.query.weight,model.layer.attention.self.key.weight)
        linear=F.linear;counts=[]
        def counted(x,w,bias=None):
            if any(w is v for v in weights):counts.append(w)
            return linear(x,w,bias)
        with patch('torch.nn.functional.linear',side_effect=counted):original_forward(model,row)
        self.assertEqual(len(counts),4);counts.clear()
        rng=torch.get_rng_state().clone();keys=list(model.state_dict())
        with patch('torch.nn.functional.linear',side_effect=counted),\
             patch.object(model.effect,'kernel',wraps=model.effect.kernel) as kernel:
            a=model(row);b=model(row)
            self.assertEqual(kernel.call_count,3) # one frozen + two learned
        self.assertEqual(len(counts),4) # two projections per forward, not four
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(keys,list(model.state_dict()))
        self.assertFalse('forward' in model.layer.attention.self.query.__dict__)

    def test_reference_invalidation_eviction_dtype_restore_and_unbound_fallback(self):
        model,row=self.make();model.REFERENCE_CACHE_BYTES=100
        with patch.object(model.effect,'kernel',wraps=model.effect.kernel) as kernel:
            model(row);model(row);self.assertEqual(kernel.call_count,3)
            with torch.no_grad():model.effect.reference_projection.add_(.01)
            model(row);self.assertEqual(kernel.call_count,5)
            with torch.no_grad():model.layer.attention.self.query.weight.add_(.01)
            model(row);self.assertEqual(kernel.call_count,7)
            other=dict(row,frozen_sequence_key='second');model(other)
            self.assertLessEqual(model._reference_bytes,100);self.assertEqual(len(model._references),1)
            model(row);self.assertEqual(kernel.call_count,11)
            unbound=dict(row);unbound.pop('frozen_sequence_key');model(unbound);model(unbound)
            self.assertEqual(kernel.call_count,15)
        model.load_state_dict(model.state_dict());model(row)
        model.double();self.assertEqual(model._reference_bytes,0)
        converted=dict(row,hidden=row['hidden'].double())
        self.assertTrue(torch.equal(model(converted),original_forward(model,converted)))

    def test_projection_cleanup_after_exception_and_instrumented_fallback(self):
        model,row=self.make();att=model.layer.attention.self
        with patch.object(model.layer,'forward',side_effect=RuntimeError('native failure')):
            with self.assertRaisesRegex(RuntimeError,'native failure'):model(row)
        self.assertNotIn('forward',att.query.__dict__);self.assertNotIn('forward',att.key.__dict__)
        calls=[];hook=att.query.register_forward_hook(lambda *args:calls.append(1))
        try:model(row);self.assertEqual(len(calls),2)
        finally:hook.remove()
        # A different tensor must never silently reuse the cached projection.
        with _frozen_projections(row['hidden'],[(att.query,torch.zeros(3,7,64))]):
            with self.assertRaisesRegex(RuntimeError,'identical native hidden'):att.query(row['hidden'].clone())
        self.assertNotIn('forward',att.query.__dict__)


if __name__=='__main__':unittest.main()
