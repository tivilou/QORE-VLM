import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from applications.rag.tests.test_top8_fidelity_speed import FidelitySpeedTests


class CLSNumerics(unittest.TestCase):
    def test_combined_nonfinite_loss_gate_still_precedes_optimizer(self):
        from applications.rag.top8_runtime_io_v004 import step_diagnostics
        p=nn.Parameter(torch.ones(2));p.grad=torch.ones(2)
        for loss in (float('nan'),float('inf')):
            with self.assertRaisesRegex(RuntimeError,'finite training-only objective'):
                step_diagnostics([torch.tensor(loss)],[torch.tensor(1)],[p],8)
            self.assertTrue(torch.equal(p.detach(),torch.ones(2)))

    def test_both_backbones_all_active_arms_masks_outputs_gradients_and_replay(self):
        fixture=FidelitySpeedTests()
        for backbone in ('dpr','bge'):
            for arm in ('MLP','GIVENS','PRODUCT','QUANTUM'):
                for mode in ('none','bool','float'):
                    with self.subTest(backbone=backbone,arm=arm,mask=mode):
                        a,row=fixture.make(backbone,arm,heads=2);b=copy.deepcopy(a);b.cls_only=True
                        if mode!='none':
                            mask=torch.ones(3,1,1,7,dtype=torch.bool);mask[...,-2:]=False
                            row=dict(row,mask=mask if mode=='bool' else torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,torch.finfo(torch.float32).min))
                        pa=[p for p in a.parameters() if p.requires_grad];pb=[p for p in b.parameters() if p.requires_grad]
                        oa=torch.optim.AdamW(pa,lr=.001,weight_decay=0);ob=torch.optim.AdamW(pb,lr=.001,weight_decay=0)
                        for step in range(3):
                            oa.zero_grad();ob.zero_grad();sa=a(row);sb=b(row)
                            torch.testing.assert_close(sa,sb,atol=3e-5,rtol=3e-5)
                            la=torch.nn.functional.softplus(-(sa[0]-sa[2]));lb=torch.nn.functional.softplus(-(sb[0]-sb[2]))
                            la.backward();lb.backward()
                            for p,q in zip(pa,pb):
                                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=.005)
                            oa.step();ob.step()
                        restored=copy.deepcopy(b)
                        self.assertTrue(torch.equal(b(row),restored(row)))
                        # The final FFN consumes CLS only, including padded inputs.
                        seen=[];hook=b.layer.intermediate.register_forward_pre_hook(lambda m,args:seen.append(args[0].shape[1]))
                        try:b(row)
                        finally:hook.remove()
                        self.assertEqual(seen,[1])

    def test_prediction_and_replay_read_each_chunk_once(self):
        from applications.rag.top8_portable_v001 import scores_and_replay
        class Scorer(nn.Module):
            kind='QUANTUM'
            def forward(self,row):return row['x']*2
        rows=[({'x':torch.arange(3.)},None),({'x':torch.arange(2.)},None)]
        with patch('applications.rag.top8_portable_v001.sequence_rows',return_value=iter(rows)) as reader:
            a,b=scores_and_replay(Scorer(),Scorer(),'FIDELITY',{},None,Path('.'),lambda:None)
        self.assertEqual(reader.call_count,1);self.assertTrue(torch.equal(a,b));self.assertEqual(len(a),5)

    def test_prefetch_never_regenerates_or_imports_cuda(self):
        from applications.rag.top8_cache_v002 import FrozenSequences,object_bytes,sequence_signature
        import hashlib,json
        with tempfile.TemporaryDirectory() as d:
            reader=FrozenSequences(Path(d),dict(sequence_cache_bytes=1024**2,backbone='fake'))
            payload=dict(hidden=torch.randn(2,4,8),mask=None,native_scores=torch.zeros(2))
            entry=dict(regenerate=True,signature=sequence_signature(**dict(hidden=payload['hidden'],mask=None,native=payload['native_scores'])),legacy_name='unused')
            key=hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest()
            reader.disk.put(key,object_bytes(payload),'training');reader.prefetch(entry)
            with patch('torch.cuda.init',side_effect=AssertionError('CUDA forbidden')):actual=reader.read(entry,'training')
            self.assertTrue(torch.equal(actual['hidden'],payload['hidden']));self.assertEqual(reader.prefetch_hits,1)
            reader.close()


if __name__=='__main__':unittest.main()
