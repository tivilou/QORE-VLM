"""Synthetic CPU evidence; does not certify native CUDA speed or async copies."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from applications.rag.top8_fidelity_runtime_v007 import (
    CircuitKernel,DeviceLRU,SequenceDeviceCache,batch_rows,memory_plan)
from applications.rag.tests import test_top8_fidelity_speed as fixture


class ThroughputTests(unittest.TestCase):
    def test_gate_matrix_matches_original_all_arms_and_gradients(self):
        from applications.rag.qfidelity_v002 import Fidelity
        g=torch.Generator().manual_seed(481)
        for arm in ('MLP','GIVENS','PRODUCT','QUANTUM'):
            a=Fidelity(arm,19,'cpu');b=copy.deepcopy(a);fast=CircuitKernel(arm,'cpu')
            q=torch.randn(3,2,64,generator=g);k=torch.randn(3,2,9,64,generator=g)
            x=a.kernel(q,k);y=fast.kernel(b,q,k)
            torch.testing.assert_close(x,y,atol=3e-6,rtol=2e-5)
            (x.square().mean()).backward();(y.square().mean()).backward()
            for p,r in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,r.grad,atol=1e-5,rtol=.003)
            self.assertTrue(all(bool(f) for f in fast.take_flags()));self.assertEqual(fast.take_flags(),[])
            xr=a.kernel(q,k,True);yr=fast.kernel(b,q,k,True)
            torch.testing.assert_close(xr,yr,atol=3e-6,rtol=2e-5)
            self.assertTrue(all(bool(f) for f in fast.take_flags()))

    def test_full_and_cls_dpr_bge_active_scores_gradients_adam_replay(self):
        for backbone in ('dpr','bge'):
            for cls in (False,True):
                for arm in ('MLP','GIVENS','PRODUCT','QUANTUM'):
                    for mode in ('none','bool','float'):
                        with self.subTest(backbone=backbone,cls=cls,arm=arm,mask=mode):
                            a,row=fixture.FidelitySpeedTests().make(backbone,arm,2);a.cls_only=cls
                            with torch.no_grad():
                                for name,p in a.effect.named_parameters():p.add_(.013)
                            b=copy.deepcopy(a);b.enable_throughput(memory_plan('cpu'))
                            if mode!='none':
                                mask=torch.ones(3,1,1,7,dtype=torch.bool);mask[...,-2:]=False
                                row=dict(row,mask=mask if mode=='bool' else torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,torch.finfo(torch.float32).min))
                            chunks=[]
                            for j,(lo,hi) in enumerate(((0,2),(2,3))):
                                r={k:(v[lo:hi] if isinstance(v,torch.Tensor) else v) for k,v in row.items()}
                                r['frozen_sequence_key']=str(j);chunks.append((r,torch.zeros(hi-lo)))
                            pa=[p for p in a.parameters() if p.requires_grad];pb=[p for p in b.parameters() if p.requires_grad]
                            oa=torch.optim.AdamW(pa,lr=.001,weight_decay=0);ob=torch.optim.AdamW(pb,lr=.001,weight_decay=0)
                            for _ in range(2):
                                oa.zero_grad();ob.zero_grad()
                                sa=torch.cat([a(r) for r,_ in chunks]);sb=torch.cat([b(r) for r,_ in batch_rows(iter(chunks))])
                                torch.testing.assert_close(sa,sb,atol=4e-5,rtol=4e-5)
                                F.softplus(-(sa[0]-sa[2])).backward();F.softplus(-(sb[0]-sb[2])).backward()
                                for p,q in zip(pa,pb):torch.testing.assert_close(p.grad,q.grad,atol=4e-5,rtol=.01)
                                self.assertTrue(all(bool(f) for f in b.take_kernel_flags()));oa.step();ob.step()
                            restored=copy.deepcopy(b);restored.clear_references()
                            with torch.no_grad():
                                sb=torch.cat([b(r) for r,_ in batch_rows(iter(chunks))]);sr=torch.cat([restored(r) for r,_ in batch_rows(iter(chunks))])
                            self.assertTrue(torch.equal(sb,sr))

    def test_native_shapes_order_broadcast_mask_and_full_exhaustion(self):
        visited=[]
        def rows():
            for i,(b,l) in enumerate(((2,7),(3,7),(1,5),(2,5),(2,7))):
                visited.append(i);yield dict(hidden=torch.full((b,l,4),float(i)),mask=torch.ones(1,1,1,l,dtype=torch.bool),readout=torch.full((b,4),float(i)),frozen_sequence_key=str(i)),torch.full((b,),float(i))
        result=list(batch_rows(rows(),5));self.assertEqual(visited,list(range(5)))
        self.assertEqual([r['hidden'].shape[:2] for r,_ in result],[(5,7),(3,5),(2,7)])
        self.assertTrue(torch.equal(torch.cat([n for _,n in result]),torch.tensor([0.,0.,1.,1.,1.,2.,3.,3.,4.,4.])))
        for row,_ in result:self.assertIn(row['mask'].shape[0],(1,len(row['hidden'])))

    def test_device_cache_no_grad_bound_and_inplace_invalidation(self):
        cache=DeviceLRU(32);cache.put('a',torch.ones(4));cache.put('b',torch.ones(4));self.assertEqual(cache.bytes,32)
        self.assertIsNotNone(cache.get('a'));cache.put('c',torch.ones(4));self.assertIsNone(cache.get('b'))
        x=cache.get('a');x.add_(1)
        with self.assertRaisesRegex(RuntimeError,'immutable'):cache.get('a')
        with self.assertRaisesRegex(RuntimeError,'graphs'):cache.put('x',torch.ones(3,requires_grad=True))
        cache.clear();self.assertEqual(cache.bytes,0)

    def test_sequence_cache_bound_no_speculative_regeneration_and_identical_values(self):
        class Reader:
            memory={};disk=type('Disk',(),{'entries':{}})()
            def __init__(self):self.calls=0
            def read(self,entry,role):
                self.calls+=1;return dict(hidden=torch.arange(24.).reshape(2,3,4),mask=None,native_scores=torch.zeros(2))
        reader=Reader();cache=SequenceDeviceCache('cpu',1024)
        self.assertIsNone(cache.read(reader,{'k':1},'training',prefetch=True));self.assertEqual(reader.calls,0)
        a=cache.read(reader,{'k':1},'training');b=cache.read(reader,{'k':1},'training')
        self.assertEqual(reader.calls,1);self.assertTrue(torch.equal(a['hidden'],b['hidden']));self.assertEqual(cache.cache.hits,1)
        cache.close();self.assertEqual(cache.cache.bytes,0)

    def test_flags_detect_nonfinite_before_optimizer_without_per_kernel_host_bool(self):
        from applications.rag.qfidelity_v002 import Fidelity
        from applications.rag.top8_runtime_io_v004 import step_diagnostics
        e=Fidelity('QUANTUM',3,'cpu');fast=CircuitKernel('QUANTUM','cpu')
        with patch.object(torch.Tensor,'__bool__',side_effect=AssertionError('per-kernel host sync')):
            fast.kernel(e,torch.zeros(1,1,64),torch.zeros(1,1,3,64))
        flags=fast.take_flags();self.assertTrue(bool(flags[0]))
        fast.kernel(e,torch.full((1,1,64),float('nan')),torch.zeros(1,1,3,64));flags=fast.take_flags()
        p=torch.nn.Parameter(torch.ones(2));p.grad=torch.ones(2)
        with self.assertRaisesRegex(RuntimeError,'finite data gradients'):
            step_diagnostics([torch.tensor(1.)],[torch.tensor(1)],[p],8,extra_flags=flags)
        self.assertTrue(torch.equal(p.detach(),torch.ones(2)))

    def test_checkpoint_restart_keeps_next_update_and_cold_cache_replay(self):
        model,row=fixture.FidelitySpeedTests().make('dpr','QUANTUM',2)
        model.cls_only=True;model.enable_throughput(memory_plan('cpu'))
        row=dict(row,frozen_sequence_key='restart-source')
        parameters=lambda m:[p for p in m.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(parameters(model),lr=.001,weight_decay=0)
        def update(m,o):
            o.zero_grad();s=m(row);F.softplus(-(s[0]-s[2])).backward()
            self.assertTrue(all(bool(v) for v in m.take_kernel_flags()));o.step()
        update(model,opt)
        restored=copy.deepcopy(model);restored.enable_throughput(memory_plan('cpu'))
        other=torch.optim.AdamW(parameters(restored),lr=.001,weight_decay=0)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        update(model,opt);update(restored,other)
        for a,b in zip(parameters(model),parameters(restored)):self.assertTrue(torch.equal(a,b))
        with torch.no_grad():self.assertTrue(torch.equal(model(row),restored(row)))

    def test_v7_resumes_v6_without_changing_old_receipt_or_science(self):
        import json
        from applications.rag.top8_feature_io_v003 import resume_binding
        directory=Path(__file__).resolve().parents[3]/'configs/experiments/triviaqa_smallset_20261112'
        six=json.loads((directory/'primary54-compatibility.v006.json').read_bytes())
        seven=json.loads((directory/'fidelity-throughput-compatibility.v007.json').read_bytes())
        original=dict(sources=six['accepted_previous_sources'][0],seeds=[7],epochs=4)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,old=resume_binding(original,dict(original,sources=six['after_sources']),root)
            raw=old.read_bytes();current=dict(original,sources=seven['after_sources'])
            chosen,new=resume_binding(original,current,root)
            self.assertEqual(chosen,original);self.assertEqual(old.read_bytes(),raw)
            self.assertEqual(new.name,'fidelity-throughput-continuation.v7.json')
            self.assertIn(old.name,[r['name'] for r in json.loads(new.read_bytes())['previous_continuations']])
            with self.assertRaises(RuntimeError):resume_binding(original,dict(current,epochs=3),root)

    def test_throughput_telemetry_renders_measured_cache_and_wait(self):
        from scripts.smallset_progress import Progress,render,Dashboard
        p=Progress('synthetic','log');p.event(dict(qduet_progress=1,event='throughput',time=1,throughput=dict(
            candidates=32,device_input_bytes=2*1024**3,device_input_limit=4*1024**3,
            device_input_hits=19,projection_hits=7,reader_wait_seconds=1.25)))
        text=render(Dashboard(),p.snapshot(now=2),now=2)
        self.assertIn('同形候选块上限 32',text);self.assertIn('2.00 GiB/4.00 GiB',text)
        self.assertIn('QKV命中 7',text);self.assertIn('等待 1.2s',text)


if __name__=='__main__':unittest.main()
