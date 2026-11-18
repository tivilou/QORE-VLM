"""Synthetic CPU tests: diagnostic batching and sequence-free FIDELITY HEAD."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.top8_runtime_io_v004 import step_diagnostics,finite_parameters,head_sequence_rows


def legacy_diagnostics(losses,counts,parameters,denominator,fusion=False,empty=0):
    total=0.;pairs=0;without=empty
    for loss,count in zip(losses,counts):
        total+=float(loss.detach())/denominator;pairs+=int(count);without+=int(count==0)
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    active=[float(p.grad.abs().max()) for p in parameters if p.grad is not None]
    assert (active or fusion) and (pairs==0 or max(active)>0)
    return total,pairs,without


def cpu_row(features,index=None):return {k:v if index is None else v[index] for k,v in features.items()}


class RuntimeIOTests(unittest.TestCase):
    def test_metrics_gradients_five_adam_steps_and_optimizer_state_equal(self):
        g=torch.Generator().manual_seed(81)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(27);a=nn.Sequential(nn.Linear(8,4),nn.Tanh(),nn.Linear(4,1))
        b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=0)
        ob=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=0)
        xs=[torch.randn(n,8,generator=g) for n in (7,0,3,5,2,8,4,1)]
        ys=[torch.arange(len(x))%2 for x in xs]
        for _ in range(5):
            reports=[]
            for model,opt,new in ((a,oa,False),(b,ob,True)):
                opt.zero_grad(set_to_none=True);losses=[];counts=[]
                for x,y in zip(xs,ys):
                    s=model(x).squeeze(-1);mask=y[:,None]>y[None,:];count=mask.sum()
                    loss=(F.softplus(-(s[:,None]-s[None,:]))*mask).sum()/count.clamp_min(1)
                    self.assertTrue(bool(torch.isfinite(loss)))
                    (loss/8).backward();losses.append(loss.detach());counts.append(count)
                parameters=list(model.parameters())
                report=(step_diagnostics if new else legacy_diagnostics)(losses,counts,parameters,8)
                reports.append(report)
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step()
                self.assertTrue(finite_parameters(parameters))
            self.assertEqual(reports[0],reports[1])
            for p,q in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(p.grad,q.grad));self.assertTrue(torch.equal(p,q))
            for i in oa.state_dict()['state']:
                for k,v in oa.state_dict()['state'][i].items():self.assertTrue(torch.equal(v,ob.state_dict()['state'][i][k]))

    def test_gradient_and_parameter_failure_gates_preserved(self):
        p=nn.Parameter(torch.ones(3));loss=[torch.tensor(1.)];count=[torch.tensor(1)]
        p.grad=torch.tensor([float('nan'),1.,0.])
        with self.assertRaisesRegex(RuntimeError,'finite data gradients'):step_diagnostics(loss,count,[p],8)
        p.grad=torch.zeros(3)
        with self.assertRaisesRegex(RuntimeError,'nonzero actual'):step_diagnostics(loss,count,[p],8)
        self.assertEqual(step_diagnostics(loss,[torch.tensor(0)],[p],8),(0.125,0,1))
        p.grad=None
        with self.assertRaisesRegex(RuntimeError,'nonzero actual'):step_diagnostics(loss,count,[p],8)
        self.assertEqual(step_diagnostics([],[],[p],8,fusion=True,empty=8),(0.,0,8))
        self.assertEqual(step_diagnostics(loss,[torch.tensor(0)],[p],8,fusion=True,empty=2),(0.125,0,3))
        with torch.no_grad():p[0]=float('inf')
        self.assertFalse(finite_parameters([p]))

    def test_host_accumulation_exact_with_cancellation_and_no_rng_change(self):
        p=nn.Parameter(torch.ones(2));p.grad=torch.ones(2)
        losses=[torch.tensor(x,dtype=torch.float64) for x in (1e16,1.,-1e16,1e-10)]
        counts=[torch.tensor(x) for x in (0,600,1,0)]
        state=torch.get_rng_state().clone()
        self.assertEqual(step_diagnostics(losses,counts,[p],8),legacy_diagnostics(losses,counts,[p],8))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_fidelity_head_keeps_sparse_original_chunk_shapes_scores_and_gradients(self):
        from applications.rag.tests.test_top8_cache_numerics import NativeFixture
        from applications.rag.top8_methods_v001 import Fidelity
        from applications.rag.top8_portable_v001 import scores_for
        g=torch.Generator().manual_seed(11)
        for kind in ('dpr','bge'):
            native=NativeFixture(kind,Path('.'))
            with torch.no_grad():features={'readout':native.bridge(torch.randn(11,64,generator=g))}
            keep=[0,2,3,7,8,9,14,17,18,20,49];sizes=[3,1,7]
            record=dict(retained_indices=keep,native_scores=[float(i)/7 for i in range(50)],
                        sequences=[dict(positions=keep[:3]),dict(positions=keep[3:4]),dict(positions=keep[4:])])
            a=Fidelity(native.head,native.last,native.bridge,'HEAD',17);b=copy.deepcopy(a)
            expected=[];offset=0
            for n in sizes:expected.append(a({'readout':features['readout'][offset:offset+n]}));offset+=n
            expected=torch.cat(expected);calls=[]
            hook=b.head.register_forward_pre_hook(lambda module,args:calls.append(len(args[0])))
            try:
                with patch('applications.rag.top8_portable_v001.gpu_row',side_effect=cpu_row),\
                     patch('applications.rag.top8_portable_v001.sequence_rows',side_effect=AssertionError('HEAD opened full sequences')):
                    actual=scores_for(b,'FIDELITY',record,features,Path('.'),lambda:None)
                    rows=list(head_sequence_rows(record,features,lambda:None))
                self.assertEqual(calls,sizes);self.assertTrue(torch.equal(actual,expected))
                for (row,anchor),entry in zip(rows,record['sequences']):
                    self.assertTrue(torch.equal(anchor,torch.tensor([record['native_scores'][i] for i in entry['positions']])))
                loss_a=F.softplus(-(expected[0]-expected[-1]));loss_b=F.softplus(-(actual[0]-actual[-1]))
                loss_a.backward();loss_b.backward();self.assertTrue(torch.equal(a.head.weight.grad,b.head.weight.grad))
                self.assertTrue(torch.equal(torch.argsort(expected),torch.argsort(actual)))
            finally:hook.remove()

    def test_head_chunk_alignment_and_full_exhaustion_rejected(self):
        with patch('applications.rag.top8_portable_v001.gpu_row',side_effect=cpu_row):
            r=dict(retained_indices=[0,2],native_scores=[0.]*50,sequences=[dict(positions=[0])])
            with self.assertRaisesRegex(RuntimeError,'exhausted'):list(head_sequence_rows(r,{'readout':torch.ones(2,4)},lambda:None))
            r['sequences']=[dict(positions=[0,1])]
            with self.assertRaisesRegex(RuntimeError,'alignment'):list(head_sequence_rows(r,{'readout':torch.ones(2,4)},lambda:None))

    def test_active_fidelity_still_uses_original_sequence_reader(self):
        from applications.rag.top8_portable_v001 import scores_for
        class Active:
            kind='MLP'
            def __call__(self,row):return row['readout'].sum(-1)
        with patch('applications.rag.top8_portable_v001.sequence_rows',return_value=iter([({'readout':torch.ones(2,4)},None)])) as original,\
             patch('applications.rag.top8_runtime_io_v004.head_sequence_rows',side_effect=AssertionError('active arm bypassed sequences')):
            self.assertTrue(torch.equal(scores_for(Active(),'FIDELITY',{}, {},'.',lambda:None),torch.tensor([4.,4.])))
            original.assert_called_once()

    def test_v2_continuation_preserves_v1_and_accepts_only_exact_ancestors(self):
        from applications.rag.top8_feature_io_v003 import resume_binding
        directory=Path(__file__).resolve().parents[3]/'configs/experiments/triviaqa_smallset_20261112'
        v1=directory/'feature-io-compatibility.v001.json';v2=directory/'runtime-speed-compatibility.v002.json'
        one=json.loads(v1.read_bytes());two=json.loads(v2.read_bytes())
        original=dict(sources=one['before_sources'],seeds=[7],epochs=4,caches=[{'sha':'exact'}])
        first=dict(original,sources=one['after_sources']);latest=dict(original,sources=two['after_sources'])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            selected,p=resume_binding(original,first,root,v1)
            prior=p.read_bytes()
            selected,new=resume_binding(original,latest,root,v2)
            self.assertEqual(selected,original);self.assertEqual(p.read_bytes(),prior)
            self.assertEqual(new.name,'runtime-speed-continuation.v2.json')
            self.assertEqual(json.loads(new.read_bytes())['previous_continuation']['name'],p.name)
            self.assertEqual(resume_binding(original,latest,root,v2),(original,new))
            wrong=dict(latest,seeds=[8])
            with self.assertRaisesRegex(RuntimeError,'cannot change'):resume_binding(original,wrong,root,v2)
            broken=json.loads(prior);broken['actual_runtime_binding']['epochs']=1;p.write_text(json.dumps(broken))
            with self.assertRaisesRegex(RuntimeError,'cannot change'):resume_binding(original,latest,root,v2)
        with tempfile.TemporaryDirectory() as d:
            selected,new=resume_binding(first,latest,Path(d),v2)
            self.assertEqual(selected,first);self.assertNotIn('previous_continuation',json.loads(new.read_bytes()))


if __name__=='__main__':unittest.main()
