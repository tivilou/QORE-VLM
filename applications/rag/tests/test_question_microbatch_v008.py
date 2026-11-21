"""Synthetic CPU comparisons of actual production losses and whole updates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from applications.rag.top8_microbatch_v008 import Microbatch,losses_for_questions,question_losses,merge_rows
from applications.rag.top8_methods_v001 import build,objective
from applications.rag.tests.test_top8_cache_numerics import NativeFixture
from applications.rag.tests import test_top8_fidelity_speed as fixture


class QuestionMicrobatchTests(unittest.TestCase):
    def records(self):
        return [dict(id=str(i),retained_indices=list(range(n)),targets=[0]*n if i==2 else [j%2 for j in range(n)])
                for i,n in enumerate((5,3,4,2,5,3,4,2))]
    def compare(self,a,b):
        for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(n,m)
            # AdamW eps=1e-8 amplifies float32 noise on dormant/gauge circuit
            # directions. Check them explicitly within the two-update absolute
            # step envelope, not with a false parameter-bit-equivalence claim.
            weak=(p.grad.abs()<1e-6)&(q.grad.abs()<1e-6) if p.grad is not None else torch.zeros_like(p,dtype=torch.bool)
            torch.testing.assert_close(p[~weak],q[~weak],atol=5e-5,rtol=.002)
            if weak.any():self.assertLessEqual(float((p[weak]-q[weak]).abs().max()),.004001)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-5,rtol=.005)
    def test_all_feature_methods_and_two_backbones_two_adam_updates(self):
        # Test real production entry, nested features, uneven lengths and a
        # zero-pair question; never flatten pairwise targets across questions.
        for backbone in ('dpr','bge'):
            for idea in ('MIXED_DEPTH','VALUE_GEOMETRY','LOCAL_REUPLOAD','NATIVE_BORN','FEATURE_GATE','LOCAL_FFN','ORDINAL'):
                with self.subTest(backbone=backbone,idea=idea):
                    torch.manual_seed(93);native=NativeFixture(backbone,Path('.'));d=native.head.in_features
                    a=build(idea,'QUANTUM',17,native.head,native.last,native.bridge,(torch.zeros(d),torch.ones(d)),2)
                    # Continuing active parameters, rather than only the null
                    # initialization with structurally dormant circuit angles.
                    with torch.no_grad():
                        for p in a.parameters():
                            if p.requires_grad:p.add_(.017)
                    b=copy.deepcopy(a);records=self.records();features={}
                    for r in records:
                        n=len(r['targets']);features[r['id']]=dict(readout=torch.randn(n,d),depths=torch.randn(n,3,d),
                            context=torch.randn(n,d),residual=torch.randn(n,d),final=torch.randn(n,d),
                            z=torch.randn(n,native.last.intermediate.dense.out_features),ffn_residual=torch.randn(n,d),
                            mixed={'QUANTUM':torch.randn(n,2)})
                    pa=[p for p in a.parameters() if p.requires_grad];pb=[p for p in b.parameters() if p.requires_grad]
                    oa=torch.optim.AdamW(pa,lr=.001,weight_decay=0);ob=torch.optim.AdamW(pb,lr=.001,weight_decay=0)
                    for _ in range(2):
                        oa.zero_grad()
                        for r in records:
                            loss,_=objective(a,idea,features[r['id']],torch.tensor(r['targets']))
                            (loss/8).backward()
                        control=Microbatch()
                        ls,cs=control.backward(records,lambda part:losses_for_questions(b,idea,part,features,'.',lambda:None,'cpu'),pb,ob)
                        self.assertEqual(len(ls),8);self.assertEqual(int(cs[2]),0)
                        self.compare(a,b);torch.nn.utils.clip_grad_norm_(pa,1.);torch.nn.utils.clip_grad_norm_(pb,1.)
                        oa.step();ob.step();self.compare(a,b)
                        for r in records:
                            with torch.no_grad():torch.testing.assert_close(a(features[r['id']]),b(features[r['id']]),atol=5e-5,rtol=.001)
                        self.assertEqual(control.state()['accumulation_groups'],1)
                    for x,y in zip(oa.state.values(),ob.state.values()):
                        for k in ('exp_avg','exp_avg_sq'):torch.testing.assert_close(x[k],y[k],atol=5e-6,rtol=.01)

    def test_fidelity_cross_question_native_masks_chunks_and_gradients(self):
        from applications.rag.top8_fidelity_runtime_v007 import memory_plan,batch_rows
        for backbone in ('dpr','bge'):
            for arm in ('MLP','GIVENS','PRODUCT','QUANTUM'):
                for mask_kind in ('none','bool','float'):
                    with self.subTest(backbone=backbone,arm=arm,mask=mask_kind):
                        a,row=fixture.FidelitySpeedTests().make(backbone,arm);a.cls_only=True;a.enable_throughput(memory_plan('cpu'))
                        with torch.no_grad():
                            for p in a.effect.parameters():p.add_(.013)
                        b=copy.deepcopy(a);records=self.records();chunks={}
                        for i,r in enumerate(records):
                            n=len(r['targets']);h=torch.randn(n,row['hidden'].shape[1],row['hidden'].shape[2],generator=torch.Generator().manual_seed(100+i))
                            m=None
                            if mask_kind!='none':
                                m=torch.ones(n,1,1,h.shape[1],dtype=torch.bool);m[...,-1]=False
                                if mask_kind=='float':m=torch.zeros_like(m,dtype=h.dtype).masked_fill(~m,torch.finfo(h.dtype).min)
                            chunks[r['id']]=[(dict(hidden=h[:2],mask=None if m is None else m[:2],frozen_sequence_key=r['id']+'a'),torch.zeros(min(n,2))),
                                (dict(hidden=h[2:],mask=None if m is None else m[2:],frozen_sequence_key=r['id']+'b'),torch.zeros(max(0,n-2)))] if n>2 else [(dict(hidden=h,mask=m,frozen_sequence_key=r['id']),torch.zeros(n))]
                        pa=[p for p in a.parameters() if p.requires_grad];pb=[p for p in b.parameters() if p.requires_grad]
                        oa=torch.optim.AdamW(pa,lr=.001,weight_decay=0);ob=torch.optim.AdamW(pb,lr=.001,weight_decay=0)
                        for _ in range(2):
                            oa.zero_grad()
                            for r in records:
                                s=torch.cat([a(x) for x,_ in batch_rows(iter(chunks[r['id']]),32)])
                                ls,_=question_losses([s],[torch.tensor(r['targets'])]);(ls[0]/8).backward()
                            with patch('applications.rag.top8_portable_v001.sequence_rows',side_effect=lambda root,r,f,c:iter(chunks[r['id']])):
                                Microbatch().backward(records,lambda part:losses_for_questions(b,'FIDELITY',part,{r['id']:{} for r in records},'.',lambda:None,'cpu'),pb,ob)
                            self.compare(a,b);a.take_kernel_flags();b.take_kernel_flags()
                            torch.nn.utils.clip_grad_norm_(pa,1.);torch.nn.utils.clip_grad_norm_(pb,1.);oa.step();ob.step();self.compare(a,b)

    def test_forward_backward_oom_discards_partial_gradients_and_rng(self):
        torch.manual_seed(56);a=nn.Linear(3,1);b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
        records=list(range(8));x=torch.randn(8,3);rng=torch.get_rng_state();discarded=[];calls=[]
        def losses(model,part):
            value=model(x[part]).flatten()*torch.rand(len(part))
            return [v.square() for v in value],[torch.tensor(1) for v in value]
        # The successful fallback route is tier2. Force OOM both before any
        # backward and after partial gradients from the first four questions.
        controller=Microbatch()
        def injected(part):
            calls.append(list(part))
            if len(part)==8 or (len(part)==4 and part[0]==4):
                torch.rand(7);raise torch.cuda.OutOfMemoryError('injected allocation failure')
            return losses(b,part)
        controller.backward(records,injected,b.parameters(),ob,lambda:discarded.append(1))
        after=torch.get_rng_state();torch.set_rng_state(rng);reference=Microbatch();reference.size=2
        reference.backward(records,lambda p:losses(a,p),a.parameters(),oa)
        self.compare(a,b);self.assertTrue(torch.equal(after,torch.get_rng_state()));oa.step();ob.step();self.compare(a,b)
        self.assertEqual(controller.size,2);self.assertEqual(controller.ooms,2);self.assertEqual(len(discarded),2)
        self.assertEqual(controller.last_groups,4);self.assertEqual(len(oa.state),len(ob.state))

    def test_smallest_oom_stops_other_errors_propagate_no_optimizer_update(self):
        m=nn.Linear(1,1);o=torch.optim.AdamW(m.parameters());initial=copy.deepcopy(m.state_dict())
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            Microbatch().backward([0],lambda _:(_ for _ in ()).throw(torch.cuda.OutOfMemoryError()),m.parameters(),o)
        with self.assertRaisesRegex(ValueError,'identity'):
            Microbatch().backward([0],lambda _:(_ for _ in ()).throw(ValueError('identity')),m.parameters(),o)
        self.assertFalse(o.state)
        for k,v in initial.items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_oom_raised_inside_autograd_backward_retries_one_whole_update(self):
        armed=[True]
        class FailOnce(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x):return x.clone()
            @staticmethod
            def backward(ctx,gradient):
                if armed[0]:
                    armed[0]=False;raise torch.cuda.OutOfMemoryError('backward allocation')
                return gradient
        torch.manual_seed(34);a=nn.Linear(3,1);b=copy.deepcopy(a);x=torch.randn(8,3)
        oa=torch.optim.AdamW(a.parameters());ob=torch.optim.AdamW(b.parameters());c=Microbatch()
        def fn(model,part,inject=False):
            value=model(x[part]).flatten()
            if inject:value=FailOnce.apply(value)
            return [v.square() for v in value],[torch.tensor(1)]*len(part)
        with patch.object(ob,'step',wraps=ob.step) as step:
            c.backward(list(range(8)),lambda part:fn(b,part,True),b.parameters(),ob)
            self.assertEqual(step.call_count,0)
            reference=Microbatch();reference.size=4
            reference.backward(list(range(8)),lambda part:fn(a,part),a.parameters(),oa)
            self.compare(a,b);oa.step();ob.step();self.compare(a,b);self.assertEqual(step.call_count,1)
        self.assertEqual(c.size,4);self.assertEqual(c.ooms,1)

    def test_persisted_tier_fresh_optimizer_restart(self):
        torch.manual_seed(33);m=nn.Linear(3,1);o=torch.optim.AdamW(m.parameters(),lr=.001);control=Microbatch();control.size=4
        x=torch.randn(8,3)
        def update(model,opt,c):
            c.backward(list(range(8)),lambda part:([v.square() for v in model(x[part]).flatten()],[torch.tensor(1)]*len(part)),model.parameters(),opt);opt.step()
        update(m,o,control)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'checkpoint.bin';torch.save(dict(parameters=m.state_dict(),optimizer=o.state_dict(),microbatch=control.state()),p)
            saved=torch.load(p,weights_only=True);restored=nn.Linear(3,1);restored.load_state_dict(saved['parameters']);new=torch.optim.AdamW(restored.parameters(),lr=.001);new.load_state_dict(saved['optimizer']);c=Microbatch(saved['microbatch'])
            update(m,o,control);update(restored,new,c)
            for a,b in zip(m.parameters(),restored.parameters()):self.assertTrue(torch.equal(a,b))
            self.assertEqual(c.size,4)

    def test_question_boundaries_nested_fields_and_invalid_merge(self):
        rows=[dict(readout=torch.ones(n,3),mixed={'QUANTUM':torch.zeros(n,2)}) for n in (2,3)]
        self.assertEqual(merge_rows(rows)['mixed']['QUANTUM'].shape,(5,2))
        with self.assertRaisesRegex(RuntimeError,'fields'):merge_rows([rows[0],{'other':torch.zeros(2)}])
        ls,cs=question_losses([torch.tensor([100.,100.],requires_grad=True),torch.tensor([-100.,-100.],requires_grad=True)],
                              [torch.tensor([0,0]),torch.tensor([1,1])])
        self.assertEqual([int(c) for c in cs],[0,0]);self.assertEqual([float(l) for l in ls],[0.,0.])

    def test_microbatch_progress_is_visible_without_fidelity_cache(self):
        from scripts.smallset_progress import render
        from scripts.smallset_progress import Progress
        from scripts.progress_ui import Dashboard
        state=Progress('test','test.log');state.event(dict(qduet_progress=1,event='throughput',time=1,throughput=dict(Microbatch().state(),route='question-microbatch-v8')))
        text=render(Dashboard(),state.snapshot(now=2),2)
        self.assertIn('microbatch',text);self.assertIn('OOM',text)

    def test_v8_preserves_v7_receipt_and_rejects_scientific_changes(self):
        from applications.rag.top8_feature_io_v003 import resume_binding
        directory=Path(__file__).resolve().parents[3]/'configs/experiments/triviaqa_smallset_20261112'
        seven=json.loads((directory/'fidelity-throughput-compatibility.v007.json').read_bytes())
        eight=json.loads((directory/'question-microbatch-compatibility.v008.json').read_bytes())
        original=dict(sources=seven['accepted_previous_sources'][0],seeds=[7],epochs=4,batch=8)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,old=resume_binding(original,dict(original,sources=seven['after_sources']),root)
            raw=old.read_bytes();current=dict(original,sources=eight['after_sources'])
            chosen,new=resume_binding(original,current,root)
            self.assertEqual(chosen,original);self.assertEqual(old.read_bytes(),raw)
            self.assertEqual(new.name,'question-microbatch-continuation.v8.json')
            self.assertIn(old.name,[r['name'] for r in json.loads(new.read_bytes())['previous_continuations']])
            with self.assertRaises(RuntimeError):resume_binding(original,dict(current,batch=16),root)

    def test_real_train_entry_saves_microbatch_and_fresh_resume_replays(self):
        from applications.rag.top8_portable_v001 import train_cell
        torch.manual_seed(55);native=NativeFixture('dpr',Path('.'));records=self.records();features={}
        for r in records:
            r['role']='training';features[r['id']]=dict(readout=torch.randn(len(r['targets']),64))
        r=dict(id='dev',role='development',retained_indices=list(range(50)),targets=None,candidate_ids=list(range(50)))
        records.append(r);features['dev']=dict(readout=torch.randn(50,64))
        class Journal:
            value=None
            def load(self):return copy.deepcopy(self.value)
            def save(self,v):self.value=copy.deepcopy(v)
        journal=Journal();manifest=dict(rows=records,label_classes=2)
        with tempfile.TemporaryDirectory() as d,patch('applications.rag.top8_backbone_v001.load_modules',return_value=(native.last,native.bridge,native.head)),\
             patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self),patch.object(nn.Module,'cuda',lambda self,*a,**k:self):
            out=Path(d)/'first';out.mkdir()
            result=train_cell('NATIVE_BORN','QUANTUM',7,manifest,features,d,out,1,None,lambda:None,journal)
            self.assertEqual(result['steps'],1);self.assertEqual(journal.value['microbatch']['physical_questions'],8)
            self.assertEqual(journal.value['next_start'],8);self.assertEqual(len(journal.value['records']),1)
            other=Path(d)/'resumed';other.mkdir()
            resumed=train_cell('NATIVE_BORN','QUANTUM',7,manifest,features,d,other,1,None,lambda:None,journal)
            self.assertEqual(resumed['steps'],1)
            first=json.loads((out/'predictions-NATIVE_BORN-QUANTUM-s7.json').read_bytes())
            second=json.loads((other/'predictions-NATIVE_BORN-QUANTUM-s7.json').read_bytes())
            self.assertEqual(first,second)


if __name__=='__main__':unittest.main()
