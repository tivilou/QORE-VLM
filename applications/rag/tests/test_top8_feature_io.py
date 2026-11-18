"""CPU synthetic IO/gradient tests; no pretrained assets or dataset examples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from applications.rag.qvalue_portable_v001 import tensor_file
from applications.rag.top8_cache_v002 import FeatureRows
from applications.rag.top8_feature_io_v003 import (
    FieldBank, project_features, training_moments, field_bank, close_fields,
    required_fields, compatible_binding,
)


def fixture(root):
    g=torch.Generator().manual_seed(81);values={};rows=[]
    for i in range(5):
        row={k:torch.randn(4,8,generator=g) for k in ('readout','context','residual','final','ffn_residual','depth0')}
        row['depths']=torch.randn(4,3,8,generator=g);row['z']=torch.randn(4,32,generator=g)
        if i>=3:row['readout']+=1000
        key='q'+str(i);values[key]=row
        record=tensor_file(root,'question-'+str(i)+'.bin',row)
        rows.append(dict(id=key,role='training' if i<3 else 'development',retained_indices=list(range(4)),feature_file=record))
    manifest=dict(rows=rows,backbone='dpr');(root/'cache.json').write_text(json.dumps(manifest))
    return values,manifest,FeatureRows(root,rows,maximum=0)


class FeatureIOTests(unittest.TestCase):
    def tearDown(self):close_fields()

    def test_requested_fields_exact_and_no_per_step_source_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values,m,source=fixture(root)
            state=torch.get_rng_state().clone()
            view=project_features(source,m,root,('readout',))
            bank=field_bank(source,m,root);self.assertEqual(bank.metrics['source_questions_read'],5)
            for _ in range(8):
                for key in values:
                    actual=view[key];self.assertEqual(set(actual),{'readout'})
                    self.assertTrue(torch.equal(actual['readout'],values[key]['readout']))
            self.assertEqual(bank.metrics['source_questions_read'],5)
            self.assertEqual(bank.metrics['column_payload_bytes'],5*4*8*4)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            modified=view['q0']['readout'];modified.zero_()
            self.assertTrue(torch.equal(view['q0']['readout'],values['q0']['readout']))

    def test_auxiliary_mixed_terms_survive_without_loading_full_shard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values,m,source=fixture(root)
            source.set_aux('q0','mixed',{'QUANTUM':torch.ones(4,2)})
            view=project_features(source,m,root,('depths','mixed'))
            self.assertTrue(torch.equal(view['q0']['depths'],values['q0']['depths']))
            self.assertEqual(view['q0']['mixed']['QUANTUM'].shape,(4,2))

    def test_stats_bit_equal_training_only_and_reused_across_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values,m,source=fixture(root)
            x=torch.cat([values[r['id']]['readout'] for r in m['rows'] if r['role']=='training'])
            mean=x.mean(0);std=x.std(0,unbiased=False);std=torch.where(std>1e-8,std,torch.ones_like(std))
            for _ in range(3):
                a,b=training_moments(source,m,root)
                self.assertTrue(torch.equal(a,mean));self.assertTrue(torch.equal(b,std))
            bank=field_bank(source,m,root)
            self.assertEqual(bank.metrics['statistics_computed'],1)
            close_fields();again=FeatureRows(root,m['rows'],maximum=0)
            a,b=training_moments(again,m,root);bank=field_bank(again,m,root)
            self.assertEqual(bank.metrics['statistics_computed'],0)
            self.assertEqual(bank.metrics['source_questions_read'],0)
            self.assertTrue(torch.equal(a,mean));self.assertTrue(torch.equal(b,std))

    def test_partial_column_is_preserved_and_never_treated_as_committed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,m,source=fixture(root);count=0
            def interrupt():
                nonlocal count
                count+=1
                if count==3:raise InterruptedError('synthetic interruption')
            bank=FieldBank(source,m,root,interrupt)
            with self.assertRaises(InterruptedError):bank.ensure(('readout',))
            part=next(bank.root.glob('*.part'));raw=part.read_bytes()
            self.assertFalse((bank.root/'readout.json').exists())
            bank.checkpoint=lambda:None;bank.ensure(('readout',))
            self.assertEqual(part.read_bytes(),raw)
            self.assertEqual(json.loads((bank.root/'readout.json').read_bytes())['file']['name'],'readout-g0001.f32')

    def test_tampered_column_and_training_binding_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,m,source=fixture(root);bank=FieldBank(source,m,root,lambda:None)
            bank.moments();meta=bank.root/'training-statistics.json'
            row=json.loads(meta.read_bytes());row['binding']['training_ids'].reverse();meta.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError,'normalization binding'):FieldBank(source,m,root,lambda:None).moments()
            receipt=json.loads((bank.root/'readout.json').read_bytes())
            p=bank.root/receipt['file']['name'];p.write_bytes(b'X'*p.stat().st_size)
            with self.assertRaisesRegex(RuntimeError,'size/SHA'):FieldBank(source,m,root,lambda:None).ensure(('readout',))

    def test_interrupted_json_receipts_preserved_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,m,source=fixture(root);bank=FieldBank(source,m,root,lambda:None)
            bank.root.mkdir(parents=True)
            for name in ('readout.json','training-statistics.json'):
                (bank.root/(name+'.part')).write_bytes(b'uncommitted receipt')
            bank.moments()
            for name in ('readout.json','training-statistics.json'):
                self.assertTrue((bank.root/name).is_file())
                self.assertEqual((bank.root/(name+'.interrupted-0')).read_bytes(),b'uncommitted receipt')

    def test_compatibility_allows_only_exact_sources_and_identical_experiment(self):
        old=dict(sources={'portable.py':'old'},seeds=[7],epochs=4,caches=[{'sha':'locked'}])
        new=dict(old,sources={'portable.py':'new','io.py':'new'})
        spec=dict(before_sources=old['sources'],after_sources=new['sources'])
        self.assertIs(compatible_binding(old,new,spec),old)
        for changed in (dict(new,epochs=1),dict(new,seeds=[8]),dict(new,caches=[])):
            with self.assertRaisesRegex(RuntimeError,'cannot change'):compatible_binding(old,changed,spec)
        with self.assertRaisesRegex(RuntimeError,'exact reviewed'):compatible_binding(old,dict(new,sources={}),spec)

    def test_every_nonsequence_arm_scores_losses_gradients_and_adam_equal(self):
        from applications.rag.tests.test_top8_cache_numerics import NativeFixture
        from applications.rag.top8_cache_v002 import capture_chunk
        from applications.rag.top8_methods_v001 import ARMS,build,objective
        g=torch.Generator().manual_seed(15)
        cases=0
        for kind in ('dpr','bge'):
            native=NativeFixture(kind,Path('.'))
            encoded=native.tokenize('synthetic question',['paragraph'+str(i) for i in range(4)])
            h,_,_,_=capture_chunk(native,encoded)
            with torch.no_grad():
                features={k:h[k].cpu() for k in ('context','ffn_residual','z','final')}
                features['readout']=native.bridge(h['final']);features['residual']=h['prefix'][:,0]
                features['depths']=torch.stack([native.bridge(h['depth0']),native.bridge(h['prefix'][:,0]),native.bridge(h['final'])],1)
                features['mixed']={a:torch.randn(4,2,generator=g) for a in ARMS['MIXED_DEPTH'][2:]}
            for idea,arms in ARMS.items():
                if idea=='FIDELITY':continue  # Its unchanged sequence reader has separate real-layer tests.
                for arm in arms:
                    with self.subTest(kind=kind,idea=idea,arm=arm):
                        a=build(idea,arm,31,native.head,native.last,native.bridge,
                                (torch.zeros(64),torch.ones(64)),label_classes=2)
                        b=copy.deepcopy(a)
                        row=project_features({'q':features},{},'.',required_fields(idea,arm))['q']
                        sa=a(features);sb=b(row);self.assertTrue(torch.equal(sa,sb))
                        labels=torch.tensor([0,1,0,1]);la,_=objective(a,idea,features,labels);lb,_=objective(b,idea,row,labels)
                        self.assertTrue(torch.equal(la,lb));la.backward();lb.backward()
                        for p,q in zip(a.parameters(),b.parameters()):
                            self.assertEqual(p.grad is None,q.grad is None)
                            if p.grad is not None:self.assertTrue(torch.equal(p.grad,q.grad))
                        oa=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.001,weight_decay=0)
                        ob=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=.001,weight_decay=0)
                        oa.step();ob.step()
                        for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))
                        self.assertTrue(torch.equal(torch.argsort(a(features)),torch.argsort(b(row))))
                        cases+=1
        self.assertEqual(cases,106)

    def test_completed_cell_is_skipped_and_old_pending_journal_retained(self):
        from applications.rag.qvalue_portable_v001 import json_file,digest,publish
        from applications.rag.top8_suite_v002 import run_recoverable
        from applications.rag.top8_recovery_v001 import Journal
        from applications.rag.top8_feature_io_v003 import resume_binding
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'matrix';output.mkdir();cache=root/'cache.json';cache.write_text('{}')
            old=dict(schema='top8.recoverable-run.v1',sources={'module':'old'},
                caches=[dict(path=str(cache.resolve()),sha256=digest(cache))],seeds=[7,8],ideas=['FEATURE_GATE'],
                epochs=4,canary=False,question_batch=8,lr=.001,weight_decay=0,gradient_clip=1.)
            contract=json_file(output,'run-binding.json',old)
            first=output/'dpr-FEATURE_GATE-HEAD-s7';first.mkdir()
            artifact=json_file(first,'predictions.json',dict(rows=[]))
            file=dict(artifact,path=first.name+'/'+artifact['name'])
            json_file(first,'disposition.json',dict(status='complete',binding=old,result={'preserved':True},files=[file]))
            pending=output/'dpr-FEATURE_GATE-HEAD-s8';pending.mkdir()
            journal=Journal(pending/'recovery',dict(matrix=old,idea='FEATURE_GATE',arm='HEAD',seed=8,backbone='dpr'))
            checkpoint=publish(journal.root,'step-00000032-g0.bin',b'opaque synthetic checkpoint')
            json_file(journal.root,'step-00000032.json',dict(step=32,binding=journal.binding,tensor=checkpoint))
            before={p.relative_to(output).as_posix():p.read_bytes() for p in output.rglob('*') if p.is_file()}
            spec=root/'compat.json';spec.write_text(json.dumps(dict(before_sources=old['sources'],after_sources={'module':'new'})))
            def adopt(a,b,out):return resume_binding(a,b,out,spec)
            calls=[]
            def train(*args,**kw):
                calls.append(args[2]);self.assertEqual(kw['journal'].latest_record()['step'],32)
                self.assertEqual(kw['journal'].binding['matrix'],old)
                return dict(files=[],steps=4000)
            metadata=dict(backbone='dpr',rows=[dict(id='t',role='training'),dict(id='v',role='development',
                          candidate_ids=list(range(50)),native_scores=[0.]*50)])
            with patch('applications.rag.top8_recovery_v001.source_binding',return_value={'module':'new'}),\
                 patch('applications.rag.top8_feature_io_v003.resume_binding',side_effect=adopt),\
                 patch('applications.rag.top8_portable_v001.load_cache',return_value=(metadata,{})),\
                 patch('applications.rag.top8_portable_v001.train_cell',side_effect=train),\
                 patch('applications.rag.top8_methods_v001.ARMS',{'FEATURE_GATE':('HEAD',)}):
                seal=run_recoverable([cache],output,[7,8],['FEATURE_GATE'])
            self.assertEqual(calls,[8]);self.assertEqual(len(seal['results']),2)
            for name,raw in before.items():self.assertEqual((output/name).read_bytes(),raw)
            self.assertEqual(digest(output/'run-binding.json'),contract['sha256'])
            receipt=json.loads((output/'feature-io-continuation.v1.json').read_bytes())
            self.assertEqual(receipt['actual_runtime_binding']['sources'],{'module':'new'})


if __name__=='__main__':unittest.main()
