"""Bootstrap/manifest/publication tests: standard library only."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.run_local_born_smallset import main,source_snapshot,PROTOCOL
from applications.rag.local_born_io_v001 import immutable_json,verify_seal
from scripts.smallset_progress import Progress


class DeliveryTests(unittest.TestCase):
    def test_plan_only_and_transitive_scientific_source(self):
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(['--plan-only','--workspace',d+'/absent']),0)
            value=json.loads(out.getvalue());self.assertFalse(value['scientific_execution'])
            p=value['protocol'];self.assertEqual(p['sealed_prediction_groups'],2+2*len(p['arms'])*len(p['seeds']))
            self.assertEqual(p['steps_per_learning_job'],p['roles']['train']//p['logical_questions']*p['epochs'])
            self.assertFalse(Path(d,'absent').exists())
        source=source_snapshot()['files']
        for n in ['toy100_local_born_v001.py','toy100_output_frame_v001.py','toy100_global_ffn_v001.py','local_born_io_v001.py','model_input_guards.py','top8_cache_v002.py']:
            self.assertIn('applications/rag/'+n,source)

    def test_interrupted_immutable_publish_and_seal_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch('applications.rag.local_born_io_v001.os.link',side_effect=OSError('injected publication interruption')):
                with self.assertRaises(OSError):immutable_json(root,'predictions-dpr.json',dict(rows=[]))
            self.assertEqual(len(list(root.glob('*.part'))),1)
            record=immutable_json(root,'predictions-dpr.json',dict(rows=[]))
            immutable_json(root,'predictions-dpr.json',dict(rows=[]))
            with self.assertRaises(RuntimeError):immutable_json(root,'predictions-dpr.json',dict(rows=[1]))
            seal=dict(files=[dict(path=record['name'],size=record['size'],sha256=record['sha256'])],failures=[],all_predictions_before_evaluation_labels=True,expected_prediction_groups=1)
            immutable_json(root,'joint-prediction-seal.json',seal);verify_seal(root)
            (root/record['name']).write_text('corrupt')
            with self.assertRaises(RuntimeError):verify_seal(root)

    def test_progress_custom_name_preserves_legacy_default(self):
        p=Progress('test','/tmp/log');self.assertIn('8+2',p.snapshot()['name'])
        p.event(dict(qduet_progress=1,event='schedule',time=0,jobs=['a'],name='Local Born · 9 arms'))
        self.assertEqual(p.snapshot()['name'],'Local Born · 9 arms')
        import ast
        engine=Path(__file__).resolve().parents[1]/'local_born_smallset_v001.py'
        states=[kw.value.value for n in ast.walk(ast.parse(engine.read_bytes())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='emit' for kw in n.keywords if kw.arg=='state' and isinstance(kw.value,ast.Constant)]
        self.assertEqual(set(states),{'running','computed'})

    def test_real_ui_child_uses_the_computed_completion_vocabulary(self):
        import sys
        from scripts.smallset_progress import supervise
        code="from scripts.smallset_events import emit;emit('schedule',name='Local Born',jobs=['native-cache-dpr']);emit('cell',job='native-cache-dpr',state='running');emit('cell',job='native-cache-dpr',state='computed')"
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            result=supervise([sys.executable,'-B','-c',code],d,'append',.02,cwd=Path(__file__).resolve().parents[3])
            self.assertEqual(result,0)
            snapshot=json.loads((Path(d)/'progress/latest.json').read_bytes())
            self.assertEqual(snapshot['name'],'Local Born');self.assertTrue(all(j['state']=='computed' for j in snapshot['jobs']))

    def test_coordinator_complete_barrier_evaluation_and_upload_only(self):
        """Real coordinator/evaluator/exporter; fake science/assets/transport."""
        import types
        from applications.rag.junior_transport_v001 import digest
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);protocol=json.loads(PROTOCOL.read_bytes())
            protocol.update(roles=dict(train=8,valid=1,test=1),epochs=1,steps_per_learning_job=1)
            pp=work/'protocol.json';pp.write_text(json.dumps(protocol));source=dict(files={},git_revision='synthetic')
            uploads=[];calls=[]
            class Exchange:
                def __init__(self,*a):pass
                def upload(self,path,name):
                    uploads.append(name);return dict(path=name,size=path.stat().st_size,sha256=digest(path))
            @contextlib.contextmanager
            def lock(*a):yield 17
            def materialize(raw,release,data):
                data.mkdir();roles={'t':'train','v':'valid','x':'test'}
                (data/'roles.json').write_text(json.dumps(roles));(data/'train.jsonl').write_text('training fixture')
                rows=[]
                for q,role in [('v','valid'),('x','test')]:
                    ps=[dict(id=f'p{i}',text='synthetic') for i in range(50)]
                    rows.append(dict(id=q,component_id=q,question=q,paragraphs=ps))
                    (data/(role+'.labels.jsonl')).write_text(json.dumps(dict(id=q,labels={p['id']:i%2 for i,p in enumerate(ps)}))+'\n')
                (data/'evaluation.inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            def runner(argv,**kw):
                calls.append(argv);self.assertIn('--_science',argv);self.assertEqual(kw['pass_fds'],(17,))
                matrix=work/'run/matrix';matrix.mkdir()
                binding=dict(protocol=protocol,source=source,input_files={n:digest(work/'data'/n) for n in ['train.jsonl','evaluation.inputs.jsonl','roles.json']})
                immutable_json(matrix,'run-binding.json',binding);declared=[matrix/'run-binding.json']
                predictions=dict(rows=[dict(id=q,candidate_ids=[f'p{i}' for i in range(50)],scores=[float(-i) for i in range(50)],ranking=list(range(50))) for q in ['v','x']])
                for k in ['dpr','bge']:
                    (matrix/('frozen-'+k+'.bin')).write_bytes(b'synthetic metadata');declared.append(matrix/('frozen-'+k+'.bin'))
                    immutable_json(matrix,'predictions-native-'+k+'.json',predictions);declared.append(matrix/('predictions-native-'+k+'.json'))
                for arm in protocol['arms']:
                    for seed in protocol['seeds']:
                        folder=matrix/(arm+'-s'+str(seed));journal=folder/'journal';journal.mkdir(parents=True)
                        b=dict(binding,arm=arm,seed=seed);immutable_json(journal,'binding.json',b)
                        cp=journal/'step-00000001-g0.bin';cp.write_bytes(b'synthetic checkpoint, not scientific state')
                        record=dict(step=1,binding=b,tensor=dict(name=cp.name,size=cp.stat().st_size,sha256=digest(cp)))
                        immutable_json(journal,'step-00000001.json',record);pred={}
                        for k in ['dpr','bge']:
                            pred[k]=immutable_json(folder,'predictions-'+k+'.json',predictions);declared.append(folder/pred[k]['name'])
                        immutable_json(folder,'result.json',dict(binding=b,arm=arm,seed=seed,step=1,checkpoint=record,predictions=pred))
                        declared += [folder/'result.json',journal/'binding.json',journal/'step-00000001.json',cp]
                files=[dict(path=p.relative_to(matrix).as_posix(),size=p.stat().st_size,sha256=digest(p)) for p in declared]
                immutable_json(matrix,'joint-prediction-seal.json',dict(files=files,failures=[],all_predictions_before_evaluation_labels=True,arms=protocol['arms'],seeds=protocol['seeds'],roles=protocol['roles'],expected_prediction_groups=56))
                return types.SimpleNamespace(returncode=0)
            # No actual download, model import, GPU or external upload.
            with contextlib.ExitStack() as stack:
                for target,value in {
                    'scripts.run_local_born_smallset.PROTOCOL':pp,
                    'scripts.run_local_born_smallset.source_snapshot':lambda:source,
                    'scripts.run_local_born_smallset.campaign_lock':lock,
                    'scripts.run_local_born_smallset.environment':lambda _:Path('python-fixture'),
                    'scripts.run_local_born_smallset.Exchange':Exchange,
                    'scripts.run_local_born_smallset.shutil.disk_usage':lambda _:types.SimpleNamespace(free=100*1024**3),
                    'scripts.run_local_born_smallset.subprocess.run':runner,
                    'scripts.run_junior_experiments.configuration':lambda _:dict(url='https://fixture.invalid',token='synthetic'),
                    'scripts.run_two_datasets.AssetStore':lambda *a:lambda _:work/'fake-assets',
                    'applications.rag.triviaqa_smallset_v001.materialize':materialize,
                }.items():stack.enter_context(patch(target,value))
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['--workspace',str(work),'--display','off']),0)
                self.assertEqual(len(calls),1);self.assertTrue((work/'run/upload-complete.json').exists())
                self.assertEqual(len(json.loads((work/'run/matrix/smallset-evaluation.json').read_bytes())['metrics']),56)
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['--workspace',str(work),'--display','off','--upload-only']),0)
                self.assertEqual(len(calls),1);self.assertGreater(len(uploads),56)


if __name__=='__main__':unittest.main()
