"""Coordinator fixtures: no scientific imports, GPU, private data or network."""
import copy
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from scripts import run_two_datasets as joint
from scripts import run_junior_experiments as single


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.campaign = json.loads(joint.CAMPAIGN.read_bytes())
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_plan_exact_786_cells_without_scientific_import(self):
        _, report = joint.resolve(self.campaign, {}, list(joint.DATASETS), 'formal')
        self.assertEqual(report['total_learned_cells'], 786)
        self.assertEqual(report['learned_cells_per_dataset'], 393)
        self.assertFalse(report['ready'])
        self.assertFalse(report['formal_scale_certified'])

    def test_gap_stops_both_before_any_work_or_exchange(self):
        with self.assertRaisesRegex(ValueError, 'nq_open'):
            joint.execute(self.campaign, {}, list(joint.DATASETS), 'formal', self.root/'work',
                exchange_factory=lambda *a:self.fail('network touched'))
        self.assertFalse((self.root/'work').exists())

    def test_existing_b1_only_explicit_triviaqa_reproduction(self):
        plans, report = joint.resolve(self.campaign, {}, ['triviaqa'], 'reproduction')
        self.assertTrue(report['ready'])
        self.assertTrue(plans['triviaqa']['populations']['reproduction']['exposed_population'])
        _, both = joint.resolve(self.campaign, {}, list(joint.DATASETS), 'reproduction')
        self.assertFalse(both['ready'])

    def test_wrong_dataset_and_missing_formal_qualification_rejected(self):
        release = copy.deepcopy(json.loads(single.CAMPAIGN.read_bytes())['populations']['reproduction'])
        release.update(population='formal', exposed_population=False, independent_confirmation=True,
                       qualification_evidence=['fixture'], dataset='nq_open')
        config = {'datasets':{'triviaqa':{'populations':{'formal':release}}}}
        _, report = joint.resolve(self.campaign, config, ['triviaqa'], 'formal')
        self.assertFalse(report['ready'])
        release['dataset'] = 'triviaqa'
        _, report = joint.resolve(self.campaign, config, ['triviaqa'], 'formal')
        self.assertIn('headroom', report['readiness'][0]['reason'])

    def fixture(self, name, train_question, eval_question):
        root = self.root/name; root.mkdir()
        def row(identity, question):
            return dict(id=name+identity, component_id=name+identity, question=question,
                        paragraphs=[dict(id=str(i),text='synthetic text') for i in range(50)])
        train = row('train',train_question); train['labels']={'0':2,'1':0}
        evaluation = row('eval',eval_question)
        (root/'train.jsonl').write_text(json.dumps(train)+'\n')
        (root/'eval.jsonl').write_text(json.dumps(evaluation)+'\n')
        (root/'labels.json').write_text('invalid JSON: must remain unopened')
        release = dict(counts=dict(train=1,evaluation=1),
            models={'dpr':'models/dpr-reader-multiset-base','bge':'models/bge-reranker-v2-m3'},
            inputs=dict(train='train.jsonl',eval_inputs='eval.jsonl',eval_labels='labels.json'),
            files=[dict(path='models/'+kind+'/config.json',size=1,sha256='f'*64)
                   for kind in ['dpr-reader-multiset-base','bge-reranker-v2-m3']])
        return root,dict(populations={'formal':release})

    def test_cross_dataset_normalized_train_eval_leak_rejected(self):
        a, pa=self.fixture('a',' Shared   QUESTION ','a eval')
        b, pb=self.fixture('b','b train','shared question')
        with self.assertRaisesRegex(ValueError,'cross-dataset'):
            joint.audit_inputs({'nq_open':a,'triviaqa':b}, {'nq_open':pa,'triviaqa':pb}, 'formal')

    def test_audit_never_parses_eval_labels_and_rejects_model_mismatch(self):
        a, pa=self.fixture('a','a train','a eval'); b,pb=self.fixture('b','b train','b eval')
        staged={'nq_open':a,'triviaqa':b};plans={'nq_open':pa,'triviaqa':pb}
        self.assertFalse(joint.audit_inputs(staged,plans,'formal')['evaluation_label_values_opened'])
        pb['populations']['formal']['files'][0]['sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'model bytes'):
            joint.audit_inputs(staged,plans,'formal')

    def evaluation(self,n,delta,seeds=(1,2)):
        metrics={}
        for kind in ['dpr','bge']:
            base=dict(questions=n,denominator=5*n,useful_lower=n,useful_upper=n,
                      direct_lower=n,direct_upper=n,unknown=0)
            metrics['predictions-'+kind+'-NATIVE.json']=base
            for seed in seeds:
                metrics[f'{kind}-A-QUANTUM-s{seed}/predictions.json']=dict(base,
                    useful_lower=n+delta,useful_upper=n+delta)
                metrics[f'{kind}-A-HEAD-s{seed}/predictions.json']=dict(base)
        return dict(metrics=metrics)

    def test_equal_dataset_rates_not_pooled_raw_totals(self):
        result=joint.summarize({'nq_open':self.evaluation(10,2),
            'triviaqa':self.evaluation(100,-10)},['nq_open','triviaqa'],['A'],[1,2])
        row=result['ranking'][0]
        self.assertAlmostEqual(row['equal_weight_useful_delta_lower'],.05)
        self.assertFalse(row['improves_all_dataset_models'])
        self.assertTrue(row['by_dataset_model']['nq_open/dpr']['classical_controls'])

    def test_missing_seed_never_ranked_and_unknowns_preserved(self):
        evaluation=self.evaluation(10,2,seeds=(1,))
        self.assertFalse(joint.summarize({'nq_open':evaluation},['nq_open'],['A'],[1,2])['ranking'][0]['rankable'])
        evaluation=self.evaluation(10,0)
        evaluation['metrics']['dpr-A-QUANTUM-s1/predictions.json'].update(useful_upper=12,unknown=2)
        row=joint.summarize({'nq_open':evaluation},['nq_open'],['A'],[1,2])['ranking'][0]
        self.assertGreater(row['equal_weight_useful_delta_upper'],row['equal_weight_useful_delta_lower'])

    def test_plan_only_is_read_only_and_config_credentials_not_printed(self):
        config=self.root/'config.json';config.write_text(json.dumps(dict(url='private-host',token='SECRET')))
        output=io.StringIO()
        with patch('sys.stdout',output):
            self.assertEqual(joint.main(['--plan-only','--config',str(config)]),0)
        self.assertNotIn('SECRET',output.getvalue())
        self.assertFalse((self.root/'work').exists())

    def test_two_dataset_execution_and_terminal_reentry_only_uploads(self):
        a,pa=self.fixture('a','a train','a eval');b,pb=self.fixture('b','b train','b eval')
        overrides={}
        for name,plan in [('nq_open',pa),('triviaqa',pb)]:
            release=plan['populations']['formal']
            release.update(status='ready',population='reproduction',dataset=name,exposed_population=True,
                archive=dict(exchange_path='five_ideas/fixture.tar',size=1,sha256='a'*64))
            overrides[name]=dict(populations=dict(reproduction=release))
        config=dict(url='http://fixture',token='fixture',datasets=overrides)
        class Transport:
            fail=False
            def __init__(self,*args):pass
            def json(self,*args):return {'ok':True}
            def upload(self,path,target):
                if self.fail:raise RuntimeError('fixture upload failure')
                return dict(path=target,sha256=joint.digest(path))
        @contextmanager
        def fake_lock(work):
            work.mkdir(parents=True,exist_ok=True);yield 9876
        calls=[];environments=[]
        def runner(command,**kwargs):
            calls.append(command)
            self.assertIn(9876,kwargs['pass_fds'])
            self.assertEqual(command[command.index('--seeds')+1:command.index('--ideas')],
                             ['20261102','20261103','20261104'])
            self.assertEqual(command[command.index('--fusion-ideas')+1:],['CROSS_SCORE','CLASS_FUSION'])
            output=Path(command[command.index('--output')+1]);matrix=output/'matrix';matrix.mkdir(parents=True)
            (matrix/'joint-prediction-seal.json').write_text('{"files":[]}')
            (matrix/'evaluation.json').write_text('{"metrics":{}}')
            return subprocess.CompletedProcess(command,0)
        def environment(*args):environments.append(1);return 'fixture-python'
        def stage(release):return a if release['dataset']=='nq_open' else b
        with patch.object(joint,'campaign_lock',fake_lock),patch.object(joint.AssetStore,'__call__',side_effect=stage):
            self.assertEqual(joint.execute(self.campaign,config,list(joint.DATASETS),'reproduction',self.root/'work',
                runner=runner,environment=environment,exchange_factory=Transport),0)
            self.assertEqual(len(calls),2);self.assertEqual(len(environments),1)
            Transport.fail=True
            with self.assertRaisesRegex(RuntimeError,'upload failure'):
                joint.execute(self.campaign,config,list(joint.DATASETS),'reproduction',self.root/'work',
                    runner=lambda *a,**k:self.fail('retrained terminal cells'),
                    environment=lambda *a:self.fail('environment on upload-only'),exchange_factory=Transport)
            Transport.fail=False
            self.assertEqual(joint.execute(self.campaign,config,list(joint.DATASETS),'reproduction',self.root/'work',
                runner=lambda *a,**k:self.fail('retrained terminal cells'),
                environment=lambda *a:self.fail('environment on upload-only'),exchange_factory=Transport),0)


if __name__=='__main__':
    unittest.main()
