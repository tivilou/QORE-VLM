"""Stdlib fixtures for fusion identity, dispatch, recovery, seal and reporting.

No Torch/model import or execution, private data, GPU, internet or downloads.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from applications.rag import qduet_fusion_v001 as fusion
from applications.rag.top8_matrix_plan_v001 import rank_metrics
from applications.rag.qvalue_portable_v001 import digest, json_file
from scripts import run_junior_experiments as single
from scripts import run_two_datasets as joint


class FusionTests(unittest.TestCase):
    def caches(self):
        train = dict(id='train', component_id='ct', role='training', candidate_ids=list(map(str,range(50))),
            retained_indices=[0,3], targets=[2,0], native_scores=list(map(float,range(50))))
        dev = dict(train, id='eval', component_id='ce', role='development', retained_indices=list(range(50)), targets=None)
        a = dict(schema='top8.native-cache.v1', backbone='dpr', train_sha256='a'*64,
                 eval_input_sha256='b'*64, evaluation_labels_present=False, rows=[train,dev],
                 model_id='facebook/dpr-reader-multiset-base', revision='ab1974d6023bc4dd27cd713c9ac2b6247509244d')
        b = copy.deepcopy(a); b['backbone'] = 'bge'
        b.update(model_id='BAAI/bge-reranker-v2-m3', revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')
        return a,b

    def test_exact_population_and_original50_normalization_even_unreviewed(self):
        a,b = self.caches(); rows = fusion.aligned_rows(a,b)
        self.assertEqual(len(rows[0]['paired_scores']),50)
        self.assertEqual(rows[0]['retained_indices'],[0,3])
        self.assertEqual(rows[0]['targets'],[2,0])
        self.assertIsNone(rows[1]['targets'])

    def test_old_cache_wrong_population_candidate_role_and_targets_rejected(self):
        for key,value in [('native_scores',None), ('candidate_ids',list(reversed(list(map(str,range(50)))))),
                          ('targets',[0,2]), ('component_id','wrong'), ('role','development')]:
            a,b = self.caches(); b['rows'][0][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                fusion.aligned_rows(a,b)
        a,b = self.caches(); b['train_sha256']='c'*64
        with self.assertRaises(RuntimeError): fusion.aligned_rows(a,b)
        a,b = self.caches(); a['rows'][0]['native_scores'][1]=float('nan')
        with self.assertRaises(RuntimeError): fusion.aligned_rows(a,b)

    def test_fusion_needs_both_models_and_separate_registry(self):
        self.assertEqual(fusion.fusion_plan(['dpr','bge'],list(fusion.FUSION_ARMS)),list(fusion.FUSION_ARMS))
        for models,ideas in [(['dpr'],['CROSS_SCORE']), (['bge'],['CLASS_FUSION']),
                              (['dpr','bge'],['CROSS_SCORE','CROSS_SCORE']), (['dpr','bge'],['madeup'])]:
            with self.assertRaises(RuntimeError): fusion.fusion_plan(models,ideas)

    def test_default_matrix_786_cells_with_45_fusion_cells_per_dataset(self):
        campaign=json.loads(joint.CAMPAIGN.read_bytes())
        plans,report=joint.resolve(campaign,{},list(joint.DATASETS),'formal')
        self.assertEqual(report['total_learned_cells'],786)
        self.assertEqual(report['learned_cells_per_dataset'],393)
        self.assertEqual(report['fusion_ideas'],list(fusion.FUSION_ARMS))
        plans,_=joint.resolve(campaign,{},['triviaqa'],'reproduction')
        plan=plans['triviaqa']; release=plan['populations']['reproduction']
        command=single.matrix_argv('python',Path('.'),Path('out'),release,plan,False)
        self.assertEqual(command[command.index('--fusion-ideas')+1:],list(fusion.FUSION_ARMS))

    def metrics(self, seeds=(1,2)):
        base=dict(useful_lower=10,useful_upper=10,direct_lower=5,direct_upper=5,
                  unknown=0,questions=4,denominator=20)
        m={'predictions-dpr-NATIVE.json':base,'predictions-bge-NATIVE.json':dict(base,useful_lower=9,useful_upper=9)}
        for seed in seeds:
            m[f'fusion-CROSS_SCORE-QUANTUM-s{seed}/predictions.json']=dict(base,useful_lower=12,useful_upper=13,unknown=1)
            m[f'fusion-CROSS_SCORE-LINEAR-s{seed}/predictions.json']=base
        return m

    def test_single_fusion_output_per_seed_both_native_baselines_controls_and_bounds(self):
        rows,groups=rank_metrics(dict(ideas=[],seeds=[1,2],fusion_ideas=['CROSS_SCORE']),self.metrics())
        self.assertEqual(groups['dual'],[]); row=groups['fusion'][0]
        self.assertTrue(row['rankable']); self.assertEqual(row['output_rankings_per_seed'],1)
        self.assertEqual(len(row['per_seed']),2)
        self.assertEqual(row['by_native']['dpr']['per_seed']['1']['useful_delta'],[2,3])
        self.assertEqual(row['by_native']['bge']['per_seed']['1']['useful_delta'],[3,4])
        self.assertEqual(len(row['classical_controls']),2)
        self.assertEqual(rows,[row]); self.assertEqual(row['quantum_attribution'],'NOT_ESTABLISHED')

    def test_missing_seed_or_channel_never_complete_and_cross_dataset_keeps_raw_values(self):
        seal=dict(ideas=[],seeds=[1,2],fusion_ideas=['CROSS_SCORE'])
        self.assertFalse(rank_metrics(seal,self.metrics((1,)))[0][0]['rankable'])
        m=self.metrics(); del m['predictions-bge-NATIVE.json']
        self.assertFalse(rank_metrics(seal,m)[0][0]['rankable'])
        m=self.metrics(); m['predictions-bge-NATIVE.json']['questions']=5
        invalid=rank_metrics(seal,m)[0][0]
        self.assertFalse(invalid['rankable']); self.assertEqual(invalid['by_native']['bge']['per_seed'],{})
        report=joint.summarize({'nq_open':dict(metrics=self.metrics()),'triviaqa':dict(metrics=self.metrics())},
                               ['nq_open','triviaqa'],[],[1,2],['CROSS_SCORE'])
        row=report['fusion_results'][0]
        self.assertTrue(row['rankable']); self.assertEqual(report['ranking'],[])
        self.assertEqual(row['by_dataset']['nq_open']['per_seed']['1']['useful_lower'],12)

    def test_completed_and_failed_fusion_cells_not_retrained_all_artifacts_returned_for_joint_seal(self):
        a,b=self.caches()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); paths=[]
            for kind,manifest in [('dpr',a),('bge',b)]:
                folder=root/kind;folder.mkdir(); f=folder/'frozen-modules.bin';f.write_bytes(b'fixture')
                manifest['files']=[dict(name=f.name,size=f.stat().st_size,sha256=digest(f))]
                p=folder/'cache.json';p.write_text(json.dumps(manifest));paths.append(p)
            output=root/'matrix'; output.mkdir(); calls=[]
            def train(idea,arm,seed,rows,features,head,destination,*args):
                calls.append((idea,arm,seed))
                if arm=='POLY': raise RuntimeError('retained explicit failure')
                record=json_file(destination,'predictions-fixture.json',dict(rows=[]))
                return dict(files=[record])
            fake=SimpleNamespace(load=lambda *a,**k:dict(head={}),cuda=SimpleNamespace(empty_cache=lambda:None))
            with patch.dict(sys.modules,torch=fake), patch('applications.rag.top8_portable_v001.load_cache',return_value=(a,{})), \
                    patch.object(fusion,'train_fusion_cell',side_effect=train):
                result,failed,files=fusion.run_fusions(paths,output,[1],['CROSS_SCORE'],{'fixed':True})
                self.assertEqual(len(calls),7);self.assertEqual(len(result),6);self.assertEqual(len(failed),1)
                self.assertEqual(len(files),6); self.assertTrue(all('path' in r for r in files))
                self.assertFalse((output/'joint-prediction-seal.json').exists())
                again=fusion.run_fusions(paths,output,[1],['CROSS_SCORE'],{'fixed':True})
                self.assertEqual(len(calls),7); self.assertEqual(again,(result,failed,files))
                with self.assertRaisesRegex(RuntimeError,'binding'):
                    fusion.run_fusions(paths,output,[1],['CROSS_SCORE'],{'fixed':False})


if __name__=='__main__': unittest.main()
