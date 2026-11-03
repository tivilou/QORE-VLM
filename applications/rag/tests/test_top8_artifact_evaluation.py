"""Pure-stdlib artifact tests; never imports a scientific runtime."""
import json
from pathlib import Path
import tempfile
import unittest
from applications.rag.qvalue_portable_v001 import json_file
from applications.rag.top8_portable_v001 import evaluate


class SparseRankingTests(unittest.TestCase):
    def test_missing_labels_bound_gain_and_missing_backbone_is_unranked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);ids=[str(i) for i in range(50)]
            files=[]
            for name,ranking in [('predictions-bge-NATIVE.json',list(range(50))),
                                 ('predictions-FEATURE_GATE-QUANTUM-s7.json',[5,6,7,8,9]+list(range(5))+list(range(10,50)))]:
                if 'QUANTUM' in name:
                    cell=root/'bge-FEATURE_GATE-QUANTUM-s7';cell.mkdir();record=json_file(cell,name,dict(rows=[dict(id='q',candidate_ids=ids,ranking=ranking)]));record['path']=cell.name+'/'+name
                else:record=json_file(root,name,dict(rows=[dict(id='q',candidate_ids=ids,ranking=ranking)]))
                files.append(record)
            json_file(root,'joint-prediction-seal.json',dict(all_predictions_before_evaluation_labels=True,
                files=files,ideas=['FEATURE_GATE'],seeds=[7]))
            labels=root/'labels.json';labels.write_text(json.dumps({'q':{'0':1,'1':2,'2':0,'3':0,'4':0,'5':1,'6':1,'7':0,'8':0}}))
            result=evaluate(root,labels)
            metric=result['metrics']['bge-FEATURE_GATE-QUANTUM-s7/predictions-FEATURE_GATE-QUANTUM-s7.json']
            self.assertEqual((metric['useful_lower'],metric['useful_upper'],metric['unknown']),(2,3,1))
            self.assertFalse(result['ranking'][0]['rankable'])
            self.assertEqual(result['ranking'][0]['by_model']['bge']['mean_native_delta_lower'],0)
            self.assertEqual(result['ranking'][0]['by_model']['bge']['mean_native_delta_upper'],1)

    def test_modified_sealed_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);r=json_file(root,'predictions-bge-NATIVE.json',dict(rows=[]))
            json_file(root,'joint-prediction-seal.json',dict(all_predictions_before_evaluation_labels=True,
                files=[r],ideas=[],seeds=[]))
            (root/r['name']).write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'sealed prediction'):
                evaluate(root,root/'does-not-exist-labels.json')


if __name__=='__main__':unittest.main()
