"""Stdlib acceptance of role isolation, exact source binding and sealed evaluation."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from applications.rag import triviaqa_smallset_v001 as small
from applications.rag.junior_transport_v001 import digest


class SmallsetTests(unittest.TestCase):
    def fixtures(self,root):
        counts={'train':2,'valid':1,'test':1};files=[]
        for role,count in counts.items():
            with gzip.open(root/(role+'.inputs.jsonl.gz'),'wb') as inputs, \
                    (root/(role+'.labels.private.jsonl')).open('wb') as labels:
                for i in range(count):
                    q=role+str(i)
                    r=dict(id=q,component_id=hashlib.sha256(q.encode()).hexdigest(),question='Question '+q,
                           paragraphs=[dict(id=str(j),text='Passage '+str(j)) for j in range(50)])
                    raw=small.encoded(r);inputs.write(raw)
                    v=dict(id=q,component_id=r['component_id'],source_row_sha256=hashlib.sha256(raw).hexdigest(),
                           labels=[dict(candidate_id=str(j),useful=int(j<3),decision_source='fixture') for j in range(50)])
                    labels.write(small.encoded(v))
            for suffix in ('.inputs.jsonl.gz','.labels.private.jsonl'):
                path=root/(role+suffix);files.append(dict(path=path.name,size=path.stat().st_size,sha256=digest(path)))
        return dict(schema='qduet.triviaqa-smallset-release.v1',dataset='triviaqa',status='ready',
                    roles=counts,label_semantics='binary_useful',semantic_gold=False,files=files)

    def test_roles_original50_binary_labels_and_idempotent_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=self.fixtures(root)
            with patch.object(small,'ROLES',release['roles']):
                r=small.materialize(root,release,root/'converted')
                self.assertEqual(r['headroom']['valid']['oracle_top5'],3)
                self.assertNotIn('test',r['headroom'])
                self.assertEqual(r,small.materialize(root,release,root/'converted'))
                for line in (root/'converted/evaluation.inputs.jsonl').read_text().splitlines():
                    self.assertEqual(set(json.loads(line)),{'id','component_id','question','paragraphs'})
                with (root/'converted/train.jsonl').open('a') as f:f.write('corrupt\n')
                with self.assertRaises(ValueError):small.materialize(root,release,root/'converted')

    def test_wrong_row_hash_candidate_identity_extra_labels_and_nonbinary_rejected(self):
        for kind in ('hash','candidate','extra','unknown'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.fixtures(root);path=root/'valid.labels.private.jsonl'
                v=json.loads(path.read_bytes())
                if kind=='hash':v['source_row_sha256']='0'*64
                if kind=='candidate':v['labels'][0]['candidate_id']='not-the-candidate'
                if kind=='unknown':v['labels'][0]['useful']=None
                path.write_bytes(small.encoded(v)+(small.encoded(v) if kind=='extra' else b''))
                with self.assertRaises(ValueError):list(small.converted_rows(root,'valid',1))

    def test_binary_evaluation_requires_exact_seal_rows_ranking_and_role_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=self.fixtures(root);data=root/'data'
            with patch.object(small,'ROLES',release['roles']):small.materialize(root,release,data)
            out=root/'matrix';out.mkdir();rows=[]
            for role in ('valid','test'):
                rows.append(dict(id=role+'0',candidate_ids=list(map(str,range(50))),
                                 scores=list(map(float,range(50))),ranking=list(reversed(range(50)))))
            p=out/'predictions-dpr-NATIVE.json';small.save(p,dict(rows=rows))
            seal=dict(all_predictions_before_evaluation_labels=True,failures=[],
                      files=[dict(name=p.name,size=p.stat().st_size,sha256=digest(p))])
            small.save(out/'joint-prediction-seal.json',seal)
            r=small.evaluate(out,data)
            self.assertEqual(r['direct_metric'],'NOT_APPLICABLE')
            self.assertEqual(r['metrics'][p.name]['test'],dict(questions=1,useful=0,denominator=5))
            p.write_text('{}')
            with self.assertRaises(ValueError):small.evaluate(out,data)


if __name__=='__main__':unittest.main()
