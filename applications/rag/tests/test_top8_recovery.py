"""Journal integrity tests using ordinary bytes; no Torch import or simulation."""
import json
from pathlib import Path
import tempfile
import unittest
from applications.rag.qvalue_portable_v001 import publish, json_file
from applications.rag.top8_recovery_v001 import Journal, new_generation
from applications.rag.top8_suite_v002 import verify_files


class RecoveryTests(unittest.TestCase):
    def test_fidelity_upgrade_keeps_prior_receipts_and_exact_scientific_binding(self):
        from applications.rag.top8_feature_io_v003 import resume_binding
        directory=Path(__file__).resolve().parents[3]/'configs/experiments/triviaqa_smallset_20261112'
        v1=directory/'feature-io-compatibility.v001.json';v2=directory/'runtime-speed-compatibility.v002.json'
        v3=directory/'fidelity-speed-compatibility.v003.json'
        one=json.loads(v1.read_bytes());two=json.loads(v2.read_bytes());three=json.loads(v3.read_bytes())
        for ancestor in three['accepted_previous_sources']:
            original=dict(sources=ancestor,seeds=[20261102],epochs=4,caches=[dict(sha256='fixed')])
            current=dict(original,sources=three['after_sources'])
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);priors={}
                if ancestor==one['before_sources']:
                    _,p=resume_binding(original,dict(original,sources=one['after_sources']),root,v1);priors[p.name]=p.read_bytes()
                if ancestor in two['accepted_previous_sources']:
                    _,p=resume_binding(original,dict(original,sources=two['after_sources']),root,v2);priors[p.name]=p.read_bytes()
                selected,p=resume_binding(original,current,root,v3)
                self.assertEqual(selected,original);self.assertEqual(p.name,'fidelity-speed-continuation.v3.json')
                for name,raw in priors.items():self.assertEqual((root/name).read_bytes(),raw)
                self.assertEqual(len(json.loads(p.read_bytes())['previous_continuations']),len(priors))
                self.assertEqual(resume_binding(original,current,root,v3),(original,p))
                with self.assertRaisesRegex(RuntimeError,'cannot change'):
                    resume_binding(original,dict(current,epochs=3),root,v3)
                with self.assertRaisesRegex(RuntimeError,'exact reviewed'):
                    resume_binding(original,dict(current,sources={'unknown':'source'}),root,v3)
                if priors:
                    name=next(iter(priors));bad=json.loads(priors[name]);bad['changes']=['tampered']
                    (root/name).write_text(json.dumps(bad))
                    with self.assertRaisesRegex(RuntimeError,'same recorded'):
                        resume_binding(original,current,root,v3)

    def test_latest_committed_step_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binding={'source':'locked','seed':7}
            journal=Journal(root,binding)
            record=publish(root,'step-00000032-g0.bin',b'opaque checkpoint')
            json_file(root,'step-00000032.json',dict(step=32,binding=binding,tensor=record))
            # Interrupted later bytes without a commit receipt cannot become latest.
            publish(root,'step-00000064-g0.bin',b'unsealed')
            self.assertEqual(journal.latest_record()['step'],32)
            with self.assertRaisesRegex(RuntimeError,'configuration/source/cache'):
                Journal(root,dict(binding,seed=8))
            (root/record['name']).write_bytes(b'altered')
            with self.assertRaisesRegex(RuntimeError,'cache file size/SHA'):
                journal.latest_record()

    def test_new_generations_preserve_old_and_reject_escaping_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);first=new_generation(root);second=new_generation(root)
            self.assertNotEqual(first,second)
            r=publish(first,'predictions.json',b'[]')
            value=dict(r,path=first.name+'/'+r['name'])
            verify_files(root,[value])
            with self.assertRaisesRegex(RuntimeError,'immutable completed cell'):
                verify_files(root,[dict(value,path='../outside.json')])


if __name__=='__main__':unittest.main()
