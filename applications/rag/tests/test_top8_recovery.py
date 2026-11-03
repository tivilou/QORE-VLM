"""Journal integrity tests using ordinary bytes; no Torch import or simulation."""
import json
from pathlib import Path
import tempfile
import unittest
from applications.rag.qvalue_portable_v001 import publish, json_file
from applications.rag.top8_recovery_v001 import Journal, new_generation
from applications.rag.top8_suite_v002 import verify_files


class RecoveryTests(unittest.TestCase):
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
