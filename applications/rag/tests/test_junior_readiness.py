"""Local release diagnosis and mirror-byte verification; stdlib fixtures only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from applications.rag.junior_readiness_v001 import local_models
from scripts import run_two_datasets as joint


class ReadinessTests(unittest.TestCase):
    def test_missing_release_diagnosis_never_starts_work_or_network(self):
        with tempfile.TemporaryDirectory() as root:
            out=io.StringIO(); work=Path(root)/'work'
            with contextlib.redirect_stdout(out), patch.object(joint,'execute',side_effect=AssertionError('execution')):
                self.assertEqual(joint.main(['--diagnose','--workspace',str(work)]),0)
            report=json.loads(out.getvalue())
            self.assertEqual(report['category'],'DATA_RELEASE_NOT_READY')
            self.assertFalse(report['ready']); self.assertFalse(report['private_service_contacted'])
            self.assertFalse(report['can_skip_release_gate']); self.assertFalse(work.exists())
            self.assertFalse(report['available_exposed_reproduction']['formal'])

    def test_config_credentials_not_in_diagnosis(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=Path(root)/'cfg.json'; cfg.write_text(json.dumps(dict(url='http://private.invalid',token='PRIVATE_SENTINEL')))
            out=io.StringIO()
            with contextlib.redirect_stdout(out): self.assertEqual(joint.main(['--diagnose','--config',str(cfg)]),0)
            self.assertNotIn('PRIVATE_SENTINEL',out.getvalue()); self.assertNotIn('private.invalid',out.getvalue())
            self.assertTrue(json.loads(out.getvalue())['private_access_config_present'])

    def test_actionable_default_guard_before_credentials_download_or_install(self):
        out=io.StringIO()
        with contextlib.redirect_stderr(out), patch.object(joint.single,'configuration',side_effect=AssertionError('prompt')):
            self.assertEqual(joint.main([]),2)
        self.assertIn('DATA_RELEASE_NOT_READY',out.getvalue())
        self.assertIn('--diagnose',out.getvalue()); self.assertIn('项目负责人',out.getvalue())

    def test_mirror_files_need_full_pinned_hash_not_master_or_file_size(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'models/dpr/config.json'; p.parent.mkdir(parents=True); p.write_bytes(b'valid')
            row=dict(path='models/dpr/config.json',size=5,sha256=hashlib.sha256(b'valid').hexdigest())
            campaign=dict(populations=dict(reproduction=dict(models={'dpr':'models/dpr'},files=[row])))
            def status():return local_models(campaign,root)
            self.assertTrue(status()['all_pinned_model_bytes_match'])
            p.write_bytes(b'other'); self.assertEqual(status()['models']['dpr']['files'][0]['status'],'SHA256_MISMATCH')
            p.write_bytes(b'x'); self.assertEqual(status()['models']['dpr']['files'][0]['status'],'SIZE_MISMATCH')
            p.unlink(); self.assertEqual(status()['models']['dpr']['files'][0]['status'],'MISSING')
            campaign['populations']['reproduction']['files']=[]
            self.assertFalse(status()['all_pinned_model_bytes_match'])

    def test_model_check_alone_cannot_implicitly_start_training(self):
        out=io.StringIO()
        with contextlib.redirect_stderr(out):self.assertEqual(joint.main(['--check-local-models','.']),2)
        self.assertIn('requires --diagnose',out.getvalue())


if __name__=='__main__':unittest.main()
