"""Bootstrap/transport fixtures only: no model, GPU, private dataset or internet."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from applications.rag.junior_transport_v001 import digest, install_package, relative
from scripts import run_junior_experiments as junior


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tar(self, entries):
        path = self.root / 'package.tar'
        with tarfile.open(path, 'w') as tar:
            for name, data, kind in entries:
                entry = tarfile.TarInfo(name); entry.size = len(data)
                if kind == 'link':
                    entry.type = tarfile.SYMTYPE; entry.linkname = '../escape'; entry.size = 0
                tar.addfile(entry, io.BytesIO(data) if kind != 'link' else None)
        return path

    def test_traversal_absolute_and_ambiguous_paths_rejected(self):
        for name in ['../x', '/etc/x', 'a/../x', 'a\\b', 'a//b']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative(name)

    def test_duplicate_and_symlink_packages_rejected_before_install(self):
        import hashlib
        record = dict(path='a', size=1, sha256=hashlib.sha256(b'x').hexdigest())
        for entries in [[('a', b'x', 'file'), ('a', b'x', 'file')], [('a', b'', 'link')]]:
            with self.assertRaises(ValueError):
                install_package(self.tar(entries), self.root / 'assets', [record])
            self.assertFalse((self.root / 'assets/a').exists())

    def test_hash_failure_leaves_no_installed_target(self):
        row = dict(path='a', size=1, sha256='0' * 64)
        with self.assertRaises(ValueError):
            install_package(self.tar([('a', b'x', 'file')]), self.root / 'assets', [row])
        self.assertFalse((self.root / 'assets/a').exists())

    def fixture(self):
        source = self.root / 'source'; source.mkdir()
        def question(identity, component):
            return dict(id=identity, component_id=component, question='question ' + identity,
                        paragraphs=[dict(id=str(i), text='paragraph') for i in range(50)])
        train = question('train', 'c1'); train['labels'] = {'0': 2, '1': 0}
        evaluation = question('eval', 'c2')
        (source / 'train.jsonl').write_text(json.dumps(train) + '\n')
        (source / 'eval.jsonl').write_text(json.dumps(evaluation) + '\n')
        (source / 'labels.json').write_text('{"must_not_read_before_seal":true}')
        entries = [(p.name, p.read_bytes(), 'file') for p in source.iterdir()]
        archive = self.tar(entries)
        rows = [dict(path=p.name, size=p.stat().st_size, sha256=digest(p)) for p in source.iterdir()]
        release = dict(status='ready', population='reproduction', exposed_population=True,
            archive=dict(exchange_path='five_ideas/fixture.tar', size=archive.stat().st_size, sha256=digest(archive)),
            files=rows, counts=dict(train=1,evaluation=1),
            models=dict(dpr='models/dpr', bge='models/bge'),
            inputs=dict(train='train.jsonl',eval_inputs='eval.jsonl',eval_labels='labels.json'))
        ideas = json.loads(junior.CAMPAIGN.read_bytes())['dual_ideas']
        campaign = dict(populations={'reproduction':release,'formal':dict(status='unavailable',reason='no qualified full population')},
                        dual_ideas=ideas, dpr_only_ideas=[], implemented_ideas=ideas, reserve_bytes=0)
        class ExchangeFixture:
            fail = False
            uploads = []
            def __init__(self, *a): pass
            def json(self, *a): return {'ok':True}
            def download(self, row, target): shutil.copyfile(archive, target)
            def upload(self, path, remote):
                if self.fail: raise RuntimeError('fixture transport unavailable')
                self.uploads.append(remote)
                return dict(path=remote,size_bytes=Path(path).stat().st_size,sha256=digest(path))
        return source, campaign, ExchangeFixture

    def test_formal_gap_stops_before_download_environment_or_science(self):
        _, campaign, transport = self.fixture()
        with self.assertRaisesRegex(ValueError, 'no qualified'):
            junior.execute(campaign,dict(url='http://fixture',token='test'), 'formal', self.root/'work',
                           runner=lambda *a,**k:self.fail('science invoked'),
                           environment=lambda *a:self.fail('environment invoked'),exchange_factory=transport)
        self.assertFalse((self.root/'work').exists())

    def test_role_component_leak_rejected(self):
        source, campaign, _ = self.fixture()
        p=source/'eval.jsonl'; row=json.loads(p.read_bytes()); row['component_id']='c1'; p.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'role leakage'):
            junior.validate_inputs(source,campaign['populations']['reproduction'])

    def test_eval_label_values_remain_unopened_during_input_validation(self):
        source, campaign, _ = self.fixture()
        (source/'labels.json').write_text('not parsed until predictions are sealed')
        self.assertEqual(junior.validate_inputs(source,campaign['populations']['reproduction']),dict(train=1,evaluation=1))

    def test_eight_dual_schedule_and_no_fabricated_extra_dpr(self):
        source,campaign,_=self.fixture();release=campaign['populations']['reproduction']
        command=junior.matrix_argv('python',source,self.root/'output',release,campaign,False)
        self.assertEqual(command[command.index('--ideas')+1:],campaign['dual_ideas'])
        self.assertNotIn('--dpr-only-ideas', command)

    def test_upload_failure_reruns_transport_without_retraining(self):
        _,campaign,transport=self.fixture();work=self.root/'work';calls=[]
        def runner(command,**kwargs):
            calls.append(command)
            output=Path(command[command.index('--output')+1]); matrix=output/'matrix';matrix.mkdir(parents=True)
            prediction=matrix/'predictions.json';prediction.write_text('{"fixture":true}')
            (matrix/'joint-prediction-seal.json').write_text(json.dumps(dict(files=[dict(path=prediction.name,size=prediction.stat().st_size,sha256=digest(prediction))])))
            (matrix/'evaluation.json').write_text('{"fixture":true}')
            return subprocess.CompletedProcess(command,0)
        config=dict(url='http://fixture',token='test')
        transport.fail=True
        with self.assertRaisesRegex(RuntimeError,'transport unavailable'):
            junior.execute(campaign,config,'reproduction',work,runner=runner,
                           environment=lambda *a:'python',exchange_factory=transport)
        self.assertTrue((work/'run-reproduction/pending-upload.json').exists())
        transport.fail=False
        self.assertEqual(junior.execute(campaign,config,'reproduction',work,runner=runner,
                         environment=lambda *a:self.fail('unnecessary environment'),exchange_factory=transport),0)
        self.assertEqual(len(calls),1)
        self.assertTrue((work/'run-reproduction/upload-complete.json').exists())
        self.assertTrue(any(x.endswith('predictions.json') for x in transport.uploads))


if __name__ == '__main__':
    unittest.main()
