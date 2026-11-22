"""Synthetic file/transport tests only; no model, data release or GPU imports."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from applications.rag.smallset_controls_v001 import collect, inventory


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n', encoding='utf-8', newline='\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExistingControlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.root = self.work/'run-smallset'
        self.primary = self.root/'matrix-primary54'
        self.legacy = self.root/'matrix'
        cache_sha = {kind: save(self.root/f'cache-{kind}/cache.json', {'fixture':kind})
                     for kind in ('dpr', 'bge')}
        self.binding = dict(seeds=[20261102], epochs=4, question_batch=8, canary=False,
                            caches=[dict(path=kind, sha256=sha) for kind, sha in cache_sha.items()],
                            sources={'fixture.py':'0'*64})
        sha = save(self.primary/'run-binding.json', self.binding)
        seal_sha = save(self.primary/'joint-prediction-seal.json', {'fixture':True})
        save(self.primary/'adoption.json', {'fixture':True})
        save(self.root/'execution.json', dict(returncode=0, all_cells_complete=True,
             experiment_profile='primary54', roles=dict(train=8000, valid=1000, test=1000)))
        save(self.legacy/'run-binding.json', self.binding)
        self.request = dict(schema='qduet.existing-control-request.v1', primary_run_id='fixture',
            primary_joint_seal_sha256=seal_sha, original_run_binding_sha256=sha,
            seeds=[20261102], roles=dict(train=8000, valid=1000, test=1000), steps=4000,
            cache_sha256=cache_sha, control_arms={'MIXED_DEPTH':['HEAD','CONCAT']},
            max_upload_bytes=1024**2, max_prediction_bytes=128*1024, max_metadata_bytes=128*1024)
        self.request_path = self.work/'request.json'
        save(self.request_path, self.request)
        self.cell = 'dpr-MIXED_DEPTH-HEAD-s20261102'
        prefix = self.legacy/self.cell/'run-0000'
        records = []
        for name in ['predictions-MIXED_DEPTH-HEAD-s20261102.json', 'checkpoint-fixture.bin']:
            sha = save(prefix/name, {'fixture':'no real predictions or tensors'})
            records.append(dict(name=name, path=f'{self.cell}/run-0000/{name}',
                                size=(prefix/name).stat().st_size, sha256=sha))
        self.disposition = dict(status='complete', binding=self.binding, files=records,
            result=dict(idea='MIXED_DEPTH', arm='HEAD', seed=20261102, steps=4000,
                        training_questions=8000, evaluation_questions=2000, trainable_parameters=768))
        self.disposition_path = self.legacy/self.cell/'disposition.json'
        save(self.disposition_path, self.disposition)

    def test_fixed_roster_and_only_completed_predictions(self):
        result = inventory(self.work, self.request)
        self.assertEqual(result['summary']['expected'], 4)
        self.assertEqual(result['summary']['complete'], 1)
        self.assertEqual(result['summary']['not_present'], 3)
        self.assertFalse(any(x['path'].endswith('.bin') for x in result['files']))
        self.assertFalse(result['pre_label_joint_seal_inferred'])
        self.assertFalse(result['label_join'])

    def test_no_legacy_is_absence_not_failure_or_zero_effect(self):
        self.disposition_path.unlink()
        (self.legacy/'run-binding.json').unlink()
        r = inventory(self.work, self.request)
        self.assertFalse(r['legacy_matrix_present'])
        self.assertEqual(r['summary']['not_present'], 4)
        self.assertTrue(r['no_controls_is_not_zero_effect'])

    def test_failed_control_is_retained(self):
        save(self.disposition_path, dict(status='failed', binding=self.binding))
        r = inventory(self.work, self.request)
        self.assertEqual(r['summary']['failed'], 1)
        self.assertEqual(r['summary']['complete'], 0)

    def test_checkpoint_only_is_not_complete_or_missing(self):
        self.disposition_path.unlink()
        save(self.legacy/self.cell/'recovery/binding.json',
             dict(matrix=self.binding, backbone='dpr', idea='MIXED_DEPTH', arm='HEAD', seed=20261102))
        r = inventory(self.work, self.request)
        self.assertEqual(r['summary']['checkpoint_only'], 1)
        self.assertEqual(r['summary']['complete'], 0)
        self.assertEqual(r['summary']['not_present'], 3)

    def test_changed_primary_seal_rejected(self):
        save(self.primary/'joint-prediction-seal.json', {'changed':True})
        with self.assertRaisesRegex(ValueError, 'different primary'):
            inventory(self.work, self.request)

    def test_changed_population_binding_rejected(self):
        save(self.legacy/'run-binding.json', dict(self.binding, epochs=1))
        with self.assertRaisesRegex(ValueError, 'different source/input'):
            inventory(self.work, self.request)

    def test_changed_current_cache_rejected(self):
        save(self.root/'cache-dpr/cache.json', {'changed':True})
        with self.assertRaisesRegex(ValueError, 'cache inventory changed'):
            inventory(self.work, self.request)

    def test_changed_completed_checkpoint_rejected_without_loading_tensor(self):
        save(self.legacy/self.disposition['files'][1]['path'], {'changed':True})
        with self.assertRaisesRegex(ValueError, 'sealed control file changed'):
            inventory(self.work, self.request)

    def test_incomplete_step_rejected(self):
        self.disposition['result']['steps'] = 3999
        save(self.disposition_path, self.disposition)
        with self.assertRaisesRegex(ValueError, 'partial control'):
            inventory(self.work, self.request)

    def test_duplicate_declared_file_rejected(self):
        self.disposition['files'].append(self.disposition['files'][0])
        save(self.disposition_path, self.disposition)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inventory(self.work, self.request)

    def test_cross_cell_path_rejected(self):
        self.disposition['files'][0]['path'] = '../outside.json'
        save(self.disposition_path, self.disposition)
        with self.assertRaisesRegex(ValueError, 'crosses cell boundary'):
            inventory(self.work, self.request)

    def test_upload_cap_stops_before_output_or_network(self):
        self.request['max_upload_bytes'] = 1
        save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, 'upload cap'):
            collect(self.work, self.request_path, upload=True)
        self.assertFalse((self.work/'control-receipts').exists())

    def test_metadata_only_mode_leaves_sources_identical(self):
        before = {p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = collect(self.work, self.request_path)
        self.assertFalse(result['uploaded'])
        self.assertFalse(result['training_launched'])
        self.assertTrue(all(p.read_bytes() == raw for p, raw in before.items()))

    def test_missing_config_has_no_interactive_prompt(self):
        with self.assertRaisesRegex(ValueError, 'private junior.json'):
            collect(self.work, self.request_path, upload=True, config_path=self.work/'missing.json')

    def fake_exchange(self, bad=False, fail=False):
        calls = []
        class Fake:
            def __init__(self, url, token):
                pass
            def upload(self, path, remote):
                calls.append(remote)
                if fail:
                    raise RuntimeError('synthetic transport failure')
                raw = Path(path).read_bytes()
                return dict(path=remote, size_bytes=len(raw),
                            sha256='0'*64 if bad else hashlib.sha256(raw).hexdigest())
        config = self.work/'private.json'
        save(config, dict(url='http://fixture.invalid', token='fixture-token'))
        return Fake, config, calls

    def test_upload_exact_receipts_and_manifest_last(self):
        factory, config, calls = self.fake_exchange()
        r = collect(self.work, self.request_path, upload=True, config_path=config,
                    exchange_factory=factory)
        self.assertTrue(r['uploaded'])
        self.assertTrue(calls[-1].endswith('/upload.json'))
        self.assertTrue(calls[-2].endswith('/inventory.json'))
        self.assertFalse(any(x.endswith('.bin') for x in calls))

    def test_receipt_mismatch_stops(self):
        factory, config, calls = self.fake_exchange(bad=True)
        with self.assertRaisesRegex(ValueError, 'receipt mismatch'):
            collect(self.work, self.request_path, upload=True, config_path=config,
                    exchange_factory=factory)
        self.assertEqual(len(calls), 1)

    def test_transport_failure_retains_pending_without_retry(self):
        factory, config, calls = self.fake_exchange(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'synthetic transport'):
            collect(self.work, self.request_path, upload=True, config_path=config,
                    exchange_factory=factory)
        self.assertEqual(len(calls), 1)
        pending = list(self.work.glob('control-receipts/*/upload.json'))
        self.assertEqual(len(pending), 1)
        self.assertEqual(json.loads(pending[0].read_bytes())['status'], 'pending')

    def test_entry_mode_bypasses_training_and_asset_preparation(self):
        from scripts import run_triviaqa_smallset as entry
        with patch('applications.rag.smallset_controls_v001.collect', return_value={'stage':'fixture'}) as collect_mock, \
             patch.object(entry, 'prepare_environment', side_effect=AssertionError('training route')), \
             patch.object(entry, 'check_delivery', side_effect=AssertionError('training preflight')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(entry.main(['--upload-existing-controls', '--workspace', str(self.work)]), 0)
        self.assertEqual(json.loads(output.getvalue()), {'stage':'fixture'})
        self.assertTrue(collect_mock.call_args.kwargs['upload'])


if __name__ == '__main__':
    unittest.main()
