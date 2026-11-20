"""Cache admission/capacity software checks. Tensor fixtures are tiny CPU-only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from applications.rag.top8_cache_v002 import DiskLRU, sequence_capacity_plan, owned_scratch_bytes, cache_capacity


class CapacityPolicy(unittest.TestCase):
    def key(self, text):return hashlib.sha256(text.encode()).hexdigest()

    def test_auto_counts_filesystem_once_and_keeps_reserves(self):
        with tempfile.TemporaryDirectory() as d:
            roots=[Path(d)/n for n in ('dpr','bge')]
            for r in roots:r.mkdir()
            with patch('applications.rag.top8_cache_v002.shutil.disk_usage') as usage:
                usage.return_value.free=400*1024**3
                plan=sequence_capacity_plan(roots)
                self.assertEqual(set(plan['limits'].values()),{156*1024**3})
                self.assertEqual(usage.call_count,1)
                self.assertEqual(len(plan['filesystems']),1)
            self.assertEqual(set(Path(d).rglob('*')),set(roots))

    def test_resume_adds_only_indexed_owned_bytes_and_explicit_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=DiskLRU(root/'bounded-sequences-v2',100)
            cache.put(self.key('owned'),b'12345');cache.close()
            (root/'legacy.bin').write_bytes(b'ignored')
            self.assertEqual(owned_scratch_bytes(root),5)
            with patch('applications.rag.top8_cache_v002.shutil.disk_usage') as usage:
                usage.return_value.free=88*1024**3
                self.assertEqual(sequence_capacity_plan([root])['limits'][str(root.resolve())],5)
                self.assertEqual(sequence_capacity_plan([root],0)['limits'][str(root.resolve())],0)
                self.assertEqual(sequence_capacity_plan([root],12)['limits'][str(root.resolve())],12*1024**3)
                usage.return_value.free=1000*1024**3
                self.assertEqual(sequence_capacity_plan([root])['limits'][str(root.resolve())],256*1024**3)
                usage.return_value.free=20*1024**3
                self.assertEqual(sequence_capacity_plan([root])['limits'][str(root.resolve())],0)

    def test_training_stays_admitted_across_full_scan_and_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            cache=DiskLRU(d,10,protect_training=True)
            cache.put(self.key('eval'),b'12345','development')
            cache.put(self.key('train1'),b'12345','training')
            cache.put(self.key('train2'),b'12345','training')
            self.assertIsNone(cache.get(self.key('eval')))
            for i in range(20):
                self.assertFalse(cache.put(self.key(str(i)),b'12345','training'))
                self.assertFalse(cache.put(self.key('other-eval'),b'12345','development'))
            self.assertIsNotNone(cache.get(self.key('train1'),'training'))
            self.assertIsNotNone(cache.get(self.key('train2'),'training'))
            self.assertEqual(cache.evictions,1);cache.close()

    def test_old_role_promoted_and_free_space_blocks_optional_writes(self):
        with tempfile.TemporaryDirectory() as d:
            cache=DiskLRU(d,10,protect_training=True,minimum_free_bytes=10)
            with patch('applications.rag.top8_cache_v002.shutil.disk_usage') as usage:
                usage.return_value.free=100
                cache.put(self.key('old'),b'12345')
                cache.get(self.key('old'),'training')
                self.assertEqual(cache.entries[self.key('old')]['role'],'training')
                usage.return_value.free=12
                self.assertFalse(cache.put(self.key('eval'),b'12345','development'))
            self.assertIsNotNone(cache.get(self.key('old')));cache.close()

    def test_bad_capacity_or_inventory_rejected(self):
        for value in ('bad',-1):
            with self.assertRaises(ValueError):cache_capacity(value)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scratch=root/'bounded-sequences-v2';scratch.mkdir()
            (scratch/'lru.json').write_text(json.dumps({'bad':dict(name='../other',size=9)}))
            with self.assertRaisesRegex(RuntimeError,'inventory entry'):owned_scratch_bytes(root)


class HotPrefixPolicy(unittest.TestCase):
    def test_eval_replay_uses_bounded_immutable_hot_bytes(self):
        import torch
        from applications.rag.top8_cache_v002 import FrozenSequences, sequence_signature
        from applications.rag.tests.test_top8_cache_numerics import FakeBackbone,populations
        from applications.rag.top8_backbone_v001 import extract
        from applications.rag.top8_portable_v001 import load_cache
        with tempfile.TemporaryDirectory() as d,patch('applications.rag.top8_backbone_v001.Backbone',FakeBackbone):
            root=Path(d);train,evaluation=populations(root)
            file=extract('dpr',root,train,evaluation,root/'cache',label_classes=2)
            manifest,_=load_cache(file);reader=FrozenSequences(file.parent,manifest,maximum=0)
            entry=manifest['rows'][-1]['sequences'][0]
            try:
                expected=reader.read(entry,'development')
                original=expected['hidden'].clone();expected['hidden'].zero_()
                again=reader.read(entry,'development')
                self.assertTrue(torch.equal(again['hidden'],original))
                self.assertEqual(reader.regenerated,1);self.assertEqual(reader.memory_hits,1)
                self.assertEqual(sequence_signature(again['hidden'],again['mask'],again['native_scores']),entry['signature'])
                self.assertFalse(list((file.parent/'bounded-sequences-v2').glob('*.bin')))
                with patch('applications.rag.top8_cache_v002.SEQUENCE_MEMORY_BYTES',1):
                    reader.memory.clear();reader.memory_bytes=0
                    reader.read(entry,'development');reader.read(entry,'development')
                    self.assertEqual(reader.regenerated,3)
                    self.assertEqual(reader.memory_bytes,0)
                self.assertGreater(reader.stats()['host_regeneration_seconds'],0)
                self.assertEqual(reader.stats()['reads'],4)
            finally:reader.close()


if __name__=='__main__':unittest.main()
