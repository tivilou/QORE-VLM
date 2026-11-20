"""Mechanical scratch ownership, bounds and crash recovery; stdlib only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from applications.rag.top8_cache_v002 import DiskLRU, _json_commit


class BoundedStorage(unittest.TestCase):
    def key(self, name): return hashlib.sha256(name.encode()).hexdigest()

    def test_lru_limit_restart_and_no_outside_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); outside=root/'prefix-000000-00.bin';outside.write_bytes(b'legacy')
            cache=DiskLRU(root/'scratch',10)
            cache.put(self.key('a'),b'12345');cache.put(self.key('b'),b'56789')
            self.assertEqual(cache.get(self.key('a')),b'12345')
            cache.put(self.key('c'),b'78901')
            self.assertIsNone(cache.get(self.key('b')))
            self.assertLessEqual(sum(x['size'] for x in cache.entries.values()),10)
            cache.close()
            cache=DiskLRU(root/'scratch',5)
            self.assertLessEqual(sum(x['size'] for x in cache.entries.values()),5)
            self.assertEqual(outside.read_bytes(),b'legacy');cache.close()

    def test_corrupt_and_uncommitted_scratch_recomputed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=DiskLRU(root,20);key=self.key('a');cache.put(key,b'expected')
            (root/(key+'.bin')).write_bytes(b'altered')
            self.assertIsNone(cache.get(key));cache.close()
            (root/(self.key('uncommitted')+'.bin')).write_bytes(b'bytes without index')
            (root/(self.key('pending')+'.bin.part')).write_bytes(b'partial')
            cache=DiskLRU(root,20)
            self.assertFalse(list(root.glob('*.bin*')));cache.close()

    def test_zero_budget_and_oversize_blob_never_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            cache=DiskLRU(d,0);cache.put(self.key('x'),b'some bytes')
            self.assertIsNone(cache.get(self.key('x')))
            self.assertFalse(list(Path(d).glob('*.bin')));cache.close()

    def test_hits_batch_access_only_index_writes_and_close_flushes(self):
        with tempfile.TemporaryDirectory() as d, patch('applications.rag.top8_cache_v002.time.monotonic',return_value=0.):
            cache=DiskLRU(d,20);key=self.key('a');cache.put(key,b'expected')
            with patch.object(cache,'_save',wraps=cache._save) as save:
                for _ in range(200):self.assertEqual(cache.get(key),b'expected')
                self.assertEqual(save.call_count,0)
                cache.close();self.assertEqual(save.call_count,1)
            cache=DiskLRU(d,20)
            self.assertEqual(cache.get(key),b'expected');cache.close()

    def test_periodic_flush_corruption_and_failed_close_release_lock(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('applications.rag.top8_cache_v002.time.monotonic',return_value=0.):
                cache=DiskLRU(d,20);key=self.key('a');cache.put(key,b'expected')
            with patch('applications.rag.top8_cache_v002.time.monotonic',return_value=31.),\
                 patch.object(cache,'_save',wraps=cache._save) as save:
                self.assertEqual(cache.get(key),b'expected');self.assertEqual(save.call_count,1)
            (Path(d)/(key+'.bin')).write_bytes(b'corrupt')
            self.assertIsNone(cache.get(key))
            self.assertNotIn(key,json.loads((Path(d)/'lru.json').read_bytes()))
            cache.put(key,b'expected');cache.get(key)
            with patch.object(cache,'_save',side_effect=OSError('synthetic flush failure')):
                with self.assertRaisesRegex(OSError,'flush failure'):cache.close()
            self.assertTrue(cache.lock.closed)
            other=DiskLRU(d,20);self.assertEqual(other.get(key),b'expected');other.close()

    def test_writer_lock_and_corrupt_index_reject(self):
        with tempfile.TemporaryDirectory() as d:
            cache=DiskLRU(d,10)
            with self.assertRaisesRegex(RuntimeError,'already in use'):DiskLRU(d,10)
            cache.close()
            (Path(d)/'lru.json').write_text(json.dumps({'bad':dict(name='../other',size=1,sha256='bad')}))
            with self.assertRaisesRegex(RuntimeError,'owned entry'):DiskLRU(d,10)

    def test_interrupted_question_receipt_kept_and_new_receipt_committed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'question.json.part').write_bytes(b'{partial')
            _json_commit(root,'question.json',dict(complete=True))
            self.assertEqual((root/'question.json.interrupted-0').read_bytes(),b'{partial')
            self.assertEqual(json.loads((root/'question.json').read_bytes()),dict(complete=True))


if __name__=='__main__':unittest.main()
