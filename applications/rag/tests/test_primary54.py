import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from applications.rag.top8_primary_v001 import adopt_primary,selected_arms,freeze_profile
from applications.rag.qvalue_portable_v001 import json_file
from scripts.smallset_progress import Progress
from scripts.smallset_resources import cgroup_memory,process_memory,resource_lines


class PrimaryTests(unittest.TestCase):
    def test_v6_keeps_original_binding_and_verifies_v5_receipt(self):
        from applications.rag.top8_feature_io_v003 import resume_binding
        directory=Path(__file__).resolve().parents[3]/'configs/experiments/triviaqa_smallset_20261112'
        five=json.loads((directory/'progress-ui-compatibility.v005.json').read_bytes())
        six=json.loads((directory/'primary54-compatibility.v006.json').read_bytes())
        original=dict(sources=six['accepted_previous_sources'][0],seeds=[7],epochs=4)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,p=resume_binding(original,dict(original,sources=five['after_sources']),root,directory/'progress-ui-compatibility.v005.json')
            old=p.read_bytes();current=dict(original,sources=six['after_sources'])
            selected,path=resume_binding(original,current,root)
            self.assertEqual(selected,original);self.assertEqual(p.read_bytes(),old)
            self.assertEqual(path.name,'primary54-continuation.v6.json')
            self.assertTrue(any(x['name']==p.name for x in json.loads(path.read_bytes())['previous_continuations']))
            with self.assertRaises(RuntimeError):resume_binding(original,dict(current,epochs=3),root)

    def test_roster_is_54_not_393_or_395(self):
        from scripts.run_triviaqa_smallset import main
        import contextlib,io
        for profile,total in [('primary54',54),('full',393)]:
            with contextlib.redirect_stdout(io.StringIO()) as stream:main(['--plan-only','--experiment-profile',profile])
            plan=json.loads(stream.getvalue())['matrix'];self.assertEqual(plan['learned_cells'],total)
        p=Progress('test','log');p.event(dict(qduet_progress=1,event='schedule',time=1,jobs=['q']))
        p.event(dict(qduet_progress=1,event='cell',time=2,job='historical-probe',state='computed'))
        self.assertEqual(set(p.jobs),{'q'})

    def test_adoption_preserves_complete_failed_and_checkpoint_without_changing_source(self):
        from applications.rag.top8_recovery_v001 import Journal
        from applications.rag.qvalue_portable_v001 import publish
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'old';dest=Path(d)/'new';source.mkdir();binding=dict(sources={'old':'sha'},seeds=[7])
            json_file(source,'run-binding.json',binding)
            complete=source/'q';complete.mkdir();record=json_file(complete,'predictions.json',dict(rows=[]))
            files=[dict(record,path='q/predictions.json')]
            json_file(complete,'disposition.json',dict(binding=binding,status='complete',files=files,result={}))
            failed=source/'f';failed.mkdir();json_file(failed,'disposition.json',dict(binding=binding,status='failed',failure={'kept':True}))
            journal=Journal(source/'r/recovery',dict(matrix=binding))
            tensor=publish(journal.root,'step-00000032-g0.bin',b'opaque')
            json_file(journal.root,'step-00000032.json',dict(step=32,binding=journal.binding,tensor=tensor))
            before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            adopt_primary(source,dest,['q','f','r']);adopt_primary(source,dest,['q','f','r'])
            self.assertEqual(json.loads((dest/'f/disposition.json').read_bytes())['status'],'failed')
            self.assertEqual(Journal(dest/'r/recovery',journal.binding).latest_record()['step'],32)
            self.assertEqual(before,{str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            freeze_profile(dest,'primary54',['q']);freeze_profile(dest,'primary54',['q'])
            with self.assertRaises(RuntimeError):freeze_profile(dest,'full',['q','f'])
            (source/'evaluation.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'after evaluation'):adopt_primary(source,Path(d)/'other',['q'])

    def test_resource_cgroup_process_group_and_missing_values(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);proc=root/'proc';cg=root/'cg';p=proc/'42';p.mkdir(parents=True);cg.mkdir()
            (p/'cgroup').write_text('0::/nested\n');(cg/'memory.current').write_text('4096');(cg/'memory.max').write_text('8192')
            (p/'stat').write_text('42 (name with spaces) R 1 42 42 0')
            (p/'status').write_text('VmRSS:\t4 kB\n')
            self.assertEqual(cgroup_memory(42,cg,proc),dict(used=4096,limit=8192))
            self.assertEqual(process_memory(42,proc),(4096,[42]))
            text='\n'.join(resource_lines(dict(sampled_at=1,process_tree_rss=None,gpus=[]),None,22))
            self.assertIn('待测',text);self.assertIn('陈旧',text)

    def test_primary_upload_excludes_old_matrix_and_keeps_new_seal(self):
        from scripts.run_junior_experiments import declared_artifacts
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);matrix=root/'matrix-primary54';matrix.mkdir()
            row=json_file(matrix,'predictions-dpr-NATIVE.json',dict(rows=[]))
            json_file(matrix,'joint-prediction-seal.json',dict(files=[row]))
            old=root/'matrix';old.mkdir();(old/'secret-old-log').write_text('unused')
            names={r['path'] for r in declared_artifacts(root,'matrix-primary54')}
            self.assertEqual(names,{'matrix-primary54/joint-prediction-seal.json','matrix-primary54/predictions-dpr-NATIVE.json'})


if __name__=='__main__':unittest.main()
