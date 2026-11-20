"""Synthetic CLI/event lifecycle tests; no pretrained models, data or network."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from scripts.progress_ui import Dashboard, Terminal, Throughput, fit_line
from scripts.smallset_progress import Progress, render, supervise, watch


class ProgressTests(unittest.TestCase):
    def model(self):return Progress('attempt','runner.log')
    def event(self, m, event, stamp, **fields):
        m.event(dict(qduet_progress=1,event=event,time=stamp,pid=7,**fields))

    def test_resume_origin_phase_reset_and_no_fake_completion(self):
        m=self.model();self.event(m,'schedule',10,jobs=['a','b'])
        self.event(m,'cell',10,job='a',state='running')
        self.event(m,'progress',12,completed=100,total=200,origin=100,unit='steps',phase='训练')
        self.event(m,'progress',20,completed=140,total=200,unit='steps',phase='训练',checkpoint_step=140)
        text=render(Dashboard(),m.snapshot(20),20)
        self.assertIn('4.00 steps/s',text);self.assertIn('全套剩余 ETA 待测',text)
        self.assertIn('已提交 step 140',text)
        self.event(m,'progress',30,completed=0,total=2,unit='questions_checked',phase='回放未封存',origin=0)
        self.assertEqual(m.jobs['a']['state'],'running')
        self.assertEqual(m.jobs['a']['started_at'],30)
        self.event(m,'progress',40,completed=2,total=2,unit='questions_checked',phase='回放未封存')
        self.assertEqual(m.jobs['a']['state'],'running')
        self.event(m,'cell',41,job='a',state='computed')
        self.assertEqual(m.jobs['b']['state'],'pending')

    def test_outer_heartbeat_cannot_refresh_worker_or_checkpoint(self):
        m=self.model();self.event(m,'cell',1,job='a',state='running')
        self.event(m,'progress',2,completed=32,total=100,unit='steps',phase='训练',checkpoint_step=32)
        a=m.snapshot(10);b=m.snapshot(200)
        self.assertEqual(a['worker_event_at'],b['worker_event_at'])
        self.assertEqual(b['jobs'][0]['checkpoint_at'],2)
        self.assertIn('暂停估计',render(Dashboard(),b,200))
        self.assertEqual(b['jobs'][0]['state'],'running')

    def test_redraw_cannot_invent_rate_observations(self):
        m=Throughput();m.observe(100,10,10);m.observe(200,20,20)
        result=m.observe(200,20,30)
        self.assertEqual(result['short'],10);self.assertEqual(result['average'],5)

    def test_failure_and_interruption_states(self):
        for pause,state in ((False,'failed'),(True,'paused')):
            m=self.model();self.event(m,'cell',1,job='a',state='running');m.finish(1,pause)
            self.assertEqual(m.jobs['a']['state'],state)
            self.assertNotIn('入口完成门通过',m.result)

    def test_terminal_bounds_redirection_and_cleanup(self):
        class TTY(io.StringIO):
            def isatty(self):return True
        stream=TTY()
        with patch('sys.stdout',stream),patch.dict(os.environ,{'TERM':'xterm'}),\
             patch('shutil.get_terminal_size',return_value=os.terminal_size((30,8))):
            with self.assertRaises(ValueError),Terminal('auto') as term:
                term.show('研究进度'*100);raise ValueError('test')
        self.assertIn('\x1b[?25h',stream.getvalue());self.assertIn('\x1b[?1049l',stream.getvalue())
        self.assertLessEqual(len(fit_line('x'*100,30)),29)
        plain=io.StringIO()
        with patch('sys.stdout',plain),Terminal('auto') as term:term.show('append snapshot')
        self.assertNotIn('\x1b',plain.getvalue())

    def test_cli_dispatch_keeps_exact_workspace_and_plan_is_read_only(self):
        from scripts.run_triviaqa_smallset import main
        with tempfile.TemporaryDirectory() as d,patch('scripts.smallset_progress.supervise',return_value=17) as call:
            self.assertEqual(main(['--workspace',d,'--resume','--display','append']),17)
            cmd=call.call_args.args[0]
            self.assertEqual(cmd[1:3],['-u','-B']);self.assertEqual(Path(cmd[3]).name,'run_triviaqa_smallset.py')
            self.assertIn('--resume',cmd);self.assertEqual(cmd[-1],'--_progress-worker')
        with tempfile.TemporaryDirectory() as d:
            output=io.StringIO()
            with contextlib.redirect_stdout(output):main(['--workspace',d,'--plan-only'])
            plan=json.loads(output.getvalue())['matrix']
            self.assertEqual(plan['learned_cells'],54)
            import ast
            from scripts.smallset_events import jobs_for
            root=Path(__file__).resolve().parents[3]
            def registry(name,key):
                tree=ast.parse((root/name).read_text(encoding='utf-8'))
                return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==key)
            roster=jobs_for(plan,registry('applications/rag/top8_methods_v001.py','ARMS'),
                           registry('applications/rag/qduet_fusion_v001.py','FUSION_ARMS'))
            self.assertEqual(len(roster),54);self.assertEqual(len(set(roster)),54)
            self.assertFalse(list(Path(d).iterdir()))

    def test_actual_child_argv_logs_final_snapshot_and_watch_no_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);worker=root/'worker.py'
            worker.write_text("from scripts.smallset_events import emit\nprint('raw stdout')\nemit('schedule',jobs=['one'])\nemit('cell',job='one',state='running')\nemit('progress',completed=32,total=32,unit='steps',phase='training',checkpoint_step=32)\nemit('cell',job='one',state='computed')\n",encoding='utf-8')
            with patch.dict(os.environ,{'PYTHONPATH':str(Path.cwd())}),contextlib.redirect_stdout(io.StringIO()):
                code=supervise([sys.executable,'-u','-B',str(worker)],root,'append',.02)
            self.assertEqual(code,0)
            latest=root/'progress/latest.json';snapshot=json.loads(latest.read_bytes())
            self.assertEqual(snapshot['jobs'][0]['state'],'computed')
            log=Path(snapshot['log_path']);self.assertIn('raw stdout',log.read_text())
            self.assertTrue((log.parent/'jobs/one.log').is_file())
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(watch(root,'append',.01),0)
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_nonzero_and_reader_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(supervise([sys.executable,'-B','-c','raise SystemExit(7)'],d,'append',.01),7)
            self.assertIn('实际 exit 7',json.loads((Path(d)/'progress/latest.json').read_bytes())['return_status'])
        children=[];real=subprocess.Popen
        def launch(*a,**kw):
            p=real(*a,**kw);children.append(p);return p
        def bad_reader(stream,log,events,folder):events.put(OSError('reader fixture failure'))
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()),\
             patch('scripts.smallset_progress.subprocess.Popen',side_effect=launch),\
             patch('scripts.smallset_progress.read_events',side_effect=bad_reader):
            with self.assertRaisesRegex(OSError,'reader fixture'):
                supervise([sys.executable,'-B','-c','import time;time.sleep(30)'],d,'append',.01)
        self.assertIsNotNone(children[0].poll())

    def test_zero_worker_exit_with_pending_jobs_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PYTHONPATH':str(Path.cwd())}),\
             contextlib.redirect_stdout(io.StringIO()):
            code="from scripts.smallset_events import emit;emit('schedule',jobs=['not_done'])"
            self.assertEqual(supervise([sys.executable,'-B','-c',code],d,'append',.01),2)
            result=json.loads((Path(d)/'progress/latest.json').read_bytes())
            self.assertIn('worker实际 exit 0',result['return_status'])
            self.assertNotIn('入口完成门通过',result['return_status'])

    def test_broken_console_terminates_owned_child(self):
        children=[];real=subprocess.Popen
        def launch(*a,**kw):
            p=real(*a,**kw);children.append(p);return p
        with tempfile.TemporaryDirectory() as d,patch('scripts.smallset_progress.subprocess.Popen',side_effect=launch),\
             patch('scripts.smallset_progress.Terminal.show',side_effect=BrokenPipeError('closed console')):
            with self.assertRaises(BrokenPipeError):
                supervise([sys.executable,'-B','-c','import time;time.sleep(30)'],d,'append',.01)
        self.assertIsNotNone(children[0].poll())

    @unittest.skipIf(os.name=='nt','POSIX process-group synthetic check')
    def test_interrupt_reaps_owned_worker_and_descendant(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile=Path(d)/'descendant.pid'
            code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)"
            fired=False
            def frame(*a):
                nonlocal fired
                if pidfile.exists() and not fired:
                    fired=True;raise KeyboardInterrupt
            with patch('scripts.smallset_progress.Terminal.show',side_effect=frame):
                self.assertEqual(supervise([sys.executable,'-c',code,str(pidfile)],d,'append',.01),130)
            pid=int(pidfile.read_text());state=Path(f'/proc/{pid}/stat')
            if state.exists():self.assertEqual(state.read_text().split()[2],'Z')
            self.assertEqual(json.loads((Path(d)/'progress/latest.json').read_bytes())['phase'],'已中断')

    def test_progress_upgrade_retains_previous_receipt(self):
        from applications.rag.top8_feature_io_v003 import resume_binding
        directory=Path(__file__).resolve().parents[3]/'configs/experiments/triviaqa_smallset_20261112'
        four=directory/'cache-policy-compatibility.v004.json';old=json.loads(four.read_bytes())
        latest=json.loads((directory/'progress-ui-compatibility.v005.json').read_bytes())
        for ancestor in latest['accepted_previous_sources']:
            original=dict(sources=ancestor,seeds=[7],epochs=4,caches=[{'sha':'fixed'}])
            current=dict(original,sources=latest['after_sources'])
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);previous=None
                if ancestor in old['accepted_previous_sources']:
                    _,previous=resume_binding(original,dict(original,sources=old['after_sources']),root,four)
                    saved=previous.read_bytes()
                _,p=resume_binding(original,current,root)
                self.assertEqual(p.name,'progress-ui-continuation.v5.json')
                if previous:self.assertEqual(previous.read_bytes(),saved)
                with self.assertRaisesRegex(RuntimeError,'cannot change'):
                    resume_binding(original,dict(current,seeds=[8]),root)


if __name__=='__main__':unittest.main()
