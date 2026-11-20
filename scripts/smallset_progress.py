"""Local one-child CLI supervisor and read-only watch; no scientific imports."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from scripts.progress_ui import Dashboard, Terminal, duration


def atomic(path, value):
    part=path.with_suffix('.json.part')
    with part.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False);f.flush()
    os.replace(part,path)


class Progress:
    def __init__(self, attempt, log):
        self.attempt=attempt;self.jobs={};self.current=None;self.phase='启动／准备'
        self.log_path=str(log);self.checkpoint='待测；显示心跳不代表检查点已保存'
        self.result='等待完成';self.last_event=None;self.cache=None
        self.worker_exit_code=None
        self.resources=None;self.allocator=None;self.roster=None

    def event(self, e):
        if e.get('qduet_progress')!=1:return
        kind=e.get('event');stamp=e['time'];self.last_event=stamp
        if kind=='schedule':
            self.roster=set(e['jobs'])
            for key in e['jobs']:self.jobs.setdefault(key,dict(id=key,state='pending'))
        elif kind=='cell':
            if self.roster is not None and e['job'] not in self.roster:return
            key=e['job'];job=self.jobs.setdefault(key,dict(id=key))
            job.update(state=e['state'])
            if e['state']=='running':
                self.allocator=None
                self.current=key;job.update(started_at=stamp,attempt_started_at=stamp,pid=e['pid'],attempt=self.attempt,phase='加载／初始化',
                                            completed=None,total=None,progress_at=None)
            else:
                if self.current==key:self.current=None
        elif kind=='phase':
            self.phase=e['phase']
            if self.current:self.jobs[self.current]['phase']=e['phase']
        elif kind=='progress' and self.current:
            job=self.jobs[self.current]
            if job.get('unit')!=e['unit']:
                # Training, prediction and extraction have separate denominators
                # and clock origins. Never carry a training rate into evaluation.
                job.update(run_completed_start=e.get('origin',e['completed']))
                if job.get('unit') is not None:job['started_at']=stamp
            job.update(completed=e['completed'],total=e.get('total'),unit=e['unit'],progress_at=stamp,
                       phase=e['phase'],metrics=e.get('metrics',{}))
            self.phase=e['phase']
            if e.get('checkpoint_step') is not None:
                self.checkpoint=f"{self.current}: 已提交 step {e['checkpoint_step']}（本地）；云端持久化未认证"
                job['checkpoint_at']=stamp
        elif kind=='device' and self.current:
            self.jobs[self.current].update(device=e['device'],execution=e.get('execution','backend待测'))
        elif kind=='cache':self.cache=e['cache']
        elif kind=='allocator':self.allocator=e
        elif kind=='outcome':self.result=e['result']

    def finish(self, code, interrupted=False):
        if self.current:
            self.jobs[self.current]['state']='paused' if interrupted else 'failed'
            self.current=None
        if interrupted:
            self.phase='已中断';self.result='未宣称当前batch已保存；从最近完整检查点恢复'
        else:
            self.phase='入口已退出'
            self.result=f'worker实际 exit {self.worker_exit_code} / 界面入口 exit {code}；'+(
                '入口完成门通过' if code==0 else '未完成，检查完整日志；不自动重试')

    def snapshot(self, now=None, watch=False):
        now=time.time() if now is None else now
        return dict(schema='qduet.cli-progress.v1',name='TriviaQA · DPR / BGE · 8+2方案',
            attempt=self.attempt,watch_only=watch,phase=self.phase,jobs=list(self.jobs.values()),
            log_path=self.log_path,checkpoint_status=self.checkpoint,return_status=self.result,
            supervisor_heartbeat_at=now,worker_event_at=self.last_event,cache=self.cache,
            resources=self.resources,allocator=self.allocator)


def render(dashboard, snapshot, now=None):
    now=time.time() if now is None else now
    lines=dashboard.render(snapshot,now).splitlines()
    from scripts.smallset_resources import resource_lines
    lines[3:3]=resource_lines(snapshot.get('resources'),snapshot.get('allocator'),now)
    outer=snapshot.get('supervisor_heartbeat_at');worker=snapshot.get('worker_event_at')
    lines.insert(3,'父进程心跳 '+(duration(now-outer)+' 前' if outer is not None else '待测')+
                 ' | worker事件 '+(duration(now-worker)+' 前' if worker is not None else '待测')+
                 ' | 全套剩余 ETA 待测（不按组合数量外推）')
    for i,line in enumerate(lines):
        if line.startswith(('已计算：','排队：','失败：','暂停：')) and len(line)>190:
            names=line.split('：',1)[1].split(', ')
            lines[i]=line.split('：',1)[0]+f'（共{len(names)}项）：'+', '.join(names[:2])+' …'
    cache=snapshot.get('cache')
    if cache:
        fraction=cache.get('verified_hit_fraction')
        lines.insert(-3,'序列缓存（本reader累计）：命中 '+(f'{100*fraction:.1f}%' if fraction is not None else '待测')+
            ' | 重算 '+str(cache.get('regenerated_chunks'))+
            ' | host重算秒 '+str(round(cache.get('host_regeneration_seconds',0),1)))
    for job in snapshot.get('jobs',[]):
        if job.get('state')=='running' and job.get('checkpoint_at') is not None:
            lines.append('本地检查点提交于 '+duration(now-job['checkpoint_at'])+' 前（独立于父心跳）')
    return '\n'.join(lines)


def read_events(stream, log, events, job_directory):
    current=None;child_log=None
    try:
        with log.open('ab',buffering=0) as f:
            for raw in stream:
                f.write(raw)
                if child_log is not None:child_log.write(raw)
                try:e=json.loads(raw)
                except (ValueError,UnicodeDecodeError):continue
                if not isinstance(e,dict):continue
                if e.get('qduet_progress')!=1:
                    if e.get('stage') in ('sequence_cache','sequence_cache_final'):
                        e=dict(qduet_progress=1,event='cache',time=time.time(),cache=e)
                    elif e.get('stage')=='extract':
                        e=dict(qduet_progress=1,event='phase',time=time.time(),
                            phase=f"提取 {e['backbone']} {e['completed_questions']}/{e['total_questions']} 题（块已提交）")
                    else:continue
                events.put(e)
                if e.get('event')=='cell':
                    if child_log is not None:child_log.close();child_log=None
                    if e.get('state')=='running':
                        name=e['job']
                        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
                            raise ValueError('invalid progress job name')
                        child_log=(job_directory/(name+'.log')).open('ab',buffering=0);child_log.write(raw)
    except BaseException as error:events.put(error)
    finally:
        if child_log is not None:child_log.close()
        stream.close()


def live_group(group):
    """Linux ownership metadata only; zombies have no running computation."""
    proc=Path('/proc')
    if not proc.is_dir():return []
    live=[]
    for path in proc.glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group and fields[0] not in ('Z','X'):live.append(int(path.parent.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return live


def stop_child(child):
    # Own process group only. The existing worker resumes from its last committed
    # journal; this does not claim a safe save of the currently executing batch.
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
    else:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name!='nt':
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        else:child.kill()
        child.wait(timeout=5)
    if os.name!='nt':
        # wait() reaps only the direct worker. Account for its own process-group
        # descendants finishing signal/exit IO before reporting clean shutdown.
        deadline=time.monotonic()+5
        while live_group(child.pid):
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:break
            if time.monotonic()>=deadline:
                raise RuntimeError('Owned worker group still nonquiescent after cleanup; inspect logs')
            time.sleep(.02)


def supervise(command, workspace, display='auto', interval=5., cwd=None):
    progress=Path(workspace)/'progress';progress.mkdir(parents=True,exist_ok=True)
    attempt=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]
    folder=progress/attempt;folder.mkdir();jobs=folder/'jobs';jobs.mkdir()
    log=folder/'runner.log';model=Progress(attempt,log);dashboard=Dashboard();events=queue.Queue()
    latest=progress/'latest.json';child=None;reader=None;sampler=None;handlers={};interrupted=False;code=1
    # A second UI must never become a second owner of this same workspace.
    from applications.rag.top8_cache_v002 import DiskLRU
    lock=DiskLRU(progress/'owner',0)
    def interrupt(signum, frame):raise KeyboardInterrupt
    try:
        with Terminal(display) as terminal:
            try:
                env=dict(os.environ,QDUET_PROGRESS_EVENTS='1',PYTHONUNBUFFERED='1')
                child=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                       start_new_session=os.name!='nt')
                from scripts.smallset_resources import Sampler
                sampler=Sampler(child.pid)
                for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
                if hasattr(signal,'SIGHUP'):handlers[signal.SIGHUP]=signal.signal(signal.SIGHUP,interrupt)
                reader=threading.Thread(target=read_events,args=(child.stdout,log,events,jobs),daemon=True);reader.start()
                while True:
                    while True:
                        try:event=events.get_nowait()
                        except queue.Empty:break
                        if isinstance(event,BaseException):raise event
                        model.event(event)
                    model.resources=sampler.latest
                    snapshot=model.snapshot();atomic(latest,snapshot);terminal.show(render(dashboard,snapshot))
                    if child.poll() is not None and not reader.is_alive() and events.empty():
                        model.worker_exit_code=child.returncode;code=child.returncode
                        if code==0 and any(j.get('state')!='computed' for j in model.jobs.values()):code=2
                        break
                    time.sleep(interval)
            except KeyboardInterrupt:interrupted=True;code=130
            finally:
                # A second interrupt must not abandon children during cleanup.
                for sig in handlers:signal.signal(sig,signal.SIG_IGN)
                if child is not None:
                    stop_child(child)
                    if sampler is not None:sampler.close()
                    if reader is not None:reader.join(timeout=3)
                    if child.stdout is not None and (reader is None or not reader.is_alive()):child.stdout.close()
                model.finish(code,interrupted);snapshot=model.snapshot()
                atomic(latest,snapshot);atomic(folder/'final.json',snapshot)
                terminal.show(render(dashboard,snapshot))
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        lock.close()
    return code


def watch(workspace, display='auto', interval=5.):
    path=Path(workspace)/'progress/latest.json';dashboard=Dashboard()
    with Terminal(display) as terminal:
        try:
            while True:
                snapshot=json.loads(path.read_bytes());snapshot['watch_only']=True
                terminal.show(render(dashboard,snapshot))
                if snapshot.get('phase') in ('入口已退出','已中断'):return 0
                time.sleep(interval)
        except KeyboardInterrupt:return 0
