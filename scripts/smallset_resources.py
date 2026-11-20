"""Best-effort parent telemetry. No Torch import, CUDA context or synchronization."""
from pathlib import Path
import csv
import io
import os
import subprocess
import threading
import time


def number(value):
    try:return float(value.strip())
    except (ValueError,AttributeError):return None


def cgroup_memory(pid,root=Path('/sys/fs/cgroup'),proc=Path('/proc')):
    try:
        for line in (proc/str(pid)/'cgroup').read_text().splitlines():
            _,controllers,relative=line.split(':',2)
            if controllers and 'memory' not in controllers.split(','):continue
            base=root if not controllers else root/'memory'
            choices=[base/relative.lstrip('/'),base]
            for path in choices:
                current=path/('memory.current' if not controllers else 'memory.usage_in_bytes')
                maximum=path/('memory.max' if not controllers else 'memory.limit_in_bytes')
                if current.is_file():
                    used=int(current.read_text());text=maximum.read_text().strip()
                    limit=None if text=='max' else int(text)
                    if limit is not None and limit>=1<<60:limit=None
                    return dict(used=used,limit=limit)
    except (OSError,ValueError):pass
    return None


def process_memory(pgid,proc=Path('/proc')):
    rows=[]
    try:paths=proc.iterdir()
    except OSError:return None,[]
    for p in paths:
        if not p.name.isdigit():continue
        try:
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[2])!=pgid:continue
            status=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x)
            rss=int(status.get('VmRSS','0 kB').split()[0])*1024
            rows.append((int(p.name),rss))
        except (OSError,ValueError,IndexError):continue
    return (sum(rss for _,rss in rows) if rows else None),[pid for pid,_ in rows]


def query_gpu():
    command=['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=1.5)
        if result.returncode:return []
        return [dict(id=row[0].strip(),uuid=row[1].strip(),util=number(row[2]),
                     used=number(row[3]),total=number(row[4]))
                for row in csv.reader(io.StringIO(result.stdout)) if len(row)==5]
    except (OSError,subprocess.TimeoutExpired):return []


class Sampler:
    def __init__(self,pid,interval=5.):
        self.pid=pid;self.interval=interval;self.latest=None;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        while not self.stop.is_set():
            try:
                rss,pids=process_memory(self.pid)
                self.latest=dict(sampled_at=time.time(),process_tree_rss=rss,pids=pids,
                    cgroup=cgroup_memory(self.pid),gpus=query_gpu())
            except Exception:self.latest=dict(sampled_at=time.time(),error='resource sample unavailable')
            self.stop.wait(self.interval)
    def close(self):self.stop.set();self.thread.join(timeout=2.)


def gib(value):return '待测' if value is None else f'{value/1024**3:.2f} GiB'


def resource_lines(resource,allocator,now):
    if not resource:return ['资源采样：采集中（不初始化CUDA）']
    age=max(0,now-resource['sampled_at']);suffix=f' | 采样 {age:.0f}s 前'+('（陈旧）' if age>15 else '')
    lines=['内存：自有进程组 RSS合计 '+gib(resource.get('process_tree_rss'))+'（共享页可能重复）'+suffix]
    group=resource.get('cgroup')
    lines.append('容器内存：'+(gib(group['used'])+' / '+(gib(group['limit']) if group['limit'] is not None else '未设硬上限') if group else '待测'))
    for gpu in resource.get('gpus',[]):
        lines.append(f"GPU物理{gpu['id']}：整卡显存 "+gib(None if gpu['used'] is None else gpu['used']*1024**2)+' / '+
            gib(None if gpu['total'] is None else gpu['total']*1024**2)+
            ' | 整卡利用率 '+('待测' if gpu['util'] is None else f"{gpu['util']:.0f}%"))
    if not resource.get('gpus'):lines.append('整卡显存／利用率：待测（nvidia-smi不可用）')
    if allocator:
        age=max(0,now-allocator['time'])
        lines.append('当前worker显存：张量 '+gib(allocator.get('allocated'))+' | 预留 '+gib(allocator.get('reserved'))+
            ' | 本进程峰值 '+gib(allocator.get('peak_allocated'))+f' | 采样 {age:.0f}s 前'+('（陈旧）' if age>15 else ''))
    else:lines.append('当前worker张量／预留／峰值显存：待测')
    return lines
