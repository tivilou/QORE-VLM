"""Portable research CLI display; stdlib only, no process or training ownership.

Display/rate primitives adapted from the QKAN-Mamba CLI dashboard. Copy this
module into a project's tools directory and feed already-owned runner snapshots.
"""
from collections import deque
import datetime as dt
import math
import os
import shutil
import sys
import time
import unicodedata

def fit_line(text, columns):
    """Bound terminal cells, including Chinese, so redraw never wraps/scrolls."""
    output=[];used=0;limit=max(1,columns-1)
    for char in text:
        if ord(char)<32 or ord(char)==127: continue
        width=0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in 'WF' else 1)
        if used+width>limit:
            while output and used>limit-1:
                removed=output.pop();used-=0 if unicodedata.combining(removed) else (2 if unicodedata.east_asian_width(removed) in 'WF' else 1)
            return ''.join(output)+'…'
        output.append(char);used+=width
    return ''.join(output)


class Terminal:
    def __init__(self, mode):
        if mode not in ('auto', 'inplace', 'append'): raise ValueError('Unknown display mode')
        self.live=mode!='append' and sys.stdout.isatty() and os.environ.get('TERM','')!='dumb'
        self.entered=False

    def show(self, text):
        if not self.live:
            print(text,flush=True);return
        if not self.entered:
            sys.stdout.write('\x1b[?1049h\x1b[?25l')
            self.entered=True
        size=shutil.get_terminal_size(fallback=(100,30))
        lines=text.strip('\n').splitlines();height=max(1,size.lines-1)
        if len(lines)>height:
            lines=lines[:max(0,height-1)]+['窗口高度不足；放大窗口，或使用 --display append 查看完整快照。']
        frame='\n'.join(fit_line(line,size.columns) for line in lines)
        sys.stdout.write('\x1b[H\x1b[J'+frame);sys.stdout.flush()

    def close(self):
        if self.entered:
            sys.stdout.write('\x1b[?25h\x1b[?1049l');sys.stdout.flush()
            self.entered=False

    def __enter__(self): return self

    def __exit__(self, *error): self.close()


def duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds<0: return '待测'
    seconds=int(seconds);hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class Throughput:
    """Count committed work against worker timestamps, independently of redraws."""
    def __init__(self,window=60.):
        self.window=window;self.points=deque(maxlen=256);self.origin=None;self.scope='本次观察'
        self.short=None

    def observe(self,count,stamp,now,origin=None):
        if (not isinstance(count,int) or count<0 or not isinstance(stamp,(int,float))
                or not math.isfinite(stamp) or stamp>now+1): return self.result(now)
        if self.points and stamp<self.points[-1][0]: return self.result(now)
        if self.points and count<self.points[-1][1]:
            self.points.clear();self.origin=None;self.short=None;self.scope='本次观察'
        if not self.points:
            self.origin=(stamp,count)
            if (origin and isinstance(origin[0],(int,float)) and math.isfinite(origin[0])
                    and isinstance(origin[1],int) and 0<=origin[1]<=count and origin[0]<=stamp):
                self.origin=origin;self.scope='本次进程'
        if not self.points or stamp>self.points[-1][0]:
            self.points.append((stamp,count))
            cutoff=stamp-self.window
            while len(self.points)>2 and self.points[1][0]<=cutoff: self.points.popleft()
            elapsed=stamp-self.points[0][0]
            if elapsed>0: self.short=(count-self.points[0][1])/elapsed
        return self.result(now)

    def result(self,now):
        if not self.points: return dict(short=None,average=None,scope=self.scope,age=None)
        stamp,count=self.points[-1];elapsed=now-self.origin[0]
        # Repeated redraws do not erase short throughput. Average includes waits.
        average=(count-self.origin[1])/elapsed if elapsed>0 and (count>self.origin[1] or self.short is not None) else None
        return dict(short=self.short,average=average,scope=self.scope,age=max(0,now-stamp))


def speed_text(value,unit,precision=2):
    return f'{value:.{precision}f} {unit}' if value is not None else '采集中'


def speed_lines(stats,unit,remaining,precision=2):
    stale=stats['age'] is not None and stats['age']>120
    label='最近测量，阶段进度久未更新' if stale else '近60s平滑，窗口逐步积累'
    lines=[f"  短期平滑速度 {speed_text(stats['short'],unit,precision)}（{label}）",
           f"  长期平均速度 {speed_text(stats['average'],unit,precision)}（{stats['scope']}，含等待/保存）"]
    average=stats['average']
    eta=remaining/average if average and remaining>=0 and not stale else None
    estimate='暂停估计（阶段进度久未更新）' if stale else (duration(eta) if eta is not None else '采集中')
    lines.append('  当前计数阶段剩余 ETA '+estimate+'（按长期平均；不含后续验证/测试/回传及排队任务）')
    return lines


class Dashboard:
    """Render in-memory snapshots; the runner retains signals, IO and scheduling."""
    def __init__(self): self.rates = {}

    def render(self, snapshot, now=None):
        now = time.time() if now is None else now
        jobs = snapshot.get('jobs', [])
        groups = {name: [] for name in ('computed', 'running', 'pending', 'failed', 'paused', 'unknown')}
        for job in jobs:
            state = job.get('state', 'unknown')
            groups[state if state in groups else 'unknown'].append(str(job['id']))
        clock = dt.datetime.fromtimestamp(now, dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S %Z')
        lines = [f"[{clock}] 实验：{snapshot.get('name', '未命名')}",
            f"总计 {len(jobs)} | 完成 {len(groups['computed'])} | 运行 {len(groups['running'])} | "
            f"待跑 {len(groups['pending'])} | 失败 {len(groups['failed'])} | 暂停 {len(groups['paused'])}"]
        if groups['unknown']: lines.append('状态未知：'+', '.join(groups['unknown']))
        lines.append('模式：'+('只读观察（不启动或恢复任务）' if snapshot.get('watch_only') else '训练与显示一体'))
        # Values are supplied by the runner; this module never probes a GPU or PID.
        for device in snapshot.get('devices', []):
            lines.append(f"设备 {device['id']} | {device.get('summary', '遥测待测')}")
        if snapshot.get('phase'): lines.append('入口阶段：'+str(snapshot['phase']))
        for job in jobs:
            if job.get('state') != 'running': continue
            started = job.get('started_at'); stamp = job.get('progress_at')
            fresh_process = (isinstance(started,(int,float)) and math.isfinite(started)
                and isinstance(stamp,(int,float)) and math.isfinite(stamp) and started <= stamp <= now+1)
            lines.append(f"任务 {job['id']}")
            lines.append(f"  {job.get('device', '设备待测')} | PID {job.get('pid', '?')} | "
                f"本次耗时 {duration(now-job.get('attempt_started_at',started)) if isinstance(started,(int,float)) else '待测'}")
            lines.append('  阶段：'+str(job.get('phase', '待测')))
            if not fresh_process:
                lines.append('  初始化/恢复中，等待本次进程进度；速度与 ETA 待测。')
                continue
            count = job.get('completed'); total = job.get('total'); unit = job.get('unit', 'steps')
            valid = type(count) is int and count >= 0
            known_total = type(total) is int and total >= 0
            percent = f' ({100*count/total:.2f}%)' if valid and known_total and total > 0 else ''
            lines.append(f"  进度 {count if valid else '待测'}/{total if known_total else '待测'} {unit}{percent}"+
                (' | '+str(job['position']) if job.get('position') else ''))
            if job.get('metrics'):
                lines.append('  '+' | '.join(f'{k} {v}' for k,v in job['metrics'].items()))
            lines.append(f'  阶段进度 {int(max(0, now-stamp))}s 前')
            if job.get('execution'): lines.append('  实际执行：'+str(job['execution']))
            if valid:
                # Identity must change on every fresh worker attempt; no inherited speed.
                key = (str(job['id']), str(job.get('attempt', job.get('pid'))), started, unit)
                meter = self.rates.setdefault(key, Throughput())
                stats = meter.observe(count, stamp, now, origin=(started,job.get('run_completed_start')))
                lines.extend(speed_lines(stats, unit+'/s', max(0,total-count) if known_total else -1))
        for label,state in (('已完成','computed'),('排队','pending'),('失败','failed'),('暂停','paused')):
            lines.append(label+'：'+(', '.join(groups[state]) or '无'))
        lines.append('检查点：'+str(snapshot.get('checkpoint_status', '待测；不由显示心跳推断')))
        lines.append('回传：'+str(snapshot.get('return_status', '待测')))
        lines.append('日志：'+str(snapshot.get('log_path', '待测')))
        # Bound historical meters when a scheduler serves many fresh attempts.
        if len(self.rates) > max(32,4*len(jobs)):
            active = {(str(j['id']),str(j.get('attempt',j.get('pid'))),j.get('started_at'),j.get('unit','steps'))
                      for j in jobs if j.get('state') == 'running'}
            self.rates = {k:v for k,v in self.rates.items() if k in active}
        return '\n'.join(lines)


def demo():
    """Synthetic preview only. No worker, device query, file write or sleep."""
    now=time.time(); snapshot=dict(name='研究实验 CLI 模板（演示）',
        devices=[dict(id='GPU 0',summary='利用率 92% | 显存 8.2/24.0 GiB（演示数据）')],
        checkpoint_status='最近完整状态已保存（演示）',return_status='等待计算完成',log_path='logs/runner.log',
        jobs=[dict(id='model-dataset-seed',state='running',pid=1234,device='GPU 0',attempt='demo',
            started_at=now-100,progress_at=now,run_completed_start=1000,completed=2000,total=10000,
            unit='samples',phase='train',position='Epoch 2/10',metrics={'loss':0.25},execution='BF16（演示）'),
            dict(id='next-task',state='pending')])
    with Terminal('auto') as terminal: terminal.show(Dashboard().render(snapshot,now))


if __name__ == '__main__': demo()
