"""Optional display events only; no model, scheduler or filesystem ownership."""
import json
import os
import time


def emit(event, **fields):
    if os.environ.get('QDUET_PROGRESS_EVENTS') == '1':
        print(json.dumps(dict(qduet_progress=1,event=event,time=time.time(),pid=os.getpid(),**fields)),flush=True)


def jobs_for(plan, arms, fusion_arms):
    if plan.get('experiment_profile')=='primary54':
        arms={k:('QUANTUM',) for k in arms};fusion_arms={k:('QUANTUM',) for k in fusion_arms}
    return [f'{kind}-{idea}-{arm}-s{seed}'
            for kind,ideas in plan['backbone_ideas'].items() for idea in ideas
            for seed in plan['seeds'] for arm in arms[idea]] + [
            f'fusion-{idea}-{arm}-s{seed}' for idea in plan['fusion_ideas']
            for seed in plan['seeds'] for arm in fusion_arms[idea]]


_allocator_at=0.
def allocator_event():
    global _allocator_at
    if os.environ.get('QDUET_PROGRESS_EVENTS')!='1' or time.monotonic()-_allocator_at<5:return
    import torch
    if not torch.cuda.is_initialized():return
    _allocator_at=time.monotonic()
    emit('allocator',allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
         peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
