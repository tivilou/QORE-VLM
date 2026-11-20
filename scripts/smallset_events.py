"""Optional display events only; no model, scheduler or filesystem ownership."""
import json
import os
import time


def emit(event, **fields):
    if os.environ.get('QDUET_PROGRESS_EVENTS') == '1':
        print(json.dumps(dict(qduet_progress=1,event=event,time=time.time(),pid=os.getpid(),**fields)),flush=True)


def jobs_for(plan, arms, fusion_arms):
    return [f'{kind}-{idea}-{arm}-s{seed}'
            for kind,ideas in plan['backbone_ideas'].items() for idea in ideas
            for seed in plan['seeds'] for arm in arms[idea]] + [
            f'fusion-{idea}-{arm}-s{seed}' for idea in plan['fusion_ideas']
            for seed in plan['seeds'] for arm in fusion_arms[idea]]
