"""Physical question batching; logical loss and optimizer accounting stay fixed.

Only forward/backward CUDA OOM is retried. AdamW is called by the owner after
this transaction succeeds. A failure inside AdamW uses durable journal recovery.
"""
from __future__ import annotations
import gc
import torch
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need


def merge_rows(rows):
    """Concatenate candidate axes, recursively, without mixing question losses."""
    need(bool(rows), 'nonempty physical batch')
    def merge(values):
        first=values[0]
        if isinstance(first,dict):
            need(all(isinstance(v,dict) and set(v)==set(first) for v in values),'common feature fields')
            return {k:merge([v[k] for v in values]) for k in first}
        need(all(isinstance(v,torch.Tensor) and v.ndim and v.shape[1:]==first.shape[1:]
                 and v.dtype==first.dtype and v.device==first.device for v in values),'compatible candidate features')
        return torch.cat(values,0)
    return merge(rows)


def question_losses(scores, targets, probabilities=None):
    losses=[];counts=[]
    for i,(s,y) in enumerate(zip(scores,targets)):
        need(s.ndim==1 and len(s)==len(y),'complete question score/target alignment')
        mask=y[:,None]>y[None,:];count=mask.sum()
        loss=(F.softplus(-(s[:,None]-s[None,:]))*mask).sum()/count.clamp_min(1)
        if probabilities is not None and len(y):
            loss=loss-probabilities[i].gather(-1,y[:,None]).squeeze(-1).clamp_min(1e-7).log().mean()
        losses.append(loss);counts.append(count)
    need(len(scores)==len(targets) and bool(losses),'complete physical batch losses')
    return losses,counts


def losses_for_questions(model,idea,records,features,root,checkpoint,device=None):
    """Keep native ragged sequence chunks; split scores before any pairwise loss."""
    from applications.rag.top8_portable_v001 import sequence_rows
    from applications.rag.top8_runtime_io_v004 import head_sequence_rows
    from applications.rag.top8_fidelity_runtime_v007 import batch_rows
    device=device or next(model.parameters()).device
    lengths=[len(r['retained_indices']) for r in records]
    targets=[torch.tensor(r['targets'],device=device,dtype=torch.long) for r in records]
    need(all(len(y)==n for y,n in zip(targets,lengths)),'original reviewed candidate count')
    probabilities=None
    if idea!='FIDELITY':
        def move(v):
            return {k:move(x) for k,x in v.items()} if isinstance(v,dict) else v.to(device)
        row=move(merge_rows([features[r['id']] for r in records]))
        s=model(row)
        if idea=='ORDINAL' and model.kind!='HEAD':
            _,p=model.inner.effect(row['readout'])
            if model.label_classes==2:p=torch.stack((p[:,0],p[:,1:].sum(-1)),-1)
            probabilities=list(p.split(lengths))
    else:
        def rows():
            for record in records:
                checkpoint();f=features[record['id']]
                yield from (head_sequence_rows(record,f,checkpoint) if model.kind=='HEAD'
                            else sequence_rows(root,record,f,checkpoint))
        source=rows()
        if getattr(model,'throughput',None) is not None:source=batch_rows(source,model.candidate_batch)
        parts=[model(row) for row,_ in source]
        s=torch.cat(parts) if parts else model.head.weight.sum().reshape(1)[:0]
    need(len(s)==sum(lengths),'exhausted all physical batch candidates')
    return question_losses(list(s.split(lengths)),targets,probabilities)


class Microbatch:
    """Finite physical tiers within an unchanged eight-question logical update."""
    def __init__(self,state=None):
        self.size=8;self.ooms=0;self.last_reason=None;self.last_groups=0;self.logical=0
        if state is not None:
            need(state.get('schema')=='top8.microbatch.v8' and state['physical_questions'] in (1,2,4,8),
                 'known microbatch recovery tier')
            self.size=state['physical_questions'];self.ooms=state['oom_retries'];self.last_reason=state['last_reason']
    def state(self):
        return dict(schema='top8.microbatch.v8',physical_questions=self.size,logical_questions=self.logical,
                    accumulation_groups=self.last_groups,oom_retries=self.ooms,last_reason=self.last_reason)
    def backward(self,records,losses_fn,parameters,optimizer,discard=lambda:None):
        need(0<len(records)<=8,'fixed logical update of at most eight questions')
        params=list(parameters);self.logical=len(records)
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if params[0].is_cuda else None
        # Modules used here are sample independent and frozen native layers are
        # eval-only. No optimizer or parameter mutation occurs inside losses_fn.
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                result=self._backward(records,losses_fn)
            except torch.cuda.OutOfMemoryError:
                pass
            else:
                return result
            # Outside the exception scope: its traceback no longer owns graphs.
            optimizer.zero_grad(set_to_none=True);discard();gc.collect()
            if cuda_rng is not None:torch.cuda.empty_cache()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            if self.size==1:raise torch.cuda.OutOfMemoryError('physical microbatch=1 exhausted; committed journal retained')
            self.size//=2;self.ooms+=1;self.last_reason='forward/backward CUDA OOM: whole logical update retried'
    def _backward(self,records,losses_fn):
        losses=[];counts=[];groups=0
        for start in range(0,len(records),self.size):
            part=records[start:start+self.size];ls,cs=losses_fn(part)
            need(len(ls)==len(cs)==len(part),'one normalized loss per question')
            # Includes zero-pair questions in the original logical denominator.
            torch.stack(ls).sum().div(len(records)).backward()
            losses.extend(x.detach() for x in ls);counts.extend(x.detach() for x in cs);groups+=1
        self.last_groups=groups
        return losses,counts


def discard_transients(model,reader=None):
    """Drop graph-free acceleration caches and diagnostics after failed backward."""
    if hasattr(model,'take_kernel_flags'):model.take_kernel_flags()
    if hasattr(model,'clear_references'):model.clear_references()
    cache=getattr(model,'_projection_cache',None)
    if cache is not None:cache.clear()
    cache=getattr(reader,'device_cache',None)
    if cache is not None:cache.close()
