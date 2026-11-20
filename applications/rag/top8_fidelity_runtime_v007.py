"""Bounded device working set, shape-preserving batches and exact circuit algebra.

Only admitted workers import Torch. No labels, optimizer or population choices
live here. Batch only equal native padded shapes; never pad or truncate anew.
"""
from collections import OrderedDict
import hashlib
import json
import math
import time
import torch
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need


def memory_plan(device):
    if torch.device(device).type!='cuda':
        return dict(input_bytes=32*1024**2,projection_bytes=8*1024**2,
                    reference_bytes=4*1024**2,candidates=32)
    free,total=torch.cuda.mem_get_info(device)
    # Two independent scorers can coexist at final replay. Leave >=4GiB for
    # the frozen backbone, live graphs, allocator and unrelated allocations.
    budget=min(6*1024**3,total//3,max(0,free-4*1024**3))
    return dict(input_bytes=min(4*1024**3,budget*2//3),
                projection_bytes=budget//12,reference_bytes=budget//24,
                candidates=32 if total>=16*1024**3 else 16,
                observed_free=free,observed_total=total)


def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from tensors(v)
    elif isinstance(value,(tuple,list)):
        for v in value:yield from tensors(v)


class DeviceLRU:
    """Detached immutable tensors only; live graphs never enter the cache."""
    def __init__(self,maximum):
        self.maximum=maximum;self.bytes=0;self.items=OrderedDict();self.hits=0;self.misses=0
    def get(self,key):
        if key not in self.items:self.misses+=1;return None
        value,size,versions=self.items.pop(key)
        need(all(t._version==v and not t.requires_grad for t,v in versions),'immutable device cache tensor changed')
        self.items[key]=(value,size,versions);self.hits+=1;return value
    def put(self,key,value):
        ts=list(tensors(value));need(all(not t.requires_grad for t in ts),'cache cannot retain autograd graphs')
        size=sum(t.numel()*t.element_size() for t in ts)
        if size>self.maximum:return
        if key in self.items:self.bytes-=self.items.pop(key)[1]
        while self.items and self.bytes+size>self.maximum:self.bytes-=self.items.popitem(last=False)[1][1]
        self.items[key]=(value,size,[(t,t._version) for t in ts]);self.bytes+=size
    def clear(self):self.items.clear();self.bytes=0


class SequenceDeviceCache:
    """One reader-owned GPU cache and <=2 pinned <=32MiB staging buffers."""
    def __init__(self,device,maximum):
        self.device=torch.device(device);self.cache=DeviceLRU(maximum)
        self.stream=torch.cuda.Stream(device=device) if self.device.type=='cuda' else None
        self.staging=[];self.read_seconds=0.;self.transfers=0;self.prefetches=0
    def _key(self,entry):return hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest()
    def _release(self,wait=False):
        if wait:
            for event,_ in self.staging:event.synchronize()
            self.staging=[]
        else:self.staging=[(e,v) for e,v in self.staging if not e.query()]
    def read(self,reader,entry,role,prefetch=False):
        key=self._key(entry);value=self.cache.get(key)
        if value is not None:
            payload,event=value
            if event is not None:
                current=torch.cuda.current_stream(self.device);current.wait_event(event)
                for t in tensors(payload):t.record_stream(current)
            return payload
        # Do not regenerate a missing prefix speculatively. First-use native
        # reconstruction and full original SHA checks remain in the main reader.
        if prefetch and (self.cache.maximum==0 or key not in reader.memory and key not in reader.disk.entries):return None
        started=time.monotonic();cpu=reader.read(entry,role);self.read_seconds+=time.monotonic()-started
        size=sum(t.numel()*t.element_size() for t in tensors(cpu));self._release()
        asynchronous=self.stream is not None and size<=32*1024**2 and len(self.staging)<2
        if asynchronous:
            pinned={k:v.pin_memory() if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
            with torch.cuda.stream(self.stream):
                payload={k:v.to(self.device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in pinned.items()}
                event=torch.cuda.Event();event.record(self.stream)
            self.staging.append((event,pinned))
            if not prefetch:
                current=torch.cuda.current_stream(self.device);current.wait_event(event)
                for t in tensors(payload):t.record_stream(current)
        else:
            payload={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in cpu.items()};event=None
        self.cache.put(key,(payload,event));self.transfers+=1;self.prefetches+=int(prefetch)
        return payload
    def close(self):self._release(wait=True);self.cache.clear()
    def stats(self):
        return dict(device_input_bytes=self.cache.bytes,device_input_limit=self.cache.maximum,
                    device_input_hits=self.cache.hits,device_input_misses=self.cache.misses,
                    device_transfers=self.transfers,device_prefetches=self.prefetches,
                    reader_wait_seconds=self.read_seconds)


def shape_key(row):
    mask=row['mask'];b=len(row['hidden'])
    # Original batch-broadcast masks are expanded in B only, never sequence L.
    need(mask is None or mask.shape[0] in (1,b),'native mask batch dimension')
    return (tuple(row['hidden'].shape[1:]),str(row['hidden'].dtype),
            None if mask is None else (str(mask.dtype),tuple(mask.shape[1:])),
            tuple(sorted((k,tuple(v.shape[1:]),str(v.dtype)) for k,v in row.items()
                         if isinstance(v,torch.Tensor) and k not in ('hidden','mask'))))


def batch_rows(rows,maximum=32):
    """Exhaust every original chunk; combine consecutive equal native shapes."""
    pending=[];n=0;key=None
    def combine():
        if len(pending)==1:return pending[0]
        out={}
        for name,value in pending[0][0].items():
            if name=='mask':
                out[name]=None if value is None else torch.cat([
                    r[name].expand(len(r['hidden']),*r[name].shape[1:]) for r,_ in pending])
            elif isinstance(value,torch.Tensor):out[name]=torch.cat([r[name] for r,_ in pending])
        identities=[r.get('frozen_sequence_key') for r,_ in pending]
        if all(identities):out['frozen_sequence_key']=hashlib.sha256(json.dumps(identities).encode()).hexdigest()
        return out,torch.cat([native for _,native in pending])
    for row,native in rows:
        current=shape_key(row);size=len(row['hidden'])
        if pending and (current!=key or n+size>maximum):
            yield combine();pending=[];n=0
        pending.append((row,native));n+=size;key=current
    if pending:yield combine()


class CircuitKernel:
    """Compose twelve common 8x8 gates once, apply to every native Q/K state.

    Same q0-LSB RZ*RY encoding and controlled RY order as qfidelity_v002.
    Changes floating-point association, not the circuit or its parameters.
    """
    def __init__(self,kind,device):
        self.kind=kind;self.pending=None;self.calls=0
        indices=torch.arange(8,device=device);self.identity=torch.eye(8,device=device)
        self.encoding=[]
        for q in range(3):
            bit=(indices>>q)&1
            self.encoding.append((indices^(1<<q),2*bit-1))
        planes=((0,3),(1,6),(2,5),(0,7),(1,4),(3,6),
                (0,5),(1,7),(2,6),(3,4),(0,6),(2,7))
        self.gates=[]
        for layer in range(2):
            for control,target in [(None,0),(None,1),(None,2),(0,1),(1,2),(2,0)]:
                j=len(self.gates);partner=indices^(1<<target);active=torch.ones(8,device=device,dtype=torch.bool)
                if control is not None and kind!='PRODUCT':active=((indices>>control)&1).bool()
                if kind=='GIVENS':
                    a,b=planes[j];partner=indices.clone();partner[a]=b;partner[b]=a
                    active=(indices==a)|(indices==b);sign=torch.where(indices==a,-1,1)
                else:sign=2*((indices>>target)&1)-1
                self.gates.append((partner,active,sign))
    def flag(self,value):
        value=value.detach();self.pending=value if self.pending is None else self.pending&value
    def take_flags(self):value=self.pending;self.pending=None;return [] if value is None else [value]
    def encode(self,state,z):
        for q,(partner,sign) in enumerate(self.encoding):
            angle=z[...,q:q+1]/2
            state=(angle.cos()*state+angle.sin()*sign*state.index_select(-1,partner))*torch.exp(.5j*z[...,q:q+1]*sign)
        return state
    def states(self,effect,x,reference):
        p=lambda n:effect.parameter_value(n,reference)
        z=math.pi*torch.tanh(F.linear(F.layer_norm(x,(64,)),p('projection'),p('bias')))
        if self.kind=='MLP':
            raw=F.linear(torch.tanh(F.linear(z,p('w1'),p('b1'))),p('w2'),p('b2'))
            norm=torch.linalg.vector_norm(raw,dim=-1,keepdim=True);self.flag((norm>1e-8).all())
            return raw/norm
        state=torch.zeros(z.shape[:-1]+(8,),device=z.device,dtype=torch.complex64 if z.dtype==torch.float32 else torch.complex128)
        state[...,0]=1;state=self.encode(state,z)
        matrix=self.identity.to(state.dtype)
        for j,(partner,active,sign) in enumerate(self.gates):
            angle=p('theta')[j]/2
            matrix=torch.where(active,angle.cos(),1)*matrix+torch.where(active,angle.sin()*sign,0)*matrix.index_select(-1,partner)
        return self.encode(state@matrix,z)
    def kernel(self,effect,q,k,reference=False):
        self.calls+=1;s=self.states(effect,torch.cat((q.unsqueeze(2),k),2),reference)
        value=(s[:,:,0:1].conj()*s[:,:,1:]).sum(-1).abs().square()
        self.flag(torch.isfinite(value).all()&(value<=1.00002).all())
        return value
