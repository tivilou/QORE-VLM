"""Lossless field-only CPU maps and once-per-cache training normalization.

No model, optimizer, order, dtype or GPU computation changes. Original shards
remain immutable. New column files are optional, SHA-bound derived artifacts.
"""
from __future__ import annotations
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
from applications.rag.qvalue_portable_v001 import need, digest, checked_member, json_file, tensor_file
from applications.rag.top8_cache_v002 import _json_commit


def required_fields(idea, arm):
    # readout is also required by the unchanged first-two-question null check.
    fields = ['readout']
    if idea == 'MIXED_DEPTH':
        fields += ['depths']
        if arm not in ('HEAD', 'CONCAT'): fields += ['mixed']
    elif idea == 'VALUE_GEOMETRY':
        fields += ['depths'] if arm in ('HEAD', 'CONCAT', 'MLP') else ['context', 'residual', 'final']
    elif idea in ('LOCAL_FFN', 'LOCAL_REUPLOAD') and arm != 'HEAD':
        fields += ['z', 'ffn_residual']
    return tuple(fields)


class ProjectedRows(Mapping):
    def __init__(self, source, fields, bank=None):
        self.source=source; self.fields=tuple(fields); self.bank=bank
    def __iter__(self): return iter(self.source)
    def __len__(self): return len(self.source)
    def __getitem__(self, key):
        if self.bank is None:
            row=self.source[key]
            return {name:row[name] for name in self.fields}
        result={name:self.bank.row(name,key) for name in self.fields if name!='mixed'}
        if 'mixed' in self.fields:
            result['mixed']=self.source.aux[key]['mixed']
        return result


def _generation(root, stem, suffix):
    for i in range(10000):
        name=f'{stem}-g{i:04d}{suffix}'
        if not (root/name).exists() and not (root/(name+'.part')).exists(): return name
    raise RuntimeError('derived feature generation limit')


class FieldBank:
    def __init__(self, source, manifest, cache_root, checkpoint):
        self.source=source; self.rows=manifest['rows']; self.checkpoint=checkpoint
        self.cache_root=Path(cache_root)
        self.identity=digest(self.cache_root/'cache.json')
        self.root=self.cache_root/'feature-columns-v3'/self.identity
        self.maps={}; self.indices={}; self.statistics=None
        self.metrics=dict(source_questions_read=0, columns_built=0, statistics_computed=0,
                          column_payload_bytes=0)

    def ensure(self, names):
        import torch
        from applications.rag.top8_cache_v002 import DiskLRU
        names=[n for n in names if n!='mixed' and n not in self.maps]
        need(all(n in ('readout','depths','context','residual','final','z','ffn_residual') for n in names),
             'only registered feature columns')
        if not names:return
        self.root.mkdir(parents=True,exist_ok=True)
        # Only the derived namespace is locked; legacy caches are never edited.
        lock=DiskLRU(self.root/'writer',0)
        streams={}; pending={}; descriptions={}
        try:
            for name in names:
                receipt=self.root/(name+'.json')
                if receipt.exists():continue
                target=_generation(self.root,name,'.f32')
                p=self.root/(target+'.part')
                streams[name]=p.open('xb');pending[name]=(target,p)
                descriptions[name]=dict(cache_sha256=self.identity,field=name,dtype='float32',rows={},elements=0)
            if streams:
                for row in self.rows:
                    self.checkpoint(); value=self.source[row['id']]
                    self.metrics['source_questions_read']+=1
                    for name,stream in streams.items():
                        x=value[name]
                        need(x.dtype==torch.float32 and x.device.type=='cpu' and
                             len(x)==len(row['retained_indices']) and bool(torch.isfinite(x).all()),
                             'complete finite original FP32 field')
                        x=x.contiguous(); desc=descriptions[name]
                        desc['rows'][row['id']]=dict(offset=desc['elements'],shape=list(x.shape))
                        desc['elements']+=x.numel();stream.write(x.numpy().tobytes())
                for name,stream in streams.items():
                    self.checkpoint();stream.flush();os.fsync(stream.fileno());stream.close()
                    target,p=pending[name]; os.link(p,self.root/target);p.unlink()
                    if os.name!='nt':
                        fd=os.open(self.root,os.O_DIRECTORY)
                        try:os.fsync(fd)
                        finally:os.close(fd)
                    file=self.root/target
                    record=dict(name=target,size=file.stat().st_size,sha256=digest(file))
                    need(record['size']==4*descriptions[name]['elements'],'complete column size')
                    checked_member(self.root,record)
                    _json_commit(self.root,name+'.json',dict(descriptions[name],file=record))
                    self.metrics['columns_built']+=1
            for name in names:
                desc=json.loads((self.root/(name+'.json')).read_bytes())
                need(desc['cache_sha256']==self.identity and desc['field']==name and desc['dtype']=='float32',
                     'derived column exact source binding')
                expected={r['id']:len(r['retained_indices']) for r in self.rows}
                need(set(desc['rows'])==set(expected),'complete column population')
                offset=0
                for r in self.rows:
                    part=desc['rows'][r['id']];shape=part['shape']
                    need(shape and shape[0]==expected[r['id']] and all(type(x)is int and x>=0 for x in shape)
                         and part['offset']==offset,'ordered complete column slices')
                    offset+=math.prod(shape)
                need(offset==desc['elements'] and desc['file']['size']==4*offset,'exhausted column layout')
                file=checked_member(self.root,desc['file'])
                self.maps[name]=torch.from_file(str(file),shared=False,size=offset,dtype=torch.float32) if offset else torch.empty(0)
                self.indices[name]=desc['rows'];self.metrics['column_payload_bytes']+=desc['file']['size']
        finally:
            for stream in streams.values():
                if not stream.closed:stream.close()
            lock.close()

    def row(self, name, key):
        part=self.indices[name][key]; n=math.prod(part['shape']); start=part['offset']
        # A private row copy prevents callers from mutating the mapped source.
        return self.maps[name][start:start+n].reshape(part['shape']).clone()

    def moments(self):
        import torch
        from applications.rag.top8_cache_v002 import DiskLRU
        if self.statistics is not None:return tuple(x.clone() for x in self.statistics)
        self.ensure(('readout',))
        training=[r for r in self.rows if r['role']=='training']
        identity=dict(cache_sha256=self.identity,training_ids=[r['id'] for r in training],
                      algorithm='original_cpu_cat_mean_std_unbiased_false_v1')
        lock=DiskLRU(self.root/'statistics-writer',0)
        try:
            receipt=self.root/'training-statistics.json'
            if receipt.exists():
                meta=json.loads(receipt.read_bytes());need(meta['binding']==identity,'training-only normalization binding')
                saved=torch.load(checked_member(self.root,meta['tensor']),map_location='cpu',weights_only=True)
                mean,std=saved['mean'],saved['std']
            else:
                self.checkpoint()
                # Exactly the previous CPU concatenation/reduction order, NOT a streaming approximation.
                values=torch.cat([self.row('readout',r['id']) for r in training])
                mean=values.mean(0);std=values.std(0,unbiased=False)
                std=torch.where(std>1e-8,std,torch.ones_like(std))
                self.checkpoint()
                name=_generation(self.root,'training-statistics','.bin')
                record=tensor_file(self.root,name,dict(mean=mean,std=std))
                _json_commit(self.root,receipt.name,dict(binding=identity,tensor=record))
                self.metrics['statistics_computed']+=1
            need(mean.dtype==std.dtype==torch.float32 and mean.shape==std.shape and mean.ndim==1
                 and bool(torch.isfinite(mean).all()) and bool(torch.isfinite(std).all()) and bool((std>0).all()),
                 'finite original FP32 normalization')
            self.statistics=(mean,std)
        finally:lock.close()
        return tuple(x.clone() for x in self.statistics)


_BANKS={}


def field_bank(features, manifest, root, checkpoint=lambda:None):
    from applications.rag.top8_cache_v002 import FeatureRows
    if not isinstance(features,FeatureRows):return None
    key=(str(Path(root).resolve()),id(features))
    if key not in _BANKS:_BANKS[key]=FieldBank(features,manifest,root,checkpoint)
    return _BANKS[key]


def project_features(features, manifest, root, fields, checkpoint=lambda:None):
    bank=field_bank(features,manifest,root,checkpoint)
    if bank is not None:bank.ensure(fields)
    return ProjectedRows(features,fields,bank)


def training_moments(features, manifest, root, checkpoint=lambda:None):
    bank=field_bank(features,manifest,root,checkpoint)
    if bank is not None:return bank.moments()
    import torch
    values=torch.cat([features[r['id']]['readout'] for r in manifest['rows'] if r['role']=='training'])
    mean=values.mean(0);std=values.std(0,unbiased=False)
    return mean,torch.where(std>1e-8,std,torch.ones_like(std))


def close_fields(root=None):
    prefix=None if root is None else str(Path(root).resolve())
    for key in list(_BANKS):
        if prefix is None or key[0]==prefix:del _BANKS[key]


def compatible_binding(previous, current, specification):
    """Exact approved old/new source pair only; every scientific/input option equal."""
    accepted=specification.get('accepted_previous_sources',[specification.get('before_sources')])
    need(previous.get('sources') in accepted and
         current.get('sources')==specification['after_sources'],
         'performance upgrade requires the exact reviewed source versions')
    need({k:v for k,v in previous.items() if k!='sources'}==
         {k:v for k,v in current.items() if k!='sources'},
         'performance upgrade cannot change inputs, models, seeds or training configuration')
    return previous


def resume_binding(previous, current, output, specification_path=None):
    if previous==current:return previous,None
    directory=Path(__file__).resolve().parents[2]/'configs/experiments/triviaqa_smallset_20261112'
    spec_path=Path(specification_path) if specification_path is not None else directory/'runtime-speed-compatibility.v002.json'
    if specification_path is None and not spec_path.exists():spec_path=directory/'feature-io-compatibility.v001.json'
    specification=json.loads(spec_path.read_bytes())
    selected=compatible_binding(previous,current,specification)
    version2=specification.get('schema')=='top8.runtime-speed-compatibility.v2'
    output=Path(output)
    receipt=dict(schema='top8.performance-continuation.v1',original_matrix_binding=previous,
                 actual_runtime_binding=current,compatibility_sha256=digest(spec_path),
                 changes=['field-only lossless IO','identical training statistics reused'],
                 original_bindings_and_completed_results_unchanged=True)
    name='feature-io-continuation.v1.json'
    if version2:
        name='runtime-speed-continuation.v2.json';receipt['schema']='top8.performance-continuation.v2'
        receipt['changes']+=['batched diagnostic transfers; original question backward order',
                             'FIDELITY HEAD readout-only original chunk shapes']
        legacy=output/'feature-io-continuation.v1.json'
        if legacy.exists():
            # An existing v1 audit stays immutable; certify its exact original
            # source pair, complete configuration and original receipt contents.
            old=json.loads(legacy.read_bytes());old_spec=directory/'feature-io-compatibility.v001.json'
            compatible_binding(previous,old['actual_runtime_binding'],json.loads(old_spec.read_bytes()))
            expected=dict(schema='top8.performance-continuation.v1',original_matrix_binding=previous,
                actual_runtime_binding=old['actual_runtime_binding'],compatibility_sha256=digest(old_spec),
                changes=['field-only lossless IO','identical training statistics reused'],
                original_bindings_and_completed_results_unchanged=True)
            need(old==expected,'exact immutable prior performance continuation')
            receipt['previous_continuation']=dict(name=legacy.name,size=legacy.stat().st_size,sha256=digest(legacy))
    path=output/name
    if path.exists():need(json.loads(path.read_bytes())==receipt,'same recorded performance continuation')
    elif not (output/'joint-prediction-seal.json').exists():_json_commit(output,name,receipt)
    return selected,path if path.exists() else None
