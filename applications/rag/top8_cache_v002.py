"""Sharded immutable features and a bounded, disposable FP32 prefix working set.

No scientific imports at module import. Regenerated prefixes must match the
original extraction's complete tensor hashes, not just its final scalar scores.
"""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import time
from applications.rag.qvalue_portable_v001 import need, digest, checked_member, json_file, tensor_file

LAYOUT = 'question-shards-fp32-v2'
DEFAULT_SEQUENCE_BYTES = 32 * 1024**3
FEATURE_MEMORY_BYTES = 128 * 1024**2
SEQUENCE_MEMORY_BYTES = 256 * 1024**2
AUTO_MAXIMUM_BYTES = 256 * 1024**3
AUTO_GROWTH_BYTES = 48 * 1024**3


def cache_capacity(value):
    """CLI accepts auto or the existing explicit nonnegative GiB limit."""
    if str(value) == 'auto':return 'auto'
    try:amount = int(value)
    except (TypeError, ValueError):raise ValueError('sequence cache must be auto or nonnegative integer GiB')
    if amount < 0:raise ValueError('sequence cache must be auto or nonnegative integer GiB')
    return amount


def owned_scratch_bytes(root):
    """Count only indexed existing payloads; never count legacy/partial files."""
    root = Path(root)/'bounded-sequences-v2';index = root/'lru.json'
    if not index.exists():return 0
    need(not root.is_symlink() and not index.is_symlink(), 'owned scratch inventory')
    entries = json.loads(index.read_bytes());need(isinstance(entries, dict), 'scratch inventory index')
    total = 0
    for key, row in entries.items():
        need(re.fullmatch('[0-9a-f]{64}', key) is not None and row['name'] == key+'.bin'
             and type(row['size']) is int and row['size'] >= 0, 'owned scratch inventory entry')
        path = root/row['name'];need(not path.is_symlink(), 'owned scratch inventory payload')
        if path.is_file():
            size = path.stat().st_size
            need(size == row['size'], 'scratch inventory size')
            total += size
    return total


def sequence_capacity_plan(roots, requested='auto', minimum_free_bytes=40*1024**3):
    """One allocation per filesystem, after extraction; no reservation or writes.

    Auto leaves free-space headroom plus a stated growth allowance. Existing own
    scratch is added back so resuming a filled cache does not shrink it twice.
    Limits are caps, not preallocated files or a claim that all prefixes fit.
    """
    requested = cache_capacity(requested)
    need(minimum_free_bytes >= 0, 'nonnegative cache filesystem headroom')
    groups = {};limits = {};observations = []
    for root in sorted({str(Path(r).resolve()) for r in roots}):
        path = Path(root);groups.setdefault(path.stat().st_dev, []).append(path)
    for paths in groups.values():
        free = shutil.disk_usage(paths[0]).free
        existing = sum(owned_scratch_bytes(p) for p in paths)
        available = max(0, free+existing-minimum_free_bytes-AUTO_GROWTH_BYTES)
        cap = min(AUTO_MAXIMUM_BYTES, available//len(paths)) if requested == 'auto' else requested*1024**3
        for path in paths:limits[str(path)] = cap
        observations.append(dict(roots=[str(p) for p in paths],observed_free_bytes=free,
            existing_owned_payload_bytes=existing,per_model_limit_bytes=cap,
            minimum_free_bytes=minimum_free_bytes,growth_allowance_bytes=AUTO_GROWTH_BYTES,
            automatic=requested == 'auto'))
    return dict(stage='sequence_cache_policy',requested=requested,limits=limits,
                filesystems=observations,cpu_hot_limit_bytes_per_reader=SEQUENCE_MEMORY_BYTES,
                training_admission='retain admitted training chunks; evict evaluation first')


def tensor_hash(value):
    if value is None:
        return None
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def object_bytes(value):
    import torch
    out = io.BytesIO(); torch.save(value, out)
    return out.getvalue()


def load_tensor(raw):
    import torch
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


class DiskLRU:
    """Single-writer scratch namespace; only indexed, digest-named blobs evicted."""
    def __init__(self, root, maximum, protect_training=False, minimum_free_bytes=0):
        self.root = Path(root)
        need(maximum >= 0 and not self.root.is_symlink(), 'valid bounded sequence cache')
        self.maximum = maximum
        self.protect_training = protect_training;self.minimum_free_bytes = minimum_free_bytes
        self.evictions = self.skipped_writes = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root/'writer.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.lock.seek(0); self.lock.write(b'0'); self.lock.flush(); self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            self.lock.close(); raise RuntimeError('Sequence cache is already in use by another runner')
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        self._access_dirty = False
        self._last_access_save = time.monotonic()
        self.index_path = self.root/'lru.json'
        self.entries = json.loads(self.index_path.read_bytes()) if self.index_path.exists() else {}
        need(isinstance(self.entries, dict), 'sequence LRU index')
        for key, row in self.entries.items():
            need(re.fullmatch('[0-9a-f]{64}', key) is not None and
                 row['name'] == key+'.bin' and type(row['size']) is int and row['size'] >= 0 and
                 re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None,
                 'sequence LRU owned entry')
        # A killed write can leave one uncommitted scratch blob. Its name is
        # exclusive to this namespace; never touch legacy extraction files.
        for path in self.root.iterdir():
            if re.fullmatch(r'[0-9a-f]{64}\.bin(?:\.part)?', path.name):
                need(not path.is_symlink(), 'scratch file cannot be a symlink')
                if path.name.endswith('.part') or path.stem not in self.entries:
                    path.unlink()
        self._trim(0)
        self._save()

    def _save(self):
        temporary = self.root/'lru.json.part'
        with temporary.open('wb') as stream:
            stream.write(json.dumps(self.entries, sort_keys=True).encode())
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, self.index_path)
        self._access_dirty = False
        self._last_access_save = time.monotonic()

    def _trim(self, incoming, role=None):
        while self.entries and sum(x['size'] for x in self.entries.values())+incoming > self.maximum:
            # Pin the admitted training subset instead of cyclically evicting it
            # across epochs. Optional evaluation data cannot displace training.
            priority = {'development':0, 'unknown':1, 'training':2}
            eligible = [k for k,r in self.entries.items() if role is None or not self.protect_training
                        or priority.get(r.get('role','unknown'),1) < priority.get(role,1)]
            if not eligible:return False
            key = min(eligible, key=lambda k: (priority.get(self.entries[k].get('role','unknown'),1)
                      if self.protect_training else 0, self.entries[k]['used']))
            row = self.entries.pop(key); path = self.root/row['name']
            need(not path.is_symlink(), 'scratch eviction cannot follow a symlink')
            path.unlink(missing_ok=True);self.evictions += 1
        return sum(x['size'] for x in self.entries.values())+incoming <= self.maximum

    def get(self, key, role='unknown'):
        row = self.entries.get(key)
        if row is None:
            return None
        path = self.root/row['name']
        need(not path.is_symlink(), 'scratch reader cannot follow a symlink')
        raw = path.read_bytes() if path.is_file() else None
        if raw is None or len(raw) != row['size'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            self.entries.pop(key); path.unlink(missing_ok=True); self._save(); return None
        # Payload membership/bytes are durable on put. Losing only recent LRU
        # ages on a crash affects eviction order, never the accepted tensors.
        row['used'] = time.time_ns(); self._access_dirty = True
        if role == 'training' or row.get('role','unknown') == 'unknown':row['role'] = role
        if time.monotonic()-self._last_access_save >= 30:
            self._save()
        return raw

    def put(self, key, raw, role='unknown'):
        need(re.fullmatch('[0-9a-f]{64}', key) is not None, 'owned scratch key')
        need(role in ('training','development','unknown'), 'known sequence cache role')
        if key in self.entries:return True  # Published payloads are immutable.
        if len(raw) > self.maximum or (self.minimum_free_bytes and
                shutil.disk_usage(self.root).free-len(raw) < self.minimum_free_bytes):
            self.skipped_writes += 1;return False
        before = self.evictions
        if not self._trim(len(raw), role):
            if self.evictions != before:self._save()
            self.skipped_writes += 1;return False
        path = self.root/(key+'.bin'); temporary = self.root/(key+'.bin.part')
        need(not path.is_symlink() and not temporary.is_symlink(), 'safe scratch output')
        with temporary.open('wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.entries[key] = dict(name=path.name, size=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest(), used=time.time_ns(),role=role)
        self._save()
        return True

    def close(self):
        if not self.lock.closed:
            try:
                if getattr(self, '_access_dirty', False):self._save()
            finally:
                if os.name == 'nt':
                    import msvcrt
                    self.lock.seek(0); msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
                self.lock.close()


class FeatureRows(Mapping):
    """Read one whole question on demand; keep a bounded CPU tensor LRU."""
    def __init__(self, root, rows, maximum=FEATURE_MEMORY_BYTES):
        self.root = Path(root); self.rows = {r['id']:r for r in rows}
        need(len(self.rows) == len(rows), 'unique feature shard keys')
        self.maximum = maximum; self.resident = 0; self.cache = OrderedDict(); self.aux = {}

    def __iter__(self): return iter(self.rows)
    def __len__(self): return len(self.rows)

    def __getitem__(self, key):
        import torch
        if key in self.cache:
            value, size = self.cache.pop(key); self.cache[key] = (value, size)
        else:
            row = self.rows[key]
            value = torch.load(checked_member(self.root, row['feature_file']),
                               map_location='cpu', weights_only=True)
            n = len(row['retained_indices'])
            need(all(isinstance(x, torch.Tensor) and x.dtype == torch.float32 and len(x) == n and
                     bool(torch.isfinite(x).all()) for x in value.values()), 'finite FP32 feature shard')
            size = sum(x.numel()*x.element_size() for x in value.values())
            while self.cache and self.resident+size > self.maximum:
                _, (_, old) = self.cache.popitem(last=False); self.resident -= old
            if size <= self.maximum:
                self.cache[key] = (value, size); self.resident += size
        return dict(value, **self.aux.get(key, {}))

    def set_aux(self, key, name, value):
        need(key in self.rows, 'known auxiliary feature identity')
        self.aux.setdefault(key, {})[name] = value


def sequence_signature(hidden, mask, native):
    need(str(hidden.dtype)=='torch.float32' and str(native.dtype)=='torch.float32',
         'complete prefix and native scores stay FP32')
    return dict(hidden_sha256=tensor_hash(hidden), mask_sha256=tensor_hash(mask),
                native_sha256=tensor_hash(native), hidden_shape=list(hidden.shape),
                hidden_dtype=str(hidden.dtype),mask_dtype=None if mask is None else str(mask.dtype),
                mask_shape=None if mask is None else list(mask.shape),native_dtype=str(native.dtype))


class FrozenSequences:
    def __init__(self, root, manifest, checkpoint=lambda: None, maximum=None, minimum_free_bytes=0):
        self.root = Path(root); self.manifest = manifest; self.checkpoint = checkpoint
        self.disk = DiskLRU(self.root/'bounded-sequences-v2',
            manifest['sequence_cache_bytes'] if maximum is None else maximum,
            protect_training=True,minimum_free_bytes=minimum_free_bytes)
        self.backbone = None
        self.hits = self.legacy_hits = self.regenerated = 0
        self.memory = OrderedDict();self.memory_bytes = self.memory_hits = 0
        self.reads = 0;self.read_seconds = self.regeneration_seconds = 0.
        self._reported_at = time.monotonic()
        self._prefetch=None;self._prefetch_pool=None;self.prefetch_hits=0

    def prefetch(self,entry):
        """One <=32MiB immutable disk read ahead; no tensors/CUDA/RNG in thread."""
        if not entry.get('regenerate'):return
        key=hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest()
        if key in self.memory:return
        row=self.disk.entries.get(key)
        if row is None or row['size']>32*1024**2:return
        if self._prefetch is not None:
            _,old=self._prefetch
            if not old.done():return
        from concurrent.futures import ThreadPoolExecutor
        if self._prefetch_pool is None:self._prefetch_pool=ThreadPoolExecutor(max_workers=1)
        path=self.disk.root/row['name'];size=row['size'];sha=row['sha256']
        def read():
            try:
                if path.is_symlink():return None
                with path.open('rb') as f:raw=f.read(size+1)
                return raw if len(raw)==size and hashlib.sha256(raw).hexdigest()==sha else None
            except OSError:return None
        self._prefetch=(key,self._prefetch_pool.submit(read))

    def _valid(self, payload, entry):
        return sequence_signature(payload['hidden'], payload['mask'], payload['native_scores']) == entry['signature']

    def _remember(self, key, raw):
        # Immutable serialized bytes prevent a caller from modifying a cached
        # tensor. Deserialization and complete signature checks remain on hits.
        if len(raw) > SEQUENCE_MEMORY_BYTES:return
        if key in self.memory:self.memory_bytes -= len(self.memory.pop(key))
        while self.memory and self.memory_bytes+len(raw) > SEQUENCE_MEMORY_BYTES:
            _, old = self.memory.popitem(last=False);self.memory_bytes -= len(old)
        self.memory[key] = raw;self.memory_bytes += len(raw)

    def stats(self):
        hit_count = self.memory_hits+self.hits+self.legacy_hits
        return dict(backbone=self.manifest['backbone'],reads=self.reads,
            memory_hits=self.memory_hits,working_set_hits=self.hits,verified_legacy_hits=self.legacy_hits,
            regenerated_chunks=self.regenerated,verified_hit_fraction=hit_count/self.reads if self.reads else None,
            host_read_seconds=self.read_seconds,host_regeneration_seconds=self.regeneration_seconds,
            sequence_payload_limit_bytes=self.disk.maximum,
            sequence_payload_bytes=sum(r['size'] for r in self.disk.entries.values()),
            cpu_hot_payload_bytes=self.memory_bytes,cpu_hot_limit_bytes=SEQUENCE_MEMORY_BYTES,
            prefetch_hits=self.prefetch_hits,prefetch_max_bytes=32*1024**2,
            evicted_chunks=self.disk.evictions,skipped_optional_writes=self.disk.skipped_writes)

    def read(self, entry, role='unknown'):
        started = time.monotonic();self.reads += 1
        try:return self._read(entry, role)
        finally:
            now = time.monotonic();self.read_seconds += now-started
            if now-self._reported_at >= 60:
                print(json.dumps(dict(stage='sequence_cache',**self.stats())),flush=True)
                self._reported_at = now

    def _read(self, entry, role):
        import torch
        key = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        if key in self.memory:
            raw = self.memory.pop(key);self.memory[key] = raw
            payload = load_tensor(raw)
            need(self._valid(payload, entry), 'memory full-prefix hash mismatch')
            self.memory_hits += 1
            return payload
        raw=None
        if self._prefetch is not None and self._prefetch[0]==key:
            _,future=self._prefetch;self._prefetch=None;raw=future.result()
            if raw is not None:
                row=self.disk.entries.get(key)
                if row is None or len(raw)!=row['size'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raw=None
                else:
                    row['used']=time.time_ns();self.disk._access_dirty=True
                    if role=='training' or row.get('role','unknown')=='unknown':row['role']=role
                    self.prefetch_hits+=1
        if raw is None:raw = self.disk.get(key, role)
        if raw is not None:
            payload = load_tensor(raw)
            need(self._valid(payload, entry), 'cached full-prefix hash mismatch')
            self.hits += 1;self._remember(key, raw)
            return payload
        legacy = self.root/entry['legacy_name']
        need(Path(entry['legacy_name']).name == entry['legacy_name'], 'legacy prefix basename')
        if legacy.is_file() and not legacy.is_symlink():
            try:
                old = torch.load(legacy, map_location='cpu', weights_only=True)
                if old['positions'] == entry['positions'] and self._valid(old, entry):
                    self.legacy_hits += 1;self._remember(key, object_bytes(old))
                    return old
            except (OSError, RuntimeError, KeyError, EOFError, ValueError, pickle.UnpicklingError):
                pass  # Disposable unsealed cache; recompute and check original hashes.
        regeneration_started = time.monotonic();self.checkpoint()
        if self.backbone is None:
            from applications.rag.top8_backbone_v001 import Backbone
            # Loading a frozen model can initialize temporary random parameters.
            # Cache hits/misses must not change the training RNG stream.
            devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                self.backbone = Backbone(self.manifest['backbone'], self.manifest['model_root'])
        backbone = self.backbone
        package = torch.load(checked_member(self.root, entry['tokens']), map_location='cpu', weights_only=True)
        tokens = package[entry['chunk']]
        device = getattr(backbone, 'device', 'cuda')
        encoded = {k:v.to(device) for k,v in tokens['encoded'].items()}
        captured = {}
        def before(module, args, kw):
            need(not captured, 'one regenerated prefix capture')
            captured['hidden'] = (args[0] if args else kw['hidden_states']).detach()
            captured['mask'] = kw.get('attention_mask', args[1] if len(args)>1 else None)
        hook = backbone.last.register_forward_pre_hook(before, with_kwargs=True)
        try:
            with torch.no_grad():
                logits = backbone.logits(encoded)
        finally:
            hook.remove()
        index = torch.tensor(tokens['keep'], dtype=torch.long, device=device)
        mask = captured['mask']
        if mask is not None:
            if mask.shape[0] == len(logits): mask = mask.index_select(0, index)
            mask = mask.detach().cpu()
        payload = dict(hidden=captured['hidden'].index_select(0,index).cpu(), mask=mask,
                       native_scores=logits.index_select(0,index).detach().cpu())
        need(self._valid(payload, entry), 'regenerated FP32 hidden/mask/native hashes differ from extraction')
        self.regenerated += 1
        self.regeneration_seconds += time.monotonic()-regeneration_started
        raw = object_bytes(payload);self._remember(key, raw)
        self.checkpoint(); self.disk.put(key, raw, role); self.checkpoint()
        return payload

    def close(self):
        if self._prefetch_pool is not None:self._prefetch_pool.shutdown(wait=True,cancel_futures=True)
        self._prefetch=None;self._prefetch_pool=None
        self.backbone = None;self.memory.clear();self.memory_bytes = 0
        try:print(json.dumps(dict(stage='sequence_cache_final',**self.stats())),flush=True)
        finally:self.disk.close()


_SEQUENCES = {}
_SEQUENCE_BUDGET = None
_SEQUENCE_LIMITS = {}
_SEQUENCE_MINIMUM_FREE = 0


def set_sequence_budget(maximum=None, limits=None, minimum_free_bytes=0):
    global _SEQUENCE_BUDGET, _SEQUENCE_LIMITS, _SEQUENCE_MINIMUM_FREE
    need(not _SEQUENCES and (maximum is None or maximum >= 0) and minimum_free_bytes >= 0,
         'set scratch limit before sequence readers start')
    _SEQUENCE_BUDGET = maximum;_SEQUENCE_LIMITS = dict(limits or {})
    need(all(type(v) is int and v >= 0 for v in _SEQUENCE_LIMITS.values()), 'nonnegative per-root cache caps')
    _SEQUENCE_MINIMUM_FREE = minimum_free_bytes


def sequence_reader(root, manifest, checkpoint):
    key = str(Path(root).resolve())
    if key not in _SEQUENCES:
        if manifest is None: manifest = json.loads((Path(root)/'cache.json').read_bytes())
        _SEQUENCES[key] = FrozenSequences(root, manifest, checkpoint,
            _SEQUENCE_LIMITS.get(key, _SEQUENCE_BUDGET),_SEQUENCE_MINIMUM_FREE)
    return _SEQUENCES[key]


def sequence_cache_stats(root):
    reader = _SEQUENCES.get(str(Path(root).resolve()))
    return None if reader is None else reader.stats()


def close_sequences(root=None):
    keys = list(_SEQUENCES) if root is None else [str(Path(root).resolve())]
    for key in keys:
        reader = _SEQUENCES.pop(key, None)
        if reader is not None: reader.close()


def _json_commit(root, name, value):
    """Preserve an uncommitted interrupted write before publishing its receipt."""
    root = Path(root); pending = root/(name+'.part')
    if pending.exists():
        need(not pending.is_symlink(), 'safe interrupted JSON output')
        n = 0
        while (root/(name+f'.interrupted-{n}')).exists(): n += 1
        pending.rename(root/(name+f'.interrupted-{n}'))
    return json_file(root, name, value)


def _generation(root, stem):
    n = 0
    while any((Path(root)/(stem+f'-g{n:04d}'+suffix)).exists()
              for suffix in ('.bin', '.bin.part')):
        n += 1
    return stem+f'-g{n:04d}.bin'


def capture_chunk(backbone, encoded):
    """Same native full forward, CLS features and explicit-zero math check as v1."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    captured = {}; handles = []
    def before_depth(module, args, kw):
        need('depth0' not in captured, 'one depth capture')
        captured['depth0'] = (args[0] if args else kw['hidden_states'])[:,0].detach()
    def before_last(module, args, kw):
        need('prefix' not in captured, 'one last prefix capture')
        captured['prefix'] = (args[0] if args else kw['hidden_states']).detach()
        captured['mask'] = kw.get('attention_mask', args[1] if len(args)>1 else None)
    def after_attention(module, args, out): captured['context'] = out[0][:,0].detach()
    def before_intermediate(module, args):
        captured['ffn_residual'] = args[0][:,0].detach()
        captured['z'] = module.dense(args[0][:,0]).detach()
    def after_last(module, args, out): captured['final'] = out[0][:,0].detach()
    handles += [backbone.layers[-2].register_forward_pre_hook(before_depth, with_kwargs=True),
                backbone.last.register_forward_pre_hook(before_last, with_kwargs=True),
                backbone.last.attention.self.register_forward_hook(after_attention),
                backbone.last.intermediate.register_forward_pre_hook(before_intermediate),
                backbone.last.register_forward_hook(after_last)]
    try:
        with torch.no_grad():
            logits = backbone.logits(encoded)
            for h in handles: h.remove()
            replay = backbone.head(backbone.bridge(captured['final'])).squeeze(-1)
            tail = backbone.tail(captured['context'], captured['prefix'][:,0])
            error = max(float((logits-replay).abs().max()), float((tail-captured['final']).abs().max()))
            need(torch.allclose(logits,replay,atol=2e-5,rtol=2e-6) and
                 torch.allclose(tail,captured['final'],atol=2e-5,rtol=2e-6), 'actual native readout/tail replay')
            hidden = captured['prefix']; mask = captured['mask']; length = hidden.shape[1]
            zero = torch.zeros((len(hidden),backbone.heads,length,length),device=hidden.device)
            if mask is not None:
                zero = zero.masked_fill(~mask,torch.finfo(zero.dtype).min) if mask.dtype==torch.bool else zero+mask
            with sdpa_kernel(SDPBackend.MATH):
                cls = backbone.last(hidden,attention_mask=zero,output_attentions=False)[0][:,0]
                math_logit = backbone.head(backbone.bridge(cls)).squeeze(-1)
            need(torch.allclose(math_logit,logits,atol=2e-5,rtol=2e-6),
                 'every original chunk native/explicit zero math logit parity')
            math_error = float((math_logit-logits).abs().max())
    finally:
        for h in handles: h.remove()
    return captured, logits, error, math_error


def extract_bounded(kind, model_root, train_path, eval_path, output, checkpoint=lambda:None,
                    label_classes=3, resume=False, sequence_cache_bytes=DEFAULT_SEQUENCE_BYTES):
    output=Path(output)
    if output.exists():need(resume and output.is_dir(),'existing extraction requires --resume')
    else:output.mkdir(parents=True)
    writer=DiskLRU(output/'extraction-writer-v2',0)
    try:
        return _extract_bounded(kind,model_root,train_path,eval_path,output,checkpoint,label_classes,
                                resume=True,sequence_cache_bytes=sequence_cache_bytes)
    finally:writer.close()


def _extract_bounded(kind, model_root, train_path, eval_path, output, checkpoint=lambda:None,
                     label_classes=3, resume=False, sequence_cache_bytes=DEFAULT_SEQUENCE_BYTES):
    """Commit one whole question atomically; prefix tensors are hash-bound recipes.

    An old unsealed v1 extraction has no durable feature/index records to resume.
    Preserve its prefix files and verify/reuse them lazily against the new full
    prefix hashes. Never misrepresent those files as completed question shards.
    """
    import torch
    from applications.rag.top8_backbone_v001 import Backbone, MODELS
    from applications.rag.qvalue_portable_v001 import inputs
    output = Path(output)
    if output.exists(): need(resume and output.is_dir(), 'existing extraction requires --resume')
    else: output.mkdir(parents=True)
    need(label_classes in (2,3) and sequence_cache_bytes >= 0, 'explicit cache configuration')
    train = inputs(train_path,True); evaluation = inputs(eval_path,False)
    need(all(all(y in range(label_classes) for y in r['labels'].values()) for r in train),
         'training labels match declared semantics')
    for key in ('id','component_id','question'):
        need(not {r[key] for r in train}&{r[key] for r in evaluation}, 'disjoint role identities')
    identity = dict(layout=LAYOUT,backbone=kind,model_id=MODELS[kind][0],revision=MODELS[kind][1],
        model_root=str(Path(model_root).resolve()),train_sha256=digest(train_path),eval_input_sha256=digest(eval_path),
        label_classes=label_classes,
        extraction_sources={name:digest(Path(__file__).with_name(name))
                            for name in ('top8_backbone_v001.py','top8_cache_v002.py','model_input_guards.py')})
    binding = output/'extraction-binding-v2.json'
    if binding.exists(): need(json.loads(binding.read_bytes())==identity, 'immutable extraction resume binding')
    else: _json_commit(output,binding.name,identity)
    header_path = output/'extraction-header-v2.json'; backbone = None
    if header_path.exists():
        header = json.loads(header_path.read_bytes())
        need(header['binding']==identity, 'header extraction binding')
        checked_member(output, header['frozen'])
        for record in header['assets']:
            p = Path(model_root)/record['name']
            need(p.is_file() and p.stat().st_size==record['size'] and digest(p)==record['sha256'],
                 'same pinned model assets on extraction resume')
    else:
        checkpoint(); backbone = Backbone(kind,model_root)
        frozen = dict(layer=backbone.last.cpu().state_dict(),bridge=backbone.bridge.cpu().state_dict(),
                      head=backbone.head.cpu().state_dict())
        path = output/'frozen-modules.bin'
        if path.exists():
            need(not path.is_symlink(), 'safe legacy frozen module')
            old = torch.load(path,map_location='cpu',weights_only=True)
            need(set(old)==set(frozen) and all(set(old[k])==set(v) and
                 all(torch.equal(old[k][n],x) for n,x in v.items()) for k,v in frozen.items()),
                 'legacy frozen modules must match current pinned model')
            record = dict(name=path.name,size=path.stat().st_size,sha256=digest(path))
        else:
            # A crash before the immutable frozen file was committed preserves
            # its pending bytes under a different name, as for JSON receipts.
            pending = output/'frozen-modules.bin.part'
            if pending.exists(): pending.rename(output/_generation(output,'uncommitted-frozen'))
            record = tensor_file(output,path.name,frozen)
        device = getattr(backbone,'device','cuda')
        backbone.last.to(device); backbone.bridge.to(device); backbone.head.to(device)
        header = dict(binding=identity,frozen=record,config=backbone.config.to_dict(),assets=backbone.asset_records)
        _json_commit(output,header_path.name,header)
    records = [header['frozen']]; metadata = []; maximum_error=maximum_math_error=0.; truncated=0
    population = [('training',r) for r in train]+[('development',r) for r in evaluation]
    for number,(role,row) in enumerate(population):
        checkpoint(); receipt = output/f'question-{number:06d}-v2.json'
        source_sha = hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if receipt.exists():
            saved = json.loads(receipt.read_bytes())
            need(saved['source_sha256']==source_sha and saved['role']==role and saved['binding']==digest(binding),
                 'same complete question/role/source on extraction resume')
            for record in saved['files']: checked_member(output,record)
        else:
            if backbone is None: backbone = Backbone(kind,model_root)
            device = getattr(backbone,'device','cuda')
            shared=[]; native=[]; descriptors=[]; packages=[]; row_error=row_math=0.; row_truncated=0
            for start in range(0,50,8):
                checkpoint(); paragraphs=row['paragraphs'][start:start+8]; texts=[p['text'] for p in paragraphs]
                lengths=backbone.tokenize(row['question'],texts,False)['input_ids']
                row_truncated += sum(len(x)>backbone.limit for x in lengths)
                encoded={k:v.to(device) for k,v in backbone.tokenize(row['question'],texts).items()}
                captured,logits,error,math_error=capture_chunk(backbone,encoded)
                row_error=max(row_error,error);row_math=max(row_math,math_error)
                ids=[p['id'] for p in paragraphs]
                keep=[i for i,pid in enumerate(ids) if role!='training' or pid in row['labels']]
                index=torch.tensor(keep,dtype=torch.long,device=device)
                vectors={k:captured[k].index_select(0,index).cpu() for k in
                         ('depth0','context','ffn_residual','z','final')}
                with torch.no_grad():
                    vectors['residual']=captured['prefix'][:,0].index_select(0,index).cpu()
                    vectors['readout']=backbone.bridge(captured['final']).index_select(0,index).cpu()
                    vectors['depths']=torch.stack([backbone.bridge(captured['depth0']),
                        backbone.bridge(captured['prefix'][:,0]),backbone.bridge(captured['final'])],1).index_select(0,index).cpu()
                shared.append(vectors); native.extend(logits.detach().cpu().tolist())
                if keep:
                    mask=captured['mask']
                    if mask is not None:
                        if mask.shape[0]==len(ids): mask=mask.index_select(0,index)
                        mask=mask.detach().cpu()
                    hidden=captured['prefix'].index_select(0,index).cpu()
                    expected=logits.index_select(0,index).detach().cpu()
                    packages.append(dict(encoded={k:v.detach().cpu().clone() for k,v in encoded.items()},keep=keep))
                    descriptors.append(dict(regenerate=True,chunk=len(packages)-1,positions=[start+i for i in keep],
                        legacy_name=f'prefix-{number:06d}-{start:02d}.bin',
                        signature=sequence_signature(hidden,mask,expected)))
                del captured,encoded,logits,index
            feature={k:torch.cat([c[k] for c in shared]) for k in shared[0]}
            fr=tensor_file(output,_generation(output,f'features-{number:06d}'),feature)
            tr=tensor_file(output,_generation(output,f'tokens-{number:06d}'),packages)
            for entry in descriptors: entry['tokens']=tr
            ids=[p['id'] for p in row['paragraphs']]
            keep=[i for i,pid in enumerate(ids) if role!='training' or pid in row['labels']]
            need(role=='training' or len(keep)==50, 'full evaluation membership')
            record=dict(id=row['id'],component_id=row['component_id'],role=role,candidate_ids=ids,
                retained_indices=keep,sequences=descriptors,feature_file=fr,
                targets=[row['labels'][ids[i]] for i in keep] if role=='training' else None,native_scores=native)
            saved=dict(source_sha256=source_sha,role=role,binding=digest(binding),row=record,files=[fr,tr],
                       max_error=row_error,max_math_error=row_math,truncated=row_truncated)
            _json_commit(output,receipt.name,saved)
            del shared,feature,packages
        records.extend(saved['files']); metadata.append(saved['row'])
        maximum_error=max(maximum_error,saved['max_error']);maximum_math_error=max(maximum_math_error,saved['max_math_error'])
        truncated+=saved['truncated']
        if number%32==0 or number+1==len(population):
            print(json.dumps(dict(stage='extract',backbone=kind,completed_questions=number+1,
                                  total_questions=len(population),storage_layout=LAYOUT)),flush=True)
    result=dict(schema='top8.native-cache.v1',storage_layout=LAYOUT,label_classes=label_classes,backbone=kind,
        model_id=MODELS[kind][0],revision=MODELS[kind][1],config=header['config'],assets=header['assets'],
        model_root=identity['model_root'],sequence_cache_bytes=sequence_cache_bytes,files=records,rows=metadata,
        max_native_replay_error=maximum_error,max_native_math_error=maximum_math_error,truncated_rows=truncated,
        train_sha256=identity['train_sha256'],eval_input_sha256=identity['eval_input_sha256'],
        evaluation_labels_present=False,native_chunk_size=8,full_chunks_before_training_filter=True,
        bge_bridge='frozen dense/tanh; original scalar out_proj' if kind=='bge' else 'identity')
    _json_commit(output,'cache.json',result)
    return output/'cache.json'
