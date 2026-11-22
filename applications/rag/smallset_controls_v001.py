"""Read-only recovery of already completed controls; no scientific imports."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import uuid

from applications.rag.junior_transport_v001 import (
    Exchange, atomic_json, digest, member, relative, require,
)


def read_json(root, name, limit):
    path = member(root, name)
    require(path.is_file() and path.stat().st_size <= limit, 'metadata missing or exceeds limit: '+name)
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    require(len(raw) <= limit, 'metadata grew beyond limit: '+name)
    return json.loads(raw)


def inventory(workspace, request):
    """Bind the old matrix to this specific completed primary run, not old B1."""
    work = Path(workspace).expanduser().absolute()
    require(not work.is_symlink(), 'symlink workspace')
    root = member(work, 'run-smallset')
    limit = request['max_metadata_bytes']
    require(request['schema'] == 'qduet.existing-control-request.v1', 'known recovery request')
    primary = member(root, 'matrix-primary54')
    require(digest(member(primary, 'joint-prediction-seal.json')) ==
            request['primary_joint_seal_sha256'], 'different primary result; do not mix runs')
    require(digest(member(primary, 'run-binding.json')) ==
            request['original_run_binding_sha256'], 'different original source/input binding')
    primary_binding = read_json(primary, 'run-binding.json', limit)
    execution = read_json(root, 'execution.json', limit)
    require(execution.get('returncode') == 0 and execution.get('all_cells_complete') is True
            and execution.get('experiment_profile') == 'primary54', 'primary run is not complete')
    require(execution['roles'] == request['roles'], 'different data roles')
    require(primary_binding['seeds'] == request['seeds'] and primary_binding['epochs'] == 4
            and primary_binding['question_batch'] == 8 and primary_binding['canary'] is False,
            'different seed/epoch/batch scope')
    require(sorted(c['sha256'] for c in primary_binding['caches']) ==
            sorted(request['cache_sha256'].values()), 'different input/model cache bindings')
    # Read hashes only, never decode questions, training labels or features.
    for kind, sha in request['cache_sha256'].items():
        require(digest(member(root, 'cache-'+kind+'/cache.json')) == sha,
                'current cache inventory changed: '+kind)
    records = {}

    def add(name, expected=None):
        path = member(root, name)
        require(path.is_file(), 'regular recovery file required: '+name)
        record = dict(path=name, size=path.stat().st_size, sha256=digest(path))
        require(path.is_file() and record['size'] <= request['max_prediction_bytes'],
                'bounded regular recovery file: '+name)
        if expected:
            require(record['size'] == expected['size'] and record['sha256'] == expected['sha256'],
                    'completed control bytes changed: '+name)
        require(name not in records or records[name] == record, 'conflicting duplicate recovery file')
        records[name] = record

    for name in ('execution.json', 'matrix-primary54/run-binding.json',
                 'matrix-primary54/joint-prediction-seal.json', 'matrix-primary54/adoption.json'):
        add(name)
    legacy = member(root, 'matrix')
    legacy_present = (legacy/'run-binding.json').is_file()
    if legacy_present:
        require(digest(member(legacy, 'run-binding.json')) == request['original_run_binding_sha256'],
                'legacy matrix belongs to a different source/input population')
        add('matrix/run-binding.json')
        for path in sorted(legacy.glob('*-continuation.v*.json')):
            add('matrix/'+path.name)
    rows = []
    for kind in request['cache_sha256']:
        for idea, arms in request['control_arms'].items():
            for seed in request['seeds']:
                for arm in arms:
                    require(arm != 'QUANTUM', 'control request must not collect quantum cells')
                    key = f'{kind}-{idea}-{arm}-s{seed}'
                    relative(key)
                    name = 'matrix/'+key+'/disposition.json'
                    row = dict(job=key, backbone=kind, idea=idea, arm=arm, seed=seed,
                               status='not_present')
                    rows.append(row)
                    if not member(root, name).exists():
                        recovery = 'matrix/'+key+'/recovery/binding.json'
                        if member(root, recovery).is_file():
                            binding = read_json(root, recovery, limit)
                            require(binding.get('matrix') == primary_binding and
                                    (binding.get('backbone'), binding.get('idea'),
                                     binding.get('arm'), binding.get('seed')) == (kind, idea, arm, seed),
                                    'checkpoint-only control binding mismatch: '+key)
                            add(recovery)
                            row['status'] = 'checkpoint_only'
                        continue
                    require(legacy_present, 'unbound legacy disposition')
                    value = read_json(root, name, limit)
                    require(value['binding'] == primary_binding, 'control original binding mismatch: '+key)
                    add(name)
                    if value['status'] != 'complete':
                        require(value['status'] == 'failed', 'unknown disposition state: '+key)
                        row['status'] = 'failed'
                        continue
                    result = value['result']
                    require((result['idea'], result['arm'], result['seed']) == (idea, arm, seed),
                            'control method/arm/seed mismatch: '+key)
                    require(result['steps'] == request['steps'] and result['training_questions'] == 8000
                            and result['evaluation_questions'] == 2000, 'partial control run: '+key)
                    files = value['files']
                    require(files and len({r.get('path', r.get('name')) for r in files}) == len(files),
                            'duplicate or empty completed cell manifest: '+key)
                    predictions = []
                    for record in files:
                        rel = record.get('path', record.get('name'))
                        require(isinstance(rel, str) and rel.startswith(key+'/'),
                                'control file crosses cell boundary: '+key)
                        path = member(legacy, rel)
                        require(path.is_file() and path.stat().st_size == record['size']
                                and digest(path) == record['sha256'], 'sealed control file changed: '+key)
                        if path.name == f'predictions-{idea}-{arm}-s{seed}.json':
                            add('matrix/'+rel, record)
                            predictions.append('matrix/'+rel)
                    require(len(predictions) == 1, 'one exact completed prediction required: '+key)
                    # Do not copy/reload checkpoint tensors or regenerate predictions.
                    row.update(status='complete', prediction=predictions[0], steps=result['steps'],
                               trainable_parameters=result.get('trainable_parameters'),
                               attention_route=result.get('attention_route'),
                               throughput_route=result.get('throughput_route'))
    complete = sum(x['status'] == 'complete' for x in rows)
    failed = sum(x['status'] == 'failed' for x in rows)
    partial = sum(x['status'] == 'checkpoint_only' for x in rows)
    total = sum(x['size'] for x in records.values())
    require(total <= request['max_upload_bytes'], 'recovery exceeds upload cap; nothing uploaded')
    return dict(schema='qduet.existing-control-inventory.v1', primary_run_id=request['primary_run_id'],
                request=request, original_run_binding_sha256=request['original_run_binding_sha256'],
                legacy_matrix_present=legacy_present, cells=rows, files=list(records.values()),
                summary=dict(expected=len(rows), complete=complete, failed=failed,
                             checkpoint_only=partial, not_present=len(rows)-complete-failed-partial,
                             upload_bytes=total),
                new_training=False, new_predictions=False, label_join=False,
                source_outputs_modified=False, pre_label_joint_seal_inferred=False,
                no_controls_is_not_zero_effect=True)


def collect(workspace, request_path, *, upload=False, config_path=None, exchange_factory=Exchange):
    request = json.loads(Path(request_path).read_bytes())
    value = inventory(workspace, request)  # Validate everything before network or output creation.
    work = Path(workspace).expanduser().absolute()
    parent = member(work, 'control-receipts')
    parent.mkdir(exist_ok=True)
    run_id = 'controls-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output = member(parent, run_id)
    output.mkdir(exist_ok=False)
    atomic_json(output/'inventory.json', value)
    summary = dict(stage='existing_controls_inventoried', **value['summary'],
                   inventory=str(output/'inventory.json'), training_launched=False,
                   label_join=False, uploaded=False)
    if not upload:
        return summary
    # Reuse the configured private exchange; do not prompt for or print credentials.
    candidates = [Path(config_path).expanduser()] if config_path else [
        Path(os.environ.get('QDUET_HANDOFF_CONFIG', '~/.config/qduet/junior.json')).expanduser(),
        Path('/etc/qore-upload/client.json')]
    config = next((json.loads(p.read_bytes()) for p in candidates if p.is_file()), None)
    if config is None and os.environ.get('QORE_EXCHANGE_URL') and os.environ.get('QORE_EXCHANGE_TOKEN'):
        config = dict(url=os.environ['QORE_EXCHANGE_URL'], token=os.environ['QORE_EXCHANGE_TOKEN'])
    require(config is not None, 'existing private junior.json configuration required; no training started')
    exchange = exchange_factory(config['url'], config['token'])
    prefix = 'five_ideas/top8-junior-controls/'+run_id
    pending = dict(schema='qduet.existing-control-upload.v1', status='pending',
                   prefix=prefix, primary_run_id=request['primary_run_id'],
                   files=value['files'], receipts=[])
    atomic_json(output/'upload.json', pending)
    for record in value['files']:
        path = member(work/'run-smallset', record['path'])
        require(path.stat().st_size == record['size'] and digest(path) == record['sha256'],
                'source changed during recovery; upload stopped')
        receipt = exchange.upload(path, prefix+'/'+record['path'])
        require(receipt == dict(path=prefix+'/'+record['path'], size_bytes=record['size'],
                                sha256=record['sha256']), 'control upload receipt mismatch')
        pending['receipts'].append(receipt)
        atomic_json(output/'upload.json', pending)
    inv = output/'inventory.json'
    receipt = exchange.upload(inv, prefix+'/inventory.json')
    require(receipt == dict(path=prefix+'/inventory.json', size_bytes=inv.stat().st_size,
                            sha256=digest(inv)), 'inventory upload receipt mismatch')
    pending.update(status='uploaded', inventory_receipt=receipt)
    atomic_json(output/'upload.json', pending)
    final = output/'upload.json'
    receipt = exchange.upload(final, prefix+'/upload.json')
    require(receipt == dict(path=prefix+'/upload.json', size_bytes=final.stat().st_size,
                            sha256=digest(final)), 'final upload receipt mismatch')
    summary.update(stage='existing_controls_uploaded', uploaded=True, exchange_namespace=prefix)
    return summary
