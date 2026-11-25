"""Prospective toy study runner: fixed full-state recovery, joint prediction seal.

Only the explicitly allocated host executes this module. Formal full-smallset
experiments remain the junior collaborator's responsibility.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from applications.rag.coherent_qk_v001 import Adapter, CoRead, ARMS
from applications.rag.toy100_qk_task_v001 import extract, rows, train_targets, Reader, order
from applications.rag.qvalue_portable_v001 import need, digest, json_file, checked_member
from applications.rag.top8_recovery_v001 import Journal

SEEDS = (20261122, 20261123, 20261124)
EPOCHS, BATCH, LR = 20, 8, .001


def event(path, **values):
    path = Path(path)
    old = json.loads(path.read_bytes())
    old.update(values, observed_at=time.time(), source_event=old['source_event']+1)
    old['scientific_execution'] = values.get('scientific_execution', 'running')
    old['unit'] = 'optimizer steps' if values.get('phase') in ('training', 'starting_cell', 'cell_complete') else 'questions/groups'
    old['detail'] = 'Fixed toy100 common QK study; no heldout-label join before complete joint seal'
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(old, f, allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp, path)


def loss(scores, labels):
    y = labels.to(scores.device)
    high, low = scores[y == 1], scores[y == 0]
    if not len(high) or not len(low):
        return scores.sum()*0
    return F.softplus(low[None, :]-high[:, None]).mean()


def factory(backbone, arm, seed, joint=False):
    if arm == 'HEAD':
        module = copy.deepcopy(backbone.head)
        module.weight.requires_grad_(True)
        module.bias.requires_grad_(False)
        return module.eval()
    if joint:
        return CoRead(backbone.head, backbone.dimension, arm, seed).cuda().eval()
    return Adapter(backbone.dimension, arm, seed).cuda().eval()


def scores(reader, index, arm, module, role='train', capture=None):
    if isinstance(module, CoRead):
        return reader.scores(role, index, module.qk, module.head, capture)
    return reader.scores(role, index, None if arm == 'HEAD' else module,
                         module if arm == 'HEAD' else None, capture)


def canary(reader, targets, arm, seed, joint=False):
    """Two training-only informative questions, never a utility selection gate."""
    fixed = [i for i, y in enumerate(targets) if 0 < int(y.sum()) < 50][:2]
    need(len(fixed) == 2, 'two informative training probes')
    module = factory(reader.backbone, arm, seed, joint)
    opt = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=LR,
                           weight_decay=0., foreach=False)
    reader.clear_references()
    before = []
    for i in fixed:
        z = scores(reader, i, arm, module)
        expected = reader.get('train', i)['native']
        need(torch.allclose(z, expected, atol=2e-4, rtol=2e-5), 'actual initial reference/null')
        before.append(z.detach())
        (loss(z, targets[i])/len(fixed)).backward()
    gradients = {n: float(p.grad.abs().max()) if p.grad is not None else None
                 for n, p in module.named_parameters() if p.requires_grad}
    need(all(v is not None and torch.isfinite(p.grad).all() for (n, p), v in
             [( (n, p), gradients[n]) for n, p in module.named_parameters() if p.requires_grad]),
         'finite actual gradients for owned trainable parameters')
    if arm == 'HEAD':
        need(gradients['weight'] > 0, 'HEAD active weight gradient')
    else:
        need(gradients['qk.projection' if joint else 'projection'] > 0, 'active upstream projection gradient')
        if joint:
            need(gradients['head.weight'] > 0, 'joint original readout gradient')
    opt.step();opt.zero_grad(set_to_none=True)
    changed = 0.
    with torch.no_grad():
        for i, old in zip(fixed, before):
            changed = max(changed, float((scores(reader, i, arm, module)-old).abs().max()))
    need(changed > 0, 'one real update changes finite logits, not necessarily ranking')
    reader.clear_references()
    return dict(arm=arm, training_indices=fixed, gradient_maxima=gradients,
                one_update_max_logit_change=changed,
                trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad))


def cell(root, binding, reader, targets, arm, seed, emit, joint=False):
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    if (root/'result.json').is_file():
        result = json.loads((root/'result.json').read_bytes())
        checked_member(root, result['predictions'])
        need(result['binding'] == binding, 'completed cell exact binding')
        return result
    module = factory(reader.backbone, arm, seed, joint)
    reader.clear_references()
    opt = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=LR,
                           weight_decay=0., foreach=False)
    permutation = torch.Generator(device='cpu').manual_seed(seed+17)
    batches = [batch for _ in range(EPOCHS) for batch in
               torch.randperm(80, generator=permutation).reshape(-1, BATCH).tolist()]
    journal = Journal(root/'journal', binding)
    saved = journal.load()
    start = 0
    if saved is not None:
        module.load_state_dict(saved['module'])
        opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start = saved['step']
    started = time.time()
    for step in range(start, len(batches)):
        opt.zero_grad(set_to_none=True)
        total = 0.
        for i in batches[step]:
            z = scores(reader, i, arm, module)
            value = loss(z, targets[i])/BATCH
            need(bool(torch.isfinite(value)), 'finite train-only question loss')
            value.backward();total += float(value.detach())
        params = [p for p in module.parameters() if p.requires_grad]
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),
             'finite owned gradients before optimizer')
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        if (step+1) % 20 == 0:
            journal.save(dict(step=step+1, module=module.state_dict(), optimizer=opt.state_dict(),
                              cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
            emit(phase='training', arm=arm, seed=seed, current=step+1, total=len(batches),
                 loss=total, elapsed=time.time()-started,
                 gpu_allocated=torch.cuda.memory_allocated(), gpu_reserved=torch.cuda.memory_reserved())
    predictions = []
    with torch.no_grad():
        for i in range(20):
            z = scores(reader, i, arm, module, 'holdout')
            need(bool(torch.isfinite(z).all()), '50 finite label-free heldout logits')
            predictions.append(dict(row_index=i, scores=z.cpu().tolist(), top5=order(z.cpu().tolist())[:5]))
    record = json_file(root, 'predictions.json', predictions)
    # Actual checkpoint reconstruction and full20-question score/rank replay.
    restored = factory(reader.backbone, arm, seed, joint)
    final = journal.load();restored.load_state_dict(final['module'])
    reader.clear_references()
    maximum = 0.
    with torch.no_grad():
        for pred in predictions:
            z = scores(reader, pred['row_index'], arm, restored, 'holdout')
            old = z.new_tensor(pred['scores'])
            need(torch.allclose(z, old, atol=1e-6, rtol=1e-6) and
                 order(z.cpu().tolist())[:5] == pred['top5'], 'actual checkpoint prediction replay')
            maximum = max(maximum, float((z-old).abs().max()))
    result = dict(binding=binding, step=len(batches), predictions=record,
                  checkpoint=journal.latest_record(), replay_maximum=maximum,
                  trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad),
                  elapsed=time.time()-started)
    json_file(root, 'result.json', result)
    reader.clear_references()
    return result


def evaluate(data, output, joint):
    """Called only after the complete32-group joint seal, never during training."""
    population = rows(data)['holdout']
    targets = [json.loads(x) for x in (Path(data)/'holdout.labels.private.jsonl').open('rb')]
    need(len(targets) == 20, 'complete heldout target population')
    ys = []
    for row, label in zip(population, targets):
        need(label['id'] == row['id'] and label['component_id'] == row['component_id']
             and label['source_row_sha256'] == row['raw_sha256'], 'heldout target binding')
        need(len(label['labels']) == 50 and all(p['id'] == y['candidate_id']
             and y['useful'] in (0, 1) for p, y in zip(row['paragraphs'], label['labels'])),
             'complete ordered binary heldout targets')
        ys.append([x['useful'] for x in label['labels']])
    metrics = []
    for group in joint:
        source = Path(output)/group['directory']
        predictions = json.loads(checked_member(source, group['predictions']).read_bytes())
        if group['arm'] == 'NATIVE':
            predictions = [p for p in predictions if p['role'] == 'holdout']
        need([p['row_index'] for p in predictions] == list(range(20)), 'all20 question score rows')
        useful = sum(sum(ys[i][j] for j in order(p['scores'])[:5]) for i, p in enumerate(predictions))
        metrics.append(dict(backbone=group['backbone'], arm=group['arm'], seed=group['seed'],
                            useful=useful, denominator=100))
    return dict(metrics=metrics, oracle=sum(min(5, sum(y)) for y in ys), questions=20, denominator=100,
                claim_ceiling='fixed100-question exploratory toy; no fullsmallset generalization',
                joint_seal_sha256=digest(Path(output)/'joint-seal.json'))


def main():
    p = argparse.ArgumentParser()
    for name in ('data', 'models', 'output', 'progress', 'binding'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--variant', choices=('qk','qk_coread'), default='qk')
    p.add_argument('--cache-source')
    p.add_argument('--reuse-head-root')
    args = p.parse_args()
    output = Path(args.output);output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(args.binding).read_bytes())
    repo = Path(__file__).resolve().parents[2]
    for name, sha in source['files'].items():
        need(digest(repo/name) == sha, 'exact staged source/config')
    need(torch.cuda.is_available(), 'allocated CUDA device')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    emit = lambda **kw: event(args.progress, **kw)
    protocol = dict(variant=args.variant, seeds=SEEDS, arms=ARMS, epochs=EPOCHS, logical_batch=BATCH, lr=LR,
                    weight_decay=0., clip_norm=1., optimizer='AdamW foreach=False',
                    pair_loss='all binary positive-negative pairs, mean per question, fixed batch denominator8',
                    fixed_checkpoint='final200', source=source,
                    score_semantics='native_full + modified_slice - frozen_native_slice; fixed same-input numerical anchoring across all arms',
                    environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                     device=torch.cuda.get_device_name(), deterministic=True, tf32=False))
    if (output/'protocol.json').is_file():
        need(json.loads((output/'protocol.json').read_bytes()) == json.loads(json.dumps(protocol)), 'frozen protocol')
    else:
        json_file(output, 'protocol.json', protocol)
    populations = rows(args.data)
    targets = train_targets(args.data, populations['train'])
    joint = []
    for kind in ('dpr', 'bge'):
        cache = (Path(args.cache_source) if args.cache_source else output)/('cache-'+kind)
        backbone, manifest = extract(kind, Path(args.models)/kind, args.data, cache, emit)
        joint.append(dict(backbone=kind, arm='NATIVE', seed=None, directory=str(cache.resolve()),
                          predictions=manifest['native']))
        reader = Reader(cache, manifest, backbone)
        for arm in ARMS:
            if arm == 'HEAD' and args.reuse_head_root:
                origin = Path(args.reuse_head_root)
                old_protocol = json.loads((origin/'protocol.json').read_bytes())
                for field in ('seeds','epochs','logical_batch','lr','weight_decay','clip_norm','optimizer','pair_loss','fixed_checkpoint','environment','score_semantics'):
                    need(old_protocol[field] == json.loads(json.dumps(protocol))[field], 'unchanged HEAD training protocol')
                need((origin/'joint-seal.json').is_file() and (origin/'result.json').is_file(), 'HEAD source completed and sealed')
                for seed in SEEDS:
                    folder = origin/(kind+'-HEAD-s'+str(seed))
                    previous = json.loads((folder/'result.json').read_bytes())
                    need(previous['binding']['arm'] == 'HEAD' and previous['binding']['seed'] == seed
                         and previous['binding']['cache'] == digest(cache/'cache.json')
                         and previous['step'] == 200 and previous['replay_maximum'] <= 1e-6,
                         'reuse exact complete same-input HEAD checkpoint')
                    checked_member(folder, previous['predictions'])
                    checked_member(folder/'journal', previous['checkpoint']['tensor'])
                    joint.append(dict(backbone=kind, arm='HEAD', seed=seed,
                                      directory=str(folder.resolve()), predictions=previous['predictions']))
                continue
            key = kind+'-'+arm+'-canary.json'
            if not (output/key).is_file():
                result = canary(reader, targets, arm, SEEDS[0], args.variant == 'qk_coread')
                json_file(output, key, result)
            for seed in SEEDS:
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                directory = kind+'-'+arm+'-s'+str(seed)
                binding = dict(backbone=kind, arm=arm, seed=seed,
                               protocol=digest(output/'protocol.json'), cache=digest(cache/'cache.json'), variant=args.variant)
                emit(phase='starting_cell', backbone=kind, arm=arm, seed=seed, current=0, total=200)
                result = cell(output/directory, binding, reader, targets, arm, seed, emit,
                              args.variant == 'qk_coread')
                joint.append(dict(backbone=kind, arm=arm, seed=seed, directory=directory,
                                  predictions=result['predictions']))
                emit(phase='cell_complete', backbone=kind, arm=arm, seed=seed, current=200, total=200)
        del reader, backbone
        gc.collect();torch.cuda.empty_cache()
    need(len(joint) == 32, 'all predeclared both-model arms/seeds plus native')
    if not (output/'joint-seal.json').is_file():
        json_file(output, 'joint-seal.json', joint)
    else:
        need(json.loads((output/'joint-seal.json').read_bytes()) == joint, 'joint seal exact reuse')
    result = evaluate(args.data, output, joint)
    if not (output/'result.json').is_file():
        json_file(output, 'result.json', result)
    emit(phase='completed', current=32, total=32, scientific_execution='completed')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
