"""Two-scorer fusion adapters for the portable campaign.

Planning, alignment and reporting use stdlib only. Scientific imports occur
inside the admitted training child. Both frozen forward costs belong to every
fusion prediction; these cells are never counted as single-DPR/BGE adapters.
"""
from __future__ import annotations
import gc
import json
import math
from pathlib import Path
from applications.rag.qvalue_portable_v001 import need, digest, checked_member, json_file, tensor_file

FUSION_ARMS = {
    'CROSS_SCORE': ('HEAD', 'LINEAR', 'POLY', 'TRIG', 'MLP', 'PRODUCT', 'QUANTUM'),
    'CLASS_FUSION': ('HEAD', 'FULLCLASS', 'LINEAR', 'POLY', 'TRIG', 'MLP', 'PRODUCT', 'QUANTUM'),
}


def fusion_plan(available, ideas=None):
    ideas = list(ideas or [])
    need(len(ideas) == len(set(ideas)) and set(ideas) <= set(FUSION_ARMS),
         'unique implemented fusion ideas required')
    need(not ideas or set(available) == {'dpr', 'bge'}, 'fusion requires both DPR and BGE')
    return ideas


def aligned_rows(dpr, bge):
    """Bind full original Top50, role/order/targets and exact shared inputs.

    Complete training scores are needed even on unreviewed positions ONLY for
    training-only normalization. Supervision uses retained reviewed rows only.
    Old caches lacking these scores must be regenerated in a new workspace.
    """
    need(dpr['backbone'] == 'dpr' and bge['backbone'] == 'bge', 'ordered DPR/BGE caches')
    for channel, identity in [(dpr, ('facebook/dpr-reader-multiset-base', 'ab1974d6023bc4dd27cd713c9ac2b6247509244d')),
                              (bge, ('BAAI/bge-reranker-v2-m3', '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'))]:
        need((channel['model_id'], channel['revision']) == identity, 'pinned fusion model identity')
    for key in ('schema', 'train_sha256', 'eval_input_sha256'):
        need(dpr[key] == bge[key], 'same original input bytes across fusion channels')
    need(dpr['evaluation_labels_present'] is False and bge['evaluation_labels_present'] is False,
         'label-free evaluation caches')
    need(dpr['schema'] == 'top8.native-cache.v1' and len(dpr['rows']) == len(bge['rows']),
         'same complete fusion population')
    seen = set(); roles = {'training': set(), 'development': set()}; rows = []
    for a, b in zip(dpr['rows'], bge['rows']):
        for key in ('id', 'component_id', 'role', 'candidate_ids', 'retained_indices', 'targets'):
            need(a[key] == b[key], 'fusion question/component/role/candidate/target alignment')
        need(a['id'] not in seen and a['role'] in roles, 'unique valid fusion question')
        seen.add(a['id']); roles[a['role']].add(a['component_id'])
        ids = a['candidate_ids']; keep = a['retained_indices']
        need(len(ids) == 50 and len(set(ids)) == 50 and keep == sorted(set(keep))
             and all(type(i) is int and 0 <= i < 50 for i in keep), 'original50 retained positions')
        if a['role'] == 'development':
            need(a['targets'] is None and keep == list(range(50)), 'complete label-free development')
        else:
            need(isinstance(a['targets'], list) and len(a['targets']) == len(keep)
                 and all(type(y) is int and y in (0, 1, 2) for y in a['targets']),
                 'reviewed training targets only')
        for channel in (a, b):
            values = channel['native_scores']
            need(isinstance(values, list) and len(values) == 50
                 and all(type(v) in (int, float) and math.isfinite(v) for v in values),
                 'complete finite original50 scores required; old cache needs new workspace')
        rows.append(dict(a, paired_scores=list(map(list, zip(a['native_scores'], b['native_scores'])))))
    need(roles['training'] and roles['development'] and not roles['training'] & roles['development'],
         'disjoint nonempty fusion components')
    return rows


def _build(idea, arm, seed, head, mean, std, priors, full_training_scores, denominator=None):
    import torch
    if idea == 'CROSS_SCORE':
        from applications.rag.qfusion_v001 import Scorer, raw
        model = Scorer(arm, head['weight'], head['bias'], mean, std, seed, float(std[0])).cuda()
        if arm not in ('HEAD', 'LINEAR') and denominator is None:
            with torch.no_grad():
                denominator = max(1e-6, float(raw(arm, torch.atan(model.features(full_training_scores)),
                                                 model.reference).std(unbiased=False)))
        model.denominator = denominator if denominator is not None else 1.
    else:
        from applications.rag.qclass_fusion_v001 import Scorer
        model = Scorer(arm, head['weight'], head['bias'], mean, std, seed, float(std[0]), priors=priors).cuda()
    return model.eval()


def train_fusion_cell(idea, arm, seed, rows, features, head, output, epochs,
                      steps_limit, checkpoint, journal):
    """Same original losses/AdamW/order, arbitrary population and sparse labels.

    Each whole question contributes equally, including questions with no
    reviewed ordered pairs. No developmental labels reach this function.
    """
    import torch
    if idea == 'CROSS_SCORE':
        from applications.rag.qfusion_v001 import loss_for
    else:
        from applications.rag.qclass_fusion_v001 import loss_for
    train = [r for r in rows if r['role'] == 'training']
    dev = [r for r in rows if r['role'] == 'development']
    scores = {r['id']: torch.tensor(r['paired_scores'], dtype=torch.float32) for r in rows}
    full = torch.cat([scores[r['id']] for r in train]).cuda()
    mean = full.mean(0); std = full.std(0, unbiased=False)
    need(bool((std > 1e-6).all()), 'nonconstant training-only score channels')
    labels = torch.tensor([y for r in train for y in r['targets']], device='cuda', dtype=torch.long)
    counts = torch.bincount(labels, minlength=3).float()
    need(bool((counts > 0).all()) or idea != 'CLASS_FUSION', 'three training classes for class fusion')
    priors = counts / counts.sum().clamp_min(1)
    model = _build(idea, arm, seed, head, mean, std, priors, full)
    denominator = model.denominator
    del full, labels
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(parameters.values(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)

    def values(row, all50=False):
        index = list(range(50)) if all50 else row['retained_indices']
        h = features[row['id']]['readout'].cuda()
        need(h.shape == (len(index), 768), 'DPR readout/retained-position alignment')
        s = scores[row['id']][index].cuda()
        return h, s

    with torch.no_grad():
        maximum_null = 0.
        for row in train[:2]:
            h, s = values(row)
            if not len(h):
                continue
            expected = torch.nn.functional.linear(h, head['weight'].cuda(), head['bias'].cuda()).squeeze(-1)
            need(torch.allclose(expected, s[:, 0], atol=2e-5, rtol=2e-6), 'actual DPR readout/raw-score parity')
            if idea == 'CROSS_SCORE' or arm == 'HEAD':
                actual = model(h, s)
                need(torch.allclose(actual, expected, atol=2e-5, rtol=2e-6), 'fusion initial native null')
                maximum_null = max(maximum_null, float((actual-expected).abs().max()))
    records = []; generator = torch.Generator().manual_seed(seed); step = 0
    saved = journal.load()
    if saved is not None:
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); generator.set_state(saved['generator'])
        torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step = saved['step']; records = saved['records']
        need(len(records) == step and saved['denominator'] == denominator, 'fusion recovery normalization/history')
    for epoch in range(epochs):
        if saved is not None and epoch < saved['epoch']:
            continue
        restoring = saved is not None and epoch == saved['epoch']
        order = saved['order'] if restoring else torch.randperm(len(train), generator=generator).tolist()
        need(sorted(order) == list(range(len(train))), 'complete fusion training permutation')
        for start in range(0, len(order), 8):
            if restoring and start < saved['next_start']:
                continue
            checkpoint(); batch = [train[i] for i in order[start:start+8]]
            optimizer.zero_grad(set_to_none=True); total = 0.; supervised = 0; empty = 0
            for row in batch:
                h, s = values(row)
                if not len(h):
                    empty += 1
                    continue
                y = torch.tensor(row['targets'], device='cuda', dtype=torch.long)
                loss, n = loss_for(model, h[None], y[None], s[None])
                need(bool(torch.isfinite(loss)), 'finite fusion objective')
                (loss/len(batch)).backward(); total += float(loss.detach())/len(batch)
                supervised += int(n.sum()); empty += int(n.sum() == 0)
            active = [p.grad for p in parameters.values() if p.grad is not None]
            need(all(bool(torch.isfinite(g).all()) for g in active), 'finite fusion gradients')
            need(not supervised or any(bool((g != 0).any()) for g in active), 'active fusion objective')
            norm = torch.nn.utils.clip_grad_norm_(parameters.values(), 1.); optimizer.step(); step += 1
            need(all(bool(torch.isfinite(p).all()) for p in parameters.values()), 'finite fusion parameters')
            records.append(dict(step=step, loss=total, supervision=supervised,
                                empty_questions=empty, gradient_norm=float(norm)))
            if step % 32 == 0 or start+8 >= len(order) or (steps_limit and step >= steps_limit):
                journal.save(dict(step=step, epoch=epoch, next_start=start+8, order=order,
                    model={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
                    denominator=denominator, optimizer=optimizer.state_dict(), generator=generator.get_state(),
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), records=records))
            if step % 32 == 0:
                print(json.dumps(dict(stage='fusion_training', idea=idea, arm=arm, seed=seed, step=step)), flush=True)
            if steps_limit and step >= steps_limit:
                break
        if steps_limit and step >= steps_limit:
            break
    state = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    key = f'{idea}-{arm}-s{seed}'
    files = [tensor_file(output, 'checkpoint-'+key+'.bin', dict(model=state, denominator=denominator))]
    # Replay the serialized file, not just the in-memory copy.
    stored = torch.load(checked_member(output, files[0]), map_location='cpu', weights_only=True)
    replay = _build(idea, arm, seed, head, mean, std, priors, None, stored['denominator'])
    replay.load_state_dict(stored['model'], strict=True)
    predictions = []
    with torch.no_grad():
        for row in dev:
            checkpoint(); h, s = values(row, all50=True)
            a = model(h, s); b = replay(h, s)
            need(a.shape == (50,) and bool(torch.isfinite(a).all()) and torch.equal(a, b),
                 'all50 serialized fusion checkpoint bit replay')
            v = a.cpu().tolist()
            predictions.append(dict(id=row['id'], candidate_ids=row['candidate_ids'], scores=v,
                                    ranking=sorted(range(50), key=lambda i:(-v[i], i))))
    files += [json_file(output, 'predictions-'+key+'.json', dict(rows=predictions)),
              json_file(output, 'training-'+key+'.json', dict(records=records))]
    result = dict(idea=idea, arm=arm, seed=seed, steps=step, training_questions=len(train),
        evaluation_questions=len(dev), trainable_parameters=sum(p.numel() for p in parameters.values()),
        max_initial_null_error=maximum_null, prediction_replay_error=0., normalization_training_only=True,
        training_class_priors=priors.cpu().tolist(), full_frozen_forward_channels=['dpr', 'bge'],
        sample_trace_v2='DEFERRED_NOT_PASSED', files=files)
    del model, replay, optimizer, parameters, scores
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_fusions(caches, output, seeds, ideas, binding, *, canary=False, checkpoint=lambda:None):
    """Return artifacts to the SAME joint seal as all eight single-model ideas."""
    import torch
    from applications.rag.top8_portable_v001 import load_cache
    from applications.rag.top8_recovery_v001 import Journal, new_generation
    from applications.rag.top8_suite_v002 import verify_files
    manifests = {json.loads(Path(p).read_bytes())['backbone']: Path(p) for p in caches}
    fusion_plan(manifests, ideas)
    manifest, features = load_cache(manifests['dpr'])
    bge = json.loads(manifests['bge'].read_bytes())
    rows = aligned_rows(manifest, bge)
    if canary:
        rows = ([r for r in rows if r['role'] == 'training'][:2] +
                [r for r in rows if r['role'] == 'development'][:2])
    record = next(r for r in manifest['files'] if r['name'] == 'frozen-modules.bin')
    head = torch.load(checked_member(manifests['dpr'].parent, record), map_location='cpu', weights_only=True)['head']
    results = {}; failures = []; files = []
    for idea in ideas:
        for seed in seeds:
            for arm in FUSION_ARMS[idea]:
                checkpoint(); key = f'fusion-{idea}-{arm}-s{seed}'
                cell = output/key; cell.mkdir(exist_ok=True); disposition = cell/'disposition.json'
                if disposition.exists():
                    value = json.loads(disposition.read_bytes())
                    need(value['binding'] == binding, 'completed fusion cell binding')
                    if value['status'] == 'complete':
                        verify_files(output, value['files']); results[key] = value['result']; files.extend(value['files'])
                    else:
                        failures.append(value['failure'])
                    continue
                journal = Journal(cell/'recovery', dict(matrix=binding, idea=idea, arm=arm, seed=seed, scope='fusion'))
                generation = new_generation(cell)
                try:
                    result = train_fusion_cell(idea, arm, seed, rows, features, head, generation,
                                              1 if canary else 4, 1 if canary else None, checkpoint, journal)
                    owned = [dict(r, path=(generation.relative_to(output)/r['name']).as_posix()) for r in result['files']]
                    json_file(cell, disposition.name, dict(status='complete', binding=binding, result=result, files=owned))
                    results[key] = result; files.extend(owned)
                except Exception as error:
                    failure = dict(scope='fusion', idea=idea, arm=arm, seed=seed,
                                   error_type=type(error).__name__, message=str(error)[:1000])
                    json_file(cell, disposition.name, dict(status='failed', binding=binding, failure=failure))
                    failures.append(failure); gc.collect(); torch.cuda.empty_cache()
    del features; gc.collect(); torch.cuda.empty_cache()
    return results, failures, files
