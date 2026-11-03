"""Portable Q-VALUE reproduction and prospective study runner.

No downloads, host configuration, scheduler credentials, or dataset construction.
The output is private: it contains learned weights and per-candidate predictions.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024**2):
            h.update(block)
    return h.hexdigest()


def publish(root, name, raw):
    """Exclusive immutable file, atomic publication and full readback."""
    root = Path(root)
    need(Path(name).name == name, 'output basename required')
    target = root/name
    temporary = root/(name+'.part')
    need(not target.exists(), 'refuse output overwrite')
    with temporary.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    # Hardlink is exclusive and atomic; no overwrite race at publication.
    os.link(temporary, target)
    temporary.unlink()
    if os.name != 'nt':
        fd = os.open(root, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    record = dict(name=name, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    need(target.stat().st_size == record['size'] and digest(target) == record['sha256'],
         'published file readback')
    return record


def json_file(root, name, value):
    return publish(root, name, (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode())


def tensor_file(root, name, value):
    import torch
    b = io.BytesIO()
    torch.save(value, b)
    return publish(root, name, b.getvalue())


def checked_member(root, record):
    need(Path(record['name']).name == record['name'], 'cache member basename')
    path = Path(root)/record['name']
    need(path.is_file() and not path.is_symlink() and
         path.stat().st_size == record['size'] and digest(path) == record['sha256'],
         'cache file size/SHA mismatch')
    return path


def inputs(path, training):
    """Explicit raw-input schema; evaluation conditions contain no targets."""
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            expected = {'id', 'component_id', 'question', 'paragraphs'}
            need(set(row) == expected | ({'labels'} if training else set()),
                 'exact input fields; evaluation labels/answers forbidden')
            need(isinstance(row['question'], str) and row['question'].strip(), 'nonempty question')
            need(len(row['paragraphs']) == 50 and all(set(p) == {'id', 'text'} and
                 isinstance(p['text'], str) and p['text'].strip() for p in row['paragraphs']),
                 'exactly50 nonempty raw paragraphs, no titles')
            ids = [p['id'] for p in row['paragraphs']]
            need(all(isinstance(x, str) and x for x in ids) and len(set(ids)) == 50,
                 'unique paragraph identities')
            if training:
                need(isinstance(row['labels'], dict) and set(row['labels']) <= set(ids) and
                     all(type(y) is int and y in (0, 1, 2) for y in row['labels'].values()),
                     'only reviewed training tiers; absent labels never none')
            need(isinstance(row['id'], str) and row['id'] and isinstance(row['component_id'], str)
                 and row['component_id'], 'question and component identities')
            rows.append(row)
    need(rows and len({x['id'] for x in rows}) == len(rows), 'nonempty unique questions')
    return rows


def extract_cache(train_path, eval_path, model_root, output, checkpoint=lambda: None):
    """One original full forward per original8-row chunk; cache full native values."""
    import torch
    from applications.rag.qcls_five_arm_driver_v001 import offline_reader
    from applications.rag.qvalue_geometry_v002 import NativeTail
    train = inputs(train_path, True)
    evaluation = inputs(eval_path, False)
    need(not {r['id'] for r in train} & {r['id'] for r in evaluation}, 'question role leakage')
    need(not {r['component_id'] for r in train} & {r['component_id'] for r in evaluation},
         'declared components cross roles')
    need(not {hashlib.sha256(r['question'].encode()).hexdigest() for r in train} &
         {hashlib.sha256(r['question'].encode()).hexdigest() for r in evaluation},
         'identical question crosses roles')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = Path(__file__).resolve().parents[2]/'configs/experiments/reader_joint_witness_b1_model_assets.json'
    # HF snapshots commonly use symlinks and include README files. Materialize
    # only the five pinned loader files, without altering the caller's cache.
    asset_bytes = assets.read_bytes().replace(b'\r\n', b'\n')
    specification = next(r for r in json.loads(asset_bytes)['models']
                         if r['model_id'] == 'facebook/dpr-reader-multiset-base')
    view = output/'offline-model'; view.mkdir()
    for record in specification['tokenizer_files']+[specification['weight']]:
        source = Path(model_root)/record['name']
        need(source.is_file() and source.stat().st_size == record['bytes'] and
             digest(source) == record['sha256'], 'pinned offline model member')
        shutil.copyfile(source, view/record['name'])
        checkpoint()
    normalized_assets = output/'model-assets.json'
    with normalized_assets.open('xb') as f:
        f.write(asset_bytes)
    model, tokenizer, receipt = offline_reader(view, normalized_assets, device=torch.device('cuda:0'))
    layers = model.span_predictor.encoder.bert_model.encoder.layer
    tail = NativeTail(layers[-1])
    cache = {}; metadata = []; max_tail_error = 0.; truncated_rows = 0
    for role, population in [('training', train), ('development', evaluation)]:
        for row in population:
            checkpoint(); full = []; native = []
            for start in range(0, 50, 8):
                paragraphs = row['paragraphs'][start:start+8]
                texts = [p['text'] for p in paragraphs]
                kwargs = dict(questions=[row['question']]*len(texts), texts=texts)
                original = tokenizer(**kwargs, truncation=False, padding=False)['input_ids']
                truncated_rows += sum(len(x) > 350 for x in original)
                encoded = tokenizer(**kwargs, truncation=True, max_length=350,
                                    padding=True, return_tensors='pt').to('cuda')
                captured = {}; handles = []
                def before10(module, args, kw):
                    need('block10' not in captured, 'one block10 capture')
                    captured['block10'] = (args[0] if args else kw['hidden_states'])[:, 0].detach()
                def before12(module, args, kw):
                    need('residual' not in captured, 'one last-layer capture')
                    captured['residual'] = (args[0] if args else kw['hidden_states'])[:, 0].detach()
                def context(module, args, result):
                    need('context' not in captured, 'one context capture')
                    captured['context'] = result[0][:, 0].detach()
                def final(module, args, result):
                    need('final' not in captured, 'one final capture')
                    captured['final'] = result[0][:, 0].detach()
                try:
                    handles = [layers[-2].register_forward_pre_hook(before10, with_kwargs=True),
                               layers[-1].register_forward_pre_hook(before12, with_kwargs=True),
                               layers[-1].attention.self.register_forward_hook(context),
                               layers[-1].register_forward_hook(final)]
                    with torch.no_grad():
                        scores = model(**encoded, output_attentions=False).relevance_logits
                finally:
                    for handle in handles:
                        handle.remove()
                with torch.no_grad():
                    h = torch.stack([captured[k] for k in ('context', 'residual', 'block10', 'final')], 1)
                    replay = tail(h[:, 0], h[:, 1])
                    max_tail_error = max(max_tail_error, float((replay-h[:, 3]).abs().max()))
                    need(torch.allclose(replay, h[:, 3], atol=2e-5, rtol=2e-6), 'actual native tail replay')
                    need(torch.equal(model.span_predictor.qa_classifier(h[:, 3]).squeeze(-1), scores),
                         'native relevance head identity')
                full.append(h.cpu()); native.extend(scores.cpu().tolist())
            packed = torch.cat(full)
            with torch.no_grad():
                full_scores = model.span_predictor.qa_classifier(packed[:,3].cuda()).squeeze(-1).cpu()
                need(torch.allclose(full_scores, torch.tensor(native), atol=1e-5, rtol=1e-6),
                     'full50 cached native relevance agrees with original8-row forwards')
                native = full_scores.tolist()
            ids = [p['id'] for p in row['paragraphs']]
            keep = [i for i, q in enumerate(ids) if role != 'training' or q in row['labels']]
            need(row['id'] not in cache, 'reject duplicate cache before insertion')
            cache[row['id']] = packed[keep].clone()
            metadata.append(dict(id=row['id'], component_id=row['component_id'], role=role,
                candidate_ids=ids, retained_indices=keep,
                targets=[row['labels'][ids[i]] for i in keep] if role == 'training' else None,
                native_scores=native if role == 'development' else None))
            checkpoint()
    files = [tensor_file(output, 'features.bin', cache),
             tensor_file(output, 'tail.bin', {k:v.cpu() for k,v in tail.state_dict().items()}),
             tensor_file(output, 'head.bin', {k:v.cpu() for k,v in model.span_predictor.qa_classifier.state_dict().items()})]
    manifest = dict(schema='qvalue.portable-cache.v1', rows=metadata, files=files,
        source=dict(train_sha256=digest(train_path), eval_inputs_sha256=digest(eval_path),
                    model_receipt=receipt, native_tail_max_error=max_tail_error,
                    truncated_rows=truncated_rows, full50_before_train_filter=True),
        evaluation_labels_present=False)
    json_file(output, 'cache.json', manifest)
    return output/'cache.json'


def run_cache(manifest_path, output, *, canary=False, checkpoint=lambda: None):
    """All controls share features, supervision, ordering and optimizer budget."""
    import torch
    from torch.nn import functional as F
    from transformers import BertConfig
    from transformers.models.bert.modeling_bert import BertLayer
    from applications.rag.qvalue_geometry_v002 import ARMS, NativeTail, Scorer
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_bytes())
    need(manifest['schema'] == 'qvalue.portable-cache.v1' and
         manifest['evaluation_labels_present'] is False, 'label-free portable input contract')
    members = {r['name']:checked_member(manifest_path.parent, r) for r in manifest['files']}
    need(set(members) == {'features.bin', 'tail.bin', 'head.bin'}, 'complete cache inventory')
    values = {n:torch.load(p, map_location='cpu', weights_only=True) for n,p in members.items()}
    features = values['features.bin']; rows = manifest['rows']
    need(len({r['id'] for r in rows}) == len(rows) and set(features) == {r['id'] for r in rows},
         'unique complete cache identities')
    train = [r for r in rows if r['role'] == 'training']
    evaluation = [r for r in rows if r['role'] == 'development']
    need(len(train) >= 2 and evaluation and len(train)+len(evaluation) == len(rows), 'complete two roles')
    need(not {r['component_id'] for r in train} & {r['component_id'] for r in evaluation},
         'declared component role isolation')
    for row in rows:
        h = features[row['id']]; indices = row['retained_indices']
        need(len(row['candidate_ids']) == len(set(row['candidate_ids'])) == 50 and
             indices == sorted(set(indices)) and all(type(i) is int and 0 <= i < 50 for i in indices),
             'complete stable candidate order and retained row identity')
        need(h.shape == (len(indices), 4, 768) and h.dtype == torch.float32 and
             not h.requires_grad and bool(torch.isfinite(h).all()), 'finite original feature layout')
        if row['role'] == 'training':
            need(len(row['targets']) == len(indices) and all(type(y) is int and y in (0,1,2)
                 for y in row['targets']), 'reviewed training labels only')
        else:
            need(indices == list(range(50)) and row['targets'] is None and len(row['native_scores']) == 50,
                 'all50 evaluation candidates, no evaluation labels')
    if canary:
        train, evaluation = train[:2], evaluation[:2]
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    need(not output.resolve().is_relative_to(manifest_path.parent) and
         not manifest_path.parent.is_relative_to(output.resolve()), 'input/output roots disjoint')
    layer = BertLayer(BertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072,
                                hidden_act='gelu', layer_norm_eps=1e-12)).cuda().eval().requires_grad_(False)
    tail = NativeTail(layer); tail.load_state_dict(values['tail.bin'], strict=True)
    head = torch.nn.Linear(768, 1, device='cuda').eval()
    head.load_state_dict(values['head.bin'], strict=True); head.requires_grad_(False)
    records = []; checkpoints = []; summaries = {}; epochs = 1 if canary else 4
    seeds = [20261102] if canary else [20261102, 20261103, 20261104]
    native = []
    for row in evaluation:
        scores = head(features[row['id']][:,3].cuda()).squeeze(-1).detach().cpu()
        need(torch.equal(scores, torch.tensor(row['native_scores'], dtype=torch.float32)), 'native cache score bit replay')
        native.append(dict(id=row['id'], candidate_ids=row['candidate_ids'], scores=scores.tolist(),
                           ranking=sorted(range(50), key=lambda i:(-float(scores[i]), i))))
    records.append(json_file(output, 'predictions-native.json', dict(rows=native)))
    for seed in seeds:
        models = {k:Scorer(head, tail, k, seed).eval() for k in ARMS}
        optimizers = {k:torch.optim.AdamW(m.trainable(), lr=.001, betas=(.9,.999), eps=1e-8,
                                         weight_decay=0) for k,m in models.items()}
        generator = torch.Generator(device='cpu').manual_seed(seed); steps = 0
        for epoch in range(epochs):
            order = torch.randperm(len(train), generator=generator).tolist()
            for start in range(0, len(train), 8):
                checkpoint(); batch = [train[i] for i in order[start:start+8]]
                parts = [features[r['id']].cuda() for r in batch]; sizes = [len(h) for h in parts]
                packed = torch.cat(parts)
                for kind, model in models.items():
                    optimizers[kind].zero_grad(set_to_none=True)
                    score = model(packed, prepared=model.prepare()); objective = 0.
                    for r, s in zip(batch, score.split(sizes)):
                        y = torch.tensor(r['targets'], device='cuda'); pairs = y[:,None] > y[None,:]
                        loss = F.softplus(-(s[:,None]-s[None,:]))[pairs].sum()/max(int(pairs.sum()),1)
                        objective = objective+loss/len(batch)
                    need(bool(torch.isfinite(objective)), 'finite whole-question objective')
                    objective.backward()
                    need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.trainable()),
                         'finite train-only gradients')
                    torch.nn.utils.clip_grad_norm_(model.trainable(), 1.)
                    optimizers[kind].step(); model.check_frozen({})
                    del score, objective, loss
                steps += 1; checkpoint()
        for kind, model in models.items():
            key = 's%d_%s' % (seed,kind.lower()); checkpoint()
            state = {k:v.detach().cpu() for k,v in model.owned_state_dict().items()}
            checkpoint_record = tensor_file(output, 'checkpoint-'+key+'.bin', state)
            checkpoints.append(checkpoint_record)
            restored = Scorer(head, tail, kind, seed).eval()
            restored.load_owned_state(torch.load(output/checkpoint_record['name'], map_location='cpu', weights_only=True))
            prediction = []
            with torch.no_grad():
                operator = model.prepare(); other = restored.prepare()
                for row in evaluation:
                    checkpoint(); h = features[row['id']].cuda()
                    scores = model(h, prepared=operator).cpu()
                    need(torch.equal(scores, restored(h, prepared=other).cpu()) and bool(torch.isfinite(scores).all()),
                         'complete checkpoint prediction replay')
                    vals = scores.tolist()
                    prediction.append(dict(id=row['id'], candidate_ids=row['candidate_ids'], scores=vals,
                        ranking=sorted(range(50), key=lambda i:(-vals[i], i))))
            records.append(json_file(output, 'predictions-'+key+'.json', dict(rows=prediction)))
            summaries[key] = dict(steps=steps, epochs=epochs, training_questions=len(train),
                evaluation_questions=len(evaluation), trainable_parameters=sum(p.numel() for p in model.trainable()))
            del restored
        del models, optimizers
    # This is the first point at which a caller may open evaluation targets.
    joint = json_file(output, 'joint-prediction-seal.json', dict(files=records+checkpoints,
        cache_manifest_sha256=digest(manifest_path), all_predictions_before_evaluation_labels=True,
        canary=canary, seeds=seeds, arms=list(ARMS), summaries=summaries))
    return joint


def evaluate(output, labels_path):
    """Read target file only after every predictor/checkpoint is jointly sealed."""
    output = Path(output); seal = output/'joint-prediction-seal.json'
    joint = json.loads(seal.read_bytes())
    predictions = {}
    for record in joint['files']:
        path = checked_member(output, record)
        if record['name'].startswith('predictions-'):
            predictions[record['name']] = json.loads(path.read_bytes())['rows']
    # Explicit sparse targets: no missing item is silently assigned tier0.
    labels = json.loads(Path(labels_path).read_bytes())
    need(isinstance(labels, dict), 'target mapping expected')
    result = {}
    for name, rows in predictions.items():
        useful = direct = unknown = direct_questions = 0
        for row in rows:
            known = labels.get(row['id'], {})
            need(isinstance(known, dict) and all(type(y) is int and y in (0,1,2) for y in known.values()),
                 'valid reviewed evaluation tiers')
            selected = [known.get(row['candidate_ids'][i]) for i in row['ranking'][:5]]
            useful += sum(y in (1,2) for y in selected)
            direct += sum(y == 2 for y in selected)
            unknown += sum(y is None for y in selected)
            direct_questions += int(2 in selected)
        result[name] = dict(useful_lower=useful, useful_upper=useful+unknown,
            direct_lower=direct, direct_upper=direct+unknown, unknown=unknown,
            direct_questions_lower=direct_questions, selected_denominator=5*len(rows),
            complete=unknown == 0)
    json_file(output, 'evaluation.json', dict(joint_seal_sha256=digest(seal),
        labels_sha256=digest(labels_path), results=result, quantum_attribution='not_established'))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path)
    p.add_argument('--train', type=Path)
    p.add_argument('--eval-inputs', type=Path)
    p.add_argument('--eval-labels', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--canary', action='store_true')
    args = p.parse_args(argv)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import transformers
    need(torch.cuda.is_available() and transformers.__version__ == '4.44.2',
         'CUDA and transformers4.44.2 required; use the documented environment')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    repo = Path(__file__).resolve().parents[2]
    data = Path(os.environ.get('QVALUE_DATA_DIR', str(repo/'data/qvalue')))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or repo/'outputs'/('qvalue-'+stamp)
    if args.cache is None:
        model = args.model or (Path(os.environ['QVALUE_MODEL_DIR']) if 'QVALUE_MODEL_DIR' in os.environ else None)
        if model is None:
            hf_home = Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))
            hub = Path(os.environ.get('HF_HUB_CACHE', str(hf_home/'hub')))
            candidate = hub/'models--facebook--dpr-reader-multiset-base/snapshots/ab1974d6023bc4dd27cd713c9ac2b6247509244d'
            if candidate.is_dir():
                model = candidate
        need(model is not None, 'set --model or QVALUE_MODEL_DIR to the pinned offline snapshot')
        args.cache = extract_cache(args.train or data/'train.jsonl', args.eval_inputs or data/'eval.inputs.jsonl',
                                   model, output.with_name(output.name+'-cache'))
    joint = run_cache(args.cache, output, canary=args.canary)
    result = evaluate(output, args.eval_labels) if args.eval_labels else None
    print(json.dumps(dict(output=str(output.resolve()), seal_sha256=joint['sha256'],
        mode='CANARY_NOT_SCIENTIFIC_RESULT' if args.canary else 'COMPLETE_PREDECLARED_RUN',
        evaluation=result, claim='Existing exposed-B1 L2 vs frozen DPR; quantum increment unproven')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
