"""Fixed80/20 task adapter for common scorer plugins, allocated GPU only.

Reuse pinned Backbone, immutable artifacts and full-state Journal. Prefixes come
from native full8-candidate forwards. Only the final CLS dependency slice is used
for training, after real full/CLS null equality and ranking checks.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import torch
from applications.rag.top8_backbone_v001 import Backbone
from applications.rag.qvalue_portable_v001 import need, digest, json_file, tensor_file, checked_member
from applications.rag.coherent_qk_v001 import cls_scores


def rows(root):
    root = Path(root)
    prep = json.loads((root/'preparation.json').read_bytes())
    for record in prep['files']:
        # Population projection uses the explicit path/size/SHA schema; cache
        # artifacts use name/size/SHA. Adapt the field, never bypass the hash.
        checked_member(root, dict(name=record['path'], size=record['size'], sha256=record['sha256']))
    result = {}
    for role, count in (('train', 80), ('holdout', 20)):
        with gzip.open(root/(role+'.inputs.jsonl.gz'), 'rb') as f:
            raw = list(f)
        population = [json.loads(x) for x in raw]
        need(len(population) == count and all(len(x['paragraphs']) == 50 for x in population),
             'fixed complete toy roles')
        for row, line in zip(population, raw):
            row['raw_sha256'] = hashlib.sha256(line).hexdigest()
        result[role] = population
    need(len({r['component_id'] for rs in result.values() for r in rs}) == 100,
         '100 unique role-disjoint components')
    return result


def train_targets(root, population):
    """Never opens holdout labels. Complete candidate-order and raw-row binding."""
    path = Path(root)/'train.labels.private.jsonl'
    labels = [json.loads(line) for line in path.open('rb')]
    need(len(labels) == len(population) == 80, 'fixed complete training targets')
    values = []
    for row, target in zip(population, labels):
        need(target['id'] == row['id'] and target['component_id'] == row['component_id']
             and target['source_row_sha256'] == row['raw_sha256'], 'training target row binding')
        need(len(target['labels']) == 50 and all(p['id'] == y['candidate_id']
             and type(y['useful']) is int and y['useful'] in (0, 1)
             for p, y in zip(row['paragraphs'], target['labels'])), '50 ordered training targets')
        values.append(torch.tensor([y['useful'] for y in target['labels']], dtype=torch.long))
    return values


def order(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def extract(kind, model_root, data, output, event=lambda **kw: None):
    """One common native cache per backbone; no evaluator/holdout-label access."""
    output = Path(output)
    populations = rows(data)
    backbone = Backbone(kind, model_root)
    if (output/'cache.json').is_file():
        manifest = json.loads((output/'cache.json').read_bytes())
        need(manifest['kind'] == kind, 'cache backbone')
        expected = {(role, i): row['raw_sha256'] for role, rs in populations.items()
                    for i, row in enumerate(rs)}
        actual = {(r['role'], r['row_index']): r['raw_sha256'] for r in manifest['questions']}
        need(len(manifest['questions']) == 100 and actual == expected, 'reused cache exact population/role binding')
        for r in manifest['records']:
            checked_member(output, r)
        return backbone, manifest
    output.mkdir(parents=True, exist_ok=False)
    records, metadata, native_predictions = [], [], []
    maximum = 0.
    reordered_questions = changed_native_top5 = 0
    for role, population in populations.items():
        for ri, row in enumerate(population):
            prefixes, validity, native, tokens = [], [], [], []
            for start in range(0, 50, 8):
                texts = [p['text'] for p in row['paragraphs'][start:start+8]]
                encoded = backbone.tokenize(row['question'], texts).to('cuda')
                captured = {}
                def before(module, args, kwargs):
                    need(not captured, 'single native prefix capture')
                    captured['hidden'] = (args[0] if args else kwargs['hidden_states']).detach()
                hook = backbone.last.register_forward_pre_hook(before, with_kwargs=True)
                try:
                    with torch.no_grad():
                        score = backbone.logits(encoded)
                finally:
                    hook.remove()
                prefixes.append(captured['hidden'].cpu())
                validity.append(encoded['attention_mask'].bool().cpu())
                native.extend(score.cpu().tolist())
                tokens.append(dict(start=start, shape=list(encoded['input_ids'].shape),
                                   ids_sha256=hashlib.sha256(encoded['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                                   mask_sha256=hashlib.sha256(encoded['attention_mask'].cpu().numpy().tobytes()).hexdigest()))
                del encoded, captured, score
            length = max(x.shape[1] for x in prefixes)
            hidden = torch.zeros(50, length, backbone.dimension)
            valid = torch.zeros(50, length, dtype=torch.bool)
            start = 0
            for h, v in zip(prefixes, validity):
                b, length0, _ = h.shape
                hidden[start:start+b, :length0] = h
                valid[start:start+b, :length0] = v
                start += b
            with torch.no_grad():
                h, v = hidden.cuda(), valid.cuda()
                replay = cls_scores(backbone.last, backbone.bridge, backbone.head, h,
                                    v[:, None, None, :], v)
                expected = replay.new_tensor(native)
                error = float((replay-expected).abs().max())
                need(torch.allclose(replay, expected, atol=2e-4, rtol=2e-5),
                     'full-native / padded CLS dependency slice numeric parity')
                # Native-full and short-query fused SDPA can differ at rounding
                # scale. Preserve the original native scores as the fixed anchor,
                # rather than silently changing their near-tie rankings.
                reordered_questions += int(order(native) != order(replay.cpu().tolist()))
                changed_native_top5 += int(set(order(native)[:5]) != set(order(replay.cpu().tolist())[:5]))
                maximum = max(maximum, error)
            name = f'{role}-{ri:03d}.bin'
            record = tensor_file(output, name, dict(hidden=hidden, valid=valid,
                                  native=torch.tensor(native), native_slice=replay.detach().cpu()))
            records.append(record)
            metadata.append(dict(role=role, row_index=ri, tensor=record, raw_sha256=row['raw_sha256'],
                                 question_id=row['id'], component_id=row['component_id'], tokens=tokens))
            native_predictions.append(dict(role=role, row_index=ri, scores=native))
            event(phase='extracting', backbone=kind, current=len(metadata), total=100,
                  maximum_null_error=maximum)
    native_record = json_file(output, 'native-predictions.json', native_predictions)
    records.append(native_record)
    manifest = dict(kind=kind, records=records, questions=metadata, native=native_record,
                    max_native_slice_error=maximum, parameter_dtype='torch.float32',
                    attention_type=type(backbone.last.attention.self).__name__,
                    input_chunk=8, cached_question_batch=50, extra_padding_masked=True,
                    unanchored_reordered_questions=reordered_questions,
                    unanchored_changed_top5_questions=changed_native_top5,
                    score_semantics='native_full + modified_slice - frozen_native_slice; fixed label-independent numerical offset')
    json_file(output, 'cache.json', manifest)
    return backbone, manifest


class Reader:
    """Host-resident prefixes and bounded GPU QKV/reference working set."""
    def __init__(self, root, manifest, backbone, gpu_bytes=8*1024**3):
        self.backbone, self.limit = backbone, gpu_bytes
        self.host, self.gpu = {}, OrderedDict()
        self.bytes = 0
        for row in manifest['questions']:
            path = checked_member(root, row['tensor'])
            self.host[row['role'], row['row_index']] = torch.load(path, map_location='cpu', weights_only=True)

    def get(self, role, index):
        key = role, index
        if key in self.gpu:
            self.gpu.move_to_end(key)
            return self.gpu[key]
        host = self.host[key]
        b = self.backbone
        with torch.no_grad():
            h, valid = host['hidden'].cuda(), host['valid'].cuda()
            attn = b.last.attention.self
            value = dict(hidden=h, valid=valid, mask=valid[:, None, None, :],
                         precomputed=(attn.query(h[:, :1]), attn.key(h), attn.value(h)),
                         native=host['native'].cuda(), native_slice=host['native_slice'].cuda(), references={})
            size = sum(x.numel()*x.element_size() for x in (h, valid, *value['precomputed']))
            # Reference7-float/token work sets for all arms/seeds are separately
            # evicted at cell boundaries; account for their maximum one-cell size.
            size += valid.numel()*7*4
        need(size <= self.limit, 'one complete question GPU working set')
        while self.gpu and self.bytes+size > self.limit:
            _, old = self.gpu.popitem(last=False)
            self.bytes -= old['accounted_bytes']
        value['accounted_bytes'] = size
        self.gpu[key] = value
        self.bytes += size
        return value

    def clear_references(self):
        for value in self.gpu.values():
            value['references'].clear()

    def scores(self, role, index, adapter=None, head=None, capture=None):
        value = self.get(role, index)
        ref = None
        if adapter is not None:
            key = id(adapter)
            if key not in value['references']:
                with torch.no_grad():
                    value['references'][key] = adapter.features(value['hidden'], True)
            ref = value['references'][key]
        b = self.backbone
        modified = cls_scores(b.last, b.bridge, b.head if head is None else head,
                              value['hidden'], value['mask'], value['valid'], adapter, capture,
                              value['precomputed'], ref)
        # The fixed offset has no trainable parameters/target dependence. It is
        # shared by HEAD and every QK control, preserves original native anchors,
        # and contributes no gradient. Do not claim full-layer machine identity.
        return value['native'] + (modified - value['native_slice'])
