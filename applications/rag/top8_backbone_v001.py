"""Native DPR/BGE adapters and shared immutable caches. Queue worker only.

The BGE bridge retains dense/tanh before its original final linear readout.
No model download, title insertion, answer supervision, or evaluation-label read.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from torch import nn
from applications.rag.qvalue_portable_v001 import (
    need, digest, inputs, json_file, tensor_file, checked_member,
)
from applications.rag.qvalue_geometry_v002 import NativeTail

MODELS = {
    'dpr': ('facebook/dpr-reader-multiset-base',
            'ab1974d6023bc4dd27cd713c9ac2b6247509244d', 350),
    'bge': ('BAAI/bge-reranker-v2-m3',
            '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', 512),
}


class ReadoutBridge(nn.Module):
    def __init__(self, dense=None):
        super().__init__()
        self.dense = copy.deepcopy(dense) if dense is not None else nn.Identity()
        self.requires_grad_(False).eval()

    def forward(self, cls):
        need(not self.training and all(not p.requires_grad for p in self.parameters()),
             'frozen native readout bridge')
        value = self.dense(cls)
        return value.tanh() if isinstance(self.dense, nn.Linear) else value


class Backbone:
    def __init__(self, kind, root):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import DPRReader, DPRReaderTokenizer
        need(kind in MODELS, 'explicit backbone')
        self.kind = kind
        root = Path(root).resolve(strict=True)
        manifest = Path(__file__).resolve().parents[2]/'configs/experiments/reader_joint_witness_b1_model_assets.json'
        spec = next(x for x in json.loads(manifest.read_bytes())['models']
                    if x['model_id'] == MODELS[kind][0])
        self.asset_records = []
        for r in spec['tokenizer_files']+[spec['weight']]:
            p = root/r['name']
            need(p.is_file() and p.stat().st_size == r['bytes'] and digest(p) == r['sha256'],
                 'pinned existing model bytes')
            self.asset_records.append(dict(name=r['name'], size=r['bytes'], sha256=r['sha256']))
        if kind == 'dpr':
            self.model = DPRReader.from_pretrained(str(root), local_files_only=True).float().cuda().eval()
            self.tokenizer = DPRReaderTokenizer.from_pretrained(str(root), local_files_only=True)
            base = self.model.span_predictor.encoder.bert_model
            self.head = self.model.span_predictor.qa_classifier
            self.bridge = ReadoutBridge().cuda()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(root), local_files_only=True).float().cuda().eval()
            self.tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, use_fast=True)
            base = self.model.roberta
            need(type(self.model).__name__ == 'XLMRobertaForSequenceClassification', 'native BGE architecture')
            self.head = self.model.classifier.out_proj
            self.bridge = ReadoutBridge(self.model.classifier.dense).cuda()
        self.model.requires_grad_(False)
        self.layers = base.encoder.layer
        self.last = self.layers[-1]
        self.tail = NativeTail(self.last)
        self.config = self.model.config
        self.dimension = self.config.hidden_size
        self.heads = self.config.num_attention_heads
        need(self.dimension//self.heads == 64 and self.head.out_features == 1, 'native64 value heads/scalar logit')
        need(self.tokenizer.padding_side == 'right', 'original right padding')
        self.limit = MODELS[kind][2]

    def tokenize(self, question, texts, truncated=True):
        kw = dict(truncation=truncated, padding=True if truncated else False)
        if truncated:
            kw.update(max_length=self.limit, return_tensors='pt')
        if self.kind == 'dpr':
            return self.tokenizer(questions=[question]*len(texts), texts=texts, **kw)
        return self.tokenizer([question]*len(texts), texts, **kw)

    def logits(self, encoded):
        out = self.model(**encoded, output_attentions=False, return_dict=True)
        return out.relevance_logits if self.kind == 'dpr' else out.logits.squeeze(-1)


def extract(kind, model_root, train_path, eval_path, output, checkpoint=lambda: None):
    """Cache all native chunks first; retain reviewed training rows afterwards."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train = inputs(train_path, True)
    evaluation = inputs(eval_path, False)
    for key in ('id', 'component_id'):
        need(not {r[key] for r in train} & {r[key] for r in evaluation}, 'disjoint role identities')
    need(not {r['question'] for r in train} & {r['question'] for r in evaluation}, 'question role leakage')
    backbone = Backbone(kind, model_root)
    records = []; metadata = []; maximum_error = 0.; maximum_math_error = 0.; truncated = 0; features = {}
    records.append(tensor_file(output, 'frozen-modules.bin', dict(
        layer=backbone.last.cpu().state_dict(), bridge=backbone.bridge.cpu().state_dict(),
        head=backbone.head.cpu().state_dict())))
    backbone.last.cuda(); backbone.bridge.cuda(); backbone.head.cuda()
    for role, population in [('training', train), ('development', evaluation)]:
        for row in population:
            checkpoint(); shared = []; native = []; sequences = []
            for start in range(0, 50, 8):
                checkpoint(); paragraphs = row['paragraphs'][start:start+8]
                texts = [p['text'] for p in paragraphs]
                lengths = backbone.tokenize(row['question'], texts, False)['input_ids']
                truncated += sum(len(x) > backbone.limit for x in lengths)
                encoded = backbone.tokenize(row['question'], texts).to('cuda')
                captured = {}; handles = []
                def before_depth(module, args, kw):
                    need('depth0' not in captured, 'one depth capture')
                    captured['depth0'] = (args[0] if args else kw['hidden_states'])[:, 0].detach()
                def before_last(module, args, kw):
                    need('prefix' not in captured, 'one last prefix capture')
                    h = args[0] if args else kw['hidden_states']
                    captured['prefix'] = h.detach()
                    captured['mask'] = kw.get('attention_mask', args[1] if len(args)>1 else None)
                def after_attention(module, args, out):
                    captured['context'] = out[0][:, 0].detach()
                def before_intermediate(module, args):
                    captured['ffn_residual'] = args[0][:, 0].detach()
                    captured['z'] = module.dense(args[0][:, 0]).detach()
                def after_last(module, args, out):
                    captured['final'] = out[0][:, 0].detach()
                handles += [backbone.layers[-2].register_forward_pre_hook(before_depth, with_kwargs=True),
                            backbone.last.register_forward_pre_hook(before_last, with_kwargs=True),
                            backbone.last.attention.self.register_forward_hook(after_attention),
                            backbone.last.intermediate.register_forward_pre_hook(before_intermediate),
                            backbone.last.register_forward_hook(after_last)]
                try:
                    with torch.no_grad():
                        logits = backbone.logits(encoded)
                        # Captures belong only to the full original forward. The
                        # CLS-only tail and last-layer math replay must not fire them.
                        for h in handles:h.remove()
                        replay = backbone.head(backbone.bridge(captured['final'])).squeeze(-1)
                        tail = backbone.tail(captured['context'], captured['prefix'][:, 0])
                        error = max(float((logits-replay).abs().max()), float((tail-captured['final']).abs().max()))
                        need(torch.allclose(logits, replay, atol=2e-5, rtol=2e-6) and
                             torch.allclose(tail, captured['final'], atol=2e-5, rtol=2e-6), 'actual native readout/tail replay')
                        maximum_error = max(maximum_error, error)
                        from torch.nn.attention import SDPBackend, sdpa_kernel
                        hidden=captured['prefix'];mask=captured['mask'];length=hidden.shape[1]
                        zero=torch.zeros((len(hidden),backbone.heads,length,length),device=hidden.device)
                        if mask is not None:
                            zero=zero.masked_fill(~mask,torch.finfo(zero.dtype).min) if mask.dtype==torch.bool else zero+mask
                        with sdpa_kernel(SDPBackend.MATH):
                            math_cls=backbone.last(hidden,attention_mask=zero,output_attentions=False)[0][:,0]
                            math_logit=backbone.head(backbone.bridge(math_cls)).squeeze(-1)
                        need(torch.allclose(math_logit,logits,atol=2e-5,rtol=2e-6),'every original chunk native/explicit zero math logit parity')
                        maximum_math_error=max(maximum_math_error,float((math_logit-logits).abs().max()))
                        del zero,math_cls,math_logit
                finally:
                    for h in handles: h.remove()
                ids = [p['id'] for p in paragraphs]
                keep = [i for i, pid in enumerate(ids) if role != 'training' or pid in row['labels']]
                index = torch.tensor(keep, dtype=torch.long, device='cuda')
                vectors = {k:captured[k].index_select(0,index).cpu() for k in
                           ('depth0', 'context', 'ffn_residual', 'z', 'final')}
                vectors['residual'] = captured['prefix'][:, 0].index_select(0,index).cpu()
                vectors['readout'] = backbone.bridge(captured['final']).index_select(0,index).detach().cpu()
                vectors['depths'] = torch.stack([backbone.bridge(captured['depth0']),
                    backbone.bridge(captured['prefix'][:, 0]), backbone.bridge(captured['final'])],1).index_select(0,index).detach().cpu()
                shared.append(vectors); native.extend(logits.detach().cpu().tolist())
                if keep:
                    mask = captured['mask']
                    if mask is not None:
                        mask = mask.detach()
                        if mask.shape[0] == len(ids): mask = mask.index_select(0,index)
                        mask = mask.cpu()
                    payload = dict(hidden=captured['prefix'].index_select(0,index).cpu(), mask=mask,
                        native_scores=logits.index_select(0,index).detach().cpu(),
                        positions=[start+i for i in keep],
                        token_sha256=digest_tensor(encoded['input_ids']),
                        attention_sha256=digest_tensor(encoded['attention_mask']))
                    name=f'prefix-{len(metadata):06d}-{start:02d}.bin'
                    r=tensor_file(output,name,payload);records.append(r);sequences.append(r)
                del captured, encoded, logits, index
            features[row['id']] = {k:torch.cat([c[k] for c in shared]) for k in shared[0]}
            ids = [p['id'] for p in row['paragraphs']]
            keep = [i for i,pid in enumerate(ids) if role != 'training' or pid in row['labels']]
            need(role=='training' or len(keep)==50, 'reviewed membership/full evaluation')
            metadata.append(dict(id=row['id'], component_id=row['component_id'], role=role,
                candidate_ids=ids, retained_indices=keep, sequences=sequences,
                targets=[row['labels'][ids[i]] for i in keep] if role=='training' else None,
                # Fusion normalizes BOTH complete training score channels;
                # unreviewed positions remain excluded from training targets.
                native_scores=native))
    records.append(tensor_file(output,'features.bin',features))
    result=dict(schema='top8.native-cache.v1',backbone=kind,model_id=MODELS[kind][0],
        revision=MODELS[kind][1],config=backbone.config.to_dict(),assets=backbone.asset_records,
        files=records,rows=metadata,max_native_replay_error=maximum_error,max_native_math_error=maximum_math_error,truncated_rows=truncated,
        train_sha256=digest(train_path),eval_input_sha256=digest(eval_path),
        evaluation_labels_present=False,native_chunk_size=8,full_chunks_before_training_filter=True,
        bge_bridge='frozen dense/tanh; original scalar out_proj' if kind=='bge' else 'identity')
    json_file(output,'cache.json',result)
    return output/'cache.json'


def digest_tensor(x):
    import hashlib
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_modules(manifest, root):
    from transformers import BertConfig, XLMRobertaConfig
    from transformers.models.bert.modeling_bert import BertLayer
    from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLayer
    kind=manifest['backbone'];config=(BertConfig if kind=='dpr' else XLMRobertaConfig)(**manifest['config'])
    # Transformers 4.44.2 exposes SDPA for BERT, but XLM-R only has its
    # original eager implementation. Reconstruct the native class, not a
    # backend name unsupported by the pinned BGE runtime.
    config._attn_implementation='sdpa' if kind=='dpr' else 'eager'
    layer=(BertLayer if kind=='dpr' else XLMRobertaLayer)(config).float().cuda().eval().requires_grad_(False)
    d=config.hidden_size
    bridge=ReadoutBridge(nn.Linear(d,d) if kind=='bge' else None).cuda()
    head=nn.Linear(d,1).cuda().eval().requires_grad_(False)
    r=next(x for x in manifest['files'] if x['name']=='frozen-modules.bin')
    value=torch.load(checked_member(root,r),map_location='cuda',weights_only=True)
    layer.load_state_dict(value['layer'],strict=True);bridge.load_state_dict(value['bridge'],strict=True)
    head.load_state_dict(value['head'],strict=True)
    return layer,bridge,head
