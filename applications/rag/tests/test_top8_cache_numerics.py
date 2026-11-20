"""Synthetic CPU tensors only: no pretrained model, dataset, network or CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from applications.rag.top8_cache_v002 import (
    FeatureRows, FrozenSequences, capture_chunk, extract_bounded, sequence_signature, tensor_hash,object_bytes,
)
from applications.rag.qvalue_portable_v001 import checked_member, tensor_file
from applications.rag.top8_portable_v001 import load_cache


class SelfAttention(nn.Module):
    def forward(self, h): return (h*.5,)


class Intermediate(nn.Module):
    def __init__(self):super().__init__();self.dense=nn.Linear(8,32)
    def forward(self,h):return self.dense(h)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__();self.attention=nn.Module();self.attention.self=SelfAttention()
        self.intermediate=Intermediate();self.output=nn.Linear(32,8)
    def forward(self,hidden_states,attention_mask=None,output_attentions=False):
        residual=self.attention.self(hidden_states)[0]+hidden_states
        return (self.output(self.intermediate(residual).relu())+residual,)


class Prefix(nn.Module):
    def forward(self,h,attention_mask=None):return h*.75+.1


class TinyConfig:
    def to_dict(self):return dict(hidden_size=8)


class FakeBackbone:
    """Deterministic CPU stand-in with the same native hook/row interfaces."""
    calls=0
    def __init__(self,kind,root):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.layers=nn.ModuleList([Prefix(),TinyLayer()]).eval().requires_grad_(False)
            self.head=nn.Linear(8,1).eval().requires_grad_(False)
        self.last=self.layers[-1];self.bridge=nn.Identity();self.heads=2
        self.device='cpu';self.limit=12;self.config=TinyConfig();self.asset_records=[]
    def tail(self,context,residual):
        x=context+residual
        return self.last.output(self.last.intermediate.dense(x).relu())+x
    def tokenize(self,question,texts,truncated=True):
        seqs=[[1]+[2+(ord(x)%17) for x in question+' '+t]+[2] for t in texts]
        if not truncated:return dict(input_ids=seqs)
        seqs=[s[:self.limit] for s in seqs];length=max(map(len,seqs))
        return dict(input_ids=torch.tensor([s+[0]*(length-len(s)) for s in seqs]),
                    attention_mask=torch.tensor([[1]*len(s)+[0]*(length-len(s)) for s in seqs]))
    def logits(self,encoded):
        type(self).calls+=1
        h=encoded['input_ids'].float().unsqueeze(-1).expand(-1,-1,8)/10
        mask=encoded['attention_mask'][:,None,None,:].bool()
        h=self.layers[0](h,attention_mask=mask)
        h=self.last(h,attention_mask=mask,output_attentions=False)[0]
        return self.head(h[:,0]).squeeze(-1)


class NativeFixture(FakeBackbone):
    """Real pinned Transformers layer classes, random tiny CPU weights only."""
    def __init__(self,kind,root):
        from transformers import BertConfig,XLMRobertaConfig
        from transformers.models.bert.modeling_bert import BertLayer
        from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLayer
        from applications.rag.qvalue_geometry_v002 import NativeTail
        from applications.rag.top8_backbone_v001 import ReadoutBridge
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            cls=BertConfig if kind=='dpr' else XLMRobertaConfig
            config=cls(hidden_size=64,num_attention_heads=1,intermediate_size=128,
                       hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
            config._attn_implementation='sdpa' if kind=='dpr' else 'eager'
            self.layers=nn.ModuleList([Prefix(),(BertLayer if kind=='dpr' else XLMRobertaLayer)(config)]).eval().requires_grad_(False)
            self.head=nn.Linear(64,1).eval().requires_grad_(False)
            self.bridge=ReadoutBridge(nn.Linear(64,64) if kind=='bge' else None)
        self.last=self.layers[-1];self.tail=NativeTail(self.last);self.heads=1
        self.device='cpu';self.limit=12;self.config=config;self.asset_records=[]
    def logits(self,encoded):
        h=(encoded['input_ids'].float().unsqueeze(-1)+torch.arange(64).float()/64).sin()
        mask=torch.zeros_like(encoded['attention_mask'],dtype=torch.float32)[:,None,None,:]
        mask=mask.masked_fill(encoded['attention_mask'][:,None,None,:]==0,torch.finfo(torch.float32).min)
        h=self.layers[0](h,attention_mask=mask)
        h=self.last(h,attention_mask=mask,output_attentions=False)[0]
        return self.head(self.bridge(h[:,0])).squeeze(-1)


def populations(root):
    paths=[]
    for role in ('train','eval'):
        rows=[]
        for j in range(2):
            ids=[f'{role}-{j}-{i}' for i in range(50)]
            r=dict(id=f'{role}-{j}',component_id=hashlib.sha256(f'{role}-{j}'.encode()).hexdigest(),
                   question=f'question{role}{j}',paragraphs=[dict(id=p,text='text '+str(i)) for i,p in enumerate(ids)])
            if role=='train':r['labels']={p:i%2 for i,p in enumerate(ids)}
            rows.append(r)
        p=root/(role+'.jsonl');p.write_text(''.join(json.dumps(r)+'\n' for r in rows));paths.append(p)
    return paths


class CacheNumerics(unittest.TestCase):
    def test_migrate_unsealed_legacy_extraction_without_deleting_its_prefixes(self):
        with tempfile.TemporaryDirectory() as d,patch('applications.rag.top8_backbone_v001.Backbone',FakeBackbone):
            root=Path(d);train,evaluation=populations(root);output=root/'cache';output.mkdir()
            b=FakeBackbone('dpr',root)
            tensor_file(output,'frozen-modules.bin',dict(layer=b.last.state_dict(),bridge=b.bridge.state_dict(),head=b.head.state_dict()))
            question=json.loads(train.read_text().splitlines()[0])
            encoded=b.tokenize(question['question'],[p['text'] for p in question['paragraphs'][:8]])
            captured,logits,_,_=capture_chunk(b,encoded)
            tensor_file(output,'prefix-000000-00.bin',dict(hidden=captured['prefix'],mask=captured['mask'],
                        native_scores=logits,positions=list(range(8))))
            before={p.name:p.read_bytes() for p in output.iterdir()}
            cache=extract_bounded('dpr',root,train,evaluation,output,label_classes=2,resume=True)
            for name,raw in before.items():self.assertEqual((output/name).read_bytes(),raw)
            manifest,_=load_cache(cache);reader=FrozenSequences(output,manifest)
            try:
                reader.read(manifest['rows'][0]['sequences'][0])
                self.assertEqual(reader.legacy_hits,1);self.assertEqual(reader.regenerated,0)
            finally:reader.close()

    def test_uncommitted_question_files_never_become_a_completed_question(self):
        from applications.rag.top8_cache_v002 import _json_commit
        with tempfile.TemporaryDirectory() as d,patch('applications.rag.top8_backbone_v001.Backbone',FakeBackbone):
            root=Path(d);train,evaluation=populations(root)
            def crash(output,name,value):
                if name=='question-000000-v2.json':raise InterruptedError('after tensors before commit')
                return _json_commit(output,name,value)
            with patch('applications.rag.top8_cache_v002._json_commit',crash),self.assertRaises(InterruptedError):
                extract_bounded('dpr',root,train,evaluation,root/'cache',label_classes=2)
            before=(root/'cache/features-000000-g0000.bin').read_bytes()
            cache=extract_bounded('dpr',root,train,evaluation,root/'cache',label_classes=2,resume=True)
            self.assertEqual((root/'cache/features-000000-g0000.bin').read_bytes(),before)
            self.assertEqual(load_cache(cache)[0]['rows'][0]['feature_file']['name'],'features-000000-g0001.bin')

    def test_real_dpr_sdpa_and_bge_eager_layer_classes_with_native_bridge(self):
        for kind in ('dpr','bge'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d,\
                    patch('applications.rag.top8_backbone_v001.Backbone',NativeFixture):
                root=Path(d);train,evaluation=populations(root)
                cache=extract_bounded(kind,root,train,evaluation,root/'cache',label_classes=2,sequence_cache_bytes=0)
                self.assertFalse(list(cache.parent.glob('prefix-*.bin')))
                self.assertFalse((cache.parent/'features.bin').exists())
                self.assertEqual(len(list(cache.parent.glob('question-*-v2.json'))),4)
                manifest,features=load_cache(cache);reader=FrozenSequences(cache.parent,manifest)
                try:
                    record=manifest['rows'][0];entry=record['sequences'][0]
                    state=torch.get_rng_state().clone()
                    row=reader.read(entry)
                    self.assertTrue(torch.equal(state,torch.get_rng_state()))
                    b=NativeFixture(kind,root)
                    native=b.head(b.bridge(b.last(row['hidden'],attention_mask=row['mask'])[0][:,0])).squeeze(-1)
                    self.assertTrue(torch.equal(native,row['native_scores']))
                    self.assertEqual(features[record['id']]['readout'].shape,(50,64))
                    self.assertFalse(torch.cuda.is_initialized())
                finally:reader.close()

    def test_full_fidelity_logit_gradient_and_rank_equal_after_cache_roundtrip(self):
        from transformers import BertConfig
        from transformers.models.bert.modeling_bert import BertLayer
        from applications.rag.top8_methods_v001 import Fidelity
        torch.manual_seed(31)
        config=BertConfig(hidden_size=64,num_attention_heads=1,intermediate_size=128,
                          hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        config._attn_implementation='sdpa'
        layer=BertLayer(config).eval().requires_grad_(False)
        head=nn.Linear(64,1).requires_grad_(False);hidden=torch.randn(3,7,64)
        mask=torch.zeros(3,1,1,7);mask[1,:,:,5:]=torch.finfo(torch.float32).min
        full=dict(hidden=hidden,mask=mask,readout=layer(hidden,attention_mask=mask)[0][:,0].detach())
        from applications.rag.top8_cache_v002 import load_tensor
        restored=load_tensor(object_bytes(full))
        for arm in ('HEAD','MLP','GIVENS','PRODUCT','QUANTUM'):
            with self.subTest(arm=arm):
                a=Fidelity(head,copy.deepcopy(layer),nn.Identity(),arm,17)
                b=copy.deepcopy(a)
                sa=a(full);sb=b(restored)
                self.assertTrue(torch.equal(sa,sb))
                self.assertEqual(torch.argsort(sa).tolist(),torch.argsort(sb).tolist())
                loss_a=torch.nn.functional.softplus(-(sa[0]-sa[1]))
                loss_b=torch.nn.functional.softplus(-(sb[0]-sb[1]))
                loss_a.backward();loss_b.backward()
                for p,q in zip(a.parameters(),b.parameters()):
                    self.assertEqual(p.grad is None,q.grad is None)
                    if p.grad is not None:self.assertTrue(torch.equal(p.grad,q.grad))

    def test_native_features_fp32_and_full_prefix_regeneration_hashes(self):
        with tempfile.TemporaryDirectory() as d,patch('applications.rag.top8_backbone_v001.Backbone',FakeBackbone):
            root=Path(d);train,evaluation=populations(root)
            cache=extract_bounded('dpr',root,train,evaluation,root/'cache',label_classes=2,sequence_cache_bytes=100000)
            manifest,features=load_cache(cache);self.assertIsInstance(features,FeatureRows)
            record=manifest['rows'][0];entry=record['sequences'][0]
            reader=FrozenSequences(cache.parent,manifest)
            try:
                actual=reader.read(entry);same=reader.read(entry)
                self.assertTrue(torch.equal(actual['hidden'],same['hidden']))
                self.assertEqual(reader.regenerated,1);self.assertEqual(reader.hits+reader.memory_hits,1)
                package=torch.load(checked_member(cache.parent,entry['tokens']),weights_only=True)
                b=FakeBackbone('dpr',root);captured,logits,_,_=capture_chunk(b,package[entry['chunk']]['encoded'])
                self.assertTrue(torch.equal(actual['hidden'],captured['prefix']))
                original=features[record['id']]
                for name in ('depth0','context','ffn_residual','z','final'):
                    self.assertTrue(torch.equal(original[name][:8],captured[name]))
                    self.assertEqual(original[name].dtype,torch.float32)
                self.assertTrue(torch.equal(original['readout'][:8],b.bridge(captured['final'])))
                # A changed tensor never passes merely because scalar scores match.
                changed=copy.deepcopy(entry);changed['signature']['hidden_sha256']='0'*64
                with self.assertRaisesRegex(RuntimeError,'hashes differ'):reader.read(changed)
            finally:reader.close()

    def test_question_commit_resume_skips_only_verified_complete_questions(self):
        with tempfile.TemporaryDirectory() as d,patch('applications.rag.top8_backbone_v001.Backbone',FakeBackbone):
            root=Path(d);train,evaluation=populations(root);calls=0
            def stop():
                nonlocal calls
                calls+=1
                if (root/'cache/question-000000-v2.json').exists():raise InterruptedError('intentional interruption')
            with self.assertRaises(InterruptedError):
                extract_bounded('dpr',root,train,evaluation,root/'cache',stop,label_classes=2)
            self.assertFalse((root/'cache/cache.json').exists())
            before=(root/'cache/question-000000-v2.json').read_bytes();FakeBackbone.calls=0
            cache=extract_bounded('dpr',root,train,evaluation,root/'cache',label_classes=2,resume=True)
            self.assertEqual(FakeBackbone.calls,3*7)
            self.assertEqual((root/'cache/question-000000-v2.json').read_bytes(),before)
            self.assertEqual(len(load_cache(cache)[0]['rows']),4)
            with self.assertRaisesRegex(RuntimeError,'resume binding'):
                extract_bounded('dpr',root,train,evaluation,root/'cache',label_classes=3,resume=True)

    def test_feature_memory_lru_aux_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[]
            for i in range(3):
                r=tensor_file(root,f'f{i}.bin',dict(readout=torch.full((2,8),float(i))))
                rows.append(dict(id=str(i),retained_indices=[0,1],feature_file=r))
            store=FeatureRows(root,rows,maximum=64)
            store.set_aux('0','mixed',{'Q':torch.ones(2)})
            for key in ('0','1','2','0'):
                self.assertTrue(torch.equal(store[key]['readout'],torch.full((2,8),float(key))))
                self.assertLessEqual(store.resident,64)
            self.assertIn('mixed',store['0'])
            (root/'f1.bin').write_bytes(b'altered')
            with self.assertRaisesRegex(RuntimeError,'SHA mismatch'):store['1']

    def test_verified_legacy_chunks_reused_without_regeneration_or_deletion(self):
        with tempfile.TemporaryDirectory() as d,patch('applications.rag.top8_backbone_v001.Backbone',FakeBackbone):
            root=Path(d);train,evaluation=populations(root)
            cache=extract_bounded('dpr',root,train,evaluation,root/'cache',label_classes=2)
            manifest,_=load_cache(cache);entry=manifest['rows'][0]['sequences'][0]
            reader=FrozenSequences(cache.parent,dict(manifest,sequence_cache_bytes=0))
            value=reader.read(entry);reader.close()
            tensor_file(cache.parent,entry['legacy_name'],dict(value,positions=entry['positions']))
            path=cache.parent/entry['legacy_name'];before=path.read_bytes()
            reader=FrozenSequences(cache.parent,manifest)
            try:
                self.assertEqual(tensor_hash(reader.read(entry)['hidden']),entry['signature']['hidden_sha256'])
                self.assertEqual(reader.regenerated,0);self.assertEqual(reader.legacy_hits,1)
                self.assertEqual(path.read_bytes(),before)
            finally:reader.close()


if __name__=='__main__':unittest.main()
