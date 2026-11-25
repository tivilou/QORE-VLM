"""Question-sharded native FFN task adapter for the qualified local Born factory.

No evaluation target reads, full-sequence dumps, fitting, or network clients.
"""
from collections import OrderedDict
import hashlib,json,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need,digest,inputs,checked_member,tensor_file
from applications.rag.local_born_io_v001 import immutable_json as json_file
from applications.rag.top8_backbone_v001 import Backbone
from applications.rag.toy100_ffn_export_v001 import frozen_tail
from applications.rag.toy100_output_frame_v001 import output_frames


def populations(data,counts):
    data=Path(data);roles=json.loads((data/'roles.json').read_bytes())
    train=inputs(data/'train.jsonl',True);evaluation=inputs(data/'evaluation.inputs.jsonl',False)
    rows=train+evaluation
    need(len(train)==counts['train'] and len(evaluation)==sum(v for k,v in counts.items() if k!='train'),'complete declared population')
    need(len({x['id'] for x in rows})==len(rows) and set(roles)=={x['id'] for x in rows},'unique complete role identities')
    need(all(roles[x['id']]=='train' for x in train) and all(roles[x['id']]!='train' for x in evaluation),'training role boundary')
    seen={}
    for row in rows:
        role=roles[row['id']]
        need(role in counts,'declared role only')
        for key in [row['component_id'],'q:'+hashlib.sha256(' '.join(row['question'].casefold().split()).encode()).hexdigest()]:
            need(key not in seen or seen[key]==role,'component/question cross-role leakage');seen[key]=role
        if role=='train':need(set(row['labels'])=={p['id'] for p in row['paragraphs']} and all(type(y)is int and y in (0,1) for y in row['labels'].values()),'complete binary TRAIN labels')
    need({k:sum(roles[x['id']]==k for x in rows) for k in counts}==counts,'exact role counts')
    return rows,roles


def input_identity(row):
    return hashlib.sha256(json.dumps({k:row[k] for k in ['id','component_id','question','paragraphs']},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def new_tensor(root,stem,value):
    generation=0
    while (root/(stem+f'-g{generation}.bin')).exists() or (root/(stem+f'-g{generation}.bin.part')).exists():generation+=1
    return tensor_file(root,stem+f'-g{generation}.bin',value)


def extract(kind,model_root,rows,roles,root,source_key,emit=lambda **kw:None):
    """One native forward per original8-candidate block; resume sealed questions."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    binding=dict(kind=kind,source=source_key,inputs=[input_identity(r) for r in rows],roles=[roles[r['id']] for r in rows])
    bpath=root/'binding.json'
    if bpath.exists():need(json.loads(bpath.read_bytes())==binding,'native cache binding changed')
    else:need(all(p.name.startswith('binding.json.') and p.name.endswith('.part') for p in root.iterdir()),'unbound cache not empty');json_file(root,'binding.json',binding)
    if (root/'cache.json').exists():
        value=json.loads((root/'cache.json').read_bytes());need(value['binding']==binding,'cache manifest identity')
        checked_member(root,value['metadata'])
        need(len(value['questions'])==len(rows),'complete cached population')
        for i,record in enumerate(value['questions']):
            need(record['index']==i and record['input_sha256']==binding['inputs'][i] and record['id']==rows[i]['id'] and
                record['candidate_ids']==[p['id'] for p in rows[i]['paragraphs']],'cache exact row/candidate order')
            checked_member(root,record['tensor'])
        return value
    backbone=Backbone(kind,model_root)
    cpu=lambda module:{n:v.detach().cpu() for n,v in module.state_dict().items()}
    metadata=dict(original_head=cpu(backbone.head),ffn_dense=cpu(backbone.last.output.dense),
        ffn_normalization=cpu(backbone.last.output.LayerNorm),ffn_bridge=cpu(backbone.bridge.dense) if kind=='bge' else None,
        ffn_eps=backbone.last.output.LayerNorm.eps,model_assets=backbone.asset_records)
    need(type(backbone.last.intermediate.intermediate_act_fn).__name__=='GELUActivation','native exact GELU')
    questions=[];maximum=0.
    for i,row in enumerate(rows):
        need(shutil.disk_usage(root).free>=8*1024**3,'Native cache needs8GiB free; expand server filesystem and resume')
        receipt=root/f'question-{i:05d}.json';identity=binding['inputs'][i]
        if receipt.exists():
            record=json.loads(receipt.read_bytes())
            need(record['index']==i and record['input_sha256']==identity and record['id']==row['id'] and record['binding_sha256']==digest(bpath),'sealed question identity')
            checked_member(root,record['tensor'])
        else:
            parts=[];tokens=[]
            for start in range(0,50,8):
                encoded=backbone.tokenize(row['question'],[p['text'] for p in row['paragraphs'][start:start+8]]).to('cuda');captured={}
                def before(module,args):
                    need('z' not in captured,'single native intermediate capture')
                    captured.update(z=module.dense(args[0][:,0]).detach(),residual=args[0][:,0].detach())
                def after(module,args,out):captured['features']=backbone.bridge(out[0][:,0]).detach()
                hs=[backbone.last.intermediate.register_forward_pre_hook(before),backbone.last.register_forward_hook(after)]
                try:
                    with torch.no_grad():native=backbone.logits(encoded)
                finally:
                    for handle in hs:handle.remove()
                with torch.no_grad():
                    tail=frozen_tail(captured['z'],captured['residual'],metadata['ffn_dense'],metadata['ffn_normalization'],metadata['ffn_bridge'],metadata['ffn_eps'])
                    expected=captured['features'];error=float((tail-expected).abs().max())
                    need(torch.allclose(tail,expected,atol=2e-5,rtol=2e-5),'native FFN/bridge numeric replay')
                    need(torch.allclose(backbone.head(expected).squeeze(-1),native,atol=2e-5,rtol=2e-5),'original head identity')
                parts.append(dict(z=captured['z'].cpu(),residual=captured['residual'].cpu(),features=expected.cpu(),native=native.detach().cpu()))
                tokens.append(dict(start=start,shape=list(encoded['input_ids'].shape),ids_sha256=hashlib.sha256(encoded['input_ids'].cpu().numpy().tobytes()).hexdigest(),mask_sha256=hashlib.sha256(encoded['attention_mask'].cpu().numpy().tobytes()).hexdigest()))
                maximum=max(maximum,error)
                del captured,expected,tail,native,encoded
            payload={n:torch.cat([p[n] for p in parts]) for n in ['z','residual','features','native']}
            need(all(len(t)==50 and bool(torch.isfinite(t).all()) for t in payload.values()),'complete finite question FFN record')
            record=dict(index=i,id=row['id'],component_id=row['component_id'],role=roles[row['id']],input_sha256=identity,
                candidate_ids=[p['id'] for p in row['paragraphs']],tensor=new_tensor(root,f'question-{i:05d}',payload),tokens=tokens,binding_sha256=digest(bpath))
            json_file(root,receipt.name,record)
        questions.append(record);emit(phase='提取 '+kind,completed=i+1,total=len(rows),unit='questions')
    del backbone
    torch.cuda.empty_cache()
    metadata['ffn_frames']=output_frames(metadata['ffn_dense']['weight'])
    first=torch.load(checked_member(root,questions[0]['tensor']),weights_only=True)
    need(questions[0]['role']=='train','factory diagnostic uses first TRAIN only')
    # Only these two inherited bookkeeping buffers use this first query. They
    # are unused by HEAD/local-mode scoring; real-data parity tests cover this.
    metadata.update(ffn_z=dict(train=first['z'][None]),native=dict(train=first['native'][None]),
        features=dict(train=first['features'][None]),portable_bookkeeping='First TRAIN query only; inherited scale buffers unused by all147arms. Not full TRAIN statistics.')
    meta=new_tensor(root,'metadata',metadata)
    value=dict(schema='qduet.local-born-native-cache.v1',binding=binding,metadata=meta,questions=questions,maximum_tail_error_new_questions=maximum,
        input_chunk=8,features_dtype='torch.float32',full_sequences_stored=False,evaluation_labels_opened=False)
    json_file(root,'cache.json',value);return value


class QuestionStore:
    """Hash-verified bounded CPU LRU; no caching of full sequence tensors."""
    def __init__(self,roots,manifests,limit=512*1024**2):
        self.roots={k:Path(v) for k,v in roots.items()};self.manifests=manifests;self.limit=limit;self.cache=OrderedDict();self.bytes=0
        self.packed={k:torch.load(checked_member(self.roots[k],m['metadata']),weights_only=True) for k,m in manifests.items()}
    def one(self,k,i):
        key=k,i
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key][0]
        record=self.manifests[k]['questions'][i];v=torch.load(checked_member(self.roots[k],record['tensor']),weights_only=True)
        need(set(v)=={'z','residual','features','native'} and all(len(t)==50 and bool(torch.isfinite(t).all()) for t in v.values()),'finite complete cached50')
        size=sum(t.numel()*t.element_size() for t in v.values())
        while self.cache and self.bytes+size>self.limit:
            _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
        if size<=self.limit:self.cache[key]=(v,size);self.bytes+=size
        return v
    def batch(self,indices,device='cuda'):
        features={};native={};coords={}
        for k in ['dpr','bge']:
            vs=[self.one(k,i) for i in indices]
            features[k]=torch.stack([x['features'] for x in vs]).to(device)
            native[k]=torch.stack([x['native'] for x in vs]).to(device)
            coords[k]=torch.cat((torch.stack([x['z'] for x in vs]),torch.stack([x['residual'] for x in vs])),-1).to(device)
        return features,native,coords
