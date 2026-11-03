"""Run the selected eight mechanisms on native DPR and BGE, with shared caches.

All prediction cells are sealed before reading evaluation labels. Failed cells
remain failed, missing labels stay unknown, and mixed precision is not used.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from applications.rag.qvalue_portable_v001 import need, digest, json_file, tensor_file, checked_member


def load_cache(path):
    import torch
    p=Path(path);m=json.loads(p.read_bytes())
    need(m['schema']=='top8.native-cache.v1' and m['evaluation_labels_present'] is False,'native label-free cache')
    rows=m['rows'];need(len({r['id'] for r in rows})==len(rows),'unique cache identities')
    for r in rows:
        need(len(r['candidate_ids'])==50 and len(set(r['candidate_ids']))==50,'same50 identities')
        if r['role']=='training':need(len(r['targets'])==len(r['retained_indices']),'reviewed training supervision')
        else:need(r['targets'] is None and r['retained_indices']==list(range(50)),'target-free complete50 evaluation')
    record=next(r for r in m['files'] if r['name']=='features.bin')
    features=torch.load(checked_member(p.parent,record),map_location='cpu',weights_only=True)
    need(set(features)=={r['id'] for r in rows},'complete cache population')
    for r in rows:
        need(all(len(v)==len(r['retained_indices']) and bool(torch.isfinite(v).all()) for v in features[r['id']].values()),'finite complete shared native values')
    return m,features


def gpu_row(features, index=None):
    def convert(v):
        if isinstance(v,dict):return {k:convert(x) for k,x in v.items()}
        return (v if index is None else v[index]).cuda()
    return {k:convert(v) for k,v in features.items()}


def sequence_rows(root, record, features, checkpoint):
    """Full-value exhaustion of every original retained chunk, in original order."""
    import torch
    offset=0
    for entry in record['sequences']:
        checkpoint();p=checked_member(root,entry);x=torch.load(p,map_location='cuda',weights_only=True)
        n=len(x['positions']);out=gpu_row(features,slice(offset,offset+n));out.update(hidden=x['hidden'],mask=x['mask'])
        expected=record['retained_indices'][offset:offset+n]
        need(expected==x['positions'],'complete sequence/native feature alignment')
        yield out,x['native_scores'];offset+=n
    need(offset==len(record['retained_indices']),'full sequence reader exhausted')


def scores_for(model, idea, record, features, root, checkpoint):
    import torch
    if idea!='FIDELITY':return model(gpu_row(features))
    scores=[]
    for row,native in sequence_rows(root,record,features,checkpoint):scores.append(model(row))
    return torch.cat(scores) if scores else model.head.weight.sum().reshape(1)[:0]


def loss_for(model,idea,record,features,root,checkpoint):
    import torch
    from torch.nn import functional as F
    from applications.rag.top8_methods_v001 import objective
    y=torch.tensor(record['targets'],device='cuda',dtype=torch.long)
    if idea!='FIDELITY':return objective(model,idea,gpu_row(features),y)
    s=scores_for(model,idea,record,features,root,checkpoint);mask=y[:,None]>y[None,:];count=mask.sum()
    return (F.softplus(-(s[:,None]-s[None,:]))*mask).sum()/count.clamp_min(1),count


def train_cell(idea,arm,seed,manifest,features,cache_root,output,epochs,steps_limit,checkpoint,journal=None):
    import torch
    from applications.rag.top8_backbone_v001 import load_modules
    from applications.rag.top8_methods_v001 import build
    train=[r for r in manifest['rows'] if r['role']=='training'];dev=[r for r in manifest['rows'] if r['role']=='development']
    layer,bridge,head=load_modules(manifest,cache_root)
    values=torch.cat([features[r['id']]['readout'] for r in train])
    mean=values.mean(0).cuda();std=values.std(0,unbiased=False);std=torch.where(std>1e-8,std,torch.ones_like(std)).cuda()
    model=build(idea,arm,seed,head,layer,bridge,(mean,std)).cuda()
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    need(parameters and len({id(p) for p in parameters.values()})==len(parameters),'exact trainable parameter ownership')
    optimizer=torch.optim.AdamW(list(parameters.values()),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    max_null=0.;native_math_error=0.
    # Fixed first two complete training questions. No outcome-selected probes.
    with torch.no_grad():
        for r in train[:2]:
            f=features[r['id']]
            if idea=='FIDELITY':
                for row,native in sequence_rows(cache_root,r,f,checkpoint):
                    actual=model(row);error=float((actual-native).abs().max())
                    need(torch.allclose(actual,native,atol=2e-5,rtol=2e-6),'fixed native/zero/frozen-reference attention parity')
                    max_null=max(max_null,error);native_math_error=max(native_math_error,error)
            else:
                expected=head(f['readout'].cuda()).squeeze(-1);actual=model(gpu_row(f))
                need(torch.allclose(actual,expected,atol=2e-5,rtol=2e-6),'fixed native/reference null')
                max_null=max(max_null,float((actual-expected).abs().max()))
    records=[];generator=torch.Generator().manual_seed(seed);step=0;started=time.monotonic()
    saved=journal.load() if journal is not None else None
    if saved is not None:
        need(set(saved['parameters'])==set(parameters),'complete recovery parameter ownership')
        with torch.no_grad():
            for n,p in parameters.items():p.copy_(saved['parameters'][n].to(p.device))
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step=saved['step'];records=saved['records']
        need(len(records)==step,'complete recovery training history')
    for epoch in range(epochs):
        if saved is not None and epoch<saved['epoch']:continue
        restoring=saved is not None and epoch==saved['epoch']
        order=saved['order'] if restoring else torch.randperm(len(train),generator=generator).tolist()
        need(sorted(order)==list(range(len(train))),'complete training permutation')
        for start in range(0,len(order),8):
            if restoring and start<saved['next_start']:continue
            checkpoint();batch=[train[i] for i in order[start:start+8]];optimizer.zero_grad(set_to_none=True)
            total=0.;pairs=0;without=0
            # Every complete question contributes 1/B, including zero-pair cases.
            for r in batch:
                loss,count=loss_for(model,idea,r,features[r['id']],cache_root,checkpoint)
                need(bool(torch.isfinite(loss)),'finite training-only objective')
                (loss/len(batch)).backward();total+=float(loss.detach())/len(batch);pairs+=int(count);without+=int(count==0)
            need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters.values()),'finite data gradients')
            active={n:float(p.grad.abs().max()) for n,p in parameters.items() if p.grad is not None}
            need(active and (pairs==0 or max(active.values())>0),'nonzero actual objective gradient when pairs exist')
            norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.);optimizer.step();step+=1
            need(all(bool(torch.isfinite(p).all()) for p in parameters.values()),'finite trained parameters')
            records.append(dict(step=step,loss=total,ordered_pairs=pairs,without_ordered_pairs=without,gradient_norm=float(norm)))
            if journal is not None and (step%32==0 or start+8>=len(order) or (steps_limit and step>=steps_limit)):
                journal.save(dict(step=step,epoch=epoch,next_start=start+8,order=order,
                    parameters={n:p.detach().cpu().clone() for n,p in parameters.items()},
                    optimizer=optimizer.state_dict(),generator=generator.get_state(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),records=records))
            if step%32==0:print(json.dumps(dict(stage='training',idea=idea,arm=arm,seed=seed,step=step)),flush=True)
            if steps_limit and step>=steps_limit:break
        if steps_limit and step>=steps_limit:break
    checkpoint();key=f'{idea}-{arm}-s{seed}';files=[]
    compact={n:p.detach().cpu().clone() for n,p in parameters.items()}
    files.append(tensor_file(output,'checkpoint-'+key+'.bin',compact))
    # Rebuild from fixed native modules and seed, then load only owned parameters.
    replay=build(idea,arm,seed,head,layer,bridge,(mean,std)).cuda()
    restored={n:p for n,p in replay.named_parameters() if p.requires_grad}
    need(set(restored)==set(compact),'complete compact trainable schema')
    with torch.no_grad():
        for n,p in restored.items():p.copy_(compact[n].to(p.device))
    predictions=[];replay_error=0.
    with torch.no_grad():
        for r in dev:
            checkpoint();f=features[r['id']];s=scores_for(model,idea,r,f,cache_root,checkpoint)
            s2=scores_for(replay,idea,r,f,cache_root,checkpoint)
            need(s.shape==(50,) and bool(torch.isfinite(s).all()) and torch.equal(s,s2),'all50 prediction/final checkpoint bit replay')
            replay_error=max(replay_error,float((s-s2).abs().max()));values=s.cpu().tolist()
            ranking=sorted(range(50),key=lambda i:(-values[i],i))
            predictions.append(dict(id=r['id'],candidate_ids=r['candidate_ids'],scores=values,ranking=ranking))
    files.append(json_file(output,'predictions-'+key+'.json',dict(rows=predictions)))
    files.append(json_file(output,'training-'+key+'.json',dict(records=records)))
    result=dict(idea=idea,arm=arm,seed=seed,steps=step,training_questions=len(train),evaluation_questions=len(dev),
        trainable_parameters=sum(p.numel() for p in parameters.values()),max_initial_null_error=max_null,
        representative_native_math_error=native_math_error,prediction_replay_error=replay_error,
        seconds=time.monotonic()-started,files=files,sample_trace_v2='DEFERRED_NOT_PASSED')
    del optimizer,model,replay,layer,bridge,head;gc.collect();torch.cuda.empty_cache()
    return result


def run_matrix(cache_paths,output,seeds,ideas,epochs=4,canary=False,checkpoint=lambda:None):
    import torch
    from applications.rag.top8_methods_v001 import ARMS,mixed_banks
    output=Path(output);output.mkdir(parents=True,exist_ok=False);results={};files=[];failures=[]
    for path in cache_paths:
        manifest,features=load_cache(path);root=Path(path).parent;kind=manifest['backbone']
        train=[r for r in manifest['rows'] if r['role']=='training'];dev=[r for r in manifest['rows'] if r['role']=='development']
        need(train and dev,'both roles present')
        if canary:
            selected=train[:2]+dev[:2];manifest=dict(manifest,rows=selected)
            features={r['id']:features[r['id']] for r in selected}
        elif 'MIXED_DEPTH' in ideas:
            bank=mixed_banks(features,manifest['rows'],checkpoint)
            files.append(tensor_file(output,'banks-'+kind+'.bin',bank));del bank;torch.cuda.empty_cache()
        native=[dict(id=r['id'],candidate_ids=r['candidate_ids'],scores=r['native_scores'],ranking=sorted(range(50),key=lambda i:(-r['native_scores'][i],i))) for r in manifest['rows'] if r['role']=='development']
        files.append(json_file(output,'predictions-'+kind+'-NATIVE.json',dict(rows=native)))
        # Run cached-feature methods first; sequence-attention is the last cell family.
        execution_order=[x for x in ideas if x!='FIDELITY']+(['FIDELITY'] if 'FIDELITY' in ideas else [])
        for idea in execution_order:
            if canary and idea=='MIXED_DEPTH':
                # Bank fit needs five populated folds: no two-question fake pass.
                failures.append(dict(backbone=kind,idea=idea,stage='canary',status='DEFERRED_BANK_FIT'))
                continue
            for seed in seeds:
                for arm in ARMS[idea]:
                    checkpoint();cell=output/(kind+'-'+idea+'-'+arm+'-s'+str(seed));cell.mkdir()
                    try:
                        result=train_cell(idea,arm,seed,manifest,features,root,cell,1 if canary else epochs,1 if canary else None,checkpoint)
                        files.extend(dict(r,path=str(cell.name+'/'+r['name'])) for r in result['files'])
                        results[cell.name]=result
                        json_file(cell,'complete.json',result)
                    except Exception as e:
                        # Preserve concrete failures, never retry or silently remove a control.
                        failure=dict(backbone=kind,idea=idea,arm=arm,seed=seed,error_type=type(e).__name__,message=str(e)[:1000])
                        json_file(cell,'failed.json',failure);failures.append(failure)
                        gc.collect();torch.cuda.empty_cache()
                        if isinstance(e,(KeyboardInterrupt,MemoryError)):raise
        del features;gc.collect();torch.cuda.empty_cache()
    # Manifest inventories all predictions/checkpoints before target access.
    seal=dict(schema='top8.joint-predictions.v1',seeds=seeds,ideas=ideas,canary=canary,
        requested_backbones=len(cache_paths),results=results,failures=failures,files=files,
        all_predictions_before_evaluation_labels=True)
    json_file(output,'joint-prediction-seal.json',seal)
    return seal


def evaluate(output,labels_path):
    from statistics import mean
    output=Path(output);seal=json.loads((output/'joint-prediction-seal.json').read_bytes())
    need(seal['all_predictions_before_evaluation_labels'] is True,'joint seal precedes labels')
    # First verify every sealed file, then and only then open the sparse targets.
    predictions={}
    for r in seal['files']:
        path=output/r.get('path',r['name']);need(path.resolve().is_relative_to(output.resolve()) and path.is_file() and path.stat().st_size==r['size'] and digest(path)==r['sha256'],'sealed prediction/checkpoint bytes')
        if r['name'].startswith('predictions-'):
            rows=json.loads(path.read_bytes())['rows'];predictions[path.relative_to(output).as_posix()]=rows
    labels=json.loads(Path(labels_path).read_bytes());metrics={}
    for key,rows in predictions.items():
        useful=direct=unknown=directq=0
        for r in rows:
            known=labels.get(r['id'],{});need(set(known)<=set(r['candidate_ids']) and all(type(y)is int and y in (0,1,2) for y in known.values()),'sparse exact evaluation labels')
            selected=[known.get(r['candidate_ids'][i]) for i in r['ranking'][:5]]
            useful+=sum(y in (1,2) for y in selected);direct+=sum(y==2 for y in selected);unknown+=sum(y is None for y in selected);directq+=int(2 in selected)
        metrics[key]=dict(useful_lower=useful,useful_upper=useful+unknown,direct_lower=direct,direct_upper=direct+unknown,direct_questions_lower=directq,unknown=unknown,questions=len(rows),denominator=5*len(rows))
    ranking=[]
    for idea in seal['ideas']:
        deltas=[];by_model={};complete=True
        for kind in ('dpr','bge'):
            base=metrics.get('predictions-'+kind+'-NATIVE.json')
            cells=[(k,v) for k,v in metrics.items() if k.startswith(kind+'-'+idea+'-QUANTUM-s')]
            if not base or len(cells)!=len(seal['seeds']):complete=False;continue
            lower=mean(v['useful_lower']-base['useful_upper'] for k,v in cells)
            upper=mean(v['useful_upper']-base['useful_lower'] for k,v in cells)
            by_model[kind]=dict(mean_native_delta_lower=lower,mean_native_delta_upper=upper,seeds=len(cells))
            deltas.append((lower,upper))
        ranking.append(dict(idea=idea,by_model=by_model,rankable=complete,
            equal_weight_native_delta_lower=mean(v[0] for v in deltas) if complete else None,
            equal_weight_native_delta_upper=mean(v[1] for v in deltas) if complete else None))
    ranking.sort(key=lambda r:(not r['rankable'],-r['equal_weight_native_delta_lower'] if r['rankable'] else 0,r['idea']))
    result=dict(schema='top8.evaluation.v1',joint_seal_sha256=digest(output/'joint-prediction-seal.json'),labels_sha256=digest(labels_path),metrics=metrics,ranking=ranking,
        ranking_rule='mean of DPR and BGE seed-mean useful improvements over each own native; lower bounds; incomplete cells unranked',
        classical_controls_separate=True,quantum_attribution='not_established',exposed_population=True)
    json_file(output,'evaluation.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dpr-model',type=Path);p.add_argument('--bge-model',type=Path)
    p.add_argument('--train',type=Path);p.add_argument('--eval-inputs',type=Path);p.add_argument('--eval-labels',type=Path)
    p.add_argument('--cache',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261102,20261103,20261104])
    p.add_argument('--ideas',nargs='+');p.add_argument('--canary',action='store_true');p.add_argument('--extract-only',action='store_true')
    args=p.parse_args(argv)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch,transformers
    need(torch.cuda.is_available() and transformers.__version__=='4.44.2','CUDA and pinned Transformers4.44.2')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from applications.rag.top8_backbone_v001 import extract
    from applications.rag.top8_methods_v001 import IDEAS
    ideas=args.ideas or list(IDEAS);need(set(ideas)<=set(IDEAS) and len(set(ideas))==len(ideas),'selected registry only')
    need(not args.output.exists(),'fresh output root');args.output.mkdir(parents=True)
    caches=list(args.cache)
    for kind,root in [('dpr',args.dpr_model),('bge',args.bge_model)]:
        if root:
            need(args.train and args.eval_inputs,'explicit separate raw conditions')
            caches.append(extract(kind,root,args.train,args.eval_inputs,args.output/('cache-'+kind)))
    need(caches,'at least one backbone cache')
    if args.extract_only:return 0
    run_matrix(caches,args.output/'matrix',args.seeds,ideas,canary=args.canary)
    if args.eval_labels:evaluate(args.output/'matrix',args.eval_labels)
    return 0


if __name__=='__main__':raise SystemExit(main())
