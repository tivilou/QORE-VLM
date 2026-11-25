"""Portable task runner, reusing the exact147 factory, loss and full-state Journal.

This file does not acquire assets or open validation/test targets. The coordinator
joins targets only after the whole fixed campaign prediction seal is committed.
"""
import gc,json,os,shutil,time
from pathlib import Path
import torch
from torch.nn import functional as F
from applications.rag.qvalue_portable_v001 import need,digest,checked_member
from applications.rag.local_born_io_v001 import RecoveryJournal as Journal,immutable_json as json_file,member,verify_seal,verify_campaign
from applications.rag.toy100_local_born_v001 import factory,ARMS
from applications.rag.toy100_shared_study_v001 import combined_objective,model_batch_loss,frozen_parameters,verify_activity
from applications.rag.local_born_cache_v001 import QuestionStore,extract,populations
from scripts.smallset_events import emit,allocator_event


def policy():
    need(torch.cuda.is_available(),'CUDA GPU required')
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def schedule(n,epochs,seed):
    need(n>0 and n%8==0 and epochs>0,'complete logical eight-question batches')
    g=torch.Generator(device='cpu').manual_seed(seed+17)
    return [batch for _ in range(epochs) for batch in torch.randperm(n,generator=g).reshape(-1,8).tolist()]


def score(module,store,indices):
    features,native,coords=store.batch(indices)
    with torch.no_grad():reference=module.references(coords,native,features)
    original={k:F.linear(features[k],store.packed[k]['original_head']['weight'].to(features[k]),store.packed[k]['original_head']['bias'].to(features[k])).squeeze(-1) for k in features}
    return module(features,native,original,coords,reference),native


def checked_cell(folder,binding,evaluation_count,steps):
    p=folder/'result.json'
    if not p.exists():return None
    v=json.loads(p.read_bytes());need(v['binding']==binding and v['step']==steps,'completed cell binding/endpoint')
    for k in ['dpr','bge']:
        pred=json.loads(checked_member(folder,v['predictions'][k]).read_bytes());need(len(pred['rows'])==evaluation_count,'completed cell full evaluation')
    journal=Journal(folder/'journal',binding);need(journal.latest_record()['step']==steps,'final full state present')
    return v


def run_cell(folder,binding,store,rows,ntrain,arm,seed,epochs,checkpoint_every=100):
    """Identical coupled gradients; bounded sharded input replaces all-GPU toy bank."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    batches=schedule(ntrain,epochs,seed);steps=len(batches);job=f'local-born-{arm}-s{seed}'
    done=checked_cell(folder,binding,len(rows)-ntrain,steps)
    if done is not None:emit('cell',job=job,state='computed');return done
    emit('cell',job=job,state='running');emit('device',device='cuda:0 / '+torch.cuda.get_device_name(),execution='FP32 native tail/head; FP64 exact147 critic; no TF32')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    module=factory(store.packed,arm,seed).cuda().eval();frozen=frozen_parameters(module)
    parameters=[p for p in module.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=.001,weight_decay=0.,foreach=False)
    with torch.no_grad():
        initial,native=score(module,store,list(range(8)));null=max(float((initial[k]-native[k]).abs().max()) for k in native)
        need(null<=1e-5,'initial native/null on fixed first8 TRAIN questions')
    active={n:0. for n,p in module.named_parameters() if p.requires_grad};last_loss=None;start=0
    journal=Journal(folder/'journal',binding);saved=journal.load()
    if saved is not None:
        start=saved['step'];need(0<start<=steps,'valid resume step')
        module.load_state_dict(saved['module']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        active=saved['gradient_maxima'];last_loss=saved['loss']
    targets=[torch.tensor([r['labels'][p['id']] for p in r['paragraphs']],device='cuda',dtype=torch.long) for r in rows[:ntrain]]
    shared_probe=saved.get('shared_gradient_probe') if saved is not None else None
    begun=time.monotonic();emit('phase',phase='训练')
    for index in range(start,steps):
        batch=batches[index];opt.zero_grad(set_to_none=True);values,_=score(module,store,batch)
        individual={k:model_batch_loss(k,batch,z,targets) for k,z in values.items()};value=combined_objective(individual)
        need(bool(torch.isfinite(value)),'finite coupled TRAIN objective')
        if index==0 and module.shared is not None:
            parameter=module.gradient_probe_parameter();probes={k:torch.autograd.grad(v,parameter,retain_graph=True)[0] for k,v in individual.items()}
            need(all(float(g.abs().max())>0 for g in probes.values()),'both model data gradients active')
            expected=(probes['dpr']+probes['bge'])/2;shared_probe={k:float(g.abs().max()) for k,g in probes.items()}
        value.backward()
        if index==0 and module.shared is not None:need(torch.allclose(parameter.grad,expected,atol=1e-6,rtol=1e-4),'exact coupled gradient ownership')
        for name,p in module.named_parameters():
            if p.requires_grad:
                need(p.grad is not None and bool(torch.isfinite(p.grad).all()),'finite complete owned gradients')
                active[name]=max(active[name],float(p.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);opt.step();last_loss=float(value.detach());step=index+1
        if step%checkpoint_every==0 or step==steps:
            journal.save(dict(step=step,module=module.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),gradient_maxima=active,loss=last_loss,shared_gradient_probe=shared_probe))
        if step%20==0 or step==steps:
            need(shutil.disk_usage(folder).free>=8*1024**3,'Training needs8GiB free; expand filesystem and resume latest state')
            emit('progress',phase='训练',completed=step,total=steps,unit='steps',origin=start,metrics=dict(loss=last_loss),
                checkpoint_step=step if step%checkpoint_every==0 or step==steps else None);allocator_event()
    verify_activity(module,active,frozen)
    restored=factory(store.packed,arm,seed).cuda().eval();restored.load_state_dict(journal.load()['module'])
    with torch.no_grad():
        ids=list(range(ntrain,min(ntrain+8,len(rows))))
        old,_=score(module,store,ids);new,_=score(restored,store,ids)
        replay=max(float((old[k]-new[k]).abs().max()) for k in old);need(replay==0,'first complete evaluation batch restored exactly')
    del module,opt,saved,parameters,targets;gc.collect();torch.cuda.empty_cache()
    predictions={k:[] for k in ['dpr','bge']};emit('phase',phase='验证／测试统一预测（未读标签）')
    for begin in range(ntrain,len(rows),8):
        ids=list(range(begin,min(begin+8,len(rows))))
        with torch.no_grad():values,_=score(restored,store,ids)
        for k,values0 in values.items():
            need(bool(torch.isfinite(values0).all()),'finite complete evaluation scores')
            for i,scores in zip(ids,values0.cpu().tolist()):
                predictions[k].append(dict(id=rows[i]['id'],candidate_ids=[p['id'] for p in rows[i]['paragraphs']],scores=scores,
                    ranking=sorted(range(50),key=lambda j:(-scores[j],j))))
        emit('progress',phase='验证／测试统一预测（未读标签）',completed=ids[-1]+1-ntrain,total=len(rows)-ntrain,unit='questions');allocator_event()
    records={}
    for k,rs in predictions.items():
        name='predictions-'+k+'.json';path=folder/name
        # A prediction interrupted before result publication is checked against a
        # fresh final-checkpoint replay, never blindly adopted or overwritten.
        if path.exists():
            need(json.loads(path.read_bytes())==dict(rows=rs),'unsealed prediction final-checkpoint replay mismatch')
            records[k]=dict(name=name,size=path.stat().st_size,sha256=digest(path))
        else:records[k]=json_file(folder,name,dict(rows=rs))
    result=dict(binding=binding,arm=arm,seed=seed,step=steps,predictions=records,checkpoint=journal.latest_record(),
        null_first8_train=null,replay_first8_evaluation=replay,gradient_maxima=active,shared_gradient_probe=shared_probe,
        loss=last_loss,elapsed_this_attempt=time.monotonic()-begun,resumed_step=start,evaluation_labels_opened=False)
    json_file(folder,'result.json',result);emit('cell',job=job,state='computed');del restored;gc.collect();torch.cuda.empty_cache();return result


def execute(data,models,output,cache,protocol,source,minimum_free_gib=8):
    policy();need(protocol['logical_questions']==8 and list(ARMS)==protocol['arms'],'exact147 arm/batch roster');data=Path(data);output=Path(output);cache=Path(cache);output.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    need(shutil.disk_usage(cache.parent).free>=minimum_free_gib*1024**3,'Expand experiment filesystem: native question cache needs free space')
    rows,roles=populations(data,protocol['roles']);ntrain=protocol['roles']['train']
    binding=dict(protocol=protocol,source=source,input_files={n:digest(data/n) for n in ['train.jsonl','evaluation.inputs.jsonl','roles.json']})
    p=output/'run-binding.json'
    if p.exists():need(json.loads(p.read_bytes())==binding,'run source/protocol/input mismatch; use separate workspace')
    else:need(all(p.name.startswith('run-binding.json.') and p.name.endswith('.part') for p in output.iterdir()),'unbound run directory not empty');json_file(output,p.name,binding)
    if (output/'joint-prediction-seal.json').exists():
        seal=verify_campaign(output,binding,protocol,rows[ntrain:]);return dict(complete=True,prediction_groups=seal['expected_prediction_groups'],joint_seal_sha256=digest(output/'joint-prediction-seal.json'))
    jobs=['native-cache-'+k for k in ['dpr','bge']]+[f'local-born-{a}-s{s}' for a in ARMS for s in protocol['seeds']]
    emit('schedule',name='TriviaQA · 双模型 Local Born · 9臂×3种子',jobs=jobs)
    roots={k:cache/k for k in ['dpr','bge']};manifests={}
    for k in roots:
        job='native-cache-'+k;emit('cell',job=job,state='running')
        emit('device',device='cuda:0 / '+torch.cuda.get_device_name(),execution='original frozen FP32 full forwards, native8 candidate chunks')
        def event(**kw):emit('progress',**kw);allocator_event()
        manifests[k]=extract(k,Path(models[k]),rows,roles,roots[k],binding,event)
        emit('cell',job=job,state='computed')
    store=QuestionStore(roots,manifests,limit=protocol['cpu_question_cache_mib']*1024**2)
    declared=[output/'run-binding.json']
    for k in roots:
        path=output/('frozen-'+k+'.bin');original=checked_member(roots[k],manifests[k]['metadata'])
        if not path.exists():
            import uuid
            part=path.with_name(path.name+'.'+uuid.uuid4().hex+'.part');shutil.copyfile(original,part)
            with part.open('rb') as f:os.fsync(f.fileno())
            os.link(part,path);part.unlink()
        need(digest(path)==digest(original),'pinned factory frame/tail metadata')
        declared.append(path)
    for k in ['dpr','bge']:
        path=output/('predictions-native-'+k+'.json')
        rs=[]
        for i in range(ntrain,len(rows)):
            scores=store.one(k,i)['native'].tolist()
            rs.append(dict(id=rows[i]['id'],candidate_ids=manifests[k]['questions'][i]['candidate_ids'],scores=scores,ranking=sorted(range(50),key=lambda j:(-scores[j],j))))
        values=dict(rows=rs)
        if path.exists():need(json.loads(path.read_bytes())==values,'native prediction mismatch')
        else:json_file(output,path.name,values)
        declared.append(path)
    for arm in ARMS:
        for seed in protocol['seeds']:
            folder=output/(arm+'-s'+str(seed))
            result=run_cell(folder,dict(binding,arm=arm,seed=seed),store,rows,ntrain,arm,seed,protocol['epochs'],protocol['checkpoint_every'])
            record=result['checkpoint']
            declared += [folder/'result.json',folder/'journal/binding.json',folder/'journal'/f"step-{record['step']:08d}.json",checked_member(folder/'journal',record['tensor'])]
            declared += [checked_member(folder,result['predictions'][k]) for k in ['dpr','bge']]
    expected=2+2*len(ARMS)*len(protocol['seeds']);files=[]
    for path in sorted(declared):
        r=member(output,path);files.append(dict(path=r['name'],size=r['size'],sha256=r['sha256']))
    need(sum(Path(x['path']).name.startswith('predictions-') for x in files)==expected,'complete joint prediction groups')
    seal=dict(schema='qduet.local-born-joint.v1',files=files,failures=[],all_predictions_before_evaluation_labels=True,
        restored_checkpoint_inference=True,roles=protocol['roles'],seeds=protocol['seeds'],arms=list(ARMS),expected_prediction_groups=expected)
    p=output/'joint-prediction-seal.json'
    if p.exists():need(json.loads(p.read_bytes())==seal,'joint seal changed')
    else:json_file(output,p.name,seal)
    verify_campaign(output,binding,protocol,rows[ntrain:])
    return dict(complete=True,prediction_groups=expected,joint_seal_sha256=digest(p))
