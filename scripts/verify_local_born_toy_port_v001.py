"""Affected-adapter acceptance only: reuse existing toy/native banks, no new data.

Run inside the accepted owner allocation. It never acquires fullsmallset assets.
"""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from applications.rag.qvalue_portable_v001 import digest,checked_member,need
from applications.rag.local_born_io_v001 import immutable_json,verify_seal
from applications.rag.local_born_cache_v001 import new_tensor,input_identity
from applications.rag.local_born_smallset_v001 import execute
from applications.rag.toy100_qk_task_v001 import rows,train_targets


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;out=a.output;out.mkdir(parents=True,exist_ok=False)
    original=root/'data/toy100-high-ceiling-v001';population=rows(original);targets=train_targets(original,population['train'])
    data=out/'data';data.mkdir();source=dict(accepted147_sha256=digest(root/'local-born-run-v001/output/joint-seal.json'),purpose='task-adapter parity; no fullsmallset')
    allrows=[];roles={}
    for role in ['train','holdout']:
        for i,r in enumerate(population[role]):
            value={k:r[k] for k in ['id','component_id','question','paragraphs']}
            if role=='train':value['labels']={p['id']:int(y) for p,y in zip(value['paragraphs'],targets[i])}
            allrows.append(value);roles[r['id']]='train' if role=='train' else 'valid'
    for name,rs in [('train.jsonl',allrows[:80]),('evaluation.inputs.jsonl',allrows[80:])]:
        (data/name).write_text(''.join(json.dumps(r)+'\n' for r in rs))
    immutable_json(data,'roles.json',roles)
    protocol=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/local_born_smallset_20261125/protocol.v001.json').read_bytes())
    protocol.update(roles=dict(train=80,valid=20,test=0),seeds=[20261122],steps_per_learning_job=200,checkpoint_every=20,
        coupled_learning_jobs=9,sealed_prediction_groups=20,acceptance_scope='Existing toy only; not a new efficacy study')
    binding=dict(protocol=protocol,source=source,input_files={n:digest(data/n) for n in ['train.jsonl','evaluation.inputs.jsonl','roles.json']})
    frame_root=root/'output-frame-run-v001/ffn-readouts';records=json.loads((frame_root/'result.json').read_bytes())
    for k in ['dpr','bge']:
        packed=torch.load(checked_member(frame_root,records[k]['compact_file']),weights_only=True)
        need(packed['population_sha256']==digest(original/'population.private.json'),'exact same existing toy population')
        cache=out/'native-cache'/k;cache.mkdir(parents=True)
        cb=dict(kind=k,source=binding,inputs=[input_identity(r) for r in allrows],roles=[roles[r['id']] for r in allrows])
        immutable_json(cache,'binding.json',cb);qs=[]
        tensors={n:torch.cat([packed[n]['train'],packed[n]['holdout']]) for n in ['ffn_z','ffn_residual','features','native']}
        for i,row in enumerate(allrows):
            # torch.save(view) retains its entire backing storage. Materialize
            # this one question, as the production extractor's torch.cat does.
            value={n:tensors[m][i].clone() for n,m in [('z','ffn_z'),('residual','ffn_residual'),('features','features'),('native','native')]}
            qs.append(dict(index=i,id=row['id'],candidate_ids=[p['id'] for p in row['paragraphs']],input_sha256=cb['inputs'][i],
                tensor=new_tensor(cache,f'question-{i:05d}',value)))
        metadata={n:copy.deepcopy(packed[n]) for n in ['original_head','ffn_dense','ffn_normalization','ffn_bridge','ffn_eps','ffn_frames']}
        for n in ['ffn_z','features','native']:metadata[n]=dict(train=packed[n]['train'][:1].clone())
        manifest=dict(binding=cb,metadata=new_tensor(cache,'metadata',metadata),questions=qs)
        immutable_json(cache,'cache.json',manifest)
    result=execute(data,dict(dpr='unused',bge='unused'),out/'matrix',out/'native-cache',protocol,source)
    seal=verify_seal(out/'matrix');need(seal['expected_prediction_groups']==20,'full native plus9coupled arms onfixedseed')
    differences={};changed=0;count=0
    for arm in protocol['arms']:
        cell=json.loads((out/'matrix'/f'{arm}-s20261122/result.json').read_bytes())
        oldfolder=root/'local-born-run-v001/output'/f'{arm}-s20261122';oldcell=json.loads((oldfolder/'result.json').read_bytes())
        for k in ['dpr','bge']:
            new=json.loads(checked_member(out/'matrix'/f'{arm}-s20261122',cell['predictions'][k]).read_bytes())['rows']
            old=json.loads(checked_member(oldfolder,oldcell['predictions'][k]).read_bytes());need(len(old)==len(new)==20,'same complete heldout')
            maximum=max(abs(x-y) for aa,bb in zip(old,new) for x,y in zip(aa['scores'],bb['scores']))
            need(maximum<=2e-5,'portable score parity within original FP32 tolerance')
            flips=sum(aa['top5']!=bb['ranking'][:5] for aa,bb in zip(old,new));need(flips==0,'all20completeTop5 rankings preserved')
            differences[arm+'/'+k]=dict(maximum_score_difference=maximum,changed_top5=flips,values=1000);changed+=flips;count+=1000
    # Sealed completion is repeatable even with later coordinator reports;
    # completed cells/cache do not reconstruct or train during this branch.
    immutable_json(out/'matrix','smallset-evaluation.json',dict(acceptance_test=True,not_actual_smallset=True))
    repeated=execute(data,dict(dpr='unused',bge='unused'),out/'matrix',out/'native-cache',protocol,source)
    need(repeated==result,'completed sealed resume unaffected by post-seal reports')
    report=dict(status='PASS',scope='same existing toy80/20 single fixedseed task-adapter acceptance',
        source_147=source,groups=20,complete_fixed_seed=20261122,learning_jobs=9,steps_each=200,
        compared_scores=count,changed_top5=changed,parity=differences,sealed_resume_identical=True,
        maximum_score_difference=max(v['maximum_score_difference'] for v in differences.values()),
        no_owner_fullsmallset=True,no_new_assets_or_labels=True,scientific_efficacy_claim='Reuse all3seed147; this tests the changed adapter only')
    immutable_json(out,'acceptance.json',report);print(json.dumps(report))


if __name__=='__main__':main()
