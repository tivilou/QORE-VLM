#!/usr/bin/env python3
"""Bounded training-only LP proposals with native nonlinear score verification."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,platform,sys,time
import numpy as np
import torch
import scipy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from applications.rag import reader_membership_training as learning
from applications.rag.quantum_reader_rotation import make_rotation_head,RotationConfig
from applications.rag.protected_boundary_inputs import load_bundle
from applications.rag.protected_update_feasibility import score_jacobian,rescue_targets,boundary_system,bounded_proposal,membership_counts
from scripts.collab.lib.source_snapshot import archive_sources,verify_sources
CONFIG=ROOT/'configs/experiments/reader_protected_update_feasibility.json'
PLAN=ROOT/'configs/experiments/reader_protected_update_feasibility_plan.json'
SOURCES=(
    'applications/rag/protected_update_feasibility.py',
    'scripts/collab/five_ideas/run_reader_protected_update_feasibility.py',
    'configs/experiments/reader_protected_update_feasibility.json',
    'configs/experiments/reader_protected_update_feasibility_plan.json',
    'applications/rag/reader_membership_training.py','applications/rag/quantum_reader_rotation.py',
    'applications/rag/protected_boundary_inputs.py','applications/rag/protected_boundary_learning.py',
    'applications/rag/scale_invariant_top5_objective.py','applications/rag/support_boundary_bridge.py',
    'applications/rag/support_supervision_axes.py','scripts/collab/lib/source_snapshot.py',
    '.ai-progress/workstreams/rag-selector/refs/reader_membership_screen_20261017_recovery.json')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


class Features:
    def __init__(self,path,records):self.path,self.records=path,records
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        if i>=len(self):raise IndexError
        prefix=self.records[i]['prefix']
        with np.load(self.path,allow_pickle=False) as z:
            return {k:torch.from_numpy(z[prefix+'__'+k]) for k in ('cls','pooled','base')}


def inputs(args,cfg):
    if args.synthetic:
        torch.manual_seed(23)
        weight,bias=torch.randn(8),torch.tensor(.2)
        features=[];records=[];targets=[]
        for i in range(2):
            cls=torch.randn(50,8);pooled=torch.randn(50,40);pooled[:,:8]=cls
            base=cls@weight+bias
            order=sorted(range(50),key=lambda j:(-float(base[j]),j))
            features.append(dict(cls=cls,pooled=pooled,base=base))
            targets.append(dict(baseline_top5_indices=order[:5],immutable_selected_indices=order[:3],eligible_positive_indices=order[5:7]))
            records.append(dict(prefix=f'training_{i+1:03d}',question_sha256=hashlib.sha256(str(i).encode()).hexdigest(),candidate_retrieved_ranks=list(range(1,51))))
        return features,records,targets,weight,bias,{'synthetic_only':True}
    paths={'source_values_sha256':args.source/'membership_values.npz','source_feature_sha256':args.features/'reader_features.npz',
        'source_index_sha256':args.features/'cache_index.json','source_bundle_sha256':args.bundle}
    for key,path in paths.items():
        if path.is_symlink() or digest(path)!=cfg[key]:raise ValueError('input hash mismatch: '+key)
    index=json.loads((args.features/'cache_index.json').read_text())
    records=[r for r in index['records'] if r['lane']=='training']
    if len(records)!=59 or [r['number'] for r in records]!=list(range(1,60)):raise ValueError('train order/coverage drift')
    rows,qualification=load_bundle(args.bundle)
    targets=[r['target'] for r in rows]
    for r,row in zip(records,rows):
        case=row['case']
        if hashlib.sha256(case['question'].encode()).hexdigest()!=r['question_sha256']:raise ValueError('question identity drift')
        ids=[hashlib.sha256(str(p['id']).encode()).hexdigest() for p in case['candidates']]
        if ids!=r['candidate_id_sha256'] or [p['retrieved_rank'] for p in case['candidates']]!=r['candidate_retrieved_ranks']:raise ValueError('candidate identity/order drift')
        if row['target']['baseline_top5_indices']!=r['baseline_top5_indices']:raise ValueError('target baseline drift')
    if len(rows)!=59 or sum(len(t['immutable_selected_indices']) for t in targets)!=183:raise ValueError('qualified population drift')
    with np.load(paths['source_feature_sha256'],allow_pickle=False) as z:
        weight,bias=torch.from_numpy(z['relevance_weight']),torch.from_numpy(z['relevance_bias'])
    return Features(paths['source_feature_sha256'],records),records,targets,weight,bias,qualification


@torch.no_grad()
def native_forward(head,features,targets,ranks,theta):
    head.theta.copy_(torch.as_tensor(theta,dtype=head.theta.dtype).reshape_as(head.theta))
    scores=np.stack([head(f['cls'],f['pooled'])['scores'].numpy() for f in features])
    return scores,membership_counts(scores,targets,ranks)


def validate_numeric_trace(directory):
    trace=json.loads((directory/'selector_trace.json').read_text())
    if trace['schema_version']!='sample-trace.v2' or digest(directory/'feasibility_values.npz')!=trace['values_sha256']:raise ValueError('trace binding drift')
    with np.load(directory/'feasibility_values.npz',allow_pickle=False) as z:
        if set(z.files)!=set(trace['values']):raise ValueError('trace coverage drift')
        for key,spec in trace['values'].items():
            a=z[key]
            if list(a.shape)!=spec['shape'] or str(a.dtype)!=spec['dtype'] or learning.array_hash(a)!=spec['sha256']:raise ValueError('coordinate integrity drift')
        for point in trace['checkpoints']:
            scores,jac,base=(z[point[k]] for k in ('scores64','jacobians','base_scores'))
            state={k:torch.from_numpy(z[v].copy()) for k,v in point['head_state'].items()}
            head=make_rotation_head(state['weight'],state['bias'],family=point['id'].split('_')[0],config=RotationConfig(**trace['config']['source_rotation']))
            head.load_state_dict(state)
            exact=copy.deepcopy(head).double()
            for i,refs in enumerate(point['complete_first2_outputs']):
                feature={k:torch.from_numpy(z[v].copy()) for k,v in trace['samples'][i]['inputs'].items()}
                result,jac_replay=score_jacobian(exact,{k:v.double() for k,v in feature.items()})
                for k,v in refs.items():np.testing.assert_allclose(result[k].numpy(),z[v],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(jac_replay.numpy(),jac[i],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(np.linalg.norm(result['states'].numpy(),axis=-1),1.,rtol=1e-10,atol=1e-10)
            for trial in point['trials']:
                if not trial['solver']['success']:continue
                a,b,lineage=boundary_system(scores,jac,base,trace['training_targets'],trial['proposal'],protected=trial['mode']=='global_protection',margin=trace['config']['budget']['normalized_margin'])
                np.testing.assert_allclose(a@z[trial['delta']]-b,z[trial['linear_slack']],rtol=1e-9,atol=1e-9)
                if 'constraint_matrix' in trial:
                    np.testing.assert_array_equal(a,z[trial['constraint_matrix']]);np.testing.assert_array_equal(b,z[trial['constraint_rhs']])
                for check in trial['nonlinear']:
                    computed=membership_counts(z[check['scores']],trace['training_targets'],trace['ranks'])
                    if computed!=check['membership']:raise ValueError('nonlinear membership drift')
                    native=check['membership']
                    rescued=trial['proposal']['added'] in native['selected_indices'][trial['proposal']['case']]
                    if check['safe_witness']!=(native['rescued']>0 and native['lost_protected']==0 and rescued):raise ValueError('false safe witness')
                    updated=state['theta'].numpy().astype(np.float64)+check['fraction']*z[trial['delta']].reshape(state['theta'].shape)
                    with torch.no_grad():head.theta.copy_(torch.as_tensor(updated,dtype=state['theta'].dtype))
                    for i,sample in enumerate(trace['samples']):
                        feature={k:torch.from_numpy(z[v].copy()) for k,v in sample['inputs'].items()}
                        with torch.no_grad():actual=head(feature['cls'],feature['pooled'])['scores'].numpy()
                        np.testing.assert_array_equal(actual,z[check['scores']][i])
    return {'status':'passed','complete_arrays':len(trace['values']),'constraint_and_native_membership_replay':True}


def run(args):
    torch.set_num_threads(1)
    cfg=json.loads(CONFIG.read_text())
    if cfg['families']!=['quantum','classical'] or cfg['checkpoints']!=['initial','final']:raise ValueError('allowlist drift')
    plan=json.loads(PLAN.read_text())
    if plan['authorization'] not in ('approved','implemented'):raise ValueError('architecture not approved')
    started=time.monotonic();run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.output/run_id;out.mkdir(parents=True,exist_ok=False)
    snapshot=archive_sources(ROOT,SOURCES,out/'executed_sources.zip')
    features,records,targets,weight,bias,qualification=inputs(args,cfg)
    ranks=[r['candidate_retrieved_ranks'] for r in records]
    base=np.stack([f['base'].numpy() for f in features])
    proposals=rescue_targets(base,targets,cfg['budget']['targets_per_checkpoint'])
    arrays=learning.DetachedValueStore(out)
    save=lambda key,value:learning.save_value(arrays,key,value)
    trace={'schema_version':'sample-trace.v2','artifact_type':'reader_protected_update_feasibility','run_id':run_id,
        'synthetic_only':args.synthetic,'config':cfg,'training_targets':targets,'ranks':ranks,'sample_selection':'first2 source-order training cases, every checkpoint; first target attempt full constraints',
        'stages':['data','preprocess','scoring','selection','diagnosis'],'population':{'training':len(records),'evaluation':0},'samples':[],'checkpoints':[]}
    for i,feature in enumerate(features):
        if i<2:trace['samples'].append({'id':records[i]['prefix'],'inputs':{k:save(f'input_{i}__{k}',v) for k,v in feature.items()}})
    stopped=False
    for family in cfg['families']:
        for checkpoint in cfg['checkpoints']:
            tag=family+'_'+checkpoint
            head=make_rotation_head(weight,bias,family=family,config=RotationConfig(**cfg['source_rotation']))
            if not args.synthetic:
                method=family+'_membership'
                with np.load(args.source/'membership_values.npz',allow_pickle=False) as z:
                    state={k:torch.from_numpy(z[method+'__'+checkpoint+'__'+k].copy()) for k in head.state_dict()}
                head.load_state_dict(state)
            elif checkpoint=='final':
                with torch.no_grad():head.theta.copy_(torch.randn_like(head.theta)*.05)
            theta=head.theta.detach().numpy().astype(np.float64).copy()
            exact=copy.deepcopy(head).double()
            score_rows=[];jac_rows=[];full=[]
            for i,f in enumerate(features):
                f64={k:v.double() for k,v in f.items()}
                output,jac=score_jacobian(exact,f64)
                score_rows.append(output['scores'].numpy());jac_rows.append(jac.numpy())
                if i<2:
                    native=exact(f64['cls'][:2],f64['pooled'][:2])['scores']
                    expected=torch.stack([torch.autograd.grad(native[j],exact.theta,retain_graph=True)[0].flatten() for j in range(2)])
                    torch.testing.assert_close(jac[:2],expected,rtol=1e-9,atol=1e-10)
                    full.append({k:save(f'{tag}__case{i}__{k}',v) for k,v in output.items()})
            scores,jacobians=np.stack(score_rows),np.stack(jac_rows)
            score32,zero=native_forward(head,features,targets,ranks,theta)
            np.testing.assert_allclose(scores,score32,rtol=2e-5,atol=3e-5)
            if checkpoint=='initial':
                np.testing.assert_allclose(score32,base,rtol=1e-5,atol=3e-5)
                if zero['rescued'] or zero['lost_protected']:raise ValueError('identity null drift')
                if zero['selected_indices']!=[t['baseline_top5_indices'] for t in targets]:raise ValueError('full null selection drift')
            if not args.synthetic:
                with np.load(args.source/'membership_values.npz',allow_pickle=False) as z:
                    if checkpoint=='final':
                        for i,record in enumerate(records):
                            np.testing.assert_allclose(score32[i],z[record['prefix']+'__'+method+'__scores'],rtol=2e-5,atol=3e-5)
            point={'id':tag,'head_state':{k:save(f'{tag}__head__{k}',v) for k,v in head.state_dict().items()},
                'scores64':save(tag+'__scores64',scores),'jacobians':save(tag+'__jacobians',jacobians),'base_scores':save(tag+'__base',base),
                'zero_scores':save(tag+'__zero_scores',score32),'zero_membership':zero,'complete_first2_outputs':full,'real_adjoint_autograd_rows':4,'trials':[],'safe_witness':None}
            for attempt,proposal in enumerate(proposals,1):
                if time.monotonic()-started>cfg['budget']['total_seconds']:stopped=True;break
                for mode in cfg['proposal_modes']:
                    key=f'{tag}__a{attempt}__{mode}'
                    a,b,lineage=boundary_system(scores,jacobians,base,targets,proposal,protected=mode=='global_protection',margin=cfg['budget']['normalized_margin'])
                    solver,delta=bounded_proposal(a,b,cfg['budget']['radius'],cfg['budget']['lp_time_limit_seconds'])
                    trial={'attempt':attempt,'mode':mode,'proposal':proposal,'solver':solver,'nonlinear':[]}
                    if delta is not None:
                        trial.update(delta=save(key+'__delta',delta),linear_slack=save(key+'__linear_slack',a@delta-b))
                        if attempt==1:
                            trial.update(constraint_matrix=save(key+'__A',a),constraint_rhs=save(key+'__b',b),constraint_lineage=save(key+'__lineage',lineage))
                        for fraction in cfg['budget']['step_fractions']:
                            score,health=native_forward(head,features,targets,ranks,theta+fraction*delta.reshape(theta.shape))
                            witness=health['rescued']>0 and health['lost_protected']==0 and proposal['added'] in health['selected_indices'][proposal['case']]
                            result={'fraction':fraction,'scores':save(key+f'__f{fraction}__scores',score),'membership':health,'safe_witness':witness}
                            trial['nonlinear'].append(result)
                            if witness:break
                    point['trials'].append(trial)
                    if mode=='global_protection' and any(r['safe_witness'] for r in trial['nonlinear']):point['safe_witness']={'attempt':attempt,'proposal':proposal,'mode':mode}
                print(f'{tag}: target{attempt}; protected LP={point["trials"][-1]["solver"]["status"]}; safe={bool(point["safe_witness"])}',flush=True)
                if point['safe_witness']:break
            trace['checkpoints'].append(point)
            del exact,head,output,expected,native
            if stopped:break
        if stopped:break
    trace['values']=learning.describe_values(arrays)
    arrays.archive(out/'feasibility_values.npz');arrays.clear()
    trace['values_sha256']=digest(out/'feasibility_values.npz')
    verify_sources(ROOT,snapshot)
    trace['provenance']={'source_snapshot':snapshot,'config_sha256':digest(CONFIG),'qualification':qualification,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'device':'cpu'},
        'evaluation_fields_accessed':False,'new_optimizer_updates':0,'parameter_search':'bounded LP proposals on training only'}
    write(out/'selector_trace.json',trace)
    readiness=validate_numeric_trace(out)
    summary={'schema_version':'rag.reader_protected_update_result.v1','run_id':run_id,'synthetic_only':args.synthetic,
        'status':'budget_exhausted' if stopped else 'completed','readiness':readiness,'elapsed_seconds':time.monotonic()-started,
        'checkpoints':[{'id':p['id'],'zero_membership':{k:v for k,v in p['zero_membership'].items() if k!='selected_indices'},
            'safe_witness':p['safe_witness'],'attempts':max([t['attempt'] for t in p['trials']],default=0),
            'native_best':[{**{k:v for k,v in r['membership'].items() if k!='selected_indices'},'mode':t['mode'],'fraction':r['fraction']} for t in p['trials'] for r in t['nonlinear'] if r['safe_witness']]} for p in trace['checkpoints']],
        'provenance':trace['provenance'],'limitations':cfg['limits'],'claim_ceiling':'L0_train_only_feasibility_not_utility'}
    write(out/'summary.json',summary)
    write(out/'upload_manifest.json',{'schema_version':'collaborator-exchange.manifest.v1','target_directory':'five_ideas/reader_protected_update_feasibility/'+run_id,
        'exchange_files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file()]})
    print(json.dumps({'output':str(out),'summary':summary},indent=2),flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--features',type=Path,default=ROOT/'.cache/reader-membership-input-v1')
    parser.add_argument('--source',type=Path,default=ROOT/'exchange/five_ideas/quantum_reader_membership_screen_100/20261017T152312389403Z')
    parser.add_argument('--bundle',type=Path,default=ROOT/'.cache/protected_boundary_reader_inputs/20261011-v1/frozen_support_bundle.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'exchange/five_ideas/reader_protected_update_feasibility')
    run(parser.parse_args())
