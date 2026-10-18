#!/usr/bin/env python3
"""Bounded sequential native-safe updates on the frozen59 training cache."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,platform,sys,time
import numpy as np
import scipy
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from applications.rag import reader_membership_training as learning
from applications.rag.quantum_reader_rotation import make_rotation_head,RotationConfig
from applications.rag.protected_update_feasibility import (score_jacobian,rescue_targets,
    bounded_proposal,membership_counts,sequential_membership_counts,
    next_sequential_proposal,sequential_boundary_system)
from scripts.collab.five_ideas.run_reader_protected_update_feasibility import inputs,digest,native_forward
from scripts.collab.lib.source_snapshot import archive_sources,verify_sources

CONFIG=ROOT/'configs/experiments/reader_sequential_protected_updates.json'
PLAN=ROOT/'configs/experiments/reader_sequential_protected_updates_plan.json'
SOURCES=(
    'applications/rag/protected_update_feasibility.py',
    'applications/rag/quantum_reader_rotation.py',
    'applications/rag/reader_membership_training.py',
    'applications/rag/protected_boundary_inputs.py',
    'applications/rag/protected_boundary_learning.py',
    'applications/rag/scale_invariant_top5_objective.py',
    'applications/rag/support_boundary_bridge.py',
    'applications/rag/support_supervision_axes.py',
    'scripts/collab/five_ideas/run_reader_protected_update_feasibility.py',
    'scripts/collab/five_ideas/run_reader_sequential_protected_updates.py',
    'scripts/collab/lib/source_snapshot.py',
    'configs/experiments/reader_sequential_protected_updates.json',
    'configs/experiments/reader_sequential_protected_updates_plan.json',
    '.ai-progress/workstreams/rag-selector/refs/reader_protected_feasibility_20261017_recovery.json')


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def compact_membership(value):
    return {k:v for k,v in value.items() if k!='selected_indices'}


def accepted_lists(score_matrix,targets,ranks):
    selected=membership_counts(score_matrix,targets,ranks)['selected_indices']
    return [sorted(set(s)&set(t['eligible_positive_indices'])) for s,t in zip(selected,targets)]


def validate_trace(directory):
    trace=json.loads((directory/'selector_trace.json').read_text())
    if trace['schema_version']!='sample-trace.v2' or digest(directory/'sequential_values.npz')!=trace['values_sha256']:
        raise ValueError('trace binding drift')
    with np.load(directory/'sequential_values.npz',allow_pickle=False) as z:
        if set(z.files)!=set(trace['values']):raise ValueError('trace coverage drift')
        for key,spec in trace['values'].items():
            value=z[key]
            if list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype'] or learning.array_hash(value)!=spec['sha256']:
                raise ValueError('numeric coordinate drift: '+key)
        for point in trace['checkpoints']:
            parameter_count=int(np.prod(z[point['head_state']['theta']].shape))
            before=point['initial_membership']
            current_rescue=before['eligible_rescued']
            current_accepted=[set() for _ in trace['training_targets']]
            for step in point['steps']:
                np.testing.assert_array_equal(z[step['scores64']].shape,(len(trace['training_targets']),50))
                np.testing.assert_array_equal(z[step['jacobians']].shape,(len(trace['training_targets']),50,parameter_count))
                accepted=None
                for trial in step['trials']:
                    if not trial['solver']['success']:continue
                    delta=z[trial['delta']]
                    slack=z[trial['linear_slack']]
                    if float(slack.min()) < -2e-7 or float(np.abs(delta).max()) > trace['config']['budget']['radius']+1e-8:
                        raise ValueError('invalid saved LP proposal')
                    for native in trial['native']:
                        health=sequential_membership_counts(z[native['scores']],trace['training_targets'],trace['ranks'],current_accepted)
                        if compact_membership(health)!=native['membership']:
                            raise ValueError('native membership reconstruction drift')
                        safe=(health['eligible_rescued']>current_rescue and health['lost_protected']==0 and
                              health['accepted_lost']==0 and trial['proposal']['added'] in health['selected_indices'][trial['proposal']['case']])
                        if safe!=native['accepted']:
                            raise ValueError('false native acceptance')
                        if safe:
                            if accepted is not None:raise ValueError('multiple accepted proposals in one step')
                            accepted=native
                            current_rescue=health['eligible_rescued']
                            current_accepted=[set(v) for v in native['accepted_after']]
                if step['accepted']!=(accepted is not None):raise ValueError('step acceptance drift')
    return {'status':'passed','complete_arrays':len(trace['values']),
            'sequential_membership_and_numeric_replay':True}


def run(args):
    torch.set_num_threads(1)
    cfg=json.loads(CONFIG.read_text())
    plan=json.loads(PLAN.read_text())
    if cfg['families']!=['quantum','classical'] or cfg['checkpoints']!=['initial']:
        raise ValueError('head allowlist drift')
    if plan['authorization']!='implemented':raise ValueError('architecture not implemented')
    started=time.monotonic();run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.output/run_id;out.mkdir(parents=True,exist_ok=False)
    snapshot=archive_sources(ROOT,SOURCES,out/'executed_sources.zip')
    features,records,targets,weight,bias,qualification=inputs(args,cfg)
    ranks=[r['candidate_retrieved_ranks'] for r in records]
    base=np.stack([f['base'].numpy() for f in features])
    candidates=rescue_targets(base,targets,cfg['budget']['candidate_limit'])
    arrays=learning.DetachedValueStore(out)
    save=lambda key,value:learning.save_value(arrays,key,value)
    trace={'schema_version':'sample-trace.v2','artifact_type':'reader_sequential_protected_updates',
        'run_id':run_id,'synthetic_only':args.synthetic,'config':cfg,'population':{'training':len(records),'evaluation':0},
        'training_targets':targets,'ranks':ranks,'candidate_order':candidates,
        'sample_selection':'first2 source-order training cases at every relinearized step; all59 numeric scores/Jacobians/trials',
        'stages':['data','preprocess','scoring','selection','diagnosis'],'samples':[],'checkpoints':[]}
    for i in range(min(2,len(features))):
        feature=features[i]
        trace['samples'].append({'id':records[i]['prefix'],'inputs':{k:save(f'input_{i}__{k}',v) for k,v in feature.items()}})
    for family in cfg['families']:
        head=make_rotation_head(weight,bias,family=family,config=RotationConfig(**cfg['source_rotation']))
        if not args.synthetic:
            with np.load(args.source/'membership_values.npz',allow_pickle=False) as z:
                state={k:torch.from_numpy(z[family+'_membership__initial__'+k].copy()) for k in head.state_dict()}
            head.load_state_dict(state)
        theta=head.theta.detach().numpy().astype(np.float64).copy()
        initial_scores,initial_native=native_forward(head,features,targets,ranks,theta)
        initial=sequential_membership_counts(initial_scores,targets,ranks,[set() for _ in targets])
        if initial['lost_protected'] or initial['accepted_lost'] or initial['eligible_rescued']:
            raise ValueError('initial identity membership drift')
        if initial['selected_indices']!=[t['baseline_top5_indices'] for t in targets]:
            raise ValueError('initial Reader Top5 drift')
        point={'id':family+'_initial','head_state':{k:save(f'{family}__head__{k}',v) for k,v in head.state_dict().items()},
            'initial_scores':save(f'{family}__initial_scores',initial_scores),
            'initial_membership':compact_membership(initial),'steps':[],'stop_reason':None}
        accepted=[set() for _ in targets];current_scores=initial_scores;current_rescue=0;cursor=0
        for step_number in range(1,cfg['budget']['max_steps']+1):
            if time.monotonic()-started>cfg['budget']['total_seconds']:
                point['stop_reason']='global_time_budget';break
            exact=copy.deepcopy(head).double()
            with torch.no_grad():exact.theta.copy_(torch.as_tensor(theta,dtype=exact.theta.dtype))
            rows=[];jacrows=[];sample_outputs=[]
            for i,feature in enumerate(features):
                output,jac=score_jacobian(exact,{k:v.double() for k,v in feature.items()})
                rows.append(output['scores'].numpy());jacrows.append(jac.numpy())
                if i<2:sample_outputs.append({k:save(f'{family}__s{step_number}__case{i}__{k}',v) for k,v in output.items()})
            scores64,jacobians=np.stack(rows),np.stack(jacrows)
            np.testing.assert_allclose(scores64,current_scores,rtol=2e-5,atol=3e-5)
            step={'step':step_number,'theta_before':save(f'{family}__s{step_number}__theta_before',theta),
                'scores64':save(f'{family}__s{step_number}__scores64',scores64),
                'jacobians':save(f'{family}__s{step_number}__jacobians',jacobians),
                'membership_before':compact_membership(sequential_membership_counts(current_scores,targets,ranks,accepted)),
                'accepted_before':[sorted(v) for v in accepted],'complete_first2_outputs':sample_outputs,
                'trials':[],'accepted':False}
            attempts=0
            while attempts<cfg['budget']['max_attempts_per_step'] and cursor<len(candidates):
                proposal,next_cursor=next_sequential_proposal(current_scores,targets,ranks,candidates,accepted,cursor)
                cursor=next_cursor
                if proposal is None:break
                attempts+=1;key=f'{family}__s{step_number}__a{attempts}'
                matrix,rhs,lineage=sequential_boundary_system(scores64,jacobians,base,targets,ranks,proposal,accepted,cfg['budget']['normalized_margin'])
                solver,delta=bounded_proposal(matrix,rhs,cfg['budget']['radius'],cfg['budget']['lp_time_limit_seconds'])
                trial={'attempt':attempts,'proposal':proposal,'solver':solver,'native':[],
                    'constraint_matrix':save(key+'__A',matrix),'constraint_rhs':save(key+'__b',rhs),
                    'constraint_lineage':save(key+'__lineage',lineage)}
                if delta is not None:
                    trial.update(delta=save(key+'__delta',delta),linear_slack=save(key+'__slack',matrix@delta-rhs))
                    for fraction in cfg['budget']['step_fractions']:
                        candidate_theta=theta+fraction*delta.reshape(theta.shape)
                        trial_scores,_=native_forward(head,features,targets,ranks,candidate_theta)
                        health=sequential_membership_counts(trial_scores,targets,ranks,accepted)
                        safe=(health['eligible_rescued']>current_rescue and health['lost_protected']==0 and
                              health['accepted_lost']==0 and proposal['added'] in health['selected_indices'][proposal['case']])
                        next_accepted=accepted_lists(trial_scores,targets,ranks) if safe else [sorted(v) for v in accepted]
                        native={'fraction':fraction,'scores':save(key+f'__f{fraction}__scores',trial_scores),
                            'membership':compact_membership(health),'accepted':safe,'accepted_after':next_accepted}
                        trial['native'].append(native)
                        if safe:
                            theta=candidate_theta;current_scores=trial_scores;current_rescue=health['eligible_rescued']
                            accepted=[set(v) for v in next_accepted]
                            step['accepted']=True;step['accepted_trial']={'attempt':attempts,'fraction':fraction,
                                'proposal':proposal,'membership':compact_membership(health),'accepted_after':next_accepted,
                                'theta_after':save(key+'__theta_after',theta)}
                            break
                step['trials'].append(trial)
                print(f'{family}: step{step_number} target{proposal["cursor"]}; lp={solver["status"]}; accepted={step["accepted"]}; rescue={current_rescue}',flush=True)
                if step['accepted']:break
                native_forward(head,features,targets,ranks,theta)
            point['steps'].append(step)
            if not step['accepted']:
                point['stop_reason']='no_native_safe_strict_increase';break
        if point['stop_reason'] is None:
            point['stop_reason']='maximum_accepted_steps'
        point['final_membership']=compact_membership(sequential_membership_counts(current_scores,targets,ranks,accepted))
        point['final_theta']=save(f'{family}__final_theta',theta)
        point['accepted_final']=[sorted(v) for v in accepted]
        trace['checkpoints'].append(point)
    trace['values']=learning.describe_values(arrays)
    arrays.archive(out/'sequential_values.npz');arrays.clear()
    trace['values_sha256']=digest(out/'sequential_values.npz')
    verify_sources(ROOT,snapshot)
    trace['provenance']={'source_snapshot':snapshot,'config_sha256':digest(CONFIG),'qualification':qualification,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'device':'cpu'},
        'evaluation_fields_accessed':False,'new_optimizer_updates':0,'parameter_search':'sequential bounded LP on training59 only'}
    write(out/'selector_trace.json',trace)
    readiness=validate_trace(out)
    summary={'schema_version':'rag.reader_sequential_protected_result.v1','run_id':run_id,'synthetic_only':args.synthetic,
        'status':'completed','elapsed_seconds':time.monotonic()-started,'readiness':readiness,
        'checkpoints':[{'id':p['id'],'accepted_steps':sum(s['accepted'] for s in p['steps']),
            'final_membership':p['final_membership'],'accepted_final_total':sum(map(len,p['accepted_final'])),
            'attempts':sum(len(s['trials']) for s in p['steps']),'stop_reason':p['stop_reason']} for p in trace['checkpoints']],
        'provenance':trace['provenance'],'claim_ceiling':'L0_train_only_sequential_feasibility','limitations':cfg['limits']}
    write(out/'summary.json',summary)
    write(out/'upload_manifest.json',{'schema_version':'collaborator-exchange.manifest.v1',
        'target_directory':'five_ideas/reader_sequential_protected_updates/'+run_id,
        'exchange_files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file()]})
    print(json.dumps({'output':str(out),'summary':summary},indent=2),flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--features',type=Path,default=ROOT/'.cache/reader-membership-input-v1')
    parser.add_argument('--source',type=Path,default=ROOT/'exchange/five_ideas/quantum_reader_membership_screen_100/20261017T152312389403Z')
    parser.add_argument('--bundle',type=Path,default=ROOT/'.cache/protected_boundary_reader_inputs/20261011-v1/frozen_support_bundle.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'exchange/five_ideas/reader_sequential_protected_updates')
    run(parser.parse_args())
