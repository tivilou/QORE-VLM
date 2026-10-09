from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
root=Path('/home/Q-DUET-VLM/QORE-VLM-phase1-gate'); sys.path.insert(0,str(root))
from scripts.collab.five_ideas import run_semantic_reader_supervision_audit as runner
from applications.rag import semantic_supervision_audit as a
run=Path('/srv/qore-collab-uploads/five_ideas/semantic_reader_supervision_audit/20261009T045728Z')
inputs=Path('/srv/qore-collab-uploads/five_ideas/semantic_reader_supervision_audit_input/repair_20261008T145652Z')
summary=runner.load_json(run/'summary.json'); manifest=runner.load_json(run/'upload_manifest.json')
receipts=[]
for r in manifest['exchange_files']+[{'name':'upload_manifest.json'}]:
 name=r['name']; p=run/name; receipt=runner.load_json(run/(name+'.upload.json'))
 assert receipt['sha256']==a.digest_file(p) and receipt['size_bytes']==p.stat().st_size and receipt['path']==manifest['target_directory']+'/'+name
 if 'sha256' in r: assert r['sha256']==receipt['sha256'] and r['bytes']==receipt['size_bytes']
 receipts.append({'name':name,'sha256':receipt['sha256'],'bytes':receipt['size_bytes']})
assert summary['provenance']['code_revision']=='31a5c932de9b525cdf34eed688e9890226402cb9'
for path,sha in summary['provenance']['source_hashes'].items():
 assert hashlib.sha256(subprocess.check_output(['git','show','31a5c93:'+path],cwd=root)).hexdigest()==sha,path
assert hashlib.sha256(subprocess.check_output(['git','show','31a5c93:configs/experiments/semantic_reader_supervision_audit.json'],cwd=root)).hexdigest()==summary['provenance']['config_sha256']
assert a.digest_file(inputs/'input_manifest.json')==runner.SOURCE_MANIFEST_SHA
for r in runner.load_json(inputs/'input_manifest.json')['files']:
 assert a.digest_file(inputs/r['name'])==r['sha256'] and (inputs/r['name']).stat().st_size==r['bytes']
cohort=runner.load_json(inputs/'cohort.json')
with np.load(inputs/'head_checkpoints.npz',allow_pickle=False) as checkpoint:
 # Historical validator uses exact float dict equality; preserve it and audit
 # cross-environment scalar statistics with an explicitly bounded comparator.
 import inspect, math
 scalar_differences=[]
 def scalar_equal(left,right,path='$'):
  if isinstance(left,dict):
   if not isinstance(right,dict) or set(left)!=set(right): return False
   return all(scalar_equal(left[k],right[k],path+'.'+k) for k in left)
  if isinstance(left,list):
   return isinstance(right,list) and len(left)==len(right) and all(scalar_equal(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(left,right)))
  if isinstance(left,float) and isinstance(right,(float,int)):
   if left!=right: scalar_differences.append({'field':path,'saved':left,'recomputed':right,'absolute_difference':abs(left-right)})
   return math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-9)
  return type(left)==type(right) and left==right
 source=inspect.getsource(runner.validate_trace)
 old='if row["boundary"] != audit.boundary_diagnostic(baseline, saved, mask, candidates) or row["compression"] != audit.compression_diagnostic(baseline, saved):'
 new='if not scalar_equal(row["boundary"], audit.boundary_diagnostic(baseline, saved, mask, candidates), "boundary") or not scalar_equal(row["compression"], audit.compression_diagnostic(baseline, saved), "compression"):'
 assert old in source
 env=dict(runner.__dict__); env['scalar_equal']=scalar_equal
 exec(source.replace(old,new),env)
 validation=env['validate_trace'](run,cohort,checkpoint)
 validation['scalar_arithmetic_cross_environment_tolerance']='pass_at_1e-9'

trace=runner.load_json(run/'case_study.json'); cases=trace['cases']; review=runner.load_json(run/'support_review.json')
assert not trace['fixture'] and not trace['training_called'] and not trace['generator_called'] and not trace['evaluation_called']
assert trace['provenance']==summary['provenance']==manifest['provenance']
assert all(x['support_label'] is None for c in review['cases'] for x in c['passages'])
methods={}; changes=[]; case_table=[]
for m in a.METHODS:
 recompute={'weak_positive_selected_total':sum(c['methods'][m]['weak_positive_count'] for c in cases),'changed_top5_cases':sum(set(c['methods'][m]['selected_indices'])!=set(c['methods']['frozen_reader_topk']['selected_indices']) for c in cases),'boundary_pairs':sum(c['methods'][m]['boundary']['pair_count'] for c in cases),'strict_crossings':sum(c['methods'][m]['boundary']['strict_crossings'] for c in cases)}
 assert recompute==summary['methods'][m]
 slopes=[c['methods'][m]['compression']['score_on_base_slope'] for c in cases]
 correlations=[c['methods'][m]['compression']['delta_base_pearson'] for c in cases if c['methods'][m]['compression']['delta_base_pearson'] is not None]
 pairs=[p for c in cases for p in c['methods'][m]['boundary']['pairs']]
 pairs_with=[p for p in pairs if p['matched_weak_negative_control']]
 methods[m]={**recompute,'median_score_on_base_slope':float(np.median(slopes)),'median_delta_base_correlation':float(np.median(correlations)) if correlations else None,'favorable_weak_positive_relative_correction_pairs':sum(p['relative_correction']>0 for p in pairs),'matched_control_pairs':len(pairs_with),'favorable_wrong_relative_correction_pairs':sum(p['matched_weak_negative_control']['relative_correction']>0 for p in pairs_with),'positive_more_favored_than_wrong_pairs':sum(p['relative_correction']>p['matched_weak_negative_control']['relative_correction'] for p in pairs_with),'bound_unreachable_weak_positive_pairs':sum(not p['within_bound_necessary_not_sufficient'] for p in pairs)}
 for n,c in enumerate(cases,1):
  orig=set(c['methods']['frozen_reader_topk']['selected_indices']); selected=set(c['methods'][m]['selected_indices'])
  if selected!=orig:
   changes.append({'audit_case_number':n,'usable_case_index':c['usable_case_index'],'method':m,'question_sha256':c['question_sha256'],'weak_delta':c['methods'][m]['weak_positive_count']-c['methods']['frozen_reader_topk']['weak_positive_count'],'added':[{'candidate_index':i,'reader_rank':c['candidates'][i]['reader_rank'],'retrieved_rank':c['candidates'][i]['retrieved_rank'],'weak_positive':c['candidates'][i]['weak_positive'],'review_item_id':c['candidates'][i]['question_passage_review_id']} for i in sorted(selected-orig)],'removed':[{'candidate_index':i,'reader_rank':c['candidates'][i]['reader_rank'],'retrieved_rank':c['candidates'][i]['retrieved_rank'],'weak_positive':c['candidates'][i]['weak_positive'],'review_item_id':c['candidates'][i]['question_passage_review_id']} for i in sorted(orig-selected)]})
for n,c in enumerate(cases,1):
 case_table.append({'audit_case_number':n,'usable_case_index':c['usable_case_index'],'question_sha256':c['question_sha256'],'sample_role':c['sample_role'],'future_role':c['future_role'],'total_weak_positive':sum(p['weak_positive'] for p in c['candidates']),'base_weak_top5':c['methods']['frozen_reader_topk']['weak_positive_count'],'base_top5_reader_ranks':[c['candidates'][i]['reader_rank'] for i in c['methods']['frozen_reader_topk']['selected_indices']]})
with np.load(run/'audit_values.npz',allow_pickle=False) as values:
 saturation={m:float(np.mean(np.concatenate([np.abs(values[c['value_prefix']+'__'+m+'__encoded']).ravel()>.99 for c in cases]))) for m in a.TRAINABLE}
report={'schema_version':'rag.semantic_supervision_result_audit.v1','source_run':manifest['target_directory'],'source_revision':summary['provenance']['code_revision'],'artifact_receipts':receipts,'source_identity':'pass','independent_replay':validation,'strict_historical_validator':'failed_exact_scalar_dict_equality','scalar_tolerance':{'rtol':1e-9,'atol':1e-9,'differing_scalars':len(scalar_differences),'max_absolute_difference':max([r['absolute_difference'] for r in scalar_differences],default=0),'examples':scalar_differences[:5]},'sample_case_count':32,'candidate_count':1600,'support_review_labels_filled':0,'training_run':False,'generator_run':False,'evaluation_run':False,'heldout_validation':False,'claim_ceiling':'L0_diagnostic','methods':methods,'encoding_abs_gt_099_fraction':saturation,'all_changed_cases':changes,'case_table':case_table,'limits':['weak labels not support accuracy','old training cases not unseen validation','60 pairs are dependent, repeated candidates','no full1600 support adjudication']}
a.write_json('/tmp/semantic_supervision_result_20261009T045728Z_audit.json',report)
rawchanges=[{**x,'question':cases[x['audit_case_number']-1]['question'],'answers':cases[x['audit_case_number']-1]['answers'],'added_full':[cases[x['audit_case_number']-1]['candidates'][p['candidate_index']] for p in x['added']],'removed_full':[cases[x['audit_case_number']-1]['candidates'][p['candidate_index']] for p in x['removed']]} for x in changes]
a.write_json('/tmp/semantic_supervision_result_20261009T045728Z_changed_case_details.json',rawchanges)
print(json.dumps({'source_identity':'pass','receipts':len(receipts),'replay':validation,'methods':methods,'saturation':saturation,'changes':changes},indent=2))
