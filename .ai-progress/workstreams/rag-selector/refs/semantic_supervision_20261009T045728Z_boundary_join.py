from pathlib import Path
import json,hashlib,collections,argparse
parser=argparse.ArgumentParser(description='Validate explicit single-model judgments and join frozen audit scores; never infer semantic labels.')
parser.add_argument('--source-dir',required=True,type=Path)
parser.add_argument('--records-dir',required=True,type=Path)
parser.add_argument('--output-dir',required=True,type=Path)
args=parser.parse_args();C=args.source_dir;R=args.records_dir;O=args.output_dir;O.mkdir(parents=True,exist_ok=True)
load=lambda n:json.loads((C/n).read_text(encoding='utf-8'))
record_names={'partial_support_review.json':'semantic_supervision_20261009T045728Z_partial_support_review.json','boundary_review_scope.json':'semantic_supervision_20261009T045728Z_boundary_scope.json','new_judgments.txt':'semantic_supervision_20261009T045728Z_boundary_judgments.txt'}
def record(n):
    return R/n if (R/n).exists() else R/record_names[n]
trace=load('case_study.json'); blind=load('support_review.json'); old=json.loads((record('partial_support_review.json')).read_text(encoding='utf-8'));scope=json.loads((record('boundary_review_scope.json')).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(C/'case_study.json')==scope['source_case_sha256']
assert sha(C/'support_review.json')==scope['source_review_sha256']==old['source_support_review_sha256']
expected=set()
for k,c in enumerate(trace['cases'],1):
    for pair in c['methods']['frozen_reader_topk']['boundary']['pairs']:
        expected.update((k,i) for i in (pair['weak_positive_index'],pair['weak_negative_index']))
        if pair['matched_weak_negative_control']:expected.add((k,pair['matched_weak_negative_control']['candidate_index']))
    for m in c['methods'].values():expected.update((k,i) for i in m['selected_indices'])
assert {(r['case_number'],r['candidate_index']) for r in scope['items']}==expected
assert len(scope['items'])==len(expected)==220
for r in scope['items']:
    c=trace['cases'][r['case_number']-1]
    assert r['review_case_id']==c['question_sha256'] and r['review_item_id']==c['candidates'][r['candidate_index']]['question_passage_review_id']
lookup={p['review_item_id']:(c,p) for c in blind['cases'] for p in c['passages']}
judgments={r['review_item_id']:r for r in old['judgments']}
new=[]; keys=set()
for line in (record('new_judgments.txt')).read_text(encoding='utf-8').splitlines():
    if not line.strip():continue
    key,label,quote,reason=line.split('|',3); case,index=map(int,key.split(':')); c=trace['cases'][case-1]; p=c['candidates'][index]
    assert key not in keys;keys.add(key)
    rid=p['question_passage_review_id']; assert rid not in judgments
    assert label in ('direct','partial','irrelevant','contradictory','uncertain')
    assert quote=='-' or quote in p['text'],(key,quote)
    assert label not in ('direct','partial','contradictory') or quote!='-'
    r={'audit_case_number':case,'candidate_index':index,'review_case_id':c['question_sha256'],'review_item_id':rid,'support_label':label,'support_quote':None if quote=='-' else quote,'rationale':reason,'selection':'frozen_registered_boundary_controls_and_all_selected_union','reviewer':'primary_agent_single_model','score_blind_when_reviewed':False,'scores_withheld_in_current_reading':True,'official_gold':False,'confidence':'provisional_single_reviewer','source_text_sha256':p['text_sha256']}
    new.append(r);judgments[rid]=r
assert len(new)==209 and len(judgments)==282
assert all(x['review_item_id'] in judgments for x in scope['items'])
assert sum(p['support_label'] is not None for c in blind['cases'] for p in c['passages'])==0
for r in judgments.values():
    c,p=lookup[r['review_item_id']]; assert r['review_case_id']==c['review_case_id']
    if r['support_quote']: assert r['support_quote'] in p['text']
labels=('direct','partial','irrelevant','contradictory','uncertain')
quality={2:('conditional_scenario_missing','Neither-vessel原文限于特定迎面相遇或绝对权利；问题未给场景。'),4:('serialized_formula_subscript_missing','公式氧下标4在导出文本中丢失，严格分辨CaSO4/CaSO3需保留原始公式。'),10:('reference_order_conflict','first约束与原文following migration of followers冲突。'),15:('election_vs_office_ambiguous','2016选举胜者与2016在任总统不同，参考名单合并两者。'),24:('event_start_ambiguous','首次创制/使用与1942官方认可的事件未区分。'),28:('question_reference_type_mismatch','when时间问题的参考答案是体育场；且未给赛季。'),30:('adaptation_version_unspecified','影片与舞台版相同角色不同演员；问题未声明版本。'),31:('acquisition_vs_constitution_conflict','1898领土获得与1950制宪授权是不同事件。'),32:('jurisdiction_and_authority_scope_unspecified','哪个联邦制度、哪种最终权威未声明，片段未给完整关系。')}
qflags=[{'audit_case_number':i,'review_case_id':trace['cases'][i-1]['question_sha256'],'flag':flag,'rationale':reason,'use_in_new_training':'blocked_pending_target_qualification','reference_rewritten':False,'official_gold_correction':False} for i,(flag,reason) in quality.items()]
review={'schema_version':'rag.semantic_support_partial_review.v1','status':'declared_boundary_scope_complete_full_population_partial','source_support_review_sha256':sha(C/'support_review.json'),'source_case_sha256':sha(C/'case_study.json'),'prior_review_sha256':sha(record('partial_support_review.json')),'scope_manifest_sha256':sha(record('boundary_review_scope.json')),'producer_labels_filled':0,'full_population':1600,'declared_scope_count':220,'declared_scope_reviewed':220,'distinct_reviewed_passages':282,'unreviewed_population':1318,'prior_judgments_retained':73,'new_judgments':209,'new_review_is_independent_blind':False,'question_target_flags':qflags,'judgments':list(judgments.values()),'limits':['single-model provisional interpretation, not three-model consensus or official gold','all1600 population not complete; scope selected from already observed trace','old heads trained on all434, no heldout validation','do not infer corpus noise rate or causal supervision effect','do not use this282 diagnostic review to train old exposed100 Silver panel or assume434 complete labels']}
def get(c,i):return judgments[c['candidates'][i]['question_passage_review_id']]['support_label']
methods={};case_stats=[]
for m in trace['cases'][0]['methods']:
    counts=collections.Counter(); clear=collections.Counter()
    for k,c in enumerate(trace['cases'],1):
        counts.update(get(c,i) for i in c['methods'][m]['selected_indices'])
        if k not in quality:clear.update(get(c,i) for i in c['methods'][m]['selected_indices'])
    methods[m]={'selected_slots':160,'label_counts':{l:counts[l] for l in labels},'direct_or_partial_count':counts['direct']+counts['partial'],'question_target_provisionally_unflagged':23,'unflagged_label_counts':{l:clear[l] for l in labels},'not_confirmed_accuracy':True}
pair_rows=[]
for k,c in enumerate(trace['cases'],1):
    base=c['methods']['frozen_reader_topk']
    cs={'audit_case_number':k,'target_flag':quality.get(k,(None,None))[0],'selected':{},'missed_direct_in_declared_scope':[]}
    for m,x in c['methods'].items():cs['selected'][m]=dict(collections.Counter(get(c,i) for i in x['selected_indices']))
    ids={x['candidate_index'] for x in scope['items'] if x['case_number']==k}
    for i in sorted(ids-set(base['selected_indices'])):
        if get(c,i)=='direct':cs['missed_direct_in_declared_scope'].append({'candidate_index':i,'reader_rank':c['candidates'][i]['reader_rank'],'weak_positive':c['candidates'][i]['weak_positive'],'review_item_id':c['candidates'][i]['question_passage_review_id']})
    case_stats.append(cs)
    for j,p in enumerate(base['boundary']['pairs']):
        pc=p['weak_positive_index'];nc=p['weak_negative_index'];ctrl=p['matched_weak_negative_control']['candidate_index'] if p['matched_weak_negative_control'] else None
        positive=get(c,pc); negative=get(c,nc);control=get(c,ctrl) if ctrl is not None else None
        row={'audit_case_number':k,'pair_ordinal':j,'positive_reader_rank':p['positive_reader_rank'],'positive_review_id':c['candidates'][pc]['question_passage_review_id'],'negative_review_id':c['candidates'][nc]['question_passage_review_id'],'control_review_id':c['candidates'][ctrl]['question_passage_review_id'] if ctrl is not None else None,'weak_positive_support_label':positive,'weak_negative_support_label':negative,'matched_control_support_label':control,'target_flag':quality.get(k,(None,None))[0],'clean_direct_vs_irrelevant_pair':k not in quality and positive=='direct' and negative=='irrelevant','clean_matched_control':k not in quality and positive=='direct' and negative=='irrelevant' and control=='irrelevant','methods':{}}
        for m,x in c['methods'].items():
            mp=x['boundary']['pairs'][j];assert mp['weak_positive_index']==pc and mp['weak_negative_index']==nc
            row['methods'][m]={'relative_correction':mp['relative_correction'],'control_relative_correction':mp['matched_weak_negative_control']['relative_correction'] if ctrl is not None else None,'crossed':mp['crossed']}
        pair_rows.append(row)
scopeweak=collections.Counter();scopeother=collections.Counter()
for x in scope['items']:
    c=trace['cases'][x['case_number']-1];p=c['candidates'][x['candidate_index']];lab=judgments[x['review_item_id']]['support_label']
    (scopeweak if p['weak_positive'] else scopeother).update([lab])
clean=[p for p in pair_rows if p['clean_direct_vs_irrelevant_pair']];matched=[p for p in pair_rows if p['clean_matched_control']]
contrast={m:{'clean_dependent_pair_count':len(clean),'clean_distinct_questions':len({p['audit_case_number'] for p in clean}),'positive_relative_correction':sum(p['methods'][m]['relative_correction']>0 for p in clean),'strict_crossings':sum(p['methods'][m]['crossed'] for p in clean),'clean_matched_pairs':len(matched),'positive_more_favored_than_matched_irrelevant':sum(p['methods'][m]['relative_correction']>p['methods'][m]['control_relative_correction'] for p in matched)} for m in methods}
report={'schema_version':'rag.semantic_support_boundary_join.v1','source_run':'five_ideas/semantic_reader_supervision_audit/20261009T045728Z','claim_ceiling':'L0_diagnostic','coverage':{'declared_scope':220,'declared_scope_reviewed':220,'reviewed_population':282,'all_population':1600,'new_judgments':209,'old_preserved':73,'all_selected_slots_covered':True,'all_registered_boundary_control_items_covered':True,'all_swaps_covered':True},'semantic_modes':'single_primary_model, provisional, new scored context not independent blind','methods':methods,'scope_only_weak_positive_label_counts':dict(scopeweak),'scope_only_weak_negative_label_counts':dict(scopeother),'scope_noise_rates_are_not_population_rates':True,'target_flags':qflags,'registered_pair_count':len(pair_rows),'registered_pairs_weak_positive_labels':dict(collections.Counter(p['weak_positive_support_label'] for p in pair_rows)),'clean_contrast_diagnostic':contrast,'per_case':case_stats,'pairs':pair_rows,'limits':review['limits']}
for name,data in [('boundary_support_review.json',review),('boundary_support_join.json',report)]: (O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('coverage','methods','scope_only_weak_positive_label_counts','scope_only_weak_negative_label_counts','clean_contrast_diagnostic')},ensure_ascii=False,indent=2))
print('missed_direct',[(x['audit_case_number'],[(p['reader_rank'],p['weak_positive']) for p in x['missed_direct_in_declared_scope']]) for x in case_stats if x['missed_direct_in_declared_scope']])
