"""Full1600 reconstruction, immutable history, literal quotes and tamper checks."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-dir', required=True, type=Path)
p.add_argument('--records-dir', required=True, type=Path)
args = p.parse_args()
prefix = 'semantic_supervision_20261009T045728Z_complete_'
spec = importlib.util.spec_from_file_location('complete_validator', args.records_dir/(prefix+'validate.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ann = m.load(args.records_dir/(prefix+'annotations.json'))
review, report = m.assemble(args.source_dir, args.records_dir, ann)
delta, manifest = m.storage(review, ann, args.records_dir, prefix)
assert delta == m.load(args.records_dir/(prefix+'review_delta.json'))
assert manifest == m.load(args.records_dir/(prefix+'review_manifest.json'))
assert report == m.load(args.records_dir/(prefix+'join.json'))
checks = ['exact_delta_manifest_join_replay']
assert review['unreviewed_population'] == 0
assert set(review['reviewed_count_by_case'].values()) == {50}
assert len({r['review_item_id'] for r in review['judgments']}) == 1600
checks.append('full32_by50_unique_complete')
prior = m.load(args.records_dir/ann['prior_review_file'])
assert review['judgments'][:783] == prior['judgments']
assert review['question_target_flags'] == prior['question_target_flags']
assert len(review['question_target_flags']) == 9
checks.append('prior783_metadata_and9_flags_unchanged')
assert len(delta['judgments']) == 817
assert report['new_label_counts'] == {'irrelevant':673, 'partial':105, 'direct':33, 'uncertain':6}
checks.append('explicit817_only_no_default_classification')
assert all(not r['formal_training_qualified'] for r in report['fully_reviewed_cases'])
assert manifest['official_gold'] is False and manifest['training_qualified'] is False
checks.append('coverage_not_training_or_gold_qualification')
assert report['method_totals_existing_selected_union'] == m.load(args.records_dir/'semantic_supervision_20261009T045728Z_fullcase_09_12_join.json')['method_totals_existing_selected_union']
checks.append('all160_selected_slots_per_method_unchanged')
assert all(len(p.read_bytes()) < 1048576 for p in args.records_dir.glob(prefix+'*'))
assert hashlib.sha256((args.records_dir/manifest['delta_file']).read_bytes()).hexdigest() == manifest['delta_sha256']
checks.append('sub1MiB_shards_hash_reconstruction')
diagnostics = m.load(args.records_dir/(prefix+'diagnostics.json'))
assert diagnostics['label_counts'] == {'direct':97, 'irrelevant':1259, 'partial':223, 'uncertain':18, 'contradictory':3}
assert diagnostics['unflagged_zero_direct_cases'] == [11,22,25]
assert diagnostics['unflagged_direct_slot_gap_total'] == 25
assert diagnostics['unflagged_reader_direct_selected'] == 48
assert diagnostics['unflagged_direct_slot_ceiling'] == 73
checks.append('audit_only_support_ceiling_not_answer_accuracy')
assert diagnostics['provisional_gap_only_against_selected_partial_cases'] == [3,9,19,29]
assert diagnostics['provisional_rescue_with_selected_irrelevant_cases'] == [5,6,8,13,16,17,21,26,27]
checks.append('partial_not_hard_negative_retention_cases_separated')
assert report['fully_reviewed_cases'][17]['methods']['frozen_reader_topk']['provisional_direct_slot_gap'] == 0
assert report['fully_reviewed_cases'][22]['methods']['frozen_reader_topk']['provisional_direct_slot_gap'] == 0
checks.append('more_than5_direct_not_forced_into5')

def reject(name, mutate):
    a = copy.deepcopy(ann)
    mutate(a)
    try:
        m.assemble(args.source_dir, args.records_dir, a)
    except (AssertionError, KeyError, IndexError):
        checks.append(name)
    else:
        raise AssertionError('Invalid input accepted: '+name)

reject('duplicate_explicit_id', lambda a: a['judgments'].append(copy.deepcopy(a['judgments'][0])))
reject('missing_passage_not_negative', lambda a: a['judgments'].pop())
reject('false_partial_completion', lambda a: a.update(fully_reviewed_case_numbers=list(range(1,32))))
reject('wrong_source_hash', lambda a: a.update(source_support_review_sha256='0'*64))
reject('wrong_prior_review_hash', lambda a: a.update(prior_review_sha256='0'*64))
reject('wrong_blind_id', lambda a: a['judgments'][0].update(review_item_id='0'*64))
reject('wrong_case_ordinal', lambda a: a['judgments'][0].update(blind_ordinal=0))
reject('wrong_reviewer', lambda a: a.update(reviewer='unrecorded_external_consensus'))
direct = next(i for i,r in enumerate(ann['judgments']) if r['support_label']=='direct')
partial = next(i for i,r in enumerate(ann['judgments']) if r['support_label']=='partial')
reject('fabricated_support_quote', lambda a: a['judgments'][direct].update(support_quote='NOT IN SOURCE 847243'))
reject('direct_missing_quote', lambda a: a['judgments'][direct].update(support_quote=None))
reject('partial_missing_quote', lambda a: a['judgments'][partial].update(support_quote=None))
reject('unread_null_not_negative', lambda a: a['judgments'][0].update(support_label=None))
reject('unknown_label', lambda a: a['judgments'][0].update(support_label='auto_gold'))
reject('missing_rationale', lambda a: a['judgments'][0].update(rationale=' '))
print(json.dumps({'checks_passed':len(checks),'checks':checks,'source_rewritten':False,
                  'training_or_inference':False,'reviewed':1600,'remaining':0}))
