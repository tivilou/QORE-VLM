"""Readback and adversarial checks for incremental explicit support labels."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-dir', required=True, type=Path)
p.add_argument('--records-dir', required=True, type=Path)
args = p.parse_args()
prefix = 'semantic_supervision_20261009T045728Z_fullcase_01_04_'
spec = importlib.util.spec_from_file_location('fullcase_validator', args.records_dir/(prefix+'validate.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ann = m.load(args.records_dir/(prefix+'annotations.json'))
review, report = m.assemble(args.source_dir, args.records_dir, ann)
assert review == m.load(args.records_dir/(prefix+'review.json'))
assert report == m.load(args.records_dir/(prefix+'join.json'))
assert report['coverage']['remaining'] == 1151
assert report['fully_reviewed_cases'][2]['methods']['frozen_reader_topk']['missed_direct'][0]['reader_rank'] == 9
assert report['fully_reviewed_cases'][2]['methods']['frozen_reader_topk']['missed_direct'][0]['weak_positive'] is False
checks = ['exact_record_replay_and_new_direct_identity']

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
reject('missing_passage_not_complete', lambda a: a['judgments'].pop())
reject('false_full32_coverage', lambda a: a.update(fully_reviewed_case_numbers=list(range(1,33))))
reject('wrong_source_hash', lambda a: a.update(source_support_review_sha256='0'*64))
reject('wrong_prior_review_hash', lambda a: a.update(prior_review_sha256='0'*64))
reject('wrong_blind_id', lambda a: a['judgments'][0].update(review_item_id='0'*64))
direct = next(i for i, x in enumerate(ann['judgments']) if x['support_label']=='direct')
reject('fabricated_support_quote', lambda a: a['judgments'][direct].update(support_quote='NOT IN SOURCE 847243'))
reject('direct_missing_quote', lambda a: a['judgments'][direct].update(support_quote=None))
reject('unread_null_not_negative', lambda a: a['judgments'][0].update(support_label=None))
print(json.dumps({'checks_passed':len(checks),'checks':checks,'source_rewritten':False,
                  'training_or_inference':False,'reviewed':449,'remaining':1151}))
