"""Validate explicit primary-model labels; join frozen scores only after review.

No semantic classifier, default label, inference, or training is implemented.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

PREFIX = 'semantic_supervision_20261009T045728Z_'
LABELS = ('direct', 'partial', 'irrelevant', 'contradictory', 'uncertain')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def assemble(source, records, annotations):
    prior_name = annotations['prior_review_file']
    assert Path(prior_name).name == prior_name and prior_name.startswith(PREFIX)
    old_path = records / prior_name
    prior_count = annotations['expected_prior_count']
    new_count = annotations['declared_new_judgments']
    total_count = prior_count + new_count
    old = load(old_path)
    assert sha(source / 'support_review.json') == old['source_support_review_sha256']
    assert sha(source / 'case_study.json') == old['source_case_sha256']
    blind = load(source / 'support_review.json')
    trace = load(source / 'case_study.json')
    assert len(blind['cases']) == len(trace['cases']) == 32
    assert all(len(c['passages']) == 50 for c in blind['cases'])
    assert all(p['support_label'] is None for c in blind['cases'] for p in c['passages'])
    assert annotations['reviewer'] == 'primary_agent_single_model'
    assert annotations['source_support_review_sha256'] == sha(source / 'support_review.json')
    assert annotations['prior_review_sha256'] == sha(old_path)
    assert annotations['new_case_numbers'] == list(range(13,33))
    assert annotations['fully_reviewed_case_numbers'] == list(range(1,33))
    assert prior_count == 783 and new_count == 817
    all_items = {}
    for k, (bc, tc) in enumerate(zip(blind['cases'], trace['cases']), 1):
        assert bc['review_case_id'] == tc['question_sha256']
        assert bc['question'] == tc['question']
        by_id = {p['question_passage_review_id']: (i, p) for i, p in enumerate(tc['candidates'])}
        assert len(by_id) == 50
        for n, p in enumerate(bc['passages'], 1):
            assert p['review_item_id'] not in all_items
            i, tp = by_id[p['review_item_id']]
            assert p['text'] == tp['text'] and p['title'] == tp['title']
            assert hashlib.sha256(tp['text'].encode()).hexdigest() == tp['text_sha256']
            all_items[p['review_item_id']] = (k, n, bc, i, tp)
    judgments = {r['review_item_id']: r for r in old['judgments']}
    assert len(judgments) == len(old['judgments']) == prior_count
    for r in judgments.values():
        k, n, bc, i, tp = all_items[r['review_item_id']]
        assert r['review_case_id'] == bc['review_case_id']
        assert r['support_label'] in LABELS and r['rationale'].strip()
        if r['support_quote']:
            assert r['support_quote'] in tp['text']
    new = []
    ordinals = set()
    for a in annotations['judgments']:
        k, n = a['audit_case_number'], a['blind_ordinal']
        assert isinstance(k, int) and isinstance(n, int)
        assert k in annotations['new_case_numbers'] and 1 <= n <= 50
        assert (k, n) not in ordinals
        ordinals.add((k, n))
        bc = blind['cases'][k - 1]
        rid = bc['passages'][n - 1]['review_item_id']
        assert rid == a['review_item_id'] and rid not in judgments
        _, _, _, i, tp = all_items[rid]
        label, quote = a['support_label'], a['support_quote']
        assert label in LABELS and a['rationale'].strip()
        assert quote is None or (isinstance(quote, str) and quote.strip() and quote in tp['text'])
        assert label not in ('direct', 'partial', 'contradictory') or quote is not None
        r = {'audit_case_number': k, 'blind_ordinal': n, 'candidate_index': i,
             'review_case_id': bc['review_case_id'], 'review_item_id': rid,
             'support_label': label, 'support_quote': quote, 'rationale': a['rationale'],
             'selection': 'original_audit_case_order_remaining_full_top50',
             'reviewer': annotations['reviewer'], 'score_blind_when_reviewed': False,
             'scores_withheld_in_current_reading': True, 'official_gold': False,
             'confidence': 'provisional_single_reviewer', 'source_text_sha256': tp['text_sha256']}
        judgments[rid] = r
        new.append(r)
    assert len(new) == new_count
    assert all(p['review_item_id'] in judgments for c in blind['cases'] for p in c['passages'])
    assert len(judgments) == total_count
    # Preserve all prior labels/metadata byte-independent, not only their counts.
    assert list(judgments.values())[:prior_count] == old['judgments']
    coverage = {str(k): sum(p['review_item_id'] in judgments for p in c['passages'])
                for k, c in enumerate(blind['cases'], 1)}
    fully_reviewed = [int(k) for k, v in coverage.items() if v == 50]
    assert fully_reviewed == annotations['fully_reviewed_case_numbers']
    flags = {r['audit_case_number']: r for r in old['question_target_flags']}
    review = dict(old)
    review.update(schema_version='rag.semantic_support_incremental_review.v1',
                  status='all32_full_cases_complete_provisional_not_training_qualified',
                  prior_review_sha256=sha(old_path), prior_judgments_retained=prior_count,
                  new_judgments=new_count, distinct_reviewed_passages=len(judgments),
                  unreviewed_population=1600-len(judgments),
                  fully_reviewed_case_numbers=fully_reviewed, reviewed_count_by_case=coverage,
                  judgments=list(judgments.values()))
    review['limits'] = [
        'Single primary-model provisional labels, not official gold or three-model consensus.',
        'Not independently blind; previous scores/results were seen, withheld in this reading.',
        'All1600 audit passages reviewed; whole434 training targets remain uncertified.',
        'Existing trained heads saw these questions; no heldout utility or training authorization.',
        'Missing, uncertain, partial and question-target-flagged data are not automatic negatives.',
        'Counts describe this selected32 training-case audit only, not the entire434; exposed100 Silver labels remain evaluation-only.']
    methods = {}
    for name in trace['cases'][0]['methods']:
        counts = Counter(judgments[c['candidates'][i]['question_passage_review_id']]['support_label']
                         for c in trace['cases'] for i in c['methods'][name]['selected_indices'])
        methods[name] = {'selected_slots': 160, 'label_counts': {l: counts[l] for l in LABELS},
                         'not_confirmed_answer_accuracy': True}
    rows = []
    for k in fully_reviewed:
        c = trace['cases'][k-1]
        labeled = [(i, p, judgments[p['question_passage_review_id']]) for i, p in enumerate(c['candidates'])]
        count = Counter(r['support_label'] for i, p, r in labeled)
        row = {'audit_case_number': k, 'review_case_id': c['question_sha256'],
               'population_label_counts': {l: count[l] for l in LABELS},
               'question_target_flag': flags.get(k), 'formal_training_qualified': False,
               'provisional_direct_slot_ceiling': min(5, count['direct']),
               'frozen_weak_boundary_pair_count': len(c['methods']['frozen_reader_topk']['boundary']['pairs']),
               'methods': {}}
        original_fifth = c['methods']['frozen_reader_topk']['selected_indices'][-1]
        for name, m in c['methods'].items():
            selected = set(m['selected_indices'])
            sc = Counter(r['support_label'] for i, p, r in labeled if i in selected)
            missed = [{'candidate_index': i, 'review_item_id': p['question_passage_review_id'],
                       'reader_rank': p['reader_rank'], 'weak_positive': p['weak_positive'],
                       'support_quote': r['support_quote'],
                       'reader_gap_to_original_rank5': p['reader_score']-c['candidates'][original_fifth]['reader_score'],
                       'relative_correction_vs_original_rank5':
                           (m['scores'][i]-p['reader_score'])-
                           (m['scores'][original_fifth]-c['candidates'][original_fifth]['reader_score'])}
                      for i, p, r in labeled if i not in selected and r['support_label'] == 'direct']
            row['methods'][name] = {'selected_label_counts': {l: sc[l] for l in LABELS},
                                    'missed_direct': missed,
                                    'provisional_direct_slot_gap': min(5, count['direct']) - sc['direct']}
        rows.append(row)
    report = {'schema_version': 'rag.semantic_support_fullcase_join.v1',
              'claim_ceiling': 'L0_diagnostic', 'source_case_sha256': sha(source/'case_study.json'),
              'source_support_review_sha256': sha(source/'support_review.json'),
              'coverage': {'reviewed_population':total_count, 'all_population':1600, 'remaining':1600-total_count,
                           'new_judgments':new_count, 'prior_preserved':prior_count,
                           'fully_reviewed_cases':fully_reviewed, 'by_case':coverage},
              'new_label_counts':dict(Counter(r['support_label'] for r in new)),
              'method_totals_existing_selected_union':methods, 'fully_reviewed_cases':rows,
              'limits':review['limits']}
    return review, report


def storage(review, annotations, records, prefix):
    prior_count = annotations['expected_prior_count']
    delta = {k:v for k,v in review.items() if k != 'judgments'}
    delta['schema_version'] = 'rag.semantic_support_review_delta.v1'
    delta['prior_review_file'] = annotations['prior_review_file']
    delta['judgments'] = review['judgments'][prior_count:]
    payload = json.dumps(delta, ensure_ascii=False, indent=2)+'\n'
    manifest = {'schema_version':'rag.complete_provisional_support_review_manifest.v1',
        'prior_file':annotations['prior_review_file'], 'prior_sha256':sha(records/annotations['prior_review_file']),
        'delta_file':prefix+'review_delta.json', 'delta_sha256':hashlib.sha256(payload.encode('utf-8')).hexdigest(),
        'prior_count':prior_count, 'delta_count':len(delta['judgments']),
        'merged_count':len(review['judgments']), 'remaining':review['unreviewed_population'],
        'source_case_sha256':review['source_case_sha256'],
        'source_support_review_sha256':review['source_support_review_sha256'],
        'merge_rule':'prior.judgments followed by delta.judgments; preserve all prior metadata',
        'official_gold':False, 'training_qualified':False, 'claim_ceiling':'L0_diagnostic'}
    return delta, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output-prefix', required=True)
    p.add_argument('--source-dir', required=True, type=Path)
    p.add_argument('--records-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    assert Path(args.annotations).name == args.annotations
    assert Path(args.output_prefix).name == args.output_prefix
    annotations = load(args.records_dir/args.annotations)
    review, report = assemble(args.source_dir, args.records_dir, annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    delta, manifest = storage(review, annotations, args.records_dir, args.output_prefix)
    for kind, data in [('review_delta',delta),('review_manifest',manifest),('join',report)]:
        payload = json.dumps(data, ensure_ascii=False, indent=2)+'\n'
        assert len(payload.encode('utf-8')) < 1048576, kind
        # Hash the same physical UTF-8/LF bytes on Windows and Linux.
        (args.output_dir/(args.output_prefix+kind+'.json')).write_bytes(payload.encode('utf-8'))
    print(json.dumps({'coverage':report['coverage'], 'new_label_counts':report['new_label_counts'],
                      'methods':report['method_totals_existing_selected_union']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
