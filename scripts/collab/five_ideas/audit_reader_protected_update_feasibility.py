#!/usr/bin/env python3
"""Independently reopen all59 native forwards for bounded feasibility witnesses.

This reads only the frozen training inputs and the declared diagnostic run.
It does not search new directions or open evaluation judgments.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag.quantum_reader_rotation import make_rotation_head, RotationConfig
from applications.rag.protected_update_feasibility import membership_counts
from applications.rag.protected_boundary_inputs import load_bundle
from scripts.collab.five_ideas.run_reader_protected_update_feasibility import inputs, digest, validate_numeric_trace


def audit(args):
    torch.set_num_threads(1)
    run = args.run
    trace = json.loads((run / 'selector_trace.json').read_text())
    if trace['synthetic_only'] or trace['population'] != {'training': 59, 'evaluation': 0}:
        raise ValueError('audit requires actual training-only population')
    cfg = trace['config']
    args.synthetic = False
    features, records, targets, weight, bias, _ = inputs(args, cfg)
    if targets != trace['training_targets']:
        raise ValueError('qualification target drift')
    ranks = [r['candidate_retrieved_ranks'] for r in records]
    if ranks != trace['ranks']:
        raise ValueError('rank identity drift')
    snapshot = trace['provenance']['source_snapshot']
    if digest(run / snapshot['file']) != snapshot['sha256']:
        raise ValueError('source archive integrity drift')
    with zipfile.ZipFile(run / snapshot['file']) as archive:
        manifest = json.loads(archive.read('source_manifest.json'))
        if manifest['source_hashes'] != snapshot['source_hashes']:
            raise ValueError('archived manifest drift')
        for name, expected in manifest['source_hashes'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('archived source content drift')
    checks = validate_numeric_trace(run)
    rows, _ = load_bundle(args.bundle)
    report = {'schema_version': 'rag.reader_protected_update_audit.v1',
              'run_id': trace['run_id'], 'status': 'passed',
              'trace_sha256': digest(run / 'selector_trace.json'),
              'values_sha256': digest(run / 'feasibility_values.npz'),
              'auditor_sha256': digest(Path(__file__)),
              'source_files_verified': len(snapshot['source_hashes']),
              'readiness': checks, 'all59_native_baselines': 0,
              'all59_native_witnesses': 0, 'checkpoints': [],
              'data_access': 'training59 only; no evaluation100/validation87 judgments or features',
              'claim_ceiling': 'L0_training_conditioned_feasibility'}
    with np.load(run / 'feasibility_values.npz', allow_pickle=False) as z:
        for point in trace['checkpoints']:
            state = {k: torch.from_numpy(z[v].copy()) for k, v in point['head_state'].items()}
            family, checkpoint = point['id'].split('_')
            with np.load(args.source / 'membership_values.npz', allow_pickle=False) as source:
                for k, v in state.items():
                    np.testing.assert_array_equal(v.numpy(), source[family + '_membership__' + checkpoint + '__' + k])
            head = make_rotation_head(weight, bias, family=family, config=RotationConfig(**cfg['source_rotation']))
            head.load_state_dict(state)

            def forward(theta):
                with torch.no_grad():
                    head.theta.copy_(torch.as_tensor(theta, dtype=state['theta'].dtype))
                    return np.stack([head(f['cls'], f['pooled'])['scores'].numpy() for f in features])

            original = state['theta'].numpy().astype(np.float64)
            baseline = forward(original)
            np.testing.assert_array_equal(baseline, z[point['zero_scores']])
            if membership_counts(baseline, targets, ranks) != point['zero_membership']:
                raise ValueError('baseline membership drift')
            report['all59_native_baselines'] += 1
            record = {'id': point['id'], 'baseline_rescued': point['zero_membership']['rescued'],
                      'baseline_lost_protected': point['zero_membership']['lost_protected'],
                      'trials': [], 'verified_witnesses': [], 'fixed_first2': []}
            for trial in point['trials']:
                record['trials'].append({'attempt': trial['attempt'], 'mode': trial['mode'],
                    'proposal': trial['proposal'], 'solver': trial['solver'],
                    'actual_checks': [{k: v for k, v in r.items() if k not in ('scores', 'membership')}
                        | {k: v for k, v in r['membership'].items() if k != 'selected_indices'}
                        for r in trial['nonlinear']]})
                # Replay every accepted witness, including rescue-only witnesses;
                # do not silently verify just the chosen global-protection arm.
                for check in trial['nonlinear']:
                    if not check['safe_witness']:
                        continue
                    delta = z[trial['delta']].reshape(original.shape)
                    scores = forward(original + check['fraction'] * delta)
                    np.testing.assert_array_equal(scores, z[check['scores']])
                    actual = membership_counts(scores, targets, ranks)
                    if actual != check['membership'] or actual['lost_protected'] or actual['rescued'] < 1:
                        raise ValueError('all59 actual witness drift')
                    case = trial['proposal']['case']
                    if trial['proposal']['added'] not in actual['selected_indices'][case]:
                        raise ValueError('declared rescue missing')
                    report['all59_native_witnesses'] += 1
                    changes = []
                    for i, target in enumerate(targets):
                        before = target['baseline_top5_indices']
                        after = actual['selected_indices'][i]
                        if set(before) == set(after):
                            continue
                        changes.append({'id': records[i]['prefix'], 'before': before, 'after': after,
                            'added': sorted(set(after) - set(before)), 'removed': sorted(set(before) - set(after)),
                            'eligible_rescued': sorted(set(after) & set(target['eligible_positive_indices'])),
                            'lost_protected': sorted(set(target['immutable_selected_indices']) - set(after))})
                    record['verified_witnesses'].append({'attempt': trial['attempt'], 'mode': trial['mode'],
                        'fraction': check['fraction'], 'rescued': actual['rescued'], 'lost_protected': 0,
                        'infinity_norm': float(np.abs(check['fraction'] * delta).max()),
                        'proposed_case': records[case]['prefix'], 'question': rows[case]['case']['question'],
                        'changes': changes})
            for i in range(2):
                record['fixed_first2'].append({'id': records[i]['prefix'],
                    'question': rows[i]['case']['question'],
                    'baseline_top5': targets[i]['baseline_top5_indices'],
                    'checkpoint_top5': point['zero_membership']['selected_indices'][i],
                    'protected': targets[i]['immutable_selected_indices'],
                    'eligible': targets[i]['eligible_positive_indices']})
            report['checkpoints'].append(record)
    output = run / 'independent_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'path': str(output), 'status': report['status'],
                     'all59_native_baselines': report['all59_native_baselines'],
                     'all59_native_witnesses': report['all59_native_witnesses']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    audit(parser.parse_args())
