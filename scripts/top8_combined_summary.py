"""Compact exploratory ranking: archived DPR seeds plus current BGE owner seeds.

This is explicitly not the matched, prospective two-backbone three-seed matrix.
Only aggregate metrics are consumed; no candidate texts or labels are opened.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(selection, bge):
    metrics=bge['metrics'];base=metrics['predictions-bge-NATIVE.json'];rows=[]
    for idea in selection['selected']:
        key=idea['key'];cells=[(name,v) for name,v in metrics.items()
            if name.startswith('bge-'+key+'-QUANTUM-s')]
        classical={}
        for name,v in metrics.items():
            if not name.startswith('bge-'+key+'-') or '-QUANTUM-' in name:continue
            arm=name.split('/')[0].split('-')[2]
            classical.setdefault(arm,[]).append(v)
        entry=dict(idea=key,initial_dpr_rank=idea['rank'],bge_completed_quantum_seeds=len(cells),
            dpr_mean_useful_lower=mean(idea['useful_lower']),
            dpr_mean_useful_upper=mean(idea['useful_upper']),
            dpr_archived_seeds=len(idea['useful_lower']),
            dpr_evidence_sha256=idea['evidence_sha256'],
            bge_controls={arm:dict(useful_lower=mean(x['useful_lower'] for x in values),
                useful_upper=mean(x['useful_upper'] for x in values)) for arm,values in classical.items()})
        if cells:
            lo=mean(v['useful_lower'] for _,v in cells);hi=mean(v['useful_upper'] for _,v in cells)
            dlo=entry['dpr_mean_useful_lower']-selection['baselines']['dpr']['useful']
            dhi=entry['dpr_mean_useful_upper']-selection['baselines']['dpr']['useful']
            blo=lo-base['useful_upper'];bhi=hi-base['useful_lower']
            entry.update(bge_useful_lower=lo,bge_useful_upper=hi,bge_native_delta_lower=blo,
                bge_native_delta_upper=bhi,dpr_native_delta_lower=dlo,dpr_native_delta_upper=dhi,
                provisional_equal_weight_delta_lower=(dlo+blo)/2,
                provisional_equal_weight_delta_upper=(dhi+bhi)/2)
        rows.append(entry)
    rows.sort(key=lambda x:(not x['bge_completed_quantum_seeds'],
        -x.get('provisional_equal_weight_delta_lower',float('-inf')),x['idea']))
    for i,row in enumerate(rows,1):row['display_rank']=i if row['bge_completed_quantum_seeds'] else None
    return dict(schema='top8.provisional-dual-summary.v1',rows=rows,bge_native=base,
        matching_seed_matrix=False,independent_confirmation=False,exposed_development=True,
        interpretation='Exploratory historical DPR plus owner BGE; unequal seed counts; interval overlap is unresolved.',
        definitive_dual_rank='PENDING junior same-code same-seed full matrix',quantum_attribution='NOT_ESTABLISHED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--bge-evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(json.loads(a.selection.read_bytes()),json.loads(a.bge_evaluation.read_bytes()))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()
