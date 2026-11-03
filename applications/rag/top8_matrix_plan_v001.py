"""Standard-library planning for common and backbone-specific experiment cells.

Scheduling a method on DPR alone does not assert that BGE is incompatible. A
missing requested result is a failure/incomplete result, never an N/A or zero.
"""
from __future__ import annotations


def matrix_plan(available, registry, common=None, dpr_only=None, bge_only=None):
    available = list(available)
    registry = list(registry)
    dpr_only, bge_only = list(dpr_only or []), list(bge_only or [])
    if not available or len(set(available)) != len(available) or not set(available) <= {'dpr', 'bge'}:
        raise ValueError('unique available DPR/BGE backbones required')
    for kind, values in [('dpr', dpr_only), ('bge', bge_only)]:
        if values and kind not in available:
            raise ValueError(kind + '-only ideas require that backbone model/cache')
    common = list(common) if common is not None else [
        idea for idea in registry if idea not in dpr_only + bge_only]
    selected = common + dpr_only + bge_only
    if not selected or len(selected) > 8 or len(set(selected)) != len(selected):
        raise ValueError('one to eight unique ideas; common and exclusive lists must be disjoint')
    if not set(selected) <= set(registry):
        raise ValueError('selected ideas must have an implemented registered adapter')
    scheduled = {kind: common + (dpr_only if kind == 'dpr' else bge_only)
                 for kind in available}
    if any(not values for values in scheduled.values()):
        raise ValueError('each supplied backbone must have a scheduled idea')
    return dict(schema='top8.backbone-plan.v1', ideas=selected,
                backbone_ideas=scheduled,
                not_scheduled=[dict(backbone=kind, idea=idea, status='NOT_SCHEDULED',
                    reason='explicit backbone scope; not a failed or zero-valued result')
                    for kind in ('dpr', 'bge') for idea in selected
                    if idea not in scheduled.get(kind, [])])


def rank_metrics(seal, metrics):
    """Rank like scopes together; preserve legacy seals' two-backbone contract."""
    from statistics import mean
    groups = {'dual': [], 'dpr_only': [], 'bge_only': [], 'fusion': []}
    schedule = seal.get('backbone_ideas')
    ranking = []
    for idea in seal['ideas']:
        required = [kind for kind in ('dpr', 'bge')
                    if schedule is None or idea in schedule.get(kind, [])]
        if not required:
            raise ValueError('sealed idea has no requested backbone')
        group = 'dual' if len(required) == 2 else required[0] + '_only'
        by_model = {}; deltas = []; complete = True
        for kind in required:
            base = metrics.get('predictions-' + kind + '-NATIVE.json')
            prefix = kind + '-' + idea + '-QUANTUM-s'
            cells = [(key, value) for key, value in metrics.items() if key.startswith(prefix)]
            identities = [key.split('/')[0][len(prefix):] for key, _ in cells]
            expected = [str(seed) for seed in seal['seeds']]
            if not base or not expected or sorted(identities) != sorted(expected):
                complete = False
                continue
            lower = mean(value['useful_lower'] - base['useful_upper'] for _, value in cells)
            upper = mean(value['useful_upper'] - base['useful_lower'] for _, value in cells)
            by_model[kind] = dict(mean_native_delta_lower=lower,
                                  mean_native_delta_upper=upper, seeds=len(cells))
            deltas.append((lower, upper))
        lower = mean(value[0] for value in deltas) if complete else None
        upper = mean(value[1] for value in deltas) if complete else None
        entry = dict(idea=idea, comparison_group=group, requested_backbones=required,
            not_scheduled_backbones=[kind for kind in ('dpr', 'bge') if kind not in required],
            by_model=by_model, rankable=complete,
            scope_native_delta_lower=lower, scope_native_delta_upper=upper,
            equal_weight_native_delta_lower=lower if group == 'dual' else None,
            equal_weight_native_delta_upper=upper if group == 'dual' else None)
        groups[group].append(entry)
    for group, entries in groups.items():
        entries.sort(key=lambda row: (not row['rankable'],
            -row['scope_native_delta_lower'] if row['rankable'] else 0, row['idea']))
        previous = None; rank = None
        for index, row in enumerate(entries, 1):
            if row['rankable']:
                value = row['scope_native_delta_lower']
                if value != previous:
                    rank = index
                row['scope_rank'] = rank
                previous = value
            else:
                row['scope_rank'] = None
        ranking.extend(entries)
    # A fusion cell consumes both raw score channels but produces ONE ranking.
    # Report per-seed raw counts against each native, never fabricate two cells.
    for idea in seal.get('fusion_ideas', []):
        prefix = 'fusion-' + idea + '-QUANTUM-s'
        cells = [(k,v) for k,v in metrics.items() if k.startswith(prefix)]
        expected = sorted(map(str, seal['seeds']))
        complete = bool(expected) and sorted(k.split('/')[0][len(prefix):] for k,_ in cells) == expected
        per_seed = {k.split('/')[0][len(prefix):]:v for k,v in cells}
        by_native = {}
        for kind in ('dpr', 'bge'):
            base = metrics.get('predictions-'+kind+'-NATIVE.json')
            complete = complete and base is not None
            if base is not None:
                aligned = {s:v for s,v in per_seed.items()
                           if v.get('questions') == base.get('questions')
                           and v.get('denominator') == base.get('denominator')
                           and isinstance(base.get('questions'), int) and base['questions'] > 0
                           and base.get('denominator') == 5*base['questions']}
                complete = complete and len(aligned) == len(per_seed)
                by_native[kind] = dict(native=base, per_seed={s:dict(
                    useful_delta=[v['useful_lower']-base['useful_upper'], v['useful_upper']-base['useful_lower']],
                    direct_delta=[v['direct_lower']-base['direct_upper'], v['direct_upper']-base['direct_lower']])
                    for s,v in aligned.items()})
        entry = dict(idea=idea, comparison_group='fusion', requested_backbones=['dpr', 'bge'],
            output_rankings_per_seed=1, rankable=bool(complete), per_seed=per_seed, by_native=by_native,
            classical_controls={k:v for k,v in metrics.items()
                if k.startswith('fusion-'+idea+'-') and '-QUANTUM-' not in k},
            scope_rank=None, quantum_attribution='NOT_ESTABLISHED')
        groups['fusion'].append(entry); ranking.append(entry)
    return ranking, groups
