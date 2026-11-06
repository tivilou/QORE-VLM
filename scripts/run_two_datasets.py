#!/usr/bin/env python3
"""Independent NQ-open/TriviaQA matrices; same pinned backbones, no pooled training.

Standard-library coordinator. --plan-only writes nothing and needs no credentials.
Missing qualified releases stop both datasets before downloads or scientific work.
Model/data packages are private, hash-verified, and cached on the runner's server.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unicodedata

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_junior_experiments as single
from applications.rag.junior_transport_v001 import (
    Exchange, atomic_json, digest, install_package, member, require,
)

CAMPAIGN = ROOT / 'configs/experiments/idea_search_20261103/two-dataset-campaign.v001.json'
DATASETS = ('nq_open', 'triviaqa')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def registry():
    # Never import Torch/model implementation for planning or bootstrap validation.
    import ast
    tree = ast.parse((ROOT/'applications/rag/top8_methods_v001.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in n.targets))


def resolve(campaign, config, datasets, population):
    require(datasets and len(set(datasets)) == len(datasets) and set(datasets) <= set(DATASETS),
            'unique registered datasets required')
    template = json.loads((ROOT/campaign['base_campaign']).read_bytes())
    # The two-dataset campaign extends, rather than mutates, the old eight-only entry.
    template['fusion_ideas'] = campaign.get('fusion_ideas', [])
    from applications.rag.qduet_fusion_v001 import fusion_plan, FUSION_ARMS
    fusion_plan(['dpr', 'bge'], template['fusion_ideas'])
    arms = registry(); ideas = template['dual_ideas'] + template['dpr_only_ideas']
    require(template['seeds'] and len(set(template['seeds'])) == len(template['seeds']), 'fixed unique seeds')
    require(len(template['dual_ideas']) == 8 and len(set(ideas)) == len(ideas) and set(ideas) <= set(arms),
            'eight implemented ideas plus distinct DPR additions required')
    plans = {}; readiness = []
    for name in datasets:
        spec = campaign['datasets'][name]
        release = config.get('datasets', {}).get(name, {}).get('populations', {}).get(
            population, spec['populations'].get(population))
        # Explicit reference reuses the existing pinned B1 package without inventing NQ data.
        if release and 'base_population' in release:
            release = copy.deepcopy(template['populations'][release['base_population']])
            release['dataset'] = name
        candidate = copy.deepcopy(template)
        candidate['dataset'] = name
        candidate['populations'] = {population: release or dict(status='unavailable', reason='No release registered')}
        reason = None
        try:
            bound = single.package_for(candidate, population)
            require(bound.get('dataset') == name, 'release dataset identity mismatch')
            require(bound.get('models') == campaign['models'], 'same pinned DPR/BGE model layout required')
            if population == 'formal':
                require(bound.get('qualification', {}).get('cross_dataset_question_audit')
                        and bound['qualification'].get('model_visible_label_audit')
                        and bound['qualification'].get('oracle_or_headroom_evidence'),
                        'formal release requires overlap, visible-text labels and headroom evidence')
            plans[name] = candidate
        except (ValueError, KeyError, TypeError) as error:
            reason = str(error)
        readiness.append(dict(dataset=name, ready=reason is None, reason=reason))
    per_dataset = len(template['seeds']) * (2*sum(len(arms[i]) for i in template['dual_ideas'])
                         + sum(len(arms[i]) for i in template['dpr_only_ideas'])
                         + sum(len(FUSION_ARMS[i]) for i in template['fusion_ideas']))
    report = dict(schema='qduet.two-dataset-plan.v1', datasets=list(datasets), population=population,
        training='independent_per_dataset', cross_dataset_transfer='NOT_SCHEDULED',
        backbones=['dpr', 'bge'], models=campaign['model_identities'],
        seeds=template['seeds'], ideas=template['dual_ideas'], dpr_only=template['dpr_only_ideas'],
        fusion_ideas=template['fusion_ideas'], fusion_arms={i:FUSION_ARMS[i] for i in template['fusion_ideas']},
        epochs=4, learned_cells_per_dataset=per_dataset, total_learned_cells=per_dataset*len(datasets),
        readiness=readiness, ready=all(r['ready'] for r in readiness),
        aggregation='equal dataset/model weight; useful per question delta; unknown bounds retained',
        scientific_execution=False, formal_scale_certified=False)
    return plans, report


class AssetStore:
    """One verified extraction per exact archive/member inventory; no local relay."""
    def __init__(self, root, exchange, reserve):
        self.root = Path(root); self.exchange = exchange; self.reserve = reserve
        self.staged = {}

    def __call__(self, release):
        binding = dict(archive=release['archive'], files=release['files'])
        key = canonical_sha(binding)
        if key in self.staged:
            return self.staged[key]
        path = self.root/key; path.mkdir(parents=True, exist_ok=True)
        archive = path/'assets.tar'; assets = path/'assets'
        receipt = path/'installed.json'
        if receipt.exists():
            require(json.loads(receipt.read_bytes()) == binding, 'asset inventory drift')
            for row in release['files']:
                p = member(assets, row['path'])
                require(p.is_file() and p.stat().st_size == row['size'] and digest(p) == row['sha256'],
                        'installed asset bytes changed')
        else:
            require(shutil.disk_usage(path).free >= release['archive']['size']
                    + sum(r['size'] for r in release['files']) + self.reserve,
                    'insufficient server storage; expand project disk')
            self.exchange.download(release['archive'], archive)
            install_package(archive, assets, release['files'])
            atomic_json(receipt, binding)
        self.staged[key] = assets
        return assets


def audit_inputs(staged, plans, population):
    """Cross-dataset train/evaluation identity check without opening label values.

    NFKC/case/whitespace normalization is for identity checks, never model text.
    It is not a semantic duplicate certification. Shared corpus passages allowed.
    """
    seen = {'train': set(), 'evaluation': set()}; counts = {}; model_inventory = None
    duplicate_eval = 0
    for name, assets in staged.items():
        release = plans[name]['populations'][population]
        counts[name] = single.validate_inputs(assets, release)
        inventory = sorted((kind, row['path'].split(prefix+'/', 1)[1], row['size'], row['sha256'])
            for kind, prefix in release['models'].items() for row in release['files']
            if row['path'].startswith(prefix+'/'))
        require(inventory and all(any(r[0] == k for r in inventory) for k in ('dpr', 'bge')),
                'both pinned model inventories required')
        require(model_inventory is None or inventory == model_inventory, 'different pretrained model bytes across datasets')
        model_inventory = inventory
        for role, key in [('train', 'train'), ('evaluation', 'eval_inputs')]:
            current = set()
            with member(assets, release['inputs'][key]).open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    normalized = ' '.join(unicodedata.normalize('NFKC', row['question']).casefold().split())
                    h = hashlib.sha256(normalized.encode()).hexdigest()
                    require(h not in current, 'normalized duplicate question within role')
                    current.add(h)
            duplicate_eval += len(current & seen['evaluation']) if role == 'evaluation' else 0
            seen[role].update(current)
    require(not seen['train'] & seen['evaluation'], 'cross-dataset train/evaluation question leakage')
    return dict(counts=counts, train_eval_overlap=0, evaluation_overlap=duplicate_eval,
                semantic_duplicate_certification=False, evaluation_label_values_opened=False,
                shared_pretrained_models=True)


def summarize(evaluations, datasets, ideas, seeds, fusion_ideas=None):
    """Average native-relative rates, never raw totals from different populations."""
    rows = []
    for idea in ideas:
        by_cell = {}; intervals = []; direct_intervals = []; complete = True
        for name in datasets:
            evaluation = evaluations.get(name, {})
            metrics = evaluation.get('metrics', {})
            for kind in ('dpr', 'bge'):
                base = metrics.get('predictions-'+kind+'-NATIVE.json')
                prefix = kind+'-'+idea+'-QUANTUM-s'
                cells = [(key, value) for key, value in metrics.items() if key.startswith(prefix)]
                identities = [key.split('/')[0][len(prefix):] for key, _ in cells]
                if not base or sorted(identities) != sorted(map(str, seeds)):
                    complete = False; continue
                n = base['questions']
                require(type(n) is int and n > 0 and base['denominator'] == 5*n
                        and all(v['questions'] == n and v['denominator'] == 5*n for _, v in cells),
                        'common exact evaluation denominator required')
                lo = sum(v['useful_lower']-base['useful_upper'] for _, v in cells)/len(seeds)/n
                hi = sum(v['useful_upper']-base['useful_lower'] for _, v in cells)/len(seeds)/n
                dlo = sum(v['direct_lower']-base['direct_upper'] for _, v in cells)/len(seeds)/n
                dhi = sum(v['direct_upper']-base['direct_lower'] for _, v in cells)/len(seeds)/n
                by_cell[name+'/'+kind] = dict(questions=n, mean_useful_delta_per_question=[lo, hi],
                    mean_direct_delta_per_question=[dlo, dhi],
                    selected_unreviewed=sum(v['unknown'] for _, v in cells),
                    classical_controls={k:v for k,v in metrics.items()
                        if k.startswith(kind+'-'+idea+'-') and '-QUANTUM-' not in k})
                intervals.append((lo, hi)); direct_intervals.append((dlo, dhi))
        rows.append(dict(idea=idea, rankable=complete, by_dataset_model=by_cell,
            equal_weight_useful_delta_lower=sum(x[0] for x in intervals)/len(intervals) if complete else None,
            equal_weight_useful_delta_upper=sum(x[1] for x in intervals)/len(intervals) if complete else None,
            improves_all_dataset_models=complete and all(x[0]>0 for x in intervals),
            direct_protected_all=complete and all(x[0]>=0 for x in direct_intervals)))
    rows.sort(key=lambda r:(not r['rankable'], -r['equal_weight_useful_delta_lower'] if r['rankable'] else 0, r['idea']))
    previous = None; rank = None
    for index, row in enumerate(rows, 1):
        if row['rankable']:
            if row['equal_weight_useful_delta_lower'] != previous:
                rank = index
            previous = row['equal_weight_useful_delta_lower']
            row['rank'] = rank
        else:
            row['rank'] = None
    fusion = []
    for idea in fusion_ideas or []:
        by_dataset = {}; complete = True
        for name in datasets:
            from applications.rag.top8_matrix_plan_v001 import rank_metrics
            _, groups = rank_metrics(dict(ideas=[], seeds=seeds, fusion_ideas=[idea]),
                                     evaluations.get(name, {}).get('metrics', {}))
            entry = groups['fusion'][0]; complete = complete and entry['rankable']
            by_dataset[name] = entry
        fusion.append(dict(idea=idea, comparison_group='fusion', rankable=complete,
                           by_dataset=by_dataset, combined_with_single_model_ranking=False))
    return dict(schema='qduet.two-dataset-summary.v2', datasets=list(datasets), ranking=rows,
                fusion_results=fusion,
                quantum_attribution='NOT_ESTABLISHED', cross_dataset_transfer='NOT_RUN')


@contextmanager
def campaign_lock(work):
    require(os.name == 'posix', 'execution requires the junior Linux server; --plan-only works anywhere')
    import fcntl
    work.mkdir(parents=True, exist_ok=True)
    fd = os.open(work/'two-dataset.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def execute(campaign, config, datasets, population, work, *, prepare_only=False,
            runner=subprocess.run, environment=single.prepare_environment, exchange_factory=Exchange):
    plans, report = resolve(campaign, config, datasets, population)
    from applications.rag.junior_readiness_v001 import release_error
    require(report['ready'], release_error(report))
    work = Path(work).resolve()
    with campaign_lock(work) as lock_fd:
        exchange = exchange_factory(config['url'], config['token'])
        # Immutable before downloads, labels or any model execution.
        bound = dict(campaign=canonical_sha(campaign), plans=canonical_sha(plans), population=population,
                     coordinator_sources={name:digest(ROOT/name) for name in
                        ['scripts/run_two_datasets.py','scripts/run_junior_experiments.py',
                         'applications/rag/junior_transport_v001.py']})
        path = work/'two-dataset-binding.json'
        if path.exists():
            require(json.loads(path.read_bytes()) == bound, 'changed dataset/campaign requires a new workspace')
        else:
            atomic_json(path, bound)
        store = AssetStore(work/'shared-assets', exchange, campaign['reserve_bytes'])
        staged = {name:store(plan['populations'][population]) for name,plan in plans.items()}
        audit = audit_inputs(staged, plans, population)
        atomic_json(work/'input-audit.json', audit)
        if prepare_only:
            return 0
        # Reuse one environment, and inherit BOTH locks across the scientific child.
        def locked_runner(command, **kwargs):
            fds = tuple(kwargs.pop('pass_fds', ()))
            return runner(command, pass_fds=tuple(set(fds+(lock_fd,))), **kwargs)
        shared_python = None
        def shared_environment(*args):
            nonlocal shared_python
            if shared_python is None:
                shared_python = environment(work, config, runner)
            return shared_python
        statuses = {}; evaluations = {}
        for name, plan in plans.items():
            code = single.execute(plan, config | {'populations':plan['populations']}, population,
                work/'datasets'/name, runner=locked_runner,
                environment=shared_environment, exchange_factory=exchange_factory,
                asset_store=lambda release:staged[name])
            statuses[name] = code
            output = work/'datasets'/name/('run-'+population)
            if (output/'matrix/evaluation.json').is_file():
                execution = json.loads((output/'execution.json').read_bytes())
                # Failed/incomplete matrices cannot receive a combined rank.
                if code == 0 and execution['scientific_success']:
                    evaluations[name] = json.loads((output/'matrix/evaluation.json').read_bytes())
        report = summarize(evaluations, datasets, campaign['dual_ideas'], campaign['seeds'],
                           campaign.get('fusion_ideas', []))
        report['returncodes'] = statuses
        report['population'] = population
        report['input_audit_sha256'] = digest(work/'input-audit.json')
        atomic_json(work/'two-dataset-summary.json', report)
        prefix = 'five_ideas/top8-two-dataset-results/'+canonical_sha(bound)[:24]
        receipt = exchange.upload(work/'two-dataset-summary.json', prefix+'/summary.json')
        atomic_json(work/'two-dataset-upload.json', receipt)
        return 0 if all(code == 0 for code in statuses.values()) else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--population', choices=['formal', 'reproduction'], default='formal')
    p.add_argument('--workspace', type=Path, default=Path('qduet-two-dataset-work'))
    p.add_argument('--config', type=Path)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--diagnose', action='store_true', help='Read-only release diagnosis; no downloads, prompts or science.')
    p.add_argument('--check-local-models', type=Path,
                   help='With --diagnose, hash existing models/ beneath this root against the pinned inventory.')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args(argv)
    try:
        campaign = json.loads(CAMPAIGN.read_bytes())
        config = json.loads(args.config.read_bytes()) if args.config else {}
        # Auto-discover the same private file; do not prompt/log secrets during planning.
        if not args.config:
            options = [Path(os.environ.get('QDUET_HANDOFF_CONFIG','~/.config/qduet/junior.json')).expanduser(),
                       Path('/etc/qore-upload/client.json')]
            for candidate in options:
                if candidate.is_file():
                    config = json.loads(candidate.read_bytes()); break
        _, report = resolve(campaign, config, args.datasets, args.population)
        if args.check_local_models is not None and not args.diagnose:
            raise ValueError('--check-local-models requires --diagnose; no implicit training')
        if args.diagnose:
            from applications.rag.junior_readiness_v001 import diagnosis, local_models
            value = diagnosis(report, access_config_present=bool(config.get('url') and config.get('token')))
            if args.check_local_models is not None:
                template = json.loads((ROOT/campaign['base_campaign']).read_bytes())
                value['local_models'] = local_models(template, args.check_local_models)
            print(json.dumps(value, ensure_ascii=False, indent=2)); return 0
        if args.plan_only:
            print(json.dumps(report, ensure_ascii=False, indent=2)); return 0
        from applications.rag.junior_readiness_v001 import release_error
        require(report['ready'], release_error(report))
        if not config.get('url') or not config.get('token'):
            config.update(single.configuration(args.config))
        return execute(campaign, config, args.datasets, args.population, args.workspace,
                       prepare_only=args.prepare_only)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('未完成: '+type(error).__name__+': '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
