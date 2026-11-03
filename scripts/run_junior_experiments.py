#!/usr/bin/env python3
"""One entry: authenticated assets -> environment -> frozen matrix -> result receipts.

Default is FORMAL and refuses an unavailable or exposed dataset release. Use
--population reproduction explicitly for the existing exposed B1 population.
No scientific imports occur in this bootstrap; training runs in a child.
"""
from __future__ import annotations
import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from applications.rag.junior_transport_v001 import (
    Exchange, atomic_json, digest, install_package, member, record_ok, relative, require,
)

CAMPAIGN = ROOT / 'configs/experiments/idea_search_20261103/junior-campaign.v001.json'


def configuration(path=None):
    candidates = [Path(path)] if path else [
        Path(os.environ.get('QDUET_HANDOFF_CONFIG', '~/.config/qduet/junior.json')).expanduser(),
        Path('/etc/qore-upload/client.json'),
    ]
    for candidate in candidates:
        if candidate.is_file():
            value = json.loads(candidate.read_bytes())
            Exchange(value['url'], value['token'])  # Validate without logging credentials.
            return value
    if os.environ.get('QORE_EXCHANGE_URL') and os.environ.get('QORE_EXCHANGE_TOKEN'):
        return dict(url=os.environ['QORE_EXCHANGE_URL'], token=os.environ['QORE_EXCHANGE_TOKEN'])
    require(sys.stdin.isatty(), 'Private access configuration missing. Install the project-provided junior.json at ~/.config/qduet/junior.json; do not put credentials in Git or command arguments.')
    print('首次使用只需输入私有交付服务访问配置；以后自动复用。', flush=True)
    value = dict(url=input('交付服务地址: ').strip(), token=getpass.getpass('交付令牌（不显示）: '))
    Exchange(value['url'], value['token'])
    target = candidates[0]; target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream)
    return value


def package_for(campaign, population, release=None):
    value = release if release is not None else campaign['populations'][population]
    require(value.get('status') == 'ready', value.get('reason', 'Dataset release is not ready'))
    require(value['population'] == population and value['files'] and value['archive'], 'exact population release required')
    if population == 'formal':
        require(value.get('independent_confirmation') is True and value.get('exposed_population') is False
                and value.get('label_semantics') == 'reviewed_none_partial_direct'
                and value.get('qualification_evidence'), 'formal release lacks independent population/label qualification')
    require(len({r['path'] for r in value['files']}) == len(value['files']), 'unique release files required')
    return value


def validate_inputs(root, release):
    """Check roles and exact original Top50; no answer-string supervision construction."""
    counts = {}; questions = set(); components = set()
    for role, name in [('train', release['inputs']['train']), ('evaluation', release['inputs']['eval_inputs'])]:
        ids = set(); role_components = set(); role_questions = set(); count = 0
        with member(root, name).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                keys = {'id', 'component_id', 'question', 'paragraphs'} | ({'labels'} if role == 'train' else set())
                require(set(row) == keys, 'unexpected input fields; evaluation answers/labels forbidden')
                require(all(isinstance(row[k], str) and row[k] for k in ('id', 'component_id', 'question')),
                        'invalid input identity')
                require(row['id'] not in ids, 'duplicate question identity')
                candidates = row['paragraphs']
                require(len(candidates) == 50 and all(set(p) == {'id', 'text'} and
                        isinstance(p['id'], str) and p['id'] and isinstance(p['text'], str) and p['text'].strip()
                        for p in candidates) and len({p['id'] for p in candidates}) == 50, 'original50 input required')
                if role == 'train':
                    require(isinstance(row['labels'], dict) and set(row['labels']) <= {p['id'] for p in candidates}
                            and all(type(v) is int and v in (0, 1, 2) for v in row['labels'].values()), 'reviewed training tiers required')
                ids.add(row['id']); role_components.add(row['component_id'])
                role_questions.add(hashlib.sha256(row['question'].encode()).hexdigest()); count += 1
        require(count == release['counts'][role], 'release population count mismatch')
        require(not role_components & components and not role_questions & questions, 'component/question role leakage')
        require(not ids & counts.get('train_ids', set()), 'question ID role leakage')
        components.update(role_components); questions.update(role_questions)
        counts[role] = count
        if role == 'train':
            counts['train_ids'] = ids
    counts.pop('train_ids', None)
    # Evaluation labels are hash-bound here but their values are opened only after the joint prediction seal.
    require(member(root, release['inputs']['eval_labels']).is_file(), 'separate evaluation labels required')
    return counts


def prepare_environment(work, config, runner=subprocess.run):
    py = Path(config.get('python', work / 'environment/bin/python')).expanduser()
    if not py.is_file():
        require('python' not in config, 'Configured Python executable does not exist')
        runner([sys.executable, '-m', 'venv', str(work / 'environment')], check=True)
    require(py.is_file(), 'Python environment was not created')
    # Exact existing compatible installations can be reused without another pip pass.
    command = [str(py), '-B', '-c',
        "import torch,transformers,tokenizers,numpy,safetensors;assert torch.__version__.split('+')[0]=='2.5.1';assert torch.cuda.is_available();assert transformers.__version__=='4.44.2';assert tokenizers.__version__=='0.19.1';assert numpy.__version__=='1.26.4';assert safetensors.__version__=='0.7.0'"]
    probe = runner(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode:
        runner([str(py), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-qvalue.txt')], check=True)
        runner(command, check=True)
    return py


def matrix_argv(py, assets, output, release, campaign, resume):
    common = campaign['dual_ideas']; extras = campaign['dpr_only_ideas']
    require(len(common) == 8 and len(set(common + extras)) == len(common + extras), 'eight dual plus distinct qualified DPR additions')
    require(all(x in campaign['implemented_ideas'] for x in common + extras), 'unimplemented selected mechanism')
    command = [str(py), '-B', str(ROOT / 'scripts/run_top8.py'),
        '--dpr-model', str(member(assets, release['models']['dpr'])),
        '--bge-model', str(member(assets, release['models']['bge'])),
        '--train', str(member(assets, release['inputs']['train'])),
        '--eval-inputs', str(member(assets, release['inputs']['eval_inputs'])),
        '--eval-labels', str(member(assets, release['inputs']['eval_labels'])),
        '--output', str(output), '--seeds', *map(str, campaign['seeds']), '--ideas', *common]
    if extras:
        command += ['--dpr-only-ideas', *extras]
    if resume:
        command += ['--resume']
    return command


def declared_artifacts(output):
    """Inventory only the joint-sealed closure plus named compact run records."""
    output = Path(output); rows = {}
    def add(name, expected=None):
        path = member(output, name)
        require(path.is_file(), 'declared result missing: ' + name)
        row = dict(path=name, size=path.stat().st_size, sha256=digest(path))
        if expected:
            require(row['size'] == expected['size'] and row['sha256'] == expected['sha256'], 'sealed output changed')
        require(name not in rows or rows[name] == row, 'conflicting duplicate output')
        rows[name] = row
    seal_file = output / 'matrix/joint-prediction-seal.json'
    if seal_file.exists():
        seal = json.loads(seal_file.read_bytes()); add('matrix/joint-prediction-seal.json')
        for row in seal['files']:
            add('matrix/' + (row['path'] if 'path' in row else row['name']), row)
    for name in ['execution.json', 'dataset-release.json', 'matrix/run-binding.json', 'matrix/evaluation.json']:
        if (output / name).is_file():
            add(name)
    for path in sorted((output / 'matrix').glob('*/disposition.json')):
        add(path.relative_to(output).as_posix())
    require(rows, 'no declared results to upload')
    return list(rows.values())


def push_results(exchange, output, run_id, execution):
    prefix = 'five_ideas/top8-junior-results/' + str(relative(run_id))
    files = declared_artifacts(output)
    pending = dict(schema='qduet.junior-results.v1', run_id=run_id, execution=execution,
                   target=prefix, files=files, status='pending')
    atomic_json(output / 'pending-upload.json', pending)
    receipts = []
    for row in files:
        receipts.append(exchange.upload(member(output, row['path']), prefix + '/' + row['path']))
        atomic_json(output / 'upload-progress.json', dict(files=receipts, expected=len(files)))
    pending['status'] = 'uploaded'; pending['receipts'] = receipts
    atomic_json(output / 'result-manifest.json', pending)
    final = exchange.upload(output / 'result-manifest.json', prefix + '/result-manifest.json')
    atomic_json(output / 'upload-complete.json', dict(status='complete', manifest_receipt=final, files=len(receipts)))
    return final


def execute(campaign, config, population, work, *, resume=False, prepare_only=False, upload_only=False,
            runner=subprocess.run, environment=prepare_environment, exchange_factory=Exchange,
            asset_store=None):
    exchange = exchange_factory(config['url'], config['token'])
    release = package_for(campaign, population, config.get('populations', {}).get(population))
    # Fail before installing dependencies/downloading GB/GPU execution if formal data is unavailable.
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    lock = work / 'campaign.lock'
    if os.name == 'posix':
        import fcntl
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ValueError('another campaign process or inherited scientific child is active') from None
    else:
        fd = None
    if fd is None and lock.exists():
        old = json.loads(lock.read_bytes())
        require(old.get('host') == __import__('socket').gethostname(), 'workspace locked by another host')
        try:
            os.kill(old['pid'], 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise ValueError('another campaign process is still active')
    if fd is None:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.ftruncate(fd, 0)
    os.write(fd, json.dumps(dict(pid=os.getpid(), host=__import__('socket').gethostname())).encode())
    try:
        output = work / ('run-' + population)
        contract = dict(campaign_sha256=hashlib.sha256(json.dumps(campaign, sort_keys=True).encode()).hexdigest(),
                        release_sha256=hashlib.sha256(json.dumps(release, sort_keys=True).encode()).hexdigest())
        binding = work / 'binding.json'
        if binding.exists():
            require(json.loads(binding.read_bytes()) == contract, 'changed campaign/release requires a new workspace')
        else:
            atomic_json(binding, contract)
        if upload_only:
            execution = json.loads((output / 'execution.json').read_bytes())
            push_results(exchange, output, execution['run_id'], execution)
            return execution['returncode']
        existing = output / 'execution.json'
        if existing.exists() and json.loads(existing.read_bytes())['stage'] == 'terminal':
            execution = json.loads(existing.read_bytes())
            # Re-running the same command retries transport only, never a finished/failed scientific cell.
            push_results(exchange, output, execution['run_id'], execution)
            return execution['returncode']
        exchange.json('GET', '/healthz')
        if asset_store is None:
            free = shutil.disk_usage(work).free
            require(free >= release['archive']['size'] + sum(r['size'] for r in release['files']) + campaign['reserve_bytes'],
                    'insufficient server disk; expand before starting')
            archive = work / 'assets.tar'; assets = work / 'assets'
            exchange.download(release['archive'], archive)
            install_package(archive, assets, release['files'])
        else:
            assets = asset_store(release)
        counts = validate_inputs(assets, release)
        if prepare_only:
            print(json.dumps(dict(stage='assets_ready', population=population, counts=counts)))
            return 0
        py = environment(work, config, runner)
        if not existing.exists():
            # Training code owns its source-bound resume; explicit failed cells are not resubmitted.
            command = matrix_argv(py, assets, output, release, campaign, output.is_dir())
            # The child exclusively creates output; bootstrap records live state outside it.
            run_id = json.loads((work / 'active.json').read_bytes())['run_id'] if (work / 'active.json').exists() else (
                dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
            atomic_json(work / 'active.json', dict(run_id=run_id, stage='running', population=population))
            print(json.dumps(dict(stage='matrix_started', run_id=run_id, population=population)), flush=True)
            try:
                # Inherit the OS lock across the scientific child: killing the bootstrap
                # cannot permit a second launcher while the original child is alive.
                kwargs = dict(pass_fds=(fd,)) if os.name == 'posix' else {}
                result = runner(command, cwd=ROOT, check=False, **kwargs)
                code = result.returncode
            except OSError:
                code = 127
        else:
            execution = json.loads(existing.read_bytes()); run_id = execution['run_id']; code = execution['returncode']
        output.mkdir(parents=True, exist_ok=True)
        sealed = (output / 'matrix/joint-prediction-seal.json').is_file()
        evaluated = (output / 'matrix/evaluation.json').is_file()
        if code == 0 and not (sealed and evaluated):
            code = 3
        execution = dict(schema='qduet.junior-execution.v1', run_id=run_id, stage='terminal', returncode=code,
            population=population, exposed_population=release['exposed_population'],
            independent_confirmation=release.get('independent_confirmation', False), counts=counts,
            prediction_seal_present=sealed, evaluation_present=evaluated,
            dual_ideas=campaign['dual_ideas'], dpr_only_ideas=campaign['dpr_only_ideas'],
            scientific_success=code == 0, formal_success=code == 0 and population == 'formal')
        atomic_json(output / 'dataset-release.json', release)
        atomic_json(output / 'execution.json', execution)
        push_results(exchange, output, run_id, execution)
        print(json.dumps(dict(stage='results_uploaded', run_id=run_id, returncode=code, population=population)), flush=True)
        return code
    finally:
        os.close(fd)
        if os.name != 'posix':
            lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', choices=['formal', 'reproduction'], default='formal')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('qduet-junior-work'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--upload-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        campaign = json.loads(CAMPAIGN.read_bytes())
        # Expose the actual data gap before asking for credentials or touching the network.
        if not args.config and not os.environ.get('QDUET_HANDOFF_CONFIG') and not any(p.is_file() for p in [Path('~/.config/qduet/junior.json').expanduser(),Path('/etc/qore-upload/client.json')]):
            package_for(campaign, args.population)
        config = configuration(args.config)
        return execute(campaign, config, args.population, args.workspace,
                       prepare_only=args.prepare_only, upload_only=args.upload_only)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        # Library exceptions never include response bodies, tokens or raw question text.
        print('未完成: ' + type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
