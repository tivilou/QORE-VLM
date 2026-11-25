#!/usr/bin/env python3
"""One command: pinned assets, coupled DPR/BGE study, sealed evaluation, upload."""
import argparse,ast,contextlib,datetime,hashlib,importlib.metadata,json,os,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from applications.rag.junior_transport_v001 import require,atomic_json,digest,Exchange
from applications.rag.local_born_io_v001 import immutable_json,verify_seal,verify_campaign

PROTOCOL=ROOT/'configs/experiments/local_born_smallset_20261125/protocol.v001.json'
RELEASE=ROOT/'configs/experiments/triviaqa_smallset_20261112/junior-release.v001.json'
CAMPAIGN=ROOT/'configs/experiments/idea_search_20261103/junior-campaign.v001.json'


def source_snapshot():
    # Bind import-time local dependencies recursively. Explicit lazy execution
    # roots cover the task factory, runtime and delivery branches actually used.
    pending=[Path(__file__).relative_to(ROOT).as_posix(),
        'applications/rag/local_born_smallset_v001.py','applications/rag/local_born_cache_v001.py',
        'scripts/run_junior_experiments.py','scripts/run_two_datasets.py',
        'scripts/smallset_progress.py','scripts/smallset_resources.py',
        'applications/rag/top8_cache_v002.py',
        'applications/rag/triviaqa_smallset_v001.py']
    values={}
    while pending:
        name=pending.pop()
        if name in values:continue
        p=ROOT/name;require(p.is_file(),'missing shipped dependency: '+name);values[name]=digest(p)
        if p.suffix!='.py':continue
        for node in ast.parse(p.read_bytes()).body:
            imports=([x.name for x in node.names] if isinstance(node,ast.Import) else
                [node.module] if isinstance(node,ast.ImportFrom) and node.module and not node.level else [])
            for module in imports:
                if module.startswith(('applications.','scripts.')):
                    q=module.replace('.','/')+'.py'
                    if (ROOT/q).is_file():pending.append(q)
    for p in [PROTOCOL,RELEASE,CAMPAIGN,ROOT/'scripts/run_local_born_smallset.sh',
              ROOT/'configs/experiments/reader_joint_witness_b1_model_assets.json',
              ROOT/'configs/experiments/local_born_smallset_20261125/requirements.txt']:
        values[p.relative_to(ROOT).as_posix()]=digest(p)
    revision='unavailable; exact file hashes retained'
    try:
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
        if git.returncode==0:revision=git.stdout.strip()
    except OSError:pass
    return dict(files=dict(sorted(values.items())),git_revision=revision)


def runtime_matches(python):
    code="import importlib.metadata as m,json;print(json.dumps({n:m.version(n).split('+')[0] for n in ['torch','transformers','tokenizers']}))"
    r=subprocess.run([str(python),'-B','-c',code],capture_output=True,text=True)
    if r.returncode:return False
    try:return json.loads(r.stdout)==dict(torch='2.13.0',transformers='4.44.2',tokenizers='0.19.1')
    except ValueError:return False


def environment(work):
    require(sys.version_info>=(3,11),'Use Python3.11+ (QDUET_BOOTSTRAP_PYTHON may select the installed interpreter)')
    if runtime_matches(sys.executable):return Path(sys.executable)
    py=work/'runtime/bin/python'
    if not py.exists():subprocess.run([sys.executable,'-m','venv',str(work/'runtime')],check=True)
    if not runtime_matches(py):
        subprocess.run([str(py),'-m','pip','install','-r',str(PROTOCOL.parent/'requirements.txt')],check=True)
    require(runtime_matches(py),'pinned runtime installation incomplete');return py


def sealed_campaign(work,protocol):
    matrix=work/'run/matrix';binding=json.loads((matrix/'run-binding.json').read_bytes())
    require(binding['protocol']==protocol,'sealed campaign protocol differs')
    for name,sha in binding['input_files'].items():require(digest(work/'data'/name)==sha,'sealed campaign input identity differs')
    with (work/'data/evaluation.inputs.jsonl').open(encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream]
    return verify_campaign(matrix,binding,protocol,rows)


@contextlib.contextmanager
def campaign_lock(work):
    require(os.name=='posix','Run this launcher on the Linux experiment server')
    import fcntl
    fd=os.open(work/'campaign.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another campaign worker owns this workspace') from None
        yield fd
    finally:os.close(fd)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('~/qduet-local-born-smallset').expanduser())
    p.add_argument('--config',type=Path)
    for name in ['resume','plan-only','prepare-only','watch','upload-only']:
        p.add_argument('--'+name,action='store_true')
    p.add_argument('--display',choices=['auto','inplace','append','off'],default='auto')
    p.add_argument('--refresh',type=float,default=5.)
    p.add_argument('--_progress-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--_science',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv);work=a.workspace.expanduser().resolve()
    require(__import__('math').isfinite(a.refresh) and a.refresh>=.5,'refresh >=0.5 seconds')
    from scripts.smallset_progress import supervise,watch
    if a.watch:return watch(work,a.display if a.display!='off' else 'append',a.refresh)
    protocol=json.loads(PROTOCOL.read_bytes());source=source_snapshot()
    if a.plan_only:
        print(json.dumps(dict(protocol=protocol,source_files=len(source['files']),
            release_status=json.loads(RELEASE.read_bytes())['status'],scientific_execution=False),ensure_ascii=False,indent=2));return 0
    if a._science:
        from applications.rag.local_born_smallset_v001 import execute
        prepared=json.loads((work/'prepared.json').read_bytes())
        result=execute(work/'data',prepared['models'],work/'run/matrix',work/'native-cache',protocol,source)
        print(json.dumps(result));return 0
    if not a._progress_worker and a.display!='off':
        return supervise([sys.executable,'-u','-B',str(Path(__file__).resolve()),
            *(sys.argv[1:] if argv is None else argv),'--_progress-worker'],work,a.display,a.refresh,cwd=ROOT)
    work.mkdir(parents=True,exist_ok=True)
    from scripts.run_junior_experiments import configuration,push_results
    from scripts.run_two_datasets import AssetStore
    from applications.rag.triviaqa_smallset_v001 import materialize,evaluate,validate_release
    from scripts.smallset_events import emit
    with campaign_lock(work) as fd:
        config=configuration(a.config);exchange=Exchange(config['url'],config['token'])
        output=work/'run';matrix=output/'matrix'
        if a.upload_only:
            sealed_campaign(work,protocol);execution=json.loads((output/'execution.json').read_bytes())
            push_results(exchange,output,execution['run_id'],execution);return 0
        release=json.loads(RELEASE.read_bytes());validate_release(release,ready=True)
        immutable_json(work,'source-binding.json',source)
        if matrix.exists() and any(matrix.iterdir()):require(a.resume,'Existing study requires --resume (completed jobs are retained)')
        emit('phase',phase='下载／校验固定smallset与两个原模型')
        store=AssetStore(work/'assets',exchange,2*1024**3)
        raw=store(release);materialize(raw,release,work/'data')
        package=json.loads(CAMPAIGN.read_bytes())['populations']['reproduction'];models=store(package)
        prepared=dict(models={k:str(models/package['models'][k]) for k in ['dpr','bge']})
        immutable_json(work,'prepared.json',prepared)
        if a.prepare_only:print(json.dumps(dict(stage='assets_ready',scientific_execution=False,roles=release['roles'])));return 0
        # 10000 questions, both FFNs/residuals/features ~22 GiB. No sequence dump.
        cached=sum(f.stat().st_size for f in (work/'native-cache').rglob('*') if f.is_file())
        complete_cache=all((work/'native-cache'/k/'cache.json').exists() for k in ['dpr','bge'])
        required_free=8*1024**3 if complete_cache else max(8*1024**3,40*1024**3-cached)
        require((matrix/'joint-prediction-seal.json').exists() or shutil.disk_usage(work).free>=required_free,
            'Need 40 GiB free for first extraction; expand server storage. Existing caches/checkpoints are preserved.')
        emit('phase',phase='准备与已验证toy相同的运行环境')
        py=environment(work);output.mkdir(exist_ok=True)
        identity=output/'identity.json'
        if not identity.exists():immutable_json(output,identity.name,dict(run_id='local-born-smallset-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
        run_id=json.loads(identity.read_bytes())['run_id']
        env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        args=[str(py),'-u','-B',str(Path(__file__).resolve()),'--workspace',str(work),'--_science','--display','off']
        started=datetime.datetime.now(datetime.timezone.utc).isoformat()
        result=subprocess.run(args,cwd=ROOT,env=env,pass_fds=(fd,))
        evaluation=None
        if result.returncode==0:
            sealed_campaign(work,protocol);emit('phase',phase='全部56组预测封存后评估valid／test')
            evaluation=evaluate(matrix,work/'data')
        execution=dict(schema='qduet.local-born-smallset-execution.v1',run_id=run_id,returncode=result.returncode,
            all_cells_complete=evaluation is not None,roles=release['roles'],arms=protocol['arms'],seeds=protocol['seeds'],
            label_semantics='binary_useful',experiment_profile='local-born',test_used_for_selection=False,
            toy_claim='scoped adaptive L3; smallset outcome pending actual run',cross_machine_resume='Final sealed checkpoint + pinned frozen metadata; active milestones local only')
        import uuid
        immutable_json(output/'attempts',uuid.uuid4().hex+'.json',dict(started=started,finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=args,**execution))
        atomic_json(output/'execution.json',execution);atomic_json(output/'dataset-release.json',release)
        emit('phase',phase='上传声明结果与最终检查点')
        push_results(exchange,output,run_id,execution)
        emit('outcome',result='已完成并上传' if evaluation is not None else '训练未完成；保留完整日志与检查点，使用 --resume')
        return result.returncode


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,KeyError,subprocess.CalledProcessError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
