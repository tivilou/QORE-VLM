#!/usr/bin/env python3
"""Acquire the frozen 8000/1000/1000 binary release and run the junior matrix."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_junior_experiments import configuration,prepare_environment,push_results
from scripts.run_two_datasets import AssetStore
from applications.rag.junior_transport_v001 import Exchange,atomic_json,digest,require
from applications.rag.triviaqa_smallset_v001 import materialize,evaluate,validate_release
from scripts.check_top8_delivery import check_delivery

RELEASE=ROOT/'configs/experiments/triviaqa_smallset_20261112/junior-release.v001.json'
CAMPAIGN=ROOT/'configs/experiments/idea_search_20261103/two-dataset-campaign.v001.json'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('~/qduet-triviaqa-smallset').expanduser())
    p.add_argument('--config',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--experiment-profile',choices=['primary54','full'],default='primary54',
                   help='primary54: quantum main effects only; full: historical393 controls.')
    p.add_argument('--display',choices=['auto','inplace','append','off'],default='auto',
                   help='Terminal dashboard; auto appends under redirection, off retains raw logs.')
    p.add_argument('--refresh',type=float,default=5.,help='Dashboard refresh seconds (>=0.5).')
    p.add_argument('--watch',action='store_true',help='Read-only display of workspace/progress/latest.json.')
    p.add_argument('--_progress-worker',action='store_true',help=argparse.SUPPRESS)
    from applications.rag.top8_cache_v002 import cache_capacity
    p.add_argument('--sequence-cache-gib',type=cache_capacity,default='auto',
                   help='auto: filesystem-aware prefix cache; or explicit nonnegative GiB per model.')
    a=p.parse_args(argv)
    if a.refresh<.5 or not __import__('math').isfinite(a.refresh):p.error('--refresh must be finite and >=0.5')
    from scripts.smallset_progress import supervise,watch
    if a.watch:return watch(a.workspace,a.display if a.display!='off' else 'append',a.refresh)
    if not a.plan_only and not a._progress_worker and a.display!='off':
        command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),
                 *(sys.argv[1:] if argv is None else argv),'--_progress-worker']
        return supervise(command,a.workspace,a.display,a.refresh,cwd=os.getcwd())
    from scripts.smallset_events import emit
    emit('phase',phase='核对交付配置')
    check_delivery()
    release=json.loads(RELEASE.read_bytes());validate_release(release,ready=not a.plan_only)
    campaign=json.loads(CAMPAIGN.read_bytes())
    base=json.loads((ROOT/campaign['base_campaign']).read_bytes())
    command=[sys.executable,'-B',str(ROOT/'scripts/run_top8.py'),
             '--experiment-profile',a.experiment_profile,
             '--dpr-model','models/dpr','--bge-model','models/bge','--output','matrix',
             '--label-classes','2','--ideas',*base['dual_ideas'],
             '--fusion-ideas',*campaign['fusion_ideas'],'--seeds',*map(str,base['seeds'])]
    if a.plan_only:
        r=subprocess.run(command+['--plan-only'],check=True,capture_output=True,text=True)
        print(json.dumps(dict(dataset='triviaqa',roles=release['roles'],label_semantics='binary_useful',
                              matrix=json.loads(r.stdout),test_used_for_selection=False,
                              owner_scientific_run=False,release_status=release['status'],
                              data_archive_bytes=release.get('archive',{}).get('size'),
                              sequence_cache_policy=a.sequence_cache_gib),indent=2))
        return 0
    config=configuration(a.config);exchange=Exchange(config['url'],config['token'])
    work=a.workspace.resolve();work.mkdir(parents=True,exist_ok=True)
    contract=dict(release=digest(RELEASE),campaign=digest(CAMPAIGN),base_campaign=digest(ROOT/campaign['base_campaign']))
    if (work/'binding.json').exists():require(json.loads((work/'binding.json').read_bytes())==contract,'new release needs a new workspace')
    else:atomic_json(work/'binding.json',contract)
    # Download data independently of old B1 models; models may be reused by exact
    # hash from the existing private model package. No owner paths are required.
    store=AssetStore(work/'assets',exchange,2*1024**3)
    emit('phase',phase='下载／校验数据包')
    raw=store(release)
    data=work/'data'
    emit('phase',phase='展开并核验共同输入')
    materialize(raw,release,data)
    emit('phase',phase='获取／核验DPR与BGE模型')
    models=store(base['populations']['reproduction'])
    if a.prepare_only:
        print(json.dumps(dict(stage='smallset_ready',roles=release['roles'],labels=500000,
                              label_semantics='binary_useful',scientific_execution=False)))
        return 0
    # Compact FP32 question shards and token recipes replace full sequence dumps.
    # Each backbone gets a bounded, disposable working set; old partial v1 files
    # are preserved and are NOT included in the new-cache estimate below.
    require(shutil.disk_usage(work).free>=40*1024**3,
            'Less than 40GiB free. Expand the experiment disk; --prepare-only remains available.')
    print(json.dumps(dict(stage='storage_plan',storage_layout='question-shards-fp32-v2',
                          sequence_cache_GiB_per_model=a.sequence_cache_gib,
                          combined_sequence_cache_limit_GiB=None if a.sequence_cache_gib=='auto' else 2*a.sequence_cache_gib,
                          compact_features_max_GiB=43.4,tokens_and_metadata_estimate_GiB=9,
                          derived_feature_columns_max_GiB=40.1,
                          measured_cache_bytes=None,minimum_free_GiB=40,
                          note='Auto caps resolved on actual cache filesystems after extraction, with 40GiB free plus 48GiB growth allowance. Caps are not preallocated. Legacy partial files are preserved.')),flush=True)
    emit('phase',phase='准备运行环境')
    py=prepare_environment(work,config)
    emit('phase',phase='软件检查（非正式实验）')
    subprocess.run([str(py),'-B','-m','unittest',
                    'applications.rag.tests.test_top8_binary_numerics',
                    'applications.rag.tests.test_top8_cache_numerics',
                    'applications.rag.tests.test_top8_feature_io',
                    'applications.rag.tests.test_top8_runtime_io','-q'],cwd=ROOT,check=True)
    output=work/'run-smallset';output.mkdir(exist_ok=True)
    command=[str(py),'-B',str(ROOT/'scripts/run_top8.py'),
             '--experiment-profile',a.experiment_profile,
             '--dpr-model',str(models/base['populations']['reproduction']['models']['dpr']),
             '--bge-model',str(models/base['populations']['reproduction']['models']['bge']),
             '--train',str(data/'train.jsonl'),'--eval-inputs',str(data/'evaluation.inputs.jsonl'),
             '--output',str(output),'--label-classes','2','--minimum-free-gib','40',
             '--sequence-cache-gib',str(a.sequence_cache_gib),'--ideas',*base['dual_ideas'],
             '--fusion-ideas',*campaign['fusion_ideas'],'--seeds',*map(str,base['seeds'])]
    if a.resume:command.append('--resume')
    else:
        require(not any(output.iterdir()),'existing run requires --resume')
        output.rmdir() # suite owns exclusive creation
    result=subprocess.run(command)
    evaluation=None
    matrix=output/('matrix-primary54' if a.experiment_profile=='primary54' else 'matrix')
    if (matrix/'joint-prediction-seal.json').exists():
        emit('phase',phase='联合预测封存后的验证／测试评估')
        evaluation=evaluate(matrix,data)
    run_id='triviaqa-smallset-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    execution=dict(schema='qduet.smallset-execution.v1',returncode=result.returncode,roles=release['roles'],
                   experiment_profile=a.experiment_profile,quantum_attribution_certified=False,
                   all_cells_complete=evaluation is not None and not evaluation['failures'],
                   label_semantics='binary_useful',formal_full_dataset=False,test_used_for_selection=False)
    require(output.is_dir(),'run failed before producing output')
    atomic_json(output/'execution.json',execution);atomic_json(output/'dataset-release.json',release)
    # The new binary report is explicitly included by the common uploader.
    emit('phase',phase='归档并上传结果')
    receipt=push_results(exchange,output,run_id,execution)
    emit('outcome',result='结果上传完成；all_cells_complete='+str(execution['all_cells_complete']))
    print(json.dumps(dict(stage='results_uploaded',returncode=result.returncode,receipt=receipt)))
    return result.returncode if result.returncode else (0 if execution['all_cells_complete'] else 2)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,KeyError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
