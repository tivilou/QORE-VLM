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
    p.add_argument('--sequence-cache-gib',type=int,default=32,
                   help='Bounded FP32 sequence cache per model; smaller uses more recomputation.')
    a=p.parse_args(argv)
    if a.sequence_cache_gib < 0:p.error('--sequence-cache-gib must be nonnegative')
    check_delivery()
    release=json.loads(RELEASE.read_bytes());validate_release(release,ready=not a.plan_only)
    campaign=json.loads(CAMPAIGN.read_bytes())
    base=json.loads((ROOT/campaign['base_campaign']).read_bytes())
    command=[sys.executable,'-B',str(ROOT/'scripts/run_top8.py'),
             '--dpr-model','models/dpr','--bge-model','models/bge','--output','matrix',
             '--label-classes','2','--ideas',*base['dual_ideas'],
             '--fusion-ideas',*campaign['fusion_ideas'],'--seeds',*map(str,base['seeds'])]
    if a.plan_only:
        r=subprocess.run(command+['--plan-only'],check=True,capture_output=True,text=True)
        print(json.dumps(dict(dataset='triviaqa',roles=release['roles'],label_semantics='binary_useful',
                              matrix=json.loads(r.stdout),test_used_for_selection=False,
                              owner_scientific_run=False,release_status=release['status'],
                              data_archive_bytes=release.get('archive',{}).get('size')),indent=2))
        return 0
    config=configuration(a.config);exchange=Exchange(config['url'],config['token'])
    work=a.workspace.resolve();work.mkdir(parents=True,exist_ok=True)
    contract=dict(release=digest(RELEASE),campaign=digest(CAMPAIGN),base_campaign=digest(ROOT/campaign['base_campaign']))
    if (work/'binding.json').exists():require(json.loads((work/'binding.json').read_bytes())==contract,'new release needs a new workspace')
    else:atomic_json(work/'binding.json',contract)
    # Download data independently of old B1 models; models may be reused by exact
    # hash from the existing private model package. No owner paths are required.
    store=AssetStore(work/'assets',exchange,2*1024**3)
    raw=store(release)
    data=work/'data'
    materialize(raw,release,data)
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
                          combined_sequence_cache_limit_GiB=2*a.sequence_cache_gib,
                          compact_features_max_GiB=43.4,tokens_and_metadata_estimate_GiB=9,
                          measured_cache_bytes=None,minimum_free_GiB=40,
                          note='Bounded FP32 prefixes regenerate on misses. Legacy partial files are preserved, additional to this estimate. Training outputs require extra space.')),flush=True)
    py=prepare_environment(work,config)
    subprocess.run([str(py),'-B','-m','unittest',
                    'applications.rag.tests.test_top8_binary_numerics',
                    'applications.rag.tests.test_top8_cache_numerics','-q'],cwd=ROOT,check=True)
    output=work/'run-smallset';output.mkdir(exist_ok=True)
    command=[str(py),'-B',str(ROOT/'scripts/run_top8.py'),
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
    if (output/'matrix/joint-prediction-seal.json').exists():evaluation=evaluate(output/'matrix',data)
    run_id='triviaqa-smallset-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    execution=dict(schema='qduet.smallset-execution.v1',returncode=result.returncode,roles=release['roles'],
                   all_cells_complete=evaluation is not None and not evaluation['failures'],
                   label_semantics='binary_useful',formal_full_dataset=False,test_used_for_selection=False)
    require(output.is_dir(),'run failed before producing output')
    atomic_json(output/'execution.json',execution);atomic_json(output/'dataset-release.json',release)
    # The new binary report is explicitly included by the common uploader.
    receipt=push_results(exchange,output,run_id,execution)
    print(json.dumps(dict(stage='results_uploaded',returncode=result.returncode,receipt=receipt)))
    return result.returncode if result.returncode else (0 if execution['all_cells_complete'] else 2)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,KeyError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
