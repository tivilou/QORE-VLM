"""One-command dual-backbone suite with immutable, source-bound cell recovery."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
from applications.rag.qvalue_portable_v001 import need, digest, json_file, tensor_file, checked_member


def verify_files(root, records):
    root = Path(root)
    for record in records:
        path = root/record.get('path', record['name'])
        need(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
             and path.is_file() and path.stat().st_size == record['size']
             and digest(path) == record['sha256'], 'immutable completed cell file binding')


def recovery_probe(cache, output, checkpoint=lambda:None):
    """Two fixed training questions; compare uninterrupted and reconstructed runs.

    This exercises actual optimizer/RNG restoration in the admitted worker. It
    does not pretend that reconstruction in one worker is an OS process restart.
    No development labels are opened, and no outcomes select a probe or setting.
    """
    import torch
    from applications.rag.top8_portable_v001 import load_cache, train_cell
    from applications.rag.top8_recovery_v001 import Journal, source_binding
    manifest, features=load_cache(cache)
    selected=[r for r in manifest['rows'] if r['role']=='training'][:2]+[
        r for r in manifest['rows'] if r['role']=='development'][:2]
    manifest=dict(manifest,rows=selected)
    features={r['id']:features[r['id']] for r in selected}
    output=Path(output);output.mkdir(exist_ok=False)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    verified=[]
    for idea,arm in [('FEATURE_GATE','QUANTUM'),('VALUE_GEOMETRY','DENSE')]:
        root=output/idea;root.mkdir()
        binding=dict(sources=source_binding(),cache_sha256=digest(cache),idea=idea,arm=arm,seed=20261103)
        baseline=Journal(root/'baseline-journal',binding)
        resumed=Journal(root/'resumed-journal',binding)
        def run(name,limit,journal,initial_seed):
            torch.manual_seed(initial_seed);torch.cuda.manual_seed_all(initial_seed)
            destination=root/name;destination.mkdir()
            result=train_cell(idea,arm,20261103,manifest,features,Path(cache).parent,destination,
                              2,limit,checkpoint,journal=journal)
            return destination,result
        a,ra=run('uninterrupted',2,baseline,20261103)
        run('interrupted',1,resumed,20261103)
        b,rb=run('restored',2,resumed,17)
        for field in ('parameters','optimizer','generator','cpu_rng','cuda_rng','records'):
            need(equal(baseline.load()[field],resumed.load()[field]),'actual optimizer/RNG/order recovery '+field)
        ap=next(r for r in ra['files'] if r['name'].startswith('predictions-'))
        bp=next(r for r in rb['files'] if r['name'].startswith('predictions-'))
        need(json.loads(checked_member(a,ap).read_bytes())==json.loads(checked_member(b,bp).read_bytes()),
             'all50 fixed-question prediction recovery')
        verified.append(dict(idea=idea,arm=arm,steps=2,restored_from_step=1,parameter_optimizer_rng_order_bit_equal=True))
    result=dict(schema='top8.recovery-probe.v1',cases=verified,
        actual_gpu_restore=True,process_restart_test=False,development_labels_opened=False,
        limitation='Fresh model/optimizer reconstruction in same worker; OS process-restart not certified')
    json_file(output,'recovery-probe.json',result)
    return result


def run_recoverable(caches, output, seeds, ideas, *, canary=False, checkpoint=lambda:None,
                    backbone_ideas=None, fusion_ideas=None):
    import torch
    from applications.rag.top8_portable_v001 import load_cache, train_cell
    from applications.rag.top8_methods_v001 import ARMS, mixed_banks
    from applications.rag.top8_recovery_v001 import Journal, source_binding, new_generation
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = dict(schema='top8.recoverable-run.v1', sources=source_binding(),
        caches=[dict(path=str(Path(p).resolve()), sha256=digest(p)) for p in caches],
        seeds=seeds, ideas=ideas, epochs=1 if canary else 4, canary=canary,
        question_batch=8, lr=.001, weight_decay=0, gradient_clip=1.)
    if backbone_ideas is not None:
        binding['backbone_ideas'] = backbone_ideas
    if fusion_ideas:
        binding['fusion_ideas'] = fusion_ideas
    contract = output/'run-binding.json'
    if contract.exists():
        need(json.loads(contract.read_bytes()) == binding, 'immutable matrix resume contract')
    else:
        json_file(output, contract.name, binding)
    seal_path = output/'joint-prediction-seal.json'
    if seal_path.exists():
        seal = json.loads(seal_path.read_bytes())
        verify_files(output, seal['files'])
        return seal
    need(not (output/'evaluation.json').exists(), 'never continue training after evaluation')
    results = {}; failures = []; files = []
    for path in caches:
        manifest, features = load_cache(path)
        root = Path(path).parent; kind = manifest['backbone']
        scheduled = ideas if backbone_ideas is None else backbone_ideas.get(kind, [])
        need(scheduled and set(scheduled) <= set(ideas), 'explicit nonempty backbone schedule')
        training = [r for r in manifest['rows'] if r['role']=='training']
        development = [r for r in manifest['rows'] if r['role']=='development']
        need(training and development, 'both complete roles')
        if canary:
            selected = training[:2] + development[:2]
            manifest = dict(manifest, rows=selected)
            features = {r['id']:features[r['id']] for r in selected}
        elif 'MIXED_DEPTH' in scheduled:
            # Deterministic training-only refit; no development targets are present.
            # Bank files from interrupted generations remain intact.
            bank = mixed_banks(features, manifest['rows'], checkpoint)
            generation = len(list(output.glob('banks-'+kind+'-*.bin')))
            files.append(tensor_file(output, f'banks-{kind}-{generation:04d}.bin', bank))
            del bank; torch.cuda.empty_cache()
        native = dict(rows=[dict(id=r['id'], candidate_ids=r['candidate_ids'],
            scores=r['native_scores'], ranking=sorted(range(50),key=lambda i:(-r['native_scores'][i],i)))
            for r in manifest['rows'] if r['role']=='development'])
        native_name = 'predictions-'+kind+'-NATIVE.json'
        if (output/native_name).exists():
            need(json.loads((output/native_name).read_bytes())==native, 'same native predictions on resume')
            files.append(dict(name=native_name,size=(output/native_name).stat().st_size,
                              sha256=digest(output/native_name)))
        else:
            files.append(json_file(output,native_name,native))
        order = [x for x in scheduled if x!='FIDELITY'] + (['FIDELITY'] if 'FIDELITY' in scheduled else [])
        for idea in order:
            if canary and idea=='MIXED_DEPTH':
                failures.append(dict(backbone=kind,idea=idea,status='DEFERRED_BANK_FIT'))
                continue
            for seed in seeds:
                for arm in ARMS[idea]:
                    checkpoint()
                    key = f'{kind}-{idea}-{arm}-s{seed}'
                    cell = output/key; cell.mkdir(exist_ok=True)
                    disposition = cell/'disposition.json'
                    if disposition.exists():
                        value = json.loads(disposition.read_bytes())
                        need(value['binding']==binding, 'completed cell source binding')
                        if value['status']=='complete':
                            result = value['result']; verify_files(output, value['files'])
                            results[key]=result; files.extend(value['files'])
                        else:
                            failures.append(value['failure'])
                        continue
                    journal = Journal(cell/'recovery',dict(matrix=binding,idea=idea,arm=arm,seed=seed,backbone=kind))
                    generation = new_generation(cell)
                    try:
                        result = train_cell(idea,arm,seed,manifest,features,root,generation,
                            1 if canary else 4,1 if canary else None,checkpoint,journal=journal)
                        owned = [dict(r,path=(generation.relative_to(output)/r['name']).as_posix())
                                 for r in result['files']]
                        json_file(cell,'disposition.json',dict(status='complete',binding=binding,
                            result=result,files=owned))
                        results[key]=result; files.extend(owned)
                    except Exception as e:
                        failure = dict(backbone=kind,idea=idea,arm=arm,seed=seed,
                            error_type=type(e).__name__,message=str(e)[:1000])
                        json_file(cell,'disposition.json',dict(status='failed',binding=binding,failure=failure))
                        failures.append(failure)
                        gc.collect(); torch.cuda.empty_cache()
        del features; gc.collect(); torch.cuda.empty_cache()
    if fusion_ideas:
        from applications.rag.qduet_fusion_v001 import run_fusions
        extra, failed, owned = run_fusions(caches, output, seeds, fusion_ideas, binding,
                                           canary=canary, checkpoint=checkpoint)
        results.update(extra); failures.extend(failed); files.extend(owned)
    seal = dict(schema='top8.joint-predictions.v1',seeds=seeds,ideas=ideas,canary=canary,
        requested_backbones=len(caches),results=results,failures=failures,files=files,
        all_predictions_before_evaluation_labels=True,run_binding_sha256=digest(contract),
        recovery='same immutable source/input/config; optimizer/RNG/order restored; explicit failed cells not retried')
    if fusion_ideas:
        seal['fusion_ideas'] = fusion_ideas
        seal['fusion_inputs'] = ['dpr', 'bge']
    if backbone_ideas is not None:
        seal['backbone_ideas'] = backbone_ideas
        seal['not_scheduled'] = [dict(backbone=kind, idea=idea, status='NOT_SCHEDULED')
            for kind in ('dpr', 'bge') for idea in ideas
            if idea not in backbone_ideas.get(kind, [])]
    json_file(output,seal_path.name,seal)
    return seal


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dpr-model',type=Path);parser.add_argument('--bge-model',type=Path)
    parser.add_argument('--train',type=Path);parser.add_argument('--eval-inputs',type=Path)
    parser.add_argument('--eval-labels',type=Path);parser.add_argument('--cache',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261102,20261103,20261104])
    parser.add_argument('--ideas',nargs='+');parser.add_argument('--canary',action='store_true')
    parser.add_argument('--dpr-only-ideas',nargs='+',default=[],
                        help='Run these registered ideas only on DPR; omitted BGE is N/A, not zero.')
    parser.add_argument('--bge-only-ideas',nargs='+',default=[])
    parser.add_argument('--fusion-ideas',nargs='+',default=[],
                        help='Two-scorer CROSS_SCORE/CLASS_FUSION; both native models required.')
    parser.add_argument('--plan-only',action='store_true',
                        help='Print the exact backbone/idea/arm schedule without loading models or writing output.')
    parser.add_argument('--extract-only',action='store_true')
    args=parser.parse_args(argv)
    from applications.rag.top8_matrix_plan_v001 import matrix_plan
    # Read the static registry without importing its scientific dependencies.
    import ast
    module = ast.parse(Path(__file__).with_name('top8_methods_v001.py').read_text(encoding='utf-8'))
    registry = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('IDEAS', 'ARMS')}
    available = [kind for kind, model in [('dpr',args.dpr_model),('bge',args.bge_model)] if model]
    available += [json.loads(path.read_bytes())['backbone'] for path in args.cache]
    plan = matrix_plan(available,registry['IDEAS'],args.ideas,args.dpr_only_ideas,args.bge_only_ideas)
    from applications.rag.qduet_fusion_v001 import fusion_plan, FUSION_ARMS
    plan['fusion_ideas'] = fusion_plan(available, args.fusion_ideas)
    plan['seeds'] = args.seeds
    plan['learned_cells'] = sum(len(registry['ARMS'][idea]) * len(args.seeds)
        for values in plan['backbone_ideas'].values() for idea in values)
    plan['fusion_learned_cells'] = sum(len(FUSION_ARMS[i])*len(args.seeds) for i in args.fusion_ideas)
    plan['learned_cells'] += plan['fusion_learned_cells']
    need(args.seeds and len(set(args.seeds))==len(args.seeds),'unique predeclared seeds')
    if args.plan_only:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return 0
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch,transformers
    need(torch.cuda.is_available() and transformers.__version__=='4.44.2','CUDA / Transformers4.44.2')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    from applications.rag.top8_backbone_v001 import extract
    from applications.rag.top8_methods_v001 import IDEAS
    from applications.rag.top8_portable_v001 import evaluate
    ideas=plan['ideas']
    need(ideas and len(set(ideas))==len(ideas) and set(ideas)<=set(IDEAS),'selected unique ideas')
    need(args.seeds and len(set(args.seeds))==len(args.seeds),'unique predeclared seeds')
    if args.resume:need(args.output.is_dir(),'existing output required for explicit resume')
    else:args.output.mkdir(parents=True,exist_ok=False)
    caches=list(args.cache)
    for kind,model in [('dpr',args.dpr_model),('bge',args.bge_model)]:
        if not model:continue
        cache=args.output/('cache-'+kind)/'cache.json'
        if args.resume and cache.exists():
            m=json.loads(cache.read_bytes())
            need(args.train and args.eval_inputs and m['train_sha256']==digest(args.train)
                 and m['eval_input_sha256']==digest(args.eval_inputs),'same raw input files on resume')
            for record in m['assets']:
                file=model/record['name']
                need(file.is_file() and file.stat().st_size==record['size'] and digest(file)==record['sha256'],
                     'same pinned model bytes on resume')
            caches.append(cache)
        else:
            need(args.train and args.eval_inputs,'explicit train / label-free evaluation inputs')
            caches.append(extract(kind,model,args.train,args.eval_inputs,args.output/('cache-'+kind)))
    need(caches and len({json.loads(Path(p).read_bytes())['backbone'] for p in caches})==len(caches),
         'one cache for each requested backbone')
    if args.extract_only:return 0
    seal=run_recoverable(caches,args.output/'matrix',args.seeds,ideas,canary=args.canary,
                         backbone_ideas=plan['backbone_ideas'],fusion_ideas=args.fusion_ideas)
    if args.eval_labels:
        evaluation=args.output/'matrix/evaluation.json'
        if evaluation.exists():
            r=json.loads(evaluation.read_bytes())
            need(r['labels_sha256']==digest(args.eval_labels) and r['joint_seal_sha256']==
                 digest(args.output/'matrix/joint-prediction-seal.json'),'same sealed evaluation')
        else:evaluate(args.output/'matrix',args.eval_labels)
    failures=[x for x in seal['failures'] if x.get('status')!='DEFERRED_BANK_FIT']
    print(json.dumps(dict(stage='matrix_finished',completed_cells=len(seal['results']),
        failed_cells=len(failures),evaluation_written=(args.output/'matrix/evaluation.json').exists(),
        predictions_sealed=True)),flush=True)
    return 2 if failures else 0


if __name__=='__main__':raise SystemExit(main())
