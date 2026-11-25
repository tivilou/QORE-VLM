"""Prospective coupled-backbone toy study, reusing journals and metric adapter."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from applications.rag.toy100_shared_readout_v001 import ARMS,PairedReadout,aligned_coordinates,transform_readouts
from applications.rag.toy100_qk_task_v001 import rows,train_targets,order
from applications.rag.toy100_qk_study_v001 import SEEDS,event,evaluate,loss
from applications.rag.top8_recovery_v001 import Journal
from applications.rag.qvalue_portable_v001 import need,digest,checked_member,json_file,tensor_file


def tensors(packed):
    features={k:torch.cat((p['features']['train'],p['features']['holdout'])).cuda() for k,p in packed.items()}
    native={k:torch.cat((p['native']['train'],p['native']['holdout'])).cuda() for k,p in packed.items()}
    original={k:F.linear(features[k],p['original_head']['weight'].cuda(),p['original_head']['bias'].cuda()).squeeze(-1)
              for k,p in packed.items()}
    return features,native,original


def model_batch_loss(kind,indices,scores,targets,loss_fn=loss,batch_loss=None):
    """Optional task objective with explicit row identity; legacy path unchanged."""
    if batch_loss is not None:return batch_loss(kind,indices,scores,targets)
    return torch.stack([loss_fn(scores[j],targets[i]) for j,i in enumerate(indices)]).mean()


def registered_protocol(protocol,plugin_setup):
    """Scientific metadata only: do not inherit declarations of another factory."""
    result=dict(protocol)
    result['paired_mapping']='Registered plugin input mapping is stored in train-input-only-alignment.bin; see candidate-specific declaration. No generic PCA assumption.'
    for field in ('circuit','shared','controls'):result.pop(field,None)
    if 'batch_loss_factory' in plugin_setup:
        result['loss']='Plugin-defined indexed TRAIN objective by arm, including legacy defaults when returned None; see candidate-specific mechanism/controls and source.'
    result.update(plugin_setup['protocol'])
    return result


def combined_objective(individual,auxiliary=None):
    """Optional coupled plugin objective; the accepted no-aux path is exact."""
    value=(individual['dpr']+individual['bge'])/2
    if auxiliary is not None:
        need(isinstance(auxiliary,torch.Tensor) and auxiliary.ndim==0 and bool(torch.isfinite(auxiliary)),
             'finite scalar plugin TRAIN auxiliary objective')
        value=value+auxiliary
    return value


def initial_anchor(module,features,native,original,coords):
    """Default native parity; an explicit staged plugin may supply pinned scores."""
    provider=getattr(module,'initial_score_anchor',None)
    anchor=provider(features,native,original,coords) if callable(provider) else native
    need(set(anchor)==set(native) and all(anchor[k].shape==native[k].shape and
        bool(torch.isfinite(anchor[k]).all()) and not anchor[k].requires_grad for k in native),
        'finite detached complete initial score anchor')
    return anchor


def frozen_parameters(module):
    declared=tuple(getattr(module,'declared_frozen_parameters',lambda:())())
    params=dict(module.named_parameters())
    need(len(set(declared))==len(declared) and all(n in params and not params[n].requires_grad for n in declared),
         'explicit frozen parameter ownership')
    for name in ('heads.dpr.weight','heads.bge.weight'):
        need(name in params and (params[name].requires_grad or name in declared),
             'native head either trained or explicitly frozen')
    return {n:params[n].detach().clone() for n in declared}


def verify_activity(module,active,frozen):
    params=dict(module.named_parameters())
    declared=tuple(getattr(module,'declared_frozen_parameters',lambda:())())
    need(len(set(declared))==len(declared) and set(declared)==set(frozen),
         'frozen parameter declaration unchanged during training')
    for name in ('heads.dpr.weight','heads.bge.weight'):
        need(params[name].requires_grad or name in frozen,'no undeclared freeze during training')
        if params[name].requires_grad:need(active[name]>0,'both trainable native heads train')
    need(all(not params[n].requires_grad and params[n].grad is None and torch.equal(params[n],v)
             for n,v in frozen.items()),'declared frozen parameters remain exactly unchanged')
    need(all(active[n]>0 for n in module.activity_gradient_names()),'all designated mechanism parameters trained')


def run_cell(folder,bound,packed,data,coords,mapping,targets,arm,seed,emit,private_entrances=False,factory=None,loss_fn=loss,batch_loss=None,gradient_transform=None):
    folder.mkdir(exist_ok=False)
    factory=factory or (lambda:PairedReadout(packed,arm,seed,private_entrances))
    module=factory().cuda().eval()
    frozen=frozen_parameters(module)
    parameters=[p for p in module.parameters() if p.requires_grad]
    optimizer_factory=getattr(module,'optimizer_factory',None)
    opt=(optimizer_factory(parameters) if callable(optimizer_factory) else
         torch.optim.AdamW(parameters,lr=.001,weight_decay=0.,foreach=False))
    x,native,original=data
    with torch.no_grad():
        references=module.references(coords,native,x)
    def score(indices,model=module,training=False):
        batch_coords={k:v[indices] for k,v in coords.items()}
        batch_native={k:v[indices] for k,v in native.items()}
        batch_refs=(model.references(batch_coords,batch_native,{k:v[indices] for k,v in x.items()}) if getattr(model,'batch_local_references',False)
                    else {k:v[indices] for k,v in references.items()})
        arguments=({k:v[indices] for k,v in x.items()},batch_native,
                   {k:v[indices] for k,v in original.items()},batch_coords,batch_refs)
        training_forward=getattr(model,'training_forward',None)
        if training and callable(training_forward):return training_forward(*arguments)
        values=model(*arguments)
        return (values,None) if training else values
    with torch.no_grad():
        initial=score(slice(None))
        anchor=initial_anchor(module,x,native,original,coords)
        null=max(float((initial[k]-anchor[k]).abs().max()) for k in ('dpr','bge'))
        need(null<=1e-5,'all100 initial declared-anchor/null scores')
    journal=Journal(folder/'journal',bound)
    g=torch.Generator(device='cpu').manual_seed(seed+17)
    batches=[b for _ in range(20) for b in torch.randperm(80,generator=g).reshape(-1,8).tolist()]
    active={n:0. for n,p in module.named_parameters() if p.requires_grad}
    started=time.time();last_loss=None;shared_probe=None;last_auxiliary=None;auxiliary_shared_probe=None
    for step,batch in enumerate(batches,1):
        opt.zero_grad(set_to_none=True);z,auxiliary=score(batch,training=True)
        individual={k:model_batch_loss(k,batch,z[k],targets,loss_fn,batch_loss) for k in ('dpr','bge')}
        value=combined_objective(individual,auxiliary)
        need(bool(torch.isfinite(value)),'finite same16 model-question loss')
        if step==1 and module.shared is not None:
            target=module.gradient_probe_parameter()
            probes={k:torch.autograd.grad(v,target,retain_graph=True)[0] for k,v in individual.items()}
            need(all(float(g.abs().max())>0 for g in probes.values()),'both backbones supply real shared-map data gradients')
            expected=(probes['dpr']+probes['bge'])/2
            shared_probe={k:float(g.abs().max()) for k,g in probes.items()}
            if auxiliary is not None:
                contribution=(torch.autograd.grad(auxiliary,target,retain_graph=True,allow_unused=True)[0]
                              if auxiliary.requires_grad else None)
                auxiliary_shared_probe=0. if contribution is None else float(contribution.abs().max())
                if contribution is not None:expected=expected+contribution
        value.backward(retain_graph=gradient_transform is not None)
        if step==1 and module.shared is not None:
            need(torch.allclose(target.grad,expected,atol=1e-6,rtol=1e-4),'actual shared gradient equals both task contributions plus declared auxiliary')
        if gradient_transform is not None:
            need(auxiliary is None and batch_loss is None and loss_fn is loss,
                 'gradient transform only declared original allpair objective')
            gradient_transform(module,z,batch,targets,step)
        for n,p in module.named_parameters():
            if p.requires_grad:
                need(p.grad is not None and bool(torch.isfinite(p.grad).all()),'finite actual owned data gradients')
                active[n]=max(active[n],float(p.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);opt.step();last_loss=float(value.detach())
        if gradient_transform is not None:gradient_transform.observe_update(module)
        last_auxiliary=None if auxiliary is None else float(auxiliary.detach())
        if step%20==0:
            journal.save(dict(step=step,module=module.state_dict(),optimizer=opt.state_dict(),
                              cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            emit(phase='training',current=step,total=200,loss=last_loss,
                 gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved(),elapsed=time.time()-started)
    verify_activity(module,active,frozen)
    with torch.no_grad():
        predictions=score(slice(80,100))
        training_predictions=score(slice(0,80))
        training_diagnostic={k:dict(questions=80,denominator=400,
            useful_top5=int(targets.gather(1,torch.argsort(v,dim=-1,descending=True,stable=True)[:,:5]).sum()))
            for k,v in training_predictions.items()}
        del training_predictions
        restored=factory().cuda().eval();restored.load_state_dict(journal.load()['module'])
        if restored.shared is not None:
            restored_refs=restored.references(coords,native,x)
            need(all(torch.equal(references[k],restored_refs[k]) for k in ('dpr','bge')),
                 'restored frozen-reference recomputation, not stale unverified cache')
        replays=score(slice(80,100),restored)
        maximum=max(float((predictions[k]-replays[k]).abs().max()) for k in ('dpr','bge'))
        need(maximum==0,'all40 final-checkpoint prediction replay')
        standalone={}
        for k in ('dpr','bge'):
            # Explicitcontext plugins consume own-backbone prepared windows,
            # not a PCA of finalCLS. Transfer/replay only that backbone's inputs.
            singlecoords=(coords[k][80:].cpu().to(x[k].device) if getattr(restored,'explicit_context',False)
                          else transform_readouts(x[k][80:],k,mapping))
            need(torch.allclose(singlecoords,coords[k][80:],atol=2e-5,rtol=2e-5),'stored inputmap singlebackbone replay')
            single=restored.forward_one(k,x[k][80:],native[k][80:],original[k][80:],singlecoords)
            need(torch.allclose(single,predictions[k],atol=2e-5,rtol=2e-5),'singlebackbone score parity')
            need(all(order(a.tolist())[:5]==order(b.tolist())[:5] for a,b in zip(single.cpu(),predictions[k].cpu())),
                 'singlebackbone completeTop5 parity')
            standalone[k]=float((single-predictions[k]).abs().max())
    records={}
    for kind in ('dpr','bge'):
        need(bool(torch.isfinite(predictions[kind]).all()),'complete finite50scores')
        values=[dict(row_index=i,scores=v.tolist(),top5=order(v.tolist())[:5]) for i,v in enumerate(predictions[kind].cpu())]
        records[kind]=json_file(folder,kind+'-predictions.json',values)
    result=dict(binding=bound,step=200,predictions=records,checkpoint=journal.latest_record(),
                null_maximum=null,replay_maximum=maximum,standalone_replay_maxima=standalone,
                gradient_maxima=active,shared_gradient_probe=shared_probe,loss=last_loss,
                trainable_parameters=sum(p.numel() for p in parameters),elapsed=time.time()-started,
                final_training_auxiliary=last_auxiliary,auxiliary_shared_gradient_probe=auxiliary_shared_probe,
                training_diagnostic=training_diagnostic)
    if gradient_transform is not None:result['gradient_transformation']=gradient_transform.diagnostics()
    diagnostic=getattr(module,'mechanism_diagnostics',None)
    if callable(diagnostic):result['mechanism_diagnostics']=diagnostic()
    json_file(folder,'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    for n in ('data','models','output','progress','binding','readouts','head-results'):p.add_argument('--'+n,required=True)
    p.add_argument('--private-entrances',action='store_true')
    p.add_argument('--collective',action='store_true')
    p.add_argument('--native-features',action='store_true')
    p.add_argument('--window-gap',action='store_true')
    p.add_argument('--backbone-local',action='store_true')
    p.add_argument('--amplitude',action='store_true')
    p.add_argument('--candidate-interference',action='store_true')
    p.add_argument('--plugin')
    a=p.parse_args();output=Path(a.output);output.mkdir(exist_ok=False)
    binding=json.loads(Path(a.binding).read_bytes());root=Path(__file__).resolve().parents[2]
    need(all(digest(root/n)==s for n,s in binding['files'].items()),'exact shared source closure')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    population=rows(a.data);targets=torch.stack(train_targets(a.data,population['train'])).cuda()
    readoutroot=Path(a.readouts);readoutrecords=json.loads((readoutroot/'result.json').read_bytes())
    packed={k:torch.load(checked_member(readoutroot,readoutrecords[k]['compact_file']),weights_only=True) for k in ('dpr','bge')}
    joint=[];prior=Path(a.head_results).resolve()
    for kind,p in packed.items():
        need(p['population_sha256']==digest(Path(a.data)/'population.private.json'),'exact population')
        need(p['source_row_sha256']=={k:[r['raw_sha256'] for r in rs] for k,rs in population.items()},'same100rows, roles and aligned candidate order')
        cache=prior/('cache-'+kind);need(p['cache_sha256']==digest(cache/'cache.json'),'model/native cache identity')
        manifest=json.loads((cache/'cache.json').read_bytes())
        joint.append(dict(backbone=kind,arm='NATIVE',seed=None,directory=str(cache),predictions=manifest['native']))
    plugin_setup=None
    if a.plugin:
        need(not any((a.window_gap,a.collective,a.native_features,a.private_entrances,a.backbone_local,a.amplitude,a.candidate_interference)),'one registered plugin')
        from applications.rag.toy100_plugins_v001 import prepare_plugin
        event(a.progress,phase='feature_preparation',current=0,total=2,scientific_execution='running',backbone='registered_plugin')
        plugin_setup=prepare_plugin(a.plugin,packed,targets)
        coords,mapping=plugin_setup['coords'],plugin_setup['mapping']
    elif a.window_gap:
        need(not a.collective and not a.native_features and not a.private_entrances,'one window scientificplugin')
        coords={k:torch.cat((torch.cat((p['windows']['train'],p['windows']['holdout'])),
                           torch.cat((p['window_masses']['train'],p['window_masses']['holdout']))[...,None]),-1).cuda()
                for k,p in packed.items()}
        mapping=dict(kind='own_native_attention_four_windows',readout_files={k:r['compact_file'] for k,r in readoutrecords.items()},
                     semantics='Complete source-bound ownmodel windowmeans/masses; no pairedinference and no PCA.')
    else:coords,mapping=aligned_coordinates(packed)
    if a.candidate_interference:
        need(not any((a.window_gap,a.collective,a.native_features,a.private_entrances,a.backbone_local,a.amplitude)),'one candidate-interference plugin')
        coords={k:torch.cat((coords[k],torch.cat((p['native']['train'],p['native']['holdout'])).cuda()[...,None]),-1) for k,p in packed.items()}
    map_record=tensor_file(output,'train-input-only-alignment.bin',mapping)
    if plugin_setup and callable(plugin_setup.get('restore_mapping')):
        plugin_setup['restore_mapping'](checked_member(output,map_record))
    data=tensors(packed)
    arms=plugin_setup['arms'] if plugin_setup else ARMS
    if a.collective:
        from applications.rag.toy100_collective_readout_v001 import CollectiveReadout, COLLECTIVE_ARMS
        arms=COLLECTIVE_ARMS
    if a.native_features:
        need(not a.collective and not a.private_entrances,'one explicit scientific plugin')
        from applications.rag.toy100_native_feature_readout_v001 import NativeFeatureReadout, NATIVE_FEATURE_ARMS
        arms=NATIVE_FEATURE_ARMS
    if a.window_gap:
        from applications.rag.toy100_window_gap_v001 import WindowGapReadout, WINDOW_ARMS
        arms=WINDOW_ARMS
    if a.backbone_local:
        need(not any((a.window_gap,a.collective,a.native_features,a.private_entrances)),'one backbone-local scientific plugin')
        from applications.rag.toy100_local_readout_v001 import BackboneLocalReadout
    if a.amplitude:
        need(not any((a.window_gap,a.collective,a.native_features,a.private_entrances,a.backbone_local)),'one full-amplitude plugin')
        from applications.rag.toy100_amplitude_readout_v001 import AmplitudeReadout,AMPLITUDE_ARMS
        arms=AMPLITUDE_ARMS
    if a.candidate_interference:
        from applications.rag.toy100_candidate_interference_v001 import CandidateInterferenceReadout,INTERFERENCE_ARMS
        arms=INTERFERENCE_ARMS
    protocol=dict(candidate='COLLECTIVE_WITNESS_READOUT_v001' if a.collective else ('PRIVATE_ENTRANCE_SHARED_QUANTUM_READOUT_v001' if a.private_entrances else 'PAIRED_SHARED_QUANTUM_READOUT_v001'),source=binding,arms=arms,seeds=SEEDS,
        epochs=20,batch=8,steps=200,lr=.001,weight_decay=0,clip=1.,optimizer='AdamW foreach=False',
        loss='16 model-question mean per step: both backbones, same8questions, allbinarypairs and no-pair denominator retained',
        paired_mapping='Train-only12PCA of full50question-centered readouts; BGE row-vector Procrustes toDPR fitted to4000 paired training positions',
        shared='Private permodel input projection+bias (identically initialized) with shared circuit/measurement' if a.private_entrances else 'One shared map/measurement trained jointly; independent original native-linear weights, frozen biases; frozen-reference complete map null',
        circuit='3qubit2reuploads RY/RZ andtrainRY/RX; only betweenblocks CNOT0to1,1to2;63nonidentityPaulis LSBwire0',
        controls='Fresh coupled HEAD, shared LINEAR, complex-normalized MLP-state, PRODUCT(noCNOT), DEPHASED(Z-only) and QUANTUM',
        scope='Already-exposed toy20; no confirmation/generalization/hardware advantage claim; inference needs one backbone only',
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,tf32=False,deterministic=True))
    if a.collective:
        protocol['shared']='Private input-angle entrances; shared state encoder; target-free native-weighted leave-self-out witness; coherent versus mixture controls. No63Pauli freeobservable.'
        protocol['circuit']='3qubit2reupload betweenblockCNOT, shared6firstblocktheta; terminalcommontrainunitary omitted because it cancels. Native50softmax prior fixed, full49other state witness; coherent normalized overlap versus same-stateincoherentMIXTURE.'
        protocol['controls']='HEAD, COSINE_STATE, MLP_STATE, PRODUCT, ABS_STATE, MIXTURE, QUANTUM; all use same full50context/nativeprior and frozen-initial difference exceptHEAD.'
    if a.native_features:
        protocol['candidate']='NATIVE_FEATURE_QUANTUM_READOUT_v001'
        protocol['paired_mapping']='PCA alignment computed only for unchanged interface replay; scientific plugin DOES NOT USE IT. Full own nativeD768/1024 perparagraph LayerNorm -> learned6angles.'
        protocol['shared']='Private full-native-feature angle encoders; shared stateencoder/observable; independent nativeheads. Same-shape current/frozen-reference scalar evaluation.'
        protocol['controls']='HEAD, MLP_STATE, TRIG, PRODUCT, DEPHASED, QUANTUM. Same fullinput/privateangleencoders across alladdedarms; originalhead trained equally.'
    if a.window_gap:
        protocol['candidate']='NATIVE_WINDOW_JENSEN_QUANTUM_v001'
        protocol['paired_mapping']='Own nativeattention fourwindow means/masses from pinned tokenprefix; noPCA orpairedmap. Includesquestion/separators, excludesCLS/padding.'
        protocol['shared']='Privatefullprefix-to-angle encoders andsharedcritic; nonlinear gap sum_w mass*f(window)-f(sum_w mass*window), completefrozeninitialreference difference, ownnativeheadlearned.'
        protocol['controls']='HEAD, POOLED(samequantumcritic onmeanonly), MLP_STATE/TRIG gapreplacements, PRODUCT(candidateCNOTremoved), DEPHASED, QUANTUM.'
    if a.backbone_local:
        protocol['candidate']='BACKBONE_LOCAL_QUANTUM_READOUT_v001'
        protocol['shared']='Same architecture, separate input/circuit/observable parameters for DPR/BGE with identical initial values. Joint mean loss/clipping unchanged for all arms; domain-gradient isolation delta test. No shared-weights or transfer claim.'
        protocol['controls']='HEAD, LINEAR_SHARED(compatibility name; fully local), MLP_STATE, PRODUCT, DEPHASED, QUANTUM; all non-HEAD parameters duplicated per backbone.'
    if a.amplitude:
        protocol['candidate']='FULL_NATIVE_AMPLITUDE_QUADRATIC_v001'
        protocol['paired_mapping']='No PCA enters the scientific plugin. Own nativeCLS LayerNorm/L2, DPR zero-pad768to1024; full signed amplitude on10idealqubits.'
        protocol['circuit']='Two localRY layers, inter-layer CNOT0to1...8to9,55Zsingle/pair observables; separate fitted critic parameters per backbone. Removing CNOT does not disentangle encoded inputs.'
        protocol['shared']='Common architecture; own nativeheads and local critic parameter rows, unchanged jointmean/clipping, complete initial-reference difference and gain1.'
        protocol['controls']='HEAD, MLP(4hidden), LOW_RANK(rank4 quadratic), RANDOM_QUADRATIC(55fixed directions), DIAGONAL, FROZEN_BASIS, LOCAL_ROTATIONS, QUANTUM; same information, disclosed capacity/initialscale differences.'
    if a.candidate_interference:
        protocol['candidate']='CANDIDATE_PHASE_REFLECTION_v001'
        protocol['circuit']='sqrt frozen native-softmax amplitude over50candidates (64basis states,14zero), learned diagonal phase phi=pi*tanh(w.PCA12+b), reflection2|a><a|-I, logprobability-ratio correction with1e-6 numericalfloor.'
        protocol['shared']='Common architecture,13local phaseparams permodel; unchanged nativehead and completeinitialreference difference gain1. Ownfrozen nativeprior, no online otherbackbone.'
        protocol['controls']='HEAD/no-reflection, POINTWISE(no candidatepool), COSINE_ATTENTION(classical globalphasor similarity), CLASSICAL_DIFFUSION(nonnegative1+cos affinity), MLP_CONTEXT, REAL_ONLY(nonunitary imaginary-component removal), QUANTUM.'
    if plugin_setup:
        protocol=registered_protocol(protocol,plugin_setup)
    json_file(output,'protocol.json',protocol)
    for arm in arms:
        for seed in SEEDS:
            name=arm+'-s'+str(seed)
            bound=dict(arm=arm,seed=seed,protocol=digest(output/'protocol.json'),alignment=map_record,
                       readouts={k:readoutrecords[k]['compact_file'] for k in ('dpr','bge')})
            emit=lambda **kw:event(a.progress,backbone='paired_dpr_bge',arm=arm,seed=seed,**kw)
            factory=((lambda:WindowGapReadout(packed,arm,seed)) if a.window_gap else
                     ((lambda:CollectiveReadout(packed,arm,seed)) if a.collective else ((lambda:NativeFeatureReadout(packed,arm,seed)) if a.native_features else None)))
            if a.backbone_local:factory=lambda:BackboneLocalReadout(packed,arm,seed)
            if a.amplitude:factory=lambda:AmplitudeReadout(packed,arm,seed)
            if a.candidate_interference:factory=lambda:CandidateInterferenceReadout(packed,arm,seed)
            if plugin_setup:factory=lambda:plugin_setup['factory'](arm,seed)
            loss_fn=plugin_setup.get('loss_fn',loss) if plugin_setup else loss
            batch_loss=plugin_setup['batch_loss_factory'](arm) if plugin_setup and 'batch_loss_factory' in plugin_setup else None
            gradient_transform=(plugin_setup['gradient_transform_factory'](arm,seed)
                                if plugin_setup and 'gradient_transform_factory' in plugin_setup else None)
            result=run_cell(output/name,bound,packed,data,coords,mapping,targets,arm,seed,emit,a.private_entrances,factory,loss_fn,batch_loss,gradient_transform)
            for k in ('dpr','bge'):joint.append(dict(backbone=k,arm=arm,seed=seed,directory=name,predictions=result['predictions'][k]))
    need(len(joint)==2+6*len(arms),'all coupledcells/backbonegroups plus2native')
    json_file(output,'joint-seal.json',joint)
    result=evaluate(a.data,output,joint);json_file(output,'result.json',result)
    event(a.progress,phase='completed',current=len(joint),total=len(joint),scientific_execution='completed')
    print(json.dumps(result))


if __name__=='__main__':main()
