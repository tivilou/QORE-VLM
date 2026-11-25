"""Compact frozen final-CLS FFN boundary export, no model fit or label reads."""
import argparse,copy,gc,json
from pathlib import Path
import torch
from torch.nn import functional as F
from applications.rag.toy100_qk_task_v001 import extract,Reader,rows,order
from applications.rag.toy100_qk_study_v001 import event
from applications.rag.qvalue_portable_v001 import need,digest,checked_member,tensor_file,json_file


def frozen_tail(z,residual,dense,normalization,bridge,eps):
    output=F.linear(F.gelu(z),dense['weight'].to(z),dense['bias'].to(z))
    cls=F.layer_norm(output+residual,(residual.shape[-1],),normalization['weight'].to(z),normalization['bias'].to(z),eps)
    return (F.linear(cls,bridge['weight'].to(z),bridge['bias'].to(z)).tanh() if bridge else cls)


def main():
    p=argparse.ArgumentParser()
    for n in ('data','models','output','progress','binding','cache-source','readouts'):p.add_argument('--'+n,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];binding=json.loads(Path(a.binding).read_bytes())
    need(all(digest(root/n)==s for n,s in binding['files'].items()),'exact FFN export closure')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    output=Path(a.output);output.mkdir(exist_ok=False);population=rows(a.data);source=Path(a.readouts)
    originals=json.loads((source/'result.json').read_bytes());result={}
    for kind in ('dpr','bge'):
        packed=torch.load(checked_member(source,originals[kind]['compact_file']),weights_only=True)
        cache=Path(a.cache_source)/('cache-'+kind);need(packed['cache_sha256']==digest(cache/'cache.json'),'same frozen prefix cache')
        b,manifest=extract(kind,Path(a.models)/kind,a.data,cache);reader=Reader(cache,manifest,b,gpu_bytes=2*1024**3)
        need(type(b.last.intermediate.intermediate_act_fn).__name__=='GELUActivation','native exact GELU binding')
        cpu=lambda module:{n:v.detach().cpu() for n,v in module.state_dict().items()}
        dense=cpu(b.last.output.dense);norm=cpu(b.last.output.LayerNorm);bridge=cpu(b.bridge.dense) if kind=='bge' else None;eps=b.last.output.LayerNorm.eps
        zs={};residuals={};maximum=0.;top5_differences=0
        with torch.no_grad():
            for role,rs in population.items():
                zrows=[];rrows=[]
                for index,row in enumerate(rs):
                    capture={};boundary={}
                    def hook(module,args):boundary.update(residual=args[0][:,0].detach(),z=module.dense(args[0][:,0]).detach())
                    h=b.last.intermediate.register_forward_pre_hook(hook)
                    try:reader.scores(role,index,capture=capture)
                    finally:h.remove()
                    features=frozen_tail(boundary['z'],boundary['residual'],dense,norm,bridge,eps)
                    expected=b.bridge(capture['cls']);error=float((features-expected).abs().max());maximum=max(maximum,error)
                    need(torch.allclose(features,expected,atol=2e-5,rtol=2e-5),'all50 reconstructed native FFN/bridge')
                    score=b.head(features).squeeze(-1);expected_score=b.head(expected).squeeze(-1)
                    need(torch.allclose(score,expected_score,atol=2e-5,rtol=2e-5),'all50 native FFN logit parity')
                    top5_differences+=int(set(order(score.tolist())[:5])!=set(order(expected_score.tolist())[:5]))
                    zrows.append(boundary['z'].cpu());rrows.append(boundary['residual'].cpu())
                    event(a.progress,phase='exporting_native_ffn_boundary',backbone=kind,current=index+1,total=len(rs))
                zs[role]=torch.stack(zrows);residuals[role]=torch.stack(rrows)
        need(top5_differences==0,'all100 FFN boundary Top5 unchanged')
        need(packed['source_row_sha256']=={k:[r['raw_sha256'] for r in rs] for k,rs in population.items()},'all100 original source rows')
        packed.update(ffn_z=zs,ffn_residual=residuals,ffn_dense=dense,ffn_normalization=norm,ffn_bridge=bridge,ffn_eps=eps,
                      ffn_semantics='Original lastlayer CLS preGELU andattentionoutputresidual; nativeoutputdense/LayerNorm andBGE dense+tanh frozen; no newfullmodel forward orlabels')
        rec=tensor_file(output,kind+'-readouts.bin',packed)
        result[kind]=dict(compact_file=rec,questions=100,candidates=5000,intermediate=zs['train'].shape[-1],hidden=b.dimension,
            maximum_native_tail_error=maximum,top5_differences=top5_differences,labels_decoded=False,model_fitted=False,new_full_model_forwards=0,source_cache_sha256=digest(cache/'cache.json'))
        del reader,b,packed,zs,residuals,capture,boundary,features,expected,score,expected_score;gc.collect();torch.cuda.empty_cache()
    json_file(output,'result.json',result);event(a.progress,phase='completed',current=2,total=2,scientific_execution='completed');print(json.dumps(result))


if __name__=='__main__':main()
