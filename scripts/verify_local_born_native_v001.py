"""Two fixed existing toy rows: actual native extractor acceptance, not efficacy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from applications.rag.local_born_cache_v001 import extract
from applications.rag.local_born_io_v001 import immutable_json
from applications.rag.local_born_smallset_v001 import policy
from applications.rag.toy100_qk_task_v001 import rows
from applications.rag.qvalue_portable_v001 import checked_member,need


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    policy();a.output.mkdir(parents=True,exist_ok=False);population=rows(a.root/'data/toy100-high-ceiling-v001')
    selected=[{k:r[k] for k in ['id','component_id','question','paragraphs']} for r in [population['train'][0],population['holdout'][0]]]
    roles={selected[0]['id']:'train',selected[1]['id']:'valid'};frame=a.root/'output-frame-run-v001/ffn-readouts';records=json.loads((frame/'result.json').read_bytes());report={}
    for kind in ['dpr','bge']:
        old=torch.load(checked_member(frame,records[kind]['compact_file']),weights_only=True)
        root=a.output/kind;manifest=extract(kind,a.root/'models'/kind,selected,roles,root,dict(scope='fixed firstTRAIN/firstheld native adapter only'))
        differences={}
        for i,role in enumerate(['train','holdout']):
            current=torch.load(checked_member(root,manifest['questions'][i]['tensor']),weights_only=True)
            for new_name,old_name in [('z','ffn_z'),('residual','ffn_residual'),('features','features'),('native','native')]:
                reference=old[old_name][role][0];value=current[new_name]
                maximum=float((value-reference).abs().max());need(torch.allclose(value,reference,atol=2e-5,rtol=2e-5),'actual original native extractor parity')
                differences[role+'/'+new_name]=maximum
            need(torch.equal(torch.argsort(current['native'],descending=True,stable=True)[:5],torch.argsort(old['native'][role][0],descending=True,stable=True)[:5]),'fixed original native Top5 order')
        metadata=torch.load(checked_member(root,manifest['metadata']),weights_only=True)
        basis=float((metadata['ffn_frames']['basis']-old['ffn_frames']['basis']).abs().max())
        # New metadata is always pinned, even when an SVD orientation is not unique.
        report[kind]=dict(questions=2,original_candidate_chunks=14,differences=differences,
            new_frame_vs_old_basis_maximum=basis,new_frame_pinned_sha256=manifest['metadata']['sha256'],
            maximum_tail_replay_error=manifest['maximum_tail_error_new_questions'])
        del old,metadata
    result=dict(status='PASS',scope='Actual native forwards on first TRAIN and first exposed-held toy questions, both backbones; no fullsmallset or labels',backbones=report)
    immutable_json(a.output,'acceptance.json',result);print(json.dumps(result))


if __name__=='__main__':main()
