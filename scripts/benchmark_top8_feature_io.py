#!/usr/bin/env python3
"""Synthetic CPU-only IO microbenchmark; not a full-run/GPU ETA predictor."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('baseline','optimized'),required=True)
    a=p.parse_args()
    import torch
    from applications.rag.qvalue_portable_v001 import tensor_file
    from applications.rag.top8_cache_v002 import FeatureRows
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(31)
    questions,candidates,width,passes=64,50,128,4
    with tempfile.TemporaryDirectory(prefix='qduet-synthetic-feature-io-') as d:
        root=Path(d);rows=[]
        for i in range(questions):
            value={k:torch.randn(candidates,width,generator=g) for k in
                   ('depth0','context','residual','final','ffn_residual','readout')}
            value['depths']=torch.randn(candidates,3,width,generator=g)
            value['z']=torch.randn(candidates,4*width,generator=g)
            record=tensor_file(root,str(i)+'.bin',value)
            rows.append(dict(id=str(i),role='training' if i<48 else 'development',
                             retained_indices=list(range(candidates)),feature_file=record))
        manifest=dict(rows=rows);(root/'cache.json').write_text(json.dumps(manifest))
        source=FeatureRows(root,rows,maximum=0)
        train=[r for r in rows if r['role']=='training']
        if a.mode=='optimized':
            from applications.rag.top8_feature_io_v003 import project_features,training_moments,field_bank,close_fields
            started=time.perf_counter();view=project_features(source,manifest,root,('readout',))
            build_seconds=time.perf_counter()-started
            before=field_bank(source,manifest,root).metrics['source_questions_read']
        else:view=source;build_seconds=0;before=0
        checksum=0.;payload=0;started=time.perf_counter()
        for _ in range(passes):
            for row in rows:
                value=view[row['id']]
                payload+=sum(x.numel()*x.element_size() for x in value.values())
                checksum+=float(value['readout'].sum())
        warm_seconds=time.perf_counter()-started
        started=time.perf_counter();statistics=[]
        for _ in range(3):
            if a.mode=='optimized':mean,std=training_moments(source,manifest,root)
            else:
                values=torch.cat([source[r['id']]['readout'] for r in train])
                mean=values.mean(0);std=values.std(0,unbiased=False)
                std=torch.where(std>1e-8,std,torch.ones_like(std))
            statistics.append((mean,std))
        stats_seconds=time.perf_counter()-started
        assert all(torch.equal(mean,x) and torch.equal(std,y) for x,y in statistics)
        if a.mode=='optimized':
            metrics=dict(field_bank(source,manifest,root).metrics)
            warm_source_reads=metrics['source_questions_read']-before
            close_fields()
        else:metrics=dict(statistics_computed=3);warm_source_reads=questions*passes
        print(json.dumps(dict(mode=a.mode,cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),
            synthetic_shape=[questions,candidates,width],passes=passes,fields_per_row=len(value),
            projected_transfer_payload_bytes=payload,warm_source_question_reads=warm_source_reads,
            checksum=checksum,mean=mean.tolist(),std=std.tolist(),
            column_build_seconds=build_seconds,warm_read_seconds=warm_seconds,
            three_statistics_calls_seconds=stats_seconds,metrics=metrics,
            scope='CPU synthetic IO only; payload counts are not timed GPU transfers; no full-run speedup claim')))


if __name__=='__main__':main()
