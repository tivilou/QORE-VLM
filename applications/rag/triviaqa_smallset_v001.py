"""Mechanical input conversion and binary evaluation; no model imports.

Validation and test stay separate in reports. No direct/partial labels are
inferred. All selected methods, controls and seeds predict before label join.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import math
from pathlib import Path
from applications.rag.junior_transport_v001 import digest, member, record_ok, require

ROLES = {'train': 8000, 'valid': 1000, 'test': 1000}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()


def save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def validate_release(release,ready=True):
    require(not ready or release.get('status')=='ready',
            'TRIVIAQA_SMALLSET_ARCHIVE_NOT_PUBLISHED: input/labels complete; owner archive publication is pending.')
    require(release['schema']=='qduet.triviaqa-smallset-release.v1'
            and (release['status']=='ready' or not ready) and release['dataset']=='triviaqa'
            and release['roles']==ROLES and release['label_semantics']=='binary_useful'
            and release['semantic_gold'] is False, 'exact operational binary smallset release required')
    require(len(release['files'])==6 and len({r['path'] for r in release['files']})==6,
            'six unique input and label members')
    for role in ROLES:
        require({role+'.inputs.jsonl.gz',role+'.labels.private.jsonl'} <=
                {r['path'] for r in release['files']}, 'three complete roles required')


def converted_rows(root, role, expected):
    """Read both complete source streams; bind labels to actual original bytes."""
    count=0
    with gzip.open(member(root,role+'.inputs.jsonl.gz'),'rb') as source, \
            member(root,role+'.labels.private.jsonl').open('rb') as labels:
        for raw in source:
            require(raw.endswith(b'\n'),'unterminated source row')
            row=json.loads(raw); vote=json.loads(next(labels))
            require(set(row)=={'id','component_id','question','paragraphs'},'exact label-free input')
            require(vote['id']==row['id'] and vote['component_id']==row['component_id']
                    and vote['source_row_sha256']==hashlib.sha256(raw).hexdigest(), 'label/source row binding')
            ps=row['paragraphs']; ys=vote['labels']
            require(len(ps)==len(ys)==50 and len({p['id'] for p in ps})==50,'original50 identities')
            require(all(set(p)=={'id','text'} and isinstance(p['id'],str) and p['id']
                        and isinstance(p['text'],str) and p['text'].strip() for p in ps), 'raw nonempty passages')
            require(all(p['id']==y['candidate_id'] and type(y['useful']) is int
                        and y['useful'] in (0,1) for p,y in zip(ps,ys)), 'ordered binary labels')
            require(all(isinstance(row[k],str) and row[k].strip() for k in ('id','component_id','question')),
                    'nonempty identity and question')
            count+=1
            yield row,{y['candidate_id']:y['useful'] for y in ys}
        require(labels.read()==b'' and count==expected,'complete role/label EOF and denominator')


def materialize(root, release, output):
    validate_release(release)
    for record in release['files']:
        require(record_ok(root,record),'input/label member SHA mismatch')
    output=Path(output)
    binding=hashlib.sha256(encoded(release)).hexdigest()
    if (output/'materialization.json').exists():
        result=json.loads((output/'materialization.json').read_bytes())
        require(result['release_sha256']==binding,'materialized release changed')
        require(all(record_ok(output,r) for r in result['files']),'materialized bytes changed')
        return result
    output.mkdir(parents=True,exist_ok=False)
    seen_ids=set();seen_components=set();seen_questions=set();roles={};metrics={};files=[]
    with (output/'train.jsonl').open('xb') as train, (output/'evaluation.inputs.jsonl').open('xb') as evaluation:
        for role,n in release['roles'].items():
            ids=set();components=set();questions=set();oracle=useful=0
            with (output/(role+'.labels.jsonl')).open('xb') as targets:
                for row,labels in converted_rows(root,role,n):
                    require(row['id'] not in ids and row['id'] not in seen_ids,'duplicate question ID')
                    normalized=' '.join(row['question'].casefold().split())
                    q=hashlib.sha256(normalized.encode()).hexdigest()
                    require(q not in seen_questions and row['component_id'] not in seen_components,
                            'question/component role leakage')
                    ids.add(row['id']);components.add(row['component_id']);questions.add(q)
                    if role=='train':train.write(encoded(dict(row,labels=labels)))
                    else:evaluation.write(encoded(row))
                    targets.write(encoded(dict(id=row['id'],labels=labels)))
                    # Training/validation headroom only; test targets are copied,
                    # never used for tuning, filtering, ranking or headroom reporting.
                    if role!='test':
                        positives=sum(labels.values());oracle+=min(5,positives);useful+=positives
                    roles[row['id']]=role
            seen_ids.update(ids);seen_components.update(components);seen_questions.update(questions)
            if role!='test':metrics[role]=dict(questions=len(ids),positive_positions=useful,oracle_top5=oracle,
                                             denominator=5*len(ids))
    save(output/'roles.json',roles)
    for p in sorted(output.iterdir()):
        files.append(dict(path=p.name,size=p.stat().st_size,sha256=digest(p)))
    result=dict(schema='qduet.smallset-materialization.v1',release_sha256=binding,
                roles=release['roles'],files=files,headroom=metrics,test_used_for_selection=False,
                label_semantics='binary_useful',labels_created=0)
    save(output/'materialization.json',result)
    return result


def evaluate(output, data):
    """Verify joint seal first; then join binary targets for each fixed role."""
    output=Path(output);data=Path(data)
    seal_path=output/'joint-prediction-seal.json'
    seal=json.loads(seal_path.read_bytes())
    require(seal['all_predictions_before_evaluation_labels'] is True,'joint seal before targets')
    records={}
    for r in seal['files']:
        name=r.get('path',r.get('name'));record=dict(path=name,size=r['size'],sha256=r['sha256'])
        require(name not in records and record_ok(output,record),'unique sealed files and exact bytes')
        records[name]=record
    roles=json.loads((data/'roles.json').read_bytes());targets={}
    for role in ('valid','test'):
        with (data/(role+'.labels.jsonl')).open() as stream:
            for line in stream:
                r=json.loads(line)
                require(r['id'] not in targets and roles[r['id']]==role,'unique evaluation targets')
                require(len(r['labels'])==50 and all(type(y)is int and y in (0,1) for y in r['labels'].values()),
                        'complete useful labels')
                targets[r['id']]=r['labels']
    expected={k for k,v in roles.items() if v!='train'}
    require(set(targets)==expected,'exact validation and test population')
    metrics={}
    for name,record in records.items():
        if not Path(name).name.startswith('predictions-'):continue
        value=json.loads(member(output,name).read_bytes());seen=set()
        counts={r:dict(questions=0,useful=0,denominator=0) for r in ('valid','test')}
        for row in value['rows']:
            q=row['id'];ids=row['candidate_ids'];scores=row['scores'];rank=row['ranking']
            require(q in expected and q not in seen and len(ids)==len(set(ids))==len(scores)==len(rank)==50,
                    'complete unique prediction population')
            require(all(type(x) in (float,int) and math.isfinite(x) for x in scores)
                    and rank==sorted(range(50),key=lambda i:(-scores[i],i)), 'finite scores and native index ties')
            require(set(ids)==set(targets[q]),'prediction target identities')
            seen.add(q);m=counts[roles[q]];m['questions']+=1;m['denominator']+=5
            m['useful']+=sum(targets[q][ids[i]] for i in rank[:5])
        require(seen==expected,'every prediction contains both complete evaluation roles')
        metrics[name]=counts
    require(metrics,'nonempty joint predictions')
    result=dict(schema='qduet.smallset-binary-evaluation.v1',metrics=metrics,failures=seal['failures'],
                joint_seal_sha256=digest(seal_path),label_semantics='binary_useful',
                targets_sha256={r:digest(data/(r+'.labels.jsonl')) for r in ('valid','test')},
                direct_metric='NOT_APPLICABLE',test_used_for_selection=False,
                all_predictions_before_evaluation_labels=True)
    target=output/'smallset-evaluation.json'
    if target.exists():require(json.loads(target.read_bytes())==result,'immutable evaluation')
    else:save(target,result)
    return result
