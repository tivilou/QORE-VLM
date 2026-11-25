"""Portable publication and journals; reuse the accepted full-state reader."""
import hashlib,json,math,os,uuid
from pathlib import Path
from applications.rag.top8_recovery_v001 import Journal
from applications.rag.qvalue_portable_v001 import need,digest


def immutable_json(root,name,value):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    need(Path(name).name==name,'publication basename')
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();target=root/name
    if target.exists():
        need(target.read_bytes()==raw,'immutable JSON mismatch');return member(root,target)
    part=root/(name+'.'+uuid.uuid4().hex+'.part')
    with part.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    try:os.link(part,target)
    except FileExistsError:need(target.read_bytes()==raw,'concurrent publication mismatch')
    part.unlink()
    fd=os.open(root,os.O_DIRECTORY) if os.name=='posix' else None
    if fd is not None:
        try:os.fsync(fd)
        finally:os.close(fd)
    need(target.read_bytes()==raw,'published JSON readback')
    return member(root,target)


def member(root,path):
    root=Path(root);path=Path(path)
    need(path.is_file() and not path.is_symlink(),'regular declared artifact')
    return dict(name=path.relative_to(root).as_posix(),size=path.stat().st_size,sha256=digest(path))


def verify_seal(root):
    root=Path(root);seal=json.loads((root/'joint-prediction-seal.json').read_bytes())
    names=[]
    for r in seal['files']:
        p=root/r['path'];need(not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts,'safe sealed member')
        actual=member(root,p);need(actual==dict(name=r['path'],size=r['size'],sha256=r['sha256']),'sealed member hash')
        names.append(r['path'])
    need(len(names)==len(set(names)) and sum(Path(n).name.startswith('predictions-') for n in names)==seal['expected_prediction_groups'],'unique complete sealed predictions')
    need(not seal['failures'] and seal['all_predictions_before_evaluation_labels'],'complete prediction barrier')
    return seal


def verify_campaign(root,binding,protocol,evaluation_rows):
    """Validate the requested roster, not a self-reported seal group count."""
    root=Path(root);seal=verify_seal(root);expected=2+2*len(protocol['arms'])*len(protocol['seeds'])
    need(seal['expected_prediction_groups']==expected and seal['arms']==protocol['arms'] and seal['seeds']==protocol['seeds'] and seal['roles']==protocol['roles'],'sealed exact requested campaign')
    allowed={'run-binding.json','frozen-dpr.bin','frozen-bge.bin','predictions-native-dpr.json','predictions-native-bge.json'}
    steps=protocol['roles']['train']//protocol['logical_questions']*protocol['epochs']
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            folder=root/(arm+'-s'+str(seed));cell=json.loads((folder/'result.json').read_bytes())
            need(cell['binding']==dict(binding,arm=arm,seed=seed) and cell['step']==steps and cell['arm']==arm and cell['seed']==seed,'exact completed cell identity/endpoint')
            cp=cell['checkpoint'];need(cp['step']==steps and cp['binding']==cell['binding'],'final full-state binding')
            record=folder/'journal'/f'step-{steps:08d}.json'
            need(json.loads(record.read_bytes())==cp and json.loads((folder/'journal/binding.json').read_bytes())==cell['binding'],'final journal/manifest identity')
            from applications.rag.qvalue_portable_v001 import checked_member
            paths=[folder/'result.json',record,folder/'journal/binding.json',checked_member(folder/'journal',cp['tensor'])]
            paths += [checked_member(folder,cell['predictions'][k]) for k in ['dpr','bge']]
            allowed.update(p.relative_to(root).as_posix() for p in paths)
    actual={r['path'] for r in seal['files']};need(actual==allowed,'seal closure is the exact complete requested roster')
    expected_ids=[r['id'] for r in evaluation_rows]
    for name in actual:
        if not Path(name).name.startswith('predictions-'):continue
        rows=json.loads((root/name).read_bytes())['rows'];need([r['id'] for r in rows]==expected_ids,'every group preserves all evaluation identities/order')
        for row,source in zip(rows,evaluation_rows):
            scores=row['scores'];need(row['candidate_ids']==[p['id'] for p in source['paragraphs']] and len(scores)==50 and
                all(type(x) in (int,float) and math.isfinite(x) for x in scores) and row['ranking']==sorted(range(50),key=lambda i:(-scores[i],i)),'complete finite original candidate order/native ties')
    return seal


class RecoveryJournal(Journal):
    def __init__(self,root,binding):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.binding=binding
        if not (self.root/'binding.json').exists():
            need(all(p.name.startswith('binding.json.') and p.name.endswith('.part') for p in self.root.iterdir()),'unbound journal is not empty; do not adopt checkpoints')
        immutable_json(self.root,'binding.json',binding)

    def save(self,value):
        from applications.rag.local_born_cache_v001 import new_tensor
        step=value['step'];need(step>0,'positive checkpoint')
        tensor=new_tensor(self.root,f'step-{step:08d}',value)
        immutable_json(self.root,f'step-{step:08d}.json',dict(step=step,binding=self.binding,tensor=tensor))
