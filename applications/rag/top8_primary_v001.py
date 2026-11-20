"""Primary-effect subset and immutable adoption of corresponding historical cells."""
from pathlib import Path
import json
import os
import shutil
from applications.rag.qvalue_portable_v001 import need,digest,json_file


def selected_arms(registry,profile):
    need(profile in ('primary54','full'),'known experiment profile')
    return {idea:('QUANTUM',) if profile=='primary54' else arms for idea,arms in registry.items()}


def adopt_primary(source,destination,jobs):
    """Hardlink/copy SHA-verified compact results or latest recovery; never write source."""
    from applications.rag.top8_suite_v002 import verify_files
    from applications.rag.top8_recovery_v001 import Journal
    source=Path(source);destination=Path(destination)
    if not (source/'run-binding.json').exists():return
    need(not (source/'evaluation.json').exists(),'cannot select a new subset after evaluation')
    destination.mkdir(parents=True,exist_ok=True)
    def copy_file(relative):
        a=source/relative;b=destination/relative
        need(a.resolve().is_relative_to(source.resolve()) and not a.is_symlink() and a.is_file(),'safe adoption source')
        sha=digest(a)
        if b.exists():need(not b.is_symlink() and digest(b)==sha,'same adopted bytes');return
        b.parent.mkdir(parents=True,exist_ok=True)
        temp=b.with_name(b.name+'.adopting')
        if temp.exists():need(digest(temp)==sha,'incomplete adoption needs inspection')
        else:
            try:os.link(a,temp)
            except OSError:
                with a.open('rb') as src,temp.open('xb') as dst:
                    shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
        need(digest(temp)==sha,'adopted file SHA');os.replace(temp,b)
    copy_file('run-binding.json')
    original=json.loads((source/'run-binding.json').read_bytes())
    for p in source.glob('*-continuation.v*.json'):copy_file(p.name)
    reused=[]
    for key in jobs:
        need(key and all(c.isalnum() or c in '_-' for c in key),'safe selected cell')
        cell=source/key;disposition=cell/'disposition.json'
        if (destination/key/'disposition.json').exists():continue
        if disposition.exists():
            value=json.loads(disposition.read_bytes())
            need(value['binding']==original,'adopted cell original binding')
            if value['status']=='complete':
                verify_files(source,value['files'])
                for record in value['files']:copy_file(record.get('path',record['name']))
            # Preserve failures, never turn adoption into a retry.
            copy_file(key+'/disposition.json');reused.append(dict(job=key,state=value['status']))
        elif (cell/'recovery/binding.json').exists():
            binding=json.loads((cell/'recovery/binding.json').read_bytes())
            need(binding['matrix']==original,'recovery original matrix binding')
            journal=Journal(cell/'recovery',binding);record=journal.latest_record()
            copy_file(key+'/recovery/binding.json')
            if record:
                copy_file(key+'/recovery/'+record['tensor']['name'])
                copy_file(key+'/recovery/'+f"step-{record['step']:08d}.json")
            reused.append(dict(job=key,state='checkpoint',step=record['step'] if record else 0))
    if not (destination/'adoption.json').exists():
        json_file(destination,'adoption.json',dict(schema='top8.primary-adoption.v1',
            source_binding_sha256=digest(source/'run-binding.json'),selected_jobs=jobs,reused=reused,
            source_outputs_unchanged=True,selection_uses_labels_or_scores=False))


def freeze_profile(output,profile,jobs):
    value=dict(schema='top8.experiment-profile.v1',profile=profile,jobs=jobs,
        objective='method versus native baseline' if profile=='primary54' else 'full controls and attribution',
        quantum_attribution_certified=False)
    path=Path(output)/'experiment-profile.json'
    if path.exists():need(json.loads(path.read_bytes())==value,'immutable experiment profile')
    else:json_file(Path(output),path.name,value)
