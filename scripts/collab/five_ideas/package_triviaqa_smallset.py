#!/usr/bin/env python3
"""Package six already-sealed input/label files without relabeling or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import tarfile
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def package(spec,output):
    cpu=time.process_time();wall=time.monotonic();output=Path(output)
    output.mkdir(mode=0o700,exist_ok=False)
    archive=output/'triviaqa-8000-1000-1000-binary.tar';rows=[]
    with tarfile.open(archive,'x') as tar:
        for r in spec['members']:
            path=Path(r['source'])
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size==r['size'] and digest(path)==r['sha256'],'sealed input changed'
            assert Path(r['path']).name==r['path'] and not r['path'].startswith('.')
            info=tar.gettarinfo(str(path),arcname=r['path'])
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o600
            with path.open('rb') as stream:tar.addfile(info,stream)
            rows.append({k:r[k] for k in ('path','size','sha256')})
            assert time.process_time()-cpu<50,'packaging CPU suballocation reached'
    # Restore/hash the actual tar streams; never infer archive validity from size.
    with tarfile.open(archive) as tar:
        seen=set()
        for entry in tar:
            expected=next(r for r in rows if r['path']==entry.name)
            assert entry.isfile() and entry.name not in seen and entry.size==expected['size']
            seen.add(entry.name);h=hashlib.sha256()
            with tar.extractfile(entry) as stream:
                for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
            assert h.hexdigest()==expected['sha256']
        assert seen=={r['path'] for r in rows}
    result=dict(schema='qduet.smallset-package-receipt.v1',status='six_members_packaged_readback_verified',
                archive=dict(path=str(archive),size=archive.stat().st_size,sha256=digest(archive)),
                files=rows,member_readback_SHA_EOF=True,labels_created=0,GPU_tasks=0,
                CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
                maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (output/'package.safe.json').open('x') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(package(json.loads(Path(a.spec).read_bytes()),a.output)))
