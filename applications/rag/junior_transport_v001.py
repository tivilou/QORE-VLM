"""Private, bounded transport and safe package installation; standard library only."""
from __future__ import annotations
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from urllib.parse import quote, urlsplit

CHUNK = 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def relative(value):
    require(isinstance(value, str) and value and '\\' not in value, 'invalid relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and
            all(re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}', p)
                and p not in ('.', '..') for p in path.parts), 'unsafe relative path')
    return path


def member(root, name):
    path = Path(root).joinpath(*relative(name).parts)
    require(path.resolve().is_relative_to(Path(root).resolve()) and
            not any(p.is_symlink() for p in [path, *path.parents]), 'symlink/path escape')
    return path


def record_ok(root, row):
    path = member(root, row['path'])
    require(type(row['size']) is int and row['size'] >= 0 and
            re.fullmatch('[0-9a-f]{64}', row['sha256']), 'invalid pinned file')
    return path.is_file() and path.stat().st_size == row['size'] and digest(path) == row['sha256']


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'symlink JSON target')
    part = path.with_name(path.name + '.part')
    require(not part.is_symlink(), 'symlink JSON partial')
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()
    with part.open('wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(part, path)


class Exchange:
    """No redirect, no token in URL, stream limits, immutable receipt verification."""
    def __init__(self, url, token, timeout=300):
        p = urlsplit(url)
        require(p.scheme in ('https', 'http') and p.hostname and not
                (p.username or p.password or p.query or p.fragment), 'invalid private exchange URL')
        # HTTP is supported for the owner's explicitly authorized existing service.
        require(isinstance(token, str) and token and not any(c in token for c in '\r\n\0'), 'invalid private token')
        self.url, self.token, self.timeout = p, token, timeout

    def connection(self):
        cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        return cls(self.url.hostname, self.url.port, timeout=self.timeout)

    def endpoint(self, route):
        return self.url.path.rstrip('/') + route

    def json(self, method, route, value=None):
        conn = self.connection()
        raw = json.dumps(value or {}).encode()
        try:
            conn.request(method, self.endpoint(route), body=raw if method == 'POST' else None,
                         headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
            response = conn.getresponse()
            body = response.read(1024 * 1024 + 1)
            require(response.status in (200, 201) and len(body) <= 1024 * 1024,
                    'exchange metadata response rejected: HTTP ' + str(response.status))
            return json.loads(body)
        except (OSError, http.client.HTTPException):
            raise RuntimeError('private exchange transport unavailable') from None
        finally:
            conn.close()

    def download(self, row, target):
        require(type(row['size']) is int and 0 <= row['size'] <= 16 * 1024**3 and
                re.fullmatch('[0-9a-f]{64}', row['sha256']), 'invalid download binding')
        remote = str(relative(row['exchange_path']))
        require(remote.startswith('five_ideas/'), 'project namespace required')
        target = Path(target)
        require(not target.is_symlink(), 'symlink download target')
        if target.exists():
            require(target.is_file() and target.stat().st_size == row['size'] and
                    digest(target) == row['sha256'], 'conflicting downloaded asset')
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + '.part')
        require(not part.is_symlink(), 'symlink download partial')
        offset = part.stat().st_size if part.exists() else 0
        require(offset <= row['size'], 'oversized download partial')
        if offset < row['size']:
            conn = self.connection()
            try:
                headers = {'Authorization': 'Bearer ' + self.token}
                if offset:
                    headers['Range'] = 'bytes=' + str(offset) + '-'
                conn.request('GET', self.endpoint('/files/' + quote(remote, safe='/')), headers=headers)
                response = conn.getresponse()
                append = bool(offset and response.status == 206)
                require(response.status in (200, 206) and (not append or
                        response.getheader('Content-Range', '').startswith('bytes ' + str(offset) + '-')),
                        'download response/range rejected: HTTP ' + str(response.status))
                require(response.status != 206 or append, 'unsolicited partial response')
                count = offset if append else 0
                with part.open('ab' if append else 'wb') as stream:
                    while block := response.read(CHUNK):
                        count += len(block)
                        require(count <= row['size'], 'download byte cap exceeded')
                        stream.write(block)
                    stream.flush(); os.fsync(stream.fileno())
            except (OSError, http.client.HTTPException):
                raise RuntimeError('private package transfer interrupted; partial retained for resume') from None
            finally:
                conn.close()
        require(part.stat().st_size == row['size'] and digest(part) == row['sha256'], 'download size/SHA mismatch')
        os.link(part, target); part.unlink()

    def upload(self, path, remote):
        path = Path(path); remote = str(relative(remote))
        require(remote.startswith('five_ideas/') and path.is_file() and not path.is_symlink(), 'invalid upload')
        size, sha = path.stat().st_size, digest(path)
        require(size <= 10 * 1024**3, 'exchange single-file limit exceeded')
        self.json('POST', '/api/directories', {'path': remote.rsplit('/', 1)[0]})
        conn = self.connection()
        try:
            conn.putrequest('PUT', self.endpoint('/upload/' + quote(remote, safe='/')))
            for key, value in {'Authorization': 'Bearer ' + self.token,
                               'Content-Length': str(size), 'Content-Type': 'application/octet-stream'}.items():
                conn.putheader(key, value)
            conn.endheaders()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(CHUNK), b''):
                    conn.send(block)
            response = conn.getresponse(); raw = response.read(1024 * 1024 + 1)
            require(response.status in (200, 201) and len(raw) <= 1024 * 1024,
                    'upload rejected: HTTP ' + str(response.status))
            receipt = json.loads(raw)
            require(receipt.get('path') == remote and receipt.get('size_bytes') == size and
                    receipt.get('sha256') == sha, 'upload receipt size/SHA mismatch')
            return {'path': remote, 'size_bytes': size, 'sha256': sha}
        except (OSError, http.client.HTTPException):
            raise RuntimeError('result upload interrupted; pending manifest retained') from None
        finally:
            conn.close()


def install_package(archive, root, records):
    """Validate every tar member before writing; resume only identical pinned files."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    expected = {r['path']: r for r in records}
    require(len(expected) == len(records) and expected, 'unique pinned package members required')
    for row in records:
        member(root, row['path'])
        require(type(row['size']) is int and row['size'] >= 0 and
                re.fullmatch('[0-9a-f]{64}', row['sha256']), 'invalid member binding')
    with tarfile.open(archive, 'r:*') as tar:
        seen = set()
        for entry in tar:
            name = str(relative(entry.name))
            require(name not in seen and name in expected and entry.isfile() and
                    not entry.issym() and not entry.islnk() and entry.size == expected[name]['size'],
                    'unexpected/duplicate/nonregular tar member')
            seen.add(name)
        require(seen == set(expected), 'missing package members')
        for entry in tar.getmembers():
            row = expected[entry.name]; target = member(root, row['path'])
            if target.exists():
                require(record_ok(root, row), 'conflicting installed package file')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + '.install-part')
            require(not part.exists() and not part.is_symlink(), 'stale package install partial')
            h = hashlib.sha256(); count = 0
            try:
                with tar.extractfile(entry) as source, part.open('xb') as output:
                    while block := source.read(CHUNK):
                        count += len(block); require(count <= row['size'], 'member size cap')
                        output.write(block); h.update(block)
                    output.flush(); os.fsync(output.fileno())
                require(count == row['size'] and h.hexdigest() == row['sha256'], 'member SHA mismatch')
                os.link(part, target)
            finally:
                if part.is_file() and not part.is_symlink():
                    part.unlink()
