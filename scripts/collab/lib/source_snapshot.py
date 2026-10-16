"""Archive exact declared source bytes, independently of Git revision/dirty state."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile


def archive_sources(root, paths, destination):
    root = Path(root).resolve()
    destination = Path(destination)
    records, contents = {}, {}
    for relative in sorted(set(str(p).replace("\\", "/") for p in paths)):
        rel = Path(relative)
        if rel.is_absolute() or any(part in ("..", ".") for part in rel.parts):
            raise ValueError("source path must stay inside project")
        path = root / rel
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("invalid source file: " + relative)
        if path.suffix not in (".py", ".json", ".sh") or path.stat().st_size > 2 * 1024**2:
            raise ValueError("source type/size is not allowed: " + relative)
        data = path.read_bytes()
        contents[relative] = data
        records[relative] = hashlib.sha256(data).hexdigest()
    if not records:
        raise ValueError("empty source snapshot")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        archive.writestr("source_manifest.json", json.dumps({"schema_version": "source-snapshot.v1", "source_hashes": records}, indent=2) + "\n")
    return {"file": destination.name, "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "source_hashes": records}


def verify_sources(root, snapshot):
    root = Path(root)
    for name, expected in snapshot["source_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("source changed during execution: " + name)
