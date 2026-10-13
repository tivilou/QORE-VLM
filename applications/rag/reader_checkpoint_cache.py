"""Bounded-memory, byte-preserving cache for legacy DPR state dictionaries.

No Torch import or arbitrary pickle globals during conversion. Only contiguous,
whole, uniquely owned FloatStorage tensors are supported; other formats fail
closed. This changes serialization, never precision or model parameters.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import struct


MAGIC = 0x1950A86A20F9469CFC6C


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Storage:
    key: str
    count: int


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


def rebuild(storage, offset, shape, stride, *_ignored):
    return Tensor(storage, offset, tuple(shape), tuple(stride))


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"): return OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"): return rebuild
        if (module, name) == ("torch", "FloatStorage"): return "F32"
        raise ValueError("unsupported checkpoint metadata global")

    def persistent_load(self, identity):
        if not isinstance(identity, tuple) or len(identity) != 6 or identity[0] != "storage" or identity[1] != "F32" or identity[5] is not None:
            raise ValueError("only independent FP32 storages supported")
        _, _, key, location, count, _ = identity
        if not isinstance(key, str) or not isinstance(location, str) or type(count) is not int or count < 1:
            raise ValueError("invalid storage metadata")
        return Storage(key, count)


def _read_metadata(stream):
    # Restrict even the scalar/header unpickles to the same global allowlist.
    load = lambda: MetadataUnpickler(stream).load()
    if load() != MAGIC or load() != 1001: raise ValueError("unsupported legacy serialization")
    system = load()
    if not isinstance(system, dict) or system.get("little_endian") is not True:
        raise ValueError("little-endian checkpoint required")
    tensors, keys = load(), load()
    if not isinstance(tensors, OrderedDict) or not isinstance(keys, list) or len(keys) != len(set(keys)):
        raise ValueError("state dictionary/storage order invalid")
    by_storage = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor): raise ValueError("tensor-only state dictionary required")
        shape = tensor.shape
        if any(type(n) is not int or n < 1 for n in shape): raise ValueError("invalid shape")
        stride, step = [], 1
        for n in reversed(shape): stride.insert(0, step); step *= n
        if tensor.offset != 0 or tensor.stride != tuple(stride) or math.prod(shape) != tensor.storage.count or tensor.storage.key in by_storage:
            raise ValueError("only whole contiguous uniquely owned tensors supported")
        by_storage[tensor.storage.key] = (name, tensor)
    if set(keys) != set(by_storage): raise ValueError("storage inventory mismatch")
    return keys, by_storage


def convert_legacy(source, directory):
    """Stream storage bytes with 1 MiB buffers, then reopen/hash every tensor."""
    source, directory = Path(source), Path(directory)
    source_sha = file_hash(source)
    output, receipt = directory/"model.safetensors", directory/"conversion.json"
    if receipt.exists():
        document = json.loads(receipt.read_text(encoding="utf-8"))
        if document["source_sha256"] != source_sha: raise ValueError("cache source mismatch")
        verify_cache(output, document)
        return document
    if output.exists(): raise ValueError("cache without integrity receipt")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory/"model.safetensors.partial"
    if temporary.exists(): raise ValueError("unfinished conversion must be inspected")
    with source.open("rb") as original:
        keys, by_storage = _read_metadata(original)
        header, position = {"__metadata__": {"format": "pt"}}, 0
        for key in keys:
            name, tensor = by_storage[key]
            size = tensor.storage.count * 4
            header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [position, position+size]}
            position += size
        encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)
        hashes = {}
        try:
            with temporary.open("xb") as dest:
                dest.write(struct.pack("<Q", len(encoded))); dest.write(encoded)
                for key in keys:
                    name, tensor = by_storage[key]
                    count = original.read(8)
                    if len(count) != 8 or struct.unpack("<Q", count)[0] != tensor.storage.count:
                        raise ValueError("raw storage count mismatch")
                    remaining, digest = tensor.storage.count * 4, hashlib.sha256()
                    while remaining:
                        block = original.read(min(1024*1024, remaining))
                        if not block: raise ValueError("truncated storage")
                        dest.write(block); digest.update(block); remaining -= len(block)
                    hashes[name] = {**header[name], "sha256": digest.hexdigest()}
                if original.read(1): raise ValueError("unexpected trailing checkpoint bytes")
            temporary.rename(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    document = {"schema_version": "rag.reader_serialization_cache.v1", "source_sha256": source_sha,
        "source_bytes": source.stat().st_size, "output_sha256": file_hash(output), "output_bytes": output.stat().st_size,
        "precision": "unchanged_FP32", "tensor_count": len(hashes), "tensors": hashes, "buffer_bytes": 1024*1024}
    verify_cache(output, document)
    receipt.write_text(json.dumps(document, indent=2)+"\n", encoding="utf-8")
    return document


def verify_cache(path, document):
    if file_hash(path) != document["output_sha256"] or Path(path).stat().st_size != document["output_bytes"]:
        raise ValueError("converted archive integrity mismatch")
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 1024*1024: raise ValueError("unexpected header size")
        header = json.loads(stream.read(length))
        if set(header)-{"__metadata__"} != set(document["tensors"]): raise ValueError("tensor key mismatch")
        for name, tensor in document["tensors"].items():
            if header[name] != {k: tensor[k] for k in ("dtype", "shape", "data_offsets")}:
                raise ValueError("tensor metadata mismatch")
            start, end = tensor["data_offsets"]
            stream.seek(8+length+start)
            digest, remaining = hashlib.sha256(), end-start
            while remaining:
                block = stream.read(min(1024*1024, remaining))
                if not block: raise ValueError("truncated converted tensor")
                digest.update(block); remaining -= len(block)
            if digest.hexdigest() != tensor["sha256"]: raise ValueError("converted tensor bytes changed")
    return True


def load_streaming_reader(config, archive):
    """Execute actual frozen DPR modules, materializing only active module weights.

    Classifier/span heads stay resident for the adapter contract. Encoder
    parameters are FP32 mapped views during each real module forward, reset to
    meta afterwards. No autograd through the frozen encoder is supported.
    """
    import torch
    from safetensors import safe_open
    from transformers import DPRReader
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        reader = DPRReader(config).eval().requires_grad_(False)
    archive = str(Path(archive).resolve())
    with safe_open(archive, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        shapes = {name: handle.get_slice(name).get_shape() for name in keys}
    expected = dict(reader.named_parameters())
    unused = keys - set(expected)
    allowed = {"span_predictor.encoder.bert_model.pooler.dense.weight", "span_predictor.encoder.bert_model.pooler.dense.bias"}
    if set(expected)-keys or unused-allowed or reader.span_predictor.encoder.bert_model.pooler is not None:
        raise ValueError("streaming Reader parameter coverage mismatch")
    for name, parameter in expected.items():
        if list(parameter.shape) != shapes[name] or parameter.dtype != torch.float32:
            raise ValueError("streaming Reader tensor shape/precision mismatch")
    handles = []
    for name, module in reader.named_modules():
        # BERT's non-persistent buffers are deterministic architecture constants.
        for key, buffer in list(module.named_buffers(recurse=False)):
            if buffer.is_meta:
                if key == "position_ids": value = torch.arange(config.max_position_embeddings).expand((1,-1))
                elif key == "token_type_ids": value = torch.zeros(buffer.shape, dtype=torch.long)
                else: raise ValueError("unregistered meta buffer: " + name + "." + key)
                module._buffers[key] = value
        local = dict(module.named_parameters(recurse=False))
        if not local: continue
        entries = {key: (name+"."+key, tuple(parameter.shape), parameter.dtype) for key, parameter in local.items()}
        def materialize(current, _inputs, entries=entries):
            with safe_open(archive, framework="pt", device="cpu") as source:
                for key, (full, shape, dtype) in entries.items():
                    value = source.get_tensor(full)
                    if tuple(value.shape) != shape or value.dtype != dtype: raise ValueError("mapped tensor mismatch")
                    current._parameters[key] = torch.nn.Parameter(value, requires_grad=False)
        def release(current, _inputs, _outputs, entries=entries):
            for key, (_, shape, dtype) in entries.items():
                current._parameters[key] = torch.nn.Parameter(torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False)
        if name.startswith("span_predictor.encoder."):
            handles.append(module.register_forward_pre_hook(materialize))
            handles.append(module.register_forward_hook(release, always_call=True))
        else:
            materialize(module, ())
    reader._streaming_handles = handles
    reader._streaming_archive = archive
    return reader, sorted(unused)
