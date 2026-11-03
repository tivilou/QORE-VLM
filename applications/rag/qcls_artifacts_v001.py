"""Private QCLS attempt artifacts; source only, not a certified trace writer.

No CLI, data lookup, scientific imports, admission, retry or restore. The queue
parent must bind the new attempt root and supervise this process. Full values
arrive from the reviewed producer as detached CPU bytes; their source lineage,
tensor semantics and real-reader comparison remain separate mandatory gates.
"""
from __future__ import annotations

import hashlib
import functools
import json
import math
import os
import re
import shutil
import stat
import struct
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


GIB = 1024**3
ATTEMPT_LIMIT = 16*GIB
TRACE_LIMIT = GIB
MIN_FREE = 20*GIB
CHUNK = 1024*1024
DTYPES = {"float32": 4, "complex64": 8, "int64": 8, "bool": 1}


class ArtifactViolation(RuntimeError):
    """Stop the single attempt; preserve existing outputs and partial files."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactViolation(message)


@dataclass(frozen=True)
class Receipt:
    relative_path: str
    size: int
    sha256: str


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "not a private regular single-link file")
    return info


def _version(info: os.stat_result) -> tuple[int, ...]:
    # Reading can update atime. It is not a mutation of the sealed content.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_nlink)


def _terminal(method):
    """Every failed public write/validation permanently poisons this writer."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            result = method(self, *args, **kwargs)
            self._owner_check()
            return result
        except BaseException:
            self.failed = True
            raise
    return wrapped


class AttemptArtifacts:
    """Exclusive single writer, fail closed, no overwrite and no recovery mode.

    On the existing P4 filesystem, link(temp, destination) is the atomic
    no-replace commit. A link/fsync failure is terminal even if the destination
    appeared: it is not a successful receipt and must never trigger a retry.
    The parent separately accounts external cache/checkpoint/temp directories;
    this class only bounds the exact empty root pre-created by its parent.
    """

    def __init__(self, parent: Path, leaf: str):
        need(parent.is_absolute(), "absolute pre-authorized parent required")
        need(bool(re.fullmatch(r"qcls-[a-z0-9][a-z0-9-]{0,95}", leaf)), "invalid new attempt leaf")
        need(parent.exists() and parent.is_dir() and not parent.is_symlink(), "missing/private parent")
        need(parent.resolve() == parent, "parent must already be canonical, no symlink ancestors")
        need(shutil.disk_usage(parent).free >= MIN_FREE, "insufficient admission disk")
        self.root = parent / leaf
        info = self.root.lstat()
        need(stat.S_ISDIR(info.st_mode) and not self.root.is_symlink() and
             info.st_dev == parent.lstat().st_dev and info.st_uid == os.geteuid() and
             stat.S_IMODE(info.st_mode) == 0o700 and not any(self.root.iterdir()),
             "empty private parent-created artifact root required")
        self.owner = (os.getpid(), uuid.uuid4().hex)
        self.root_identity = (self.root.stat().st_dev, self.root.stat().st_ino)
        self.busy = self.failed = False
        self.trace_bytes = 0
        self.committed: dict[str, Receipt] = {}
        self.aggregate_quota: Callable[[int], dict] | None = None
        self._fsync_directory(parent)
        self._fsync_directory()

    def _fsync_directory(self, directory: Path | None = None) -> None:
        descriptor = os.open(self.root if directory is None else directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _owner_check(self) -> None:
        need(os.getpid() == self.owner[0] and not self.failed, "wrong process or stopped artifact writer")
        info = self.root.lstat()
        need(stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == self.root_identity,
             "attempt root identity changed")

    def bind_aggregate_quota(self, callback: Callable[[int], dict]) -> None:
        """Bind the parent-owned three-root write-time quota exactly once."""
        self._owner_check()
        need(self.aggregate_quota is None and callable(callback) and
             not self.committed and not self.busy, "aggregate quota lifecycle")
        self.aggregate_quota = callback

    def _path(self, name: str) -> Path:
        # Flat generated names only: no IDs, input filenames, subdirectories,
        # absolute paths, traversal, alternate streams or caller-selected suffix.
        need(bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,95}\.(json|bin)", name)), "unsafe artifact name")
        need(name not in self.committed and not (self.root / name).exists(), "artifact already exists")
        return self.root / name

    def _usage(self) -> int:
        total = 0
        for entry in self.root.iterdir():
            info = entry.lstat()
            need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "foreign root entry or hardlink")
            total += info.st_size
        return total

    @_terminal
    def write(self, name: str, blocks: Iterable[bytes], size: int, *, trace: bool = False) -> Receipt:
        """Write exactly size bytes; an exception permanently stops this writer.

        The transient link counts twice against the conservative logical usage
        cap. Do not clean partial files on error; the parent records them and
        performs the queue-owned shutdown without replaying this operation.
        """
        self._owner_check()
        need(not self.busy, "single writer required")
        self.busy = True
        try:
            target = self._path(name)
            need(type(size) is int and size >= 0, "invalid declared size")
            need(not trace or self.trace_bytes + size <= TRACE_LIMIT, "full-value trace cap")
            need(self.aggregate_quota is not None, "parent aggregate write quota not bound")
            self.aggregate_quota(size)
            need(self._usage() + 2*size <= ATTEMPT_LIMIT, "attempt storage including temp cap")
            need(shutil.disk_usage(self.root).free - size >= MIN_FREE, "disk reserve would be crossed")
            temporary = self.root / (".partial-" + uuid.uuid4().hex)
            digest, actual = hashlib.sha256(), 0
            with temporary.open("xb") as stream:
                for block in blocks:
                    self._owner_check()
                    need(type(block) is bytes and 0 < len(block) <= CHUNK, "invalid/big stream block")
                    need(actual + len(block) <= size, "stream exceeds declared full size")
                    stream.write(block)
                    digest.update(block)
                    actual += len(block)
                self._owner_check()
                need(actual == size, "incomplete artifact stream")
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic no-overwrite, same filesystem, single private writer.
            self._owner_check()
            os.link(temporary, target, follow_symlinks=False)
            self._fsync_directory()
            temporary.unlink()
            self._fsync_directory()
            receipt = Receipt(name, actual, digest.hexdigest())
            self._verify(receipt)
            self._owner_check()
            self.committed[name] = receipt
            if trace:
                self.trace_bytes += actual
            return receipt
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False

    @_terminal
    def json(self, name: str, value: object, *, trace: bool = False) -> Receipt:
        self._owner_check()
        payload = (json.dumps(value, ensure_ascii=True, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        return self.write(name, (payload[i:i+CHUNK] for i in range(0, len(payload), CHUNK)),
                          len(payload), trace=trace)

    def _verify(self, receipt: Receipt) -> None:
        path = self.root / receipt.relative_path
        before = _regular(path)
        need(before.st_size == receipt.size, "receipt size mismatch")
        digest, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            need(_version(os.fstat(stream.fileno())) == _version(before), "artifact replaced before readback")
            while block := stream.read(CHUNK):
                digest.update(block)
                count += len(block)
            after_fd = os.fstat(stream.fileno())
        need(_version(after_fd) == _version(before) == _version(_regular(path)), "artifact mutated during readback")
        need(count == receipt.size and digest.hexdigest() == receipt.sha256, "artifact SHA/EOF mismatch")

    @_terminal
    def full_value(self, name: str, blocks: Iterable[bytes], *, dtype: str,
                   shape: tuple[int, ...], producer_binding_sha256: str) -> Receipt:
        """Canonical little-endian, C-order full coordinates; no sparse/summary substitute.

        No arbitrary pickle/torch.load input. The future tensor producer must
        prove detach/CPU/contiguity and compare real reader slices independently.
        A complete byte stream by itself does not certify that provenance.
        """
        self._owner_check()
        need(dtype in DTYPES and type(shape) is tuple and len(shape) <= 8, "unsupported full-value layout")
        need(all(type(n) is int and 0 <= n <= 1_000_000 for n in shape), "invalid coordinate extent")
        need(bool(re.fullmatch(r"[0-9a-f]{64}", producer_binding_sha256)), "producer binding required")
        need(name.endswith(".bin"), "numeric value must use binary suffix")
        size = math.prod(shape) * DTYPES[dtype]
        values = self.write(name, blocks, size, trace=True)
        descriptor = {"schema": "qcls.full-value.v1", "values": asdict(values), "dtype": dtype,
                      "shape": list(shape), "order": "C", "byte_order": "little",
                      "coordinate_count": math.prod(shape), "producer_binding_sha256": producer_binding_sha256,
                      "producer_semantics_verified": False, "reader_slice_replay_verified": False}
        self.json(name[:-4] + ".json", descriptor, trace=True)
        return values

    @_terminal
    def read(self, receipt: Receipt, *, maximum: int = 16*1024**2) -> bytes:
        """Consume exactly the authenticated bytes from one regular descriptor.

        This is deliberately separate from verify-then-reopen: callers parsing
        a seal/descriptor must parse this returned immutable byte string.
        """
        self._owner_check()
        need(type(receipt) is Receipt and self.committed.get(receipt.relative_path) == receipt,
             "unowned read receipt")
        need(type(maximum) is int and 0 <= receipt.size <= maximum <= ATTEMPT_LIMIT, "read byte cap")
        need(bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,95}\.(json|bin)", receipt.relative_path)),
             "unsafe authenticated read path")
        path = self.root / receipt.relative_path
        before = _regular(path)
        need(before.st_size == receipt.size, "read receipt size")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        digest, blocks, count = hashlib.sha256(), [], 0
        with os.fdopen(descriptor, "rb") as stream:
            need(_version(os.fstat(stream.fileno())) == _version(before), "read file replaced before open")
            while block := stream.read(min(CHUNK, receipt.size-count+1)):
                count += len(block)
                need(count <= receipt.size, "authenticated read grew")
                digest.update(block); blocks.append(block)
            after = os.fstat(stream.fileno())
        need(_version(after) == _version(before) == _version(_regular(path)), "read file changed")
        need(count == receipt.size and digest.hexdigest() == receipt.sha256, "read SHA/EOF/size mismatch")
        return b"".join(blocks)

    @_terminal
    def seal(self, name: str, required: tuple[Receipt, ...], metadata: dict) -> Receipt:
        """Completion manifest only for explicit, present, verified outputs.

        This does not certify the scientific completeness of the required set;
        that set must be frozen by the reviewed producer, not selected by result.
        """
        self._owner_check()
        need(required and len({item.relative_path for item in required}) == len(required),
             "empty/duplicate seal requirements")
        for receipt in required:
            need(self.committed.get(receipt.relative_path) == receipt, "unowned/uncommitted receipt")
            self._verify(receipt)
        return self.json(name, {"schema": "qcls.explicit-artifact-seal.v1", "metadata": metadata,
                                "files": [asdict(item) for item in required]})


def read_full_coordinates(root: Path, descriptor: dict, *,
                          exhausted: Callable[[dict], None] | None = None
                          ) -> Iterable[tuple[int, object]]:
    """Read/hash full binary values and check finite values; not source replay.

    Exhaust the iterator before reporting success. Coordinate ordinal maps to
    the declared C-order shape; complex coordinates consume real/imag pairs.
    No actual candidate arrays have been opened by preparing this source.
    """
    need(exhausted is None or callable(exhausted), "invalid exhaustion recorder")
    need(descriptor.get("schema") == "qcls.full-value.v1" and
         descriptor.get("order") == "C" and descriptor.get("byte_order") == "little", "full-value schema")
    dtype, shape = descriptor["dtype"], descriptor["shape"]
    need(dtype in DTYPES and isinstance(shape, list) and len(shape) <= 8 and
         all(type(n) is int and 0 <= n <= 1_000_000 for n in shape), "full-value extent")
    expected = math.prod(shape)
    need(descriptor["coordinate_count"] == expected, "full coordinate count")
    receipt = Receipt(**descriptor["values"])
    need(bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,95}\.bin", receipt.relative_path)), "unsafe value path")
    need(receipt.size == expected * DTYPES[dtype], "full-value declared byte size")
    path = root / receipt.relative_path
    before = _regular(path)
    need(before.st_size == receipt.size, "full-value size")
    digest, count, remainder = hashlib.sha256(), 0, b""
    formats = {"float32": "<f", "complex64": "<ff", "int64": "<q", "bool": "<B"}
    with path.open("rb") as stream:
        need(_version(os.fstat(stream.fileno())) == _version(before), "full-value replaced before read")
        while block := stream.read(CHUNK):
            digest.update(block)
            payload = remainder + block
            complete = len(payload) // DTYPES[dtype] * DTYPES[dtype]
            for raw in struct.iter_unpack(formats[dtype], payload[:complete]):
                value = complex(*raw) if dtype == "complex64" else raw[0]
                if dtype in ("float32", "complex64"):
                    need(all(math.isfinite(part) for part in raw), "nonfinite stored numeric value")
                if dtype == "bool":
                    need(value in (0, 1), "invalid stored bool")
                    value = bool(value)
                yield count, value
                count += 1
            remainder = payload[complete:]
        after_fd = os.fstat(stream.fileno())
    need(_version(after_fd) == _version(before) == _version(_regular(path)), "full-value changed during read")
    need(not remainder and count == expected and digest.hexdigest() == receipt.sha256,
         "full-value coordinate/SHA/EOF mismatch")
    if exhausted is not None:
        exhausted({
            "schema": "qcls.full-value-exhaustion.v1",
            "descriptor_sha256": hashlib.sha256(json.dumps(
                descriptor, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            "values": asdict(receipt), "coordinates": count, "eof": True,
        })


def verify_trace_readers(artifacts: AttemptArtifacts, forward: list, backward: list,
                         *, arms: tuple[str, ...]) -> dict:
    """Parent consumes each authenticated field to EOF before sealing a trace.

    This independently checks full-value coordinates and recorded reader slices.
    Forward provenance remains the reviewed fresh-forward producer; backward
    values explicitly do not claim an independent backward rerun.
    """
    expected = {(arm, checkpoint) for arm in arms
                for checkpoint in ("initial", "diagnostic_updated")}
    base = {"input_ids", "attention_mask", "membership", "hidden", "final_cls",
            "logits", "complete50_scores", "complete50_ranking", "top5"}
    upstream = {"query_projection", "key_projection", "value_projection", "native_context",
                "contextual_question_pool", "angles", "reference_angles", "observation",
                "reference_observation", "delta", "actual_additive_mask",
                "oracle_attention", "oracle_context"}
    backward_fields = {"loss", "pair_counts", "score_offsets", "supervised_scores",
                       "supervision_tiers", "parameters_before", "gradients", "updates"}
    rows = []
    seen_paths = set()
    for direction, records in (("forward", forward), ("backward", backward)):
        need(type(records) is list and len(records) == len(expected), "trace group cardinality")
        seen = set()
        for entry in records:
            binding = entry["binding"]
            key = (binding["arm"], binding["checkpoint"])
            need(key in expected and key not in seen, "trace arm/checkpoint duplicate/missing")
            seen.add(key)
            receipt = entry["fields"]
            need(type(receipt) is Receipt and receipt.relative_path not in seen_paths,
                 "trace receipt duplicate/type")
            seen_paths.add(receipt.relative_path)
            payload = json.loads(artifacts.read(receipt))
            fields = payload["fields"]
            names = (base | (upstream if key[0] != "HEAD768_active" else set())
                     if direction == "forward" else backward_fields)
            need(payload["schema"] == "qcls.sample-trace.field-replay.v1" and
                 set(fields) == names and entry["field_names"] == sorted(names),
                 "exact full trace field inventory")
            for name in sorted(names):
                field = fields[name]
                field_binding = dict(binding, field=name, scope=(
                    "first2_of_first_complete_preflight_question" if direction == "forward"
                    else "first_fixed_eligible_two-question_batch"))
                need(field["binding"] == field_binding and field["caller_values_equal"] is True and
                     field["replay_required"] == (direction == "forward") and
                     field["same_sample_replay_authenticated"] == (direction == "forward"),
                     "trace source/replay binding")
                if direction == "forward":
                    need(field["exact_replay_binding"] == dict(
                        binding, fresh_native_prefix=True, fresh_final_forward=True),
                        "fresh forward binding mismatch")
                else:
                    need(field["exact_replay_binding"] is None, "backward rerun falsely claimed")
                descriptor_receipt = Receipt(**field["descriptor_receipt"])
                need(descriptor_receipt.relative_path == field["artifact_stem"]+".json",
                     "trace descriptor receipt path")
                descriptor = json.loads(artifacts.read(descriptor_receipt))
                binding_sha = hashlib.sha256(json.dumps(
                    field_binding, sort_keys=True, ensure_ascii=False,
                    separators=(",", ":")).encode()).hexdigest()
                need(descriptor["producer_binding_sha256"] == binding_sha and
                     descriptor["shape"] == field["shape"] and descriptor["dtype"] == field["dtype"] and
                     descriptor["values"] == field["values"], "trace descriptor/field mismatch")
                values_receipt = Receipt(**field["values"])
                need(artifacts.committed.get(values_receipt.relative_path) == values_receipt and
                     values_receipt.relative_path not in seen_paths, "trace values receipt identity")
                seen_paths.add(values_receipt.relative_path)
                count = math.prod(field["shape"])
                selected = {0, count//2, count-1} if count else set()
                need(set(field["reader_slices"]) == {str(i) for i in selected},
                     "trace reader slice inventory")
                eof = []
                consumed = 0
                for index, value in read_full_coordinates(
                        artifacts.root, descriptor, exhausted=eof.append):
                    if index in selected:
                        sample = field["reader_slices"][str(index)]
                        need(sample["reader"] == sample["producer"] == repr(value),
                             "parent reader slice differs from producer")
                    consumed += 1
                need(consumed == count == field["coordinates"] and len(eof) == 1 and
                     eof[0] == field["final_reader_receipt"], "trace EOF completion evidence")
                rows.append(dict(direction=direction, arm=key[0], checkpoint=key[1],
                                 field=name, descriptor=asdict(descriptor_receipt), reader=eof[0]))
        need(seen == expected, "incomplete trace group")
    return {"schema": "qcls.parent-full-trace-readers.v1",
            "status": "all_declared_fields_consumed_to_verified_eof",
            "forward_groups": len(forward), "backward_groups": len(backward),
            "field_count": len(rows), "fields": rows,
            "backward_rerun_claimed": False}
