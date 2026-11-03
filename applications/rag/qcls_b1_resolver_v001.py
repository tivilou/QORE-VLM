"""Candidate-specific B1 source and TRAINING-target binding; source preparation.

No CLI, admission, model loader, label adjudication, or development evaluation.
Call only from a separately reviewed, admitted P4 parent. Never import an old
population/audit/scoring helper. Protected payloads are scanned as byte spans;
only allowlisted role/identity metadata and selected B1 bodies are decoded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing, contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


BRIDGE_SHA = "1174d30a017bf10f158787af7d0106a3f33c02f052619d0644b0fbc370beae95"
COHORT_SHA = "e371b184eb08047d2e63487828348b8f864e76579de30270a1448a7354430743"
ASSIGNMENTS_SHA = "921325a2fd9d7b158c51bae0349fdbe03ca54417ddceb6dcaee44efbb32116a9"
AUDIT_SHA = "956fe4d3749112818b7ff4a2d5b8f170a68576a8e26390f8e4752dac13b09ae7"
ASSETS_SHA = "71b67830ac16e3ba49215bf47abfe3ef2782a34722622c25077acc1b5606d0c7"
CONSENSUS_SHA = "d5c00de3f1fa618985725f43d19f5293589ef4f23f755f679a6a426229fcf20e"
SUPPLEMENT_SHA = "546c563af728510bedd4fe891dc7eca7e053233ad4267a8849bca5a97dd32184"
KEYS_SHA = "0007114ae8a9a96c83a29271104564696d8c4bdf19d4eff388abe0bfcb115406"
MODEL = "facebook__dpr-reader-multiset-base"
REVISION = "ab1974d6023bc4dd27cd713c9ac2b6247509244d"
ROLES = {"training": 512, "development": 128}
LABELS = {"none": 0, "partial": 1, "direct": 2}
TARGET_BUNDLE_SCHEMA = "qcls.role-target-bundle.v1"
TARGET_SOURCE_PROVENANCE = {
    "keys_sha256": KEYS_SHA,
    "consensus_sha256": CONSENSUS_SHA,
    "supplement_sha256": SUPPLEMENT_SHA,
}
_STRING = re.compile(rb'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"')
_ATOM = re.compile(rb'(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
MIB = 1024**2


def require(condition, message):
    if not condition:
        raise ValueError("QCLS B1 binding: " + message)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate decoded JSON member")
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=_pairs,
                      parse_constant=lambda _: require(False, "nonfinite JSON"))


class Spans:
    """Validate JSON structure without string decoding in skipped values.

    Authentic file hashes are mandatory before constructing this scanner. It
    validates structure/escapes, not UTF-8 semantics of opaque string contents.
    Decoded allowlisted values additionally pass the strict JSON decoder.
    """
    def __init__(self, data):
        self.data = data
        require(type(data) is bytes and len(data) <= 128*MIB, "JSON byte cap")
        self.end = self.skip(0)
        require(self.ws(self.end) == len(data), "trailing JSON bytes")

    def ws(self, pos):
        while pos < len(self.data) and self.data[pos] in b" \r\n\t":
            pos += 1
        return pos

    def string_end(self, pos):
        match = _STRING.match(self.data, pos)
        require(match is not None, "JSON string/escape")
        return match.end()

    def skip(self, pos, depth=0):
        require(depth <= 32, "JSON nesting cap")
        pos = self.ws(pos)
        require(pos < len(self.data), "JSON premature EOF")
        kind = self.data[pos]
        if kind == 34:
            return self.string_end(pos)
        if kind not in (91, 123):
            match = _ATOM.match(self.data, pos)
            require(match is not None, "JSON primitive")
            return match.end()
        stop = 93 if kind == 91 else 125
        pos = self.ws(pos+1)
        if pos < len(self.data) and self.data[pos] == stop:
            return pos+1
        while True:
            if kind == 123:
                pos = self.ws(self.string_end(pos))
                require(pos < len(self.data) and self.data[pos] == 58, "JSON colon")
                pos += 1
            pos = self.ws(self.skip(pos, depth+1))
            require(pos < len(self.data), "container EOF")
            if self.data[pos] == stop:
                return pos+1
            require(self.data[pos] == 44, "JSON separator")
            pos = self.ws(pos+1)

    def fields(self, span=None):
        start, end = (0, self.end) if span is None else span
        pos = self.ws(start)
        require(self.data[pos] == 123, "object expected")
        pos = self.ws(pos+1); result = {}
        while self.data[pos] != 125:
            key_end = self.string_end(pos)
            key = decode(self.data[pos:key_end])
            require(key not in result, "duplicate projected member")
            pos = self.ws(key_end)
            require(self.data[pos] == 58, "projected colon")
            begin = self.ws(pos+1); finish = self.skip(begin)
            result[key] = (begin, finish)
            pos = self.ws(finish)
            if self.data[pos] == 125:
                break
            require(self.data[pos] == 44, "projected separator")
            pos = self.ws(pos+1)
        require(pos+1 == end, "object span boundary")
        return result

    def array(self, span):
        pos = self.ws(span[0]); require(self.data[pos] == 91, "array expected")
        pos = self.ws(pos+1)
        while self.data[pos] != 93:
            end = self.skip(pos)
            yield pos, end
            pos = self.ws(end)
            if self.data[pos] == 93:
                break
            require(self.data[pos] == 44, "array separator")
            pos = self.ws(pos+1)
        require(pos+1 == span[1], "array span boundary; exhaust reader")

    def value(self, span):
        return decode(self.data[span[0]:span[1]])

    def project(self, span, names):
        fields = self.fields(span)
        return {name: self.value(fields[name]) for name in names}


class BoundFiles:
    """Same-descriptor verification, bounded reads, immutable provenance ledger.

    Parent remains responsible for aggregate CPU/RAM/disk/time and exhausting
    every iterator. No file lock or whole-machine isolation is claimed.
    """
    def __init__(self):
        self.files = {}
        self.expanded_bytes = 0

    @staticmethod
    def child(root, name):
        require(isinstance(name, str) and "\\" not in name and
                not Path(name).is_absolute() and ".." not in Path(name).parts, "relative source name")
        root = Path(root).resolve(strict=True)
        child = (root/name).resolve(strict=True)
        require(child.is_relative_to(root) and child.is_file(), "source outside bound root")
        return child

    @contextmanager
    def opened(self, path, expected, *, maximum=1024*MIB, size=None):
        path = Path(path).resolve(strict=True)
        require(re.fullmatch("[0-9a-f]{64}", expected) is not None, "file digest syntax")
        require(os.name == "posix" and hasattr(os, "O_NONBLOCK") and hasattr(os, "O_NOFOLLOW"),
                "P4 regular-file open capabilities")
        require(stat.S_ISREG(path.stat().st_mode), "source must be regular file")
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum and
                    (size is None or before.st_size == size), "regular file byte cap/size")
            def verify():
                stream.seek(0); actual = hashlib.sha256(); consumed = 0
                while True:
                    # At most one byte beyond the frozen size is observed.
                    block = stream.read(min(MIB, before.st_size-consumed+1))
                    if not block:
                        break
                    consumed += len(block)
                    require(consumed <= before.st_size and consumed <= maximum, "hash actual-read cap")
                    actual.update(block)
                after = os.fstat(stream.fileno())
                require(consumed == before.st_size and actual.hexdigest() == expected and
                        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "file drift")
                current = path.stat()
                require((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) ==
                        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "source path replacement")
                stream.seek(0)
            verify()
            record = {"sha256": expected, "bytes": before.st_size}
            key = str(path)
            require(key not in self.files or self.files[key] == record, "conflicting file binding")
            try:
                yield stream
            except GeneratorExit:
                # generator.close() swallows GeneratorExit, so validation
                # errors must escape as themselves rather than its cause.
                verify()
                raise
            except BaseException as original:
                try:
                    verify()
                except BaseException as verification_error:
                    # Preserve the first failure; no ledger success on either.
                    raise original from verification_error
                raise
            else:
                verify()
            self.files[key] = record

    def bytes(self, path, expected, *, size=None, maximum=128*MIB):
        with self.opened(path, expected, size=size, maximum=maximum) as stream:
            data = stream.read(maximum+1)
            require(len(data) <= maximum, "bounded read")
        return data

    def json(self, path, expected):
        return decode(self.bytes(path, expected, maximum=16*MIB))

    def gzip_rows(self, path, spec):
        with self.opened(path, spec["sha256"], size=spec["bytes"]) as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                while True:
                    line = stream.readline(8*MIB+1)
                    if not line:
                        break
                    self.expanded_bytes += len(line)
                    require(len(line) <= 8*MIB and line.endswith(b"\n") and
                            self.expanded_bytes <= 8*1024*MIB, "JSONL/expanded-byte cap")
                    yield line
                require(stream.read(1) == b"", "gzip EOF")


def unique_rows(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        require(identity not in result, "duplicate identity before insertion")
        result[identity] = row
    return result


class ReaderCleanupError(RuntimeError):
    def __init__(self, failures):
        super().__init__("QCLS B1 owned-reader cleanup failed")
        self.failures = tuple(failures)  # Private exception objects, never raw public logs.


def close_owned(readers, primary):
    failures = []
    for name, reader in readers:
        try:
            reader.close()
        except BaseException as error:
            failures.append((name, error))
    if failures:
        cleanup = ReaderCleanupError(failures)
        if primary is not None and not isinstance(primary, GeneratorExit):
            raise primary from cleanup
        raise cleanup


def _population(files, bridge_root):
    bridge = files.json(BoundFiles.child(bridge_root, "verification/bridge_receipt.json"), BRIDGE_SHA)
    freeze = files.json(BoundFiles.child(bridge_root, "questions/freeze_receipt.json"), bridge["freeze_receipt_sha256"])
    require(freeze["question_cohort_sha256"] == COHORT_SHA and
            freeze["assignments_sha256"] == ASSIGNMENTS_SHA and freeze["cases"] == 8531, "freeze chain")
    data = Spans(files.bytes(BoundFiles.child(bridge_root, "questions/questions.private.json"), COHORT_SHA))
    top = data.fields()
    require(data.value(top["schema_version"]) == "rag.joint_witness_question_cohort.v1", "cohort schema")
    selected = {}; protected = 0; count = 0
    components = defaultdict(list); component_roles = {}
    for i, span in enumerate(data.array(top["cases"])):
        count += 1
        metadata = data.project(span, ("role", "component_id"))
        role, component = metadata["role"], metadata["component_id"]
        require(role in (*ROLES, "validation"), "unexpected cohort role")
        require(component not in component_roles or component_roles[component] == role, "cross-role component")
        component_roles[component] = role
        if role not in ROLES:
            protected += 1
            continue  # No protected question, ID, text or semantic field decoding.
        row = data.project(span, ("split", "id", "question_sha256", "component_id", "role", "question"))
        require(row["split"] == "train" and type(row["id"]) is str and
                hashlib.sha256(row["question"].encode()).hexdigest() == row["question_sha256"], "selected question identity")
        row["source_index"] = i
        selected[i] = row; components[component].append(row)
    require(count == 8531 and protected == 7891 and
            Counter(row["role"] for row in selected.values()) == Counter(ROLES), "exact cohort denominator")
    require(len(unique_rows(selected.values(), "id")) == len(unique_rows(selected.values(), "question_sha256")) == 640,
            "unique original B1 identities")
    require(list(selected.values()) == sorted(selected.values(), key=lambda r:
            (0 if r["role"] == "training" else 1, r["component_id"], r["id"])), "original stable cohort order")
    for component, rows in components.items():
        require(digest(sorted((r["split"], r["id"]) for r in rows)) == component, "incomplete selected component")
    assignment_data = Spans(files.bytes(BoundFiles.child(bridge_root, "questions/assignments.private.json"), ASSIGNMENTS_SHA))
    field = assignment_data.fields()["assignments"]
    assignment_selected = {}; selected_components = set(components)
    role_counts = Counter()
    for span in assignment_data.array(field):
        meta = assignment_data.project(span, ("role", "component_id"))
        role_counts[meta["role"]] += 1
        require(meta["component_id"] not in selected_components or meta["role"] in ROLES, "component hidden member")
        if meta["role"] not in ROLES:
            continue  # Reserve/quarantine/validation/test identity bodies stay opaque.
        row = assignment_data.project(span, ("split", "id", "question_sha256", "component_id", "role", "hold_reason"))
        require(row["id"] not in assignment_selected and row["hold_reason"] is None, "assignment duplicate/held member")
        assignment_selected[row["id"]] = row
    require(dict(role_counts) == freeze["role_counts"] and len(assignment_selected) == 640, "complete assignment role counts")
    for row in selected.values():
        assignment = assignment_selected[row["id"]]
        require(all(row[key] == assignment[key] for key in ("split", "id", "question_sha256", "component_id", "role")),
                "assignment/cohort drift")
    # Fixed prospective order: original component hash then unchanged source position.
    train_order = tuple(r["question_sha256"] for r in sorted(selected.values(),
                        key=lambda r: (r["component_id"], r["source_index"])) if r["role"] == "training")
    preflight = tuple(sorted(c for c, rows in components.items() if rows[0]["role"] == "training")[:16])
    require(len(preflight) == 16, "fixed sixteen whole training components")
    return bridge, selected, train_order, preflight


@dataclass(frozen=True)
class BoundPopulation:
    cases: tuple
    identities: tuple
    train_order: tuple[str, ...]
    preflight_components: tuple[str, ...]
    source_manifest_sha256: str
    provenance: tuple


def _traces(files, path, spec, questions):
    """Decode already hash-bound selected trace rows with the strict JSON hook.

    The original byte-span scanner is valuable when protected members must stay
    opaque.  These trace files are the 40 frozen shards covering exactly the
    selected 0:640 B1 cohort, and SourceBoundCase validates every consumed
    semantic field.  Re-scanning every JSON string with a Python regex made the
    first admitted run spend its whole 900-second budget before model loading.
    The strict decoder keeps duplicate-key and non-finite rejection while
    preserving identical row/order/identity checks below.
    """
    current = None; rows = []; seen = set(); finished = set()
    with closing(files.gzip_rows(path, spec)) as source:
        for line in source:
            row = decode(line)
            require(type(row) is dict, "trace JSON object")
            if row.get("model_id") != MODEL or row.get("lane") != "primary_common_context":
                continue
            qhash = row.get("question_sha256")
            require(qhash in questions, "trace outside shard B1 population")
            if current is not None and qhash != current:
                require(len(rows) == 50, "incomplete primary trace group")
                finished.add(current)
                yield current, rows
                rows = []; seen = set()
            require(qhash not in finished, "noncontiguous/duplicate trace question")
            current = qhash
            key = str(row["candidate_id"])
            require(key not in seen and len(rows) < 50, "duplicate/extra primary trace before insertion")
            seen.add(key); rows.append(row)
    if current is not None:
        require(len(rows) == 50, "last primary trace group incomplete")
        yield current, rows


def resolve_source(bridge_root, inputs_root):
    """Return all640 authenticated native cases. No labels or result filtering.

    Imports the separately reviewed scientific component only INSIDE this
    admitted-runtime function; AST preparation must not call this function.
    No partially validated population is returned on any reader failure.
    """
    files = BoundFiles()
    bridge, selected, order, preflight = _population(files, bridge_root)
    export = files.json(BoundFiles.child(bridge_root, "export-v2/export_receipt.json"), bridge["export_receipt_sha256"])
    audit = files.json(BoundFiles.child(inputs_root, "tokenizer_audit_receipt.json"), AUDIT_SHA)
    require(audit["status"] == "pass", "original tokenizer audit not pass")
    inventory = files.json(BoundFiles.child(inputs_root, "token_trace_manifest.private.json"), audit["trace_manifest_sha256"])
    traces = unique_rows(inventory["files"], "name")
    from applications.rag.qcls_native_producer_v001 import SourceBoundCase
    plans = []; covered = set(); specs = []
    export_root = BoundFiles.child(bridge_root, "export-v2/export_receipt.json").parent
    for shard in export["shards"]:
        begin, end = shard["start"], shard["end"]
        require(type(begin) is int and type(end) is int and 0 <= begin < end <= 8531, "shard bounds")
        needed = set(selected) & set(range(begin, end))
        if not needed:
            continue
        require(not covered & needed, "overlapping selected shards")
        covered.update(needed)
        raw = [r for r in shard["files"] if ".candidates." in r["name"]]
        visible = [r for r in shard["files"] if ".visibility." in r["name"]]
        require(len(raw) == len(visible) == 1, "ambiguous source shard")
        name = f"{begin:05d}.token_traces.private.jsonl.gz"
        triplet = ((export_root, raw[0]), (export_root, visible[0]), (inputs_root, traces[name]))
        bound = []
        for root, spec in triplet:
            path = BoundFiles.child(root, spec["name"])
            with files.opened(path, spec["sha256"], size=spec["bytes"]):
                pass
            bound.append((path, spec)); specs.append((str(path), spec["sha256"], spec["bytes"]))
        plans.append((begin, end, needed, bound))
    require(covered == set(selected), "incomplete640 source shard cover")
    identities = tuple({k: row[k] for k in ("source_index", "id", "role", "component_id", "question_sha256")}
                       for row in selected.values())
    manifest = digest({"identities": identities, "train_order": order, "preflight": preflight,
                       "files": sorted(files.files.items()), "shards": specs})
    cases = {}
    for begin, end, needed, bound in plans:
        qhashes = {selected[i]["question_sha256"] for i in needed}
        trace_stream = _traces(files, bound[2][0], bound[2][1], qhashes)
        raw_stream = files.gzip_rows(*bound[0]); visible_stream = files.gzip_rows(*bound[1])
        try:
            count = 0
            for index, (a, b) in enumerate(zip(raw_stream, visible_stream, strict=True), begin):
                count += 1
                require(index < end, "extra source shard row")
                if index not in needed:
                    continue  # Binary rows only: no text/JSON decoding of unselected payload.
                qhash, token_rows = next(trace_stream)
                require(qhash == selected[index]["question_sha256"] and index not in cases, "source/trace stable order")
                cases[index] = SourceBoundCase.from_records(selected[index], decode(a), decode(b), token_rows,
                                                           source_manifest_sha256=manifest)
            require(count == end-begin, "source row count")
            require(next(trace_stream, None) is None, "unconsumed/extra trace group")
        finally:
            close_owned((("traces", trace_stream), ("raw", raw_stream),
                         ("visible", visible_stream)), sys.exc_info()[1])
    require(set(cases) == set(selected), "complete bound native population")
    return BoundPopulation(tuple(cases[i] for i in sorted(cases)), identities, order, preflight,
                           manifest, tuple(sorted(files.files.items())))


@dataclass(frozen=True)
class TrainingTargets:
    # Every512 question retained, including no ordered label pairs.
    rows_by_question: dict
    target_manifest_sha256: str
    provenance: tuple
    development_labels_decoded: int = 0


@dataclass(frozen=True)
class DevelopmentTargets:
    positions: dict
    target_manifest_sha256: str
    provenance: tuple
    development_labels_decoded: int = 1839


def _resolve_role_targets(population, *, bundle_path, bundle_sha256, role):
    """Read one immutable role-only bundle and bind every row to B1 source.

    The training call receives no development-target file.  The development
    call is made only after the joint prediction barrier.  This removes the
    earlier mixed-role byte-read ambiguity without changing any label.
    """
    require(type(population) is BoundPopulation and len(population.cases) == 640,
            "authenticated population required")
    require(role in ROLES and re.fullmatch("[0-9a-f]{64}", bundle_sha256),
            "role bundle identity")
    files = BoundFiles()
    data = Spans(files.bytes(bundle_path, bundle_sha256))
    root = data.fields()
    require(set(root) == {"schema_version", "role", "source_provenance", "rows"} and
            data.value(root["schema_version"]) == TARGET_BUNDLE_SCHEMA and
            data.value(root["role"]) == role and
            data.value(root["source_provenance"]) == TARGET_SOURCE_PROVENANCE,
            "role bundle schema/provenance")
    identities = unique_rows(population.identities, "question_sha256")
    cases = dict(zip(identities, population.cases, strict=True))
    expected_order = (population.train_order if role == "training" else
                      tuple(case.question_sha256 for case in population.cases
                            if case.role == "development"))
    rows = {question: {} for question in expected_order}
    items = set(); positions = set()
    names = {"item_id", "question_sha256", "role", "component_id",
             "source_question_id", "candidate_id", "retrieved_rank", "label"}
    for span in data.array(root["rows"]):
        fields = data.fields(span)
        require(set(fields) == names, "exact role-target row fields")
        row = {name: data.value(fields[name]) for name in names}
        qhash = row["question_sha256"]
        require(qhash in rows and qhash in identities, "role-target question identity")
        identity = identities[qhash]; case = cases[qhash]; rank = row["retrieved_rank"]
        require(type(row["item_id"]) is str and row["item_id"] and
                row["item_id"] not in items and type(rank) is int and 1 <= rank <= 50 and
                (qhash, rank) not in positions and row["role"] == role and
                identity["role"] == role and row["component_id"] == identity["component_id"] and
                row["source_question_id"] == identity["id"] and
                str(row["candidate_id"]) == case.candidate_ids[rank-1] and
                row["label"] in LABELS, "role-target source/label binding")
        items.add(row["item_id"]); positions.add((qhash, rank))
        rows[qhash][rank-1] = LABELS[row["label"]]
    expected_rows = 7339 if role == "training" else 1839
    require(len(rows) == ROLES[role] and len(items) == expected_rows and
            sum(map(len, rows.values())) == expected_rows and
            all(rows[question] for question in expected_order), "complete role-only target bundle")
    ordered = {question: tuple(sorted(rows[question].items())) for question in expected_order}
    provenance = tuple(sorted(files.files.items()))
    manifest = digest({"source_manifest_sha256": population.source_manifest_sha256,
                       "role": role, "bundle_sha256": bundle_sha256,
                       "source_provenance": TARGET_SOURCE_PROVENANCE,
                       "role_order": expected_order, "ordinal_rows_sha256": digest(ordered)})
    return ordered, manifest, provenance


def resolve_training_targets(population, *, bundle_path, bundle_sha256):
    """Read exactly the7339-row training bundle; no dev bundle is accepted."""
    rows, manifest, provenance = _resolve_role_targets(
        population, bundle_path=bundle_path, bundle_sha256=bundle_sha256, role="training")
    return TrainingTargets(rows, manifest, provenance)


def resolve_development_targets(population, *, bundle_path, bundle_sha256):
    """Read exactly the1839-row development bundle after prediction seal."""
    rows, manifest, provenance = _resolve_role_targets(
        population, bundle_path=bundle_path, bundle_sha256=bundle_sha256, role="development")
    positions = {(question, index): label for question, values in rows.items()
                 for index, label in values}
    return DevelopmentTargets(positions, manifest, provenance)


def bind_model_inventory(files, manifest_path, model_root):
    """Pin actual offline tokenizer/config/weight files before a future loader.

    Filename inventory is deliberately exact: no local added_tokens/config
    override or alternate checkpoint may silently participate. This authenticates
    files, not loaded tensors; verify_loaded_reader remains mandatory afterwards.
    """
    manifest = files.json(manifest_path, ASSETS_SHA)
    spec = unique_rows(manifest["models"], "id")[MODEL]
    require(spec["model_id"] == "facebook/dpr-reader-multiset-base" and spec["revision"] == REVISION and
            spec["backend"] == "dpr_reader" and spec["max_length"] == 350, "original scorer specification")
    directory = Path(model_root).resolve(strict=True)
    require(directory.is_dir(), "offline model root")
    inventory = unique_rows(spec["tokenizer_files"] + [spec["weight"]], "name")
    require({p.name for p in directory.iterdir()} == set(inventory) and
            all(p.is_file() and not p.is_symlink() for p in directory.iterdir()), "unexpected offline loader file")
    for name, entry in inventory.items():
        with files.opened(BoundFiles.child(directory, name), entry["sha256"], size=entry["bytes"]):
            pass
    return spec, digest({"model_id": spec["model_id"], "revision": REVISION,
                         "inventory": sorted(files.files.items())})
