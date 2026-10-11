"""Immutable, private support-audit bundle. No dataset retrieval or label inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from applications.rag import support_supervision_axes as axes
from applications.rag import support_boundary_bridge as bridge
from applications.rag.protected_boundary_learning import validate_row

MANIFEST_SHA = "d8c86b3d3ae0261c4f2533f67cc31e17e4dc091d7ddee6dbe6aec8a0ebd731c8"
SOURCE_SHA = "1804608795c99a71fc607ce7216b0e390301ba6c85876e0986a55e304019ff12"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build_bundle(source, final, destination):
    """Preserve every final source/witness byte; never modify the audit flags."""
    final, destination = Path(final), Path(destination)
    raw = (final / "manifest.json").read_bytes()
    if sha(raw) != MANIFEST_SHA:
        raise ValueError("final audit manifest changed")
    document = json.loads(raw)
    entries = {"source.json": Path(source).read_bytes(), "final/manifest.json": raw}
    if sha(entries["source.json"]) != SOURCE_SHA:
        raise ValueError("cohort source changed")
    for row in document["files"]:
        data = (final / row["path"]).read_bytes()
        if len(data) != row["bytes"] or sha(data) != row["sha256"]:
            raise ValueError("audit file changed: " + row["path"])
        entries["final/" + row["path"]] = data
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic zip, exclusive creation; no implicit overwrite.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return {"bytes": destination.stat().st_size, "sha256": sha(destination.read_bytes())}


def load_bundle(path):
    """Reopen all hashes, exact target compilation and actual matching witnesses."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            n.startswith("/") or "\\" in n or any(s in ("", ".", "..") for s in n.split("/")) for n in names
        ):
            raise ValueError("unsafe/duplicate bundle entry")
        raw = archive.read("final/manifest.json")
        if sha(raw) != MANIFEST_SHA:
            raise ValueError("frozen manifest mismatch")
        manifest = json.loads(raw)
        expected = {"source.json", "final/manifest.json"} | {"final/" + r["path"] for r in manifest["files"]}
        if set(names) != expected or sum(i.file_size for i in archive.infolist()) > 150_000_000:
            raise ValueError("bundle coverage/size mismatch")
        documents = {}
        for row in manifest["files"]:
            data = archive.read("final/" + row["path"])
            if len(data) != row["bytes"] or sha(data) != row["sha256"]:
                raise ValueError("frozen file mismatch: " + row["path"])
            documents[row["path"]] = json.loads(data)
        source_bytes = archive.read("source.json")
        if sha(source_bytes) != SOURCE_SHA:
            raise ValueError("source hash mismatch")
        source = json.loads(source_bytes)
    axes.validate_trace(source)
    annotations = documents["annotations.final.json"]
    targets = documents["targets.final.json"]
    compiled, _, _ = bridge.protected_draft(source, annotations)
    if compiled != targets:
        raise ValueError("frozen target compilation mismatch")
    amap = {r["question_sha256"]: r for r in annotations["cases"]}
    tmap = {r["question_sha256"]: r for r in targets["cases"]}
    witness_indices = {}
    for name, doc in documents.items():
        if not name.startswith("witnesses/"):
            continue
        witness_indices[name] = ({c["question_sha256"]: c for c in doc["cases"]},
                                 {p["review_item_id"]: (c, p) for c in doc["cases"] for p in c["items"]})
    witness_count = 0
    def audit(record, scope=False):
        nonlocal witness_count
        qualification = record["scope_qualification" if scope else "qualification"]
        if not axes.qualified(qualification, axes.qualification_binding(record, scope=scope)):
            raise ValueError("training target not qualified")
        for e in qualification["evidence"]:
            case_index, item_index = witness_indices[e["artifact_ref"]]
            if scope:
                case = case_index[record["question_sha256"]]
                axes.require(case["scope"] == record["scope"], "scope witness mismatch")
            else:
                case, item = item_index[record["review_item_id"]]
                axes.require(all(item[k] == record[k] for k in (*bridge.IDENTITY, *axes.AXES)), "item witness mismatch")
                axes.require(bool(item["rationale"]) and all(q in item["text"] for q in item["support_quotes"]), "invalid witness quote")
                if item["question_support"] in ("direct", "partial", "contradictory"):
                    axes.require(bool(item["support_quotes"]), "missing support quote")
            reviewer = case["reviewer"]
            axes.require(reviewer["identity"] == e["reviewer"] and reviewer["prior_review_exposure"] == e["prior_review_exposure"], "reviewer mismatch")
            if e["role"] == "independent_review":
                axes.require(reviewer["outside_packet_context_exposure"] is False, "exposed independent witness")
            # Actual bytes were hashed against the manifest above; bind evidence to that entry.
            expected_hash = next(r["sha256"] for r in manifest["files"] if r["path"] == e["artifact_ref"])
            axes.require(expected_hash == e["sha256"], "witness artifact hash mismatch")
            witness_count += 1
    rows = []
    for case in source["cases"]:
        qid = case["question_sha256"]
        target = tmap[qid]
        if not target["draft_pairs"]:
            continue
        validate_row(target)
        annotation = amap[qid]
        audit(annotation, True)
        by_id = {p["review_item_id"]: p for p in annotation["items"]}
        indices = set(target["eligible_positive_indices"] + target["replaceable_indices"] + target["protected_indices"])
        for index in indices:
            audit(by_id[case["candidates"][index]["review_item_id"]])
        rows.append({"case": case, "target": target, "annotation": annotation})
    if len(rows) != 59 or sum(len(r["target"]["draft_pairs"]) for r in rows) != 228:
        raise ValueError("frozen59/228 population mismatch")
    if any(r["case"]["future_role"] != "training" for r in rows):
        raise ValueError("validation entered training")
    return rows, {"source_sha256": SOURCE_SHA, "final_manifest_sha256": MANIFEST_SHA,
                  "training_cases": 59, "pairs": 228, "actual_witness_checks": witness_count,
                  "semantic_truth_certified": False, "validation_used": False}
