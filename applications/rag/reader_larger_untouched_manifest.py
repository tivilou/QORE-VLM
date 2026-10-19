"""Immutable input binding for the larger untouched Reader transport.

This module is deliberately label blind.  It verifies the already-frozen
Top-50 cohort, the pinned DPR Reader snapshot, and the two restored head
states.  It neither loads the Reader nor computes selector scores.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
import zipfile

import numpy as np

from applications.rag.reader_larger_untouched import validate_cohort


SCHEMA = "rag.reader_larger_untouched_scoring_manifest.v1"
ARM_FAMILIES = ("quantum", "classical")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "sha256": digest.hexdigest(),
    }


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"invalid SHA-256 for {label}")
    return text


def _verify_file(path: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if actual["bytes"] != int(expected["bytes"]):
        raise ValueError(f"{label} byte-size mismatch")
    if actual["sha256"] != _require_sha256(expected["sha256"], label):
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def verify_source_archive(path: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Replay a source-snapshot archive rather than trusting its outer hash only."""
    actual = _verify_file(path, expected, label)
    expected_hashes = expected.get("source_hashes")
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise ValueError(f"{label} has no declared source identities")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "source_manifest.json" not in names:
            raise ValueError(f"{label} has no source manifest")
        embedded = json.loads(archive.read("source_manifest.json"))["source_hashes"]
        if dict(embedded) != dict(expected_hashes):
            raise ValueError(f"{label} embedded source manifest mismatch")
        for relative, digest in expected_hashes.items():
            if relative not in names:
                raise ValueError(f"{label} missing source member: {relative}")
            if hashlib.sha256(archive.read(relative)).hexdigest() != _require_sha256(digest, relative):
                raise ValueError(f"{label} source member mismatch: {relative}")
    return {**actual, "source_hashes": dict(expected_hashes), "members_replayed": len(expected_hashes)}


def verify_cohort(cohort_dir: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    cohort_dir = Path(cohort_dir)
    cohort_spec = spec["cohort"]
    cohort_path = cohort_dir / "cohort.json"
    lock_path = cohort_dir / "cohort_lock.json"
    cohort_file = _verify_file(cohort_path, cohort_spec["files"]["cohort.json"], "cohort")
    lock_file = _verify_file(lock_path, cohort_spec["files"]["cohort_lock.json"], "cohort lock")
    source_spec = cohort_spec["source_snapshot"]
    source_archive = verify_source_archive(
        cohort_dir / source_spec["file"], source_spec, "cohort source snapshot"
    )
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = validate_cohort(cohort, expected_cases=int(cohort_spec["cases"]))
    if checks["cohort_identity_sha256"] != cohort_spec["cohort_identity_sha256"]:
        raise ValueError("cohort identity does not match frozen spec")
    if lock.get("schema_version") != "sample-trace.v2" or lock.get("artifact_type") != "reader_larger_untouched_cohort_lock":
        raise ValueError("unexpected cohort-lock schema")
    if lock.get("run_id") != cohort_spec["run_id"]:
        raise ValueError("cohort run identity mismatch")
    if lock.get("cohort_sha256") != cohort_file["sha256"] or lock.get("cohort_identity_sha256") != checks["cohort_identity_sha256"]:
        raise ValueError("cohort lock does not bind the cohort")
    boundary = lock.get("label_boundary")
    if not isinstance(boundary, Mapping) or not boundary or any(value is not False for value in boundary.values()):
        raise ValueError("cohort lock crossed the label boundary")
    samples = lock.get("samples")
    cases = cohort["cases"]
    if not isinstance(samples, list) or len(samples) != len(cases):
        raise ValueError("cohort lock sample coverage mismatch")
    for number, (sample, case) in enumerate(zip(samples, cases), 1):
        expected_ids = [candidate["candidate_id_sha256"] for candidate in case["candidates"]]
        expected_ranks = [int(candidate["retrieved_rank"]) for candidate in case["candidates"]]
        if sample.get("question_sha256") != case["question_sha256"]:
            raise ValueError(f"cohort-lock question mismatch at case {number}")
        if sample.get("candidate_id_sha256") != expected_ids or sample.get("candidate_retrieved_ranks") != expected_ranks:
            raise ValueError(f"cohort-lock candidate mismatch at case {number}")
    return {
        "run_id": cohort_spec["run_id"],
        "exchange_path": cohort_spec["exchange_path"],
        "cases": checks["cases"],
        "top50_rows": checks["top50_rows"],
        "cohort_identity_sha256": checks["cohort_identity_sha256"],
        "files": {"cohort.json": cohort_file, "cohort_lock.json": lock_file},
        "source_snapshot": source_archive,
        "label_boundary_verified": True,
        "lock_samples_replayed": len(samples),
    }


def verify_reader_snapshot(snapshot: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = Path(snapshot)
    reader_spec = spec["reader"]
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    files = {
        name: _verify_file(snapshot / name, expected, f"Reader {name}")
        for name, expected in sorted(reader_spec["files"].items())
    }
    return {
        "model_id": reader_spec["model_id"],
        "revision": reader_spec["revision"],
        "files": files,
        "max_length": int(reader_spec["max_length"]),
        "frozen_max_batch_size": int(reader_spec["frozen_max_batch_size"]),
        "local_snapshot_required": True,
    }


def _restore_state_record(values: np.lib.npyio.NpzFile, family: str) -> dict[str, Any]:
    keys = sorted(
        key for key in values.files
        if key == family + "__final_theta" or key.startswith(family + "__head__")
    )
    entries = [{"key": key, **array_record(values[key])} for key in keys]
    low = sorted(int(key.rsplit("_", 1)[1]) for key in keys if f"{family}__head__low_" in key)
    high = sorted(int(key.rsplit("_", 1)[1]) for key in keys if f"{family}__head__high_" in key)
    if low != list(range(40)) or high != list(range(40)) or len(entries) != 85:
        raise ValueError(f"{family} restored-head state coverage mismatch")
    by_key = {entry["key"]: entry for entry in entries}
    expected_shapes = {
        family + "__final_theta": ([40, 5], "float64"),
        family + "__head__theta": ([40, 5], "float32"),
        family + "__head__weight": ([768], "float32"),
        family + "__head__bias": ([1], "float32"),
        family + "__head__projection": ([4, 3840], "float32"),
    }
    for key, (shape, dtype) in expected_shapes.items():
        if key not in by_key or by_key[key]["shape"] != shape or by_key[key]["dtype"] != dtype:
            raise ValueError(f"{key} shape/dtype mismatch")
    for key in keys:
        value = values[key]
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise ValueError(f"non-finite checkpoint array: {key}")
    identity_entries = [
        {key: entry[key] for key in ("key", "shape", "dtype", "sha256")}
        for entry in entries
    ]
    return {
        "family": family,
        "entry_count": len(entries),
        "restore_state_sha256": canonical_json_sha256(identity_entries),
        "critical_arrays": {
            key.removeprefix(family + "__"): by_key[key]
            for key in expected_shapes
        },
        "schedule_indices": {"low": low, "high": high},
    }


def verify_sequential_checkpoint(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    checkpoint_spec = spec["sequential_checkpoint"]
    file_record = _verify_file(path, checkpoint_spec["file"], "sequential checkpoint")
    source_spec = checkpoint_spec["source_snapshot"]
    source_archive = verify_source_archive(
        path.parent / source_spec["file"], source_spec, "sequential source snapshot"
    )
    with np.load(path, allow_pickle=False) as values:
        arms = {family: _restore_state_record(values, family) for family in ARM_FAMILIES}
        for family, record in arms.items():
            expected = checkpoint_spec["arms"][family]
            if record["restore_state_sha256"] != expected["restore_state_sha256"]:
                raise ValueError(f"{family} restored-state identity mismatch")
            for name, expected_array in expected["critical_arrays"].items():
                actual = record["critical_arrays"].get(name)
                if actual is None or actual["sha256"] != expected_array["sha256"]:
                    raise ValueError(f"{family} critical-array mismatch: {name}")
        for suffix in ("head__weight", "head__bias", "head__projection", "head__theta"):
            if not np.array_equal(values["quantum__" + suffix], values["classical__" + suffix]):
                raise ValueError(f"matched-control shared state differs: {suffix}")
        if np.array_equal(values["quantum__final_theta"], values["classical__final_theta"]):
            raise ValueError("quantum and classical final theta unexpectedly identical")
    return {
        "run_id": checkpoint_spec["run_id"],
        "exchange_path": checkpoint_spec["exchange_path"],
        "file": file_record,
        "source_snapshot": source_archive,
        "source_config_sha256": checkpoint_spec["source_config_sha256"],
        "arms": arms,
        "matched_shared_state_verified": True,
        "distinct_final_theta_verified": True,
    }


def verify_method_sources(root: Path, spec: Mapping[str, Any]) -> dict[str, str]:
    root = Path(root).resolve()
    verified: dict[str, str] = {}
    for relative, expected in sorted(spec["method_source_hashes"].items()):
        path = (root / relative).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"invalid method source: {relative}")
        actual = file_sha256(path)
        if actual != _require_sha256(expected, relative):
            raise ValueError(f"method source drift: {relative}")
        verified[relative] = actual
    return verified


def build_scoring_manifest(
    spec: Mapping[str, Any],
    *,
    root: Path,
    cohort_dir: Path,
    sequential_values: Path,
    reader_snapshot: Path,
    source_snapshot: Mapping[str, Any],
    git_revision: str,
) -> dict[str, Any]:
    if spec.get("schema_version") != "rag.reader_larger_untouched_scoring_spec.v1":
        raise ValueError("unexpected scoring-spec schema")
    if not re.fullmatch(r"[0-9a-f]{40}", git_revision):
        raise ValueError("scoring source Git revision must be a full commit")
    manifest = {
        "schema_version": SCHEMA,
        "authorization": "frozen_before_scoring",
        "scoring_started": False,
        "cohort": verify_cohort(cohort_dir, spec),
        "reader": verify_reader_snapshot(reader_snapshot, spec),
        "sequential_checkpoint": verify_sequential_checkpoint(sequential_values, spec),
        "arms": spec["arms"],
        "head": spec["head"],
        "selection": spec["selection"],
        "transport_gate": spec["transport_gate"],
        "label_boundary": spec["label_boundary"],
        "provenance": {
            "git_revision": git_revision,
            "method_source_hashes": verify_method_sources(root, spec),
            "scoring_manifest_source_snapshot": dict(source_snapshot),
            "cohort_source_snapshot": spec["cohort"]["source_snapshot"],
        },
        "claim_ceiling": "input_binding_only_no_scores",
    }
    manifest["manifest_identity_sha256"] = canonical_json_sha256(manifest)
    validate_scoring_manifest(manifest)
    return manifest


def validate_scoring_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA or manifest.get("authorization") != "frozen_before_scoring":
        raise ValueError("unexpected scoring manifest")
    if manifest.get("scoring_started") is not False:
        raise ValueError("manifest must be frozen before scoring")
    if manifest.get("claim_ceiling") != "input_binding_only_no_scores":
        raise ValueError("manifest claim ceiling drift")
    cohort = manifest.get("cohort", {})
    if cohort.get("cases") != 256 or cohort.get("top50_rows") != 12800 or cohort.get("lock_samples_replayed") != 256:
        raise ValueError("manifest population mismatch")
    boundary = manifest.get("label_boundary")
    if not isinstance(boundary, Mapping) or not boundary or any(value is not False for value in boundary.values()):
        raise ValueError("manifest label boundary is not closed")
    arms = manifest.get("arms")
    if list(arms or []) != ["identity_reader", "quantum_sequential_final", "classical_sequential_final"]:
        raise ValueError("manifest arm order drift")
    if manifest.get("selection") != {"candidate_count": 50, "selected_count": 5, "tie_break": "score_descending_then_candidate_index_ascending"}:
        raise ValueError("manifest selection rule drift")
    identity = manifest.get("manifest_identity_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_identity_sha256", None)
    if identity != canonical_json_sha256(unsigned):
        raise ValueError("manifest identity mismatch")
    for label, value in (
        ("cohort", cohort.get("cohort_identity_sha256")),
        ("manifest", identity),
        ("checkpoint", manifest.get("sequential_checkpoint", {}).get("file", {}).get("sha256")),
    ):
        _require_sha256(value, label)
    return {
        "status": "passed",
        "manifest_identity_sha256": identity,
        "cases": cohort["cases"],
        "top50_rows": cohort["top50_rows"],
        "arms": len(arms),
        "scoring_started": False,
    }


__all__ = [
    "array_record",
    "build_scoring_manifest",
    "canonical_json_sha256",
    "file_sha256",
    "validate_scoring_manifest",
    "verify_cohort",
    "verify_method_sources",
    "verify_reader_snapshot",
    "verify_sequential_checkpoint",
    "verify_source_archive",
]
