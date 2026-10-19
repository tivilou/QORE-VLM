"""Pure contracts for the complete untouched NQ/Wiki-DPR population freeze.

This module never loads a model or a dataset.  It owns only exact identities,
exposure exclusion, deterministic split assignment and artifact replay.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

from applications.rag.reader_larger_untouched import (
    FORBIDDEN_COHORT_KEYS,
    file_sha256,
    key_names,
    sha256_text,
    validate_cohort as validate_spent_cohort,
)


COHORT_SCHEMA = "rag.reader_full_data_replication_cohort.v1"
TRACE_SCHEMA = "sample-trace.v2"
LOCK_TYPE = "reader_full_data_population_lock"
EXPOSURE_SOURCES = ("protected434", "historical100", "spent256")
PRE_SCORE_STAGES = ("data", "preprocess", "retrieval", "diagnosis")
ROOT = Path(__file__).resolve().parents[2]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verified_file(path: Path, spec: Mapping[str, Any], label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or invalid {label}: {path}")
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"{label} byte-size mismatch")
    if file_sha256(path) != str(spec["sha256"]):
        raise ValueError(f"{label} SHA-256 mismatch")


def _question_identity(row: Mapping[str, Any], label: str) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{label} contains an invalid question")
    identity = sha256_text(question)
    declared = row.get("question_sha256")
    if declared not in {None, identity}:
        raise ValueError(f"{label} question identity drift")
    return identity


def load_exposure_registry(
    support_bundle: Path,
    historical_panel: Path,
    spent_cohort: Path,
    registry_spec: Mapping[str, Any],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Load only question identities from three immutable exposure sources."""
    if not isinstance(registry_spec.get("union_rule"), str) or not registry_spec["union_rule"]:
        raise ValueError("exposure registry contract is missing")
    protected_spec = registry_spec["protected434"]
    historical_spec = registry_spec["historical100"]
    spent_spec = registry_spec["spent256"]
    _verified_file(support_bundle, protected_spec, "protected434")
    _verified_file(historical_panel, historical_spec, "historical100")
    _verified_file(spent_cohort, spent_spec, "spent256")

    with zipfile.ZipFile(support_bundle) as archive:
        try:
            support_document = json.loads(archive.read("source.json"))
        except KeyError as exc:
            raise ValueError("protected434 has no source.json") from exc
    support_cases = support_document.get("cases")
    if not isinstance(support_cases, list):
        raise ValueError("protected434 has no case list")
    protected = {_question_identity(row, "protected434") for row in support_cases}

    historical_document = json.loads(historical_panel.read_text(encoding="utf-8"))
    historical_cases = historical_document.get("cases")
    if not isinstance(historical_cases, list):
        raise ValueError("historical100 has no case list")
    historical = {_question_identity(row, "historical100") for row in historical_cases}

    spent_document = json.loads(spent_cohort.read_text(encoding="utf-8"))
    spent_cases = int(spent_spec["expected_question_identities"])
    checks = validate_spent_cohort(spent_document, expected_cases=spent_cases)
    if checks["cohort_identity_sha256"] != spent_spec["cohort_identity_sha256"]:
        raise ValueError("spent256 cohort identity mismatch")
    spent = {_question_identity(row, "spent256") for row in spent_document["cases"]}

    sources = {"protected434": protected, "historical100": historical, "spent256": spent}
    expected_counts = {
        name: int(registry_spec[name]["expected_question_identities"])
        for name in EXPOSURE_SOURCES
    }
    observed_counts = {name: len(sources[name]) for name in EXPOSURE_SOURCES}
    if observed_counts != expected_counts:
        raise ValueError("exposure source identity count drift")
    union = set().union(*sources.values())
    if len(union) != int(registry_spec["expected_union_identities"]):
        raise ValueError("exposure registry union drift")
    return sources, {
        "source_identity_counts": observed_counts,
        "union_identities": len(union),
        "registry_identity_sha256": canonical_json_sha256({
            name: sorted(sources[name]) for name in EXPOSURE_SOURCES
        }),
    }


def freeze_complete_population(
    rows: Sequence[Mapping[str, Any]],
    exposure_sources: Mapping[str, set[str]],
    *,
    pool_start: int,
    pool_end: int,
    split_seed: int,
    count_bounds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return every eligible identity, deterministically partitioned into A/B."""
    if len(rows) != pool_end or not (0 <= pool_start < pool_end):
        raise ValueError("complete source interval coverage mismatch")
    if set(exposure_sources) != set(EXPOSURE_SOURCES):
        raise ValueError("exposure source allowlist drift")
    lower, upper = map(int, count_bounds)
    if not (0 < lower <= upper <= pool_end - pool_start):
        raise ValueError("invalid realized-count bounds")

    union = set().union(*(set(exposure_sources[name]) for name in EXPOSURE_SOURCES))
    in_pool_by_source = {name: 0 for name in EXPOSURE_SOURCES}
    excluded_union_in_pool: set[str] = set()
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset_index in range(pool_start, pool_end):
        row = rows[dataset_index]
        if not isinstance(row, Mapping):
            raise ValueError(f"invalid source row at dataset index {dataset_index}")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"invalid question at dataset index {dataset_index}")
        question_sha = sha256_text(question)
        if question_sha in seen:
            raise ValueError("duplicate question identity in complete source interval")
        seen.add(question_sha)
        for name in EXPOSURE_SOURCES:
            if question_sha in exposure_sources[name]:
                in_pool_by_source[name] += 1
        if question_sha in union:
            excluded_union_in_pool.add(question_sha)
            continue
        source_id = str(row.get("id", dataset_index))
        split_key = sha256_text(f"{split_seed}:{dataset_index}:{question_sha}")
        eligible.append({
            "dataset_index": dataset_index,
            "source_id_sha256": sha256_text(source_id),
            "question_sha256": question_sha,
            "question": question,
            "split_order_sha256": split_key,
        })

    eligible.sort(key=lambda row: (row["split_order_sha256"], row["dataset_index"]))
    if not (lower <= len(eligible) <= upper):
        raise ValueError("realized complete-population count is outside frozen bounds")
    for position, row in enumerate(eligible):
        row["population_position"] = position + 1
        row["confirmation_split"] = "split_A" if position % 2 == 0 else "split_B"

    split_counts = {
        name: sum(row["confirmation_split"] == name for row in eligible)
        for name in ("split_A", "split_B")
    }
    if abs(split_counts["split_A"] - split_counts["split_B"]) > 1:
        raise ValueError("confirmation splits are not balanced")
    if len(eligible) + len(excluded_union_in_pool) != pool_end - pool_start:
        raise ValueError("complete source interval is not fully accounted")

    population_projection = [{
        "dataset_index": row["dataset_index"],
        "question_sha256": row["question_sha256"],
        "confirmation_split": row["confirmation_split"],
    } for row in eligible]
    split_identities = {
        name: canonical_json_sha256([
            row for row in population_projection if row["confirmation_split"] == name
        ]) for name in ("split_A", "split_B")
    }
    audit = {
        "scanned_positions": pool_end - pool_start,
        "pool_start_inclusive": pool_start,
        "pool_end_exclusive": pool_end,
        "source_identity_counts": {
            name: len(exposure_sources[name]) for name in EXPOSURE_SOURCES
        },
        "source_in_pool_counts": in_pool_by_source,
        "excluded_union_in_pool": len(excluded_union_in_pool),
        "eligible_cases": len(eligible),
        "accounted_positions": len(eligible) + len(excluded_union_in_pool),
        "split_counts": split_counts,
        "population_identity_sha256": canonical_json_sha256(population_projection),
        "split_identity_sha256": split_identities,
    }
    return eligible, audit


def cohort_identity(cases: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256([{
        "dataset_index": int(case["dataset_index"]),
        "question_sha256": str(case["question_sha256"]),
        "candidate_id_sha256": [
            str(candidate["candidate_id_sha256"]) for candidate in case["candidates"]
        ],
    } for case in cases])


def split_cohort_identities(cases: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        split: canonical_json_sha256([{
            "dataset_index": int(case["dataset_index"]),
            "question_sha256": str(case["question_sha256"]),
            "candidate_id_sha256": [
                str(candidate["candidate_id_sha256"]) for candidate in case["candidates"]
            ],
        } for case in cases if case["confirmation_split"] == split])
        for split in ("split_A", "split_B")
    }


def validate_full_cohort(
    document: Mapping[str, Any], *, count_bounds: Sequence[int]
) -> dict[str, Any]:
    if document.get("schema_version") != COHORT_SCHEMA:
        raise ValueError("unexpected complete-cohort schema")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("complete cohort has no case list")
    lower, upper = map(int, count_bounds)
    if not (lower <= len(cases) <= upper):
        raise ValueError("complete cohort case count is outside frozen bounds")
    if FORBIDDEN_COHORT_KEYS & key_names(document):
        raise ValueError("complete cohort contains a forbidden target/result field")
    boundary = document.get("label_boundary")
    if not isinstance(boundary, Mapping) or not boundary or any(value is not False for value in boundary.values()):
        raise ValueError("complete cohort crossed the label boundary")

    question_ids: list[str] = []
    positions: list[int] = []
    split_counts = {"split_A": 0, "split_B": 0}
    for case_number, case in enumerate(cases, 1):
        question = case.get("question") if isinstance(case, Mapping) else None
        candidates = case.get("candidates") if isinstance(case, Mapping) else None
        if not isinstance(question, str) or sha256_text(question) != case.get("question_sha256"):
            raise ValueError(f"question identity mismatch at case {case_number}")
        if int(case.get("population_position", -1)) != case_number:
            raise ValueError(f"population position mismatch at case {case_number}")
        split = case.get("confirmation_split")
        if split not in split_counts:
            raise ValueError(f"invalid confirmation split at case {case_number}")
        expected_split = "split_A" if (case_number - 1) % 2 == 0 else "split_B"
        if split != expected_split:
            raise ValueError(f"confirmation split order drift at case {case_number}")
        if not isinstance(candidates, list) or len(candidates) != 50:
            raise ValueError(f"Top-50 coverage mismatch at case {case_number}")
        ranks: list[int] = []
        candidate_ids: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError(f"invalid candidate at case {case_number}")
            candidate_id = str(candidate.get("id"))
            text = candidate.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"empty candidate text at case {case_number}")
            if sha256_text(candidate_id) != candidate.get("candidate_id_sha256"):
                raise ValueError(f"candidate identity mismatch at case {case_number}")
            ranks.append(int(candidate.get("retrieved_rank", -1)))
            if not math.isfinite(float(candidate.get("retrieval_score", float("nan")))):
                raise ValueError(f"non-finite retrieval score at case {case_number}")
            candidate_ids.append(candidate_id)
        if ranks != list(range(1, 51)) or len(set(candidate_ids)) != 50:
            raise ValueError(f"candidate order/uniqueness mismatch at case {case_number}")
        question_ids.append(case["question_sha256"])
        positions.append(int(case["dataset_index"]))
        split_counts[split] += 1

    if len(set(question_ids)) != len(cases) or len(set(positions)) != len(cases):
        raise ValueError("duplicate question or dataset position in complete cohort")
    if abs(split_counts["split_A"] - split_counts["split_B"]) > 1:
        raise ValueError("confirmation split imbalance")
    identity = cohort_identity(cases)
    split_identities = split_cohort_identities(cases)
    if document.get("cohort_identity_sha256") != identity:
        raise ValueError("complete cohort identity digest mismatch")
    if document.get("split_cohort_identity_sha256") != split_identities:
        raise ValueError("split cohort identity digest mismatch")
    population = document.get("population_audit")
    if not isinstance(population, Mapping):
        raise ValueError("missing population audit")
    if int(population.get("eligible_cases", -1)) != len(cases):
        raise ValueError("population/cohort count mismatch")
    if population.get("split_counts") != split_counts:
        raise ValueError("population/cohort split mismatch")
    if int(population.get("accounted_positions", -1)) != int(population.get("scanned_positions", -2)):
        raise ValueError("population interval accounting mismatch")
    return {
        "status": "passed",
        "cases": len(cases),
        "top50_rows": len(cases) * 50,
        "split_counts": split_counts,
        "cohort_identity_sha256": identity,
        "split_cohort_identity_sha256": split_identities,
        "target_boundary_verified": True,
    }


def validate_population_artifacts(run_dir: Path, *, count_bounds: Sequence[int]) -> dict[str, Any]:
    """Project-owned structural, semantic and deterministic replay validator."""
    run_dir = Path(run_dir)
    cohort_path = run_dir / "cohort.json"
    lock_path = run_dir / "full_population_lock.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "upload_manifest.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = validate_full_cohort(cohort, count_bounds=count_bounds)
    formal = cohort.get("formal_population", {})
    if (
        formal.get("dataset") != "nq_open"
        or formal.get("split") != "validation"
        or formal.get("pool_start_inclusive") != 512
        or formal.get("pool_end_exclusive") != 3610
        or formal.get("realized_count_bounds") != [2308, 2742]
    ):
        raise ValueError("formal population contract drift")
    registry = cohort.get("exposure_registry_audit", {})
    expected_source_counts = {"protected434": 434, "historical100": 100, "spent256": 256}
    if registry.get("source_identity_counts") != expected_source_counts or registry.get("union_identities") != 790:
        raise ValueError("exposure registry audit drift")
    population_audit = cohort["population_audit"]
    if population_audit.get("source_identity_counts") != expected_source_counts:
        raise ValueError("population exposure-source count drift")
    projection = [{
        "dataset_index": int(case["dataset_index"]),
        "question_sha256": case["question_sha256"],
        "confirmation_split": case["confirmation_split"],
    } for case in cohort["cases"]]
    if population_audit.get("population_identity_sha256") != canonical_json_sha256(projection):
        raise ValueError("population identity replay mismatch")
    seed = int(cohort.get("independent_confirmations", {}).get("split_seed", -1))
    order = []
    for case in cohort["cases"]:
        expected_key = sha256_text(f"{seed}:{case['dataset_index']}:{case['question_sha256']}")
        if case.get("split_order_sha256") != expected_key:
            raise ValueError("split order key replay mismatch")
        order.append((expected_key, int(case["dataset_index"])))
    if order != sorted(order):
        raise ValueError("split order is not canonical")
    if lock.get("schema_version") != TRACE_SCHEMA or lock.get("artifact_type") != LOCK_TYPE:
        raise ValueError("unexpected full-population lock schema")
    if lock.get("run_id") != cohort.get("run_id") or summary.get("run_id") != cohort.get("run_id"):
        raise ValueError("run identity mismatch")
    if lock.get("stages") != list(PRE_SCORE_STAGES):
        raise ValueError("pre-score stage coverage mismatch")
    if lock.get("cohort_sha256") != file_sha256(cohort_path):
        raise ValueError("population lock cohort hash mismatch")
    if lock.get("cohort_identity_sha256") != checks["cohort_identity_sha256"]:
        raise ValueError("population lock cohort identity mismatch")
    if lock.get("split_cohort_identity_sha256") != checks["split_cohort_identity_sha256"]:
        raise ValueError("population lock split identity mismatch")
    if lock.get("label_boundary") != cohort.get("label_boundary"):
        raise ValueError("population lock label boundary mismatch")
    samples = lock.get("samples")
    if not isinstance(samples, list) or len(samples) != checks["cases"]:
        raise ValueError("population lock sample coverage mismatch")
    for number, (sample, case) in enumerate(zip(samples, cohort["cases"]), 1):
        expected = {
            "population_position": number,
            "dataset_index": case["dataset_index"],
            "question_sha256": case["question_sha256"],
            "confirmation_split": case["confirmation_split"],
            "candidate_id_sha256": [row["candidate_id_sha256"] for row in case["candidates"]],
            "candidate_retrieved_ranks": list(range(1, 51)),
        }
        if sample != expected:
            raise ValueError(f"population lock sample drift at case {number}")
    entries = manifest.get("exchange_files")
    if not isinstance(entries, list):
        raise ValueError("upload manifest has no exchange file list")
    for entry in entries:
        path = run_dir / str(entry["name"])
        if path.stat().st_size != int(entry["bytes"]) or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"upload manifest file drift: {entry['name']}")
    if summary.get("cases") != checks["cases"] or summary.get("top50_rows") != checks["top50_rows"]:
        raise ValueError("compact summary population mismatch")
    source_snapshot = summary.get("source_snapshot", {})
    source_path = run_dir / str(source_snapshot.get("file", ""))
    if (
        not source_path.is_file()
        or source_path.stat().st_size != int(source_snapshot.get("bytes", -1))
        or file_sha256(source_path) != source_snapshot.get("sha256")
    ):
        raise ValueError("source snapshot outer identity mismatch")
    source_hashes = source_snapshot.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("source snapshot has no member identities")
    with zipfile.ZipFile(source_path) as archive:
        embedded = json.loads(archive.read("source_manifest.json"))
        if embedded.get("source_hashes") != source_hashes:
            raise ValueError("source snapshot embedded manifest mismatch")
        for relative, digest in source_hashes.items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                raise ValueError(f"source snapshot member drift: {relative}")
    provenance = lock.get("provenance", {})
    expected_config = ROOT / "configs/experiments/reader_full_data_replication.json"
    expected_plan = ROOT / "configs/experiments/reader_full_data_replication_plan.json"
    if provenance.get("config_sha256") != file_sha256(expected_config):
        raise ValueError("population lock config identity mismatch")
    if provenance.get("plan_sha256") != file_sha256(expected_plan):
        raise ValueError("population lock plan identity mismatch")
    return {
        "status": "passed",
        "structural_validation": "sample-trace.v2 project-owned pre-score profile passed",
        "semantic_validation": "complete interval accounting, exact exclusions, balanced disjoint split union, target boundary and all Top50 coordinates passed",
        "deterministic_replay": "cohort, split, trace sample, manifest and file hashes replayed",
        **checks,
        "lock_samples_replayed": len(samples),
        "manifest_files_replayed": len(entries),
        "source_members_replayed": len(source_hashes),
    }


__all__ = [
    "COHORT_SCHEMA",
    "EXPOSURE_SOURCES",
    "LOCK_TYPE",
    "PRE_SCORE_STAGES",
    "TRACE_SCHEMA",
    "canonical_json_sha256",
    "cohort_identity",
    "freeze_complete_population",
    "load_exposure_registry",
    "split_cohort_identities",
    "validate_full_cohort",
    "validate_population_artifacts",
]
