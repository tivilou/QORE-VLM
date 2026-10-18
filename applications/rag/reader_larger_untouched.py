"""Pure contracts for constructing a fresh, label-blind Top-50 cohort."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile


FORBIDDEN_COHORT_KEYS = frozenset({
    "answer", "answers", "gold", "gold_answer", "gold_answers", "silver",
    "support", "support_label", "prediction", "generation", "selected_by",
})


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_names(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.add(str(key).lower())
            found.update(key_names(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.update(key_names(child))
    return found


def exclusion_question_hashes(
    support_bundle: Path,
    historical_panel: Path,
    *,
    support_sha256: str,
    panel_sha256: str,
) -> tuple[set[str], dict[str, int]]:
    """Extract question identities only; never copy labels or answers."""
    if file_sha256(support_bundle) != support_sha256:
        raise ValueError("support-bundle hash mismatch")
    if file_sha256(historical_panel) != panel_sha256:
        raise ValueError("historical-panel hash mismatch")
    with zipfile.ZipFile(support_bundle) as archive:
        source = json.loads(archive.read("source.json"))
    support_cases = source.get("cases")
    if not isinstance(support_cases, list):
        raise ValueError("support bundle has no case list")
    support: set[str] = set()
    for row in support_cases:
        question = row.get("question") if isinstance(row, Mapping) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError("support bundle contains an invalid question")
        identity = sha256_text(question)
        if row.get("question_sha256") not in {None, identity}:
            raise ValueError("support-bundle question identity drift")
        support.add(identity)
    panel_document = json.loads(historical_panel.read_text(encoding="utf-8"))
    panel_cases = panel_document.get("cases")
    if not isinstance(panel_cases, list):
        raise ValueError("historical panel has no case list")
    panel: set[str] = set()
    for row in panel_cases:
        question = row.get("question") if isinstance(row, Mapping) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError("historical panel contains an invalid question")
        panel.add(sha256_text(question))
    union = support | panel
    return union, {
        "support_cases": len(support_cases),
        "support_unique": len(support),
        "historical_panel_cases": len(panel_cases),
        "historical_panel_unique": len(panel),
        "union_unique": len(union),
    }


def select_fresh_questions(
    rows: Sequence[Mapping[str, Any]],
    exclusions: Iterable[str],
    *,
    pool_start: int,
    pool_end: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply the frozen outcome-free hash ordering to one dataset interval."""
    if not (0 <= pool_start < pool_end <= len(rows)) or count < 1:
        raise ValueError("invalid fresh-cohort pool contract")
    excluded = set(exclusions)
    candidates: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for dataset_index in range(pool_start, pool_end):
        row = rows[dataset_index]
        question = row.get("question") if isinstance(row, Mapping) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"invalid question at dataset index {dataset_index}")
        question_sha = sha256_text(question)
        if question_sha in seen:
            raise ValueError("duplicate question identity in candidate pool")
        seen.add(question_sha)
        if question_sha in excluded:
            continue
        source_id = str(row.get("id", dataset_index))
        order_key = sha256_text(f"{seed}:{dataset_index}:{question_sha}")
        candidates.append((order_key, {
            "dataset_index": dataset_index,
            "source_id_sha256": sha256_text(source_id),
            "question_sha256": question_sha,
            "question": question,
        }))
    candidates.sort(key=lambda item: (item[0], item[1]["dataset_index"]))
    if len(candidates) < count:
        raise ValueError("not enough unexposed questions in frozen pool")
    return [item[1] for item in candidates[:count]]


def cohort_identity(cases: Sequence[Mapping[str, Any]]) -> str:
    identity = [{
        "dataset_index": int(case["dataset_index"]),
        "question_sha256": str(case["question_sha256"]),
        "candidate_id_sha256": [str(candidate["candidate_id_sha256"]) for candidate in case["candidates"]],
    } for case in cases]
    return sha256_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def validate_cohort(document: Mapping[str, Any], *, expected_cases: int) -> dict[str, Any]:
    if document.get("schema_version") != "rag.reader_larger_untouched_cohort.v1":
        raise ValueError("unexpected fresh-cohort schema")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError("fresh-cohort case count mismatch")
    if FORBIDDEN_COHORT_KEYS & key_names(document):
        raise ValueError("fresh cohort contains a forbidden target/result field")
    question_ids: list[str] = []
    for case_number, case in enumerate(cases, 1):
        question = case.get("question") if isinstance(case, Mapping) else None
        candidates = case.get("candidates") if isinstance(case, Mapping) else None
        if not isinstance(question, str) or sha256_text(question) != case.get("question_sha256"):
            raise ValueError(f"question identity mismatch at case {case_number}")
        if not isinstance(candidates, list) or len(candidates) != 50:
            raise ValueError(f"Top-50 coverage mismatch at case {case_number}")
        ranks = []
        candidate_ids = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("text"), str) or not candidate["text"].strip():
                raise ValueError(f"invalid candidate text at case {case_number}")
            candidate_id = str(candidate.get("id"))
            if sha256_text(candidate_id) != candidate.get("candidate_id_sha256"):
                raise ValueError(f"candidate identity mismatch at case {case_number}")
            ranks.append(int(candidate.get("retrieved_rank", -1)))
            if not math.isfinite(float(candidate.get("retrieval_score", float("nan")))):
                raise ValueError(f"non-finite retrieval score at case {case_number}")
            candidate_ids.append(candidate_id)
        if sorted(ranks) != list(range(1, 51)) or len(set(candidate_ids)) != 50:
            raise ValueError(f"candidate order/uniqueness mismatch at case {case_number}")
        question_ids.append(case["question_sha256"])
    if len(set(question_ids)) != expected_cases:
        raise ValueError("duplicate question identity in fresh cohort")
    identity = cohort_identity(cases)
    if document.get("cohort_identity_sha256") != identity:
        raise ValueError("fresh-cohort identity digest mismatch")
    return {"cases": expected_cases, "top50_rows": expected_cases * 50, "cohort_identity_sha256": identity}
