"""Strict observation-only alignment from official DPR positives to Wiki-DPR.

The module knows nothing about retrieval, selection, prompts, generation, or
evaluation.  It resolves only a three-link evidence chain: a normalized
question identity, an official DPR positive context, and a uniquely verified
Wiki-DPR passage identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

from applications.rag.gold_evidence_alignment import (
    GoldAlignmentError,
    find_forbidden,
    normalize_text,
    normalize_title,
    passage_identity,
)


DPR_MAPPING_STATUSES = (
    "mapped",
    "partial_wiki_dpr_identity",
    "no_question_join",
    "ambiguous_question_join",
    "no_positive_context",
    "no_wiki_dpr_identity",
    "ambiguous_wiki_dpr_identity",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def normalized_question_identity(question: Any) -> str:
    """Return the deterministic exact question-join key."""
    return normalize_text(question)


def question_fingerprint(question: Any) -> str:
    return hashlib.sha256(normalized_question_identity(question).encode("utf-8")).hexdigest()


def passage_text_fingerprint(title: Any, text: Any) -> str:
    """Hash the normalized title and complete passage text without retaining it."""
    payload = normalize_title(title) + "\0" + normalize_text(text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_text_fingerprint(text: Any) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OfficialPositiveContext:
    title: str
    text: str
    source_passage_id: str | None

    @property
    def title_text_fingerprint(self) -> str:
        return passage_text_fingerprint(self.title, self.text)

    @property
    def text_fingerprint(self) -> str:
        return exact_text_fingerprint(self.text)


@dataclass(frozen=True)
class DprQuestionEvidence:
    question_identity: str
    positives: tuple[OfficialPositiveContext, ...]
    duplicate_record_count: int
    ambiguous_records: bool


def _record_question(record: Mapping[str, Any]) -> str:
    value = record.get("question") or record.get("query") or record.get("question_text")
    if isinstance(value, Mapping):
        value = value.get("text") or value.get("question")
    return normalized_question_identity(value)


def _positive_contexts(record: Mapping[str, Any]) -> tuple[OfficialPositiveContext, ...]:
    values = record.get("positive_ctxs") or record.get("positive_contexts") or ()
    deduplicated: dict[tuple[str, str | None], OfficialPositiveContext] = {}
    for raw in _sequence(values):
        context = _mapping(raw)
        title = str(context.get("title") or "")
        text = str(context.get("text") or context.get("passage") or "")
        if not normalize_title(title) or not normalize_text(text):
            continue
        source_id = (
            context.get("passage_id")
            or context.get("psg_id")
            or context.get("id")
            or context.get("docid")
        )
        context_value = OfficialPositiveContext(
            title=title,
            text=text,
            source_passage_id=str(source_id) if source_id is not None and str(source_id) else None,
        )
        deduplicated.setdefault(
            (context_value.title_text_fingerprint, context_value.source_passage_id),
            context_value,
        )
    return tuple(deduplicated[key] for key in sorted(deduplicated))


def build_dpr_question_index(records: Iterable[Mapping[str, Any]]) -> dict[str, DprQuestionEvidence]:
    """Index official DPR records, retaining duplicate conflicts as unresolved."""
    grouped: dict[str, list[tuple[OfficialPositiveContext, ...]]] = {}
    for raw in records:
        record = _mapping(raw)
        key = _record_question(record)
        if key:
            grouped.setdefault(key, []).append(_positive_contexts(record))

    index: dict[str, DprQuestionEvidence] = {}
    for key, records_for_question in grouped.items():
        signatures = {
            tuple((context.title_text_fingerprint, context.source_passage_id) for context in contexts)
            for contexts in records_for_question
        }
        index[key] = DprQuestionEvidence(
            question_identity=key,
            positives=records_for_question[0] if len(signatures) == 1 else (),
            duplicate_record_count=len(records_for_question),
            ambiguous_records=len(signatures) > 1,
        )
    return index


def evidence_for_question(
    question: Any, index: Mapping[str, DprQuestionEvidence]
) -> tuple[str, DprQuestionEvidence | None]:
    evidence = index.get(normalized_question_identity(question))
    if evidence is None:
        return "no_question_join", None
    if evidence.ambiguous_records:
        return "ambiguous_question_join", evidence
    if not evidence.positives:
        return "no_positive_context", evidence
    return "joined", evidence


def align_dpr_positives_to_wiki(
    corpus_rows: Iterable[Mapping[str, Any]],
    joins_by_question: Mapping[str, tuple[str, DprQuestionEvidence | None]],
    *,
    progress_every: int = 1_000_000,
) -> dict[str, dict[str, Any]]:
    """Resolve official positives by corpus ID or exact passage identity only."""
    positives_by_id: dict[str, set[tuple[str, int]]] = {}
    positives_by_title_text: dict[str, set[tuple[str, int]]] = {}
    positives_by_text: dict[str, set[tuple[str, int]]] = {}
    for question_id, (join_status, evidence) in joins_by_question.items():
        if join_status != "joined" or evidence is None:
            continue
        for positive_index, positive in enumerate(evidence.positives):
            positive_key = (str(question_id), positive_index)
            if positive.source_passage_id:
                positives_by_id.setdefault(positive.source_passage_id, set()).add(positive_key)
            positives_by_title_text.setdefault(positive.title_text_fingerprint, set()).add(positive_key)
            positives_by_text.setdefault(positive.text_fingerprint, set()).add(positive_key)

    candidates: dict[tuple[str, int], dict[str, set[str]]] = {}
    for question_id, (join_status, evidence) in joins_by_question.items():
        if join_status != "joined" or evidence is None:
            continue
        for positive_index, _ in enumerate(evidence.positives):
            candidates[(str(question_id), positive_index)] = {
                "id": set(), "title_text": set(), "text": set(),
            }
    for row_index, raw in enumerate(corpus_rows):
        row = _mapping(raw)
        row_passage_id = row.get("id") or row.get("passage_id") or row.get("psg_id")
        wiki_id = passage_identity(
            row_passage_id, row.get("title"), row.get("text"), row_index
        )
        raw_id = str(row_passage_id) if row_passage_id is not None else ""
        for positive_key in positives_by_id.get(raw_id, ()):
            candidates[positive_key]["id"].add(wiki_id)
        for positive_key in positives_by_title_text.get(
            passage_text_fingerprint(row.get("title"), row.get("text")), ()
        ):
            candidates[positive_key]["title_text"].add(wiki_id)
        for positive_key in positives_by_text.get(exact_text_fingerprint(row.get("text")), ()):
            candidates[positive_key]["text"].add(wiki_id)
        if progress_every and (row_index + 1) % progress_every == 0:
            print(f"  DPR positive alignment scan: {row_index + 1} corpus rows")

    aligned: dict[str, dict[str, Any]] = {}
    for question_id, (join_status, evidence) in joins_by_question.items():
        if join_status != "joined" or evidence is None:
            aligned[str(question_id)] = {
                "mapping_status": join_status,
                "verified": set(),
                "identity_method_counts": {"official_passage_id": 0, "title_text_hash": 0, "unique_text_hash": 0},
                "official_positive_count": 0 if evidence is None else len(evidence.positives),
                "verified_positive_count": 0,
                "unresolved_positive_count": 0,
                "ambiguous_positive_count": 0,
            }
            continue
        verified: set[str] = set()
        method_counts = {"official_passage_id": 0, "title_text_hash": 0, "unique_text_hash": 0}
        unresolved = 0
        ambiguous = 0
        verified_positive_count = 0
        for positive_index, positive in enumerate(evidence.positives):
            positive_candidates = candidates[(str(question_id), positive_index)]
            matches = set()
            method = None
            if positive.source_passage_id:
                by_id = positive_candidates["id"]
                if len(by_id) == 1:
                    matches = by_id
                    method = "official_passage_id"
            if not matches:
                by_title_text = positive_candidates["title_text"]
                if len(by_title_text) == 1:
                    matches = by_title_text
                    method = "title_text_hash"
            if not matches:
                by_text = positive_candidates["text"]
                if len(by_text) == 1:
                    matches = by_text
                    method = "unique_text_hash"
            if matches and method:
                verified.update(matches)
                method_counts[method] += 1
                verified_positive_count += 1
            elif any(positive_candidates[kind] for kind in ("id", "title_text", "text")):
                ambiguous += 1
            else:
                unresolved += 1
        if verified_positive_count == len(evidence.positives):
            status = "mapped"
        elif verified_positive_count:
            status = "partial_wiki_dpr_identity"
        else:
            status = (
                "ambiguous_wiki_dpr_identity"
                if ambiguous
                else "no_wiki_dpr_identity"
            )
        aligned[str(question_id)] = {
            "mapping_status": status,
            "verified": verified,
            "identity_method_counts": method_counts,
            "official_positive_count": len(evidence.positives),
            "verified_positive_count": verified_positive_count,
            "unresolved_positive_count": unresolved,
            "ambiguous_positive_count": ambiguous,
        }
    return aligned


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarize_dpr_positive_alignment(
    samples: Sequence[Mapping[str, Any]], *, min_mapping_rate: float = 0.8
) -> dict[str, Any]:
    if not samples:
        raise GoldAlignmentError("DPR positive alignment audit has no samples")
    mapped = [row for row in samples if row.get("mapping_status") == "mapped"]
    retrieved = [row for row in mapped if bool(row.get("retrieval_hit"))]
    selected = [row for row in mapped if bool(row.get("selected_hit"))]
    official_total = sum(int(row.get("official_positive_count", 0)) for row in samples)
    verified_total = sum(int(row.get("verified_positive_count", 0)) for row in samples)
    mapping_rate = len(mapped) / len(samples)
    gate_pass = mapping_rate >= float(min_mapping_rate)
    return {
        "schema_version": 2,
        "sample_count": len(samples),
        "mapped_count": len(mapped),
        "mapping_rate": mapping_rate,
        "mapping_status_counts": {
            status: sum(row.get("mapping_status") == status for row in samples)
            for status in DPR_MAPPING_STATUSES
        },
        "identity_traceability": {
            "question_identity_rule": "normalized_question_exact",
            "official_passage_identity_rule": "official_id_or_normalized_title_full_text_sha256",
            "wiki_dpr_identity_rule": "unique_exact_corpus_id_or_full_text_identity",
            "official_positive_count": official_total,
            "verified_positive_count": verified_total,
            "verified_positive_rate": _rate(verified_total, official_total),
        },
        "strict_alignment": {
            "retrieval_top50_hit_rate_among_mapped": _rate(len(retrieved), len(mapped)),
            "selected_top5_hit_rate_among_mapped": _rate(len(selected), len(mapped)),
            "selected_top5_hit_rate_given_top50_gold": _rate(len(selected), len(retrieved)),
            "n_retrieval_hits": len(retrieved),
            "n_selected_hits": len(selected),
        },
        "decision": {
            "mapping_gate_pass": gate_pass,
            "minimum_mapping_rate": float(min_mapping_rate),
            "status": "ready_for_bottleneck_audit" if gate_pass else "mapping_incomplete",
            "interpretation": (
                "Use the explicit mapped and Top-50-conditional rates for bottleneck classification."
                if gate_pass else
                "Do not classify a retrieval, selector, or Generator bottleneck; repair label coverage first."
            ),
        },
    }


def validate_dpr_compact_result(payload: Mapping[str, Any]) -> None:
    forbidden_names = {
        "question", "query", "passages", "positive_ctxs",
        "positive_contexts", "answers", "prediction", "raw_prompt",
        "passage_text", "gold_text", "answer_text", "title", "text",
    }

    def find_dpr_forbidden(value: Any, path: str = "$root") -> list[str]:
        findings: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if str(key).lower() in forbidden_names:
                    findings.append(child_path)
                findings.extend(find_dpr_forbidden(child, child_path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                findings.extend(find_dpr_forbidden(child, f"{path}[{index}]"))
        return findings

    forbidden = sorted(set([*find_forbidden(payload), *find_dpr_forbidden(payload)]))
    if forbidden:
        raise GoldAlignmentError(f"compact result contains forbidden fields: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise GoldAlignmentError("compact result needs a non-empty samples list")
    seen: set[str] = set()
    for row in samples:
        if not isinstance(row, Mapping):
            raise GoldAlignmentError("sample must be an object")
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in seen:
            raise GoldAlignmentError("sample question IDs must be unique and non-empty")
        seen.add(question_id)
        if row.get("mapping_status") not in DPR_MAPPING_STATUSES:
            raise GoldAlignmentError(f"{question_id}: invalid DPR mapping status")
        if not isinstance(row.get("question_identity_verified"), bool):
            raise GoldAlignmentError(f"{question_id}: question_identity_verified must be boolean")
        for key in ("official_positive_count", "verified_positive_count", "unresolved_positive_count", "ambiguous_positive_count", "gold_passage_count", "top50_gold_count", "top5_gold_count"):
            value = row.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise GoldAlignmentError(f"{question_id}: {key} must be a non-negative integer")
        for key in ("retrieval_hit", "selected_hit"):
            if not isinstance(row.get(key), bool):
                raise GoldAlignmentError(f"{question_id}: {key} must be boolean")
        if row["top5_gold_count"] > row["top50_gold_count"]:
            raise GoldAlignmentError(f"{question_id}: Top-5 gold must be a subset of Top-50 gold")
        if row["top50_gold_count"] > row["gold_passage_count"]:
            raise GoldAlignmentError(f"{question_id}: Top-50 gold exceeds aligned gold passages")
        if row["selected_hit"] and not row["retrieval_hit"]:
            raise GoldAlignmentError(f"{question_id}: selected hit requires a retrieval hit")
        status = row.get("mapping_status")
        complete_chain = (
            row["question_identity_verified"]
            and row["official_positive_count"] > 0
            and row["verified_positive_count"] == row["official_positive_count"]
            and row["unresolved_positive_count"] == 0
            and row["ambiguous_positive_count"] == 0
            and row["gold_passage_count"] > 0
        )
        if status == "mapped" and not complete_chain:
            raise GoldAlignmentError(f"{question_id}: mapped sample lacks a complete identity chain")
        if status == "partial_wiki_dpr_identity" and not (
            row["question_identity_verified"]
            and 0 < row["verified_positive_count"] < row["official_positive_count"]
            and row["unresolved_positive_count"] + row["ambiguous_positive_count"] > 0
        ):
            raise GoldAlignmentError(f"{question_id}: invalid partial identity accounting")


__all__ = [
    "DPR_MAPPING_STATUSES", "DprQuestionEvidence", "OfficialPositiveContext",
    "align_dpr_positives_to_wiki", "build_dpr_question_index", "evidence_for_question",
    "exact_text_fingerprint", "normalized_question_identity", "passage_text_fingerprint",
    "question_fingerprint", "summarize_dpr_positive_alignment", "validate_dpr_compact_result",
]
