"""Gold-evidence alignment utilities for Natural Questions and Wiki-DPR.

The normal ``nq_open`` loader intentionally exposes only questions and answer
strings.  This module keeps the stricter source annotation path separate: it
extracts document titles and short-answer token spans from the original NQ
records, then aligns them to Wiki-DPR passages by title and answer text.  The
result is an audit signal only; it must never be passed back to retrieval or
selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


FORBIDDEN_FIELDS = {
    "question", "passages", "gold_answers", "prediction", "raw_prompt",
    "passage_text", "gold_text", "answer_text",
}


class GoldAlignmentError(ValueError):
    """Raised when an alignment input violates the frozen audit contract."""


def normalize_text(value: Any) -> str:
    """Normalize text for token-boundary matching without changing IDs."""
    text = str(value or "").replace("\u00a0", " ").lower()
    return " ".join(text.split())


def normalize_title(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def answer_in_passage(answer: str, passage_text: str) -> bool:
    """Return true only for a normalized token-boundary answer occurrence."""
    answer_norm = normalize_text(answer)
    text_norm = normalize_text(passage_text)
    if not answer_norm or not text_norm:
        return False
    pattern = r"(?<!\w)" + re.escape(answer_norm) + r"(?!\w)"
    return re.search(pattern, text_norm) is not None


def passage_identity(raw_id: Any, title: Any, text: Any, row_index: int) -> str:
    """Canonical ID shared by retrieved rows and the full-corpus scan."""
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id)
    digest = hashlib.sha256(
        (normalize_title(title) + "\0" + normalize_text(text)).encode("utf-8")
    ).hexdigest()[:24]
    return f"sha256:{digest}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _question_text(record: Mapping[str, Any]) -> str:
    question = record.get("question")
    if isinstance(question, Mapping):
        question = _first(question, "text", "question")
    return str(question or record.get("question_text") or "")


def _document(record: Mapping[str, Any]) -> Mapping[str, Any]:
    document = record.get("document")
    return _mapping(document)


def _tokens(record: Mapping[str, Any]) -> Sequence[Any]:
    document = _document(record)
    tokens = document.get("tokens") or record.get("tokens") or []
    return tokens if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)) else []


def _token_text(token: Any) -> tuple[str, bool]:
    if isinstance(token, Mapping):
        value = _first(token, "token", "text", "value")
        is_html = bool(token.get("is_html", False))
    else:
        value, is_html = token, False
    return str(value or ""), is_html


def _span_text(tokens: Sequence[Any], start: Any, end: Any) -> str:
    try:
        start_i, end_i = int(start), int(end)
    except (TypeError, ValueError):
        return ""
    if start_i < 0 or end_i <= start_i or start_i >= len(tokens):
        return ""
    values = [_token_text(token)[0] for token in tokens[start_i:min(end_i, len(tokens))]
              if not _token_text(token)[1]]
    return normalize_text(" ".join(values))


def _span_values(value: Any, tokens: Sequence[Any]) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    explicit = _first(value, "text", "answer", "value")
    if explicit:
        return [normalize_text(explicit)]
    text = _span_text(tokens, _first(value, "start_token", "start"),
                      _first(value, "end_token", "end"))
    return [text] if text else []


@dataclass(frozen=True)
class GoldEvidence:
    """Source-derived evidence constraints for one NQ question."""

    question_text: str
    document_title: str
    strict_answers: tuple[str, ...]
    support_answers: tuple[str, ...]
    has_short_span: bool
    source: str

    @property
    def title_key(self) -> str:
        return normalize_title(self.document_title)


def extract_gold_evidence(
    record: Mapping[str, Any], fallback_answers: Sequence[str] = ()
) -> GoldEvidence:
    """Extract the first usable NQ annotation, preserving ambiguity explicitly."""
    document = _document(record)
    title = _first(document, "title", "document_title")
    title = title or _first(record, "document_title", "title") or ""
    tokens = _tokens(record)
    annotations = record.get("annotations") or record.get("annotation") or []
    if isinstance(annotations, Mapping):
        annotations = [annotations]
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
        annotations = []

    strict: list[str] = []
    has_short = False
    for annotation_raw in annotations:
        annotation = _mapping(annotation_raw)
        short_answers = annotation.get("short_answers") or annotation.get("short_answer") or []
        if isinstance(short_answers, Mapping):
            short_answers = [short_answers]
        if isinstance(short_answers, Sequence) and not isinstance(short_answers, (str, bytes)):
            for span in short_answers:
                values = _span_values(span, tokens)
                if values:
                    has_short = True
                    strict.extend(values)

    fallback = [normalize_text(value) for value in fallback_answers if normalize_text(value)]
    strict_unique = tuple(dict.fromkeys(value for value in strict if value))
    support_unique = tuple(dict.fromkeys([*strict_unique, *fallback]))
    source = "nq_annotations" if annotations else "answer_fallback"
    return GoldEvidence(
        question_text=_question_text(record),
        document_title=str(title),
        strict_answers=strict_unique,
        support_answers=support_unique,
        has_short_span=has_short,
        source=source,
    )


def match_passage(
    *, title: Any, text: Any, evidence: GoldEvidence
) -> tuple[bool, bool]:
    """Return ``(strict_match, support_match)`` for one corpus passage."""
    if not evidence.title_key or normalize_title(title) != evidence.title_key:
        return False, False
    strict = any(answer_in_passage(answer, str(text)) for answer in evidence.strict_answers)
    support = any(answer_in_passage(answer, str(text)) for answer in evidence.support_answers)
    return strict, support


def align_corpus(
    corpus_rows: Iterable[Mapping[str, Any]],
    evidence_by_question: Mapping[str, GoldEvidence],
    *,
    progress_every: int = 1_000_000,
) -> dict[str, dict[str, Any]]:
    """Scan Wiki-DPR once and return strict/support global passage IDs.

    Only IDs are retained.  Passage titles/text are never written to the
    compact result, which keeps the collaborator handoff privacy-safe.
    """
    wanted: dict[str, list[tuple[str, GoldEvidence]]] = {}
    for question_id, evidence in evidence_by_question.items():
        if evidence.title_key:
            wanted.setdefault(evidence.title_key, []).append((str(question_id), evidence))
    matches = {
        str(question_id): {"strict": set(), "support": set(), "title_seen": 0}
        for question_id in evidence_by_question
    }
    for row_index, row_raw in enumerate(corpus_rows):
        row = _mapping(row_raw)
        title = row.get("title", "")
        title_key = normalize_title(title)
        candidates = wanted.get(title_key)
        if candidates:
            text = row.get("text", "")
            row_id = passage_identity(row.get("id"), title, text, row_index)
            for question_id, evidence in candidates:
                matches[question_id]["title_seen"] = int(matches[question_id]["title_seen"]) + 1
                strict, support = match_passage(title=title, text=text, evidence=evidence)
                if strict:
                    matches[question_id]["strict"].add(row_id)
                if support:
                    matches[question_id]["support"].add(row_id)
        if progress_every and (row_index + 1) % progress_every == 0:
            print(f"  alignment scan: {row_index + 1} corpus rows")
    return matches


def retrieved_matches(
    retrieved_rows: Sequence[Mapping[str, Any]], evidence: GoldEvidence
) -> tuple[set[str], set[str]]:
    strict: set[str] = set()
    support: set[str] = set()
    for index, row_raw in enumerate(retrieved_rows):
        row = _mapping(row_raw)
        row_id = passage_identity(row.get("id"), row.get("title"), row.get("text"), index)
        strict_match, support_match = match_passage(
            title=row.get("title", ""), text=row.get("text", ""), evidence=evidence
        )
        if strict_match:
            strict.add(row_id)
        if support_match:
            support.add(row_id)
    return strict, support


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def summarize_alignment(samples: Sequence[Mapping[str, Any]], *, min_mapping_rate: float = 0.8) -> dict[str, Any]:
    """Summarize the retrieval -> selection alignment audit."""
    if not samples:
        raise GoldAlignmentError("alignment audit has no samples")
    mapped = [row for row in samples if row.get("mapping_status") == "mapped"]
    strict_top50 = [row for row in mapped if row.get("gold_passage_count", 0) > 0]
    retrieval_hits = [row for row in mapped if bool(row.get("retrieval_hit"))]
    selected_hits = [row for row in mapped if bool(row.get("selected_hit"))]
    def rate(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        return None if not rows else sum(bool(row.get(key)) for row in rows) / len(rows)
    retrieval_rate = rate(mapped, "retrieval_hit")
    selection_rate = rate(strict_top50, "selected_hit")
    conditional_recalls = [float(row["conditional_selection_recall"]) for row in strict_top50
                           if _finite(row.get("conditional_selection_recall"))]
    mapping_rate = len(mapped) / len(samples)
    if mapping_rate < 1.0 and mapped:
        status = "mapping_incomplete"
    elif not mapped:
        status = "mapping_failed"
    else:
        status = "ready_for_bottleneck_audit"
    return {
        "schema_version": 1,
        "sample_count": len(samples),
        "mapped_count": len(mapped),
        "mapping_rate": mapping_rate,
        "mapping_status_counts": {
            status_name: sum(row.get("mapping_status") == status_name for row in samples)
            for status_name in ("mapped", "support_only", "no_matching_title", "no_strict_match", "source_unavailable")
        },
        "strict_alignment": {
            "n_with_gold": len(strict_top50),
            "retrieval_top50_hit_rate": retrieval_rate,
            "retrieval_top50_failure_rate": None if retrieval_rate is None else 1.0 - retrieval_rate,
            "selected_top5_hit_rate_conditional": selection_rate,
            "selected_top5_failure_rate_conditional": None if selection_rate is None else 1.0 - selection_rate,
            "conditional_selection_recall_mean": None if not conditional_recalls else sum(conditional_recalls) / len(conditional_recalls),
            "n_retrieval_hits": len(retrieval_hits),
            "n_selected_hits": len(selected_hits),
        },
        "support_alignment": {
            "retrieval_top50_hit_rate": rate(mapped, "support_retrieval_hit"),
            "selected_top5_hit_rate": rate(mapped, "support_selected_hit"),
        },
        "decision": {
            "status": status,
            "mapping_gate_pass": mapping_rate >= float(min_mapping_rate),
            "minimum_mapping_rate": float(min_mapping_rate),
            "interpretation": (
                "Use strict retrieval/selection bottleneck classification."
                if status == "ready_for_bottleneck_audit" else
                "Do not claim a bottleneck until the NQ-to-Wiki-DPR mapping is repaired."
            ),
        },
    }


def find_forbidden(value: Any, path: str = "$root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in FORBIDDEN_FIELDS:
                findings.append(child_path)
            findings.extend(find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden(child, f"{path}[{index}]"))
    return findings


def validate_compact_result(payload: Mapping[str, Any]) -> None:
    forbidden = find_forbidden(payload)
    if forbidden:
        raise GoldAlignmentError(f"compact result contains forbidden fields: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise GoldAlignmentError("compact result needs a non-empty samples list")
    ids: set[str] = set()
    for row in samples:
        if not isinstance(row, Mapping) or not str(row.get("question_id", "")):
            raise GoldAlignmentError("each sample needs question_id")
        question_id = str(row["question_id"])
        if question_id in ids:
            raise GoldAlignmentError(f"duplicate question_id: {question_id}")
        ids.add(question_id)
        for key in ("retrieval_hit", "selected_hit", "support_retrieval_hit", "support_selected_hit"):
            if not isinstance(row.get(key), bool):
                raise GoldAlignmentError(f"{question_id}: {key} must be boolean")
        recall = row.get("conditional_selection_recall")
        if recall is not None and not _finite(recall):
            raise GoldAlignmentError(f"{question_id}: recall must be finite or null")


__all__ = [
    "FORBIDDEN_FIELDS", "GoldAlignmentError", "GoldEvidence", "align_corpus",
    "answer_in_passage", "extract_gold_evidence", "find_forbidden", "match_passage",
    "normalize_text", "normalize_title", "passage_identity", "retrieved_matches",
    "summarize_alignment", "validate_compact_result",
]
