"""Observation-only accounting for the official DPR/NQ gold universe.

This module deliberately contains no retrieval, selector, generation, or
evaluation calls.  It turns strict source joins and canonical passage IDs into
compact audit rows and a three-way bottleneck classification.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

from applications.rag.dpr_positive_gold_alignment import (
    DPR_MAPPING_STATUSES,
    normalized_question_identity,
    validate_dpr_compact_result,
)

FORBIDDEN_FIELDS = {
    "question", "query", "passages", "positive_ctxs", "positive_contexts",
    "answers", "prediction", "raw_prompt", "passage_text", "gold_text",
    "answer_text", "title", "text",
}


def fingerprint_question(question: Any) -> str:
    return hashlib.sha256(
        normalized_question_identity(question).encode("utf-8")
    ).hexdigest()


def _count(value: Any) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mapping_identity(identity: Mapping[str, Any]) -> set[str]:
    verified = identity.get("verified", ())
    if isinstance(verified, (set, list, tuple)):
        return {str(value) for value in verified}
    return set()


def make_compact_row(
    question_id: Any,
    question: Any,
    join_status: str,
    identity: Mapping[str, Any],
    top50_ids: Iterable[Any] = (),
    top5_ids: Iterable[Any] = (),
    *,
    baseline_em: float | None = None,
    baseline_f1: float | None = None,
    generator_bucket: str | None = None,
) -> dict[str, Any]:
    """Create a privacy-safe row from canonical IDs only."""
    gold = _mapping_identity(identity)
    status = str(identity.get("mapping_status", join_status))
    if status not in DPR_MAPPING_STATUSES:
        status = "no_question_join"

    # Partial or unresolved source/corpus alignment is not evidence about the
    # frozen online path.  Retain its accounting, but remove every signal that
    # could be mistaken for retrieval, selection, or oracle headroom.
    attributable_gold = gold if status == "mapped" else set()
    top50 = {str(value) for value in top50_ids}
    top5 = {str(value) for value in top5_ids}
    top50_gold = top50 & attributable_gold
    top5_gold = top5 & attributable_gold
    row: dict[str, Any] = {
        "question_id": str(question_id),
        "question_fingerprint": fingerprint_question(question),
        "question_identity_verified": join_status == "joined",
        "mapping_status": status,
        "official_positive_count": _count(identity.get("official_positive_count")),
        "verified_positive_count": _count(identity.get("verified_positive_count")),
        "unresolved_positive_count": _count(identity.get("unresolved_positive_count")),
        "ambiguous_positive_count": _count(identity.get("ambiguous_positive_count")),
        "gold_passage_count": len(attributable_gold),
        "top50_gold_count": len(top50_gold),
        "top5_gold_count": len(top5_gold),
        "retrieval_hit": bool(top50_gold),
        "selected_hit": bool(top5_gold),
        "oracle_top5_evidence_ceiling": bool(top50_gold),
        "retrieval_top_k": len(top50),
        "selection_k": len(top5),
    }
    if baseline_em is not None:
        row["baseline_em"] = float(baseline_em)
    if baseline_f1 is not None:
        row["baseline_f1"] = float(baseline_f1)
    if generator_bucket is not None:
        row["generator_bucket"] = str(generator_bucket)
        if not row["retrieval_hit"]:
            row["bottleneck_bucket"] = "retrieval_bottleneck"
        elif not row["selected_hit"]:
            row["bottleneck_bucket"] = "selector_bottleneck"
        else:
            row["bottleneck_bucket"] = str(generator_bucket)
    if status in {
        "partial_wiki_dpr_identity", "no_wiki_dpr_identity",
        "ambiguous_wiki_dpr_identity",
    }:
        row["mapping_failure_category"] = "source_or_corpus_version_mismatch"
    else:
        row["mapping_failure_category"] = None
    return row


def summarize_universe(
    rows: Sequence[Mapping[str, Any]],
    *,
    all_questions: int,
    official_nonempty_context_questions: int,
    minimum_mapping_rate: float = 0.80,
) -> dict[str, Any]:
    """Summarize eligible accounting and conditional bottleneck strata."""
    if not rows:
        raise ValueError("eligible-universe audit has no rows")
    mapped = [row for row in rows if row.get("mapping_status") == "mapped"]
    retrieval_hits = [row for row in mapped if bool(row.get("retrieval_hit"))]
    selected_hits = [row for row in mapped if bool(row.get("selected_hit"))]
    selected_miss = [row for row in mapped if bool(row.get("retrieval_hit")) and not row.get("selected_hit")]
    generator_errors = [row for row in selected_hits if row.get("generator_bucket") == "selected_hit_generation_error"]
    status_counts = {
        status: sum(row.get("mapping_status") == status for row in rows)
        for status in DPR_MAPPING_STATUSES
    }
    mapping_rate = len(mapped) / len(rows)
    retrieval_bottleneck = len(mapped) - len(retrieval_hits)
    selector_bottleneck = len(selected_miss)
    generator_bottleneck = len(generator_errors)
    return {
        "schema_version": 1,
        "population": {
            "all_questions": int(all_questions),
            "official_nonempty_context_questions": int(official_nonempty_context_questions),
            "eligible_rows": len(rows),
            "strict_mapped_rows": len(mapped),
            "excluded_rows": int(all_questions) - len(rows),
            "source_coverage_rate": _rate(int(official_nonempty_context_questions), int(all_questions)),
        },
        "mapping": {
            "mapping_rate": mapping_rate,
            "minimum_mapping_rate": float(minimum_mapping_rate),
            "mapping_gate_pass": mapping_rate >= float(minimum_mapping_rate),
            "mapping_status_counts": status_counts,
        },
        "strict_bottleneck_counts": {
            "mapped_questions": len(mapped),
            "retrieval_hit_questions": len(retrieval_hits),
            "retrieval_bottleneck_questions": retrieval_bottleneck,
            "selector_bottleneck_questions": selector_bottleneck,
            "generator_bottleneck_questions": generator_bottleneck,
            "retrieval_top50_hit_rate": _rate(len(retrieval_hits), len(mapped)),
            "selected_top5_hit_rate_among_mapped": _rate(len(selected_hits), len(mapped)),
            "selected_top5_hit_rate_given_top50_gold": _rate(len(selected_hits), len(retrieval_hits)),
            "generator_error_rate_given_selected_hit": _rate(len(generator_errors), len(selected_hits)),
            "oracle_top5_evidence_ceiling_questions": len(retrieval_hits),
            "oracle_top5_evidence_ceiling_rate_among_mapped": _rate(len(retrieval_hits), len(mapped)),
        },
        "decision": {
            "status": "ready_for_bottleneck_audit" if mapping_rate >= float(minimum_mapping_rate) else "mapping_incomplete",
            "interpretation": (
                "Use strict Top-50 versus Top-5 versus frozen-generation strata."
                if mapping_rate >= float(minimum_mapping_rate)
                else "Do not attribute failures to retrieval, selector, or generator below the mapping gate."
            ),
        },
    }


def validate_universe_result(payload: Mapping[str, Any]) -> None:
    """Validate compact rows and the complete population accounting contract."""
    def scan(value: Any, path: str = "$root") -> list[str]:
        findings: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN_FIELDS:
                    findings.append(f"{path}.{key}")
                findings.extend(scan(child, f"{path}.{key}"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                findings.extend(scan(child, f"{path}[{index}]"))
        return findings
    forbidden = scan(payload)
    if forbidden:
        raise ValueError(f"compact result contains forbidden fields: {forbidden[:5]}")
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("compact result needs a non-empty samples list")
    validate_dpr_compact_result({"samples": rows})
    for row in rows:
        if not isinstance(row.get("question_fingerprint"), str) or len(row["question_fingerprint"]) != 64:
            raise ValueError(f"{row.get('question_id')}: invalid question fingerprint")
        if row["selected_hit"] and not row["retrieval_hit"]:
            raise ValueError(f"{row['question_id']}: selected hit requires retrieval hit")
        if not isinstance(row.get("oracle_top5_evidence_ceiling"), bool):
            raise ValueError(f"{row['question_id']}: oracle evidence ceiling must be boolean")
        if row["oracle_top5_evidence_ceiling"] != row["retrieval_hit"]:
            raise ValueError(f"{row['question_id']}: oracle evidence ceiling must equal Top-50 gold availability")


__all__ = [
    "fingerprint_question", "make_compact_row", "summarize_universe",
    "validate_universe_result",
]
