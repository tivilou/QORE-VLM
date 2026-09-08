"""Anchor-conditioned evidence support graph selector.

The selector tests a different set objective from diversity-based methods:
after reserving two high-confidence Answer Scorer anchors, it adds passages
that corroborate the current evidence set.  The graph is lexical and
question-conditioned so the online contract stays CPU-only and gold-free.

Only question text and the ordinary Top-50 candidate fields are accepted.
Evidence labels, answers, generation outputs, and evaluator fields are
rejected at the boundary.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SELECTOR_ID = "anchor_conditioned_support_graph"
SELECTOR_VERSION = "1.0"
DEFAULT_K = 5
DEFAULT_ANCHOR_COUNT = 2
SUPPORT_PASSAGE_WEIGHT = 0.70
SUPPORT_QUESTION_WEIGHT = 0.30

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
# Keep this deliberately small and fixed.  The support graph is intended to
# retain topical nouns and short identifiers (for example, ``us`` or ``uk``)
# instead of importing a library-specific stop-word vocabulary.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "in", "is", "of", "on", "or",
    "the", "to", "was", "were", "what", "when", "where", "which", "who",
}
_FORBIDDEN_FIELDS = {
    "evidence", "gold", "gold_answers", "answer", "prediction", "metrics",
    "generation_panel", "generation_consensus", "evaluator", "evaluation",
    "label", "labels", "selectors", "top_50_panel", "diagnostics",
}
_ALLOWED_FIELDS = {
    "id", "passage_id", "text", "passage", "retrieved_rank", "rank",
    "retrieval_score", "answer_scorer_score", "answer_scorer",
}


class SelectionInputError(ValueError):
    """Raised when an online candidate violates the selector contract."""


@dataclass(frozen=True)
class OnlineCandidate:
    passage_id: str
    text: str
    retrieved_rank: int
    retrieval_score: float
    answer_scorer_score: float


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    anchor_indices: tuple[int, ...]
    support_indices: tuple[int, ...]
    fallback: bool
    fallback_reason: str | None
    support_scores: tuple[float, ...]
    support_trace: tuple[dict[str, Any], ...] = ()


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _content_tokens(value: str) -> set[str]:
    return {token for token in _tokens(value) if len(token) > 1 and token not in _STOPWORDS}


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionInputError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionInputError(f"{field} must be a finite number")
    return result


def _coerce_candidate(value: OnlineCandidate | Mapping[str, Any]) -> OnlineCandidate:
    if isinstance(value, OnlineCandidate):
        return value
    if not isinstance(value, Mapping):
        raise SelectionInputError("candidate must be an object")
    keys = {str(key) for key in value.keys()}
    forbidden = sorted(keys & _FORBIDDEN_FIELDS)
    if forbidden:
        raise SelectionInputError(f"forbidden selection fields: {', '.join(forbidden)}")
    unknown = sorted(keys - _ALLOWED_FIELDS)
    if unknown:
        raise SelectionInputError(f"unknown selection fields: {', '.join(unknown)}")
    passage_id = value.get("id", value.get("passage_id"))
    text = value.get("text", value.get("passage"))
    if not isinstance(passage_id, (str, int)) or not str(passage_id).strip():
        raise SelectionInputError("candidate id must be non-empty")
    if not isinstance(text, str) or not text.strip():
        raise SelectionInputError("candidate text must be non-empty")
    rank = value.get("retrieved_rank", value.get("rank"))
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise SelectionInputError("retrieved_rank must be a positive integer")
    answer_score = value.get("answer_scorer_score", value.get("answer_scorer"))
    if answer_score is None:
        raise SelectionInputError("answer_scorer_score is required")
    return OnlineCandidate(
        passage_id=str(passage_id),
        text=text,
        retrieved_rank=rank,
        retrieval_score=_finite(value.get("retrieval_score", 0.0), "retrieval_score"),
        answer_scorer_score=_finite(answer_score, "answer_scorer_score"),
    )


def coerce_candidates(values: Sequence[OnlineCandidate | Mapping[str, Any]]) -> tuple[OnlineCandidate, ...]:
    candidates = tuple(_coerce_candidate(value) for value in values)
    if not candidates:
        raise SelectionInputError("at least one candidate is required")
    ids = [candidate.passage_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise SelectionInputError("candidate passage IDs must be unique")
    return candidates


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _support_edge(
    left: set[str],
    right: set[str],
    question_terms: set[str],
    *,
    passage_weight: float = SUPPORT_PASSAGE_WEIGHT,
    question_weight: float = SUPPORT_QUESTION_WEIGHT,
) -> float:
    """Measure corroboration while retaining a question-conditioned edge."""

    passage_overlap = _jaccard(left, right)
    # Keep the same passage-union denominator for both terms.  This makes a
    # shared question token a modest topical tie-breaker rather than a way to
    # inflate the edge score when the two passages have little content in
    # common.
    union_size = len(left | right)
    question_overlap = len(left & right & question_terms) / union_size if union_size else 0.0
    return passage_weight * passage_overlap + question_weight * question_overlap


def select(
    question: str,
    values: Sequence[OnlineCandidate | Mapping[str, Any]],
    *,
    k: int = DEFAULT_K,
    anchor_count: int = DEFAULT_ANCHOR_COUNT,
    support_passage_weight: float = SUPPORT_PASSAGE_WEIGHT,
    support_question_weight: float = SUPPORT_QUESTION_WEIGHT,
) -> SelectionResult:
    """Select exactly ``k`` passages from a frozen Top-50 candidate list."""

    if not isinstance(question, str) or not question.strip():
        raise SelectionInputError("question must be non-empty")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise SelectionInputError("k must be a positive integer")
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int) or anchor_count < 0:
        raise SelectionInputError("anchor_count must be a non-negative integer")
    support_passage_weight = _finite(support_passage_weight, "support_passage_weight")
    support_question_weight = _finite(support_question_weight, "support_question_weight")
    if support_passage_weight < 0 or support_question_weight < 0:
        raise SelectionInputError("support weights must be non-negative")
    candidates = coerce_candidates(values)
    if len(candidates) < k:
        raise SelectionInputError(f"need at least {k} candidates")
    anchor_count = min(anchor_count, k, len(candidates))
    anchor_order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -candidates[index].answer_scorer_score,
            candidates[index].retrieved_rank,
            candidates[index].passage_id,
        ),
    )
    anchors = tuple(anchor_order[:anchor_count])
    question_terms = _content_tokens(question)
    content_sets = tuple(_content_tokens(candidate.text) for candidate in candidates)
    if not question_terms:
        selected = tuple(anchor_order[:k])
        return SelectionResult(selected, anchors, tuple(), True, "empty_question_signal", tuple(), tuple())
    if not any(content_sets):
        selected = tuple(anchor_order[:k])
        return SelectionResult(selected, anchors, tuple(), True, "empty_candidate_signal", tuple(), tuple())

    selected = list(anchors)
    support_indices: list[int] = []
    support_scores: list[float] = []
    support_trace: list[dict[str, Any]] = []
    while len(selected) < k:
        available = [index for index in range(len(candidates)) if index not in selected]
        scored: list[tuple[float, float, int, str, int, float, float]] = []
        for index in available:
            edge_scores = [
                _support_edge(
                    content_sets[index],
                    content_sets[chosen],
                    question_terms,
                    passage_weight=support_passage_weight,
                    question_weight=support_question_weight,
                )
                for chosen in selected
            ]
            support_score = sum(edge_scores) / len(edge_scores) if edge_scores else 0.0
            passage_overlaps = [
                _jaccard(content_sets[index], content_sets[chosen])
                for chosen in selected
            ]
            union_sizes = [len(content_sets[index] | content_sets[chosen]) for chosen in selected]
            question_overlaps = [
                len(content_sets[index] & content_sets[chosen] & question_terms) / union_size
                if union_size else 0.0
                for chosen, union_size in zip(selected, union_sizes)
            ]
            passage_overlap_mean = sum(passage_overlaps) / len(passage_overlaps) if passage_overlaps else 0.0
            question_overlap_mean = sum(question_overlaps) / len(question_overlaps) if question_overlaps else 0.0
            scored.append(
                (
                    support_score,
                    candidates[index].answer_scorer_score,
                    -candidates[index].retrieved_rank,
                    candidates[index].passage_id,
                    index,
                    passage_overlap_mean,
                    question_overlap_mean,
                )
            )
        if not scored:
            selected = list(anchor_order[:k])
            return SelectionResult(
                tuple(selected), anchors, tuple(selected[len(anchors):]), True,
                "insufficient_candidates", tuple(support_scores), tuple(support_trace),
            )
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        best = scored[0]
        selected.append(best[4])
        support_indices.append(best[4])
        support_scores.append(best[0])
        support_trace.append(
            {
                "step": len(support_trace) + 1,
                "candidate_index": best[4],
                "candidate_id": candidates[best[4]].passage_id,
                "support_score": round(best[0], 8),
                "passage_overlap_mean": round(best[5], 8),
                "question_overlap_mean": round(best[6], 8),
                "passage_weight": support_passage_weight,
                "question_weight": support_question_weight,
                "tie_break_answer_scorer_score": candidates[best[4]].answer_scorer_score,
                "tie_break_retrieved_rank": candidates[best[4]].retrieved_rank,
            }
        )

    return SelectionResult(
        tuple(selected), anchors, tuple(support_indices), False, None,
        tuple(support_scores), tuple(support_trace),
    )


__all__ = [
    "DEFAULT_ANCHOR_COUNT",
    "DEFAULT_K",
    "OnlineCandidate",
    "SELECTOR_ID",
    "SELECTOR_VERSION",
    "SelectionInputError",
    "SelectionResult",
    "coerce_candidates",
    "select",
]
