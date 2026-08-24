"""Gold-free applicability policy for the Phase 9K diagnostic.

The policy is deliberately small and label-free.  It decides whether the
reader-span ranker may replace the frozen baseline using only candidate text,
selected passages, and reader scores.  Gold labels are never an input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

try:
    from scripts.collab.five_ideas.phase9j_probe import (
        CandidateScore,
        READER_SUPPORT_PROFILE,
        build_candidate_scores,
        choose_candidate,
    )
except ImportError:  # pragma: no cover
    from phase9j_probe import (
        CandidateScore,
        READER_SUPPORT_PROFILE,
        build_candidate_scores,
        choose_candidate,
    )


APPLICABILITY_PROFILE = "gold_free_reader_span_gate_v1"
READER_PROFILE = READER_SUPPORT_PROFILE
REASON_CODES = (
    "apply",
    "no_candidates",
    "duplicate_candidates",
    "baseline_selected",
    "baseline_supported",
    "no_candidate_consensus",
    "no_exact_span",
    "weak_reader_margin",
)


@dataclass(frozen=True)
class ApplicabilityDecision:
    """Compact, scalar-only decision trace for the frozen applicability gate."""

    apply: bool
    reason_code: str
    candidate_count: int
    unique_candidate_count: int
    chosen_mode: str | None
    baseline_score: float | None
    chosen_score: float | None
    reader_margin: float | None
    chosen_exact_span: bool | None
    baseline_exact_span: bool | None
    candidate_consensus: bool | None


def normalize_candidate(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(str(text), add_special_tokens=False)
    values = encoded.get("input_ids", []) if isinstance(encoded, Mapping) else []
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def exact_span_supported(tokenizer: Any, passages: Sequence[str], candidate: str) -> bool:
    """Return whether candidate token IDs occur contiguously in a passage."""

    candidate_ids = _token_ids(tokenizer, candidate.strip())
    if not candidate_ids:
        return False
    width = len(candidate_ids)
    for passage in passages:
        passage_ids = _token_ids(tokenizer, str(passage))
        if any(
            passage_ids[index : index + width] == candidate_ids
            for index in range(max(0, len(passage_ids) - width + 1))
        ):
            return True
    return False


def candidate_consensus(candidates: Sequence[tuple[str, str]]) -> bool:
    """Return whether both independent candidate prompts produced the same text."""
    values = {
        str(mode): normalize_candidate(text)
        for mode, text in candidates
        if str(mode) in {"extractive_span_v1", "evidence_constrained_v1"}
    }
    return (
        set(values) == {"extractive_span_v1", "evidence_constrained_v1"}
        and bool(values["extractive_span_v1"])
        and values["extractive_span_v1"] == values["evidence_constrained_v1"]
    )


def decide_applicability(
    *,
    candidates: Sequence[tuple[str, str]],
    raw_scores: Mapping[str, Mapping[str, float]],
    exact_span_modes: Mapping[str, bool],
    reader_margin_min: float = 0.0,
    min_candidate_count: int = 2,
    min_unique_candidate_count: int = 2,
    baseline_exact_span: bool = False,
    require_baseline_not_exact: bool = True,
    require_candidate_consensus: bool = True,
) -> tuple[ApplicabilityDecision, dict[str, CandidateScore]]:
    """Apply the preregistered label-free gate and return ranked scores.

    Candidate mode IDs are stable and all decisions are made before any gold
    answer or evaluator output is consulted.  The baseline is always named
    ``baseline_v1`` when present.
    """

    if not math.isfinite(float(reader_margin_min)):
        raise ValueError("reader_margin_min must be finite")
    modes = [str(mode) for mode, _ in candidates]
    unique_count = len({normalize_candidate(text) for _, text in candidates})
    if len(candidates) < min_candidate_count:
        return (
            ApplicabilityDecision(
                False,
                "no_candidates",
                len(candidates),
                unique_count,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            {},
        )
    if unique_count < min_unique_candidate_count:
        return (
            ApplicabilityDecision(
                False,
                "duplicate_candidates",
                len(candidates),
                unique_count,
                None,
                None,
                None,
                None,
                None,
                bool(baseline_exact_span),
                False,
            ),
            {},
        )
    expected_modes = set(modes)
    if expected_modes != set(raw_scores) or expected_modes != set(exact_span_modes):
        raise ValueError("candidate, score, and exact-span mode IDs must match")
    scores = build_candidate_scores(raw_scores, reader_weight=1.0)
    chosen_mode = choose_candidate(scores, READER_PROFILE)
    baseline = scores.get("baseline_v1")
    chosen = scores[chosen_mode]
    if baseline is None:
        raise ValueError("baseline_v1 must be present")
    baseline_raw = float(raw_scores["baseline_v1"]["reader_support"])
    chosen_raw = float(raw_scores[chosen_mode]["reader_support"])
    margin = chosen_raw - baseline_raw
    exact = bool(exact_span_modes[chosen_mode])
    consensus = candidate_consensus(candidates)
    if chosen_mode == "baseline_v1":
        reason = "baseline_selected"
    elif require_baseline_not_exact and baseline_exact_span:
        reason = "baseline_supported"
    elif require_candidate_consensus and not consensus:
        reason = "no_candidate_consensus"
    elif not exact:
        reason = "no_exact_span"
    elif margin < float(reader_margin_min):
        reason = "weak_reader_margin"
    else:
        reason = "apply"
    return (
        ApplicabilityDecision(
            reason == "apply",
            reason,
            len(candidates),
            unique_count,
            chosen_mode,
            baseline_raw,
            chosen_raw,
            margin,
            exact,
            bool(baseline_exact_span),
            consensus,
        ),
        scores,
    )


__all__ = [
    "APPLICABILITY_PROFILE",
    "ApplicabilityDecision",
    "REASON_CODES",
    "READER_PROFILE",
    "decide_applicability",
    "candidate_consensus",
    "exact_span_supported",
    "normalize_candidate",
]
