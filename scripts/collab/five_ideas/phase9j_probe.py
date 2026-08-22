"""Observation-only context-induced candidate scoring for Phase 9J."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


CONTEXT_LIFT_PROFILE = "context_lift_v1"
READER_SUPPORT_PROFILE = "reader_span_support_v1"
COMBINED_PROFILE = "context_lift_reader_support_v1"
RANKER_PROFILES = (CONTEXT_LIFT_PROFILE, READER_SUPPORT_PROFILE, COMBINED_PROFILE)


@dataclass(frozen=True)
class CandidateScore:
    mode: str
    context_lift: float
    reader_support: float
    context_rank: float
    reader_rank: float
    combined_rank_score: float


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean_logprob_difference(
    context_logprob: Sequence[float], empty_logprob: Sequence[float]
) -> float:
    """Return the per-token context-minus-empty log-probability lift."""
    if not context_logprob or not empty_logprob:
        raise ValueError("candidate log-probability sequences cannot be empty")
    context_mean = sum(float(value) for value in context_logprob) / len(context_logprob)
    empty_mean = sum(float(value) for value in empty_logprob) / len(empty_logprob)
    result = context_mean - empty_mean
    if not math.isfinite(result):
        raise ValueError("context lift is not finite")
    return float(result)


def _rank_normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize descending values to [0,1] with deterministic mode tie breaks."""
    if not values:
        raise ValueError("cannot rank an empty candidate mapping")
    ordered = sorted(values, key=lambda mode: (-float(values[mode]), str(mode)))
    denominator = max(1, len(ordered) - 1)
    return {
        mode: float(1.0 - index / denominator)
        for index, mode in enumerate(ordered)
    }


def build_candidate_scores(
    raw_scores: Mapping[str, Mapping[str, float]], *, reader_weight: float = 1.0
) -> dict[str, CandidateScore]:
    """Rank-normalize the two signals and form a fixed combined score."""
    if not raw_scores:
        raise ValueError("candidate scores cannot be empty")
    if not _finite(reader_weight) or float(reader_weight) < 0.0:
        raise ValueError("reader_weight must be finite and non-negative")
    context_values: dict[str, float] = {}
    reader_values: dict[str, float] = {}
    for mode, values in raw_scores.items():
        if set(values) != {"context_lift", "reader_support"}:
            raise ValueError(f"{mode}: raw score keys are not frozen")
        if not _finite(values["context_lift"]) or not _finite(values["reader_support"]):
            raise ValueError(f"{mode}: raw scores must be finite")
        context_values[str(mode)] = float(values["context_lift"])
        reader_values[str(mode)] = float(values["reader_support"])
    context_ranks = _rank_normalize(context_values)
    reader_ranks = _rank_normalize(reader_values)
    return {
        mode: CandidateScore(
            mode=mode,
            context_lift=context_values[mode],
            reader_support=reader_values[mode],
            context_rank=context_ranks[mode],
            reader_rank=reader_ranks[mode],
            combined_rank_score=context_ranks[mode]
            + float(reader_weight) * reader_ranks[mode],
        )
        for mode in context_values
    }


def choose_candidate(
    scores: Mapping[str, CandidateScore], profile: str
) -> str:
    """Choose by a fixed score with lexicographic tie breaking."""
    if profile not in RANKER_PROFILES:
        raise ValueError(f"unknown ranker profile: {profile}")
    if not scores:
        raise ValueError("cannot choose from an empty score mapping")
    field = {
        CONTEXT_LIFT_PROFILE: "context_rank",
        READER_SUPPORT_PROFILE: "reader_rank",
        COMBINED_PROFILE: "combined_rank_score",
    }[profile]
    return sorted(
        scores,
        key=lambda mode: (-float(getattr(scores[mode], field)), str(mode)),
    )[0]


def fixed_candidate_permutation(
    candidates: Sequence[tuple[str, str]]
) -> list[tuple[str, str]]:
    return list(reversed(list(candidates)))


def scores_by_permuted_order(
    candidates: Sequence[tuple[str, str]],
    raw_scores: Mapping[str, Mapping[str, float]],
    *,
    reader_weight: float = 1.0,
) -> dict[str, dict[str, str]]:
    """Return choices in original and reversed candidate order.

    Scores are keyed by stable candidate mode IDs, so the result must be
    invariant to list order. The explicit permutation argument is retained as
    an audit contract rather than used as a hidden tie-breaker.
    """
    original_modes = [mode for mode, _ in candidates]
    permuted_modes = [mode for mode, _ in fixed_candidate_permutation(candidates)]
    if set(original_modes) != set(permuted_modes) or set(original_modes) != set(raw_scores):
        raise ValueError("candidate modes and score modes do not match")
    scores = build_candidate_scores(raw_scores, reader_weight=reader_weight)
    result: dict[str, dict[str, str]] = {}
    for profile in RANKER_PROFILES:
        original = choose_candidate({mode: scores[mode] for mode in original_modes}, profile)
        permuted = choose_candidate({mode: scores[mode] for mode in permuted_modes}, profile)
        result[profile] = {
            "original_choice_mode": original,
            "permuted_choice_mode": permuted,
        }
    return result


__all__ = [
    "COMBINED_PROFILE",
    "CONTEXT_LIFT_PROFILE",
    "READER_SUPPORT_PROFILE",
    "RANKER_PROFILES",
    "CandidateScore",
    "build_candidate_scores",
    "choose_candidate",
    "fixed_candidate_permutation",
    "mean_logprob_difference",
    "scores_by_permuted_order",
]
