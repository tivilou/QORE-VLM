"""Compact metrics and gate logic for the Phase 9C rank-depth diagnostic."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


FORBIDDEN_FIELDS = {
    "question",
    "passages",
    "gold_answers",
    "prediction",
    "raw_prompt",
}
EXPECTED_CUTOFFS = (50, 100, 200)


class RankDepthError(ValueError):
    """Raised when a compact Phase 9C result is malformed."""


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise RankDepthError("Wilson interval requires a positive sample count")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def _find_forbidden(payload: Any, path: str = "$root") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child = f"{path}.{key}"
            if str(key) in FORBIDDEN_FIELDS:
                findings.append(child)
            findings.extend(_find_forbidden(value, child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_find_forbidden(value, f"{path}[{index}]"))
    return findings


def _sample_map(payload: Mapping[str, Any], name: str) -> dict[str, dict[str, Any]]:
    forbidden = _find_forbidden(payload)
    if forbidden:
        raise RankDepthError(f"{name}: forbidden fields present: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RankDepthError(f"{name}: samples must be a non-empty list")

    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict) or not sample.get("question_id"):
            raise RankDepthError(f"{name}: every sample needs question_id")
        question_id = str(sample["question_id"])
        if question_id in result:
            raise RankDepthError(f"{name}: duplicate question_id {question_id}")

        first_rank = sample.get("first_answer_rank")
        if first_rank is not None:
            if not _finite(first_rank) or int(first_rank) != float(first_rank):
                raise RankDepthError(f"{name}/{question_id}: invalid first_answer_rank")
            first_rank = int(first_rank)
            if not 1 <= first_rank <= max(EXPECTED_CUTOFFS):
                raise RankDepthError(f"{name}/{question_id}: first_answer_rank out of range")

        normalized: dict[str, Any] = {
            "first_answer_rank": first_rank,
        }
        for cutoff in EXPECTED_CUTOFFS:
            key = f"answer_hit_at_{cutoff}"
            value = sample.get(key)
            if not isinstance(value, bool):
                raise RankDepthError(f"{name}/{question_id}: {key} must be boolean")
            expected = first_rank is not None and first_rank <= cutoff
            if value != expected:
                raise RankDepthError(f"{name}/{question_id}: {key} disagrees with first_answer_rank")
            normalized[key] = value
        result[question_id] = normalized
    return result


def _cutoff_summary(rows: Sequence[Mapping[str, Any]], cutoff: int) -> dict[str, Any]:
    key = f"answer_hit_at_{cutoff}"
    hits = sum(bool(row[key]) for row in rows)
    total = len(rows)
    return {
        "n_hits": hits,
        "n_misses": total - hits,
        "hit_rate": hits / total,
        "failure_rate": 1.0 - hits / total,
        "hit_rate_wilson95": _wilson_interval(hits, total),
    }


def summarize_rank_depth(
    payload: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize first-answer ranks and classify cutoff versus retriever failure."""
    rows_by_id = _sample_map(payload, "phase9c")
    question_ids = sorted(rows_by_id)
    rows = [rows_by_id[question_id] for question_id in question_ids]
    cutoffs = {str(cutoff): _cutoff_summary(rows, cutoff) for cutoff in EXPECTED_CUTOFFS}

    first_ranks = [
        int(row["first_answer_rank"])
        for row in rows
        if row["first_answer_rank"] is not None
    ]
    rank_bins = {
        "rank_1_50": sum(1 <= rank <= 50 for rank in first_ranks),
        "rank_51_100": sum(51 <= rank <= 100 for rank in first_ranks),
        "rank_101_200": sum(101 <= rank <= 200 for rank in first_ranks),
        "not_in_200": len(rows) - len(first_ranks),
    }
    top50_failure = float(cutoffs["50"]["failure_rate"])
    top200_failure = float(cutoffs["200"]["failure_rate"])
    top200_gain = float(cutoffs["200"]["hit_rate"] - cutoffs["50"]["hit_rate"])
    max_top50_failure = float(gate.get("maximum_top50_failure_rate", 0.20))
    min_top200_gain = float(gate.get("minimum_top200_gain_over_top50", 0.10))
    max_top200_failure = float(gate.get("maximum_top200_failure_rate", 0.20))

    if top50_failure <= max_top50_failure:
        decision = "no_top50_ceiling"
    elif top200_gain >= min_top200_gain:
        decision = "cutoff_limited"
    elif top200_failure > max_top200_failure:
        decision = "retriever_limited"
    else:
        decision = "mixed_or_inconclusive"

    return {
        "schema_version": 1,
        "n_questions": len(rows),
        "cutoffs": cutoffs,
        "first_answer_rank": {
            "n_found_at_200": len(first_ranks),
            "n_not_found_at_200": len(rows) - len(first_ranks),
            "mean": _mean(first_ranks),
            "median": None if not first_ranks else float(statistics.median(first_ranks)),
            "p25": _percentile(first_ranks, 0.25),
            "p75": _percentile(first_ranks, 0.75),
            "bins": rank_bins,
        },
        "decision": {
            "primary_bottleneck": decision,
            "top200_gain_over_top50": top200_gain,
            "thresholds": {
                "maximum_top50_failure_rate": max_top50_failure,
                "minimum_top200_gain_over_top50": min_top200_gain,
                "maximum_top200_failure_rate": max_top200_failure,
            },
        },
    }


__all__ = [
    "EXPECTED_CUTOFFS",
    "FORBIDDEN_FIELDS",
    "RankDepthError",
    "summarize_rank_depth",
]
