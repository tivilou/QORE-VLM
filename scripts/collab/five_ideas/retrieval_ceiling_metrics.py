"""Compact metrics for the Phase 9B Top-50 retrieval-ceiling diagnostic."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence


FORBIDDEN_FIELDS = (
    "question",
    "passages",
    "gold_answers",
    "prediction",
    "raw_prompt",
)


class RetrievalCeilingError(ValueError):
    """Raised when a matched retrieval-ceiling matrix is malformed."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RetrievalCeilingError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RetrievalCeilingError("cannot compute a percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def paired_bootstrap(values: Sequence[float], *, repetitions: int, seed: int) -> dict[str, float]:
    if not values or repetitions < 100:
        raise RetrievalCeilingError("bootstrap needs non-empty values and at least 100 repetitions")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(repetitions):
        draws.append(_mean([values[rng.randrange(len(values))] for _ in values]))
    return {
        "mean": _mean(values),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


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


def _sample_map(
    payload: Mapping[str, Any],
    name: str,
    *,
    require_generation: bool,
) -> dict[str, dict[str, Any]]:
    forbidden = _find_forbidden(payload)
    if forbidden:
        raise RetrievalCeilingError(f"{name}: forbidden fields present: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RetrievalCeilingError(f"{name}: samples must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict) or not sample.get("question_id"):
            raise RetrievalCeilingError(f"{name}: every sample needs question_id")
        question_id = str(sample["question_id"])
        if question_id in result:
            raise RetrievalCeilingError(f"{name}: duplicate question_id {question_id}")
        hit = sample.get("answer_hit_at_retrieved")
        if not isinstance(hit, bool):
            raise RetrievalCeilingError(f"{name}/{question_id}: answer_hit_at_retrieved must be boolean")
        recall = sample.get("recall")
        if hit and not _finite(recall):
            raise RetrievalCeilingError(f"{name}/{question_id}: hit sample needs finite recall")
        row: dict[str, Any] = {
            "retrieval_hit": hit,
            "selected_hit": bool(_finite(recall) and float(recall) > 0.0),
            "recall": None if recall is None else float(recall),
        }
        if require_generation:
            for key in ("f1", "em", "generation_time_ms"):
                if not _finite(sample.get(key)):
                    raise RetrievalCeilingError(f"{name}/{question_id}: missing finite {key}")
                row[key] = float(sample[key])
        result[question_id] = row
    return result


def _same_ids(*maps: Mapping[str, Any]) -> list[str]:
    if not maps:
        raise RetrievalCeilingError("at least one sample map is required")
    ids = set(maps[0])
    if not ids or any(set(item) != ids for item in maps[1:]):
        raise RetrievalCeilingError("all configurations must contain the same question IDs")
    return sorted(ids)


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _mean([float(bool(row[key])) for row in rows])


def _conditional_rate(rows: Sequence[Mapping[str, Any]], predicate: str, value: str) -> float | None:
    eligible = [row for row in rows if bool(row[predicate])]
    if not eligible:
        return None
    return _rate(eligible, value)


def _conditional_mean(rows: Sequence[Mapping[str, Any]], predicate: str, value: str) -> float | None:
    eligible = [float(row[value]) for row in rows if bool(row[predicate])]
    return None if not eligible else _mean(eligible)


def summarize_retrieval_ceiling(
    results: Mapping[str, Mapping[str, Any]],
    *,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify retrieval, selection, and generation bottleneck signals."""
    expected = ("retrieval_top50_as", "qore_as_select", "topk_as_select", "qore_as_generate")
    missing = [name for name in expected if name not in results]
    if missing:
        raise RetrievalCeilingError(f"missing configurations: {missing}")
    top50 = _sample_map(results["retrieval_top50_as"], "retrieval_top50_as", require_generation=False)
    qore = _sample_map(results["qore_as_select"], "qore_as_select", require_generation=False)
    topk = _sample_map(results["topk_as_select"], "topk_as_select", require_generation=False)
    generate = _sample_map(results["qore_as_generate"], "qore_as_generate", require_generation=True)
    question_ids = _same_ids(top50, qore, topk, generate)

    for question_id in question_ids:
        hit = top50[question_id]["retrieval_hit"]
        if any(item[question_id]["retrieval_hit"] != hit for item in (qore, topk, generate)):
            raise RetrievalCeilingError(f"retrieval hit mismatch for {question_id}")

    top50_rows = [top50[item] for item in question_ids]
    qore_rows = [qore[item] for item in question_ids]
    topk_rows = [topk[item] for item in question_ids]
    generate_rows = [generate[item] for item in question_ids]
    retrieval_hit_rate = _rate(top50_rows, "retrieval_hit")
    retrieval_failure_rate = 1.0 - retrieval_hit_rate
    eligible = [index for index, row in enumerate(top50_rows) if row["retrieval_hit"]]
    if not eligible:
        raise RetrievalCeilingError("Top-50 retrieval found no answer-bearing question")

    qore_selected = [float(qore_rows[index]["selected_hit"]) for index in eligible]
    topk_selected = [float(topk_rows[index]["selected_hit"]) for index in eligible]
    selection_recovery = [right - left for left, right in zip(qore_selected, topk_selected)]

    qore_selection = {
        "selected_hit_rate_conditional": _mean(qore_selected),
        "selection_failure_rate_conditional": 1.0 - _mean(qore_selected),
    }
    topk_selection = {
        "selected_hit_rate_conditional": _mean(topk_selected),
        "selection_failure_rate_conditional": 1.0 - _mean(topk_selected),
    }
    qore_selection["topk_recovery_delta"] = paired_bootstrap(
        selection_recovery,
        repetitions=int(gate.get("bootstrap_repetitions", 2000)),
        seed=int(gate.get("bootstrap_seed", 9301)),
    )

    generation_hit_rows = [row for row in generate_rows if row["retrieval_hit"] and row["selected_hit"]]
    generation_miss_rows = [row for row in generate_rows if row["retrieval_hit"] and not row["selected_hit"]]
    generation = {
        "f1_mean": _mean([row["f1"] for row in generate_rows]),
        "em_mean": _mean([row["em"] for row in generate_rows]),
        "f1_when_selected_hit": None if not generation_hit_rows else _mean([row["f1"] for row in generation_hit_rows]),
        "f1_when_selected_miss": None if not generation_miss_rows else _mean([row["f1"] for row in generation_miss_rows]),
        "em_when_selected_hit": None if not generation_hit_rows else _mean([row["em"] for row in generation_hit_rows]),
        "n_when_selected_hit": len(generation_hit_rows),
        "n_when_selected_miss": len(generation_miss_rows),
        "generation_time_ms_mean": _mean([row["generation_time_ms"] for row in generate_rows]),
    }

    max_retrieval_failure = float(gate.get("maximum_retrieval_failure_rate", 0.20))
    min_selection_failure = float(gate.get("minimum_selection_failure_rate", 0.10))
    min_topk_recovery = float(gate.get("minimum_topk_recovery_delta", 0.10))
    if retrieval_failure_rate > max_retrieval_failure:
        bottleneck = "retrieval_ceiling"
    elif (
        qore_selection["selection_failure_rate_conditional"] >= min_selection_failure
        and qore_selection["topk_recovery_delta"]["mean"] >= min_topk_recovery
    ):
        bottleneck = "selection"
    else:
        bottleneck = "generation_or_answer_alignment"

    return {
        "schema_version": 1,
        "n_questions": len(question_ids),
        "retrieval": {
            "top50_answer_hit_rate": retrieval_hit_rate,
            "top50_answer_failure_rate": retrieval_failure_rate,
            "n_top50_answer_hits": sum(row["retrieval_hit"] for row in top50_rows),
        },
        "selection": {
            "qore_as_k5": qore_selection,
            "topk_as_k5": topk_selection,
            "eligible_questions_with_top50_hit": len(eligible),
        },
        "generation": generation,
        "decision": {
            "primary_bottleneck": bottleneck,
            "thresholds": {
                "maximum_retrieval_failure_rate": max_retrieval_failure,
                "minimum_selection_failure_rate": min_selection_failure,
                "minimum_topk_recovery_delta": min_topk_recovery,
            },
        },
    }


__all__ = [
    "FORBIDDEN_FIELDS",
    "RetrievalCeilingError",
    "paired_bootstrap",
    "summarize_retrieval_ceiling",
]
