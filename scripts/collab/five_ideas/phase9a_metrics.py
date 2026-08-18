"""Compact paired summaries and pre-registered gates for Phase 9A."""

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


class Phase9AError(ValueError):
    """Raised when a Phase 9A result matrix is malformed."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase9AError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase9AError("cannot compute a percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def paired_bootstrap(values: Sequence[float], *, repetitions: int, seed: int) -> dict[str, float]:
    if not values or repetitions < 100:
        raise Phase9AError("bootstrap requires non-empty values and at least 100 repetitions")
    rng = random.Random(seed)
    draws: list[float] = []
    size = len(values)
    for _ in range(repetitions):
        draws.append(_mean([values[rng.randrange(size)] for _ in range(size)]))
    return {
        "mean": _mean(values),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def _sample_map(payload: Mapping[str, Any], name: str) -> dict[str, dict[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase9AError(f"{name}: samples must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict) or not sample.get("question_id"):
            raise Phase9AError(f"{name}: every sample needs question_id")
        question_id = str(sample["question_id"])
        if question_id in result:
            raise Phase9AError(f"{name}: duplicate question_id {question_id}")
        for key in ("f1", "em", "generation_time_ms", "context_full_token_count",
                    "context_transformed_token_count", "context_reduction_ratio"):
            if not _finite(sample.get(key)):
                raise Phase9AError(f"{name}/{question_id}: missing finite {key}")
        result[question_id] = sample
    return result


def _paired_delta(
    baseline: Mapping[str, dict[str, Any]],
    variant: Mapping[str, dict[str, Any]],
    metric: str,
) -> list[float]:
    ids = sorted(set(baseline) & set(variant))
    if not ids or set(baseline) != set(variant):
        raise Phase9AError("all configurations must contain the same question IDs")
    return [float(variant[item][metric]) - float(baseline[item][metric]) for item in ids]


def _variant_summary(
    baseline: Mapping[str, dict[str, Any]],
    variant: Mapping[str, dict[str, Any]],
    *,
    name: str,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    f1_delta = _paired_delta(baseline, variant, "f1")
    em_delta = _paired_delta(baseline, variant, "em")
    baseline_tokens = [float(baseline[item]["context_full_token_count"]) for item in baseline]
    variant_tokens = [float(variant[item]["context_transformed_token_count"]) for item in baseline]
    baseline_latency = [float(baseline[item]["generation_time_ms"]) for item in baseline]
    variant_latency = [float(variant[item]["generation_time_ms"]) for item in baseline]
    full_total = sum(baseline_tokens)
    transformed_total = sum(variant_tokens)
    base_latency = sum(baseline_latency)
    new_latency = sum(variant_latency)
    span_candidates = sum(int(variant[item].get("context_span_candidates", 0)) for item in baseline)
    span_found = sum(int(variant[item].get("context_span_found", 0)) for item in baseline)
    span_truncated = sum(int(variant[item].get("context_span_truncated", 0)) for item in baseline)
    fallback_count = sum(int(variant[item].get("context_fallback_count", 0)) for item in baseline)
    span_failure_count = max(0, span_candidates - span_found) + span_truncated
    return {
        "name": name,
        "n_questions": len(f1_delta),
        "f1": {
            "baseline_mean": _mean([float(baseline[item]["f1"]) for item in baseline]),
            "variant_mean": _mean([float(variant[item]["f1"]) for item in baseline]),
            "delta": paired_bootstrap(f1_delta, repetitions=bootstrap_repetitions, seed=bootstrap_seed),
        },
        "em": {
            "baseline_mean": _mean([float(baseline[item]["em"]) for item in baseline]),
            "variant_mean": _mean([float(variant[item]["em"]) for item in baseline]),
            "delta": paired_bootstrap(em_delta, repetitions=bootstrap_repetitions, seed=bootstrap_seed + 1),
        },
        "context": {
            "baseline_tokens": full_total / len(baseline_tokens),
            "variant_tokens": transformed_total / len(variant_tokens),
            "token_reduction_ratio": 1.0 - transformed_total / full_total if full_total else 0.0,
            "baseline_generation_time_ms": base_latency / len(baseline_latency),
            "variant_generation_time_ms": new_latency / len(variant_latency),
            "generation_time_reduction_ratio": 1.0 - new_latency / base_latency if base_latency else 0.0,
            "span_candidates": span_candidates,
            "span_found": span_found,
            "span_truncated": span_truncated,
            "fallback_count": fallback_count,
            "span_truncated_fraction": span_truncated / span_candidates if span_candidates else 0.0,
            "span_failure_count": span_failure_count,
            "span_failure_fraction": span_failure_count / span_candidates if span_candidates else 0.0,
        },
    }


def summarize_context_intervention(
    results: Mapping[str, Mapping[str, Any]],
    *,
    baseline_name: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize matched evaluator results without copying raw content."""
    if baseline_name not in results:
        raise Phase9AError(f"missing baseline result {baseline_name}")
    baseline = _sample_map(results[baseline_name], baseline_name)
    variants: list[dict[str, Any]] = []
    for name, payload in results.items():
        if name == baseline_name:
            continue
        variants.append(_variant_summary(
            baseline,
            _sample_map(payload, name),
            name=name,
            bootstrap_repetitions=int(gate.get("bootstrap_repetitions", 2000)),
            bootstrap_seed=int(gate.get("bootstrap_seed", 9201)),
        ))

    minimum_delta = float(gate.get("minimum_f1_delta", 0.01))
    maximum_harm = float(gate.get("maximum_f1_harm", 0.005))
    minimum_tokens = float(gate.get("minimum_token_reduction", 0.25))
    minimum_latency = float(gate.get("minimum_generation_time_reduction", 0.15))
    maximum_span_failure = float(gate.get("maximum_reader_span_failure_fraction", 0.05))

    for item in variants:
        item["gate"] = {
            "minimum_f1_delta_met": item["f1"]["delta"]["mean"] >= minimum_delta,
            "maximum_f1_harm_met": item["f1"]["delta"]["mean"] >= -maximum_harm,
            "efficiency_met": (
                item["context"]["token_reduction_ratio"] >= minimum_tokens
                or item["context"]["generation_time_reduction_ratio"] >= minimum_latency
            ),
            "reader_span_retention_met": item["context"]["span_failure_fraction"] <= maximum_span_failure,
        }

    by_transform: dict[str, list[dict[str, Any]]] = {"uniform_head": [], "reader_window": []}
    for item in variants:
        transform = str(results[item["name"]].get("config", {}).get("context_transform", ""))
        if transform in by_transform:
            by_transform[transform].append(item)
    direction_consistency: dict[str, bool] = {}
    for transform, items in by_transform.items():
        direction_consistency[transform] = len(items) >= 2 and all(
            float(item["f1"]["delta"]["mean"]) >= 0.0 for item in items
        )

    no_harm = all(item["gate"]["maximum_f1_harm_met"] for item in variants)
    efficient_positive = [
        item for item in variants
        if item["gate"]["minimum_f1_delta_met"] and item["gate"]["efficiency_met"]
    ]
    uniform_positive = [
        item for item in efficient_positive
        if results[item["name"]].get("config", {}).get("context_transform") == "uniform_head"
    ]
    raw_reader_positive = [
        item for item in efficient_positive
        if results[item["name"]].get("config", {}).get("context_transform") == "reader_window"
    ]
    reader_positive = [
        item for item in efficient_positive
        if (
            results[item["name"]].get("config", {}).get("context_transform") == "reader_window"
            and item["gate"]["reader_span_retention_met"]
        )
    ]
    reader_beats_uniform: dict[str, bool] = {}
    for ratio in (0.5, 0.75):
        uniform = next((item for item in variants if item["name"] == f"qore_as_uniform_head_{int(ratio * 100):03d}"), None)
        reader = next((item for item in variants if item["name"] == f"qore_as_reader_window_{int(ratio * 100):03d}"), None)
        if uniform and reader:
            reader_beats_uniform[str(ratio)] = reader["f1"]["delta"]["mean"] > uniform["f1"]["delta"]["mean"]

    if (
        no_harm
        and reader_positive
        and direction_consistency.get("reader_window", False)
        and any(reader_beats_uniform.values())
    ):
        outcome = "pass_reader_window"
    elif no_harm and uniform_positive and direction_consistency.get("uniform_head", False):
        outcome = "pass_uniform_head_context_only"
    elif no_harm and raw_reader_positive and not reader_positive:
        outcome = "fail_reader_span_retention"
    elif not no_harm:
        outcome = "fail_f1_harm"
    elif not (reader_positive or uniform_positive):
        outcome = "fail_no_context_headroom"
    else:
        outcome = "fail_inconsistent_budget_direction"

    return {
        "schema_version": 1,
        "baseline": baseline_name,
        "n_questions": len(baseline),
        "variants": variants,
        "direction_consistency": direction_consistency,
        "reader_beats_uniform_by_budget": reader_beats_uniform,
        "outcome": outcome,
        "gate_thresholds": {
            "minimum_f1_delta": minimum_delta,
            "maximum_f1_harm": maximum_harm,
            "minimum_token_reduction": minimum_tokens,
            "minimum_generation_time_reduction": minimum_latency,
            "maximum_reader_span_failure_fraction": maximum_span_failure,
        },
    }


__all__ = ["FORBIDDEN_FIELDS", "Phase9AError", "paired_bootstrap", "summarize_context_intervention"]
