"""Compact metrics and preregistered gates for Phase 10D."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


FORBIDDEN_FIELDS = {
    "question",
    "passages",
    "gold_answers",
    "prediction",
    "raw_prompt",
    "prompt",
    "text",
    "token_ids",
    "selected_ids",
    "retrieved_ids",
    "gold_answer",
}
EXPECTED_MASKS = tuple(range(1, 32))
TOPOLOGY_CLASSES = (
    "singleton_sufficient",
    "strict_non_answer_assisted_interaction",
    "answer_bearing_only_distributed_interaction",
    "beyond_selected_set",
)


class Phase10DError(ValueError):
    pass


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase10DError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase10DError("cannot compute a percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def question_bootstrap(values: Sequence[float], *, repetitions: int, seed: int) -> dict[str, float] | None:
    """Bootstrap a question-level rate without creating pseudo-replicates."""

    if not values:
        return None
    if repetitions < 100:
        raise Phase10DError("bootstrap requires at least 100 repetitions")
    rng = random.Random(seed)
    draws = [_mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(repetitions)]
    return {
        "mean": _mean(values),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def _find_forbidden(value: Any, path: str = "$root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_FIELDS:
                findings.append(child_path)
            findings.extend(_find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden(child, f"{path}[{index}]"))
    return findings


def _validate_arm(value: Any, *, attempted: bool, name: str, question_id: str) -> dict[str, float | bool | None]:
    if not isinstance(value, Mapping) or set(value) != {"attempted", "em", "f1", "generation_time_ms"}:
        raise Phase10DError(f"{question_id}/{name}: unexpected arm schema")
    if value.get("attempted") is not attempted:
        raise Phase10DError(f"{question_id}/{name}: attempted flag mismatch")
    normalized: dict[str, float | bool | None] = {"attempted": attempted}
    for field in ("em", "f1", "generation_time_ms"):
        item = value.get(field)
        if attempted:
            if not _finite(item):
                raise Phase10DError(f"{question_id}/{name}: invalid {field}")
            numeric = float(item)
            if field in {"em", "f1"} and not 0.0 <= numeric <= 1.0:
                raise Phase10DError(f"{question_id}/{name}: {field} out of range")
            if field == "generation_time_ms" and numeric < 0.0:
                raise Phase10DError(f"{question_id}/{name}: negative runtime")
            normalized[field] = numeric
        elif item is not None:
            raise Phase10DError(f"{question_id}/{name}: unattempted arm has {field}")
        else:
            normalized[field] = None
    return normalized


def _validate_subset(
    value: Any,
    *,
    baseline: Mapping[str, float | bool | None],
    question_id: str,
) -> dict[str, int | float | bool]:
    expected = {
        "mask",
        "cardinality",
        "answer_match_count",
        "contains_non_answer_literal",
        "em",
        "f1",
        "generation_time_ms",
        "reused_frozen_baseline",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Phase10DError(f"{question_id}: malformed subset outcome")
    mask = value.get("mask")
    cardinality = value.get("cardinality")
    answer_matches = value.get("answer_match_count")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in (mask, cardinality, answer_matches)):
        raise Phase10DError(f"{question_id}: subset topology values must be integers")
    if mask not in EXPECTED_MASKS or cardinality != mask.bit_count() or not 0 <= answer_matches <= cardinality:
        raise Phase10DError(f"{question_id}: invalid subset mask topology")
    contains_non_answer = value.get("contains_non_answer_literal")
    if not isinstance(contains_non_answer, bool) or contains_non_answer != (answer_matches < cardinality):
        raise Phase10DError(f"{question_id}: subset answer-literal topology disagrees")
    reused = value.get("reused_frozen_baseline")
    if not isinstance(reused, bool) or reused != (mask == 31):
        raise Phase10DError(f"{question_id}: only the full mask may reuse baseline")
    normalized: dict[str, int | float | bool] = {
        "mask": mask,
        "cardinality": cardinality,
        "answer_match_count": answer_matches,
        "contains_non_answer_literal": contains_non_answer,
        "reused_frozen_baseline": reused,
    }
    for field in ("em", "f1", "generation_time_ms"):
        item = value.get(field)
        if not _finite(item):
            raise Phase10DError(f"{question_id}/mask={mask}: invalid {field}")
        numeric = float(item)
        if field in {"em", "f1"} and not 0.0 <= numeric <= 1.0:
            raise Phase10DError(f"{question_id}/mask={mask}: {field} out of range")
        if field == "generation_time_ms" and numeric < 0.0:
            raise Phase10DError(f"{question_id}/mask={mask}: negative runtime")
        normalized[field] = numeric
    if mask == 31:
        for field in ("em", "f1", "generation_time_ms"):
            if normalized[field] != baseline[field]:
                raise Phase10DError(f"{question_id}: full mask must exactly reuse baseline {field}")
    return normalized


def _validate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    forbidden = _find_forbidden(payload)
    if forbidden:
        raise Phase10DError(f"forbidden fields present: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase10DError("samples must be a non-empty list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected = {
        "question_id",
        "retrieval_hit",
        "selected_hit",
        "selected_answer_match_count",
        "selection_time_ms",
        "arms",
        "subset_outcomes",
    }
    for value in samples:
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Phase10DError("unexpected compact sample schema")
        question_id = str(value.get("question_id", ""))
        if not question_id or question_id in seen:
            raise Phase10DError(f"invalid or duplicate question ID {question_id!r}")
        seen.add(question_id)
        retrieval_hit, selected_hit = value.get("retrieval_hit"), value.get("selected_hit")
        if not isinstance(retrieval_hit, bool) or not isinstance(selected_hit, bool) or (selected_hit and not retrieval_hit):
            raise Phase10DError(f"{question_id}: invalid retrieval/selected hit flags")
        selected_matches = value.get("selected_answer_match_count")
        if not isinstance(selected_matches, int) or isinstance(selected_matches, bool) or not 0 <= selected_matches <= 5:
            raise Phase10DError(f"{question_id}: invalid selected answer-match count")
        if selected_hit != (selected_matches > 0):
            raise Phase10DError(f"{question_id}: selected hit/count disagreement")
        if not _finite(value.get("selection_time_ms")) or float(value["selection_time_ms"]) < 0.0:
            raise Phase10DError(f"{question_id}: invalid selection runtime")
        arms = value.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {"baseline", "gold_answer_copy"}:
            raise Phase10DError(f"{question_id}: unexpected arms")
        baseline = _validate_arm(arms["baseline"], attempted=True, name="baseline", question_id=question_id)
        primary_error = bool(selected_hit and baseline["em"] == 0.0)
        copy = _validate_arm(arms["gold_answer_copy"], attempted=primary_error, name="gold_answer_copy", question_id=question_id)
        eligible = bool(primary_error and copy["em"] == 1.0)
        subset_values = value.get("subset_outcomes")
        if not isinstance(subset_values, list):
            raise Phase10DError(f"{question_id}: subset outcomes must be a list")
        if not eligible:
            if subset_values:
                raise Phase10DError(f"{question_id}: ineligible question has subset probes")
            subsets: list[dict[str, int | float | bool]] = []
        else:
            if len(subset_values) != len(EXPECTED_MASKS):
                raise Phase10DError(f"{question_id}: eligible question must have all 31 subset outcomes")
            subsets = [_validate_subset(item, baseline=baseline, question_id=question_id) for item in subset_values]
            if tuple(item["mask"] for item in subsets) != EXPECTED_MASKS:
                raise Phase10DError(f"{question_id}: subset masks must be deterministic ascending 1..31")
            if int(subsets[-1]["answer_match_count"]) != selected_matches:
                raise Phase10DError(f"{question_id}: full-mask answer-match count must equal selected count")
        rows.append({
            "question_id": question_id,
            "retrieval_hit": retrieval_hit,
            "selected_hit": selected_hit,
            "selected_answer_match_count": selected_matches,
            "selection_time_ms": float(value["selection_time_ms"]),
            "arms": {"baseline": baseline, "gold_answer_copy": copy},
            "subset_outcomes": subsets,
            "eligible": eligible,
        })
    return sorted(rows, key=lambda row: row["question_id"])


def _classify_topology(row: Mapping[str, Any]) -> str:
    subsets = row["subset_outcomes"]
    if not subsets:
        raise Phase10DError("topology classification requires an eligible question")
    singleton_success = any(item["cardinality"] == 1 and item["em"] == 1.0 for item in subsets)
    successful_multi = [item for item in subsets if item["cardinality"] >= 2 and item["em"] == 1.0]
    distributed = not singleton_success and bool(successful_multi)
    strict = distributed and any(item["contains_non_answer_literal"] for item in successful_multi)
    if singleton_success:
        return "singleton_sufficient"
    if strict:
        return "strict_non_answer_assisted_interaction"
    if distributed:
        return "answer_bearing_only_distributed_interaction"
    return "beyond_selected_set"


def summarize_support_topology(payload: Mapping[str, Any], *, gate: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Summarize a compact Phase 10D result under a frozen screen gate."""

    if stage not in {"screen", "replication"}:
        raise Phase10DError("stage must be screen or replication")
    rows = _validate_rows(payload)
    primary = [row for row in rows if row["selected_hit"] and row["arms"]["baseline"]["em"] == 0.0]
    eligible = [row for row in primary if row["eligible"]]
    counts = {name: 0 for name in TOPOLOGY_CLASSES}
    for row in eligible:
        counts[_classify_topology(row)] += 1
    denominator = len(eligible)
    fractions = {name: (count / denominator if denominator else None) for name, count in counts.items()}
    strict_values = [1.0 if _classify_topology(row) == "strict_non_answer_assisted_interaction" else 0.0 for row in eligible]
    bootstrap = question_bootstrap(
        strict_values,
        repetitions=int(gate["bootstrap_repetitions"]),
        seed=int(gate["bootstrap_seed"]),
    )
    enough_primary = len(primary) >= int(gate["minimum_primary_errors"])
    enough_eligible = denominator >= int(gate["minimum_copy_control_successes"])
    strict_fraction = 0.0 if not denominator else counts["strict_non_answer_assisted_interaction"] / denominator
    fraction_gate = strict_fraction >= float(gate["strict_interaction_fraction_min"])
    ci_gate = bootstrap is not None and bootstrap["ci95_low"] >= float(gate["strict_interaction_bootstrap_ci95_low_min"])
    if not enough_primary:
        decision = "insufficient_primary_errors"
    elif not enough_eligible:
        decision = "insufficient_copy_control_successes"
    elif fraction_gate and ci_gate:
        decision = f"pass_phase10d_{stage}"
    else:
        decision = f"kill_phase10d_{stage}"
    return {
        "schema_version": 1,
        "stage": stage,
        "n_questions": len(rows),
        "cohorts": {
            "retrieval_hit": sum(row["retrieval_hit"] for row in rows),
            "selected_hit": sum(row["selected_hit"] for row in rows),
            "selected_hit_baseline_errors": len(primary),
            "copy_control_successes": denominator,
            "copy_control_failures": len(primary) - denominator,
        },
        "topology": {
            "counts": counts,
            "fractions": fractions,
            "strict_interaction_bootstrap": bootstrap,
        },
        "gate": {
            "enough_primary_errors": enough_primary,
            "enough_copy_control_successes": enough_eligible,
            "strict_interaction_fraction": fraction_gate,
            "strict_interaction_bootstrap_ci95_low": ci_gate,
        },
        "decision": {
            "primary_failure_class": decision,
            "report_only": True,
            "selection_mutation": False,
            "thresholds": {
                "minimum_primary_errors": int(gate["minimum_primary_errors"]),
                "minimum_copy_control_successes": int(gate["minimum_copy_control_successes"]),
                "strict_interaction_fraction_min": float(gate["strict_interaction_fraction_min"]),
                "strict_interaction_bootstrap_ci95_low_min": float(gate["strict_interaction_bootstrap_ci95_low_min"]),
            },
        },
    }


__all__ = [
    "EXPECTED_MASKS",
    "FORBIDDEN_FIELDS",
    "Phase10DError",
    "TOPOLOGY_CLASSES",
    "question_bootstrap",
    "summarize_support_topology",
]
