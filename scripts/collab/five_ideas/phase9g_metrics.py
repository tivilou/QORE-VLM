"""Compact metrics and report-only gate for Phase 9G."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence

FORBIDDEN_FIELDS = {"question", "passages", "gold_answers", "prediction", "raw_prompt", "oracle_passages", "gold_answer"}
CONTEXT_ARMS = ("full_plain", "full_highlight", "answer_first", "retrieved_answer_oracle")
ALL_ARMS = ("baseline", "gold_answer_copy", *CONTEXT_ARMS)
ATTRIBUTION_CLASSES = (
    "full_passage_sufficient",
    "full_passage_localization",
    "answer_passage_order",
    "multi_passage_evidence",
    "retrieved_oracle_nonincremental",
    "beyond_top50_context",
)


class Phase9GError(ValueError):
    pass


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase9GError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase9GError("cannot compute percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(values: Sequence[float], *, repetitions: int, seed: int) -> dict[str, float] | None:
    if not values:
        return None
    if repetitions < 100:
        raise Phase9GError("bootstrap requires at least 100 repetitions")
    rng = random.Random(seed)
    draws = [_mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(repetitions)]
    return {"mean": _mean(values), "ci95_low": _percentile(draws, .025), "ci95_high": _percentile(draws, .975), "bootstrap_repetitions": repetitions, "bootstrap_seed": seed}


def _find_forbidden(value: Any, path: str = "$root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in FORBIDDEN_FIELDS:
                findings.append(child_path)
            findings.extend(_find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden(child, f"{path}[{index}]") )
    return findings


def _validate_arm(arm: Mapping[str, Any], *, attempted: bool, name: str, question_id: str) -> dict[str, Any]:
    if set(arm) != {"attempted", "em", "f1", "generation_time_ms"}:
        raise Phase9GError(f"{question_id}/{name}: unexpected arm schema")
    if arm.get("attempted") is not attempted:
        raise Phase9GError(f"{question_id}/{name}: attempted flag mismatch")
    normalized = {"attempted": attempted}
    for key in ("em", "f1", "generation_time_ms"):
        value = arm.get(key)
        if attempted:
            if not _finite(value):
                raise Phase9GError(f"{question_id}/{name}: missing finite {key}")
            if key in {"em", "f1"} and not 0 <= float(value) <= 1:
                raise Phase9GError(f"{question_id}/{name}: {key} out of range")
            if key == "generation_time_ms" and float(value) < 0:
                raise Phase9GError(f"{question_id}/{name}: negative generation time")
            normalized[key] = float(value)
        elif value is not None:
            raise Phase9GError(f"{question_id}/{name}: unattempted arm has {key}")
        else:
            normalized[key] = None
    return normalized


def _empty() -> dict[str, Any]:
    return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}


def _validate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    forbidden = _find_forbidden(payload)
    if forbidden:
        raise Phase9GError(f"forbidden fields present: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase9GError("samples must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {"question_id", "retrieval_hit", "selected_hit", "selected_match_count", "oracle_match_count", "answer_first_changed", "selection_time_ms", "arms"}
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != required:
            raise Phase9GError("unexpected sample schema")
        question_id = str(sample.get("question_id", ""))
        if not question_id or question_id in seen:
            raise Phase9GError(f"invalid or duplicate question_id {question_id!r}")
        seen.add(question_id)
        retrieval_hit, selected_hit = sample.get("retrieval_hit"), sample.get("selected_hit")
        if not isinstance(retrieval_hit, bool) or not isinstance(selected_hit, bool) or (selected_hit and not retrieval_hit):
            raise Phase9GError(f"{question_id}: invalid hit flags")
        selected_count, oracle_count = sample.get("selected_match_count"), sample.get("oracle_match_count")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (selected_count, oracle_count)):
            raise Phase9GError(f"{question_id}: invalid match counts")
        if selected_hit != (selected_count > 0) or retrieval_hit != (oracle_count > 0):
            raise Phase9GError(f"{question_id}: hit/count mismatch")
        if oracle_count < selected_count:
            raise Phase9GError(f"{question_id}: oracle count is below selected count")
        answer_first_changed = sample.get("answer_first_changed")
        if not isinstance(answer_first_changed, bool):
            raise Phase9GError(f"{question_id}: answer_first_changed must be boolean")
        if not _finite(sample.get("selection_time_ms")) or float(sample["selection_time_ms"]) < 0:
            raise Phase9GError(f"{question_id}: invalid selection time")
        arms = sample.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(ALL_ARMS):
            raise Phase9GError(f"{question_id}: unexpected arms")
        baseline = _validate_arm(arms["baseline"], attempted=True, name="baseline", question_id=question_id)
        primary = bool(selected_hit and baseline["em"] == 0)
        copy = _validate_arm(arms["gold_answer_copy"], attempted=primary, name="gold_answer_copy", question_id=question_id)
        eligible = bool(primary and copy["em"] == 1)
        context = {name: _validate_arm(arms[name], attempted=eligible, name=name, question_id=question_id) for name in CONTEXT_ARMS}
        if eligible and not answer_first_changed and context["answer_first"]["em"] != baseline["em"]:
            raise Phase9GError(f"{question_id}: unchanged answer-first arm differs from baseline")
        normalized.append({"question_id": question_id, "retrieval_hit": retrieval_hit, "selected_hit": selected_hit, "selected_match_count": selected_count, "oracle_match_count": oracle_count, "answer_first_changed": answer_first_changed, "selection_time_ms": float(sample["selection_time_ms"]), "arms": {"baseline": baseline, "gold_answer_copy": copy, **context}})
    return sorted(normalized, key=lambda row: row["question_id"])


def _arm_summary(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    attempted = [row["arms"][name] for row in rows if row["arms"][name]["attempted"]]
    return {"n_attempted": len(attempted), "em_mean": None if not attempted else _mean([row["em"] for row in attempted]), "f1_mean": None if not attempted else _mean([row["f1"] for row in attempted]), "generation_time_ms_mean": None if not attempted else _mean([row["generation_time_ms"] for row in attempted])}


def summarize_context_failure(payload: Mapping[str, Any], *, gate: Mapping[str, Any]) -> dict[str, Any]:
    rows = _validate_rows(payload)
    primary = [row for row in rows if row["selected_hit"] and row["arms"]["baseline"]["em"] == 0]
    eligible = [row for row in primary if row["arms"]["gold_answer_copy"]["em"] == 1]
    counts = {name: 0 for name in ATTRIBUTION_CLASSES}
    for row in eligible:
        arms = row["arms"]
        if arms["full_plain"]["em"] == 1:
            counts["full_passage_sufficient"] += 1
        elif arms["full_highlight"]["em"] == 1:
            counts["full_passage_localization"] += 1
        elif arms["answer_first"]["em"] == 1 and row["answer_first_changed"]:
            counts["answer_passage_order"] += 1
        elif arms["retrieved_answer_oracle"]["em"] == 1 and row["oracle_match_count"] > row["selected_match_count"]:
            counts["multi_passage_evidence"] += 1
        elif arms["retrieved_answer_oracle"]["em"] == 1:
            counts["retrieved_oracle_nonincremental"] += 1
        else:
            counts["beyond_top50_context"] += 1
    denominator = len(eligible)
    fractions = {key: (value / denominator if denominator else None) for key, value in counts.items()}
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    dominant, runner_up = ranked[0], ranked[1]
    dominant_fraction = 0 if not denominator else counts[dominant] / denominator
    runner_up_fraction = 0 if not denominator else counts[runner_up] / denominator
    minimum_errors = int(gate.get("minimum_primary_errors", 10))
    minimum_copy = int(gate.get("minimum_copy_control_successes", 8))
    if len(primary) < minimum_errors:
        decision = "insufficient_primary_errors"
    elif denominator < minimum_copy:
        decision = "insufficient_copy_control_successes"
    elif dominant_fraction < float(gate.get("minimum_dominant_fraction", .4)) or dominant_fraction - runner_up_fraction < float(gate.get("minimum_dominance_margin", .1)):
        decision = "mixed_context_failure_modes"
    else:
        decision = f"{dominant}_dominant"
    repetitions, seed = int(gate.get("bootstrap_repetitions", 2000)), int(gate.get("bootstrap_seed", 9701))
    deltas = {name: paired_bootstrap([row["arms"][name]["f1"] - row["arms"]["baseline"]["f1"] for row in eligible], repetitions=repetitions, seed=seed + index) for index, name in enumerate(CONTEXT_ARMS)}
    return {"schema_version": 1, "n_questions": len(rows), "cohorts": {"retrieval_hit": sum(row["retrieval_hit"] for row in rows), "selected_hit": sum(row["selected_hit"] for row in rows), "selected_hit_baseline_errors": len(primary), "copy_control_successes": denominator, "copy_control_failures": len(primary) - denominator}, "arms": {name: _arm_summary(rows, name) for name in ALL_ARMS}, "primary_attribution": {"counts": counts, "fractions": fractions, "paired_f1_deltas": deltas}, "decision": {"primary_failure_class": decision, "dominant_class": dominant, "dominant_fraction": dominant_fraction, "runner_up_fraction": runner_up_fraction, "report_only": True, "thresholds": {"minimum_primary_errors": minimum_errors, "minimum_copy_control_successes": minimum_copy, "minimum_dominant_fraction": float(gate.get("minimum_dominant_fraction", .4)), "minimum_dominance_margin": float(gate.get("minimum_dominance_margin", .1))}}}


__all__ = ["ALL_ARMS", "CONTEXT_ARMS", "FORBIDDEN_FIELDS", "Phase9GError", "paired_bootstrap", "summarize_context_failure"]
