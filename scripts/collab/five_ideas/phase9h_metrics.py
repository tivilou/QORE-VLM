"""Compact metrics and report-only gate for Phase 9H."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence

FORBIDDEN_FIELDS = {
    "question", "passages", "gold_answers", "prediction", "raw_prompt",
    "gold_answer", "support_reason",
}
SUPPORT_LABELS = ("supported", "unsupported", "uncertain")
ATTRIBUTION_CLASSES = (
    "model_judge_unsupported",
    "answer_extraction_contract",
    "answer_format_mismatch",
    "semantic_generation_failure",
    "unresolved_residual",
    "conflicting_signals",
)
STANDARD_ARMS = ("baseline", "extractive", "gold_answer_copy")


class Phase9HError(ValueError):
    """Raised when compact Phase 9H data violates its frozen contract."""


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase9HError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase9HError("cannot compute percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(
    values: Sequence[float], *, repetitions: int, seed: int
) -> dict[str, float] | None:
    if not values:
        return None
    if repetitions < 100:
        raise Phase9HError("bootstrap requires at least 100 repetitions")
    rng = random.Random(seed)
    draws = [
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(repetitions)
    ]
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
            if str(key) in FORBIDDEN_FIELDS:
                findings.append(child_path)
            findings.extend(_find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden(child, f"{path}[{index}]"))
    return findings


def _validate_standard_arm(
    arm: Mapping[str, Any], *, attempted: bool, name: str, question_id: str
) -> dict[str, Any]:
    if set(arm) != {"attempted", "em", "f1", "generation_time_ms"}:
        raise Phase9HError(f"{question_id}/{name}: unexpected arm schema")
    if arm.get("attempted") is not attempted:
        raise Phase9HError(f"{question_id}/{name}: attempted flag mismatch")
    normalized = {"attempted": attempted}
    for key in ("em", "f1", "generation_time_ms"):
        value = arm.get(key)
        if attempted:
            if not _finite(value):
                raise Phase9HError(f"{question_id}/{name}: missing finite {key}")
            if key in {"em", "f1"} and not 0.0 <= float(value) <= 1.0:
                raise Phase9HError(f"{question_id}/{name}: {key} out of range")
            if key == "generation_time_ms" and float(value) < 0.0:
                raise Phase9HError(f"{question_id}/{name}: negative generation time")
            normalized[key] = float(value)
        elif value is not None:
            raise Phase9HError(f"{question_id}/{name}: unattempted arm has {key}")
        else:
            normalized[key] = None
    return normalized


def _validate_support_arm(
    arm: Mapping[str, Any], *, attempted: bool, question_id: str
) -> dict[str, Any]:
    if set(arm) != {"attempted", "label", "generation_time_ms"}:
        raise Phase9HError(f"{question_id}/support_judge: unexpected arm schema")
    if arm.get("attempted") is not attempted:
        raise Phase9HError(f"{question_id}/support_judge: attempted flag mismatch")
    if not attempted:
        if arm.get("label") is not None or arm.get("generation_time_ms") is not None:
            raise Phase9HError(f"{question_id}/support_judge: unattempted arm has values")
        return {"attempted": False, "label": None, "generation_time_ms": None}
    label = arm.get("label")
    elapsed = arm.get("generation_time_ms")
    if label not in SUPPORT_LABELS or not _finite(elapsed) or float(elapsed) < 0.0:
        raise Phase9HError(f"{question_id}/support_judge: invalid label or timing")
    return {
        "attempted": True,
        "label": str(label),
        "generation_time_ms": float(elapsed),
    }


def _arm_empty() -> dict[str, Any]:
    return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}


def _support_empty() -> dict[str, Any]:
    return {"attempted": False, "label": None, "generation_time_ms": None}


def _validate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    forbidden = _find_forbidden(payload)
    if forbidden:
        raise Phase9HError(f"forbidden fields present: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase9HError("samples must be a non-empty list")
    required = {
        "question_id", "retrieval_hit", "selected_hit", "selected_match_count",
        "retrieval_match_count", "selection_time_ms", "arms",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != required:
            raise Phase9HError("unexpected sample schema")
        question_id = str(sample.get("question_id", ""))
        if not question_id or question_id in seen:
            raise Phase9HError(f"invalid or duplicate question_id {question_id!r}")
        seen.add(question_id)
        retrieval_hit = sample.get("retrieval_hit")
        selected_hit = sample.get("selected_hit")
        if not isinstance(retrieval_hit, bool) or not isinstance(selected_hit, bool):
            raise Phase9HError(f"{question_id}: hit flags must be boolean")
        if selected_hit and not retrieval_hit:
            raise Phase9HError(f"{question_id}: selected hit requires retrieval hit")
        selected_count = sample.get("selected_match_count")
        retrieval_count = sample.get("retrieval_match_count")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (selected_count, retrieval_count)
        ):
            raise Phase9HError(f"{question_id}: invalid match counts")
        if selected_hit != (selected_count > 0):
            raise Phase9HError(f"{question_id}: selected/count mismatch")
        if retrieval_hit != (retrieval_count > 0):
            raise Phase9HError(f"{question_id}: retrieval/count mismatch")
        if retrieval_count < selected_count:
            raise Phase9HError(f"{question_id}: retrieval count below selected count")
        if not _finite(sample.get("selection_time_ms")) or float(sample["selection_time_ms"]) < 0:
            raise Phase9HError(f"{question_id}: invalid selection time")
        arms = sample.get("arms")
        expected_arms = {"baseline", "extractive", "gold_answer_copy", "support_judge"}
        if not isinstance(arms, Mapping) or set(arms) != expected_arms:
            raise Phase9HError(f"{question_id}: unexpected arms")
        baseline = _validate_standard_arm(
            arms["baseline"], attempted=True, name="baseline", question_id=question_id
        )
        primary = bool(selected_hit and baseline["em"] == 0.0)
        copy = _validate_standard_arm(
            arms["gold_answer_copy"], attempted=primary,
            name="gold_answer_copy", question_id=question_id,
        )
        eligible = bool(primary and copy["em"] == 1.0)
        extractive = _validate_standard_arm(
            arms["extractive"], attempted=eligible,
            name="extractive", question_id=question_id,
        )
        support = _validate_support_arm(
            arms["support_judge"], attempted=eligible, question_id=question_id
        )
        normalized.append({
            "question_id": question_id,
            "retrieval_hit": retrieval_hit,
            "selected_hit": selected_hit,
            "selected_match_count": selected_count,
            "retrieval_match_count": retrieval_count,
            "selection_time_ms": float(sample["selection_time_ms"]),
            "arms": {
                "baseline": baseline,
                "extractive": extractive,
                "gold_answer_copy": copy,
                "support_judge": support,
            },
        })
    return sorted(normalized, key=lambda row: row["question_id"])


def _standard_summary(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    attempted = [row["arms"][name] for row in rows if row["arms"][name]["attempted"]]
    return {
        "n_attempted": len(attempted),
        "em_mean": None if not attempted else _mean([row["em"] for row in attempted]),
        "f1_mean": None if not attempted else _mean([row["f1"] for row in attempted]),
        "generation_time_ms_mean": None if not attempted else _mean(
            [row["generation_time_ms"] for row in attempted]
        ),
    }


def _support_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = [row["arms"]["support_judge"] for row in rows if row["arms"]["support_judge"]["attempted"]]
    counts = {label: sum(row["label"] == label for row in attempted) for label in SUPPORT_LABELS}
    return {
        "n_attempted": len(attempted),
        "label_counts": counts,
        "generation_time_ms_mean": None if not attempted else _mean(
            [row["generation_time_ms"] for row in attempted]
        ),
    }


def summarize_answerability_generation(
    payload: Mapping[str, Any], *, gate: Mapping[str, Any]
) -> dict[str, Any]:
    rows = _validate_rows(payload)
    primary = [row for row in rows if row["selected_hit"] and row["arms"]["baseline"]["em"] == 0.0]
    eligible = [row for row in primary if row["arms"]["gold_answer_copy"]["em"] == 1.0]
    counts = {name: 0 for name in ATTRIBUTION_CLASSES}
    for row in eligible:
        baseline = row["arms"]["baseline"]
        extractive = row["arms"]["extractive"]
        support = row["arms"]["support_judge"]["label"]
        if support == "unsupported" and extractive["em"] == 1.0:
            counts["conflicting_signals"] += 1
        elif support == "unsupported":
            counts["model_judge_unsupported"] += 1
        elif extractive["em"] == 1.0:
            counts["answer_extraction_contract"] += 1
        elif baseline["f1"] > 0.0:
            counts["answer_format_mismatch"] += 1
        elif support == "supported":
            counts["semantic_generation_failure"] += 1
        else:
            counts["unresolved_residual"] += 1
    denominator = len(eligible)
    fractions = {
        key: (value / denominator if denominator else None)
        for key, value in counts.items()
    }
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    dominant, runner_up = ranked[0], ranked[1]
    dominant_fraction = 0.0 if not denominator else counts[dominant] / denominator
    runner_up_fraction = 0.0 if not denominator else counts[runner_up] / denominator
    minimum_errors = int(gate.get("minimum_primary_errors", 10))
    minimum_copy = int(gate.get("minimum_copy_control_successes", 8))
    minimum_fraction = float(gate.get("minimum_dominant_fraction", 0.40))
    minimum_margin = float(gate.get("minimum_dominance_margin", 0.10))
    if len(primary) < minimum_errors:
        decision = "insufficient_primary_errors"
    elif denominator < minimum_copy:
        decision = "insufficient_copy_control_successes"
    elif dominant_fraction < minimum_fraction or dominant_fraction - runner_up_fraction < minimum_margin:
        decision = "mixed_answerability_generation_failures"
    else:
        decision = f"{dominant}_dominant"
    repetitions = int(gate.get("bootstrap_repetitions", 2000))
    seed = int(gate.get("bootstrap_seed", 9801))
    deltas = [
        row["arms"]["extractive"]["f1"] - row["arms"]["baseline"]["f1"]
        for row in eligible
    ]
    return {
        "schema_version": 1,
        "n_questions": len(rows),
        "cohorts": {
            "retrieval_hit": sum(row["retrieval_hit"] for row in rows),
            "selected_hit": sum(row["selected_hit"] for row in rows),
            "selected_hit_baseline_errors": len(primary),
            "copy_control_successes": denominator,
            "copy_control_failures": len(primary) - denominator,
        },
        "arms": {
            "baseline": _standard_summary(rows, "baseline"),
            "extractive": _standard_summary(rows, "extractive"),
            "gold_answer_copy": _standard_summary(rows, "gold_answer_copy"),
            "support_judge": _support_summary(rows),
        },
        "primary_attribution": {
            "counts": counts,
            "fractions": fractions,
            "paired_f1_deltas": {
                "extractive": paired_bootstrap(
                    deltas, repetitions=repetitions, seed=seed
                )
            },
        },
        "decision": {
            "primary_failure_class": decision,
            "dominant_class": dominant,
            "dominant_fraction": dominant_fraction,
            "runner_up_fraction": runner_up_fraction,
            "report_only": True,
            "support_judge_is_model_signal": True,
            "thresholds": {
                "minimum_primary_errors": minimum_errors,
                "minimum_copy_control_successes": minimum_copy,
                "minimum_dominant_fraction": minimum_fraction,
                "minimum_dominance_margin": minimum_margin,
            },
        },
    }


__all__ = [
    "ATTRIBUTION_CLASSES",
    "FORBIDDEN_FIELDS",
    "Phase9HError",
    "paired_bootstrap",
    "summarize_answerability_generation",
]
