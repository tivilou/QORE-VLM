"""Compact metrics and report-only gate for Phase 9F."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence


FORBIDDEN_FIELDS = {
    "question", "passages", "gold_answers", "prediction", "raw_prompt",
    "oracle_sentences", "gold_answer",
}
EVIDENCE_ARMS = (
    "sentence_plain", "sentence_highlight", "window_plain", "window_highlight"
)
ALL_ARMS = ("baseline", "gold_answer_copy", *EVIDENCE_ARMS)


class Phase9FError(ValueError):
    """Raised when compact Phase 9F data violates its frozen contract."""


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase9FError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase9FError("cannot compute percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def paired_bootstrap(
    values: Sequence[float], *, repetitions: int, seed: int
) -> dict[str, float] | None:
    if not values:
        return None
    if repetitions < 100:
        raise Phase9FError("bootstrap requires at least 100 repetitions")
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


def _validate_arm(
    arm: Mapping[str, Any], *, attempted: bool, name: str, question_id: str
) -> dict[str, Any]:
    if set(arm) != {"attempted", "em", "f1", "generation_time_ms"}:
        raise Phase9FError(f"{question_id}/{name}: unexpected arm schema")
    if arm.get("attempted") is not attempted:
        raise Phase9FError(f"{question_id}/{name}: attempted flag mismatch")
    normalized = {"attempted": attempted}
    for key in ("em", "f1", "generation_time_ms"):
        value = arm.get(key)
        if attempted:
            if not _finite(value):
                raise Phase9FError(f"{question_id}/{name}: missing finite {key}")
            if key in {"em", "f1"} and not 0.0 <= float(value) <= 1.0:
                raise Phase9FError(f"{question_id}/{name}: {key} out of range")
            if key == "generation_time_ms" and float(value) < 0.0:
                raise Phase9FError(f"{question_id}/{name}: negative generation time")
            normalized[key] = float(value)
        elif value is not None:
            raise Phase9FError(f"{question_id}/{name}: unattempted arm has {key}")
        else:
            normalized[key] = None
    return normalized


def _validate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    forbidden = _find_forbidden(payload)
    if forbidden:
        raise Phase9FError(f"forbidden fields present: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase9FError("samples must be a non-empty list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != {
            "question_id", "retrieval_hit", "selected_hit", "evidence_match_found",
            "evidence_match_count", "sentence_variant_count", "window_variant_count",
            "selection_time_ms", "arms",
        }:
            raise Phase9FError("unexpected sample schema")
        question_id = str(sample.get("question_id", ""))
        if not question_id or question_id in seen:
            raise Phase9FError(f"invalid or duplicate question_id {question_id!r}")
        seen.add(question_id)
        retrieval_hit = sample.get("retrieval_hit")
        selected_hit = sample.get("selected_hit")
        evidence_found = sample.get("evidence_match_found")
        if not all(isinstance(value, bool) for value in (retrieval_hit, selected_hit, evidence_found)):
            raise Phase9FError(f"{question_id}: hit flags must be boolean")
        if selected_hit and not retrieval_hit:
            raise Phase9FError(f"{question_id}: selected hit requires retrieval hit")
        if evidence_found != selected_hit:
            raise Phase9FError(f"{question_id}: evidence match must track selected hit")
        counts = [
            sample.get("evidence_match_count"), sample.get("sentence_variant_count"),
            sample.get("window_variant_count"),
        ]
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts):
            raise Phase9FError(f"{question_id}: invalid evidence counts")
        if evidence_found != (counts[0] > 0) or counts[0] != counts[1] or counts[0] != counts[2]:
            raise Phase9FError(f"{question_id}: evidence counts disagree")
        selection_time = sample.get("selection_time_ms")
        if not _finite(selection_time) or float(selection_time) < 0.0:
            raise Phase9FError(f"{question_id}: invalid selection time")
        arms = sample.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(ALL_ARMS):
            raise Phase9FError(f"{question_id}: unexpected arms")
        baseline = _validate_arm(arms["baseline"], attempted=True, name="baseline", question_id=question_id)
        primary_error = bool(selected_hit and baseline["em"] == 0.0)
        copy = _validate_arm(
            arms["gold_answer_copy"], attempted=primary_error,
            name="gold_answer_copy", question_id=question_id,
        )
        eligible = bool(primary_error and copy["em"] == 1.0)
        evidence_arms = {
            name: _validate_arm(arms[name], attempted=eligible, name=name, question_id=question_id)
            for name in EVIDENCE_ARMS
        }
        normalized.append({
            "question_id": question_id,
            "retrieval_hit": retrieval_hit,
            "selected_hit": selected_hit,
            "evidence_match_found": evidence_found,
            "evidence_match_count": counts[0],
            "sentence_variant_count": counts[1],
            "window_variant_count": counts[2],
            "selection_time_ms": float(selection_time),
            "arms": {"baseline": baseline, "gold_answer_copy": copy, **evidence_arms},
        })
    return sorted(normalized, key=lambda row: row["question_id"])


def _arm_summary(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    attempted = [row["arms"][name] for row in rows if row["arms"][name]["attempted"]]
    return {
        "n_attempted": len(attempted),
        "em_mean": None if not attempted else _mean([row["em"] for row in attempted]),
        "f1_mean": None if not attempted else _mean([row["f1"] for row in attempted]),
        "generation_time_ms_mean": None if not attempted else _mean(
            [row["generation_time_ms"] for row in attempted]
        ),
    }


def summarize_evidence_sufficiency(
    payload: Mapping[str, Any], *, gate: Mapping[str, Any]
) -> dict[str, Any]:
    rows = _validate_rows(payload)
    selected = [row for row in rows if row["selected_hit"]]
    primary = [
        row for row in selected if row["arms"]["baseline"]["em"] == 0.0
    ]
    eligible = [
        row for row in primary if row["arms"]["gold_answer_copy"]["em"] == 1.0
    ]
    attribution = {
        "exact_sentence_sufficient": 0,
        "answer_localization_only": 0,
        "local_context_only": 0,
        "both_main_effects": 0,
        "interaction_only": 0,
        "beyond_local_context": 0,
    }
    for row in eligible:
        arms = row["arms"]
        if arms["sentence_plain"]["em"] == 1.0:
            attribution["exact_sentence_sufficient"] += 1
        else:
            highlight = arms["sentence_highlight"]["em"] == 1.0
            window = arms["window_plain"]["em"] == 1.0
            combined = arms["window_highlight"]["em"] == 1.0
            if highlight and window:
                attribution["both_main_effects"] += 1
            elif highlight:
                attribution["answer_localization_only"] += 1
            elif window:
                attribution["local_context_only"] += 1
            elif combined:
                attribution["interaction_only"] += 1
            else:
                attribution["beyond_local_context"] += 1

    denominator = len(eligible)
    fractions = {
        key: (value / denominator if denominator else None)
        for key, value in attribution.items()
    }
    ranked = sorted(attribution, key=lambda key: (-attribution[key], key))
    minimum_errors = int(gate.get("minimum_primary_errors", 10))
    minimum_copy = int(gate.get("minimum_copy_control_successes", 8))
    minimum_fraction = float(gate.get("minimum_dominant_fraction", 0.40))
    minimum_margin = float(gate.get("minimum_dominance_margin", 0.10))
    dominant = ranked[0]
    runner_up = ranked[1]
    dominant_fraction = 0.0 if not denominator else attribution[dominant] / denominator
    runner_up_fraction = 0.0 if not denominator else attribution[runner_up] / denominator
    if len(primary) < minimum_errors:
        decision = "insufficient_primary_errors"
    elif denominator < minimum_copy:
        decision = "insufficient_copy_control_successes"
    elif dominant_fraction < minimum_fraction or dominant_fraction - runner_up_fraction < minimum_margin:
        decision = "mixed_evidence_failure_modes"
    else:
        decision = f"{dominant}_dominant"

    repetitions = int(gate.get("bootstrap_repetitions", 2000))
    seed = int(gate.get("bootstrap_seed", 9601))
    paired: dict[str, Any] = {}
    for index, name in enumerate(EVIDENCE_ARMS):
        deltas = [
            row["arms"][name]["f1"] - row["arms"]["baseline"]["f1"]
            for row in eligible
        ]
        paired[name] = paired_bootstrap(deltas, repetitions=repetitions, seed=seed + index)

    return {
        "schema_version": 1,
        "n_questions": len(rows),
        "cohorts": {
            "retrieval_hit": sum(row["retrieval_hit"] for row in rows),
            "selected_hit": len(selected),
            "selected_hit_baseline_errors": len(primary),
            "copy_control_successes": denominator,
            "copy_control_failures": len(primary) - denominator,
        },
        "arms": {name: _arm_summary(rows, name) for name in ALL_ARMS},
        "primary_attribution": {
            "counts": attribution,
            "fractions": fractions,
            "paired_f1_deltas": paired,
        },
        "decision": {
            "primary_failure_class": decision,
            "dominant_class": dominant,
            "dominant_fraction": dominant_fraction,
            "runner_up_fraction": runner_up_fraction,
            "report_only": True,
            "thresholds": {
                "minimum_primary_errors": minimum_errors,
                "minimum_copy_control_successes": minimum_copy,
                "minimum_dominant_fraction": minimum_fraction,
                "minimum_dominance_margin": minimum_margin,
            },
        },
    }


__all__ = [
    "ALL_ARMS", "EVIDENCE_ARMS", "FORBIDDEN_FIELDS", "Phase9FError",
    "paired_bootstrap", "summarize_evidence_sufficiency",
]
