"""Compact metrics and gate for Phase 9E generation-error decomposition."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence


FORBIDDEN_FIELDS = {
    "question", "passages", "gold_answers", "prediction", "raw_prompt",
    "oracle_sentences", "gold_answer",
}
PRIMARY_ARMS = ("baseline", "extractive", "oracle_context", "oracle_extractive")


class Phase9EError(ValueError):
    """Raised when a compact Phase 9E result violates its frozen contract."""


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase9EError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase9EError("cannot compute percentile of an empty sequence")
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
        raise Phase9EError("bootstrap requires at least 100 repetitions")
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
    if arm.get("attempted") is not attempted:
        raise Phase9EError(f"{question_id}/{name}: attempted flag mismatch")
    metrics = {"attempted": attempted}
    for key in ("em", "f1", "generation_time_ms"):
        value = arm.get(key)
        if attempted:
            if not _finite(value):
                raise Phase9EError(f"{question_id}/{name}: missing finite {key}")
            if key in {"em", "f1"} and not 0.0 <= float(value) <= 1.0:
                raise Phase9EError(f"{question_id}/{name}: {key} out of range")
            if key == "generation_time_ms" and float(value) < 0.0:
                raise Phase9EError(f"{question_id}/{name}: negative generation time")
            metrics[key] = float(value)
        elif value is not None:
            raise Phase9EError(f"{question_id}/{name}: unattempted arm has {key}")
        else:
            metrics[key] = None
    return metrics


def _validate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    forbidden = _find_forbidden(payload)
    if forbidden:
        raise Phase9EError(f"forbidden fields present: {forbidden[:5]}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase9EError("samples must be a non-empty list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping) or not sample.get("question_id"):
            raise Phase9EError("every sample needs question_id")
        question_id = str(sample["question_id"])
        if question_id in seen:
            raise Phase9EError(f"duplicate question_id {question_id}")
        seen.add(question_id)
        retrieval_hit = sample.get("retrieval_hit")
        selected_hit = sample.get("selected_hit")
        oracle_found = sample.get("oracle_sentence_found")
        oracle_count = sample.get("oracle_sentence_count")
        if not all(isinstance(value, bool) for value in (retrieval_hit, selected_hit, oracle_found)):
            raise Phase9EError(f"{question_id}: hit flags must be boolean")
        if selected_hit and not retrieval_hit:
            raise Phase9EError(f"{question_id}: selected hit requires retrieval hit")
        if oracle_found != selected_hit:
            raise Phase9EError(f"{question_id}: oracle sentence must exactly track selected hit")
        if not isinstance(oracle_count, int) or isinstance(oracle_count, bool) or oracle_count < 0:
            raise Phase9EError(f"{question_id}: invalid oracle sentence count")
        if oracle_found != (oracle_count > 0):
            raise Phase9EError(f"{question_id}: oracle count disagrees with flag")
        arms = sample.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {*PRIMARY_ARMS, "gold_answer_copy"}:
            raise Phase9EError(f"{question_id}: unexpected arm schema")
        baseline = _validate_arm(arms["baseline"], attempted=True, name="baseline", question_id=question_id)
        extractive = _validate_arm(arms["extractive"], attempted=True, name="extractive", question_id=question_id)
        oracle_context = _validate_arm(
            arms["oracle_context"], attempted=selected_hit, name="oracle_context", question_id=question_id
        )
        oracle_extractive = _validate_arm(
            arms["oracle_extractive"], attempted=selected_hit, name="oracle_extractive", question_id=question_id
        )
        copy_expected = bool(
            selected_hit and baseline["em"] == 0.0 and oracle_extractive["em"] == 0.0
        )
        gold_copy = _validate_arm(
            arms["gold_answer_copy"], attempted=copy_expected,
            name="gold_answer_copy", question_id=question_id,
        )
        normalized.append({
            "question_id": question_id,
            "retrieval_hit": retrieval_hit,
            "selected_hit": selected_hit,
            "oracle_sentence_found": oracle_found,
            "oracle_sentence_count": oracle_count,
            "arms": {
                "baseline": baseline,
                "extractive": extractive,
                "oracle_context": oracle_context,
                "oracle_extractive": oracle_extractive,
                "gold_answer_copy": gold_copy,
            },
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


def summarize_generation_errors(
    payload: Mapping[str, Any], *, gate: Mapping[str, Any]
) -> dict[str, Any]:
    rows = _validate_rows(payload)
    selected = [row for row in rows if row["selected_hit"]]
    primary = [row for row in selected if row["arms"]["baseline"]["em"] == 0.0]
    repetitions = int(gate.get("bootstrap_repetitions", 2000))
    seed = int(gate.get("bootstrap_seed", 9501))

    attribution = {
        "prompt_format": 0,
        "context_localization": 0,
        "both_main_effects": 0,
        "interaction_only": 0,
        "lexical_evidence_insufficiency": 0,
        "generator_copy_failure": 0,
    }
    for row in primary:
        arms = row["arms"]
        extractive = arms["extractive"]["em"] == 1.0
        oracle = arms["oracle_context"]["em"] == 1.0
        combined = arms["oracle_extractive"]["em"] == 1.0
        if extractive and oracle:
            attribution["both_main_effects"] += 1
        elif extractive:
            attribution["prompt_format"] += 1
        elif oracle:
            attribution["context_localization"] += 1
        elif combined:
            attribution["interaction_only"] += 1
        elif arms["gold_answer_copy"]["em"] == 1.0:
            attribution["lexical_evidence_insufficiency"] += 1
        else:
            attribution["generator_copy_failure"] += 1

    denominator = len(primary)
    fractions = {
        key: (value / denominator if denominator else None)
        for key, value in attribution.items()
    }
    ranked = sorted(attribution, key=lambda key: (-attribution[key], key))
    minimum_errors = int(gate.get("minimum_primary_errors", 10))
    minimum_fraction = float(gate.get("minimum_dominant_fraction", 0.40))
    minimum_margin = float(gate.get("minimum_dominance_margin", 0.10))
    dominant = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    dominant_fraction = 0.0 if not denominator else attribution[dominant] / denominator
    runner_up_fraction = 0.0 if not denominator or runner_up is None else attribution[runner_up] / denominator
    if denominator < minimum_errors:
        decision = "insufficient_primary_errors"
    elif dominant_fraction < minimum_fraction or dominant_fraction - runner_up_fraction < minimum_margin:
        decision = "mixed_generation_errors"
    else:
        decision = f"{dominant}_dominant"

    paired_deltas: dict[str, Any] = {}
    for index, name in enumerate(("extractive", "oracle_context", "oracle_extractive")):
        deltas = [
            row["arms"][name]["f1"] - row["arms"]["baseline"]["f1"]
            for row in primary
        ]
        paired_deltas[name] = paired_bootstrap(
            deltas, repetitions=repetitions, seed=seed + index
        )

    return {
        "schema_version": 1,
        "n_questions": len(rows),
        "cohorts": {
            "retrieval_hit": sum(row["retrieval_hit"] for row in rows),
            "selected_hit": len(selected),
            "selected_hit_baseline_errors": denominator,
        },
        "arms": {
            name: _arm_summary(rows, name)
            for name in (*PRIMARY_ARMS, "gold_answer_copy")
        },
        "primary_error_attribution": {
            "counts": attribution,
            "fractions": fractions,
            "paired_f1_deltas": paired_deltas,
        },
        "decision": {
            "primary_failure_class": decision,
            "dominant_class": dominant,
            "dominant_fraction": dominant_fraction,
            "runner_up_fraction": runner_up_fraction,
            "thresholds": {
                "minimum_primary_errors": minimum_errors,
                "minimum_dominant_fraction": minimum_fraction,
                "minimum_dominance_margin": minimum_margin,
            },
        },
    }


__all__ = [
    "FORBIDDEN_FIELDS",
    "PRIMARY_ARMS",
    "Phase9EError",
    "paired_bootstrap",
    "summarize_generation_errors",
]
