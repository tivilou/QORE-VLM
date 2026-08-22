"""Compact metrics and gates for the Phase 9I diagnostic."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence


FORBIDDEN_FIELDS = {
    "question",
    "passages",
    "gold_answers",
    "gold_answer",
    "candidate_text",
    "prediction",
    "raw_prompt",
    "verifier_raw_output",
    "evaluator_trace",
    "support_reason",
}
CANDIDATE_STATUSES = {"ok", "abstain", "empty", "invalid"}
STANDARD_ARMS = ("baseline", "gold_answer_copy")
CANDIDATE_ARMS = ("extractive_span", "evidence_constrained")


class Phase9IError(ValueError):
    """Raised when compact Phase 9I data violates its contract."""


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase9IError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase9IError("cannot compute percentile of an empty sequence")
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
        raise Phase9IError("bootstrap requires at least 100 repetitions")
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


def _require_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise Phase9IError(
            f"{label}: expected keys {sorted(expected)}, got {sorted(value)}"
        )


def _validate_standard_arm(
    arm: Mapping[str, Any], *, attempted: bool, label: str
) -> None:
    _require_keys(
        arm,
        {"attempted", "em", "f1", "generation_time_ms"},
        label,
    )
    if arm["attempted"] is not attempted:
        raise Phase9IError(f"{label}: attempted mismatch")
    for key in ("em", "f1", "generation_time_ms"):
        if attempted:
            if not _finite(arm[key]) or float(arm[key]) < 0.0:
                raise Phase9IError(f"{label}: invalid {key}")
        elif arm[key] is not None:
            raise Phase9IError(f"{label}: non-attempted {key} must be null")
    if attempted and float(arm["em"]) > 1.0:
        raise Phase9IError(f"{label}: EM outside [0,1]")


def _validate_candidate_arm(
    arm: Mapping[str, Any], *, attempted: bool, label: str
) -> None:
    _require_keys(
        arm,
        {"attempted", "parse_status", "em", "f1", "generation_time_ms"},
        label,
    )
    if arm["attempted"] is not attempted:
        raise Phase9IError(f"{label}: attempted mismatch")
    if arm["parse_status"] not in CANDIDATE_STATUSES:
        raise Phase9IError(f"{label}: invalid parse_status")
    valid = attempted and arm["parse_status"] in {"ok", "abstain"}
    for key in ("em", "f1", "generation_time_ms"):
        if valid:
            if not _finite(arm[key]) or float(arm[key]) < 0.0:
                raise Phase9IError(f"{label}: invalid {key}")
        elif arm[key] is not None:
            raise Phase9IError(f"{label}: {key} must be null when not scored")
    if valid and float(arm["em"]) > 1.0:
        raise Phase9IError(f"{label}: EM outside [0,1]")


def _validate_sample(sample: Mapping[str, Any], question_id: str) -> None:
    expected = {
        "question_id",
        "retrieval_hit",
        "selected_hit",
        "retrieval_match_count",
        "selected_match_count",
        "primary_error",
        "copy_control_success",
        "eligible",
        "selection_time_ms",
        "arms",
        "candidate_set",
        "verifier",
    }
    _require_keys(sample, expected, question_id)
    if sample["question_id"] != question_id:
        raise Phase9IError(f"{question_id}: question_id mismatch")
    for key in (
        "retrieval_hit",
        "selected_hit",
        "primary_error",
        "copy_control_success",
        "eligible",
    ):
        if not isinstance(sample[key], bool):
            raise Phase9IError(f"{question_id}: {key} must be boolean")
    for key in ("retrieval_match_count", "selected_match_count"):
        if not isinstance(sample[key], int) or isinstance(sample[key], bool):
            raise Phase9IError(f"{question_id}: {key} must be integer")
        if sample[key] < 0:
            raise Phase9IError(f"{question_id}: {key} must be non-negative")
    if not _finite(sample["selection_time_ms"]) or float(sample["selection_time_ms"]) < 0:
        raise Phase9IError(f"{question_id}: invalid selection time")

    arms = sample["arms"]
    if not isinstance(arms, Mapping):
        raise Phase9IError(f"{question_id}: arms must be an object")
    _require_keys(arms, set(STANDARD_ARMS + CANDIDATE_ARMS), f"{question_id}/arms")
    _validate_standard_arm(
        arms["baseline"], attempted=True, label=f"{question_id}/baseline"
    )
    copy_attempted = bool(sample["primary_error"])
    _validate_standard_arm(
        arms["gold_answer_copy"],
        attempted=copy_attempted,
        label=f"{question_id}/gold_answer_copy",
    )
    for name in CANDIDATE_ARMS:
        _validate_candidate_arm(
            arms[name],
            attempted=bool(sample["eligible"]),
            label=f"{question_id}/{name}",
        )
    if sample["primary_error"] != (
        bool(sample["selected_hit"]) and float(arms["baseline"]["em"]) == 0.0
    ):
        raise Phase9IError(f"{question_id}: primary_error condition mismatch")
    if sample["copy_control_success"] != (
        copy_attempted and arms["gold_answer_copy"]["em"] == 1.0
    ):
        raise Phase9IError(f"{question_id}: copy control condition mismatch")
    if sample["eligible"] != (
        bool(sample["primary_error"]) and bool(sample["copy_control_success"])
    ):
        raise Phase9IError(f"{question_id}: eligibility condition mismatch")

    candidate_set = sample["candidate_set"]
    _require_keys(
        candidate_set,
        {
            "attempted",
            "candidate_count",
            "unique_candidate_count",
            "parse_failures",
            "oracle_em",
            "oracle_f1",
        },
        f"{question_id}/candidate_set",
    )
    if candidate_set["attempted"] is not bool(sample["eligible"]):
        raise Phase9IError(f"{question_id}: candidate set condition mismatch")
    for key in ("candidate_count", "unique_candidate_count", "parse_failures"):
        if not isinstance(candidate_set[key], int) or isinstance(candidate_set[key], bool):
            raise Phase9IError(f"{question_id}: {key} must be integer")
        if candidate_set[key] < 0:
            raise Phase9IError(f"{question_id}: {key} must be non-negative")
    if candidate_set["unique_candidate_count"] > candidate_set["candidate_count"]:
        raise Phase9IError(f"{question_id}: unique count exceeds candidate count")
    if candidate_set["attempted"]:
        if not 1 <= candidate_set["candidate_count"] <= 3:
            raise Phase9IError(f"{question_id}: candidate count outside [1,3]")
        if not _finite(candidate_set["oracle_em"]) or not _finite(candidate_set["oracle_f1"]):
            raise Phase9IError(f"{question_id}: invalid oracle metrics")
        if not 0.0 <= float(candidate_set["oracle_em"]) <= 1.0:
            raise Phase9IError(f"{question_id}: oracle EM outside [0,1]")
    else:
        if candidate_set["candidate_count"] != 0 or candidate_set["unique_candidate_count"] != 0:
            raise Phase9IError(f"{question_id}: empty candidate set has nonzero count")
        if candidate_set["oracle_em"] is not None or candidate_set["oracle_f1"] is not None:
            raise Phase9IError(f"{question_id}: empty candidate set has metrics")

    verifier = sample["verifier"]
    _require_keys(
        verifier,
        {
            "attempted",
            "original_choice_mode",
            "permuted_choice_mode",
            "order_agreement",
            "parse_status",
            "em",
            "f1",
            "generation_time_ms",
        },
        f"{question_id}/verifier",
    )
    if verifier["attempted"] is not bool(sample["eligible"]):
        raise Phase9IError(f"{question_id}: verifier condition mismatch")
    if verifier["parse_status"] not in {
        "ok",
        "invalid_choice",
        "no_candidates",
        "not_attempted",
    }:
        raise Phase9IError(f"{question_id}: invalid verifier status")
    if verifier["attempted"]:
        if verifier["order_agreement"] not in {True, False, None}:
            raise Phase9IError(f"{question_id}: invalid order agreement")
        valid_choice = (
            verifier["parse_status"] == "ok"
            and isinstance(verifier["original_choice_mode"], str)
            and isinstance(verifier["permuted_choice_mode"], str)
        )
        if valid_choice:
            if not isinstance(verifier["order_agreement"], bool):
                raise Phase9IError(f"{question_id}: valid verifier needs order agreement")
            if not _finite(verifier["em"]) or not _finite(verifier["f1"]):
                raise Phase9IError(f"{question_id}: invalid verifier metrics")
            if not 0.0 <= float(verifier["em"]) <= 1.0:
                raise Phase9IError(f"{question_id}: verifier EM outside [0,1]")
        else:
            if verifier["em"] is not None or verifier["f1"] is not None:
                raise Phase9IError(f"{question_id}: invalid verifier has metrics")
    else:
        if verifier["parse_status"] != "not_attempted":
            raise Phase9IError(f"{question_id}: non-attempted verifier status")
        for key in (
            "original_choice_mode",
            "permuted_choice_mode",
            "order_agreement",
            "em",
            "f1",
            "generation_time_ms",
        ):
            if verifier[key] is not None:
                raise Phase9IError(f"{question_id}: non-attempted verifier field {key}")
    if verifier["attempted"]:
        if not _finite(verifier["generation_time_ms"]) or verifier["generation_time_ms"] < 0:
            raise Phase9IError(f"{question_id}: invalid verifier time")


def validate_result(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        raise Phase9IError("result must be an object")
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase9IError(f"forbidden compact fields: {forbidden[:5]}")
    required = {
        "schema_version",
        "phase",
        "stage",
        "diagnostic_only",
        "selection_mutation",
        "report_only",
        "samples",
    }
    _require_keys(result, required | {"config"}, "result")
    if int(result.get("schema_version", -1)) != 1:
        raise Phase9IError("schema_version must be 1")
    if result.get("diagnostic_only") is not True or result.get("selection_mutation") is not False:
        raise Phase9IError("result is not observation-only")
    if result.get("report_only") is not True:
        raise Phase9IError("result is not report-only")
    if result.get("stage") not in {"screen", "formal", "replication"}:
        raise Phase9IError("invalid stage")
    samples = result.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase9IError("samples must be non-empty")
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise Phase9IError("sample must be an object")
        question_id = str(sample.get("question_id", ""))
        if not question_id or question_id in seen:
            raise Phase9IError("question IDs must be non-empty and unique")
        seen.add(question_id)
        _validate_sample(sample, question_id)
    return samples


def _arm_summary(rows: Sequence[Mapping[str, Any]], arm_name: str) -> dict[str, Any]:
    values = [
        row["arms"][arm_name]
        for row in rows
        if row["arms"][arm_name]["attempted"]
        and row["arms"][arm_name]["em"] is not None
    ]
    if not values:
        return {
            "n_attempted": 0,
            "em_mean": None,
            "f1_mean": None,
            "generation_time_ms_mean": None,
        }
    return {
        "n_attempted": len(values),
        "em_mean": _mean([float(item["em"]) for item in values]),
        "f1_mean": _mean([float(item["f1"]) for item in values]),
        "generation_time_ms_mean": _mean(
            [float(item["generation_time_ms"]) for item in values]
        ),
    }


def _candidate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [row["candidate_set"] for row in rows if row["candidate_set"]["attempted"]]
    if not values:
        return {
            "n_attempted": 0,
            "candidate_count_mean": None,
            "unique_candidate_count_mean": None,
            "parse_failures_total": 0,
            "oracle_em_mean": None,
            "oracle_f1_mean": None,
        }
    return {
        "n_attempted": len(values),
        "candidate_count_mean": _mean([v["candidate_count"] for v in values]),
        "unique_candidate_count_mean": _mean(
            [v["unique_candidate_count"] for v in values]
        ),
        "parse_failures_total": sum(v["parse_failures"] for v in values),
        "oracle_em_mean": _mean([float(v["oracle_em"]) for v in values]),
        "oracle_f1_mean": _mean([float(v["oracle_f1"]) for v in values]),
    }


def _verifier_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        row["verifier"]
        for row in rows
        if row["verifier"]["attempted"]
        and row["verifier"]["parse_status"] == "ok"
    ]
    if not values:
        return {
            "n_attempted": 0,
            "n_valid": 0,
            "order_agreement_rate": None,
            "em_mean": None,
            "f1_mean": None,
            "generation_time_ms_mean": None,
        }
    return {
        "n_attempted": sum(
            1 for row in rows if row["verifier"]["attempted"]
        ),
        "n_valid": len(values),
        "order_agreement_rate": _mean(
            [1.0 if v["order_agreement"] else 0.0 for v in values]
        ),
        "em_mean": _mean([float(v["em"]) for v in values]),
        "f1_mean": _mean([float(v["f1"]) for v in values]),
        "generation_time_ms_mean": _mean(
            [float(v["generation_time_ms"]) for v in values]
        ),
    }


def _delta_stats(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any] | None:
    values: list[float] = []
    for row in rows:
        baseline = row["arms"]["baseline"]
        if arm == "candidate_oracle":
            candidate_value = row["candidate_set"]["oracle_" + metric]
            attempted = row["candidate_set"]["attempted"]
        elif arm == "verifier":
            candidate_value = row["verifier"][metric]
            attempted = (
                row["verifier"]["attempted"]
                and row["verifier"]["parse_status"] == "ok"
            )
        else:
            candidate_value = row["arms"][arm][metric]
            attempted = row["arms"][arm]["attempted"]
        if attempted and candidate_value is not None:
            values.append(float(candidate_value) - float(baseline[metric]))
    return paired_bootstrap(values, repetitions=repetitions, seed=seed)


def summarize_phase9i(
    result: Mapping[str, Any], *, gate: Mapping[str, Any]
) -> dict[str, Any]:
    rows = validate_result(result)
    stage = str(result["stage"])
    eligible = [row for row in rows if row["eligible"]]
    selected = [row for row in rows if row["selected_hit"]]
    primary = [row for row in rows if row["primary_error"]]
    copy_success = [row for row in rows if row["copy_control_success"]]
    formal_gate = gate.get("formal", gate)
    repetitions = int(formal_gate.get("bootstrap_repetitions", 2000))
    bootstrap_seed = int(formal_gate.get("bootstrap_seed", 9901))
    oracle_em = _delta_stats(
        eligible, "candidate_oracle", "em", repetitions=repetitions, seed=bootstrap_seed
    )
    oracle_f1 = _delta_stats(
        eligible, "candidate_oracle", "f1", repetitions=repetitions, seed=bootstrap_seed + 1
    )
    verifier_em = _delta_stats(
        eligible, "verifier", "em", repetitions=repetitions, seed=bootstrap_seed + 2
    )
    verifier_f1 = _delta_stats(
        eligible, "verifier", "f1", repetitions=repetitions, seed=bootstrap_seed + 3
    )
    if stage == "screen":
        decision = "screen_contract_only"
        oracle_gate_pass = None
        verifier_gate_pass = None
    else:
        oracle_min = float(formal_gate.get("candidate_oracle_em_uplift_min", 0.05))
        oracle_low = float(formal_gate.get("candidate_oracle_ci95_low_min", 0.0))
        verifier_min = float(formal_gate.get("verifier_em_uplift_min", 0.03))
        verifier_low = float(formal_gate.get("verifier_ci95_low_min", 0.0))
        recovery_min = float(formal_gate.get("verifier_oracle_gain_recovery_min", 0.60))
        f1_loss_max = float(formal_gate.get("verifier_f1_loss_max", 0.01))
        oracle_gate_pass = bool(
            oracle_em
            and oracle_em["mean"] >= oracle_min
            and oracle_em["ci95_low"] > oracle_low
        )
        recovery = (
            verifier_em["mean"] / oracle_em["mean"]
            if oracle_em and oracle_em["mean"] > 0 and verifier_em
            else None
        )
        verifier_gate_pass = bool(
            verifier_em
            and verifier_em["mean"] >= verifier_min
            and verifier_em["ci95_low"] > verifier_low
            and recovery is not None
            and recovery >= recovery_min
            and verifier_f1 is not None
            and verifier_f1["mean"] >= -f1_loss_max
        )
        if not eligible:
            decision = "insufficient_eligible_cohort"
        elif not oracle_gate_pass:
            decision = "candidate_coverage_gate_failed"
        elif not verifier_gate_pass:
            decision = "coverage_positive_verifier_negative"
        else:
            decision = "verifier_gate_passed_replication_only"
    return {
        "schema_version": 1,
        "phase": result["phase"],
        "stage": stage,
        "n_questions": len(rows),
        "cohorts": {
            "selected_hit": len(selected),
            "selected_hit_baseline_errors": len(primary),
            "gold_copy_success": len(copy_success),
            "eligible": len(eligible),
        },
        "arms": {
            "baseline": _arm_summary(rows, "baseline"),
            "gold_answer_copy": _arm_summary(rows, "gold_answer_copy"),
            "extractive_span": _arm_summary(eligible, "extractive_span"),
            "evidence_constrained": _arm_summary(eligible, "evidence_constrained"),
            "candidate_set": _candidate_summary(eligible),
            "verifier": _verifier_summary(eligible),
        },
        "paired_deltas": {
            "candidate_oracle_em": oracle_em,
            "candidate_oracle_f1": oracle_f1,
            "verifier_em": verifier_em,
            "verifier_f1": verifier_f1,
        },
        "decision": {
            "primary_failure_class": decision,
            "report_only": True,
            "diagnostic_outputs_used_for_selection": False,
            "oracle_gate_pass": oracle_gate_pass,
            "verifier_gate_pass": verifier_gate_pass,
            "verifier_oracle_gain_recovery": (
                verifier_em["mean"] / oracle_em["mean"]
                if oracle_em and oracle_em["mean"] > 0 and verifier_em
                else None
            ),
            "thresholds": {
                "candidate_oracle_em_uplift_min": float(
                    formal_gate.get("candidate_oracle_em_uplift_min", 0.05)
                ),
                "candidate_oracle_ci95_low_min": float(
                    formal_gate.get("candidate_oracle_ci95_low_min", 0.0)
                ),
                "verifier_em_uplift_min": float(
                    formal_gate.get("verifier_em_uplift_min", 0.03)
                ),
                "verifier_ci95_low_min": float(
                    formal_gate.get("verifier_ci95_low_min", 0.0)
                ),
                "verifier_oracle_gain_recovery_min": float(
                    formal_gate.get("verifier_oracle_gain_recovery_min", 0.60)
                ),
                "verifier_f1_loss_max": float(
                    formal_gate.get("verifier_f1_loss_max", 0.01)
                ),
            },
        },
    }


__all__ = [
    "FORBIDDEN_FIELDS",
    "Phase9IError",
    "paired_bootstrap",
    "summarize_phase9i",
    "validate_result",
]
