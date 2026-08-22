"""Compact metrics and gates for the Phase 9J candidate ranker screen."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence


RANKER_PROFILES = (
    "context_lift_v1",
    "reader_span_support_v1",
    "context_lift_reader_support_v1",
)
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


class Phase9JError(ValueError):
    """Raised when compact Phase 9J data violates its contract."""


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase9JError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase9JError("cannot compute percentile of an empty sequence")
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
        raise Phase9JError("bootstrap requires at least 100 repetitions")
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


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise Phase9JError(
            f"{label}: expected keys {sorted(expected)}, got {sorted(value)}"
        )


def _standard_arm(arm: Mapping[str, Any], *, attempted: bool, label: str) -> None:
    _require_keys(arm, {"attempted", "em", "f1", "generation_time_ms"}, label)
    if arm["attempted"] is not attempted:
        raise Phase9JError(f"{label}: attempted mismatch")
    for key in ("em", "f1", "generation_time_ms"):
        if attempted:
            if not _finite(arm[key]) or float(arm[key]) < 0.0:
                raise Phase9JError(f"{label}: invalid {key}")
        elif arm[key] is not None:
            raise Phase9JError(f"{label}: non-attempted {key} must be null")
    if attempted and not 0.0 <= float(arm["em"]) <= 1.0:
        raise Phase9JError(f"{label}: EM outside [0,1]")


def _candidate_arm(arm: Mapping[str, Any], *, attempted: bool, label: str) -> None:
    _require_keys(
        arm,
        {"attempted", "parse_status", "em", "f1", "generation_time_ms"},
        label,
    )
    if arm["attempted"] is not attempted:
        raise Phase9JError(f"{label}: attempted mismatch")
    if arm["parse_status"] not in {"ok", "abstain", "empty", "invalid"}:
        raise Phase9JError(f"{label}: invalid parse status")
    valid = attempted and arm["parse_status"] in {"ok", "abstain"}
    for key in ("em", "f1", "generation_time_ms"):
        if valid:
            if not _finite(arm[key]) or float(arm[key]) < 0.0:
                raise Phase9JError(f"{label}: invalid {key}")
        elif arm[key] is not None:
            raise Phase9JError(f"{label}: invalid unscored {key}")
    if valid and not 0.0 <= float(arm["em"]) <= 1.0:
        raise Phase9JError(f"{label}: EM outside [0,1]")


def _ranker_arm(arm: Mapping[str, Any], *, attempted: bool, label: str) -> None:
    expected = {
        "attempted",
        "original_choice_mode",
        "permuted_choice_mode",
        "order_agreement",
        "parse_status",
        "em",
        "f1",
        "score_time_ms",
        "score_to_baseline_ratio",
    }
    _require_keys(arm, expected, label)
    if arm["attempted"] is not attempted:
        raise Phase9JError(f"{label}: attempted mismatch")
    if arm["parse_status"] not in {"ok", "invalid_choice", "not_attempted"}:
        raise Phase9JError(f"{label}: invalid parse status")
    if attempted and arm["parse_status"] == "ok":
        if not isinstance(arm["original_choice_mode"], str):
            raise Phase9JError(f"{label}: missing original choice")
        if not isinstance(arm["permuted_choice_mode"], str):
            raise Phase9JError(f"{label}: missing permuted choice")
        if not isinstance(arm["order_agreement"], bool):
            raise Phase9JError(f"{label}: order agreement must be boolean")
        for key in ("em", "f1", "score_time_ms", "score_to_baseline_ratio"):
            if not _finite(arm[key]) or float(arm[key]) < 0.0:
                raise Phase9JError(f"{label}: invalid {key}")
        if not 0.0 <= float(arm["em"]) <= 1.0:
            raise Phase9JError(f"{label}: EM outside [0,1]")
    else:
        for key in (
            "original_choice_mode",
            "permuted_choice_mode",
            "order_agreement",
            "em",
            "f1",
            "score_time_ms",
            "score_to_baseline_ratio",
        ):
            if arm[key] is not None:
                raise Phase9JError(f"{label}: invalid non-scored field {key}")


def validate_result(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        raise Phase9JError("result must be an object")
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase9JError(f"forbidden compact fields: {forbidden[:5]}")
    required = {
        "schema_version",
        "phase",
        "stage",
        "diagnostic_only",
        "selection_mutation",
        "report_only",
        "config",
        "samples",
    }
    _require_keys(result, required, "result")
    if result["schema_version"] != 1 or result["phase"] != "phase9j_context_lift_candidate_ranking":
        raise Phase9JError("invalid result identity")
    if result["stage"] not in {"screen", "formal", "replication"}:
        raise Phase9JError("invalid result stage")
    if result["diagnostic_only"] is not True or result["selection_mutation"] is not False:
        raise Phase9JError("result is not observation-only")
    if result["report_only"] is not True:
        raise Phase9JError("result is not report-only")
    samples = result["samples"]
    if not isinstance(samples, list) or not samples:
        raise Phase9JError("samples must be non-empty")
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise Phase9JError("sample must be an object")
        question_id = str(sample.get("question_id", ""))
        if not question_id or question_id in seen:
            raise Phase9JError("question IDs must be non-empty and unique")
        seen.add(question_id)
        _require_keys(
            sample,
            {
                "question_id",
                "retrieval_hit",
                "selected_hit",
                "primary_error",
                "copy_control_success",
                "eligible",
                "selection_time_ms",
                "arms",
                "candidate_set",
                "rankers",
            },
            question_id,
        )
        for key in (
            "retrieval_hit",
            "selected_hit",
            "primary_error",
            "copy_control_success",
            "eligible",
        ):
            if not isinstance(sample[key], bool):
                raise Phase9JError(f"{question_id}: {key} must be boolean")
        if not _finite(sample["selection_time_ms"]) or float(sample["selection_time_ms"]) < 0:
            raise Phase9JError(f"{question_id}: invalid selection time")
        arms = sample["arms"]
        _require_keys(
            arms,
            {"baseline", "gold_answer_copy", "extractive_span", "evidence_constrained"},
            f"{question_id}/arms",
        )
        _standard_arm(arms["baseline"], attempted=True, label=f"{question_id}/baseline")
        copy_attempted = bool(sample["primary_error"])
        _standard_arm(
            arms["gold_answer_copy"], attempted=copy_attempted, label=f"{question_id}/gold_answer_copy"
        )
        for name in ("extractive_span", "evidence_constrained"):
            _candidate_arm(
                arms[name], attempted=bool(sample["eligible"]), label=f"{question_id}/{name}"
            )
        if sample["primary_error"] != (
            bool(sample["selected_hit"]) and float(arms["baseline"]["em"]) == 0.0
        ):
            raise Phase9JError(f"{question_id}: primary error mismatch")
        if sample["copy_control_success"] != (
            copy_attempted and arms["gold_answer_copy"]["em"] == 1.0
        ):
            raise Phase9JError(f"{question_id}: copy control mismatch")
        if sample["eligible"] != (
            bool(sample["primary_error"]) and bool(sample["copy_control_success"])
        ):
            raise Phase9JError(f"{question_id}: eligibility mismatch")
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
            raise Phase9JError(f"{question_id}: candidate set condition mismatch")
        for key in ("candidate_count", "unique_candidate_count", "parse_failures"):
            if not isinstance(candidate_set[key], int) or candidate_set[key] < 0:
                raise Phase9JError(f"{question_id}: invalid candidate count")
        if candidate_set["candidate_count"] > 3:
            raise Phase9JError(f"{question_id}: candidate count exceeds 3")
        if candidate_set["attempted"]:
            if not 1 <= candidate_set["candidate_count"] <= 3:
                raise Phase9JError(f"{question_id}: invalid nonempty candidate count")
            for key in ("oracle_em", "oracle_f1"):
                if not _finite(candidate_set[key]) or not 0.0 <= float(candidate_set[key]) <= 1.0:
                    raise Phase9JError(f"{question_id}: invalid oracle metric")
        else:
            if any(candidate_set[key] != 0 for key in ("candidate_count", "unique_candidate_count", "parse_failures")):
                raise Phase9JError(f"{question_id}: invalid empty candidate set")
            if candidate_set["oracle_em"] is not None or candidate_set["oracle_f1"] is not None:
                raise Phase9JError(f"{question_id}: empty candidate set has metrics")
        rankers = sample["rankers"]
        _require_keys(rankers, set(RANKER_PROFILES), f"{question_id}/rankers")
        for profile in RANKER_PROFILES:
            _ranker_arm(
                rankers[profile], attempted=bool(sample["eligible"]), label=f"{question_id}/{profile}"
            )
    return samples


def _delta_stats(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float] | None:
    values: list[float] = []
    for row in rows:
        baseline = row["arms"]["baseline"]
        if arm == "candidate_oracle":
            candidate = row["candidate_set"]["oracle_" + metric]
            attempted = row["candidate_set"]["attempted"]
        else:
            candidate = row["rankers"][arm][metric]
            attempted = row["rankers"][arm]["attempted"] and row["rankers"][arm]["parse_status"] == "ok"
        if attempted and candidate is not None:
            values.append(float(candidate) - float(baseline[metric]))
    return paired_bootstrap(values, repetitions=repetitions, seed=seed)


def _arm_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    values = [
        row["arms"][arm]
        for row in rows
        if row["arms"][arm]["attempted"] and row["arms"][arm]["em"] is not None
    ]
    if not values:
        return {"n_attempted": 0, "em_mean": None, "f1_mean": None, "generation_time_ms_mean": None}
    return {
        "n_attempted": len(values),
        "em_mean": _mean([float(item["em"]) for item in values]),
        "f1_mean": _mean([float(item["f1"]) for item in values]),
        "generation_time_ms_mean": _mean([float(item["generation_time_ms"]) for item in values]),
    }


def _ranker_summary(rows: Sequence[Mapping[str, Any]], profile: str) -> dict[str, Any]:
    values = [
        row["rankers"][profile]
        for row in rows
        if row["rankers"][profile]["attempted"] and row["rankers"][profile]["parse_status"] == "ok"
    ]
    if not values:
        return {
            "n_attempted": 0,
            "n_valid": 0,
            "order_agreement_rate": None,
            "em_mean": None,
            "f1_mean": None,
            "score_time_ms_mean": None,
            "score_to_baseline_ratio_mean": None,
        }
    return {
        "n_attempted": sum(1 for row in rows if row["rankers"][profile]["attempted"]),
        "n_valid": len(values),
        "order_agreement_rate": _mean([1.0 if item["order_agreement"] else 0.0 for item in values]),
        "em_mean": _mean([float(item["em"]) for item in values]),
        "f1_mean": _mean([float(item["f1"]) for item in values]),
        "score_time_ms_mean": _mean([float(item["score_time_ms"]) for item in values]),
        "score_to_baseline_ratio_mean": _mean([float(item["score_to_baseline_ratio"]) for item in values]),
    }


def summarize_phase9j(result: Mapping[str, Any], *, gate: Mapping[str, Any]) -> dict[str, Any]:
    rows = validate_result(result)
    eligible = [row for row in rows if row["eligible"]]
    configured_profiles = result.get("config", {}).get("active_ranker_profiles", list(RANKER_PROFILES))
    active_profiles = tuple(configured_profiles)
    if not active_profiles or any(profile not in RANKER_PROFILES for profile in active_profiles):
        raise Phase9JError("active ranker profiles are invalid")
    formal_gate = gate.get("formal", gate)
    repetitions = int(formal_gate.get("bootstrap_repetitions", 2000))
    seed = int(formal_gate.get("bootstrap_seed", 9911))
    paired: dict[str, Any] = {
        "candidate_oracle_em": _delta_stats(eligible, "candidate_oracle", "em", repetitions=repetitions, seed=seed),
        "candidate_oracle_f1": _delta_stats(eligible, "candidate_oracle", "f1", repetitions=repetitions, seed=seed + 1),
    }
    for index, profile in enumerate(RANKER_PROFILES, start=2):
        paired[profile + "_em"] = _delta_stats(eligible, profile, "em", repetitions=repetitions, seed=seed + index * 2)
        paired[profile + "_f1"] = _delta_stats(eligible, profile, "f1", repetitions=repetitions, seed=seed + index * 2 + 1)
    ranker_summaries = {profile: _ranker_summary(eligible, profile) for profile in RANKER_PROFILES}
    if result["stage"] == "screen":
        decision = "screen_contract_only"
        gates = {profile: None for profile in RANKER_PROFILES}
        oracle_pass = None
    else:
        oracle_min = float(formal_gate.get("candidate_oracle_em_uplift_min", 0.05))
        oracle_low = float(formal_gate.get("candidate_oracle_ci95_low_min", 0.0))
        ranker_min = float(formal_gate.get("ranker_em_uplift_min", 0.03))
        ranker_low = float(formal_gate.get("ranker_ci95_low_min", 0.0))
        f1_loss = float(formal_gate.get("ranker_f1_loss_max", 0.01))
        order_min = float(formal_gate.get("order_agreement_min", 0.90))
        score_ratio_max = float(formal_gate.get("score_to_baseline_ratio_max", 1.30))
        oracle = paired["candidate_oracle_em"]
        oracle_pass = bool(oracle and oracle["mean"] >= oracle_min and oracle["ci95_low"] > oracle_low)
        gates = {}
        for profile in RANKER_PROFILES:
            if profile not in active_profiles:
                gates[profile] = None
                continue
            em = paired[profile + "_em"]
            f1 = paired[profile + "_f1"]
            summary = ranker_summaries[profile]
            gates[profile] = bool(
                oracle_pass
                and em
                and em["mean"] >= ranker_min
                and em["ci95_low"] > ranker_low
                and f1
                and f1["mean"] >= -f1_loss
                and summary["order_agreement_rate"] is not None
                and summary["order_agreement_rate"] >= order_min
                and summary["score_to_baseline_ratio_mean"] is not None
                and summary["score_to_baseline_ratio_mean"] <= score_ratio_max
            )
        if not eligible:
            decision = "insufficient_eligible_cohort"
        elif not oracle_pass:
            decision = "candidate_coverage_gate_failed"
        else:
            primary_profile = str(formal_gate.get("primary_ranker_profile", "context_lift_reader_support_v1"))
            if primary_profile not in active_profiles:
                decision = "primary_ranker_not_run"
            elif not gates[primary_profile]:
                decision = "combined_ranker_gate_failed" if primary_profile == "context_lift_reader_support_v1" else f"{primary_profile}_gate_failed"
            else:
                decision = "combined_ranker_gate_passed_replication_only" if primary_profile == "context_lift_reader_support_v1" else f"{primary_profile}_gate_passed_replication_only"
    return {
        "schema_version": 1,
        "phase": result["phase"],
        "stage": result["stage"],
        "n_questions": len(rows),
        "cohorts": {
            "eligible": len(eligible),
            "selected_hit": sum(1 for row in rows if row["selected_hit"]),
            "selected_hit_baseline_errors": sum(1 for row in rows if row["primary_error"]),
            "gold_copy_success": sum(1 for row in rows if row["copy_control_success"]),
        },
        "arms": {
            "baseline": _arm_summary(rows, "baseline"),
            "gold_answer_copy": _arm_summary(rows, "gold_answer_copy"),
            "extractive_span": _arm_summary(eligible, "extractive_span"),
            "evidence_constrained": _arm_summary(eligible, "evidence_constrained"),
            "rankers": ranker_summaries,
        },
        "paired_deltas": paired,
        "decision": {
            "primary_failure_class": decision,
            "report_only": True,
            "diagnostic_outputs_used_for_selection": False,
            "candidate_oracle_gate_pass": oracle_pass,
            "ranker_gates": gates,
            "thresholds": {
                "candidate_oracle_em_uplift_min": float(formal_gate.get("candidate_oracle_em_uplift_min", 0.05)),
                "candidate_oracle_ci95_low_min": float(formal_gate.get("candidate_oracle_ci95_low_min", 0.0)),
                "ranker_em_uplift_min": float(formal_gate.get("ranker_em_uplift_min", 0.03)),
                "ranker_ci95_low_min": float(formal_gate.get("ranker_ci95_low_min", 0.0)),
                "ranker_f1_loss_max": float(formal_gate.get("ranker_f1_loss_max", 0.01)),
                "order_agreement_min": float(formal_gate.get("order_agreement_min", 0.90)),
                "score_to_baseline_ratio_max": float(formal_gate.get("score_to_baseline_ratio_max", 1.30)),
            },
        },
    }


__all__ = [
    "FORBIDDEN_FIELDS",
    "Phase9JError",
    "RANKER_PROFILES",
    "paired_bootstrap",
    "summarize_phase9j",
    "validate_result",
]
