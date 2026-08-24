"""Compact validation and paired metrics for Phase 9K."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping, Sequence

from phase9k_gold_free_applicability import REASON_CODES


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
}


class Phase9KError(ValueError):
    """Raised when compact Phase 9K data violates its contract."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


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
        raise Phase9KError(f"{label}: expected keys {sorted(expected)}, got {sorted(value)}")


def paired_bootstrap(values: Sequence[float], *, repetitions: int, seed: int) -> dict[str, float] | None:
    if not values:
        return None
    if repetitions < 100:
        raise Phase9KError("bootstrap requires at least 100 repetitions")
    rng = random.Random(seed)
    draws = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(repetitions)
    ]
    ordered = sorted(float(value) for value in draws)
    low_index = int(0.025 * (len(ordered) - 1))
    high_index = int(0.975 * (len(ordered) - 1))
    return {
        "mean": float(statistics.fmean(values)),
        "ci95_low": float(ordered[low_index]),
        "ci95_high": float(ordered[high_index]),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def _arm(value: Mapping[str, Any], *, attempted: bool, label: str) -> None:
    _require_keys(value, {"attempted", "em", "f1", "generation_time_ms"}, label)
    if value["attempted"] is not attempted:
        raise Phase9KError(f"{label}: attempted mismatch")
    for key in ("em", "f1", "generation_time_ms"):
        if attempted:
            if not _finite(value[key]) or float(value[key]) < 0:
                raise Phase9KError(f"{label}: invalid {key}")
        elif value[key] is not None:
            raise Phase9KError(f"{label}: non-attempted {key} must be null")
    if attempted and not 0 <= float(value["em"]) <= 1:
        raise Phase9KError(f"{label}: EM outside [0,1]")


def _ranker(value: Mapping[str, Any], *, attempted: bool, label: str) -> None:
    expected = {
        "attempted",
        "choice_mode",
        "permuted_choice_mode",
        "order_agreement",
        "parse_status",
        "em",
        "f1",
        "generation_time_ms",
        "score_time_ms",
        "score_to_baseline_ratio",
    }
    _require_keys(value, expected, label)
    if value["attempted"] is not attempted:
        raise Phase9KError(f"{label}: attempted mismatch")
    if value["parse_status"] not in {"ok", "not_attempted"}:
        raise Phase9KError(f"{label}: invalid parse status")
    if attempted:
        if value["parse_status"] != "ok":
            raise Phase9KError(f"{label}: attempted ranker must be ok")
        if not isinstance(value["choice_mode"], str) or not isinstance(value["permuted_choice_mode"], str):
            raise Phase9KError(f"{label}: missing choice mode")
        if not isinstance(value["order_agreement"], bool):
            raise Phase9KError(f"{label}: order agreement must be boolean")
        for key in ("em", "f1", "generation_time_ms", "score_time_ms", "score_to_baseline_ratio"):
            if not _finite(value[key]) or float(value[key]) < 0:
                raise Phase9KError(f"{label}: invalid {key}")
    else:
        for key in ("choice_mode", "permuted_choice_mode", "order_agreement", "em", "f1", "generation_time_ms", "score_time_ms", "score_to_baseline_ratio"):
            if value[key] is not None:
                raise Phase9KError(f"{label}: non-attempted field {key} must be null")


def validate_result(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        raise Phase9KError("result must be an object")
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase9KError(f"forbidden compact fields: {forbidden[:5]}")
    _require_keys(
        result,
        {"schema_version", "phase", "stage", "diagnostic_only", "selection_mutation", "report_only", "config", "samples"},
        "result",
    )
    if result["schema_version"] != 1 or result["phase"] != "phase9k_gold_free_applicability":
        raise Phase9KError("invalid result identity")
    if result["stage"] not in {"screen", "formal", "replication"}:
        raise Phase9KError("invalid result stage")
    if result["diagnostic_only"] is not True or result["selection_mutation"] is not False or result["report_only"] is not True:
        raise Phase9KError("result is not observation-only")
    samples = result["samples"]
    if not isinstance(samples, list) or not samples:
        raise Phase9KError("samples must be non-empty")
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise Phase9KError("sample must be an object")
        question_id = str(sample.get("question_id", ""))
        if not question_id or question_id in seen:
            raise Phase9KError("question IDs must be non-empty and unique")
        seen.add(question_id)
        _require_keys(
            sample,
            {"question_id", "retrieval_hit", "selected_hit", "selection_time_ms", "candidate_count", "unique_candidate_count", "parse_failures", "applicability", "baseline", "ranker"},
            question_id,
        )
        for key in ("retrieval_hit", "selected_hit"):
            if not isinstance(sample[key], bool):
                raise Phase9KError(f"{question_id}: {key} must be boolean")
        if not _finite(sample["selection_time_ms"]) or float(sample["selection_time_ms"]) < 0:
            raise Phase9KError(f"{question_id}: invalid selection time")
        for key in ("candidate_count", "unique_candidate_count", "parse_failures"):
            if not isinstance(sample[key], int) or sample[key] < 0:
                raise Phase9KError(f"{question_id}: invalid {key}")
        if sample["unique_candidate_count"] > sample["candidate_count"]:
            raise Phase9KError(f"{question_id}: unique candidates exceed candidates")
        applicability = sample["applicability"]
        _require_keys(
            applicability,
            {"apply", "reason_code", "chosen_mode", "baseline_score", "chosen_score", "reader_margin", "chosen_exact_span"},
            f"{question_id}/applicability",
        )
        if not isinstance(applicability["apply"], bool) or applicability["reason_code"] not in REASON_CODES:
            raise Phase9KError(f"{question_id}: invalid applicability decision")
        if applicability["apply"]:
            if not isinstance(applicability["chosen_mode"], str) or not isinstance(applicability["chosen_exact_span"], bool):
                raise Phase9KError(f"{question_id}: applied decision missing fields")
            for key in ("baseline_score", "chosen_score", "reader_margin"):
                if not _finite(applicability[key]):
                    raise Phase9KError(f"{question_id}: invalid applicability score")
        else:
            if any(applicability[key] is not None for key in ("chosen_mode", "baseline_score", "chosen_score", "reader_margin", "chosen_exact_span")):
                raise Phase9KError(f"{question_id}: non-applied decision must have null score fields")
        _arm(sample["baseline"], attempted=True, label=f"{question_id}/baseline")
        _ranker(sample["ranker"], attempted=bool(applicability["apply"]), label=f"{question_id}/ranker")
    return samples


def _bootstrap_deltas(rows: Sequence[Mapping[str, Any]], metric: str, *, deployed: bool, repetitions: int, seed: int) -> dict[str, float] | None:
    values: list[float] = []
    for row in rows:
        baseline = float(row["baseline"][metric])
        if deployed:
            ranker = row["ranker"]
            value = float(ranker[metric]) if ranker["attempted"] else baseline
        else:
            if not row["ranker"]["attempted"]:
                continue
            value = float(row["ranker"][metric])
        values.append(value - baseline)
    return paired_bootstrap(values, repetitions=repetitions, seed=seed)


def summarize_phase9k(result: Mapping[str, Any], *, gate: Mapping[str, Any]) -> dict[str, Any]:
    rows = validate_result(result)
    formal_gate = gate.get("formal", gate)
    repetitions = int(formal_gate.get("bootstrap_repetitions", 2000))
    seed = int(formal_gate.get("bootstrap_seed", 9921))
    applied = [row for row in rows if row["applicability"]["apply"]]
    all_deltas = {
        "em": _bootstrap_deltas(rows, "em", deployed=True, repetitions=repetitions, seed=seed),
        "f1": _bootstrap_deltas(rows, "f1", deployed=True, repetitions=repetitions, seed=seed + 1),
    }
    applied_deltas = {
        "em": _bootstrap_deltas(rows, "em", deployed=False, repetitions=repetitions, seed=seed + 2),
        "f1": _bootstrap_deltas(rows, "f1", deployed=False, repetitions=repetitions, seed=seed + 3),
    }
    if result["stage"] == "screen":
        decision = "screen_contract_only"
        gate_pass = None
    else:
        em = all_deltas["em"]
        f1 = all_deltas["f1"]
        order = statistics.fmean(1.0 if row["ranker"]["order_agreement"] else 0.0 for row in applied) if applied else 0.0
        score_ratio = statistics.fmean(float(row["ranker"]["score_to_baseline_ratio"]) for row in applied) if applied else math.inf
        apply_rate = len(applied) / len(rows)
        gate_pass = bool(
            em
            and em["mean"] >= float(formal_gate.get("all_question_em_uplift_min", 0.03))
            and em["ci95_low"] > float(formal_gate.get("all_question_ci95_low_min", 0.0))
            and f1
            and f1["mean"] >= -float(formal_gate.get("all_question_f1_loss_max", 0.01))
            and float(formal_gate.get("apply_rate_min", 0.05)) <= apply_rate <= float(formal_gate.get("apply_rate_max", 0.60))
            and order >= float(formal_gate.get("order_agreement_min", 0.90))
            and score_ratio <= float(formal_gate.get("score_to_baseline_ratio_max", 1.30))
        )
        decision = "gold_free_gate_passed_replication_only" if gate_pass else "gold_free_gate_failed"
    reason_counts: dict[str, int] = {reason: 0 for reason in REASON_CODES}
    for row in rows:
        reason_counts[row["applicability"]["reason_code"]] += 1
    return {
        "schema_version": 1,
        "phase": result["phase"],
        "stage": result["stage"],
        "n_questions": len(rows),
        "cohorts": {"applied": len(applied), "apply_rate": len(applied) / len(rows), "reason_counts": reason_counts},
        "arms": {
            "baseline_em_mean": float(statistics.fmean(float(row["baseline"]["em"]) for row in rows)),
            "baseline_f1_mean": float(statistics.fmean(float(row["baseline"]["f1"]) for row in rows)),
            "ranker_attempted": len(applied),
            "ranker_em_mean": float(statistics.fmean(float(row["ranker"]["em"]) for row in applied)) if applied else None,
            "ranker_f1_mean": float(statistics.fmean(float(row["ranker"]["f1"]) for row in applied)) if applied else None,
        },
        "paired_deltas": {"all_question_deployed_em": all_deltas["em"], "all_question_deployed_f1": all_deltas["f1"], "applied_cohort_em": applied_deltas["em"], "applied_cohort_f1": applied_deltas["f1"]},
        "decision": {
            "primary_failure_class": decision,
            "report_only": True,
            "diagnostic_outputs_used_for_selection": False,
            "gold_free_gate_pass": gate_pass,
            "thresholds": {
                "all_question_em_uplift_min": float(formal_gate.get("all_question_em_uplift_min", 0.03)),
                "all_question_ci95_low_min": float(formal_gate.get("all_question_ci95_low_min", 0.0)),
                "all_question_f1_loss_max": float(formal_gate.get("all_question_f1_loss_max", 0.01)),
                "apply_rate_min": float(formal_gate.get("apply_rate_min", 0.05)),
                "apply_rate_max": float(formal_gate.get("apply_rate_max", 0.60)),
                "order_agreement_min": float(formal_gate.get("order_agreement_min", 0.90)),
                "score_to_baseline_ratio_max": float(formal_gate.get("score_to_baseline_ratio_max", 1.30)),
            },
        },
    }


__all__ = ["FORBIDDEN_FIELDS", "Phase9KError", "paired_bootstrap", "summarize_phase9k", "validate_result"]
