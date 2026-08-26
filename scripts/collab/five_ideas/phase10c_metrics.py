"""Compact metrics and predeclared gates for Phase 10C."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


FORBIDDEN_FIELDS = {
    "question", "passages", "gold_answers", "prediction", "candidate_text",
    "span_text", "raw_prompt", "evaluator_trace", "text", "token_ids",
}
EXPECTED_ARMS = ("baseline_greedy", "reader_span_energy", "matched_span_control")


class Phase10CError(ValueError):
    pass


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase10CError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap(values: Sequence[float], *, repetitions: int, seed: int) -> dict[str, float]:
    if not values or repetitions < 100:
        raise Phase10CError("bootstrap requires values and at least 100 repetitions")
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


def _sample_map(payload: Mapping[str, Any], arm: str) -> dict[str, Mapping[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase10CError(f"{arm}: samples must be non-empty")
    expected = {
        "question_id", "em", "f1", "generation_time_ms", "pipeline_time_ms",
        "selected_hit", "final_context_K", "selected_context_parity",
        "lattice_available", "lattice_span_count", "lattice_token_count",
        "energy_active_steps", "generated_token_count", "decoder_mode",
    }
    rows: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != expected:
            raise Phase10CError(f"{arm}: malformed compact sample")
        question_id = str(sample.get("question_id", ""))
        if not question_id or question_id in rows:
            raise Phase10CError(f"{arm}: question IDs must be unique")
        for key in ("em", "f1", "generation_time_ms", "pipeline_time_ms"):
            value = sample.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise Phase10CError(f"{arm}/{question_id}: invalid {key}")
        if int(sample["final_context_K"]) != 5 or sample["selected_context_parity"] is not True:
            raise Phase10CError(f"{arm}/{question_id}: frozen context contract failed")
        for key in ("lattice_span_count", "lattice_token_count", "energy_active_steps", "generated_token_count"):
            if not isinstance(sample[key], int) or sample[key] < 0:
                raise Phase10CError(f"{arm}/{question_id}: invalid {key}")
        if not isinstance(sample["selected_hit"], bool) or not isinstance(sample["lattice_available"], bool):
            raise Phase10CError(f"{arm}/{question_id}: invalid boolean")
        rows[question_id] = sample
    return rows


def _arm_summary(
    baseline: Mapping[str, Mapping[str, Any]],
    variant: Mapping[str, Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if set(baseline) != set(variant):
        raise Phase10CError("arms must have identical question IDs")
    ids = sorted(baseline)
    def metric(name: str, offset: int) -> dict[str, Any]:
        delta = [float(variant[key][name]) - float(baseline[key][name]) for key in ids]
        return {
            "baseline_mean": _mean([float(baseline[key][name]) for key in ids]),
            "variant_mean": _mean([float(variant[key][name]) for key in ids]),
            "delta": paired_bootstrap(delta, repetitions=repetitions, seed=seed + offset),
        }
    base_generation = sum(float(baseline[key]["generation_time_ms"]) for key in ids)
    variant_generation = sum(float(variant[key]["generation_time_ms"]) for key in ids)
    base_pipeline = sum(float(baseline[key]["pipeline_time_ms"]) for key in ids)
    variant_pipeline = sum(float(variant[key]["pipeline_time_ms"]) for key in ids)
    return {
        "n_questions": len(ids),
        "f1": metric("f1", 0),
        "em": metric("em", 1),
        "generation_time_ms": {"baseline_total": base_generation, "variant_total": variant_generation, "ratio": variant_generation / base_generation if base_generation else float("inf")},
        "pipeline_time_ms": {"baseline_total": base_pipeline, "variant_total": variant_pipeline, "ratio": variant_pipeline / base_pipeline if base_pipeline else float("inf")},
        "lattice_available_rate": _mean([1.0 if variant[key]["lattice_available"] else 0.0 for key in ids]),
        "energy_active_steps_mean": _mean([float(variant[key]["energy_active_steps"]) for key in ids]),
    }


def _direct_delta(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    if set(left) != set(right):
        raise Phase10CError("direct comparison arms must align")
    return paired_bootstrap(
        [float(left[key][metric]) - float(right[key][metric]) for key in sorted(left)],
        repetitions=repetitions,
        seed=seed,
    )


def summarize_phase10c(result: Mapping[str, Any], *, gate: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase10CError(f"compact result contains forbidden fields: {forbidden[:5]}")
    arms = result.get("arms")
    if not isinstance(arms, Mapping) or tuple(arms) != EXPECTED_ARMS:
        raise Phase10CError("result arms are malformed or out of order")
    baseline = _sample_map(arms["baseline_greedy"], "baseline_greedy")
    reader = _sample_map(arms["reader_span_energy"], "reader_span_energy")
    control = _sample_map(arms["matched_span_control"], "matched_span_control")
    if any(sample["decoder_mode"] != "frozen_generator" for sample in baseline.values()):
        raise Phase10CError("baseline must use the frozen Generator")
    if any(sample["decoder_mode"] != "reader_span_energy" for sample in reader.values()):
        raise Phase10CError("reader arm decoder mode is malformed")
    if any(sample["decoder_mode"] != "matched_span_control" for sample in control.values()):
        raise Phase10CError("control arm decoder mode is malformed")
    reps = int(gate.get("bootstrap_repetitions", 2000))
    seed = int(gate.get("bootstrap_seed", 10501))
    reader_summary = _arm_summary(baseline, reader, repetitions=reps, seed=seed)
    control_summary = _arm_summary(baseline, control, repetitions=reps, seed=seed + 20)
    reader_vs_control_f1 = _direct_delta(reader, control, "f1", repetitions=reps, seed=seed + 40)
    reader_vs_control_em = _direct_delta(reader, control, "em", repetitions=reps, seed=seed + 41)
    parity_gate = all(sample["selected_context_parity"] for sample in baseline.values())
    cost_gate = reader_summary["pipeline_time_ms"]["ratio"] <= float(gate["total_pipeline_cost_ratio_max"])
    harm_gate = reader_summary["f1"]["delta"]["ci95_low"] >= float(gate["screen_f1_ci95_low_min"])
    control_gate = reader_vs_control_f1["mean"] > float(gate["screen_reader_minus_control_mean_min"])
    if result.get("stage") == "screen":
        decision = "clean_screen_inconclusive" if all((parity_gate, cost_gate, harm_gate, control_gate)) else "kill_phase10c_screen"
    else:
        f1_gate = reader_summary["f1"]["delta"]["mean"] >= float(gate["formal_f1_delta_min"]) and reader_summary["f1"]["delta"]["ci95_low"] > 0.0
        em_gate = reader_summary["em"]["delta"]["mean"] >= 0.0
        control_gate = reader_vs_control_f1["ci95_low"] > 0.0
        decision = "pass_phase10c_formal" if all((parity_gate, cost_gate, f1_gate, em_gate, control_gate)) else "kill_phase10c_formal"
    return {
        "schema_version": 1,
        "decision": decision,
        "arms": {"reader_span_energy": reader_summary, "matched_span_control": control_summary},
        "reader_minus_control": {"f1": reader_vs_control_f1, "em": reader_vs_control_em},
        "gate": {"context_parity": parity_gate, "cost": cost_gate, "f1_harm": harm_gate, "reader_minus_control": control_gate},
    }


__all__ = ["EXPECTED_ARMS", "FORBIDDEN_FIELDS", "Phase10CError", "paired_bootstrap", "summarize_phase10c"]
