"""Compact paired metrics and gates for the Phase 10B bridge screen."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


FORBIDDEN_FIELDS = {"question", "passages", "gold_answers", "prediction", "candidate_text", "bridge_query", "raw_prompt", "evaluator_trace", "text"}


class Phase10BError(ValueError):
    pass


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase10BError("cannot average an empty sequence")
    return float(statistics.fmean(float(v) for v in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(v) for v in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def paired_bootstrap(values: Sequence[float], *, repetitions: int, seed: int) -> dict[str, float]:
    if not values or repetitions < 100:
        raise Phase10BError("bootstrap requires values and at least 100 repetitions")
    rng = random.Random(seed)
    draws = [_mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(repetitions)]
    return {"mean": _mean(values), "ci95_low": _percentile(draws, 0.025), "ci95_high": _percentile(draws, 0.975), "bootstrap_repetitions": repetitions, "bootstrap_seed": seed}


def _forbidden(value: Any, path: str = "$root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_FIELDS:
                found.append(child_path)
            found.extend(_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden(child, f"{path}[{index}]"))
    return found


def _sample_map(payload: Mapping[str, Any], name: str) -> dict[str, Mapping[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Phase10BError(f"{name}: samples must be non-empty")
    result: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping) or not str(sample.get("question_id", "")):
            raise Phase10BError(f"{name}: every sample needs question_id")
        question_id = str(sample["question_id"])
        if question_id in result:
            raise Phase10BError(f"{name}: duplicate question_id")
        for key in ("em", "f1", "generation_time_ms", "pipeline_time_ms"):
            value = sample.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise Phase10BError(f"{name}/{question_id}: invalid {key}")
        result[question_id] = sample
    return result


def _arm_summary(baseline: Mapping[str, Mapping[str, Any]], variant: Mapping[str, Mapping[str, Any]], *, repetitions: int, seed: int) -> dict[str, Any]:
    if set(baseline) != set(variant):
        raise Phase10BError("arms must contain identical question IDs")
    ids = sorted(baseline)
    f1 = [float(variant[q]["f1"]) - float(baseline[q]["f1"]) for q in ids]
    em = [float(variant[q]["em"]) - float(baseline[q]["em"]) for q in ids]
    base_gen = sum(float(baseline[q]["generation_time_ms"]) for q in ids)
    variant_gen = sum(float(variant[q]["generation_time_ms"]) for q in ids)
    base_pipe = sum(float(baseline[q]["pipeline_time_ms"]) for q in ids)
    variant_pipe = sum(float(variant[q]["pipeline_time_ms"]) for q in ids)
    return {
        "n_questions": len(ids),
        "f1": {"baseline_mean": _mean([float(baseline[q]["f1"]) for q in ids]), "variant_mean": _mean([float(variant[q]["f1"]) for q in ids]), "delta": paired_bootstrap(f1, repetitions=repetitions, seed=seed)},
        "em": {"baseline_mean": _mean([float(baseline[q]["em"]) for q in ids]), "variant_mean": _mean([float(variant[q]["em"]) for q in ids]), "delta": paired_bootstrap(em, repetitions=repetitions, seed=seed + 1)},
        "generation_time_ms": {"baseline_total": base_gen, "variant_total": variant_gen, "ratio": variant_gen / base_gen if base_gen else float("inf")},
        "pipeline_time_ms": {"baseline_total": base_pipe, "variant_total": variant_pipe, "ratio": variant_pipe / base_pipe if base_pipe else float("inf")},
    }


def _selected_hit_gain(baseline: Mapping[str, Mapping[str, Any]], variant: Mapping[str, Mapping[str, Any]], applied: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ids = [q for q in sorted(applied) if bool(applied[q].get("bridge_applied"))]
    if not ids:
        return {"n_applied": 0, "baseline_rate": None, "variant_rate": None, "delta": None}
    base = [_as_bool(baseline[q].get("selected_hit")) for q in ids]
    var = [_as_bool(variant[q].get("selected_hit")) for q in ids]
    return {"n_applied": len(ids), "baseline_rate": _mean(base), "variant_rate": _mean(var), "delta": _mean([v - b for v, b in zip(var, base)])}


def _as_bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def summarize_phase10b(result: Mapping[str, Any], *, gate: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = _forbidden(result)
    if forbidden:
        raise Phase10BError(f"compact result contains forbidden fields: {forbidden[:5]}")
    arms = result.get("arms")
    expected = {"baseline_frozen_query", "always_bridge", "consensus_gated_bridge"}
    if not isinstance(arms, Mapping) or set(arms) != expected:
        raise Phase10BError("result arms are malformed")
    baseline = _sample_map(arms["baseline_frozen_query"], "baseline_frozen_query")
    always = _sample_map(arms["always_bridge"], "always_bridge")
    gated = _sample_map(arms["consensus_gated_bridge"], "consensus_gated_bridge")
    for name, samples in (("always_bridge", always), ("consensus_gated_bridge", gated)):
        if any(sample.get("final_context_K") != 5 for sample in samples.values()):
            raise Phase10BError(f"{name}: final context K changed")
    apply_flags = [bool(sample.get("bridge_applied")) for sample in gated.values()]
    apply_rate = sum(apply_flags) / len(apply_flags)
    reps = int(gate.get("bootstrap_repetitions", 2000))
    seed = int(gate.get("bootstrap_seed", 10402))
    always_summary = _arm_summary(baseline, always, repetitions=reps, seed=seed + 20)
    gated_summary = _arm_summary(baseline, gated, repetitions=reps, seed=seed)
    coverage = {"initial_retrieval_hit_rate": _mean([_as_bool(sample.get("initial_retrieval_hit")) for sample in baseline.values()]), "bridge_retrieval_hit_rate": _mean([_as_bool(sample.get("bridge_retrieval_hit")) for sample in gated.values()]), "baseline_selected_hit_rate": _mean([_as_bool(sample.get("selected_hit")) for sample in baseline.values()]), "gated_selected_hit_rate": _mean([_as_bool(sample.get("selected_hit")) for sample in gated.values()])}
    selected_gain = _selected_hit_gain(baseline, gated, gated)
    f1_gate = gated_summary["f1"]["delta"]["ci95_low"] >= float(gate.get("all_question_f1_ci95_low_min", 0.0))
    apply_gate = float(gate.get("apply_rate_min", 0.05)) <= apply_rate <= float(gate.get("apply_rate_max", 0.25))
    hit_gate = selected_gain["delta"] is not None and selected_gain["delta"] >= float(gate.get("selected_hit_gain_min_on_applied", 0.05))
    cost_gate = gated_summary["pipeline_time_ms"]["ratio"] <= float(gate.get("total_pipeline_cost_ratio_max", 1.5))
    decision = "pass_phase10b_screen" if all((f1_gate, apply_gate, hit_gate, cost_gate)) else "inconclusive_or_kill_phase10b_screen"
    return {"schema_version": 1, "decision": decision, "apply_rate": apply_rate, "coverage": coverage, "selected_hit_gain_applied": selected_gain, "arms": {"always_bridge": always_summary, "consensus_gated_bridge": gated_summary}, "gate": {"all_question_f1_gate": f1_gate, "apply_rate_gate": apply_gate, "selected_hit_gain_gate": hit_gate, "cost_gate": cost_gate}}


__all__ = ["FORBIDDEN_FIELDS", "Phase10BError", "paired_bootstrap", "summarize_phase10b"]
