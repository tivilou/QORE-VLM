"""Compact paired metrics and gates for Phase 10A."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


FORBIDDEN_FIELDS = {"question", "passages", "gold_answers", "prediction", "raw_prompt", "text"}


class Phase10AError(ValueError):
    """Raised when a compact Phase 10A result is malformed."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Phase10AError("cannot average an empty sequence")
    return float(statistics.fmean(float(value) for value in values))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def paired_bootstrap(values: Sequence[float], *, repetitions: int, seed: int) -> dict[str, float]:
    if not values or repetitions < 100:
        raise Phase10AError("bootstrap requires values and at least 100 repetitions")
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
        raise Phase10AError(f"{name}: samples must be non-empty")
    result: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping) or not str(sample.get("question_id", "")):
            raise Phase10AError(f"{name}: every sample needs question_id")
        question_id = str(sample["question_id"])
        if question_id in result:
            raise Phase10AError(f"{name}: duplicate question_id")
        for key in ("em", "f1", "generation_time_ms"):
            value = sample.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise Phase10AError(f"{name}/{question_id}: invalid {key}")
        result[question_id] = sample
    return result


def _arm_summary(baseline: Mapping[str, Mapping[str, Any]], variant: Mapping[str, Mapping[str, Any]], *, repetitions: int, seed: int) -> dict[str, Any]:
    if set(baseline) != set(variant):
        raise Phase10AError("arms must contain identical question IDs")
    ids = sorted(baseline)
    f1 = [float(variant[q]["f1"]) - float(baseline[q]["f1"]) for q in ids]
    em = [float(variant[q]["em"]) - float(baseline[q]["em"]) for q in ids]
    base_time = sum(float(baseline[q]["generation_time_ms"]) for q in ids)
    variant_time = sum(float(variant[q]["generation_time_ms"]) for q in ids)
    return {
        "n_questions": len(ids),
        "f1": {"baseline_mean": _mean([float(baseline[q]["f1"]) for q in ids]), "variant_mean": _mean([float(variant[q]["f1"]) for q in ids]), "delta": paired_bootstrap(f1, repetitions=repetitions, seed=seed)},
        "em": {"baseline_mean": _mean([float(baseline[q]["em"]) for q in ids]), "variant_mean": _mean([float(variant[q]["em"]) for q in ids]), "delta": paired_bootstrap(em, repetitions=repetitions, seed=seed + 1)},
        "generation_time_ms": {"baseline_total": base_time, "variant_total": variant_time, "ratio": variant_time / base_time if base_time else float("inf")},
    }


def summarize_phase10a(result: Mapping[str, Any], *, gate: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = _forbidden(result)
    if forbidden:
        raise Phase10AError(f"compact result contains forbidden fields: {forbidden[:5]}")
    arms = result.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"baseline_k5", "always_wide", "adaptive"}:
        raise Phase10AError("result arms must be baseline_k5, always_wide, adaptive")
    baseline = _sample_map(arms["baseline_k5"], "baseline_k5")
    wide = _sample_map(arms["always_wide"], "always_wide")
    adaptive = _sample_map(arms["adaptive"], "adaptive")
    for name, samples in (("always_wide", wide), ("adaptive", adaptive)):
        for question_id, sample in samples.items():
            if sample.get("prefix_parity") is not True:
                raise Phase10AError(f"{name}/{question_id}: selected prefix parity failed")
    apply_flags = [bool(sample.get("applied")) for sample in adaptive.values()]
    apply_rate = sum(apply_flags) / len(apply_flags)
    adaptive_summary = _arm_summary(baseline, adaptive, repetitions=int(gate.get("bootstrap_repetitions", 2000)), seed=int(gate.get("bootstrap_seed", 10401)))
    wide_summary = _arm_summary(baseline, wide, repetitions=int(gate.get("bootstrap_repetitions", 2000)), seed=int(gate.get("bootstrap_seed", 10401)) + 20)
    risk_bins: dict[str, list[float]] = {"low": [], "high": []}
    threshold = float(result.get("config", {}).get("risk_threshold", 0.55))
    for question_id, sample in adaptive.items():
        risk_bins["high" if float(sample["risk_score"]) >= threshold else "low"].append(float(sample["f1"]) - float(baseline[question_id]["f1"]))
    risk_bin_summary = {name: (paired_bootstrap(values, repetitions=int(gate.get("bootstrap_repetitions", 2000)), seed=int(gate.get("bootstrap_seed", 10401)) + 50 + index) if values else None) for index, (name, values) in enumerate(risk_bins.items())}
    f1_gate = adaptive_summary["f1"]["delta"]["ci95_low"] >= float(gate.get("f1_ci95_low_min", 0.0))
    em_gate = adaptive_summary["em"]["delta"]["mean"] >= float(gate.get("em_delta_min", 0.03))
    rate_gate = float(gate.get("apply_rate_min", 0.05)) <= apply_rate <= float(gate.get("apply_rate_max", 0.30))
    cost_gate = adaptive_summary["generation_time_ms"]["ratio"] <= float(gate.get("cost_ratio_max", 1.5))
    harm_gate = all(item is None or item["ci95_low"] >= float(gate.get("risk_bin_ci95_low_min", -0.01)) for item in risk_bin_summary.values())
    return {
        "schema_version": 1,
        "decision": "pass_adaptive_context_screen" if all((f1_gate, em_gate, rate_gate, cost_gate, harm_gate)) else "inconclusive_or_kill_adaptive_context_screen",
        "apply_rate": apply_rate,
        "arms": {"always_wide": wide_summary, "adaptive": adaptive_summary},
        "risk_bins": risk_bin_summary,
        "gate": {"f1_gate": f1_gate, "em_gate": em_gate, "apply_rate_gate": rate_gate, "cost_gate": cost_gate, "risk_bin_harm_gate": harm_gate},
    }


__all__ = ["FORBIDDEN_FIELDS", "Phase10AError", "paired_bootstrap", "summarize_phase10a"]
