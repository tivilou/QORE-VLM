"""Pure aggregation and gate logic for the Phase 7D residual diagnostic."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts.collab.five_ideas.phase7b_metrics import (
    pearson_correlation,
    spearman_correlation,
    validate_compact_rows,
)


TARGET_FEATURES = ("agreement", "corroboration")
NUISANCE_OUTPUT_FEATURES = (
    "selected_answer_confidence_product",
    "selected_lexical_duplication",
    "selected_embedding_redundancy",
)


def _group_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row["selection_variant"]),
        int(row["K"]),
        str(row["generator"]),
        int(row["seed"]),
    )


def _bootstrap_correlation(
    values: Sequence[float],
    quality: Sequence[float],
    *,
    method: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    correlation = pearson_correlation if method == "pearson" else spearman_correlation
    rng = np.random.default_rng(seed)
    left = np.asarray(values, dtype=np.float64)
    right = np.asarray(quality, dtype=np.float64)
    estimates: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(left), size=len(left))
        estimate = correlation(left[indices], right[indices])
        if estimate is not None and np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return {"low": None, "high": None, "valid_samples": 0}
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "valid_samples": len(estimates),
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**32)


def _feature_names(residualization_variants: Sequence[str]) -> list[str]:
    features = [f"selected_{target}" for target in TARGET_FEATURES]
    features.extend(NUISANCE_OUTPUT_FEATURES)
    for variant in residualization_variants:
        features.extend(
            f"selected_residual_{variant}_{target}" for target in TARGET_FEATURES
        )
    return features


def _summarize_group(
    key: tuple[str, int, str, int],
    rows: list[Mapping[str, Any]],
    *,
    features: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    variant, k_value, generator, seed = key
    question_ids = [str(row["question_id"]) for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"duplicate question_id in group {key}")
    f1_values = [float(row["f1"]) for row in rows]
    em_values = [float(row["em"]) for row in rows]
    metrics: dict[str, Any] = {
        "selection_variant": variant,
        "K": k_value,
        "generator": generator,
        "seed": seed,
        "n_questions": len(rows),
        "mean_em": statistics.fmean(em_values),
        "mean_f1": statistics.fmean(f1_values),
        "feature_means": {},
        "feature_quality_association": {},
    }
    for feature in features:
        values = [float(row[feature]) for row in rows]
        pearson_f1 = pearson_correlation(values, f1_values)
        spearman_f1 = spearman_correlation(values, f1_values)
        metrics["feature_means"][feature] = statistics.fmean(values)
        metrics["feature_quality_association"][feature] = {
            "pearson_f1": pearson_f1,
            "spearman_f1": spearman_f1,
            "pearson_em": pearson_correlation(values, em_values),
            "pearson_f1_ci95": _bootstrap_correlation(
                values,
                f1_values,
                method="pearson",
                samples=bootstrap_samples,
                seed=_stable_seed(bootstrap_seed, *map(str, key), feature, "pearson"),
            ),
            "spearman_f1_ci95": _bootstrap_correlation(
                values,
                f1_values,
                method="spearman",
                samples=bootstrap_samples,
                seed=_stable_seed(bootstrap_seed, *map(str, key), feature, "spearman"),
            ),
        }
    return metrics


def _evaluate_gate(groups: list[dict[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    primary = str(gate["primary_selector"])
    selectors = [primary, *[str(value) for value in gate["consistency_selectors"]]]
    if len(selectors) != len(set(selectors)):
        raise ValueError("gate selectors must be unique")
    group_by_selector = {str(group["selection_variant"]): group for group in groups}
    missing = sorted(set(selectors) - set(group_by_selector))
    if missing:
        raise ValueError(f"gate selectors are missing result groups: {missing}")

    residual_variant = str(gate["residual_variant"])
    minimum_residual = float(gate["minimum_residual_spearman_f1"])
    minimum_improvement = float(gate["minimum_improvement_over_raw"])
    minimum_positive = int(gate["minimum_positive_selectors"])
    minimum_ci_low = float(gate["minimum_bootstrap_ci_low"])
    target_results: dict[str, Any] = {}
    for target in TARGET_FEATURES:
        raw_feature = f"selected_{target}"
        residual_feature = f"selected_residual_{residual_variant}_{target}"
        primary_associations = group_by_selector[primary]["feature_quality_association"]
        raw = primary_associations[raw_feature]["spearman_f1"]
        residual = primary_associations[residual_feature]["spearman_f1"]
        ci_low = primary_associations[residual_feature]["spearman_f1_ci95"]["low"]
        improvement = None if raw is None or residual is None else float(residual - raw)
        selector_values = {
            selector: group_by_selector[selector]["feature_quality_association"][
                residual_feature
            ]["spearman_f1"]
            for selector in selectors
        }
        positive_selectors = sum(
            value is not None and value > 0.0 for value in selector_values.values()
        )
        checks = {
            "primary_residual_at_least_minimum": residual is not None and residual >= minimum_residual,
            "residual_improves_over_raw": improvement is not None and improvement >= minimum_improvement,
            "selector_consistency": positive_selectors >= minimum_positive,
            "bootstrap_not_strongly_negative": ci_low is not None and ci_low >= minimum_ci_low,
        }
        target_results[target] = {
            "raw_primary_spearman_f1": raw,
            "residual_primary_spearman_f1": residual,
            "residual_minus_raw": improvement,
            "residual_primary_ci95_low": ci_low,
            "residual_spearman_by_selector": selector_values,
            "positive_selector_count": positive_selectors,
            "checks": checks,
            "passes": all(checks.values()),
        }
    passes = any(value["passes"] for value in target_results.values())
    return {
        "status": "pass" if passes else "fail",
        "decision": "controlled_followup_eligible" if passes else "stop_answer_identity_objective",
        "primary_endpoint": "Spearman correlation between full-nuisance residual and F1",
        "thresholds": {
            "minimum_residual_spearman_f1": minimum_residual,
            "minimum_improvement_over_raw": minimum_improvement,
            "minimum_positive_selectors": minimum_positive,
            "minimum_bootstrap_ci_low": minimum_ci_low,
        },
        "targets": target_results,
    }


def summarize_independence_rows(
    rows: list[Mapping[str, Any]],
    *,
    residualization_variants: Sequence[str],
    gate: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize compact rows and apply the pre-declared Phase 7D gate."""
    if not rows:
        raise ValueError("independence rows cannot be empty")
    validate_compact_rows(rows)
    variants = [str(value) for value in residualization_variants]
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("residualization variants must be unique and non-empty")
    features = _feature_names(variants)
    grouped: dict[tuple[str, int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["selected_count"]) != int(row["K"]):
            raise ValueError("selected_count must equal K")
        if not 0.0 <= float(row["f1"]) <= 1.0 or not 0.0 <= float(row["em"]) <= 1.0:
            raise ValueError("F1 and EM must be in [0, 1]")
        missing = [feature for feature in features if feature not in row]
        if missing:
            raise ValueError(f"row is missing diagnostic features: {missing}")
        if not all(np.isfinite(float(row[feature])) for feature in features):
            raise ValueError("diagnostic features must be finite")
        grouped[_group_key(row)].append(row)

    groups = [
        _summarize_group(
            key,
            grouped[key],
            features=features,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for key in sorted(grouped)
    ]
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "row_count": len(rows),
        "group_count": len(groups),
        "features": features,
        "groups": groups,
        "gate": _evaluate_gate(groups, gate),
    }
