"""Aggregation helpers for the Phase 7C corroboration ablation."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from scripts.collab.five_ideas.phase7b_metrics import (
    pearson_correlation,
    spearman_correlation,
    validate_compact_rows,
)


CALIBRATION_FEATURES = (
    "selected_agreement",
    "selected_conflict",
    "selected_decisive_conflict",
    "selected_corroboration",
    "selected_duplication",
)


def _fmean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _group_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row["selection_variant"]),
        int(row["K"]),
        str(row["generator"]),
        int(row["seed"]),
    )


def summarize_ablation_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate compact rows without emitting raw QA or prediction fields."""
    if not rows:
        raise ValueError("ablation rows cannot be empty")
    validate_compact_rows(rows)
    grouped: dict[tuple[str, int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["selected_count"]) != int(row["K"]):
            raise ValueError("selected_count must equal K")
        missing = [feature for feature in CALIBRATION_FEATURES if feature not in row]
        if missing:
            raise ValueError(f"row is missing ablation features: {missing}")
        grouped[_group_key(row)].append(row)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        variant, k_value, generator, seed = key
        group_rows = grouped[key]
        f1_values = [float(row["f1"]) for row in group_rows]
        em_values = [float(row["em"]) for row in group_rows]
        metrics: dict[str, Any] = {
            "selection_variant": variant,
            "K": k_value,
            "generator": generator,
            "seed": seed,
            "n_questions": len(group_rows),
            "mean_em": _fmean(em_values),
            "mean_f1": _fmean(f1_values),
            "feature_means": {},
            "feature_quality_correlation": {},
        }
        for feature in CALIBRATION_FEATURES:
            values = [float(row[feature]) for row in group_rows]
            metrics["feature_means"][feature] = _fmean(values)
            metrics["feature_quality_correlation"][feature] = {
                "pearson_f1": pearson_correlation(values, f1_values),
                "spearman_f1": spearman_correlation(values, f1_values),
                "pearson_em": pearson_correlation(values, em_values),
            }
        groups.append(metrics)

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_variant[group["selection_variant"]].append(group)
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "group_count": len(groups),
        "features": list(CALIBRATION_FEATURES),
        "groups": groups,
        "groups_by_selection_variant": {
            variant: values for variant, values in sorted(by_variant.items())
        },
    }
