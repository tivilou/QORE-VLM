"""Pure aggregation helpers for the Phase 7B calibration gate."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


CALIBRATION_FEATURES = (
    "selected_agreement",
    "selected_conflict",
    "selected_corroboration",
    "selected_duplication",
)
_FORBIDDEN_OUTPUT_KEYS = {"question", "passages", "gold_answers", "prediction", "text"}


def _fmean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def validate_compact_rows(rows: list[Mapping[str, Any]]) -> None:
    """Reject raw QA fields before a calibration payload is written."""
    for row_index, row in enumerate(rows):
        forbidden = sorted(_FORBIDDEN_OUTPUT_KEYS & {str(key).lower() for key in row})
        if forbidden:
            raise ValueError(f"row {row_index} contains forbidden raw fields: {forbidden}")


def pearson_correlation(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Return Pearson correlation, or None for a constant/empty input."""
    x = [float(value) for value in left]
    y = [float(value) for value in right]
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = rank
        cursor = end
    return ranks


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Return rank correlation without requiring scipy."""
    x = [float(value) for value in left]
    y = [float(value) for value in right]
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson_correlation(_ranks(x), _ranks(y))


def _group_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row["selection_variant"]),
        int(row["K"]),
        str(row["generator"]),
        int(row["seed"]),
    )


def summarize_calibration_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question rows and measure feature-to-quality association."""
    if not rows:
        raise ValueError("calibration rows cannot be empty")
    validate_compact_rows(rows)
    grouped: dict[tuple[str, int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["selected_count"]) != int(row["K"]):
            raise ValueError("selected_count must equal K")
        grouped[_group_key(row)].append(row)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        selection_variant, k_value, generator, seed = key
        group_rows = grouped[key]
        metrics: dict[str, Any] = {
            "selection_variant": selection_variant,
            "K": k_value,
            "generator": generator,
            "seed": seed,
            "n_questions": len(group_rows),
            "mean_em": _fmean([float(row["em"]) for row in group_rows]),
            "mean_f1": _fmean([float(row["f1"]) for row in group_rows]),
            "feature_means": {},
            "feature_quality_correlation": {},
        }
        for feature in CALIBRATION_FEATURES:
            values = [float(row[feature]) for row in group_rows]
            metrics["feature_means"][feature] = _fmean(values)
            metrics["feature_quality_correlation"][feature] = {
                "pearson_f1": pearson_correlation(values, [float(row["f1"]) for row in group_rows]),
                "spearman_f1": spearman_correlation(values, [float(row["f1"]) for row in group_rows]),
                "pearson_em": pearson_correlation(values, [float(row["em"]) for row in group_rows]),
            }
        groups.append(metrics)

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_variant[group["selection_variant"]].append(group)
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "group_count": len(groups),
        "groups": groups,
        "groups_by_selection_variant": {
            variant: values for variant, values in sorted(by_variant.items())
        },
    }
