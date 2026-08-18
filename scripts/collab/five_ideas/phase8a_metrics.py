"""Pure aggregation and gate logic for the Phase 8A diagnostic."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

import numpy as np

try:
    from applications.rag.counterfactual_influence import (
        fit_aipw_influence,
        spearman,
        validate_compact_payload,
    )
except ImportError:  # pragma: no cover - supports direct local module tests
    from counterfactual_influence import fit_aipw_influence, spearman, validate_compact_payload


FORBIDDEN_FIELDS = ("question", "passages", "gold_answers", "prediction", "raw_prompt")


def _effect_map(fit: Mapping[str, Any], field: str) -> dict[tuple[str, int], float]:
    return {
        (str(row["question_id"]), int(row["passage_index"])): float(row[field])
        for row in fit["effects"]
    }


def _paired_effect_values(
    left: Mapping[tuple[str, int], float], right: Mapping[tuple[str, int], float]
) -> tuple[list[tuple[str, int]], list[float], list[float]]:
    keys = sorted(set(left) & set(right))
    return keys, [float(left[key]) for key in keys], [float(right[key]) for key in keys]


def _cluster_bootstrap_spearman(
    keys: Sequence[tuple[str, int]],
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 100:
        raise ValueError("cluster bootstrap requires at least 100 samples")
    questions = sorted({key[0] for key in keys})
    by_question: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        by_question[key[0]].append(index)
    rng = np.random.default_rng(int(seed))
    estimates: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(questions, size=len(questions), replace=True)
        indexes = [index for question in sampled for index in by_question[str(question)]]
        estimate = spearman(
            [left[index] for index in indexes],
            [right[index] for index in indexes],
        )
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return {"low": None, "high": None, "valid_samples": 0}
    low, high = np.quantile(np.asarray(estimates, dtype=np.float64), [0.025, 0.975])
    return {"low": float(low), "high": float(high), "valid_samples": len(estimates)}


def permute_outcomes_by_complement_pair(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Reassign complete probe outcomes while preserving complement structure."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["question_id"])].append(index)
        if "probe_id" not in row or "pair_index" not in row:
            raise ValueError("placebo requires probe_id and pair_index")
    result = [dict(row) for row in rows]
    for question_id, indexes in sorted(grouped.items()):
        probes: dict[str, list[int]] = defaultdict(list)
        for index in indexes:
            probes[str(rows[index]["probe_id"])].append(index)
        outcomes: dict[str, float] = {}
        pair_probes: dict[int, list[str]] = defaultdict(list)
        for probe_id, probe_indexes in probes.items():
            pair_values = {int(rows[index]["pair_index"]) for index in probe_indexes}
            values = {float(rows[index]["outcome"]) for index in probe_indexes}
            if len(pair_values) != 1 or len(values) != 1:
                raise ValueError("probe rows must share pair_index and outcome")
            pair_index = next(iter(pair_values))
            pair_probes[pair_index].append(probe_id)
            outcomes[probe_id] = next(iter(values))
        if any(len(values) != 2 for values in pair_probes.values()):
            raise ValueError("each complement pair must contain two probes")
        seed_bytes = hashlib.sha256(
            f"phase8a-placebo:{int(seed)}:{question_id}".encode("utf-8")
        ).digest()[:8]
        question_seed = int.from_bytes(seed_bytes, "big") % (2**32)
        rng = np.random.default_rng(question_seed)
        pairs = sorted(pair_probes)
        source_pairs = [pairs[int(value)] for value in rng.permutation(len(pairs))]
        for target_pair, source_pair in zip(pairs, source_pairs):
            target_ids = sorted(pair_probes[target_pair])
            source_ids = sorted(pair_probes[source_pair])
            if int(rng.integers(0, 2)):
                source_ids.reverse()
            for target_id, source_id in zip(target_ids, source_ids):
                for index in probes[target_id]:
                    result[index]["outcome"] = outcomes[source_id]
    return result


def _fit_half(
    rows: Sequence[Mapping[str, Any]], pairs: set[int], *, folds: int, fold_seed: int, ridge: float
) -> dict[str, Any]:
    return fit_aipw_influence(
        rows,
        folds=folds,
        fold_seed=fold_seed,
        ridge=ridge,
        pair_filter=lambda row: int(row["pair_index"]) in pairs,
    )


def summarize_counterfactual_influence(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int,
    fold_seed: int,
    ridge: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    placebo_repetitions: int,
    placebo_seed: int,
    gate: Mapping[str, Any],
    outcome_name: str = "f1",
) -> dict[str, Any]:
    """Fit the frozen AIPW/DIM diagnostics and evaluate the Phase 8A gate."""

    fit_rows = [dict(row) for row in rows]
    if not fit_rows:
        raise ValueError("counterfactual rows cannot be empty")
    pair_indexes = sorted({int(row["pair_index"]) for row in fit_rows})
    if len(pair_indexes) < 2 or len(pair_indexes) % 2:
        raise ValueError("split-half analysis requires an even number of complement pairs")
    midpoint = len(pair_indexes) // 2
    first_pairs, second_pairs = set(pair_indexes[:midpoint]), set(pair_indexes[midpoint:])
    full_fit = fit_aipw_influence(
        fit_rows, folds=folds, fold_seed=fold_seed, ridge=ridge
    )
    first_fit = _fit_half(
        fit_rows, first_pairs, folds=folds, fold_seed=fold_seed, ridge=ridge
    )
    second_fit = _fit_half(
        fit_rows, second_pairs, folds=folds, fold_seed=fold_seed, ridge=ridge
    )

    first_map = _effect_map(first_fit, "aipw_influence")
    second_map = _effect_map(second_fit, "aipw_influence")
    split_keys, split_left, split_right = _paired_effect_values(first_map, second_map)
    split_half = spearman(split_left, split_right)
    split_ci = _cluster_bootstrap_spearman(
        split_keys,
        split_left,
        split_right,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    aipw_map = _effect_map(full_fit, "aipw_influence")
    dim_map = _effect_map(full_fit, "dim_influence")
    effect_keys, aipw_values, dim_values = _paired_effect_values(aipw_map, dim_map)
    aipw_dim = spearman(aipw_values, dim_values)
    by_question: dict[str, list[float]] = defaultdict(list)
    for (question_id, _), value in aipw_map.items():
        by_question[question_id].append(value)
    ranges = {
        question_id: float(max(values) - min(values))
        for question_id, values in sorted(by_question.items())
    }
    range_fraction = float(np.mean([value >= 0.10 for value in ranges.values()]))

    placebo_statistics: list[float] = []
    for repetition in range(int(placebo_repetitions)):
        placebo_rows = permute_outcomes_by_complement_pair(
            fit_rows, int(placebo_seed) + repetition
        )
        placebo_first = _fit_half(
            placebo_rows, first_pairs, folds=folds, fold_seed=fold_seed, ridge=ridge
        )
        placebo_second = _fit_half(
            placebo_rows, second_pairs, folds=folds, fold_seed=fold_seed, ridge=ridge
        )
        _, placebo_left, placebo_right = _paired_effect_values(
            _effect_map(placebo_first, "aipw_influence"),
            _effect_map(placebo_second, "aipw_influence"),
        )
        placebo_statistics.append(spearman(placebo_left, placebo_right))
    placebo_p = float(
        (1 + sum(value >= split_half for value in placebo_statistics))
        / (1 + len(placebo_statistics))
    )

    checks = {
        "split_half_spearman": split_half >= float(gate["minimum_split_half_spearman"]),
        "split_half_ci95_low": split_ci["low"] is not None
        and float(split_ci["low"]) > float(gate["minimum_split_half_ci95_low"]),
        "aipw_dim_agreement": aipw_dim >= float(gate["minimum_aipw_dim_spearman"]),
        "incremental_cross_fitted_r2": float(full_fit["incremental_cross_fitted_r2"])
        >= float(gate["minimum_incremental_cross_fitted_r2"]),
        "question_effect_range": range_fraction
        >= float(gate["minimum_questions_with_effect_range_0_10_fraction"]),
        "placebo": placebo_p <= float(gate["maximum_placebo_p_value"]),
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "outcome": str(outcome_name),
        "effect_count": len(effect_keys),
        "question_count": len(by_question),
        "effects": full_fit["effects"],
        "estimators": {
            "split_half_spearman": split_half,
            "split_half_ci95": split_ci,
            "aipw_dim_spearman": aipw_dim,
            "cross_fitted_r2": full_fit["cross_fitted_r2"],
            "nuisance_only_r2": full_fit["nuisance_only_r2"],
            "incremental_cross_fitted_r2": full_fit["incremental_cross_fitted_r2"],
            "question_effect_ranges": ranges,
            "questions_with_effect_range_0_10_fraction": range_fraction,
            "placebo_p_value": placebo_p,
            "placebo_repetitions": len(placebo_statistics),
            "placebo_split_half_spearman_mean": float(np.mean(placebo_statistics)),
        },
        "cross_fit": {
            "folds": full_fit["folds"],
            "fold_assignments": full_fit["fold_assignments"],
            "question_grouped": True,
        },
        "gate": {
            "status": "pass" if passed else "fail",
            "decision": gate["pass_action"] if passed else gate["fail_action"],
            "checks": checks,
        },
    }
    validate_compact_payload(summary, FORBIDDEN_FIELDS)
    return summary
