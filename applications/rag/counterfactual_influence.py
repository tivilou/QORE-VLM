"""Diagnostic-only counterfactual passage probes and AIPW estimation.

This module deliberately has no dependency on the selector or evaluator. It
creates compact, deterministic context probes and estimates passage inclusion
effects from already-computed numeric outcomes. The estimator output is never a
selection score in Phase 8A.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


PLUGIN_IDS = (
    "full_context_probe",
    "balanced_subset_probe",
    "doubly_robust_influence",
)

COVARIATE_NAMES = (
    "answer_score",
    "retrieval_rank",
    "original_selection_position",
    "passage_token_count",
    "other_context_size",
    "other_context_answer_score_sum",
    "other_context_token_count",
    "other_context_embedding_redundancy",
)


@dataclass(frozen=True)
class ContextProbe:
    """One compact context intervention descriptor."""

    probe_id: str
    question_id: str
    mask: tuple[int, ...]
    ordered_ranks: tuple[int, ...]
    anchor: str
    pair_index: int | None
    propensity: float | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mask"] = list(self.mask)
        payload["ordered_ranks"] = list(self.ordered_ranks)
        return payload


def stable_id(value: str, *, namespace: str) -> str:
    """Return a short stable identifier without exposing the source string."""

    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()
    return digest[:24]


def validate_compact_payload(payload: Any, forbidden_fields: Iterable[str]) -> None:
    """Reject raw-data field names anywhere in an Exchange payload."""

    forbidden = {str(value) for value in forbidden_fields}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                if key_text in forbidden:
                    raise ValueError(f"forbidden raw field at {path}.{key_text}")
                visit(child, f"{path}.{key_text}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "$")


def _seed_for(question_id: str, seed: int, label: str) -> int:
    raw = f"{label}:{seed}:{question_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def _mask_key(mask: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in mask)


def _mask_from_int(value: int, count: int) -> tuple[int, ...]:
    return tuple((value >> index) & 1 for index in range(count))


def _mask_int(mask: Sequence[int]) -> int:
    result = 0
    for index, value in enumerate(mask):
        result |= (int(value) & 1) << index
    return result


def _pair_candidates(count: int) -> list[tuple[int, int]]:
    if count <= 1:
        raise ValueError("at least two selected passages are required")
    full = (1 << count) - 1
    pairs: list[tuple[int, int]] = []
    for value in range(1, full):
        complement = full ^ value
        if value < complement:
            pairs.append((value, complement))
    return pairs


def build_context_probes(
    question_id: str,
    selected_ranks: Sequence[int],
    *,
    seed: int,
    complement_pairs: int = 8,
    include_full_anchor: bool = True,
    include_empty_anchor: bool = True,
    randomize_included_order: bool = True,
) -> list[ContextProbe]:
    """Build a deterministic full anchor, balanced masks, and empty anchor.

    For ``count=5`` and ``complement_pairs=8`` this returns 18 probes: one full
    anchor, sixteen effect-fit probes, and one empty anchor. Every selected
    passage is included exactly eight times among the effect-fit probes.
    """

    ranks = tuple(int(value) for value in selected_ranks)
    count = len(ranks)
    if count < 2 or len(set(ranks)) != count:
        raise ValueError("selected_ranks must contain unique ranks")
    pairs = _pair_candidates(count)
    if not 1 <= complement_pairs <= len(pairs):
        raise ValueError("complement_pairs exceeds the available mask pairs")

    pair_rng = np.random.default_rng(_seed_for(question_id, seed, "pair"))
    pair_order = pair_rng.permutation(len(pairs))[:complement_pairs]
    chosen_pairs = [pairs[int(index)] for index in pair_order]

    probes: list[ContextProbe] = []

    def append_probe(
        mask: tuple[int, ...],
        *,
        anchor: str,
        pair_index: int | None,
        propensity: float | None,
        order_rng: np.random.Generator | None = None,
    ) -> None:
        included = [index for index, value in enumerate(mask) if value]
        if order_rng is not None and len(included) > 1:
            included = [included[int(index)] for index in order_rng.permutation(len(included))]
        ordered = tuple(ranks[index] for index in included)
        mask_text = "".join(str(value) for value in mask)
        probe_id = stable_id(
            f"{question_id}:{mask_text}:{','.join(str(value) for value in ordered)}:{anchor}",
            namespace="phase8a-probe",
        )
        probes.append(ContextProbe(
            probe_id=probe_id,
            question_id=question_id,
            mask=mask,
            ordered_ranks=ordered,
            anchor=anchor,
            pair_index=pair_index,
            propensity=propensity,
        ))

    full_mask = (1,) * count
    empty_mask = (0,) * count
    if include_full_anchor:
        append_probe(full_mask, anchor="full", pair_index=None, propensity=1.0)

    for pair_index, pair in enumerate(chosen_pairs):
        for value in pair:
            mask = _mask_from_int(value, count)
            order_rng = (
                np.random.default_rng(_seed_for(question_id, seed + pair_index, f"order:{value}"))
                if randomize_included_order
                else None
            )
            append_probe(mask, anchor="effect", pair_index=pair_index, propensity=0.5, order_rng=order_rng)

    if include_empty_anchor:
        append_probe(empty_mask, anchor="empty", pair_index=None, propensity=0.0)
    validate_probe_design(
        probes,
        selected_count=count,
        complement_pairs=complement_pairs,
        require_anchors=include_full_anchor and include_empty_anchor,
    )
    return probes


def validate_probe_design(
    probes: Sequence[ContextProbe],
    *,
    selected_count: int,
    complement_pairs: int,
    require_anchors: bool = True,
) -> None:
    """Validate balance, complementarity, ordering, and anchor invariants."""

    expected_effect = 2 * complement_pairs
    effects = [probe for probe in probes if probe.anchor == "effect"]
    anchors = [probe for probe in probes if probe.anchor != "effect"]
    if len(effects) != expected_effect:
        raise ValueError(f"expected {expected_effect} effect probes, got {len(effects)}")
    if require_anchors and {probe.anchor for probe in anchors} != {"full", "empty"}:
        raise ValueError("full and empty anchors are required")
    if len({probe.probe_id for probe in probes}) != len(probes):
        raise ValueError("probe IDs must be unique")
    if any(len(probe.mask) != selected_count for probe in probes):
        raise ValueError("probe mask width mismatch")
    if any(probe.anchor == "effect" and probe.propensity != 0.5 for probe in probes):
        raise ValueError("effect probes must have propensity 0.5")
    for probe in probes:
        expected_order = tuple(
            index for index, value in enumerate(probe.mask) if value
        )
        if len(probe.ordered_ranks) != len(expected_order):
            raise ValueError("ordered ranks must match mask cardinality")
    pair_masks: dict[int, set[int]] = defaultdict(set)
    for probe in effects:
        if probe.pair_index is None:
            raise ValueError("effect probes require pair_index")
        pair_masks[int(probe.pair_index)].add(_mask_int(probe.mask))
    full = (1 << selected_count) - 1
    if len(pair_masks) != complement_pairs:
        raise ValueError("unexpected complement-pair count")
    for values in pair_masks.values():
        if len(values) != 2:
            raise ValueError("each pair must contain exactly two masks")
        first, second = sorted(values)
        if first ^ second != full:
            raise ValueError("pair masks are not complements")
    inclusion_counts = np.zeros(selected_count, dtype=int)
    for probe in effects:
        inclusion_counts += np.asarray(probe.mask, dtype=int)
    if not np.all(inclusion_counts == complement_pairs):
        raise ValueError(f"inclusion counts are not balanced: {inclusion_counts.tolist()}")


def _feature_matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    values = []
    for row in rows:
        covariates = row.get("covariates", {})
        values.append([float(covariates.get(name, 0.0)) for name in names])
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("covariate matrix has an invalid shape")
    if not np.isfinite(matrix).all():
        raise ValueError("covariates must be finite")
    return matrix


def _ridge_fit(features: np.ndarray, outcomes: np.ndarray, ridge: float) -> np.ndarray:
    if features.ndim != 2 or outcomes.ndim != 1 or len(features) != len(outcomes):
        raise ValueError("ridge inputs have incompatible shapes")
    design = np.column_stack([np.ones(len(features)), features])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ outcomes
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _ridge_predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    return design @ coefficients


def _standardize_from_train(
    train: np.ndarray, heldout: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return (train - mean) / scale, (heldout - mean) / scale


def _fold_for(question_id: str, folds: int, seed: int) -> int:
    if folds < 2:
        return 0
    return int.from_bytes(
        hashlib.sha256(f"fold:{seed}:{question_id}".encode("utf-8")).digest()[:8], "big"
    ) % folds


def _safe_r2(outcomes: np.ndarray, predictions: np.ndarray) -> float:
    variance = float(np.sum((outcomes - np.mean(outcomes)) ** 2))
    if variance <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((outcomes - predictions) ** 2) / variance)


def fit_aipw_influence(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int = 5,
    fold_seed: int = 8102,
    ridge: float = 1e-6,
    pair_filter: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Fit grouped cross-fitted AIPW effects from compact probe rows.

    Each row represents one passage under one randomized probe. Treatment is the
    passage's inclusion bit and propensity must be exactly 0.5 for effect-fit
    rows. The returned effect table is compact and contains no source text.
    """

    selected_rows = [dict(row) for row in rows if pair_filter is None or pair_filter(row)]
    if not selected_rows:
        raise ValueError("no rows available for AIPW fit")
    required = {"question_id", "passage_index", "treatment", "propensity", "outcome", "covariates"}
    for row_index, row in enumerate(selected_rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"AIPW row {row_index} missing fields: {missing}")
    if any(float(row["propensity"]) != 0.5 for row in selected_rows):
        raise ValueError("AIPW requires propensity 0.5 on every effect-fit row")
    if any(int(row["treatment"]) not in {0, 1} for row in selected_rows):
        raise ValueError("treatment must be binary")

    question_ids = sorted({str(row["question_id"]) for row in selected_rows})
    actual_folds = max(2, min(int(folds), len(question_ids)))
    X = _feature_matrix(selected_rows, COVARIATE_NAMES)
    y = np.asarray([float(row["outcome"]) for row in selected_rows], dtype=np.float64)
    treatment = np.asarray([int(row["treatment"]) for row in selected_rows], dtype=np.int64)
    if not np.isfinite(y).all():
        raise ValueError("outcomes must be finite")

    # Initialize all outputs so a deterministic hash fold that happens to be
    # empty (or leaves no training question) cannot leak uninitialized values.
    predictions = np.full(len(selected_rows), float(np.mean(y)), dtype=np.float64)
    nuisance_predictions = np.full(len(selected_rows), float(np.mean(y)), dtype=np.float64)
    pseudo = np.full(len(selected_rows), 0.0, dtype=np.float64)
    for fold in range(actual_folds):
        heldout = np.asarray([
            _fold_for(str(row["question_id"]), actual_folds, fold_seed) == fold
            for row in selected_rows
        ], dtype=bool)
        train = ~heldout
        if not heldout.any():
            continue
        if not train.any():
            # A one-question synthetic fixture can have no training fold. The
            # global mean fallback is explicit; real Phase 8A runs have 50
            # questions and therefore use all five grouped cross-fit folds.
            y_holdout = y[heldout]
            t_holdout = treatment[heldout]
            global_mean = float(np.mean(y))
            # With no training questions, use m0=m1=global_mean in the AIPW
            # score instead of treating the observed outcome as an effect.
            pseudo[heldout] = (
                t_holdout * (y_holdout - global_mean) / 0.5
                - (1 - t_holdout) * (y_holdout - global_mean) / 0.5
            )
            continue
        train_indexes = np.flatnonzero(train)
        heldout_indexes = np.flatnonzero(heldout)
        x_train, x_holdout = _standardize_from_train(X[train_indexes], X[heldout_indexes])
        train_treatment = treatment[train_indexes]
        train0 = train_treatment == 0
        train1 = train_treatment == 1
        global_mean = float(np.mean(y[train]))
        beta0 = _ridge_fit(x_train[train0], y[train_indexes][train0], ridge) if train0.any() else None
        beta1 = _ridge_fit(x_train[train1], y[train_indexes][train1], ridge) if train1.any() else None
        beta_nuisance = _ridge_fit(x_train, y[train_indexes], ridge)
        m0 = _ridge_predict(x_holdout, beta0) if beta0 is not None else np.full(len(x_holdout), global_mean)
        m1 = _ridge_predict(x_holdout, beta1) if beta1 is not None else np.full(len(x_holdout), global_mean)
        predictions[heldout] = np.where(treatment[heldout] == 1, m1, m0)
        nuisance_predictions[heldout] = _ridge_predict(x_holdout, beta_nuisance)
        y_holdout = y[heldout]
        t_holdout = treatment[heldout]
        pseudo[heldout] = (
            m1 - m0
            + t_holdout * (y_holdout - m1) / 0.5
            - (1 - t_holdout) * (y_holdout - m0) / 0.5
        )

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(selected_rows):
        grouped[(str(row["question_id"]), int(row["passage_index"]))].append(index)
    effects: list[dict[str, Any]] = []
    for (question_id, passage_index), indexes in sorted(grouped.items()):
        indexes_array = np.asarray(indexes, dtype=int)
        treated_indexes = [index for index in indexes if treatment[index] == 1]
        control_indexes = [index for index in indexes if treatment[index] == 0]
        if not treated_indexes or not control_indexes:
            raise ValueError("each question/passage unit requires treated and control rows")
        raw = float(np.mean(y[treated_indexes]) - np.mean(y[control_indexes]))
        effects.append({
            "question_id": question_id,
            "passage_index": passage_index,
            "aipw_influence": float(np.mean(pseudo[indexes_array])),
            "dim_influence": raw,
            "n_treated": len(treated_indexes),
            "n_control": len(control_indexes),
            "outcome_mean": float(np.mean(y[indexes_array])),
        })

    return {
        "effects": effects,
        "row_count": len(selected_rows),
        "question_count": len(question_ids),
        "folds": actual_folds,
        "cross_fitted_r2": _safe_r2(y, predictions),
        "nuisance_only_r2": _safe_r2(y, nuisance_predictions),
        "incremental_cross_fitted_r2": _safe_r2(y, predictions) - _safe_r2(y, nuisance_predictions),
        "fold_assignments": {
            str(question_id): _fold_for(str(question_id), actual_folds, fold_seed)
            for question_id in question_ids
        },
    }


def spearman(values_left: Sequence[float], values_right: Sequence[float]) -> float:
    """Tie-aware Spearman correlation without scipy."""

    left = np.asarray(values_left, dtype=np.float64)
    right = np.asarray(values_right, dtype=np.float64)
    if len(left) != len(right) or len(left) < 2:
        return 0.0

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[order[stop]] == values[order[start]]:
                stop += 1
            result[order[start:stop]] = (start + stop - 1) / 2.0
            start = stop
        return result

    left_rank = ranks(left)
    right_rank = ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_rank, right_rank) / denominator)


def permute_treatment_within_question(rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Create a treatment placebo while preserving per-question counts."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["question_id"])].append(index)
    rng = np.random.default_rng(int(seed))
    result = [dict(row) for row in rows]
    for indexes in grouped.values():
        labels = np.asarray([int(rows[index]["treatment"]) for index in indexes], dtype=np.int64)
        shuffled = labels[rng.permutation(len(labels))]
        for index, treatment in zip(indexes, shuffled):
            result[index]["treatment"] = int(treatment)
    return result


def permute_complement_preserving_treatment(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Placebo treatment labels by permuting passage positions per question.

    Rows must carry the original binary mask and passage index. A single
    permutation is applied to all complement pairs within each question, so
    every pair remains complementary and every passage keeps the same total
    inclusion count while treatment is detached from its original passage.
    """

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["question_id"])].append(index)
        if "mask" not in row or "passage_index" not in row:
            raise ValueError("complement-preserving placebo requires mask and passage_index")
    result = [dict(row) for row in rows]
    for question_id, indexes in sorted(grouped.items()):
        widths = {len(tuple(rows[index]["mask"])) for index in indexes}
        if len(widths) != 1:
            raise ValueError(f"inconsistent mask widths for question {question_id}")
        width = next(iter(widths))
        if width < 2:
            raise ValueError("placebo requires at least two passage positions")
        probes: dict[str, list[int]] = defaultdict(list)
        for index in indexes:
            probes[str(rows[index].get("probe_id", ""))].append(index)
        if "" in probes:
            raise ValueError("complement-preserving placebo requires probe_id")
        pair_probes: dict[int, list[str]] = defaultdict(list)
        probe_masks: dict[str, tuple[int, ...]] = {}
        for probe_id, probe_indexes in probes.items():
            pair_values = {int(rows[index]["pair_index"]) for index in probe_indexes}
            masks = {tuple(int(value) for value in rows[index]["mask"]) for index in probe_indexes}
            if len(pair_values) != 1 or len(masks) != 1:
                raise ValueError("probe rows must share one pair_index and one mask")
            pair_index = next(iter(pair_values))
            pair_probes[pair_index].append(probe_id)
            probe_masks[probe_id] = next(iter(masks))
        for probe_ids in pair_probes.values():
            if len(probe_ids) != 2:
                raise ValueError("each placebo pair requires two probes")
            left, right = probe_masks[probe_ids[0]], probe_masks[probe_ids[1]]
            if any(a ^ b != 1 for a, b in zip(left, right)):
                raise ValueError("placebo source masks must be complements")

        rng = np.random.default_rng(_seed_for(question_id, int(seed), "placebo"))
        target_pairs = sorted(pair_probes)
        source_pairs = [target_pairs[int(value)] for value in rng.permutation(len(target_pairs))]
        for target_pair, source_pair in zip(target_pairs, source_pairs):
            target_probe_ids = sorted(pair_probes[target_pair])
            source_probe_ids = sorted(pair_probes[source_pair])
            if int(rng.integers(0, 2)):
                source_probe_ids.reverse()
            for target_probe_id, source_probe_id in zip(target_probe_ids, source_probe_ids):
                assigned_mask = probe_masks[source_probe_id]
                for index in probes[target_probe_id]:
                    passage_index = int(rows[index]["passage_index"])
                    if not 0 <= passage_index < width:
                        raise ValueError("passage_index is outside placebo mask width")
                    result[index]["mask"] = list(assigned_mask)
                    result[index]["treatment"] = int(assigned_mask[passage_index])
    return result
