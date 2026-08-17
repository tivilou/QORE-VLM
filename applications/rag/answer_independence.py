"""Diagnostic-only residualization for answer-evidence pair features.

This module has no selector dependency. It observes already-computed candidate
pair matrices and removes variation explained by pre-declared nuisance
features without reading gold labels, generated answers, or QA metrics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


TARGET_FEATURES = ("agreement", "corroboration")
NUISANCE_FEATURES = (
    "answer_confidence_product",
    "lexical_duplication",
    "embedding_redundancy",
)
DIAGNOSTIC_PLUGIN_REGISTRY = {
    "answer_evidence_observer": {"requires": ()},
    "independence_residual": {"requires": ("answer_evidence_observer",)},
}


def _square_matrix(
    value: Any,
    *,
    name: str,
    size: int | None = None,
    bounded: bool = True,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if size is not None and matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must have a zero diagonal")
    if bounded and (np.min(matrix, initial=0.0) < -1e-10 or np.max(matrix, initial=0.0) > 1.0 + 1e-10):
        raise ValueError(f"{name} values must be in [0, 1]")
    return matrix.copy()


def _confidence_product(confidence: Sequence[float], size: int) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64)
    if values.shape != (size,):
        raise ValueError("passage_confidence must match the candidate count")
    if not np.all(np.isfinite(values)):
        raise ValueError("passage_confidence must contain only finite values")
    if np.min(values, initial=0.0) < -1e-10 or np.max(values, initial=0.0) > 1.0 + 1e-10:
        raise ValueError("passage_confidence values must be in [0, 1]")
    product = np.outer(np.clip(values, 0.0, 1.0), np.clip(values, 0.0, 1.0))
    np.fill_diagonal(product, 0.0)
    return product


def _pair_vector(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _pair_matrix(values: np.ndarray, size: int) -> np.ndarray:
    expected = size * (size - 1) // 2
    if values.shape != (expected,):
        raise ValueError(f"pair vector must contain {expected} values")
    matrix = np.zeros((size, size), dtype=np.float64)
    upper = np.triu_indices(size, k=1)
    matrix[upper] = values
    matrix[(upper[1], upper[0])] = values
    return matrix


def residualize_pairwise_feature(
    target: np.ndarray,
    nuisances: Mapping[str, np.ndarray],
    *,
    ridge: float = 1e-6,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return standardized target residuals from an unsupervised ridge fit.

    The fit is performed over all upper-triangle candidate pairs for one
    question. Both target and non-constant nuisance columns are standardized;
    the intercept is unpenalized and slopes use the configured ridge penalty.
    """
    target_matrix = _square_matrix(target, name="target")
    if not nuisances:
        raise ValueError("at least one nuisance feature is required")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative value")
    size = target_matrix.shape[0]
    if size < 3:
        raise ValueError("at least three candidates are required for residualization")

    names = tuple(str(name) for name in nuisances)
    if len(names) != len(set(names)):
        raise ValueError("nuisance feature names must be unique")
    nuisance_matrices = [
        _square_matrix(nuisances[name], name=name, size=size) for name in names
    ]
    nuisance_vectors = [_pair_vector(matrix) for matrix in nuisance_matrices]
    constant_nuisances = [
        name
        for name, values in zip(names, nuisance_vectors)
        if float(np.std(values)) <= epsilon
    ]
    y = _pair_vector(target_matrix)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    pair_count = int(y.size)
    if y_std <= epsilon:
        return np.zeros_like(target_matrix), {
            "pair_count": pair_count,
            "target_mean": y_mean,
            "target_std": y_std,
            "residual_mean": 0.0,
            "residual_std": 0.0,
            "r_squared": None,
            "design_rank": 1,
            "ridge": float(ridge),
            "coefficients": {"intercept": 0.0, **{name: 0.0 for name in names}},
            "constant_nuisances": constant_nuisances,
            "target_constant": True,
        }

    y_scaled = (y - y_mean) / y_std
    columns: list[np.ndarray] = []
    for name, values in zip(names, nuisance_vectors):
        scale = float(np.std(values))
        if scale <= epsilon:
            columns.append(np.zeros_like(values))
        else:
            columns.append((values - float(np.mean(values))) / scale)

    design = np.column_stack([np.ones(pair_count, dtype=np.float64), *columns])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    rhs = design.T @ y_scaled
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    fitted = design @ coefficients
    residual = y_scaled - fitted
    denominator = float(np.sum(y_scaled**2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator
    result = _pair_matrix(residual, size)
    return result, {
        "pair_count": pair_count,
        "target_mean": y_mean,
        "target_std": y_std,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "r_squared": float(r_squared),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "ridge": float(ridge),
        "coefficients": {
            "intercept": float(coefficients[0]),
            **{name: float(value) for name, value in zip(names, coefficients[1:])},
        },
        "constant_nuisances": constant_nuisances,
        "target_constant": False,
    }


def run_diagnostic_pipeline(
    evidence_matrices: Mapping[str, np.ndarray],
    passage_confidence: Sequence[float],
    embedding_redundancy: np.ndarray,
    *,
    plugin_ids: Sequence[str],
    residualization_variants: Mapping[str, Sequence[str]],
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Execute the explicit diagnostic plugin pipeline without selection access."""
    ordered_plugins = [str(plugin_id) for plugin_id in plugin_ids]
    if len(ordered_plugins) != len(set(ordered_plugins)):
        raise ValueError("diagnostic plugins must not be duplicated")
    if not ordered_plugins or ordered_plugins[0] != "answer_evidence_observer":
        raise ValueError("answer_evidence_observer must be the first diagnostic plugin")
    completed: list[str] = []
    for plugin_id in ordered_plugins:
        metadata = DIAGNOSTIC_PLUGIN_REGISTRY.get(plugin_id)
        if metadata is None:
            raise ValueError(f"unknown diagnostic plugin: {plugin_id}")
        missing = [item for item in metadata["requires"] if item not in completed]
        if missing:
            raise ValueError(f"diagnostic plugin {plugin_id} requires {missing}")
        completed.append(plugin_id)
    raw = {
        name: _square_matrix(evidence_matrices[name], name=name)
        for name in TARGET_FEATURES
        if name in evidence_matrices
    }
    if set(raw) != set(TARGET_FEATURES):
        raise ValueError(f"evidence matrices must include {list(TARGET_FEATURES)}")
    size = raw[TARGET_FEATURES[0]].shape[0]
    duplication = _square_matrix(
        evidence_matrices.get("duplication"),
        name="duplication",
        size=size,
    )
    embedding = _square_matrix(
        embedding_redundancy,
        name="embedding_redundancy",
        size=size,
    )
    nuisance_matrices = {
        "answer_confidence_product": _confidence_product(passage_confidence, size),
        "lexical_duplication": duplication,
        "embedding_redundancy": embedding,
    }

    output: dict[str, Any] = {
        "plugin_ids": ordered_plugins,
        "raw": raw,
        "nuisances": nuisance_matrices,
        "residuals": {},
        "fits": {},
    }
    if "independence_residual" not in ordered_plugins:
        return output
    if not residualization_variants:
        raise ValueError("residualization_variants cannot be empty")

    for variant, nuisance_names_value in residualization_variants.items():
        variant_name = str(variant)
        nuisance_names = [str(name) for name in nuisance_names_value]
        if not variant_name or not nuisance_names or len(nuisance_names) != len(set(nuisance_names)):
            raise ValueError(f"invalid residualization variant: {variant_name!r}")
        unknown = sorted(set(nuisance_names) - set(NUISANCE_FEATURES))
        if unknown:
            raise ValueError(f"variant {variant_name} uses unknown nuisances: {unknown}")
        selected_nuisances = {name: nuisance_matrices[name] for name in nuisance_names}
        output["residuals"][variant_name] = {}
        output["fits"][variant_name] = {}
        for target_name in TARGET_FEATURES:
            residual, diagnostics = residualize_pairwise_feature(
                raw[target_name], selected_nuisances, ridge=ridge
            )
            output["residuals"][variant_name][target_name] = residual
            output["fits"][variant_name][target_name] = diagnostics
    return output


def selected_pair_mean(matrix: np.ndarray, indices: Sequence[int]) -> float:
    """Return the unique selected-pair mean for a square feature matrix."""
    values = _square_matrix(matrix, name="selected feature", bounded=False)
    selected = [int(index) for index in indices]
    if len(selected) != len(set(selected)):
        raise ValueError("selected indices must be unique")
    if any(index < 0 or index >= values.shape[0] for index in selected):
        raise ValueError("selected index is out of range")
    if len(selected) < 2:
        return 0.0
    pairs = [
        float(values[left, right])
        for offset, left in enumerate(selected)
        for right in selected[offset + 1 :]
    ]
    return float(np.mean(pairs))


def diagnostic_spec_hash(
    plugin_ids: Sequence[str],
    residualization_variants: Mapping[str, Sequence[str]],
    ridge: float,
) -> str:
    """Hash the formal plugin order and residualization specification."""
    payload = {
        "plugins": [str(value) for value in plugin_ids],
        "residualization_variants": {
            str(name): [str(value) for value in values]
            for name, values in sorted(residualization_variants.items())
        },
        "ridge": float(ridge),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
