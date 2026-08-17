"""Unit tests for the Phase 7D diagnostic-only plugin pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from applications.rag.answer_independence import (
    run_diagnostic_pipeline,
    residualize_pairwise_feature,
    selected_pair_mean,
)


def _matrix(values: list[float], size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    upper = np.triu_indices(size, k=1)
    matrix[upper] = values
    matrix[(upper[1], upper[0])] = values
    return matrix


def test_observer_is_a_null_adapter_and_does_not_create_residuals():
    agreement = _matrix([0.1, 0.4, 0.7, 0.2, 0.3, 0.9], 4)
    corroboration = _matrix([0.2, 0.3, 0.5, 0.1, 0.2, 0.6], 4)
    duplication = _matrix([0.2, 0.1, 0.4, 0.3, 0.2, 0.5], 4)
    output = run_diagnostic_pipeline(
        {"agreement": agreement, "corroboration": corroboration, "duplication": duplication},
        [0.2, 0.4, 0.6, 0.8],
        duplication,
        plugin_ids=["answer_evidence_observer"],
        residualization_variants={},
    )
    assert output["residuals"] == {}
    np.testing.assert_array_equal(output["raw"]["agreement"], agreement)
    np.testing.assert_array_equal(output["raw"]["corroboration"], corroboration)


def test_residualization_removes_a_linear_nuisance_signal():
    confidence = _matrix([0.1, 0.2, 0.5, 0.7, 0.4, 0.8], 4)
    duplication = _matrix([0.3, 0.4, 0.2, 0.8, 0.5, 0.1], 4)
    embedding = _matrix([0.6, 0.2, 0.7, 0.3, 0.4, 0.9], 4)
    values = 0.4 * confidence + 0.3 * duplication + 0.2 * embedding
    target_values = values[np.triu_indices(4, k=1)]
    target_values = (target_values - target_values.min()) / (target_values.max() - target_values.min())
    target = _matrix(target_values.tolist(), 4)
    residual, fit = residualize_pairwise_feature(
        target,
        {
            "confidence": confidence,
            "duplication": duplication,
            "embedding": embedding,
        },
        ridge=0.0,
    )
    assert fit["r_squared"] > 0.999999
    assert fit["residual_std"] < 1e-6
    assert np.max(np.abs(residual)) < 1e-5


def test_pipeline_is_deterministic_and_rejects_invalid_composition():
    matrices = {
        "agreement": _matrix([0.1, 0.4, 0.7, 0.2, 0.3, 0.9], 4),
        "corroboration": _matrix([0.2, 0.3, 0.5, 0.1, 0.2, 0.6], 4),
        "duplication": _matrix([0.2, 0.1, 0.4, 0.3, 0.2, 0.5], 4),
    }
    kwargs = {
        "plugin_ids": ["answer_evidence_observer", "independence_residual"],
        "residualization_variants": {"all": ["answer_confidence_product", "lexical_duplication", "embedding_redundancy"]},
        "ridge": 1e-6,
    }
    first = run_diagnostic_pipeline(matrices, [0.2, 0.4, 0.6, 0.8], matrices["duplication"], **kwargs)
    second = run_diagnostic_pipeline(matrices, [0.2, 0.4, 0.6, 0.8], matrices["duplication"], **kwargs)
    np.testing.assert_array_equal(first["residuals"]["all"]["agreement"], second["residuals"]["all"]["agreement"])
    assert first["fits"] == second["fits"]
    with pytest.raises(ValueError, match="must be the first"):
        run_diagnostic_pipeline(
            matrices,
            [0.2, 0.4, 0.6, 0.8],
            matrices["duplication"],
            plugin_ids=["independence_residual", "answer_evidence_observer"],
            residualization_variants={"all": ["embedding_redundancy"]},
        )


def test_selected_pair_mean_uses_unique_upper_triangle_pairs():
    matrix = _matrix([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 4)
    assert selected_pair_mean(matrix, [0, 1, 2]) == pytest.approx((0.1 + 0.2 + 0.4) / 3)
    assert selected_pair_mean(matrix, [2]) == 0.0
    with pytest.raises(ValueError, match="unique"):
        selected_pair_mean(matrix, [0, 0])
