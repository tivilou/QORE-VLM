"""Synthetic tests for Phase 7B calibration aggregation."""

from __future__ import annotations

import pytest

from scripts.collab.five_ideas.phase7b_metrics import (
    pearson_correlation,
    spearman_correlation,
    summarize_calibration_rows,
    validate_compact_rows,
)


def _row(question, conflict, f1, *, seed=42, k_value=3):
    return {
        "question_id": question,
        "selection_variant": "qore_answer",
        "K": k_value,
        "generator": "reference",
        "seed": seed,
        "selected_count": k_value,
        "selected_agreement": 1.0 - conflict,
        "selected_conflict": conflict,
        "selected_corroboration": 0.5 * (1.0 - conflict),
        "selected_duplication": 0.1,
        "em": float(f1 == 1.0),
        "f1": f1,
    }


def test_correlations_handle_order_ties_and_constant_inputs():
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 2, 4], [1, 3, 3, 8]) == pytest.approx(1.0)
    assert pearson_correlation([1, 1, 1], [0, 1, 0]) is None


def test_summary_groups_by_variant_k_generator_and_seed():
    rows = [
        _row("q1", 0.1, 1.0),
        _row("q2", 0.5, 0.5),
        _row("q3", 0.9, 0.0),
        _row("q1", 0.2, 1.0, seed=43),
        _row("q2", 0.4, 0.5, seed=43),
        _row("q3", 0.8, 0.0, seed=43),
    ]
    summary = summarize_calibration_rows(rows)
    assert summary["row_count"] == 6
    assert summary["group_count"] == 2
    first = summary["groups"][0]
    assert first["n_questions"] == 3
    assert first["feature_quality_correlation"]["selected_conflict"]["pearson_f1"] < 0


def test_summary_rejects_selection_cardinality_drift():
    row = _row("q1", 0.2, 1.0)
    row["selected_count"] = 2
    with pytest.raises(ValueError, match="selected_count"):
        summarize_calibration_rows([row])


def test_compact_rows_reject_raw_fields():
    row = _row("q1", 0.2, 1.0)
    row["prediction"] = "raw answer"
    with pytest.raises(ValueError, match="forbidden raw fields"):
        validate_compact_rows([row])
