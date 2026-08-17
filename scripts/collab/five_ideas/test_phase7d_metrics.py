"""Tests for Phase 7D compact aggregation and kill criteria."""

from __future__ import annotations

import pytest

from scripts.collab.five_ideas.phase7d_metrics import summarize_independence_rows


VARIANTS = ["confidence_only", "lexical_only", "embedding_only", "all_nuisances"]
GATE = {
    "residual_variant": "all_nuisances",
    "primary_selector": "qore_answer",
    "consistency_selectors": ["topk_answer", "qore_dpr"],
    "minimum_residual_spearman_f1": 0.10,
    "minimum_improvement_over_raw": 0.05,
    "minimum_positive_selectors": 2,
    "minimum_bootstrap_ci_low": -0.05,
}


def _row(question_index: int, selector: str, *, helpful_residual: bool) -> dict[str, float | int | str | list[int] | bool]:
    quality = question_index / 49.0
    raw = 0.5 - 0.3 * quality if helpful_residual else 0.2 + 0.4 * quality
    residual = 0.1 + 0.5 * quality
    row: dict[str, float | int | str | list[int] | bool] = {
        "question_id": f"q{question_index:03d}",
        "generator": "reference",
        "seed": 42,
        "K": 5,
        "selection_variant": selector,
        "selected_count": 5,
        "selected_ranks": [0, 1, 2, 3, 4],
        "prediction_id": f"p{question_index:03d}",
        "generation_cache_hit": False,
        "em": quality,
        "f1": quality,
        "selected_agreement": raw,
        "selected_corroboration": raw,
        "selected_answer_confidence_product": 0.2 + 0.1 * quality,
        "selected_lexical_duplication": 0.3,
        "selected_embedding_redundancy": 0.4,
    }
    for variant in VARIANTS:
        row[f"selected_residual_{variant}_agreement"] = residual
        row[f"selected_residual_{variant}_corroboration"] = residual
    return row


def test_gate_passes_only_when_residual_association_is_stronger_and_consistent():
    rows = [
        _row(index, selector, helpful_residual=True)
        for selector in ["qore_answer", "topk_answer", "qore_dpr"]
        for index in range(50)
    ]
    summary = summarize_independence_rows(
        rows,
        residualization_variants=VARIANTS,
        gate=GATE,
        bootstrap_samples=500,
        bootstrap_seed=4207,
    )
    assert summary["gate"]["status"] == "pass"
    assert summary["gate"]["decision"] == "controlled_followup_eligible"
    assert summary["gate"]["targets"]["agreement"]["passes"]


def test_gate_fails_when_residual_does_not_improve_raw_association():
    rows = [
        _row(index, selector, helpful_residual=False)
        for selector in ["qore_answer", "topk_answer", "qore_dpr"]
        for index in range(50)
    ]
    summary = summarize_independence_rows(
        rows,
        residualization_variants=VARIANTS,
        gate=GATE,
        bootstrap_samples=500,
        bootstrap_seed=4207,
    )
    assert summary["gate"]["status"] == "fail"
    assert summary["gate"]["decision"] == "stop_answer_identity_objective"


def test_compact_rows_reject_raw_questions():
    row = _row(0, "qore_answer", helpful_residual=True)
    row["question"] = "raw question"
    with pytest.raises(ValueError, match="forbidden raw fields"):
        summarize_independence_rows(
            [row],
            residualization_variants=VARIANTS,
            gate=GATE,
            bootstrap_samples=100,
            bootstrap_seed=4207,
        )
