"""Synthetic aggregation tests for the Phase 7C ablation."""

from __future__ import annotations

import pytest

from scripts.collab.five_ideas.phase7c_metrics import summarize_ablation_rows


def _row(question: str, f1: float, *, variant: str = "qore_corroboration"):
    return {
        "question_id": question,
        "selection_variant": variant,
        "K": 5,
        "generator": "reference",
        "seed": 42,
        "selected_count": 5,
        "selected_ranks": [0, 1, 2, 3, 4],
        "prediction_id": "hashed",
        "generation_cache_hit": False,
        "selected_agreement": 0.8,
        "selected_conflict": 0.1,
        "selected_decisive_conflict": 0.0,
        "selected_corroboration": 0.7,
        "selected_duplication": 0.1,
        "em": float(f1 == 1.0),
        "f1": f1,
    }


def test_summary_includes_decisive_conflict_feature():
    summary = summarize_ablation_rows([_row("q1", 1.0), _row("q2", 0.0)])
    assert summary["features"][-1] == "selected_duplication"
    assert "selected_decisive_conflict" in summary["groups"][0]["feature_means"]


def test_summary_rejects_missing_decisive_conflict():
    row = _row("q1", 1.0)
    del row["selected_decisive_conflict"]
    with pytest.raises(ValueError, match="missing ablation features"):
        summarize_ablation_rows([row])
