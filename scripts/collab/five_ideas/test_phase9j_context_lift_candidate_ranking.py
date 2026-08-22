#!/usr/bin/env python3
"""Synthetic contracts for the Phase 9J observation-only rankers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phase9j_metrics import Phase9JError, summarize_phase9j, validate_result
from phase9j_probe import (
    COMBINED_PROFILE,
    CONTEXT_LIFT_PROFILE,
    READER_SUPPORT_PROFILE,
    build_candidate_scores,
    choose_candidate,
    mean_logprob_difference,
    scores_by_permuted_order,
)


GATE = {
    "formal": {
        "bootstrap_repetitions": 100,
        "bootstrap_seed": 9911,
        "candidate_oracle_em_uplift_min": 0.05,
        "candidate_oracle_ci95_low_min": 0.0,
        "ranker_em_uplift_min": 0.03,
        "ranker_ci95_low_min": 0.0,
        "ranker_f1_loss_max": 0.01,
        "order_agreement_min": 0.90,
        "score_to_baseline_ratio_max": 1.30,
    }
}


def standard(em: float) -> dict[str, object]:
    return {"attempted": True, "em": em, "f1": em, "generation_time_ms": 10.0}


def candidate(em: float | None, status: str = "ok") -> dict[str, object]:
    scored = status in {"ok", "abstain"}
    return {"attempted": True, "parse_status": status, "em": em if scored else None, "f1": em if scored else None, "generation_time_ms": 10.0 if scored else None}


def ranker(choice: str | None, em: float | None = 1.0) -> dict[str, object]:
    valid = choice is not None
    return {"attempted": True, "original_choice_mode": choice, "permuted_choice_mode": choice if valid else None, "order_agreement": True if valid else None, "parse_status": "ok" if valid else "invalid_choice", "em": em if valid else None, "f1": em if valid else None, "score_time_ms": 5.0 if valid else None, "score_to_baseline_ratio": 0.5 if valid else None}


def sample(index: int, *, valid: bool = True) -> dict[str, object]:
    choice = "extractive_span_v1" if valid else None
    return {
        "question_id": f"nq_{index}",
        "retrieval_hit": True,
        "selected_hit": True,
        "primary_error": True,
        "copy_control_success": True,
        "eligible": True,
        "selection_time_ms": 1.0,
        "arms": {
            "baseline": standard(0.0),
            "gold_answer_copy": standard(1.0),
            "extractive_span": candidate(1.0),
            "evidence_constrained": candidate(0.0),
        },
        "candidate_set": {"attempted": True, "candidate_count": 3, "unique_candidate_count": 2, "parse_failures": 0, "oracle_em": 1.0, "oracle_f1": 1.0},
        "rankers": {
            CONTEXT_LIFT_PROFILE: ranker(choice),
            READER_SUPPORT_PROFILE: ranker(choice),
            COMBINED_PROFILE: ranker(choice),
        },
    }


def result(rows: list[dict[str, object]], stage: str = "formal") -> dict[str, object]:
    return {"schema_version": 1, "phase": "phase9j_context_lift_candidate_ranking", "stage": stage, "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {}, "samples": rows}


class Phase9JTests(unittest.TestCase):
    def test_context_lift_and_rankers_are_deterministic(self):
        self.assertAlmostEqual(mean_logprob_difference([-1.0, -2.0], [-2.0, -4.0]), 1.5)
        raw = {
            "baseline_v1": {"context_lift": 0.2, "reader_support": 0.1},
            "extractive_span_v1": {"context_lift": 0.8, "reader_support": 0.5},
            "evidence_constrained_v1": {"context_lift": 0.3, "reader_support": 0.9},
        }
        scores = build_candidate_scores(raw)
        self.assertEqual(choose_candidate(scores, CONTEXT_LIFT_PROFILE), "extractive_span_v1")
        self.assertEqual(choose_candidate(scores, READER_SUPPORT_PROFILE), "evidence_constrained_v1")
        self.assertEqual(choose_candidate(scores, COMBINED_PROFILE), "evidence_constrained_v1")

    def test_order_audit_uses_stable_mode_ids(self):
        candidates = [("baseline_v1", "A"), ("extractive_span_v1", "B"), ("evidence_constrained_v1", "C")]
        raw = {mode: {"context_lift": float(index), "reader_support": float(2 - index)} for index, (mode, _) in enumerate(candidates)}
        choices = scores_by_permuted_order(candidates, raw)
        for value in choices.values():
            self.assertEqual(value["original_choice_mode"], value["permuted_choice_mode"])

    def test_schema_rejects_raw_text(self):
        compact = result([sample(1)])
        self.assertEqual(validate_result(compact)[0]["question_id"], "nq_1")
        compact["samples"][0]["candidate_text"] = "secret"
        with self.assertRaises(Phase9JError):
            validate_result(compact)

    def test_formal_gate_can_pass_synthetic_data(self):
        summary = summarize_phase9j(result([sample(i) for i in range(12)]), gate=GATE)
        self.assertTrue(summary["decision"]["candidate_oracle_gate_pass"])
        self.assertTrue(summary["decision"]["ranker_gates"][COMBINED_PROFILE])
        self.assertEqual(summary["decision"]["primary_failure_class"], "combined_ranker_gate_passed_replication_only")

    def test_invalid_ranker_cannot_pass(self):
        summary = summarize_phase9j(result([sample(i, valid=False) for i in range(12)]), gate=GATE)
        self.assertFalse(summary["decision"]["ranker_gates"][COMBINED_PROFILE])


if __name__ == "__main__":
    unittest.main()
