#!/usr/bin/env python3
"""Synthetic contract tests for the Phase 9K gold-free applicability gate."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phase9k_gold_free_applicability import decide_applicability
from phase9k_gold_free_metrics import Phase9KError, summarize_phase9k, validate_result


def decision(candidates, raw_scores, exact, *, baseline_exact_span=False):
    return decide_applicability(candidates=candidates, raw_scores=raw_scores, exact_span_modes=exact, reader_margin_min=0.0, baseline_exact_span=baseline_exact_span)


def arm(em: float) -> dict[str, object]:
    return {"attempted": True, "em": em, "f1": em, "generation_time_ms": 10.0}


def ranker(*, attempted: bool, em: float = 1.0, choice: str | None = "extractive_span_v1") -> dict[str, object]:
    if not attempted:
        return {"attempted": False, "choice_mode": None, "permuted_choice_mode": None, "order_agreement": None, "parse_status": "not_attempted", "em": None, "f1": None, "generation_time_ms": None, "score_time_ms": None, "score_to_baseline_ratio": None}
    return {"attempted": True, "choice_mode": choice, "permuted_choice_mode": choice, "order_agreement": True, "parse_status": "ok", "em": em, "f1": em, "generation_time_ms": 10.0, "score_time_ms": 2.0, "score_to_baseline_ratio": 0.2}


def sample(index: int, *, applied: bool, baseline_em: float = 0.0, ranker_em: float = 1.0) -> dict[str, object]:
    return {
        "question_id": f"nq_{index}",
        "retrieval_hit": True,
        "selected_hit": True,
        "selection_time_ms": 1.0,
        "candidate_generation_time_ms": 10.0,
        "candidate_count": 3,
        "unique_candidate_count": 3,
        "parse_failures": 0,
        "applicability": {"apply": applied, "reason_code": "apply" if applied else "baseline_selected", "chosen_mode": "extractive_span_v1" if applied else None, "baseline_score": 1.0 if applied else None, "chosen_score": 2.0 if applied else None, "reader_margin": 1.0 if applied else None, "chosen_exact_span": True if applied else None, "baseline_exact_span": False, "candidate_consensus": applied},
        "baseline": arm(baseline_em),
        "ranker": ranker(attempted=applied, em=ranker_em),
    }


def result(rows: list[dict[str, object]], stage: str = "formal") -> dict[str, object]:
    return {"schema_version": 1, "phase": "phase9k_gold_free_applicability", "stage": stage, "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"gold_used_for_decision": False}, "samples": rows}


class Phase9KTests(unittest.TestCase):
    def test_supported_nonbaseline_candidate_is_applied(self):
        candidates = [("baseline_v1", "wrong"), ("extractive_span_v1", "correct"), ("evidence_constrained_v1", "correct")]
        raw = {"baseline_v1": {"context_lift": 0.0, "reader_support": 1.0}, "extractive_span_v1": {"context_lift": 0.0, "reader_support": 3.0}, "evidence_constrained_v1": {"context_lift": 0.0, "reader_support": 2.0}}
        decision_result, _ = decision(candidates, raw, {"baseline_v1": False, "extractive_span_v1": True, "evidence_constrained_v1": True})
        self.assertTrue(decision_result.apply)
        self.assertEqual(decision_result.reason_code, "apply")

    def test_baseline_exact_span_is_not_overridden(self):
        candidates = [("baseline_v1", "supported"), ("extractive_span_v1", "correct"), ("evidence_constrained_v1", "correct")]
        raw = {"baseline_v1": {"context_lift": 0.0, "reader_support": 1.0}, "extractive_span_v1": {"context_lift": 0.0, "reader_support": 3.0}, "evidence_constrained_v1": {"context_lift": 0.0, "reader_support": 2.0}}
        decision_result, _ = decision(candidates, raw, {"baseline_v1": True, "extractive_span_v1": True, "evidence_constrained_v1": True}, baseline_exact_span=True)
        self.assertFalse(decision_result.apply)
        self.assertEqual(decision_result.reason_code, "baseline_supported")

    def test_duplicate_candidates_are_rejected(self):
        candidates = [("baseline_v1", "same"), ("extractive_span_v1", "same")]
        raw = {mode: {"context_lift": 0.0, "reader_support": float(index)} for index, (mode, _) in enumerate(candidates)}
        decision_result, _ = decision(candidates, raw, {mode: True for mode, _ in candidates})
        self.assertFalse(decision_result.apply)
        self.assertEqual(decision_result.reason_code, "duplicate_candidates")

    def test_nonexact_top_candidate_is_rejected(self):
        candidates = [("baseline_v1", "wrong"), ("extractive_span_v1", "unsupported"), ("evidence_constrained_v1", "other")]
        raw = {"baseline_v1": {"context_lift": 0.0, "reader_support": 1.0}, "extractive_span_v1": {"context_lift": 0.0, "reader_support": 3.0}, "evidence_constrained_v1": {"context_lift": 0.0, "reader_support": 2.0}}
        decision_result, _ = decision(candidates, raw, {"baseline_v1": False, "extractive_span_v1": False, "evidence_constrained_v1": True})
        self.assertFalse(decision_result.apply)
        self.assertEqual(decision_result.reason_code, "no_candidate_consensus")

    def test_summary_uses_baseline_for_nonapplied_rows(self):
        summary = summarize_phase9k(result([sample(1, applied=True), sample(2, applied=False)]), gate={"formal": {"bootstrap_repetitions": 100}})
        self.assertEqual(summary["cohorts"]["applied"], 1)
        self.assertAlmostEqual(summary["paired_deltas"]["all_question_deployed_em"]["mean"], 0.5)

    def test_schema_rejects_raw_candidate_text(self):
        compact = result([sample(1, applied=True)])
        validate_result(compact)
        compact["samples"][0]["candidate_text"] = "secret"
        with self.assertRaises(Phase9KError):
            validate_result(compact)


if __name__ == "__main__":
    unittest.main()
