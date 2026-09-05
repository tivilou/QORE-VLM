"""Synthetic contract tests for the canonical Q-DES no-GPU preflight."""

from __future__ import annotations

import json
import unittest

from applications.rag.qdes_preflight import PreflightError, analyze
from applications.rag.question_slot_parser import QuestionSlotParser


def _fixture(case_count: int = 50, top50_count: int = 50) -> dict:
    cases = []
    for case_number in range(1, case_count + 1):
        candidates = []
        for rank in range(1, top50_count + 1):
            positive = rank == 1
            text = (
                "William Shakespeare was the author who wrote Hamlet in England."
                if positive
                else f"Unrelated article {rank} discusses a different topic."
            )
            candidates.append({
                "id": f"p-{case_number}-{rank}",
                "text": text,
                "retrieved_rank": rank,
                "retrieval_score": float(100 - rank),
                # Deliberately put the generic scorer's high value on a
                # negative passage so typed and generic signals are testable.
                "answer_scorer_score": 0.1 if positive else (0.9 if rank == 2 else 0.01),
                "evidence": {"positive_consensus": positive},
            })
        cases.append({
            "case_number": case_number,
            "question_id": f"nq_{case_number}",
            "question": "who wrote Hamlet?",
            "top_50": candidates,
        })
    return {
        "schema_version": "fixture",
        "protocol": {"question_global_indices": list(range(1900, 1900 + case_count))},
        "cases": cases,
    }


class QDESPreflightTests(unittest.TestCase):
  def test_parser_does_not_invent_fallback(self) -> None:
    parser = QuestionSlotParser()
    parsed = parser.parse("who wrote Hamlet?")
    self.assertTrue(parsed.success)
    self.assertEqual(parsed.slots["answer_type"], "person")
    self.assertEqual(parsed.slots["operator"], "who")
    failed = parser.parse("Hamlet author")
    self.assertFalse(failed.success)
    self.assertEqual(failed.slots, {})


  def test_fixed_schema_and_typed_signal_are_computed(self) -> None:
    result = analyze(_fixture())
    self.assertEqual(result["metrics"]["case_count"], 50)
    self.assertEqual(result["metrics"]["top50_count_per_case"], 50)
    self.assertEqual(result["metrics"]["parser_success_rate"], 1.0)
    self.assertIsNotNone(result["metrics"]["auc_micro"]["typed_coverage"])
    self.assertIsNotNone(result["metrics"]["typed_gain_vs_answer_scorer"])
    self.assertGreater(result["metrics"]["typed_gain_vs_answer_scorer"], 0.02)
    self.assertTrue(result["gates"]["schema_gate"]["pass"])
    self.assertFalse(result["gates"]["leakage_gate"]["selection_feedback"])


  def test_old_questions_schema_is_rejected(self) -> None:
    with self.assertRaisesRegex(PreflightError, "root field 'cases'"):
      analyze({"questions": _fixture()["cases"]})


  def test_fixed_50_accounting_is_enforced(self) -> None:
    with self.assertRaisesRegex(PreflightError, "exactly 50 cases"):
      analyze(_fixture(case_count=49))
    with self.assertRaisesRegex(PreflightError, "exactly 50 candidates"):
      analyze(_fixture(top50_count=49))


  def test_passage_label_coverage_is_not_per_case_only(self) -> None:
    fixture = _fixture()
    del fixture["cases"][0]["top_50"][0]["evidence"]
    result = analyze(fixture)
    self.assertEqual(result["metrics"]["passage_label_coverage_candidate_rate"], 2499 / 2500)
    self.assertFalse(result["gates"]["label_availability_gate"]["pass"])


  def test_replay_is_deterministic_and_output_is_compact(self) -> None:
    first = analyze(_fixture())
    second = analyze(_fixture())
    self.assertEqual(first["replay_digest"], second["replay_digest"])
    serialized = json.dumps(first, sort_keys=True)
    self.assertNotIn("who wrote Hamlet", serialized)
    self.assertNotIn("William Shakespeare", serialized)
    self.assertIn("gold_answers", serialized)  # field name is an audit entry
    self.assertIn("selection_feedback", serialized)


if __name__ == "__main__":
  unittest.main()
