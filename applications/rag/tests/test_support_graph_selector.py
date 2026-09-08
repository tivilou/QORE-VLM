"""Contract tests for the anchor-conditioned support graph selector."""

from __future__ import annotations

import json
import unittest

from applications.rag.support_graph_preflight import analyze
from applications.rag.support_graph_selector import SelectionInputError, select
from applications.rag.tests.test_qdes_preflight import _fixture


def _online_fixture() -> list[dict[str, object]]:
    rows = []
    for rank in range(1, 11):
        if rank == 1:
            text = "William Shakespeare wrote Hamlet in 1600."
        elif rank == 2:
            text = "An unrelated article discusses a different topic."
        elif rank == 3:
            text = "Hamlet was written by Shakespeare and staged in London."
        elif rank == 4:
            text = "The play Hamlet is associated with the year 1600."
        else:
            text = f"Unrelated article {rank} discusses a different topic."
        rows.append({
            "id": f"p-{rank}",
            "text": text,
            "retrieved_rank": rank,
            "retrieval_score": float(100 - rank),
            "answer_scorer_score": 1.0 if rank == 1 else (0.95 if rank == 2 else 0.1),
        })
    return rows


class SupportGraphSelectorTests(unittest.TestCase):
    def test_selection_is_deterministic_and_keeps_two_anchors(self) -> None:
        first = select("who wrote Hamlet in 1600?", _online_fixture())
        second = select("who wrote Hamlet in 1600?", _online_fixture())
        self.assertEqual(first, second)
        self.assertEqual(first.anchor_indices, (0, 1))
        self.assertEqual(len(first.selected_indices), 5)
        self.assertEqual(len(set(first.selected_indices)), 5)
        self.assertEqual(len(first.support_indices), 3)

    def test_forbidden_diagnostic_fields_are_rejected(self) -> None:
        rows = _online_fixture()
        rows[0]["evidence"] = {"positive_consensus": True}
        with self.assertRaisesRegex(SelectionInputError, "forbidden selection fields"):
            select("who wrote Hamlet?", rows)

    def test_degenerate_question_signal_falls_back_to_topk(self) -> None:
        result = select("the and of", _online_fixture())
        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, "empty_question_signal")
        self.assertEqual(result.selected_indices, (0, 1, 2, 3, 4))

    def test_support_disabled_control_matches_answer_scorer_topk(self) -> None:
        rows = _online_fixture()
        disabled = select(
            "who wrote Hamlet in 1600?",
            rows,
            support_passage_weight=0.0,
            support_question_weight=0.0,
        )
        expected = tuple(sorted(
            range(len(rows)),
            key=lambda index: (
                -float(rows[index]["answer_scorer_score"]),
                int(rows[index]["retrieved_rank"]),
                str(rows[index]["id"]),
            ),
        )[:5])
        self.assertEqual(disabled.selected_indices, expected)
        self.assertEqual(len(disabled.support_trace), 3)
        self.assertTrue(all(item["support_score"] == 0.0 for item in disabled.support_trace))

    def test_preflight_keeps_labels_out_of_selection_and_compact(self) -> None:
        result = analyze(_fixture())
        serialized = json.dumps(result, sort_keys=True)
        self.assertEqual(result["metrics"]["case_count"], 50)
        self.assertEqual(result["metrics"]["top50_count_per_case"], 50)
        self.assertFalse(result["gates"]["leakage_gate"]["selection_feedback"])
        self.assertNotIn("who wrote Hamlet", serialized)
        self.assertNotIn("William Shakespeare", serialized)
        self.assertIn("top_50[].evidence", result["offline_evaluation_fields"])


if __name__ == "__main__":
    unittest.main()
