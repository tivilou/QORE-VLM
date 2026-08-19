#!/usr/bin/env python3
"""Synthetic tests for Phase 9D rank-depth replication metrics and invariants."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collab.five_ideas.retrieval_rank_depth_metrics import (  # noqa: E402
    RankDepthError,
    summarize_rank_depth,
)


def _payload(ranks: list[int | None]) -> dict:
    return {
        "samples": [
            {
                "question_id": f"q{index}",
                "first_answer_rank": rank,
                "answer_hit_at_50": rank is not None and rank <= 50,
                "answer_hit_at_100": rank is not None and rank <= 100,
                "answer_hit_at_200": rank is not None and rank <= 200,
            }
            for index, rank in enumerate(ranks)
        ]
    }


class RankDepthMetricsTests(unittest.TestCase):
    GATE = {
        "maximum_top50_failure_rate": 0.20,
        "minimum_top200_gain_over_top50": 0.10,
        "maximum_top200_failure_rate": 0.20,
    }

    def test_cutoff_limited_fixture(self):
        summary = summarize_rank_depth(
            _payload([12, 45, 51, 72, 111, None]),
            gate=self.GATE,
        )
        self.assertEqual(summary["decision"]["primary_bottleneck"], "cutoff_limited")
        self.assertAlmostEqual(summary["cutoffs"]["50"]["hit_rate"], 2 / 6)
        self.assertAlmostEqual(summary["cutoffs"]["200"]["hit_rate"], 5 / 6)
        self.assertEqual(summary["first_answer_rank"]["bins"]["rank_51_100"], 2)
        self.assertEqual(summary["first_answer_rank"]["bins"]["not_in_200"], 1)

    def test_retriever_limited_fixture(self):
        summary = summarize_rank_depth(
            _payload([12, None, None, None, None, None]),
            gate=self.GATE,
        )
        self.assertEqual(summary["decision"]["primary_bottleneck"], "retriever_limited")
        self.assertEqual(summary["cutoffs"]["200"]["n_misses"], 5)

    def test_hit_flags_must_match_first_rank(self):
        payload = _payload([51])
        payload["samples"][0]["answer_hit_at_50"] = True
        with self.assertRaises(RankDepthError):
            summarize_rank_depth(payload, gate={})

    def test_forbidden_raw_fields_are_rejected(self):
        payload = _payload([12])
        payload["samples"][0]["passages"] = ["raw"]
        with self.assertRaises(RankDepthError):
            summarize_rank_depth(payload, gate={})

    def test_summary_replay_is_deterministic(self):
        payload = _payload([1, 50, 100, 200, None])
        first = summarize_rank_depth(payload, gate={})
        second = summarize_rank_depth(payload, gate={})
        self.assertEqual(first, second)

    def test_two_disjoint_slices_can_be_combined(self):
        first = _payload([1, 51, 101, 200, None])
        second = _payload([2, 52, 102, 199, None])
        first["samples"] = [dict(row, question_id=f"nq_{250 + i}") for i, row in enumerate(first["samples"])]
        second["samples"] = [dict(row, question_id=f"nq_{450 + i}") for i, row in enumerate(second["samples"])]
        combined = summarize_rank_depth(
            {"samples": first["samples"] + second["samples"]}, gate=self.GATE
        )
        self.assertEqual(combined["n_questions"], 10)
        self.assertEqual(combined["first_answer_rank"]["bins"]["not_in_200"], 2)
        self.assertAlmostEqual(combined["cutoffs"]["200"]["hit_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()
