#!/usr/bin/env python3
"""Synthetic tests for the Phase 9B retrieval-ceiling diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collab.five_ideas.retrieval_ceiling_metrics import (  # noqa: E402
    RetrievalCeilingError,
    paired_bootstrap,
    summarize_retrieval_ceiling,
)


QUESTION_IDS = [f"q{index}" for index in range(6)]


def _payload(*, hits: list[bool], selected: list[bool], generation: bool = False):
    samples = []
    for index, question_id in enumerate(QUESTION_IDS):
        row = {
            "question_id": question_id,
            "answer_hit_at_retrieved": hits[index],
            "recall": 1.0 if selected[index] else 0.0,
        }
        if generation:
            row.update({
                "f1": 0.8 if selected[index] else 0.1,
                "em": 1.0 if selected[index] else 0.0,
                "generation_time_ms": 10.0 + index,
            })
        samples.append(row)
    return {"samples": samples}


class RetrievalCeilingMetricsTests(unittest.TestCase):
    def test_selection_bottleneck_fixture(self):
        hits = [True, True, True, True, True, False]
        qore = [True, False, True, False, True, False]
        topk = [True, True, True, True, True, False]
        results = {
            "retrieval_top50_as": _payload(hits=hits, selected=hits),
            "qore_as_select": _payload(hits=hits, selected=qore),
            "topk_as_select": _payload(hits=hits, selected=topk),
            "qore_as_generate": _payload(hits=hits, selected=qore, generation=True),
        }
        summary = summarize_retrieval_ceiling(
            results, gate={"bootstrap_repetitions": 100, "bootstrap_seed": 7}
        )
        self.assertEqual(summary["decision"]["primary_bottleneck"], "selection")
        self.assertAlmostEqual(summary["retrieval"]["top50_answer_failure_rate"], 1 / 6)
        self.assertAlmostEqual(summary["selection"]["qore_as_k5"]["selected_hit_rate_conditional"], 3 / 5)
        self.assertAlmostEqual(summary["selection"]["qore_as_k5"]["topk_recovery_delta"]["mean"], 2 / 5)
        self.assertEqual(summary["generation"]["n_when_selected_hit"], 3)
        self.assertEqual(summary["generation"]["n_when_selected_miss"], 2)

    def test_retrieval_ceiling_takes_priority(self):
        hits = [True, True, True, False, False, False]
        results = {
            "retrieval_top50_as": _payload(hits=hits, selected=hits),
            "qore_as_select": _payload(hits=hits, selected=hits),
            "topk_as_select": _payload(hits=hits, selected=hits),
            "qore_as_generate": _payload(hits=hits, selected=hits, generation=True),
        }
        summary = summarize_retrieval_ceiling(
            results, gate={"bootstrap_repetitions": 100, "bootstrap_seed": 7}
        )
        self.assertEqual(summary["decision"]["primary_bottleneck"], "retrieval_ceiling")

    def test_join_rejects_retrieval_mismatch(self):
        hits = [True, True, True, True, True, False]
        results = {
            "retrieval_top50_as": _payload(hits=hits, selected=hits),
            "qore_as_select": _payload(hits=hits, selected=hits),
            "topk_as_select": _payload(hits=hits, selected=hits),
            "qore_as_generate": _payload(hits=[*hits[:-1], True], selected=hits, generation=True),
        }
        with self.assertRaises(RetrievalCeilingError):
            summarize_retrieval_ceiling(
                results, gate={"bootstrap_repetitions": 100, "bootstrap_seed": 7}
            )

    def test_bootstrap_is_deterministic(self):
        first = paired_bootstrap([0.0, 1.0, 0.0, 1.0], repetitions=100, seed=11)
        second = paired_bootstrap([0.0, 1.0, 0.0, 1.0], repetitions=100, seed=11)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
