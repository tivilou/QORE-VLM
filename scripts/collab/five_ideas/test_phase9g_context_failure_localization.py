#!/usr/bin/env python3
"""Synthetic tests for Phase 9G context variants and report-only gate."""

from __future__ import annotations

import unittest

from phase9g_probe import (
    build_context_variants,
    build_gold_answer_copy_prompt,
    run_context_probe,
)
from phase9g_metrics import Phase9GError, summarize_context_failure


GATE = {
    "minimum_primary_errors": 2,
    "minimum_copy_control_successes": 2,
    "minimum_dominant_fraction": 0.40,
    "minimum_dominance_margin": 0.10,
    "bootstrap_repetitions": 100,
    "bootstrap_seed": 9701,
}


def arm(em=0, *, attempted=True):
    if not attempted:
        return {
            "attempted": False,
            "em": None,
            "f1": None,
            "generation_time_ms": None,
        }
    return {
        "attempted": True,
        "em": float(em),
        "f1": float(em),
        "generation_time_ms": 1.0,
    }


def row(index, category):
    values = {
        name: arm(0)
        for name in (
            "full_plain",
            "full_highlight",
            "answer_first",
            "retrieved_answer_oracle",
        )
    }
    copy = arm(0 if category == "copy_failure" else 1)
    if category == "full_passage_sufficient":
        values["full_plain"] = arm(1)
    elif category == "full_passage_localization":
        values["full_highlight"] = arm(1)
    elif category == "answer_passage_order":
        values["answer_first"] = arm(1)
    elif category == "multi_passage_evidence":
        values["retrieved_answer_oracle"] = arm(1)
    elif category not in {"beyond_top50_context", "copy_failure"}:
        raise ValueError(category)
    if copy["em"] != 1:
        values = {name: arm(attempted=False) for name in values}
    return {
        "question_id": f"nq_{index}",
        "retrieval_hit": True,
        "selected_hit": True,
        "selected_match_count": 1,
        "oracle_match_count": 2,
        "answer_first_changed": True,
        "selection_time_ms": 1.0,
        "arms": {
            "baseline": arm(0),
            "gold_answer_copy": copy,
            **values,
        },
    }


class Phase9GTests(unittest.TestCase):
    def test_baseline_forwards_existing_list_unchanged(self):
        class Generator:
            def __init__(self):
                self.calls = []

            def generate(self, question, passages):
                self.calls.append((question, passages))
                return "Paris"

        generator = Generator()
        passages = ["The answer is Paris."]
        result = run_context_probe(generator, "Where?", passages)
        self.assertEqual(result.prediction, "Paris")
        self.assertIs(generator.calls[0][1], passages)

    def test_variants_preserve_membership_and_order(self):
        selected = ["Noise.", "Paris is the capital.", "More noise."]
        retrieved = ["Noise.", "Paris is the capital.", "Paris is in France."]
        first = build_context_variants(selected, retrieved, ["Paris"])
        second = build_context_variants(selected, retrieved, ["Paris"])
        self.assertEqual(first, second)
        self.assertEqual(first.full_plain, (selected[1],))
        self.assertEqual(first.answer_first, (selected[1], selected[0], selected[2]))
        self.assertEqual(first.retrieved_answer_oracle, tuple(retrieved[1:]))
        self.assertIn("[[Paris]]", first.full_highlight[0])

    def test_copy_prompt_is_separate(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"] + messages[1]["content"]

        prompt = build_gold_answer_copy_prompt(Tokenizer(), "Where?", "Paris")
        self.assertIn("Candidate answer: Paris", prompt)

    def test_exclusive_attribution_and_report_only_gate(self):
        categories = [
            "full_passage_sufficient",
            "full_passage_localization",
            "answer_passage_order",
            "multi_passage_evidence",
            "beyond_top50_context",
        ]
        summary = summarize_context_failure(
            {"samples": [row(750 + i, name) for i, name in enumerate(categories)]},
            gate=GATE,
        )
        counts = summary["primary_attribution"]["counts"]
        self.assertEqual(sum(counts.values()), 5)
        self.assertEqual(counts["retrieved_oracle_nonincremental"], 0)
        self.assertTrue(all(counts[name] == 1 for name in categories))
        self.assertTrue(summary["decision"]["report_only"])
        self.assertEqual(
            summary["decision"]["primary_failure_class"],
            "mixed_context_failure_modes",
        )

    def test_copy_failure_is_excluded(self):
        summary = summarize_context_failure(
            {"samples": [row(800, "copy_failure")]},
            gate={**GATE, "minimum_primary_errors": 1},
        )
        self.assertEqual(summary["cohorts"]["copy_control_successes"], 0)
        self.assertEqual(
            summary["decision"]["primary_failure_class"],
            "insufficient_copy_control_successes",
        )

    def test_adaptive_condition_is_enforced(self):
        sample = row(801, "copy_failure")
        sample["arms"]["full_highlight"] = arm(0)
        with self.assertRaises(Phase9GError):
            summarize_context_failure({"samples": [sample]}, gate=GATE)

    def test_weakest_success_hierarchy_is_enforced(self):
        sample = row(802, "full_passage_sufficient")
        sample["arms"]["full_highlight"] = arm(1)
        sample["arms"]["answer_first"] = arm(1)
        summary = summarize_context_failure({"samples": [sample]}, gate={**GATE, "minimum_primary_errors": 1, "minimum_copy_control_successes": 1})
        self.assertEqual(summary["primary_attribution"]["counts"]["full_passage_sufficient"], 1)


if __name__ == "__main__":
    unittest.main()
