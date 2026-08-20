#!/usr/bin/env python3
"""Synthetic tests for Phase 9F evidence variants and report-only gate."""

from __future__ import annotations

import unittest

from evidence_sufficiency_probe import (
    build_evidence_variants,
    build_gold_answer_copy_prompt,
    run_context_probe,
)
from phase9f_metrics import Phase9FError, summarize_evidence_sufficiency


GATE = {
    "minimum_primary_errors": 2,
    "minimum_copy_control_successes": 2,
    "minimum_dominant_fraction": 0.40,
    "minimum_dominance_margin": 0.10,
    "bootstrap_repetitions": 100,
    "bootstrap_seed": 9601,
}


def arm(em, f1=None, attempted=True):
    if not attempted:
        return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}
    return {
        "attempted": True,
        "em": float(em),
        "f1": float(em if f1 is None else f1),
        "generation_time_ms": 1.0,
    }


def row(index, category):
    selected = True
    baseline = arm(0.0)
    copy = arm(1.0)
    sentence_plain = arm(0.0)
    sentence_highlight = arm(0.0)
    window_plain = arm(0.0)
    window_highlight = arm(0.0)
    if category == "exact_sentence_sufficient":
        sentence_plain = arm(1.0)
    elif category == "answer_localization_only":
        sentence_highlight = arm(1.0)
    elif category == "local_context_only":
        window_plain = arm(1.0)
    elif category == "both_main_effects":
        sentence_highlight = arm(1.0)
        window_plain = arm(1.0)
    elif category == "interaction_only":
        window_highlight = arm(1.0)
    elif category == "beyond_local_context":
        pass
    elif category == "copy_failure":
        copy = arm(0.0)
    else:
        raise ValueError(category)
    eligible = copy["em"] == 1.0
    return {
        "question_id": f"nq_{index}",
        "retrieval_hit": True,
        "selected_hit": selected,
        "evidence_match_found": True,
        "evidence_match_count": 1,
        "sentence_variant_count": 1,
        "window_variant_count": 1,
        "selection_time_ms": 1.0,
        "arms": {
            "baseline": baseline,
            "gold_answer_copy": copy,
            "sentence_plain": sentence_plain if eligible else arm(0.0, attempted=False),
            "sentence_highlight": sentence_highlight if eligible else arm(0.0, attempted=False),
            "window_plain": window_plain if eligible else arm(0.0, attempted=False),
            "window_highlight": window_highlight if eligible else arm(0.0, attempted=False),
        },
    }


class Phase9FTests(unittest.TestCase):
    def test_baseline_context_forwards_existing_list_unchanged(self):
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
        self.assertEqual(generator.calls, [("Where?", passages)])
        self.assertIs(generator.calls[0][1], passages)

    def test_variants_are_deterministic_and_local(self):
        passages = ["Before. The answer is Paris. After."]
        first = build_evidence_variants(passages, ["Paris"])
        second = build_evidence_variants(passages, ["Paris"])
        self.assertEqual(first, second)
        self.assertEqual(first.match_count, 1)
        self.assertEqual(first.sentence_plain, ("The answer is Paris.",))
        self.assertIn("[[Paris]]", first.sentence_highlight[0])
        self.assertEqual(first.window_plain, ("Before. The answer is Paris. After.",))

    def test_copy_prompt_is_separate(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return messages[0]["content"] + messages[1]["content"]
        prompt = build_gold_answer_copy_prompt(Tokenizer(), "Where?", "Paris")
        self.assertIn("Candidate answer: Paris", prompt)
        self.assertIn("Copy", prompt)

    def test_2x2_attribution_and_report_only_gate(self):
        categories = [
            "exact_sentence_sufficient", "answer_localization_only",
            "local_context_only", "both_main_effects", "interaction_only",
            "beyond_local_context",
        ]
        summary = summarize_evidence_sufficiency(
            {"samples": [row(700 + i, category) for i, category in enumerate(categories)]},
            gate={**GATE, "minimum_primary_errors": 2, "minimum_copy_control_successes": 2},
        )
        counts = summary["primary_attribution"]["counts"]
        self.assertEqual(sum(counts.values()), 6)
        self.assertEqual(counts["both_main_effects"], 1)
        self.assertTrue(summary["decision"]["report_only"])
        self.assertEqual(summary["decision"]["primary_failure_class"], "mixed_evidence_failure_modes")

    def test_copy_failure_excluded_and_evidence_arms_unattempted(self):
        sample = row(800, "copy_failure")
        summary = summarize_evidence_sufficiency(
            {"samples": [sample]},
            gate={**GATE, "minimum_primary_errors": 1},
        )
        self.assertEqual(summary["cohorts"]["selected_hit_baseline_errors"], 1)
        self.assertEqual(summary["cohorts"]["copy_control_successes"], 0)
        self.assertEqual(summary["decision"]["primary_failure_class"], "insufficient_copy_control_successes")

    def test_adaptive_evidence_condition_is_enforced(self):
        sample = row(801, "copy_failure")
        sample["arms"]["sentence_plain"] = arm(0.0)
        with self.assertRaises(Phase9FError):
            summarize_evidence_sufficiency({"samples": [sample]}, gate=GATE)


if __name__ == "__main__":
    unittest.main()
