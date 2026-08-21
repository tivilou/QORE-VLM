#!/usr/bin/env python3
"""Synthetic tests for Phase 9H answerability diagnostics."""

from __future__ import annotations

import unittest

from phase9h_probe import (
    build_extractive_prompt,
    build_gold_answer_copy_prompt,
    build_support_judge_prompt,
    parse_support_label,
    run_context_probe,
)
from phase9h_metrics import Phase9HError, summarize_answerability_generation


GATE = {
    "minimum_primary_errors": 2,
    "minimum_copy_control_successes": 2,
    "minimum_dominant_fraction": 0.40,
    "minimum_dominance_margin": 0.10,
    "bootstrap_repetitions": 100,
    "bootstrap_seed": 9801,
}


def arm(em=0, f1=None, *, attempted=True):
    if not attempted:
        return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}
    return {
        "attempted": True,
        "em": float(em),
        "f1": float(em if f1 is None else f1),
        "generation_time_ms": 1.0,
    }


def support(label=None, *, attempted=True):
    if not attempted:
        return {"attempted": False, "label": None, "generation_time_ms": None}
    return {"attempted": True, "label": label, "generation_time_ms": 1.0}


def row(index, category):
    baseline_f1 = 0.3 if category == "answer_format_mismatch" else 0.0
    extractive_em = 1 if category == "answer_extraction_contract" else 0
    support_label = {
        "model_judge_unsupported": "unsupported",
        "semantic_generation_failure": "supported",
        "conflicting_signals": "unsupported",
    }.get(category, "uncertain")
    if category == "conflicting_signals":
        extractive_em = 1
    if category not in {
        "model_judge_unsupported", "answer_extraction_contract",
        "answer_format_mismatch", "semantic_generation_failure",
        "unresolved_residual", "conflicting_signals",
    }:
        raise ValueError(category)
    return {
        "question_id": f"nq_{index}",
        "retrieval_hit": True,
        "selected_hit": True,
        "selected_match_count": 1,
        "retrieval_match_count": 2,
        "selection_time_ms": 1.0,
        "arms": {
            "baseline": arm(0, baseline_f1),
            "extractive": arm(extractive_em),
            "gold_answer_copy": arm(1),
            "support_judge": support(support_label),
        },
    }


class Phase9HTests(unittest.TestCase):
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

    def test_support_parser_is_strict_and_ambiguous(self):
        self.assertEqual(parse_support_label("SUPPORTED"), "supported")
        self.assertEqual(parse_support_label("UNSUPPORTED"), "unsupported")
        self.assertEqual(parse_support_label("UNCERTAIN"), "uncertain")
        self.assertEqual(parse_support_label("SUPPORTED, but maybe UNSUPPORTED"), "uncertain")
        self.assertEqual(parse_support_label("because the passage says so"), "uncertain")

    def test_prompts_keep_gold_and_support_roles_separate(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"] + messages[1]["content"]

        tokenizer = Tokenizer()
        extractive = build_extractive_prompt(tokenizer, "Where?", ["Paris."])
        support_prompt = build_support_judge_prompt(tokenizer, "Where?", ["Paris."], "Paris")
        copy_prompt = build_gold_answer_copy_prompt(tokenizer, "Where?", "Paris")
        self.assertIn("Extract the exact answer span", extractive)
        self.assertIn("SUPPORTED, UNSUPPORTED, or UNCERTAIN", support_prompt)
        self.assertIn("Candidate answer: Paris", copy_prompt)

    def test_hierarchical_attribution_and_report_only_gate(self):
        categories = [
            "model_judge_unsupported", "answer_extraction_contract",
            "answer_format_mismatch", "semantic_generation_failure",
            "unresolved_residual", "conflicting_signals",
        ]
        summary = summarize_answerability_generation(
            {"samples": [row(800 + i, category) for i, category in enumerate(categories)]},
            gate=GATE,
        )
        counts = summary["primary_attribution"]["counts"]
        self.assertEqual(sum(counts.values()), len(categories))
        self.assertTrue(all(counts[name] == 1 for name in categories))
        self.assertTrue(summary["decision"]["report_only"])
        self.assertTrue(summary["decision"]["support_judge_is_model_signal"])
        self.assertEqual(summary["decision"]["primary_failure_class"], "mixed_answerability_generation_failures")

    def test_copy_failure_excludes_diagnostic_arms(self):
        sample = row(900, "model_judge_unsupported")
        sample["arms"]["gold_answer_copy"] = arm(0)
        for name in ("extractive",):
            sample["arms"][name] = arm(attempted=False)
        sample["arms"]["support_judge"] = support(attempted=False)
        summary = summarize_answerability_generation(
            {"samples": [sample]}, gate={**GATE, "minimum_primary_errors": 1}
        )
        self.assertEqual(summary["cohorts"]["copy_control_successes"], 0)
        self.assertEqual(summary["decision"]["primary_failure_class"], "insufficient_copy_control_successes")

    def test_adaptive_condition_is_enforced(self):
        sample = row(901, "semantic_generation_failure")
        sample["arms"]["extractive"] = arm(attempted=False)
        with self.assertRaises(Phase9HError):
            summarize_answerability_generation({"samples": [sample]}, gate=GATE)


if __name__ == "__main__":
    unittest.main()
