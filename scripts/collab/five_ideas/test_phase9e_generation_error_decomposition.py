#!/usr/bin/env python3
"""Synthetic tests for Phase 9E probes, schema, and attribution gate."""

from __future__ import annotations

import unittest

from generation_error_probe import (
    build_extractive_prompt,
    build_gold_answer_copy_prompt,
    extract_gold_matched_sentences,
)
from phase9e_metrics import Phase9EError, summarize_generation_errors


GATE = {
    "minimum_primary_errors": 2,
    "minimum_dominant_fraction": 0.40,
    "minimum_dominance_margin": 0.10,
    "bootstrap_repetitions": 100,
    "bootstrap_seed": 9501,
}


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"] + "\n" + messages[1]["content"]


def arm(em, f1=None, attempted=True):
    if not attempted:
        return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}
    return {"attempted": True, "em": float(em), "f1": float(em if f1 is None else f1), "generation_time_ms": 1.0}


def row(index, *, category):
    selected = True
    baseline = arm(0.0)
    extractive = arm(0.0)
    oracle = arm(0.0)
    combined = arm(0.0)
    copy = arm(0.0)
    if category == "prompt_format":
        extractive = arm(1.0)
    elif category == "context_localization":
        oracle = arm(1.0)
    elif category == "both_main_effects":
        extractive = arm(1.0)
        oracle = arm(1.0)
    elif category == "interaction_only":
        combined = arm(1.0)
    elif category == "lexical_evidence_insufficiency":
        copy = arm(1.0)
    elif category == "generator_copy_failure":
        pass
    else:
        raise ValueError(category)
    return {
        "question_id": f"nq_{index}",
        "retrieval_hit": True,
        "selected_hit": selected,
        "oracle_sentence_found": True,
        "oracle_sentence_count": 1,
        "arms": {
            "baseline": baseline,
            "extractive": extractive,
            "oracle_context": oracle,
            "oracle_extractive": combined,
            "gold_answer_copy": (
                arm(0.0, attempted=False)
                if category == "interaction_only"
                else copy if category == "generator_copy_failure" else arm(0.0)
            ),
        },
    }


class ProbeTests(unittest.TestCase):
    def test_sentence_extraction_is_deterministic_and_deduplicated(self):
        passages = ["First fact. The answer is Paris. Last fact.", "The answer is Paris."]
        first = extract_gold_matched_sentences(passages, ["Paris"])
        second = extract_gold_matched_sentences(passages, ["Paris"])
        self.assertEqual(first, second)
        self.assertEqual(first, ("The answer is Paris.",))

    def test_extractive_prompt_contains_contract_but_not_hidden_state(self):
        tokenizer = FakeTokenizer()
        prompt = build_extractive_prompt(tokenizer, "Where?", ["Paris is here."])
        self.assertIn("verbatim", prompt)
        self.assertIn("Paris is here.", prompt)
        self.assertNotIn("raw_prompt", prompt)

    def test_gold_copy_prompt_is_separate_profile(self):
        prompt = build_gold_answer_copy_prompt(FakeTokenizer(), "Where?", "Paris")
        self.assertIn("Candidate answer: Paris", prompt)
        self.assertIn("Copy", prompt)

    def test_each_attribution_fixture_is_counted(self):
        categories = [
            "prompt_format", "context_localization", "both_main_effects",
            "interaction_only", "lexical_evidence_insufficiency", "generator_copy_failure",
        ]
        payload = {"samples": [row(650 + index, category=category) for index, category in enumerate(categories)]}
        summary = summarize_generation_errors(payload, gate={**GATE, "minimum_primary_errors": 2})
        self.assertEqual(summary["cohorts"]["selected_hit_baseline_errors"], 6)
        self.assertEqual(summary["primary_error_attribution"]["counts"]["both_main_effects"], 1)
        self.assertEqual(summary["decision"]["primary_failure_class"], "mixed_generation_errors")

    def test_adaptive_copy_arm_contract_is_enforced(self):
        bad = row(700, category="generator_copy_failure")
        bad["arms"]["oracle_extractive"] = arm(1.0)
        bad["arms"]["gold_answer_copy"] = arm(0.0, attempted=True)
        with self.assertRaises(Phase9EError):
            summarize_generation_errors({"samples": [bad]}, gate=GATE)

    def test_baseline_correct_rows_are_excluded_from_primary_errors(self):
        good = row(701, category="prompt_format")
        good["arms"]["baseline"] = arm(1.0)
        good["arms"]["extractive"] = arm(1.0)
        good["arms"]["oracle_context"] = arm(1.0)
        good["arms"]["oracle_extractive"] = arm(1.0)
        good["arms"]["gold_answer_copy"] = arm(0.0, attempted=False)
        summary = summarize_generation_errors({"samples": [good]}, gate=GATE)
        self.assertEqual(summary["cohorts"]["selected_hit_baseline_errors"], 0)
        self.assertEqual(summary["decision"]["primary_failure_class"], "insufficient_primary_errors")


if __name__ == "__main__":
    unittest.main()
