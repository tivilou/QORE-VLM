#!/usr/bin/env python3
"""Synthetic tests for the Phase 9A transformer and compact gate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from applications.rag.context_transform import ContextTransformError, transform_context
from scripts.collab.five_ideas.phase9a_metrics import summarize_context_intervention


class WordTokenizer:
    def __init__(self):
        self._ids: dict[str, int] = {}
        self._tokens: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self._ids:
            value = len(self._ids) + 1
            self._ids[token] = value
            self._tokens[value] = token
        return self._ids[token]

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [self._id(token) for token in text.split()]}

    def decode(self, ids, **kwargs):
        del kwargs
        return " ".join(self._tokens[int(value)] for value in ids)


class MappingEncoding(Mapping):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __getitem__(self, key):
        if key != "input_ids":
            raise KeyError(key)
        return self.input_ids

    def __iter__(self):
        return iter(("input_ids",))

    def __len__(self):
        return 1


class MappingTokenizer(WordTokenizer):
    def __call__(self, text: str, add_special_tokens: bool = False):
        encoded = super().__call__(text, add_special_tokens=add_special_tokens)
        return MappingEncoding(encoded["input_ids"])


class ContextTransformTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_full_is_byte_equivalent(self):
        passages = ["  Alpha beta.  ", "Gamma\nDelta"]
        result = transform_context(passages, tokenizer=self.tokenizer, transform="full")
        self.assertEqual(result.texts, tuple(passages))
        self.assertEqual(result.full_token_count, result.transformed_token_count)
        self.assertEqual(result.reduction_ratio, 0.0)

    def test_uniform_head_uses_fixed_budget_and_is_deterministic(self):
        passages = ["a b c d e", "f g h", "i j k l"]
        first = transform_context(
            passages, tokenizer=self.tokenizer, transform="uniform_head", budget_ratio=0.5
        )
        second = transform_context(
            passages, tokenizer=self.tokenizer, transform="uniform_head", budget_ratio=0.5
        )
        self.assertEqual(first, second)
        self.assertEqual(first.full_token_count, 12)
        self.assertEqual(first.budget_tokens, 6)
        self.assertEqual(first.transformed_token_count, 6)
        self.assertEqual(first.texts, ("a b", "f g", "i j"))

    def test_tokenizer_mapping_return_type_is_supported(self):
        result = transform_context(
            ["a b c d"],
            tokenizer=MappingTokenizer(),
            transform="uniform_head",
            budget_ratio=0.5,
        )
        self.assertEqual(result.texts, ("a b",))

    def test_reader_window_centers_top_span(self):
        result = transform_context(
            ["zero one two target four five"],
            tokenizer=self.tokenizer,
            transform="reader_window",
            budget_ratio=0.5,
            hypotheses=[[{"text": "target"}]],
        )
        self.assertEqual(result.texts, ("two target four",))
        self.assertEqual(result.span_candidates, 1)
        self.assertEqual(result.span_found, 1)
        self.assertEqual(result.span_truncated, 0)
        self.assertEqual(result.fallback_count, 0)

    def test_reader_window_counts_clipping_and_fallback(self):
        result = transform_context(
            ["a b c d", "e f g h"],
            tokenizer=self.tokenizer,
            transform="reader_window",
            budget_ratio=0.25,
            hypotheses=[[{"text": "a b"}], [{"text": "missing"}]],
        )
        self.assertEqual(result.budget_tokens, 2)
        self.assertEqual(result.transformed_token_count, 2)
        self.assertEqual(result.span_candidates, 2)
        self.assertEqual(result.span_found, 1)
        self.assertEqual(result.span_truncated, 1)
        self.assertEqual(result.fallback_count, 1)

    def test_invalid_transform_contract(self):
        with self.assertRaises(ContextTransformError):
            transform_context(["a"], tokenizer=self.tokenizer, transform="unknown")
        with self.assertRaises(ContextTransformError):
            transform_context(["a"], tokenizer=self.tokenizer, transform="uniform_head", budget_ratio=0.0)


def _result(transform: str, ratio: float, delta: float) -> dict:
    samples = []
    for index in range(6):
        full_tokens = 100 + index
        transformed = full_tokens if transform == "full" else int(full_tokens * ratio)
        samples.append({
            "question_id": f"q{index}",
            "f1": 0.50 + delta,
            "em": 0.40 + delta,
            "generation_time_ms": 100.0 if transform == "full" else 70.0,
            "context_transform": transform,
            "context_budget_ratio": ratio,
            "context_full_token_count": full_tokens,
            "context_transformed_token_count": transformed,
            "context_budget_tokens": transformed,
            "context_reduction_ratio": 1.0 - transformed / full_tokens,
            "context_span_candidates": 5 if transform == "reader_window" else 0,
            "context_span_found": 5 if transform == "reader_window" else 0,
            "context_span_truncated": 0,
            "context_fallback_count": 0,
        })
    return {"config": {"context_transform": transform, "context_budget_ratio": ratio}, "samples": samples}


class Phase9AMetricsTests(unittest.TestCase):
    def test_reader_gate_passes_on_positive_matched_fixture(self):
        results = {
            "qore_as_full": _result("full", 1.0, 0.0),
            "qore_as_uniform_head_050": _result("uniform_head", 0.5, 0.005),
            "qore_as_uniform_head_075": _result("uniform_head", 0.75, 0.004),
            "qore_as_reader_window_050": _result("reader_window", 0.5, 0.02),
            "qore_as_reader_window_075": _result("reader_window", 0.75, 0.015),
        }
        summary = summarize_context_intervention(
            results,
            baseline_name="qore_as_full",
            gate={"bootstrap_repetitions": 100, "bootstrap_seed": 9},
        )
        self.assertEqual(summary["outcome"], "pass_reader_window")
        self.assertTrue(summary["direction_consistency"]["reader_window"])
        self.assertEqual(summary["n_questions"], 6)

    def test_reader_gate_fails_when_spans_fall_back(self):
        results = {
            "qore_as_full": _result("full", 1.0, 0.0),
            "qore_as_uniform_head_050": _result("uniform_head", 0.5, 0.005),
            "qore_as_uniform_head_075": _result("uniform_head", 0.75, 0.004),
            "qore_as_reader_window_050": _result("reader_window", 0.5, 0.02),
            "qore_as_reader_window_075": _result("reader_window", 0.75, 0.015),
        }
        for name in ("qore_as_reader_window_050", "qore_as_reader_window_075"):
            for sample in results[name]["samples"]:
                sample["context_span_found"] = 0
                sample["context_fallback_count"] = 5
        summary = summarize_context_intervention(
            results,
            baseline_name="qore_as_full",
            gate={"bootstrap_repetitions": 100, "bootstrap_seed": 9},
        )
        self.assertEqual(summary["outcome"], "fail_reader_span_retention")


if __name__ == "__main__":
    unittest.main()
