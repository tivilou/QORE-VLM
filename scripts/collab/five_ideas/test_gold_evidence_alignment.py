#!/usr/bin/env python3
"""Synthetic tests for the gold-evidence alignment audit."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.rag.gold_evidence_alignment import (  # noqa: E402
    align_corpus,
    extract_gold_evidence,
    match_passage,
    passage_identity,
    summarize_alignment,
    validate_compact_result,
)


class GoldEvidenceAlignmentTests(unittest.TestCase):
    def test_extracts_short_span_without_html_tokens(self):
        record = {
            "question": {"text": "Who wrote the book?"},
            "document": {
                "title": "Example Book",
                "tokens": [
                    {"token": "The", "is_html": False},
                    {"token": "author", "is_html": False},
                    {"token": "Alice", "is_html": False},
                    {"token": "</p>", "is_html": True},
                ],
            },
            "annotations": [{"short_answers": [{"start_token": 2, "end_token": 3}]}],
        }
        evidence = extract_gold_evidence(record, ["fallback"])
        self.assertEqual(evidence.document_title, "Example Book")
        self.assertEqual(evidence.strict_answers, ("alice",))
        self.assertEqual(evidence.support_answers, ("alice", "fallback"))
        self.assertTrue(evidence.has_short_span)

    def test_alignment_requires_title_and_answer(self):
        evidence = extract_gold_evidence({
            "question": "q", "document": {"title": "Example"},
            "annotations": [{"short_answers": [{"text": "Alice"}]}],
        })
        self.assertEqual(match_passage(title="Other", text="Alice", evidence=evidence), (False, False))
        self.assertEqual(match_passage(title="Example", text="Alice went home", evidence=evidence), (True, True))
        self.assertEqual(match_passage(title="Example", text="Aliceland", evidence=evidence), (False, False))

    def test_full_alignment_and_cache_id_are_deterministic(self):
        evidence = extract_gold_evidence({
            "question": "q", "document": {"title": "Example"},
            "annotations": [{"short_answers": [{"text": "Alice"}]}],
        })
        first = align_corpus([
            {"id": 7, "title": "Example", "text": "Alice went home"},
            {"id": 8, "title": "Example", "text": "Aliceland"},
        ], {"q0": evidence}, progress_every=0)
        second = align_corpus([
            {"id": 7, "title": "Example", "text": "Alice went home"},
            {"id": 8, "title": "Example", "text": "Aliceland"},
        ], {"q0": evidence}, progress_every=0)
        self.assertEqual(first, second)
        self.assertEqual(first["q0"]["strict"], {"7"})
        self.assertEqual(passage_identity(7, "Example", "Alice went home", 0), "7")

    def test_summary_and_privacy_contract(self):
        samples = [{
            "question_id": "q0", "mapping_status": "mapped", "gold_passage_count": 1,
            "support_passage_count": 1, "top50_gold_count": 1, "top5_gold_count": 0,
            "top50_support_count": 1, "top5_support_count": 0, "retrieval_hit": True,
            "selected_hit": False, "support_retrieval_hit": True, "support_selected_hit": False,
            "conditional_selection_recall": 0.0,
        }]
        summary = summarize_alignment(samples, min_mapping_rate=0.8)
        self.assertEqual(summary["decision"]["status"], "ready_for_bottleneck_audit")
        payload = {"schema_version": 1, "samples": samples}
        validate_compact_result(payload)
        with self.assertRaises(ValueError):
            validate_compact_result({"samples": [{**samples[0], "question": "forbidden"}]})


if __name__ == "__main__":
    unittest.main()
