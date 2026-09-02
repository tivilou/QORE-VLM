#!/usr/bin/env python3
"""Synthetic contract tests for the private RAG selector case-study runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("run_rag_selector_case_study_50.py")
SPEC = importlib.util.spec_from_file_location("rag_case_study_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RAGSelectorCaseStudyTests(unittest.TestCase):
    def test_scope_is_exactly_fifty_questions(self):
        self.assertEqual(MODULE.CASE_COUNT, 50)
        self.assertEqual(MODULE.EXPECTED_POSITIONS, tuple(range(50)))
        self.assertEqual(MODULE.EXPECTED_GLOBAL_INDICES, tuple(range(1950, 2000)))

    def test_forbidden_compact_field_detection(self):
        self.assertTrue(MODULE._forbidden_fields({"nested": {"prediction": "x"}}))
        self.assertEqual(MODULE._forbidden_fields({"sha256": "abc", "bytes": 3}), [])

    def test_retrieval_metadata_maps_global_ids_and_ranks(self):
        class Dataset:
            def get_nearest_examples(self, name, query, k):
                self.args = (name, query, k)
                return [0.9, 0.8], {
                    "id": ["global-a", 99],
                    "title": ["Title A", ""],
                    "text": ["Body A", "Body B"],
                    "embeddings": [[1.0, 0.0], [0.0, 1.0]],
                }

        class Manager:
            _dataset = Dataset()

        result = MODULE._retrieve_with_metadata(Manager(), [1.0, 0.0], 2)
        self.assertEqual([item["id"] for item in result["records"]], ["global-a", "99"])
        self.assertEqual([item["retrieved_rank"] for item in result["records"]], [1, 2])
        self.assertEqual(result["records"][0]["passage"], "Title A. Body A")
        self.assertEqual(result["records"][1]["passage"], "Body B")

    def test_selector_contract_is_explicit_and_frozen(self):
        phase = {"selectors": [dict(item) for item in MODULE.EXPECTED_SELECTOR_SPECS]}
        self.assertEqual(tuple(item["id"] for item in MODULE._selector_contracts(phase)), MODULE.EXPECTED_SELECTOR_IDS)
        phase["selectors"][0]["gamma"] = 0.5
        with self.assertRaises(MODULE.CaseStudyError):
            MODULE._selector_contracts(phase)

    def test_selector_record_preserves_returned_order_and_proxy_label(self):
        records = [
            {"id": "a", "title": "A", "text": "answer", "passage": "A. answer", "retrieved_rank": 1, "retrieval_score": 0.1},
            {"id": "b", "title": "B", "text": "other", "passage": "B. other", "retrieved_rank": 2, "retrieval_score": 0.2},
        ]
        spec = {"id": "topk_as", "method": "topk", "K": 5}
        result = MODULE._selector_record(spec, [1, 0], records, [0.3, 0.4], "answer", ["answer"])
        self.assertEqual([item["id"] for item in result["selected_top_5"]], ["b", "a"])
        self.assertTrue(result["diagnostics"]["answer_string_match_is_proxy_not_strict_gold"])

    def test_markdown_and_json_have_case_content(self):
        case = {
            "dataset_slice": "slice", "retrieval": {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "cfg"},
            "generator": {"model_id": "model", "revision": "rev"},
            "cases": [{"case_number": 1, "question_id": "q", "question": "Who?", "gold_answers": ["A"],
                       "top_50": [{"retrieved_rank": 1, "title": "T", "id": "i", "retrieval_score": 0.1, "answer_scorer_score": 0.2, "text": "Body"}],
                       "selectors": [{"selector_id": "topk_as", "method": "topk", "prediction": "A", "metrics": {"em": 1.0, "f1": 1.0}, "diagnostics": {"answer_has_match_in_selected_text": True}, "selected_top_5": []}],
                       "selector_set_diagnostics": {}}],
        }
        rendered = MODULE._build_markdown(case)
        self.assertIn("Who?", rendered)
        self.assertIn("topk_as", rendered)


if __name__ == "__main__":
    unittest.main()
