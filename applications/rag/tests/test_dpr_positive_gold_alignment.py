"""Synthetic contracts for the observation-only DPR positive audit."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from applications.rag.dpr_positive_gold_alignment import (
    DPR_MAPPING_STATUSES,
    align_dpr_positives_to_wiki,
    build_dpr_question_index,
    evidence_for_question,
    passage_text_fingerprint,
    validate_dpr_compact_result,
)
from applications.rag.gold_evidence_alignment import GoldAlignmentError
from scripts.collab.five_ideas.run_dpr_positive_gold_alignment_audit import (
    _gold_info_context_diagnostics,
    _iter_dpr_positive_records,
    _source_question_diagnostics,
    _question_join_preflight,
)


class DprPositiveGoldAlignmentTests(unittest.TestCase):
    def _join(self, records: list[dict[str, object]], question_id: str = "q1"):
        index = build_dpr_question_index(records)
        status, evidence = evidence_for_question(" Who wrote it? ", index)
        return {question_id: (status, evidence)}

    def test_question_join_normalizes_whitespace_and_case(self) -> None:
        joins = self._join([
            {
                "question": "who   wrote IT?",
                "positive_ctxs": [{"id": "7", "title": "T", "text": "P"}],
            }
        ])
        self.assertEqual(joins["q1"][0], "joined")

    def test_duplicate_question_with_conflicting_positives_is_ambiguous(self) -> None:
        joins = self._join([
            {"question": "Who wrote it?", "positive_ctxs": [{"id": "7", "title": "T", "text": "P"}]},
            {"question": "who wrote it?", "positive_ctxs": [{"id": "8", "title": "T", "text": "P"}]},
        ])
        self.assertEqual(joins["q1"][0], "ambiguous_question_join")

    def test_official_passage_id_maps_without_text_retention(self) -> None:
        aligned = align_dpr_positives_to_wiki(
            [{"id": "7", "title": "Other", "text": "Different corpus rendering"}],
            self._join([{"question": "Who wrote it?", "positive_ctxs": [{"psg_id": "7", "title": "T", "text": "P"}]}]),
        )
        result = aligned["q1"]
        self.assertEqual(result["mapping_status"], "mapped")
        self.assertEqual(result["identity_method_counts"]["official_passage_id"], 1)
        self.assertEqual(result["verified_positive_count"], 1)

    def test_title_and_full_text_hash_maps_without_source_id(self) -> None:
        aligned = align_dpr_positives_to_wiki(
            [{"id": "7", "title": "A Title", "text": "Exact evidence text"}],
            self._join([{"question": "Who wrote it?", "positive_ctxs": [{"title": "A Title", "text": "Exact evidence text"}]}]),
        )
        result = aligned["q1"]
        self.assertEqual(result["mapping_status"], "mapped")
        self.assertEqual(result["identity_method_counts"]["title_text_hash"], 1)
        self.assertNotEqual(
            passage_text_fingerprint("A Title", "Exact evidence text"),
            passage_text_fingerprint("A Title", "Exact evidence text \0"),
        )

    def test_duplicate_corpus_identity_is_ambiguous(self) -> None:
        aligned = align_dpr_positives_to_wiki(
            [
                {"id": "7", "title": "A Title", "text": "Exact evidence text"},
                {"id": "8", "title": "A Title", "text": "Exact evidence text"},
            ],
            self._join([{"question": "Who wrote it?", "positive_ctxs": [{"title": "A Title", "text": "Exact evidence text"}]}]),
        )
        result = aligned["q1"]
        self.assertEqual(result["mapping_status"], "ambiguous_wiki_dpr_identity")
        self.assertEqual(result["ambiguous_positive_count"], 1)

    def test_partial_identity_is_not_strictly_mapped(self) -> None:
        aligned = align_dpr_positives_to_wiki(
            [{"id": "7", "title": "A", "text": "one"}],
            self._join([{
                "question": "Who wrote it?",
                "positive_ctxs": [
                    {"id": "7", "title": "A", "text": "one"},
                    {"id": "8", "title": "B", "text": "two"},
                ],
            }]),
        )
        result = aligned["q1"]
        self.assertEqual(result["mapping_status"], "partial_wiki_dpr_identity")
        self.assertEqual(result["verified_positive_count"], 1)
        self.assertEqual(result["unresolved_positive_count"], 1)
        self.assertIn(result["mapping_status"], DPR_MAPPING_STATUSES)

    def test_compact_validator_enforces_complete_chain_and_privacy(self) -> None:
        payload = {
            "samples": [{
                "question_id": "q1", "question_identity_verified": True,
                "mapping_status": "mapped", "official_positive_count": 1,
                "verified_positive_count": 1, "unresolved_positive_count": 0,
                "ambiguous_positive_count": 0, "gold_passage_count": 1,
                "top50_gold_count": 1, "top5_gold_count": 1,
                "retrieval_hit": True, "selected_hit": True,
            }]
        }
        validate_dpr_compact_result(payload)
        payload["samples"][0]["question"] = "forbidden"
        with self.assertRaises(GoldAlignmentError):
            validate_dpr_compact_result(payload)

    def test_gzip_json_array_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positives.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump([{"question": "A"}, {"question": "B"}], handle)
            self.assertEqual(
                [record["question"] for record in _iter_dpr_positive_records(path)],
                ["A", "B"],
            )

    def test_official_nq_test_gold_info_container_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nq-test_gold_info.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump({"data": [{
                    "question": "Who wrote it?",
                    "example_id": "e1",
                    "title": "A Title",
                    "context": "Exact evidence text",
                }]}, handle)
            rows = list(_iter_dpr_positive_records(path))
        self.assertEqual(rows[0]["question"], "Who wrote it?")
        self.assertEqual(rows[0]["positive_ctxs"][0]["id"], "e1")
        joins = self._join(rows)
        self.assertEqual(joins["q1"][0], "joined")

    def test_official_gold_info_field_diagnostics_are_aggregate_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nq-test_gold_info.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump({"data": [
                    {"question": "Q1", "question_tokens": ["Q1"],
                     "title": "Title", "context": "Context", "example_id": "e1"},
                    {"question": "Q2", "question_tokens": "bad",
                     "context": "x" * 70, "example_id": None},
                    {"question": "Q3", "title": "", "context": "  "},
                    {"question": "Q4", "document": {
                        "title": "Nested", "text": "y" * 300, "id": "d4"}},
                    {"question": "Q5", "positive_contexts": [{
                        "title": "Alternate", "text": "z" * 1100, "id": "p5"}]},
                    {"question": "outside", "title": "Other", "context": "Other context"},
                ]}, handle)
            diagnostics = _gold_info_context_diagnostics(
                path, [
                    {"question": "Q1"}, {"question": "Q2"}, {"question": "Q3"},
                    {"question": "Q4"}, {"question": "Q5"},
                ]
            )
        self.assertEqual(diagnostics["privacy"], "aggregate_counts_only")
        self.assertEqual(diagnostics["target_questions"]["record_count"], 5)
        target = diagnostics["target_questions"]
        self.assertEqual(target["adapted_context_stats"]["usable_positive_context"], 2)
        self.assertEqual(target["adapted_context_stats"]["missing_title"], 3)
        self.assertEqual(target["adapted_context_stats"]["missing_context"], 2)
        self.assertEqual(target["adapted_context_stats"]["missing_title_and_context"], 2)
        self.assertEqual(target["field_stats"]["question"]["present"], 5)
        self.assertEqual(target["field_stats"]["question_tokens"]["type_shape_counts"]["sequence"], 1)
        self.assertEqual(target["field_stats"]["question_tokens"]["type_shape_counts"]["string"], 1)
        self.assertEqual(
            target["field_path_stats"]["document.text"]["nonempty_record_count"], 1
        )
        self.assertEqual(
            target["candidate_positive_fields"]["field_stats"]
            ["positive_contexts[].text"]["nonempty_record_count"],
            1,
        )
        adapted_buckets = target["adapted_context_length_chars"][
            "record_length_bucket_counts"
        ]
        self.assertEqual(adapted_buckets["empty"], 2)
        self.assertEqual(adapted_buckets["1_63"], 1)
        self.assertEqual(adapted_buckets["64_255"], 1)
        self.assertEqual(adapted_buckets["256_1023"], 1)
        alternate_buckets = target["candidate_positive_fields"]["text_length_stats"]
        self.assertEqual(
            alternate_buckets["positive_contexts[].text"]
            ["record_length_bucket_counts"]["1024_plus"],
            1,
        )
        serialized = json.dumps(diagnostics, sort_keys=True)
        self.assertNotIn("x" * 70, serialized)
        self.assertNotIn("y" * 300, serialized)
        self.assertNotIn("z" * 1100, serialized)
        self.assertNotIn("Q1", serialized)

    def test_question_join_preflight_blocks_unreachable_mapping_gate(self) -> None:
        questions = [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}, {"id": "q4"}, {"id": "q5"}]
        joins = {
            "q1": ("no_question_join", None),
            "q2": ("joined", object()),
            "q3": ("no_question_join", None),
            "q4": ("no_positive_context", object()),
            "q5": ("joined", object()),
        }
        summary = _question_join_preflight(
            questions, joins, minimum_mapping_rate=0.8
        )
        self.assertEqual(summary["joined_count"], 2)
        self.assertEqual(summary["required_join_count"], 4)
        self.assertFalse(summary["can_reach_mapping_gate"])
        self.assertEqual(summary["join_status_counts"]["no_question_join"], 2)

    def test_source_question_diagnostics_exposes_deterministic_variant_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positives.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump([{
                    "question": "What is the answer",
                    "positive_ctxs": [{"title": "T", "text": "P"}],
                }], handle)
            diagnostics = _source_question_diagnostics(
                path,
                [{"question": "What is the answer?"}],
            )
        variants = diagnostics["key_variants"]
        self.assertEqual(variants["current_exact"]["target_source_key_overlap"], 0)
        self.assertEqual(variants["strip_punctuation"]["target_source_key_overlap"], 1)
        self.assertTrue(diagnostics["strict_join_rule_unchanged"])


if __name__ == "__main__":
    unittest.main()
