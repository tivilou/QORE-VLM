import gzip
import json
from pathlib import Path
import tempfile
import unittest

from applications.rag.dpr_positive_gold_eligible_universe import (
    make_compact_row,
    summarize_universe,
    validate_universe_result,
)
from scripts.collab.five_ideas.run_dpr_positive_gold_eligible_universe_audit import (
    _build_context_only_question_index,
    _load_config,
    _source_preflight,
)


class EligibleUniverseTests(unittest.TestCase):
    def _identity(self, status="mapped", verified=("p1",), official=1):
        return {
            "mapping_status": status,
            "official_positive_count": official,
            "verified_positive_count": len(verified),
            "unresolved_positive_count": max(0, official - len(verified)),
            "ambiguous_positive_count": 0,
            "verified": set(verified),
        }

    def test_oracle_ceiling_is_top50_availability_only(self):
        row = make_compact_row(
            "q1", "Who wrote it?", "joined", self._identity(),
            top50_ids=["p1", "p2"], top5_ids=["p2"],
        )
        self.assertTrue(row["oracle_top5_evidence_ceiling"])
        self.assertFalse(row["selected_hit"])
        self.assertEqual(row["top50_gold_count"], 1)
        self.assertEqual(row["top5_gold_count"], 0)

    def test_bottleneck_strata_and_rates(self):
        rows = [
            make_compact_row("q1", "Q1", "joined", self._identity(), ["p1"], ["p1"],
                             baseline_em=1.0, generator_bucket="selected_hit_answer_correct"),
            make_compact_row("q2", "Q2", "joined", self._identity(), ["p1"], ["p2"],
                             baseline_em=1.0, generator_bucket="no_selected_strict_gold"),
            make_compact_row("q3", "Q3", "joined", self._identity(), ["p2"], ["p2"],
                             baseline_em=0.0, generator_bucket="no_selected_strict_gold"),
            make_compact_row("q4", "Q4", "joined", self._identity(verified=("p2",)), ["p2"], ["p2"],
                             baseline_em=0.0, generator_bucket="selected_hit_generation_error"),
        ]
        summary = summarize_universe(
            rows, all_questions=6, official_nonempty_context_questions=3
        )
        self.assertEqual(summary["population"]["excluded_rows"], 2)
        self.assertEqual(summary["strict_bottleneck_counts"]["retrieval_bottleneck_questions"], 1)
        self.assertEqual(summary["strict_bottleneck_counts"]["selector_bottleneck_questions"], 1)
        self.assertEqual(summary["strict_bottleneck_counts"]["generator_bottleneck_questions"], 1)
        self.assertEqual(summary["strict_bottleneck_counts"]["oracle_top5_evidence_ceiling_questions"], 3)

    def test_partial_mapping_is_not_oracle_or_bottleneck_evidence(self):
        row = make_compact_row(
            "q1", "Q", "joined", self._identity("partial_wiki_dpr_identity", ("p1",), 2),
            ["p1"], ["p1"],
        )
        self.assertEqual(row["mapping_failure_category"], "source_or_corpus_version_mismatch")
        self.assertFalse(row["retrieval_hit"])
        self.assertFalse(row["selected_hit"])
        self.assertFalse(row["oracle_top5_evidence_ceiling"])
        self.assertEqual(row["gold_passage_count"], 0)
        payload = {"samples": [row]}
        validate_universe_result(payload)

    def test_compact_privacy_and_fingerprint(self):
        row = make_compact_row("q1", "Q", "joined", self._identity(), ["p1"], ["p1"])
        validate_universe_result({"samples": [row]})

    def test_unmapped_rows_retain_accounting_without_online_signals(self):
        row = make_compact_row(
            "q1", "Q", "joined", self._identity("no_wiki_dpr_identity", (), 1),
            ["p1"], ["p1"],
        )
        self.assertEqual(row["official_positive_count"], 1)
        self.assertEqual(row["gold_passage_count"], 0)
        self.assertFalse(row["retrieval_hit"])
        self.assertFalse(row["selected_hit"])
        self.assertFalse(row["oracle_top5_evidence_ceiling"])
        with self.assertRaises(ValueError):
            validate_universe_result({"samples": [{**row, "question": "forbidden"}]})

    def test_config_is_frozen(self):
        phase = _load_config(
            Path(__file__).resolve().parents[3]
            / "configs/experiments/dpr_positive_gold_eligible_universe_audit.yaml"
        )
        self.assertEqual(phase["dataset"]["max_samples"], 3610)
        self.assertEqual(phase["expected_source_accounting"]["official_nonempty_context_questions"], 1868)

    def test_preflight_blocks_population_mismatch_without_runtime(self):
        phase = {
            "expected_source_accounting": {
                "all_questions": 3, "official_nonempty_context_questions": 2
            }
        }
        questions = [
            {"id": "q1", "question": "Q1"},
            {"id": "q2", "question": "Q2"},
            {"id": "q3", "question": "Q3"},
        ]
        joins = {
            "q1": ("joined", object()),
            "q2": ("no_positive_context", object()),
            "q3": ("no_question_join", None),
        }
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "nq-test_gold_info.json.gz"
            with gzip.open(source_path, "wt", encoding="utf-8") as handle:
                json.dump({"data": [
                    {"question": "Q1", "title": "T1", "context": "C1"},
                    {"question": "Q2", "title": "T2", "context": "C2"},
                    {"question": "Q3", "title": "", "context": ""},
                ]}, handle)
            source = {"path": str(source_path), "records_scanned": 3, "target_records_retained": 3}
            result = _source_preflight(questions, joins, [{"id": "q1"}], source, phase, Path("."))
        self.assertFalse(result["decision"]["may_start_wiki_dpr"])

    def test_context_only_eligibility_does_not_require_a_title(self):
        phase = {
            "expected_source_accounting": {
                "all_questions": 2, "official_nonempty_context_questions": 2
            }
        }
        questions = [
            {"id": "q1", "question": "Q1"},
            {"id": "q2", "question": "Q2"},
        ]
        joins = {
            "q1": ("joined", object()),
            "q2": ("joined", object()),
        }
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "nq-test_gold_info.json.gz"
            with gzip.open(source_path, "wt", encoding="utf-8") as handle:
                json.dump({"data": [
                    {"question": "Q1", "title": "T1", "context": "C1"},
                    {"question": "Q2", "title": "", "context": "C2"},
                ]}, handle)
            source = {"path": str(source_path), "records_scanned": 2, "target_records_retained": 2}
            result = _source_preflight(questions, joins, questions, source, phase, Path("."))
        self.assertTrue(result["decision"]["may_start_wiki_dpr"])
        self.assertEqual(
            result["source_diagnostics"]["title_field"]["context_nonempty_title_empty_record_count"], 1
        )

    def test_context_only_adapter_preserves_nonempty_context_without_title(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "nq-test_gold_info.json.gz"
            with gzip.open(source_path, "wt", encoding="utf-8") as handle:
                json.dump({"data": [
                    {"question": "Q1", "title": "", "context": "C1"},
                    {"question": "Q2", "title": "", "context": ""},
                ]}, handle)
            index = _build_context_only_question_index(source_path)
        self.assertEqual(len(index["q1"].positives), 1)
        self.assertEqual(index["q1"].positives[0].title, "")
        self.assertEqual(index["q1"].positives[0].text, "C1")
        self.assertEqual(len(index["q2"].positives), 0)


if __name__ == "__main__":
    unittest.main()
