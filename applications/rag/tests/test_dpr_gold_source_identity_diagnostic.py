"""Synthetic tests for the local-only DPR source identity diagnostic."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.collab.five_ideas.run_dpr_gold_source_identity_diagnostic import (
    _dataset_info_summary,
    _source_analysis,
)


class DprGoldSourceIdentityDiagnosticTests(unittest.TestCase):
    def test_source_analysis_is_aggregate_only_and_detects_long_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gold.json.gz"
            rows = [
                {"question": "private question", "title": "T", "context": "word " * 140, "example_id": "e1"},
                {"question": "private question 2", "title": "T", "context": "short context", "example_id": "e2"},
                {"question": "private question 3", "title": "T", "context": "", "example_id": "e3"},
            ]
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                json.dump({"data": rows}, handle)
            report = _source_analysis(source, root)
        self.assertEqual(report["source_file"]["record_count"], 3)
        self.assertEqual(report["field_stats"]["context"]["nonempty_count"], 2)
        self.assertEqual(report["context_lengths"]["nonempty_context_count"], 2)
        self.assertEqual(report["hundred_word_compatibility"]["classification"], "long_context_dominant")
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("private question", serialized)
        self.assertNotIn("word word word", serialized)

    def test_dataset_info_summary_reports_features_without_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "facebook___wiki_dpr" / "config" / "dataset_info.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({
                "features": {
                    "id": {"dtype": "string"},
                    "title": {"dtype": "string"},
                    "text": {"dtype": "string"},
                    "embeddings": {"_type": "Array3D"},
                },
                "splits": {"train": {"num_examples": 21}},
                "version": "1.0.0",
                "description": "do not export",
            }), encoding="utf-8")
            report = _dataset_info_summary(root, metadata)
        self.assertIsNotNone(report)
        info = report["infos"][0]
        self.assertTrue(info["feature_presence"]["embeddings"])
        self.assertEqual(info["splits"]["train"]["num_examples"], 21)
        self.assertNotIn("do not export", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
