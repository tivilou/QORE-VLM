"""Synthetic contract tests for the ECAD no-GPU preflight."""

from __future__ import annotations

import hashlib
import unittest

from applications.rag.ecad_preflight import PAIRWISE_NLI_SCHEMA, analyze, run_preflight
from applications.rag.tests.test_qdes_preflight import _fixture


def _pairwise_fixture() -> dict:
    source = _fixture()
    cases = []
    for case in source["cases"]:
        ids = [item["id"] for item in case["top_50"]]
        pairs = []
        for i, left in enumerate(ids):
            for j, right in enumerate(ids):
                if j <= i:
                    continue
                # Make the second passage a deterministic conflict hub. This
                # tests that the preflight can detect an independent signal.
                conflict = 0.9 if (left.endswith("-2") or right.endswith("-2")) else 0.01
                pairs.append({
                    "left_id": left,
                    "right_id": right,
                    "contradiction": conflict,
                    "entailment": 0.0,
                    "neutral": 1.0 - conflict,
                })
        cases.append({"case_number": case["case_number"], "pairs": pairs})
    return {
        "schema_version": PAIRWISE_NLI_SCHEMA,
        "provenance": {
            "model_id": "fixture-nli",
            "revision": "fixture-rev",
            "config_sha256": hashlib.sha256(b"fixture-config").hexdigest(),
        },
        "cases": cases,
    }


class ECADPreflightTests(unittest.TestCase):
    def test_missing_nli_is_explicit_negative_control(self) -> None:
        result = analyze(_fixture())
        self.assertEqual(result["metrics"]["score_source"], "lexical_proxy")
        self.assertFalse(result["gates"]["pairwise_nli_gate"]["pass"])
        self.assertFalse(result["gates"]["overall"]["pass"])

    def test_pairwise_schema_and_provenance_are_required(self) -> None:
        pairwise = _pairwise_fixture()
        result = analyze(_fixture(), pairwise)
        self.assertEqual(result["metrics"]["pair_count_total"], 50 * 1225)
        self.assertEqual(result["metrics"]["score_source"], "pairwise_nli")
        self.assertTrue(result["gates"]["pairwise_nli_gate"]["pass"])
        self.assertEqual(result["metrics"]["pair_count_total"], 61250)

    def test_malformed_pairwise_pairs_fail_closed(self) -> None:
        pairwise = _pairwise_fixture()
        pairwise["cases"][0]["pairs"].pop()
        with self.assertRaisesRegex(ValueError, "must contain 1225 pairs"):
            analyze(_fixture(), pairwise)

    def test_replay_and_no_feedback_contract(self) -> None:
        first = analyze(_fixture())
        second = analyze(_fixture())
        self.assertEqual(first["replay_digest"], second["replay_digest"])
        self.assertFalse(first["gates"]["leakage_gate"]["selection_feedback"])
        self.assertIn("gold_answers", first["forbidden_selection_fields"])


if __name__ == "__main__":
    unittest.main()
