"""Identity and metric contracts for the offline Q-ARCG case audit."""

import copy
import unittest

from scripts.collab.five_ideas.analyze_qarcg_reader_screen_cases import audit, digest


def fixture():
    passages = []
    rows = []
    for i in range(50):
        positive = i < 3
        direct = i < 2
        passages.append({
            "id": str(i), "retrieved_rank": i + 1, "title": "fixture", "text": "fixture",
            "evidence": {"positive_consensus": positive, "direct_consensus": direct,
                         "positive_all_models": positive, "models": {},
                         "consensus_label": "direct" if direct else "partial" if positive else "irrelevant"},
        })
        rows.append({"case_number": 1, "candidate_id": str(i), "relevance_logit": 50.0 - i,
                     "span_logit": 2.0, "span_margin": 1.0, "start_entropy": 0.5,
                     "end_entropy": 0.5, "truncated": False})
    baseline = {"score_vector": [50.0 - i for i in range(50)],
                "selected_id_sha256": [digest(str(i)) for i in range(5)], "silver_overlap": 5}
    quantum = copy.deepcopy(baseline)
    quantum["selected_id_sha256"][-1] = digest("5")
    quantum["score_vector"][5] = 46.1
    quantum["silver_overlap"] = 4
    detail = {"cases": [{"case_number": 1, "question": "fixture question", "gold_answers": ["fixture"],
                          "top_50": passages, "selectors": [{"selector_id": "silver_oracle_common_order",
                                                             "selected_top_5": passages[:5]}]}]}
    trace = {"schema_version": "sample-trace.v2", "input_sha256": "fixture-hash", "cases": [{
        "case_number": 1, "question_sha256": digest("fixture question"),
        "candidate_id_sha256": [digest(str(i)) for i in range(50)],
        "methods": {"frozen_reader_topk": baseline, "trained_q_arcg": quantum,
                    "trained_same_budget_classical_control": copy.deepcopy(baseline)},
    }]}
    reader = {"input": {"sha256": "fixture-hash"}, "cases": rows}
    return detail, trace, reader


class CaseAuditTests(unittest.TestCase):
    def setUp(self):
        self.detail, self.trace, self.reader = fixture()

    def run_audit(self):
        return audit(self.detail, self.trace, self.reader, "fixture-hash", expected_cases=1)

    def test_filler_swap_is_not_evidence_loss(self):
        summary, private = self.run_audit()
        swap = summary["swaps"][0]
        self.assertEqual(swap["silver_overlap_delta"], -1)
        self.assertEqual(swap["positive_consensus_delta"], 0)
        self.assertEqual(swap["direct_consensus_delta"], 0)
        self.assertEqual(summary["panel_label_ceiling"]["positive_slots"], 3)
        self.assertNotIn("question", swap)
        self.assertIn("question", private[0])

    def test_input_hash(self):
        self.trace["input_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_audit()

    def test_question_identity(self):
        self.trace["cases"][0]["question_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "question identity"):
            self.run_audit()

    def test_candidate_order(self):
        self.trace["cases"][0]["candidate_id_sha256"].reverse()
        with self.assertRaisesRegex(ValueError, "candidate order"):
            self.run_audit()

    def test_duplicate_selection(self):
        selected = self.trace["cases"][0]["methods"]["trained_q_arcg"]["selected_id_sha256"]
        selected[0] = selected[1]
        with self.assertRaisesRegex(ValueError, "selected set"):
            self.run_audit()

    def test_reader_drift(self):
        self.reader["cases"][0]["relevance_logit"] += 0.1
        with self.assertRaisesRegex(ValueError, "relevance drift"):
            self.run_audit()

    def test_bad_oracle_reference(self):
        self.detail["cases"][0]["selectors"][0]["selected_top_5"][0] = self.detail["cases"][0]["top_50"][5]
        with self.assertRaisesRegex(ValueError, "panel-label ceiling"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
