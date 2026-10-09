"""Small regression for label-free boundary diagnostics and audit statistics."""
import unittest
import numpy as np
from scripts.collab.five_ideas.analyze_semantic_reader_training_repair import describe, mobility, evidence_rescue_diagnostics


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.base = np.arange(50, 0, -1, dtype=float)
        self.ranks = list(range(1, 51))

    def test_common_shift_never_changes_selection(self):
        result = mobility(self.base, self.base + 10., self.ranks)
        self.assertFalse(result["changed_top5"])
        self.assertTrue(result["range_less_than_boundary_gap"])

    def test_small_range_cannot_cross_boundary(self):
        scores = self.base.copy(); scores[5] += .9
        result = mobility(self.base, scores, self.ranks)
        self.assertEqual(result["rank5_rank6_gap"], 1.)
        self.assertTrue(result["range_less_than_boundary_gap"])
        self.assertFalse(result["changed_top5"])

    def test_sufficient_range_is_not_a_guaranteed_swap(self):
        scores = self.base.copy(); scores[49] += 2.
        result = mobility(self.base, scores, self.ranks)
        self.assertFalse(result["range_less_than_boundary_gap"])
        self.assertFalse(result["changed_top5"])

    def test_genuine_swap_is_detected(self):
        scores = self.base.copy(); scores[5] += 1.1
        self.assertTrue(mobility(self.base, scores, self.ranks)["changed_top5"])

    def test_ties_respect_rank_and_same_set_different_order(self):
        scores = self.base.copy(); scores[5] += 1.
        self.assertFalse(mobility(self.base, scores, self.ranks)["changed_top5"])
        scores = self.base.copy(); scores[4] += 10.
        self.assertFalse(mobility(self.base, scores, self.ranks)["changed_top5"])

    def test_stats_reject_missing_nonfinite(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                describe(values)
        result = describe([0., 1., 2.])
        self.assertEqual(result["median"], 1.)
        self.assertEqual(result["nonzero"], 2)

    def test_posthoc_rescue_bounds_are_not_a_selection_promise(self):
        candidates = [{"id": str(i), "retrieved_rank": i + 1,
                       "evidence": {"positive_consensus": i in (0, 5, 49)}} for i in range(50)]
        scores = self.base.copy(); scores[5] += 1.1
        trace = [{"methods": {m: {"score_vector": self.base if m == "frozen_reader_topk" else scores}
                               for m in screen_methods()}}]
        result = evidence_rescue_diagnostics([{"top_50": candidates}], trace, .25)
        self.assertEqual(result["reader_missing_positive_capacity"], 2)
        bands = result["methods"]["quantum_semantic"]["score_rank_bands"]
        self.assertEqual(bands["6_10"]["actually_selected"], 1)
        self.assertEqual(bands["21_50"]["outside_architectural_residual_bound"], 1)


def screen_methods():
    return ("frozen_reader_topk", "quantum_semantic", "classical_semantic", "quantum_scalar_control")


if __name__ == "__main__":
    unittest.main()
