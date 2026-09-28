import unittest

import numpy as np

from applications.rag.scorer_benchmark import (
    DEFAULT_SCORER_SPECS,
    ScorerBenchmarkError,
    rank_indices,
    score_summary,
)


class ScorerBenchmarkTests(unittest.TestCase):
    def test_allowlist_has_five_distinct_specs(self) -> None:
        identifiers = [spec.scorer_id for spec in DEFAULT_SCORER_SPECS]
        self.assertEqual(len(identifiers), 5)
        self.assertEqual(len(set(identifiers)), 5)
        self.assertEqual(identifiers[0], "dpr_reader_single_nq")

    def test_rank_is_deterministic_on_score_ties(self) -> None:
        selected = rank_indices(
            [0.5, 0.5, 0.9, 0.5],
            [3, 1, 2, 4],
            ["c", "a", "b", "d"],
            3,
        )
        self.assertEqual(selected, [2, 1, 0])

    def test_rank_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(ScorerBenchmarkError):
            rank_indices([0.2, float("nan")], [1, 2], ["a", "b"], 1)

    def test_score_summary_is_finite(self) -> None:
        summary = score_summary(np.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)
        self.assertAlmostEqual(summary["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
