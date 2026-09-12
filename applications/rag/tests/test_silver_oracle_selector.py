"""Unit tests for the diagnostic-only silver oracle."""

from __future__ import annotations

import copy
import unittest

from applications.rag.silver_oracle_selector import (
    SilverOracleError,
    count_consensus,
    select,
)


def passage(candidate_id: str, rank: int, score: float, labels: list[str]) -> dict:
    return {
        "id": candidate_id,
        "retrieved_rank": rank,
        "answer_scorer_score": score,
        "evidence": {
            "model_labels": labels,
            "mean_confidence": 0.8,
        },
    }


class SilverOracleSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.passages = [
            passage("negative-high", 1, 0.99, ["irrelevant"] * 3),
            passage("positive", 2, 0.10, ["partial", "partial", "irrelevant"]),
            passage("direct-low", 3, 0.01, ["direct", "direct", "irrelevant"]),
            passage("negative-fill", 4, 0.80, ["irrelevant"] * 3),
            passage("direct-high", 5, 0.50, ["direct", "direct", "direct"]),
            passage("negative-low", 6, 0.20, ["irrelevant"] * 3),
        ]

    def test_consensus_priority_then_answer_scorer_fill(self) -> None:
        result = select(self.passages, k=5)
        selected = [self.passages[index]["id"] for index in result.selected_indices]
        self.assertEqual(
            selected,
            ["direct-high", "direct-low", "positive", "negative-high", "negative-fill"],
        )
        self.assertEqual(
            [row["tier"] for row in result.trace],
            ["direct_consensus", "direct_consensus", "positive_consensus", "answer_scorer_fill", "answer_scorer_fill"],
        )
        self.assertEqual(count_consensus(self.passages), {"direct": 2, "broad": 3})

    def test_replay_is_deterministic(self) -> None:
        first = select(copy.deepcopy(self.passages), k=5)
        second = select(copy.deepcopy(self.passages), k=5)
        self.assertEqual(first, second)

    def test_requires_three_labels(self) -> None:
        invalid = copy.deepcopy(self.passages)
        invalid[0]["evidence"]["model_labels"] = ["irrelevant", "irrelevant"]
        with self.assertRaises(SilverOracleError):
            select(invalid, k=5)


if __name__ == "__main__":
    unittest.main()
