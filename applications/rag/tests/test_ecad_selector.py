"""Synthetic contract tests for ECAD pairwise-NLI selection."""

from __future__ import annotations

import hashlib
import unittest

from applications.rag.ecad_selector import (
    EXPECTED_PAIR_COUNT,
    NLI_MODEL_ID,
    NLI_REVISION,
    PairScore,
    SCHEMA_VERSION,
    ECADError,
    select,
    validate_pairwise_artifact,
)


def _candidates() -> list[dict]:
    return [
        {
            "id": f"p-{index}",
            "title": f"Title {index}",
            "text": f"passage content {index}",
            "retrieved_rank": index + 1,
            "retrieval_score": float(100 - index),
            "answer_scorer_score": float(1.0 - index * 0.02),
        }
        for index in range(6)
    ]


class ECADSelectorTests(unittest.TestCase):
    def test_conflict_penalty_avoids_conflict_hub(self) -> None:
        rows = _candidates()
        ids = [row["id"] for row in rows]
        pairs = {}
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                contradiction = 0.95 if "p-1" in {ids[left], ids[right]} else 0.0
                pairs[(ids[left], ids[right])] = PairScore(
                    ids[left], ids[right], contradiction, 0.0, 1.0 - contradiction
                )
        # The implementation uses the mean contradiction burden over the
        # already selected set; use a penalty large enough to make the hub
        # lose even on the final greedy step.
        result = select(rows, pairs, k=5, conflict_penalty=2.0)
        self.assertEqual(len(result.selected_indices), 5)
        self.assertNotIn(1, result.selected_indices)

    def test_online_contract_rejects_evidence(self) -> None:
        rows = _candidates()
        rows[0]["evidence"] = {}
        with self.assertRaisesRegex(ECADError, "forbidden"):
            select(rows, {})

    def test_pairwise_artifact_requires_complete_rank_order(self) -> None:
        rows = _candidates()
        ids = [row["id"] for row in rows]
        full = [
            {
                "left_id": f"p-{left}",
                "right_id": f"p-{right}",
                "contradiction": 0.1,
                "entailment": 0.2,
                "neutral": 0.7,
            }
            for left in range(50)
            for right in range(left + 1, 50)
        ]
        full_rows = [
            {
                "id": f"p-{index}",
                "title": f"Title {index}",
                "text": f"passage content {index}",
                "retrieved_rank": index + 1,
                "retrieval_score": float(100 - index),
                "answer_scorer_score": float(1.0 - index * 0.01),
            }
            for index in range(50)
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "ecad_pairwise_nli_artifact",
            "protocol": {
                "case_count": 1,
                "top50_count": 50,
                "pair_count_per_case": EXPECTED_PAIR_COUNT,
                "pair_order": "top50_position_i_lt_j",
            },
            "provenance": {
                "model_id": NLI_MODEL_ID,
                "revision": NLI_REVISION,
                "config_sha256": hashlib.sha256(b"fixture").hexdigest(),
            },
            "cases": [{
                "case_number": 1,
                "candidate_ids": [row["id"] for row in full_rows],
                "pairs": full,
            }],
        }
        result = validate_pairwise_artifact(payload, [{"top_50": full_rows}])
        self.assertEqual(len(result[1]), EXPECTED_PAIR_COUNT)
        payload["cases"][0]["pairs"][1]["right_id"] = "p-49"
        with self.assertRaisesRegex(ECADError, "order"):
            validate_pairwise_artifact(payload, [{"top_50": full_rows}])


if __name__ == "__main__":
    unittest.main()
