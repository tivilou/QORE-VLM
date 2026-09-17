"""Synthetic contract tests for the TREG selector and gain artifact."""

from __future__ import annotations

import hashlib
import unittest

from applications.rag.treg_selector import (
    READER_MODEL_ID,
    READER_REVISION,
    SCHEMA_VERSION,
    TREGError,
    base_indices,
    select,
    validate_gain_artifact,
)


def _candidates(count: int = 6) -> list[dict]:
    return [
        {
            "id": f"p-{index}",
            "title": f"Title {index}",
            "text": f"passage content {index}",
            "retrieved_rank": index + 1,
            "retrieval_score": float(100 - index),
            "answer_scorer_score": float(1.0 - index * 0.1),
        }
        for index in range(count)
    ]


class TREGSelectorTests(unittest.TestCase):
    def test_b4_plus_gain_selects_residual_and_orders_by_rank(self) -> None:
        rows = _candidates()
        base = base_indices(rows)
        gains = {row["id"]: float(index) for index, row in enumerate(rows) if index not in base}
        result = select(rows, gains)
        self.assertEqual(len(result.selected_indices), 5)
        self.assertEqual(result.residual_indices, (5,))
        self.assertEqual(
            list(result.selected_indices),
            sorted(result.selected_indices, key=lambda index: rows[index]["retrieved_rank"]),
        )

    def test_online_contract_rejects_evidence(self) -> None:
        rows = _candidates()
        rows[0]["evidence"] = {}
        with self.assertRaisesRegex(TREGError, "forbidden"):
            select(rows, {row["id"]: 0.0 for row in rows[1:]})

    def test_gain_artifact_arithmetic_and_identity_are_checked(self) -> None:
        rows = _candidates()
        base = base_indices(rows)
        base_ids = [rows[index]["id"] for index in base]
        residual = [row for row in rows if row["id"] not in base_ids]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "treg_reader_gain_artifact",
            "protocol": {
                "case_count": 1,
                "top50_count": 50,
                "k": 5,
                "b4_size": 4,
                "gain_definition": "Reader(q,B4+p)-Reader(q,B4)",
            },
            "provenance": {
                "model_id": READER_MODEL_ID,
                "revision": READER_REVISION,
                "config_sha256": hashlib.sha256(b"fixture").hexdigest(),
            },
            "cases": [],
        }
        # The validator is deliberately frozen to Top-50, so this test uses a
        # full synthetic pool while keeping the arithmetic easy to inspect.
        full = _candidates(50)
        base_full = base_indices(full)
        base_ids_full = [full[index]["id"] for index in base_full]
        residual_full = [row for row in full if row["id"] not in base_ids_full]
        payload["cases"] = [{
            "case_number": 1,
            "candidate_ids": [row["id"] for row in full],
            "b4_ids": base_ids_full,
            "gains": [
                {"id": row["id"], "base_score": 0.2, "with_candidate_score": 0.3, "gain": 0.1}
                for row in residual_full
            ],
        }]
        result = validate_gain_artifact(payload, [{"top_50": full}])
        self.assertEqual(len(result[1]), 46)
        payload["cases"][0]["gains"][0]["gain"] = 0.2
        with self.assertRaisesRegex(TREGError, "arithmetic"):
            validate_gain_artifact(payload, [{"top_50": full}])


if __name__ == "__main__":
    unittest.main()
