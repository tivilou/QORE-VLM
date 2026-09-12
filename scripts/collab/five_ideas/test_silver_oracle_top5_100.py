"""Contract tests for the fixed-100 silver-oracle Generator ceiling."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = Path(__file__).with_name("run_silver_oracle_top5_100.py")
SPEC = importlib.util.spec_from_file_location("silver_oracle_top5_100", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(index: int, *, positive: bool = False) -> dict:
    labels = ["direct", "direct", "irrelevant"] if positive else ["irrelevant"] * 3
    return {
        "id": f"p{index}",
        "title": f"title {index}",
        "text": f"text {index}",
        "passage": f"title {index}. text {index}",
        "retrieved_rank": index + 1,
        "retrieval_score": float(50 - index),
        "answer_scorer_score": float(index) / 50.0,
        "evidence": {"model_labels": labels, "mean_confidence": 0.9},
    }


class SilverOracleTop5100Tests(unittest.TestCase):
    def test_validate_only_contract_is_frozen(self) -> None:
        result = MODULE.validate_contract(
            ROOT / "configs/experiments/silver_oracle_top5_100.yaml",
            ROOT / "configs/experiments/silver_oracle_top5_100_plan.json",
        )
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["generator_calls"], 300)
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["wiki_dpr_started"])

    def test_baseline_membership_is_label_independent_and_order_is_common(self) -> None:
        case = {
            "top_50": [candidate(index, positive=index in {20, 21, 22, 23, 24}) for index in range(50)],
            "baseline_memberships": {
                "qore_as": ["p9", "p1", "p7", "p3", "p5"],
                "topk_as": ["p4", "p0", "p3", "p1", "p2"],
            },
        }
        first = MODULE.build_arm_passages(case)
        changed = copy.deepcopy(case)
        for row in changed["top_50"]:
            row["evidence"] = {"model_labels": ["irrelevant"] * 3, "mean_confidence": 0.5}
        second = MODULE.build_arm_passages(changed)
        for arm_id in ("qore_common_order", "topk_common_order"):
            first_ids = [row["id"] for row in first[arm_id]["passages"]]
            second_ids = [row["id"] for row in second[arm_id]["passages"]]
            self.assertEqual(first_ids, second_ids)
            ranks = [row["retrieved_rank"] for row in first[arm_id]["passages"]]
            self.assertEqual(ranks, sorted(ranks))
        oracle_ranks = [row["retrieved_rank"] for row in first["silver_oracle_common_order"]["passages"]]
        self.assertEqual(oracle_ranks, sorted(oracle_ranks))

    def test_compact_privacy_detector_rejects_raw_content(self) -> None:
        self.assertTrue(MODULE._forbidden_fields({"nested": {"prediction": "x"}}))
        self.assertFalse(MODULE._forbidden_fields({"mean_f1": 0.5, "case_count": 100}))
        public = MODULE._compact_generator_identity(
            {
                "model_id": "model",
                "revision": "revision",
                "config_sha256": "a" * 64,
                "resolution": "hf_cache",
                "model_path": "/private/collaborator/path",
            }
        )
        self.assertNotIn("model_path", public)

    def test_summary_detects_distributed_selector_responsiveness(self) -> None:
        top_50 = [candidate(index, positive=index < 5) for index in range(5)]
        cases = []
        for _ in range(100):
            cases.append(
                {
                    "top_50": top_50,
                    "arms": [
                        {
                            "arm_id": "qore_common_order",
                            "metrics": {"em": 0.0, "f1": 0.0},
                            "evidence_counts": {"direct": 0, "broad": 0},
                            "generation_ms": 1.0,
                        },
                        {
                            "arm_id": "topk_common_order",
                            "metrics": {"em": 0.0, "f1": 0.2},
                            "evidence_counts": {"direct": 1, "broad": 1},
                            "generation_ms": 1.0,
                        },
                        {
                            "arm_id": "silver_oracle_common_order",
                            "metrics": {"em": 1.0, "f1": 1.0},
                            "evidence_counts": {"direct": 5, "broad": 5},
                            "generation_ms": 1.0,
                        },
                    ],
                }
            )
        phase = MODULE._load_yaml(
            ROOT / "configs/experiments/silver_oracle_top5_100.yaml"
        )["phase"]
        result = MODULE.summarize(cases, contract={"status": "valid"}, phase=phase)
        self.assertTrue(result["gate"]["pass"])
        self.assertEqual(result["decision"], "selector_responsive_ceiling_confirmed")
        self.assertFalse(MODULE._forbidden_fields(result))

    def test_plan_declares_oracle_as_non_deployable(self) -> None:
        plan = json.loads(
            (ROOT / "configs/experiments/silver_oracle_top5_100_plan.json").read_text(encoding="utf-8")
        )
        self.assertEqual(plan["schema_version"], "research-plugin-architecture.plugin-plan.v1")
        oracle = next(row for row in plan["plugins"] if row["id"] == "silver_oracle_membership")
        self.assertIn("production_selector_registration", oracle["conflicts"])
        self.assertIn("observability_contract", oracle)


if __name__ == "__main__":
    unittest.main()
