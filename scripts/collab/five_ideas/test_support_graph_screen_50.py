"""Contract tests for the fresh-50 support-graph screen."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = Path(__file__).with_name("run_support_graph_screen_50.py")
SPEC = importlib.util.spec_from_file_location("support_graph_screen_50", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SupportGraphScreen50Tests(unittest.TestCase):
    def test_validate_only_contract_is_frozen(self) -> None:
        result = MODULE.validate_contract(
            ROOT / "configs/experiments/support_graph_screen_50.yaml",
            ROOT / "configs/experiments/support_graph_screen_50_plan.json",
        )
        self.assertEqual(result["status"], "valid")
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["wiki_dpr_started"])
        self.assertEqual(result["selected_global_indices"][0], 3150)
        self.assertEqual(result["selected_global_indices"][-1], 3199)
        self.assertEqual(result["selectors"], ["qore_as", "topk_as", "anchor_conditioned_support_graph", "support_disabled_control"])

    def test_plugin_plan_is_valid_json_and_has_v2_recovery_contract(self) -> None:
        plan = json.loads((ROOT / "configs/experiments/support_graph_screen_50_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["schema_version"], "research-plugin-architecture.plugin-plan.v2")
        self.assertEqual(plan["recovery_contract"]["candidate_id"], "anchor_conditioned_support_graph")
        self.assertIn("interaction_ablation", plan["validation"]["tests"])

    def test_control_parity_is_explicit_in_summary(self) -> None:
        cases = []
        for case_number in range(1, 51):
            selectors = []
            for selector_id in MODULE.EXPECTED_SELECTOR_IDS:
                selectors.append({
                    "selector_id": selector_id,
                    "selected_top_5": [{"id": f"p-{index}"} for index in range(5)],
                    "metrics": {"em": 0.0, "f1": 0.0},
                    "diagnostics": {"answer_has_match_in_selected_text": False},
                    "timing_ms": {"selection_and_generation": 1.0},
                })
            cases.append({
                "top_50_diagnostics": {"answer_has_match_in_text": False},
                "selectors": selectors,
            })
        summary = MODULE._screen_summary(cases, contract={"status": "valid"})
        self.assertTrue(summary["control_parity"]["support_disabled_matches_topk_order"])
        self.assertEqual(summary["decision"], "stop_candidate_after_fresh_screen")


if __name__ == "__main__":
    unittest.main()
