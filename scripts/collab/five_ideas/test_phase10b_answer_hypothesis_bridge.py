import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.collab.five_ideas.phase10b_metrics import summarize_phase10b
from scripts.collab.five_ideas.run_phase10b_answer_hypothesis_bridge import (
    Phase10BConfigError,
    validate_contract,
)


class Phase10BContractTests(unittest.TestCase):
    def test_compact_metrics_gate_arithmetic(self):
        def arm(name):
            samples = []
            for index in range(10):
                applied = name == "consensus_gated_bridge" and index < 2
                samples.append({
                    "question_id": f"q{index}", "em": 0.4 + (0.2 if applied else 0.0),
                    "f1": 0.5 + (0.2 if applied else 0.0),
                    "generation_time_ms": 100.0, "pipeline_time_ms": 100.0 if not applied else 120.0,
                    "initial_retrieval_hit": True, "bridge_retrieval_hit": applied,
                    "selected_hit": True if applied else False, "bridge_applied": applied,
                    "final_context_K": 5,
                })
            return {"samples": samples}
        result = {"arms": {"baseline_frozen_query": arm("baseline"), "always_bridge": arm("always_bridge"), "consensus_gated_bridge": arm("consensus_gated_bridge")}}
        summary = summarize_phase10b(result, gate={"bootstrap_repetitions": 100, "bootstrap_seed": 4, "selected_hit_gain_min_on_applied": 0.5, "total_pipeline_cost_ratio_max": 1.5})
        self.assertEqual(summary["decision"], "pass_phase10b_screen")

    def test_repository_contract_validates(self):
        root = Path(__file__).resolve().parents[3]
        result = validate_contract(
            root / "configs/experiments/phase10b_answer_hypothesis_bridge.yaml",
            root / "configs/experiments/phase10b_answer_hypothesis_bridge_plan.json",
        )
        self.assertEqual(result["status"], "valid")
        self.assertFalse(result["wiki_dpr_started"])

    def test_non_observation_authorization_is_rejected(self):
        root = Path(__file__).resolve().parents[3]
        config = yaml.safe_load((root / "configs/experiments/phase10b_answer_hypothesis_bridge.yaml").read_text())
        config["phase"]["authorization"] = "planned_observation_only"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(Phase10BConfigError):
                validate_contract(path, root / "configs/experiments/phase10b_answer_hypothesis_bridge_plan.json")

    def test_plan_allowlist_must_match_order(self):
        root = Path(__file__).resolve().parents[3]
        plan = json.loads((root / "configs/experiments/phase10b_answer_hypothesis_bridge_plan.json").read_text())
        plan["discovery"]["allowlist"] = list(reversed(plan["discovery"]["allowlist"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaises(Phase10BConfigError):
                validate_contract(root / "configs/experiments/phase10b_answer_hypothesis_bridge.yaml", path)


if __name__ == "__main__":
    unittest.main()
