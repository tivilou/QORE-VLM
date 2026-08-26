import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.collab.five_ideas.run_phase10b_answer_hypothesis_bridge import (
    Phase10BConfigError,
    validate_contract,
)


class Phase10BContractTests(unittest.TestCase):
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
        config["phase"]["authorization"] = "implemented_observation_only"
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
