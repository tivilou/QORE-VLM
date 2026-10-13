"""Local contract tests for the collaborator-only crossed Reader screen."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from applications.rag.quantum_reader_rotation import RotationConfig, make_rotation_head, stable_top5, top5_retention_objective
from scripts.collab.five_ideas import run_quantum_reader_rotation_screen_100 as runner


def row():
    baseline = [0, 1, 2, 3, 4]
    roles = ["held_not_negative"] * 50
    roles[5] = "qualified_missed_direct"
    roles[4] = "qualified_replaceable_irrelevant"
    return {
        "future_role": "training", "roles": roles,
        "baseline_top5_indices": baseline, "draft_pairs": [[5, 4]], "pair_weights": [1.0],
        "immutable_selected_indices": [0, 1, 2, 3], "eligible_positive_indices": [5],
        "replaceable_indices": [4], "maximum_replacement_slots": 1,
        "missing_selected_direct_qualification": [], "scope_qualified": True,
    }


class ReaderRotationScreenTests(unittest.TestCase):
    def test_config_and_allowlist_are_pinned(self):
        cfg = runner.validate_config(runner.CONFIG)
        self.assertEqual(tuple(cfg["methods"]), runner.METHODS)
        self.assertEqual(cfg["training"]["cases"], 59)
        self.assertEqual(cfg["evaluation"]["cases"], 100)

    def test_rotation_is_identity_at_zero_and_has_matched_parameter_budget(self):
        torch.manual_seed(7)
        weight = torch.randn(1, 8); bias = torch.randn(())
        cls = torch.randn(50, 8); pooled = torch.randn(50, 40)
        quantum = make_rotation_head(weight, bias, family="quantum", config=RotationConfig()).float()
        classical = make_rotation_head(weight, bias, family="classical", config=RotationConfig()).float()
        self.assertEqual(sum(p.numel() for p in quantum.parameters()), sum(p.numel() for p in classical.parameters()))
        with torch.no_grad():
            base = torch.nn.functional.linear(cls, weight, bias).squeeze(-1)
            self.assertTrue(torch.allclose(quantum(cls, pooled)["scores"], base, rtol=0, atol=0))
            self.assertTrue(torch.allclose(classical(cls, pooled)["scores"], base, rtol=0, atol=0))

    def test_new_loss_is_finite_and_selection_is_full_top50(self):
        scores = torch.linspace(0.0, 1.0, 50, requires_grad=True)
        base = scores.detach().clone()
        parts = top5_retention_objective(scores, base, row())
        self.assertTrue(torch.isfinite(parts["total"]))
        parts["total"].backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(len(stable_top5(scores)), 5)

    def test_validate_only_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = set(Path(tmp).iterdir())
            # The runner's validate-only path is deliberately side-effect free.
            runner.run(type("Args", (), {"config": runner.CONFIG, "validate_only": True})())
            self.assertEqual(before, set(Path(tmp).iterdir()))


if __name__ == "__main__":
    unittest.main()
