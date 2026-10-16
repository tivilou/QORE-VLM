"""Counterexamples that distinguish objective validity from falling raw loss."""
import copy
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from applications.rag.scale_invariant_top5_objective import (
    standardized_scores, standardized_top5_objective, membership_diagnostics,
)
from applications.rag.tests.test_quantum_reader_rotation_screen_100 import row
from applications.rag.protected_boundary_learning import boundary_objective
from applications.rag.quantum_reader_rotation import top5_retention_objective, make_rotation_head
from scripts.collab.five_ideas import run_quantum_reader_rotation_screen_100 as runner
from scripts.collab.lib.source_snapshot import archive_sources, verify_sources


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.base = torch.linspace(5., -5., 50, dtype=torch.float64)

    def test_objective_and_gradient_are_affine_invariant(self):
        source = self.base.clone()
        source[5] += .7
        original = source.clone().requires_grad_(True)
        reference = standardized_top5_objective(original, self.base, row())
        reference["total"].backward()
        for alpha, shift in ((.01, 7.), (.1, 0.), (.5, -9.), (2., 3.)):
            changed = (source * alpha + shift).requires_grad_(True)
            parts = standardized_top5_objective(changed, self.base, row())
            parts["total"].backward()
            for field in reference:
                torch.testing.assert_close(parts[field], reference[field], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(changed.grad * alpha, original.grad, rtol=1e-9, atol=1e-9)

    def test_reject_collapse_nonfinite_validation_and_bad_margin(self):
        for bad in (torch.ones_like(self.base), self.base * 1e-12, self.base * float("nan")):
            with self.assertRaises(ValueError): standardized_top5_objective(bad, self.base, row())
        invalid = dict(row(), future_role="validation")
        with self.assertRaises(ValueError): standardized_top5_objective(self.base, self.base, invalid)
        with self.assertRaises(ValueError): standardized_top5_objective(self.base, self.base, row(), margin=0.)

    def test_actual_rescue_and_retention_counterexamples(self):
        original = standardized_top5_objective(self.base, self.base, row())
        rescued = self.base.clone(); rescued[5] = (rescued[3] + rescued[4]) / 2
        good = standardized_top5_objective(rescued, self.base, row())
        self.assertLess(good["rescue"], original["rescue"])
        self.assertEqual(membership_diagnostics(rescued, row())["rescued_eligible"], 1)
        self.assertEqual(membership_diagnostics(rescued, row())["lost_protected"], 0)
        harmed = rescued.clone(); harmed[0] = -6.
        bad = standardized_top5_objective(harmed, self.base, row())
        self.assertEqual(membership_diagnostics(harmed, row())["lost_protected"], 1)
        self.assertGreater(bad["protection"], good["protection"])
        # Equal-to-boundary scores are penalized by a positive margin; a tie
        # is not accepted as evidence of a successful boundary crossing.
        tied = self.base.clone(); tied[5] = tied[4]
        self.assertGreater(standardized_top5_objective(tied, self.base, row())["rescue"], good["rescue"])

    def test_more_positives_than_slots_uses_only_feasible_capacity(self):
        target = copy.deepcopy(row())
        target["eligible_positive_indices"] = [5, 6, 7]
        target["draft_pairs"] = [[p, 4] for p in (5, 6, 7)]
        target["pair_weights"] = [1 / 3] * 3
        for p in (5, 6, 7): target["roles"][p] = "qualified_missed_direct"
        parts = standardized_top5_objective(self.base, self.base, target)
        # Lower-ranked additional alternatives are not all forced into one slot.
        expected = standardized_top5_objective(self.base, self.base, row())
        torch.testing.assert_close(parts["rescue"], expected["rescue"])

    def test_single_extreme_cannot_buy_loss_reduction(self):
        before = standardized_top5_objective(self.base, self.base, row())
        extreme = self.base.clone()
        extreme[-1] -= 100 * self.base.std(unbiased=False)
        after = standardized_top5_objective(extreme, self.base, row())
        self.assertGreater(after["total"], before["total"])
        # The actual boundary terms stay unchanged; only anchor harm increases.
        torch.testing.assert_close(after["rescue"], before["rescue"])
        torch.testing.assert_close(after["protection"], before["protection"])

    def test_quantum_and_classical_heads_receive_finite_gradients_on_cpu(self):
        g = torch.Generator().manual_seed(16)
        cls = torch.randn(50, 8, generator=g, dtype=torch.float64)
        pooled = torch.randn(50, 40, generator=g, dtype=torch.float64)
        weight = torch.randn(8, generator=g, dtype=torch.float64)
        for family in ("quantum", "classical"):
            head = make_rotation_head(weight, weight.new_tensor(0.), family=family)
            scores = head(cls, pooled)["scores"]
            target = row()
            # This synthetic row checks differentiability only, not semantic
            # qualification or correspondence to real Reader rankings.
            standardized_top5_objective(scores, scores.detach(), target)["total"].backward()
            self.assertTrue(bool(head.theta.grad.isfinite().all()))
            self.assertGreater(float(head.theta.grad.norm()), 0.)


class DeliveryTests(unittest.TestCase):
    def test_runtime_config_is_archived_and_outside_paths_rejected(self):
        relative = "configs/experiments/reader_top5_objective_audit.json"
        self.assertIn(relative, runner.runtime_sources(runner.ROOT / relative))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): runner.runtime_sources(Path(tmp) / "override.json")

    def test_losses_are_crossed_independently_of_head_family(self):
        base = torch.linspace(5., -5., 50, dtype=torch.float64)
        scores = base.clone(); scores[5] = 9.
        cfg = runner.validate_config(runner.CONFIG)
        target = row()
        for method in runner.METHODS[1:]:
            old = method.endswith("_old_loss")
            expected_fn = boundary_objective if old else top5_retention_objective
            expected = expected_fn(scores, base, target, **cfg["losses"]["old" if old else "new"])
            actual = runner.objective(method, scores, base, target, cfg)
            torch.testing.assert_close(actual["total"], expected["total"])
        self.assertNotEqual(float(runner.objective("new_quantum_old_loss", scores, base, target, cfg)["total"]),
                            float(runner.objective("new_quantum_new_loss", scores, base, target, cfg)["total"]))
        with self.assertRaises(ValueError): runner.arm_kind("anything_else_new_loss")

    def test_real_training_wrapper_is_unwrapped(self):
        torch.manual_seed(7)
        weight = torch.randn(8, dtype=torch.float64)
        cls = torch.randn(50, 8, dtype=torch.float64)
        pooled = torch.randn(50, 40, dtype=torch.float64)
        head = make_rotation_head(weight, weight.new_tensor(0.))
        base = head(cls, pooled)["scores"].detach()
        cfg = copy.deepcopy(runner.validate_config(runner.CONFIG))
        cfg["training"]["epochs"] = 1
        report, witness = runner.train_arm(torch, "new_quantum_new_loss", head,
            [{"base": base, "cls": cls, "pooled": pooled}], [{"case": {}, "target": row(), "annotation": {}}], cfg, "cpu", {})
        self.assertEqual(report["epochs"][0]["updates"], 1)
        self.assertEqual(len(witness), 1)

    def test_source_snapshot_reopens_exact_bytes_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "source.zip"
            snapshot = archive_sources(runner.ROOT, runner.SOURCES, archive)
            verify_sources(runner.ROOT, snapshot)
            with zipfile.ZipFile(archive) as z:
                for name in runner.SOURCES:
                    self.assertEqual(z.read(name), (runner.ROOT / name).read_bytes())
            bad = copy.deepcopy(snapshot)
            bad["source_hashes"][runner.SOURCES[0]] = "0" * 64
            with self.assertRaises(ValueError): verify_sources(runner.ROOT, bad)
        self.assertEqual(len(runner.validate_config(runner.CONFIG)["training"]["input"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
