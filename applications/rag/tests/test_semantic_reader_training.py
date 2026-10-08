"""Optimizer-repair, training-health, moment replay and historical-null tests."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from applications.rag.semantic_reader_training import (
    LEGACY, REPAIRED, SCALED, TrainingHealthError, gradient_health, load_optimizer,
    make_optimizer, norm, require_learning, save_optimizer, signal_health,
)
from applications.rag.quantum_semantic_reader import make_semantic_head
from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen


class TrainingRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.cfg = screen.validate_config(screen.CONFIG)
        torch.manual_seed(12)
        self.case = {"base": torch.linspace(3, -3, 50), "pooled": torch.randn(50, 40),
                     "scalar": torch.randn(50, 4).tanh(), "weak_mask": torch.arange(50) % 7 == 0}

    def test_exact_null_and_blocked_then_live_data_gradients(self):
        for classical in (False, True):
            h = make_semantic_head(8, classical=classical)
            o = make_optimizer(torch, h, self.cfg["training"], SCALED)
            out = h(self.case["base"], self.case["pooled"])
            torch.testing.assert_close(out[0], self.case["base"], rtol=0, atol=0)
            screen.torch_pairwise_rank_loss(out[0], self.case["weak_mask"]).backward()
            first = gradient_health(h)
            self.assertEqual(first["projection"], 0)
            self.assertGreater(first["scale"], 0)
            o.step(); o.zero_grad(set_to_none=True)
            screen.torch_pairwise_rank_loss(h(self.case["base"], self.case["pooled"])[0], self.case["weak_mask"]).backward()
            self.assertGreater(gradient_health(h)["projection"], 0)

    def test_groups_and_fanin_are_dimension_only(self):
        for classical in (False, True):
            h = make_semantic_head(768, classical=classical)
            opt = make_optimizer(torch, h, self.cfg["training"], SCALED)
            self.assertEqual(opt.param_groups[0]["parameter_names"], ["project.weight"])
            self.assertAlmostEqual(opt.param_groups[0]["lr"], .003 / np.sqrt(3840))
            self.assertEqual(opt.param_groups[1]["weight_decay"], 0)
            flattened = [n for g in opt.param_groups for n in g["parameter_names"]]
            self.assertEqual(sorted(flattened), sorted(n for n, _ in h.named_parameters()))
            self.assertEqual(len(flattened), len(set(flattened)))
        with self.assertRaises(ValueError):
            make_optimizer(torch, h, self.cfg["training"], "unknown")

    def test_zero_task_decay_does_not_destroy_projection_or_gate(self):
        norms = {}
        for policy in (LEGACY, SCALED):
            h = make_semantic_head(8)
            initial = norm(h.project.weight)
            opt = make_optimizer(torch, h, self.cfg["training"], policy)
            for _ in range(434):
                for p in h.parameters():
                    p.grad = torch.zeros_like(p)
                opt.step()
            norms[policy] = norm(h.project.weight) / initial
            if policy == SCALED:
                self.assertEqual(float(h.interaction.gate_bias), -4)
        self.assertLess(norms[LEGACY], 1e-8)
        self.assertGreater(norms[SCALED], .99)

    def test_repaired_fixture_reopens_optimizer_moments(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["training"]["optimizer_policy"] = SCALED
        with tempfile.TemporaryDirectory() as d:
            result = screen.produce_fixture(Path(d), cfg)
            self.assertEqual(result["optimizer_witness_replay"], "pass")
            import json
            trace = json.loads((Path(d) / "fixture_trace.json").read_text())
            self.assertEqual(trace["training"]["quantum_semantic"]["training_health"]["status"], "deferred_below_64_updates")

    def test_optimizer_corruption_is_detected(self):
        h = make_semantic_head(8)
        opt = make_optimizer(torch, h, self.cfg["training"], SCALED)
        for p in h.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        arrays = {}; save_optimizer(opt, h, arrays, "o")
        arrays.pop("o__project.weight__exp_avg_sq")
        with self.assertRaises(ValueError):
            load_optimizer(torch, opt, h, arrays, "o")

    def test_uniform_offset_is_not_effective_learning(self):
        h = make_semantic_head(8)
        output = list(h(self.case["base"], self.case["pooled"]))
        output[0] = self.case["base"] + .001
        measured = signal_health(h, [output], [self.case["base"]], norm(h.project.weight))
        with self.assertRaisesRegex(ValueError, "input-dependent"):
            require_learning(measured, {"projection": 1., "readout": 1., "scale": 1.})

    def test_failed_gate_persists_checkpoint_without_evaluation(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(output_root=Path(d), upload=False)
            cfg = {**self.cfg, "output_namespace": "five_ideas/semantic_reader_training_repair_100"}
            failure = TrainingHealthError("quantum_semantic", 64, {}, {}, "synthetic collapse")
            path = screen.save_training_failure(args, cfg, failure, {"checkpoint": np.zeros(3)}, [], [])
            saved = json.loads((path / "training_failure.json").read_text())
            self.assertFalse(saved["evaluation_started"])
            self.assertTrue((path / "training_failure_values.npz").is_file())

    def test_legacy_optimizer_remains_original(self):
        h = make_semantic_head(8)
        opt = make_optimizer(torch, h, self.cfg["training"], LEGACY)
        self.assertIs(type(opt), torch.optim.Adam)
        self.assertEqual(opt.param_groups[0]["weight_decay"], .0001)

    def test_learning_gate_is_executed_at_64_before_evaluation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["training"]["optimizer_policy"] = SCALED
        cases = [{**self.case, "pooled": torch.zeros_like(self.case["pooled"])} for _ in range(64)]
        h = make_semantic_head(8)
        arrays = {}
        with self.assertRaises(TrainingHealthError) as raised:
            screen.train_head(torch, "quantum_semantic", h, cases, cfg, "cpu", arrays)
        self.assertEqual(raised.exception.record["step"], 64)
        self.assertIn("checkpoint__quantum_semantic__health_failure__project.weight", arrays)
        self.assertEqual(arrays["health__quantum_semantic__data_gradient_norms"].shape, (64, 4))

    def test_live_training_gets_epoch_health_and_moment_records(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["training"]["optimizer_policy"] = SCALED
        cases = [self.case] * 64
        h = make_semantic_head(8)
        arrays = {}
        result = screen.train_head(torch, "quantum_semantic", h, cases, cfg, "cpu", arrays)
        self.assertEqual(result["training_health"]["status"], "passed")
        self.assertEqual([r["step"] for r in result["training_health"]["checks"]], [64, 128, 192])
        self.assertIn("optimizer__quantum_semantic__epoch3__project.weight__exp_avg_sq", arrays)

    def test_frozen_repair_entry_checks_qualification_and_source(self):
        from scripts.collab.five_ideas.run_semantic_reader_training_repair_100 import validate_repair
        self.assertEqual(validate_repair()["training"]["optimizer_policy"], SCALED)


if __name__ == "__main__":
    unittest.main()
