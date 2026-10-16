"""Synthetic delivery/causal checks; no Reader, Wiki-DPR or real fitting."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from applications.rag import reader_membership_training as learning
from scripts.collab.five_ideas import run_quantum_reader_membership_screen_100 as runner
from scripts.collab.five_ideas.validate_reader_membership_trace import validate, structural


def fixture_inputs():
    cfg = runner.config()
    cfg["reader"]["hidden_size"] = 8
    cfg["training"].update(cases=2, pairs=2, eligible_candidates=2, protected_candidates=8, maximum_slots=2)
    cfg["evaluation"]["cases"] = 2
    cfg["evaluation"]["bootstrap_replicates"] = 50
    generator = torch.Generator().manual_seed(20261016)
    weight, bias = torch.zeros(8), torch.tensor(0.)
    weight[0] = 1.
    records, features, targets, train_rows, eval_cases = [], [], [], [], []
    for lane in ("training", "evaluation"):
        for number in (1, 2):
            question = f"synthetic {lane} question {number}"
            qid = hashlib.sha256(question.encode()).hexdigest()
            candidates = [{"id": f"{lane}-{number}-{i}", "title": "synthetic", "text": f"synthetic passage {i}", "retrieved_rank": i + 1,
                "evidence": {"positive_consensus": i in (0, 1, 2, 3, 5), "direct_consensus": i in (0, 1, 5), "positive_all_models": i in (0, 1, 5)}} for i in range(50)]
            cls = torch.randn(50, 8, generator=generator)
            cls[:, 0] = torch.linspace(5., -5., 50)
            pooled = torch.randn(50, 40, generator=generator); pooled[:, :8] = cls
            base = cls @ weight + bias
            feature = {"base": base, "cls": cls, "pooled": pooled}
            features.append(feature)
            record = {"lane": lane, "number": number, "prefix": f"{lane}_{number:03d}", "question_sha256": qid,
                "candidate_id_sha256": [hashlib.sha256(c["id"].encode()).hexdigest() for c in candidates],
                "candidate_retrieved_ranks": list(range(1, 51)), "baseline_top5_indices": [0, 1, 2, 3, 4]}
            records.append(record)
            case = {"question": question, "gold_answers": ["synthetic answer"], "top_50": candidates,
                "selectors": [{"selector_id": "silver_oracle_common_order", "selected_top_5": [candidates[i] for i in (0, 1, 2, 3, 5)]}]}
            if lane == "evaluation": eval_cases.append(case); continue
            roles = ["held_not_negative"] * 50
            roles[5], roles[4] = "qualified_missed_direct", "qualified_replaceable_irrelevant"
            target = {"future_role": "training", "roles": roles, "baseline_top5_indices": [0, 1, 2, 3, 4],
                "draft_pairs": [[5, 4]], "pair_weights": [1.], "immutable_selected_indices": [0, 1, 2, 3],
                "eligible_positive_indices": [5], "replaceable_indices": [4], "maximum_replacement_slots": 1,
                "missing_selected_direct_qualification": [], "scope_qualified": True}
            targets.append(target); train_rows.append({"case": case, "target": target})
    return cfg, {"records": records}, features, weight, bias, targets, train_rows, eval_cases


def write_fixture(output):
    cfg, index, features, weight, bias, targets, rows, cases = fixture_inputs()
    runner.bind_cases(index["records"], rows, cases, cfg)
    summary = runner.execute(cfg, index, features, weight, bias, targets, cases, output, synthetic=True)
    runner.write(Path(output) / "fixture_inputs.private.json", {"config": cfg, "index": index, "targets": targets})
    np.savez_compressed(Path(output) / "fixture_input_values.npz", weight=weight.numpy(), bias=bias.numpy(),
        **{f"{i}__{k}": v.numpy() for i, f in enumerate(features) for k, v in f.items()})
    return summary


class MembershipScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "fixture"
        cls.inputs = fixture_inputs()
        cfg, index, features, weight, bias, targets, _, cases = cls.inputs
        cls.summary = runner.execute(cfg, index, features, weight, bias, targets, cases, cls.path, synthetic=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_full_trace_and_native_updates_reopen(self):
        checks = self.summary["readiness"]["checks"]
        self.assertEqual(checks["optimizer_update_replays"], 24)
        self.assertEqual(checks["head_score_replays"], 20)
        self.assertEqual(checks["full_output_replays"], 16)
        self.assertEqual(checks["selected_input_slices"], 12)
        self.assertEqual(self.summary["readiness"]["real_training_updates"], 0)
        self.assertTrue(self.summary["synthetic_only"])

    def test_explicit_loss_dispatch_and_matched_capacity(self):
        cfg, _, features, weight, bias, targets, _, _ = self.inputs
        heads = learning.make_heads(weight, bias, cfg)
        scores = features[0]["base"].clone(); scores[5] = 9.
        q = learning.objective("quantum_membership", scores, features[0]["base"], targets[0], cfg)
        c = learning.objective("classical_membership", scores, features[0]["base"], targets[0], cfg)
        old = learning.objective("quantum_retention_control", scores, features[0]["base"], targets[0], cfg)
        torch.testing.assert_close(q["total"], c["total"])
        self.assertNotAlmostEqual(float(q["total"]), float(old["total"]))
        self.assertEqual(len({h.theta.numel() for h in heads.values()}), 1)
        real = learning.make_heads(torch.ones(768), torch.tensor(0.), cfg)
        self.assertEqual({h.theta.numel() for h in real.values()}, {200})

    def test_online_selection_is_unmasked_and_training_rejects_validation(self):
        cfg, _, features, _, _, targets, _, _ = self.inputs
        scores = features[0]["base"].clone(); scores[49] = 100.
        self.assertIn(49, learning.top5(scores, list(range(1, 51))))
        target = dict(targets[0], future_role="validation")
        with self.assertRaises(ValueError): learning.objective("quantum_membership", scores, scores, target, cfg)

    def test_candidate_identity_and_order_are_bound(self):
        cfg, index, _, _, _, _, rows, cases = copy.deepcopy(self.inputs)
        runner.bind_cases(index["records"], rows, cases, cfg)
        cases[0]["top_50"][0], cases[0]["top_50"][1] = cases[0]["top_50"][1], cases[0]["top_50"][0]
        with self.assertRaises(ValueError): runner.bind_cases(index["records"], rows, cases, cfg)

    def test_target_aware_stage_and_forced_selection_rejected(self):
        cfg, index, *_ = self.inputs
        trace = json.loads((self.path / "selector_trace.json").read_text())
        trace["samples"][0]["stages"][3]["input_refs"].append("train_target:leak")
        with self.assertRaises(ValueError): structural(trace, index, cfg)
        trace = json.loads((self.path / "selector_trace.json").read_text())
        trace["cases"][0]["methods"]["frozen_reader_topk"]["selected_indices"] = [0, 1, 2, 3, 49]
        with self.assertRaises(ValueError): structural(trace, index, cfg)

    def corrupted(self, kind):
        cfg, index, features, weight, bias, targets, *_ = self.inputs
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "copy"
            shutil.copytree(self.path, directory)
            trace = json.loads((directory / "selector_trace.json").read_text())
            with np.load(directory / "membership_values.npz", allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            if kind == "gradient": key = trace["training_witnesses"][0]["raw_gradient"]
            else: key = trace["training_witnesses"][1]["optimizer_before"]["state"]["0"]["exp_avg"]
            arrays[key].flat[0] += 1.
            # Internally self-consistent hashes must NOT bypass actual replay.
            trace["values"] = learning.describe_values(arrays)
            np.savez_compressed(directory / "membership_values.npz", **arrays)
            trace["values_sha256"] = runner.digest(directory / "membership_values.npz")
            runner.write(directory / "selector_trace.json", trace)
            with self.assertRaises((ValueError, AssertionError)):
                validate(directory, cfg, index, features, weight, bias, targets)

    def test_internally_rehashed_wrong_gradient_is_rejected(self): self.corrupted("gradient")
    def test_internally_rehashed_wrong_adam_state_is_rejected(self): self.corrupted("optimizer")

    def test_bad_input_hash_fails_before_download_or_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"; p.write_text("{}")
            with patch.object(runner, "download_exchange_file") as download:
                with self.assertRaises(ValueError): runner.obtain(p, {"bytes": 2, "sha256": "0" * 64}, "five_ideas/test/bad.json", SimpleNamespace())
                download.assert_not_called()

    def test_feature_slices_are_lazy_and_preserve_every_coordinate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            source = self.inputs[2]
            np.savez_compressed(path, **{f"q{i}__{k}": v.numpy() for i, f in enumerate(source) for k, v in f.items()})
            with patch.object(runner.np, "load", wraps=np.load) as load:
                store = runner.FeatureStore(path, [f"q{i}" for i in range(len(source))])
                subset = store[:2]
                load.assert_not_called()
                self.assertIsInstance(subset, runner.FeatureStore)
                self.assertEqual(len(subset), 2)
                for expected, actual in zip(source, store):
                    for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                self.assertEqual(load.call_count, len(source))

    def test_upload_failure_keeps_pending_manifest_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ("summary.json", "readiness.json", "run_metadata.json"): (p / name).write_text("{}")
            (p / "report.md").write_text("synthetic")
            args = SimpleNamespace(preflight_only=False, upload=True, exchange_url="http://unused.invalid", token_env="UNUSED")
            with patch.object(runner, "upload_manifest", side_effect=RuntimeError("synthetic failed receipt")):
                with self.assertRaises(RuntimeError): runner.publish(p, self.inputs[0], args)
            self.assertEqual(json.loads((p / "upload_manifest.json").read_text())["upload_status"], "pending_until_receipts")
            self.assertFalse((p / "upload_receipts.json").exists())

    def test_preflight_performs_zero_updates(self):
        cfg, index, features, weight, bias, targets, _, cases = self.inputs
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(learning, "train_arm", side_effect=AssertionError("must not train")):
                summary = runner.execute(cfg, index, features, weight, bias, targets, cases, Path(tmp) / "null", preflight=True, synthetic=True)
            self.assertIsNone(summary["evaluation"])
            self.assertEqual(summary["readiness"]["checks"]["optimizer_update_replays"], 0)

    def test_pending_preflight_blocks_formal_run_before_input_or_training(self):
        with patch.object(runner, "PLAN") as plan, patch.object(runner, "config", return_value=self.inputs[0]), patch.object(runner, "obtain") as obtain:
            plan.read_text.return_value = '{"authorization": "approved"}'
            with self.assertRaisesRegex(ValueError, "Real training is gated"):
                runner.run(SimpleNamespace(validate_only=False, preflight_only=False))
            obtain.assert_not_called()


if __name__ == "__main__": unittest.main()
