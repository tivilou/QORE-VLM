"""Contract tests: target isolation, detached replay and required-upload failure."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

from applications.rag import protected_boundary_training as training
from applications.rag.quantum_semantic_reader import semantic_pool
from scripts.collab.five_ideas import run_protected_boundary_reader_screen_100 as runner
from scripts.collab.lib import protected_boundary_trace as trace


def target():
    roles = ["held_not_negative"] * 50
    roles[7] = "qualified_missed_direct"; roles[4] = "qualified_replaceable_irrelevant"
    return {"future_role": "training", "roles": roles, "baseline_top5_indices": [0, 1, 2, 3, 4],
            "draft_pairs": [[7, 4]], "pair_weights": [1.0], "immutable_selected_indices": [0, 1, 2, 3],
            "eligible_positive_indices": [7], "replaceable_indices": [4], "maximum_replacement_slots": 1,
            "scope_qualified": True, "missing_selected_direct_qualification": []}


def fixture_data():
    cfg = runner.validate_config(runner.CONFIG)
    cfg = copy.deepcopy(cfg); cfg["training"]["epochs"] = 1
    arrays, features, metadata, cases = {}, [], [], []
    generator = torch.Generator().manual_seed(441)
    for lane in ("training", "evaluation"):
        for i in (1, 2):
            hidden = torch.randn(50, 6, 8, generator=generator)
            qm = torch.zeros(50, 6, dtype=torch.bool); qm[:, 1:3] = True
            pm = torch.zeros(50, 6, dtype=torch.bool); pm[:, 3:5] = True
            pooled, weights = semantic_pool(hidden, qm, pm)
            prefix = f"{lane}_{i:03d}"
            base = torch.arange(50., 0., -1)
            f = {"base": base, "pooled": pooled, "scalar": torch.zeros(50, 4)}
            for k, v in f.items(): arrays[prefix + "__" + k] = v.numpy().copy()
            for k, v in {"hidden": hidden[0], "input_ids": torch.arange(6), "question_mask": qm[0],
                         "passage_mask": pm[0], "pooling_weights": weights[0],
                         "relevance_weight": torch.zeros(1,8), "relevance_bias": torch.tensor([50.])}.items():
                arrays[prefix + "__" + k] = v.numpy().copy()
            features.append(f); metadata.append({"lane": lane, "prefix": prefix, "question_sha256": str(i) + lane})
            cases.append({"question": lane + str(i), "answers": ["fixture"], "top_50": [
                {"id": str(j), "retrieved_rank": j + 1, "title": "fixture", "text": "synthetic body"} for j in range(50)]})
    heads = training.make_heads(torch, 8, cfg, torch.device("cpu"))
    reports, witnesses = training.train_heads(torch, cfg, heads, features[:2], [target(), target()], arrays, torch.device("cpu"), health_checks=False)
    ranked = {lane: runner.rank_cases(torch, heads, features[o:o+2], cases[o:o+2], lane, cfg, arrays)
              for lane, o in (("training", 0), ("evaluation", 2))}
    return cfg, arrays, features, metadata, cases, heads, reports, witnesses, ranked


class BoundaryRunnerTests(unittest.TestCase):
    def test_config_rejects_changed_budget(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "cfg.json"
            cfg = runner.validate_config(runner.CONFIG); cfg["training"]["epochs"] = 99
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ValueError): runner.validate_config(path)

    def test_real_data_forward_receives_only_question_and_body(self):
        case = {"question": "q", "gold_answers": ["poison"], "top_50": [
            {"id": str(i), "retrieved_rank": i+1, "title": "Title", "text": "Body", "evidence": {"poison": True}} for i in range(50)]}
        def forward(*args, **kwargs):
            self.assertEqual(args[4], "q"); self.assertEqual(args[5], ["Title. Body"] * 50)
            return torch.zeros(50), torch.zeros(50, 40), torch.zeros(50, 4), None
        with patch.object(runner, "reader_semantic_forward", side_effect=forward):
            runner.feature_case(torch, None, None, "cpu", runner.validate_config(runner.CONFIG), case, "evaluation_001", False, {})

    def test_online_selection_has_no_qualified_mask(self):
        scores = torch.arange(50., 0., -1); scores[20] = 100
        case = {"top_50": [{"id": str(i), "retrieved_rank": i+1} for i in range(50)], "question": "q", "draft_pairs": [[7,4]]}
        arrays = {}
        # No intervention heads: baseline still chooses held index20 by its score.
        with patch.object(runner, "METHODS", ("frozen_reader_topk",)):
            result = runner.rank_cases(torch, {}, [{"base": scores}], [case], "evaluation", {}, arrays)
        self.assertEqual(result[0]["methods"]["frozen_reader_topk"]["selected_indices"][0], 20)

    def test_training_update_and_artifact_roundtrip(self):
        cfg, arrays, _, metadata, cases, heads, _, witnesses, ranked = fixture_data()
        with tempfile.TemporaryDirectory() as root:
            provenance = {"code_revision": "fixture", "config_sha256": "a"*64, "reader": {"model_id": "synthetic_hidden_unit_test"}}
            result = trace.persist_trace(root, cfg, heads, arrays, metadata, witnesses, [target(), target()], cases, ranked, "cpu", provenance, synthetic=True)
            self.assertEqual(result["optimizer_update_replays"], 4)
            document = json.loads((Path(root) / "sample_trace.json").read_text(encoding="utf-8"))
            values_path = Path(root) / "semantic_values.npz"
            self.assertEqual(trace.read_coordinate(values_path, result["values_sha256"], "training_001__pooled", [49,39]), arrays["training_001__pooled"][49,39].item())
            with self.assertRaises(ValueError): trace.read_coordinate(values_path, result["values_sha256"], "training_001__pooled", [50,39])
            for sample in document["samples"]:
                stage = next(s for s in sample["stages"] if s["stage"] == "selection")
                self.assertNotIn("posthoc", json.dumps(stage))
            with np.load(values_path, allow_pickle=False) as archive:
                corrupted = copy.deepcopy(ranked)
                corrupted["evaluation"][0]["methods"]["quantum_semantic"]["selected_indices"][0] = 49
                with self.assertRaises(ValueError): trace.verify_trace(document, archive, metadata, corrupted)

    def test_parameter_budget_and_fresh_exact_null(self):
        cfg = runner.validate_config(runner.CONFIG)
        heads = training.make_heads(torch, 8, cfg, "cpu")
        base = torch.arange(50., 0., -1); pooled = torch.randn(50, 40)
        for h in heads.values(): torch.testing.assert_close(h(base, pooled)[0], base, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in heads[training.ARMS[0]].parameters()), sum(p.numel() for p in heads[training.ARMS[1]].parameters()))

    def test_required_upload_failure_keeps_recovery_manifest(self):
        args = types.SimpleNamespace(upload=True, exchange_url="http://example.invalid", token_env="EXAMPLE_TOKEN")
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "run1"; run_dir.mkdir()
            (run_dir / "summary.json").write_text('{"status":"ok"}', encoding="utf-8")
            (run_dir / "semantic_values.npz").write_bytes(b"fixture")
            with patch.object(runner, "upload_manifest", side_effect=RuntimeError("network failure")):
                with self.assertRaises(RuntimeError): runner.publish(run_dir, runner.validate_config(runner.CONFIG), args)
            manifest = json.loads((run_dir / "upload_manifest.json").read_text())
            self.assertIn("semantic_values.npz", manifest["exchange_only"])
            self.assertNotIn("semantic_values.npz", manifest["github_files"])
            self.assertFalse((run_dir / "upload_receipts.json").exists())


if __name__ == "__main__": unittest.main()
