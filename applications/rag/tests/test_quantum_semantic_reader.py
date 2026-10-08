import tempfile
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from applications.rag.quantum_semantic_reader import make_semantic_head, reader_semantic_forward, semantic_pool
from applications.rag.qarcg_reader import QARCGConfig
from scripts.collab.five_ideas.run_quantum_semantic_reader_screen_100 import CONFIG, diagnose, produce_fixture, validate_config, validate_trace_file


class SemanticReaderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.hidden = torch.randn(50, 12, 8)
        self.qm = torch.zeros(50, 12, dtype=torch.bool); self.qm[:, 1:4] = True
        self.pm = torch.zeros_like(self.qm); self.pm[:, 4:11] = True
        self.pooled, self.weights = semantic_pool(self.hidden, self.qm, self.pm)
        self.base = torch.linspace(3, -3, 50)

    def test_null_and_budget(self):
        q, c = make_semantic_head(8), make_semantic_head(8, classical=True)
        self.assertEqual(sum(p.numel() for p in q.parameters()), sum(p.numel() for p in c.parameters()))
        for key in q.project.state_dict():
            torch.testing.assert_close(q.project.state_dict()[key], c.project.state_dict()[key], rtol=0, atol=0)
        for head in (q, c):
            torch.testing.assert_close(head(self.base, self.pooled)[0], self.base, rtol=0, atol=0)

    def test_question_conditioned_pool_and_masks(self):
        self.assertEqual(tuple(self.pooled.shape), (50, 40))
        self.assertTrue(bool((self.weights[~self.pm] == 0).all()))
        torch.testing.assert_close(self.weights.sum(1), torch.ones(50))
        changed = self.hidden.clone(); changed[:, 1:4] *= -1
        self.assertFalse(torch.equal(semantic_pool(changed, self.qm, self.pm)[0], self.pooled))
        with self.assertRaises(ValueError):
            semantic_pool(self.hidden, self.qm, self.qm)

    def test_bounds_gradients_and_permutation(self):
        cfg = QARCGConfig()
        q = make_semantic_head(8, config=cfg)
        with torch.no_grad():
            q.interaction.residual_scale.fill_(0.3)
        out = q(self.base, self.pooled)
        self.assertLessEqual(float(out[2].abs().max()), cfg.residual_bound * float(self.base.std(unbiased=False)) + 1e-6)
        out[0].square().sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in q.parameters()))
        self.assertGreater(float(q.project.weight.grad.abs().sum()), 0)
        perm = torch.randperm(50)
        torch.testing.assert_close(q(self.base[perm], self.pooled[perm])[0], out[0][perm])

    def test_fixture_reopened_and_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = produce_fixture(Path(directory), validate_config(CONFIG))
            self.assertTrue(all(v == "pass" for v in result.values()))
            with np.load(Path(directory) / "semantic_values.npz", allow_pickle=False) as values:
                self.assertIn("evaluation_001__hidden", values)
                self.assertIn("checkpoint__quantum_semantic__initial__project.weight", values)
                self.assertIn("evaluation_001__quantum_scalar_control__encoded", values)

    def test_corrupt_trace_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cfg = validate_config(CONFIG)
            produce_fixture(path, cfg)
            trace_path = path / "fixture_trace.json"
            trace = json.loads(trace_path.read_text())
            trace["cases"][0]["methods"]["quantum_semantic"]["selected_retrieved_ranks"] = [46, 47, 48, 49, 50]
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            fixture_cfg = {**cfg, "evaluation": {**cfg["evaluation"], "cases": 1}}
            with self.assertRaises(ValueError):
                validate_trace_file(trace_path, path / "semantic_values.npz", fixture_cfg)

    def test_saturated_projection_gradients_finite(self):
        head = make_semantic_head(8)
        with torch.no_grad():
            head.project.bias.fill_(100)
            head.interaction.residual_scale.fill_(0.3)
        head(self.base, self.pooled)[0].sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))

    def test_actual_dpr_api_without_checkpoint_download(self):
        from transformers import DPRConfig, DPRReader
        reader = DPRReader(DPRConfig(vocab_size=30, hidden_size=8, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=16)).eval()
        class Tokenizer:
            all_special_ids = [0, 1, 2]
            def __call__(self, questions, texts, **kwargs):
                n = len(texts)
                return {"input_ids": torch.tensor([[1, 3, 4, 2, 5, 6, 7, 2]] * n),
                        "attention_mask": torch.ones(n, 8, dtype=torch.long),
                        "token_type_ids": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * n)}
        before = len(reader.span_predictor.encoder._forward_hooks)
        base, pooled, scalar, capture = reader_semantic_forward(torch, Tokenizer(), reader, "cpu", "fixture", ["fixture"] * 50, batch_size=8, capture_first=True)
        self.assertEqual(tuple(base.shape), (50,))
        self.assertEqual(tuple(pooled.shape), (50, 40))
        self.assertEqual(tuple(scalar.shape), (50, 4))
        self.assertEqual(len(reader.span_predictor.encoder._forward_hooks), before)
        self.assertEqual(tuple(capture["hidden"].shape), (8, 8))

    def test_filler_overlap_does_not_replace_primary(self):
        cfg = validate_config(CONFIG)
        cfg = {**cfg, "evaluation": {**cfg["evaluation"], "cases": 1}}
        candidates = [{"id": str(i), "retrieved_rank": i + 1,
                       "evidence": {"positive_consensus": i < 3, "direct_consensus": i < 2, "positive_all_models": i < 2}} for i in range(50)]
        case = {"top_50": candidates, "selectors": [{"selector_id": "silver_oracle_common_order", "selected_top_5": candidates[:5]}]}
        scores = list(range(50, 0, -1))
        q_scores = scores.copy(); q_scores[5] = 46.1
        trace = [{"methods": {m: {"score_vector": q_scores if m == "quantum_semantic" else scores} for m in cfg["methods"]}}]
        result = diagnose([case], trace, cfg)
        methods = result["methods"]
        self.assertEqual(methods["quantum_semantic"]["fixed_silver_set_overlap"]["total"], 4)
        self.assertEqual(methods["quantum_semantic"]["positive_consensus_count"]["total"], 3)
        self.assertFalse(result["gates"]["selection_diagnostic"])


if __name__ == "__main__":
    unittest.main()
