import unittest

import numpy as np

from applications.rag.qarcg_reader import (
    N_QUBITS,
    QARCGConfig,
    QARCGError,
    build_reader_features,
    classical_matched_observables,
    initial_parameters,
    make_torch_classical_control,
    make_torch_dpr_reader_adapter,
    make_torch_qarcg,
    parameter_count,
    quantum_observables,
    rank_indices,
    score_classical_control,
    score_qarcg,
    torch_pairwise_rank_loss,
    torch_reader_features,
)


def _reader_rows(count: int = 8) -> list[dict[str, float]]:
    return [
        {
            "relevance_logit": float(count - index),
            "span_logit": float((index * 3) % 7),
            "span_margin": float(index % 4) / 3.0,
            "start_entropy": float(index % 5) / 5.0,
            "end_entropy": float((index + 2) % 6) / 6.0,
        }
        for index in range(count)
    ]


class QARCGReaderTests(unittest.TestCase):
    def test_reader_projection_is_finite_and_bounded(self) -> None:
        features = build_reader_features(_reader_rows())
        self.assertEqual(features.shape, (8, N_QUBITS))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertLessEqual(float(np.max(np.abs(features))), 1.0)

    def test_null_initialization_is_exact_reader_baseline(self) -> None:
        rng = np.random.default_rng(42)
        features = rng.uniform(-1.0, 1.0, size=(11, N_QUBITS))
        base = rng.normal(size=11)
        config = QARCGConfig(depth=2)
        params = initial_parameters(config)
        quantum = score_qarcg(base, features, params, config)
        classical = score_classical_control(base, features, params, config)
        self.assertTrue(np.array_equal(quantum["scores"], base))
        self.assertTrue(np.array_equal(classical["scores"], base))
        self.assertTrue(np.all(quantum["residual"] == 0.0))

    def test_quantum_and_classical_arms_have_matched_budget(self) -> None:
        config = QARCGConfig(depth=3)
        params = initial_parameters(config)
        self.assertEqual(parameter_count(config), 8 * config.depth + 19)
        self.assertEqual(params.circuit.shape, (config.depth, N_QUBITS, 2))
        features = build_reader_features(_reader_rows())
        self.assertEqual(quantum_observables(features, params, config).shape, (8, 8))
        self.assertEqual(classical_matched_observables(features, params, config).shape, (8, 8))

    def test_nonzero_residual_is_bounded_and_gate_is_applicability_weight(self) -> None:
        features = build_reader_features(_reader_rows())
        base = np.linspace(-2.0, 2.0, len(features))
        params = initial_parameters()
        params.residual_scale = 2.0
        params.gate_weights[:] = 0.5
        params.residual_weights[:] = -0.25
        result = score_qarcg(base, features, params)
        self.assertTrue(np.all((result["gate"] >= 0.0) & (result["gate"] <= 1.0)))
        self.assertLessEqual(float(np.max(np.abs(result["residual"]))), 0.25 * float(base.std()) + 1e-12)
        self.assertTrue(np.any(np.abs(result["scores"] - base) > 1e-12))

    def test_stable_top_k_ties_and_invalid_shape_fail_closed(self) -> None:
        self.assertEqual(rank_indices([1.0, 1.0, 0.0], k=2, retrieved_ranks=[2, 1, 3], candidate_ids=["b", "a", "c"]), [1, 0])
        with self.assertRaises(QARCGError):
            score_qarcg(np.zeros(3), np.zeros((3, 3)))

    def test_torch_head_has_gradients_and_pairwise_loss(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        module = make_torch_qarcg(QARCGConfig(depth=1))
        control = make_torch_classical_control(QARCGConfig(depth=1))
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()),
                         sum(parameter.numel() for parameter in control.parameters()))
        base = torch.linspace(-1.0, 1.0, 6)
        features = torch.tensor(build_reader_features(_reader_rows(6)), dtype=torch.float32)
        scores, gate, residual, _ = module(base, features)
        self.assertTrue(torch.equal(scores, base))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        module.residual_scale.data.fill_(0.1)
        scores, _, _, _ = module(base, features)
        loss = torch_pairwise_rank_loss(scores, torch.tensor([True, True, False, False, False, False]))
        loss.backward()
        self.assertIsNotNone(module.residual_scale.grad)
        self.assertTrue(torch.isfinite(module.residual_scale.grad))
        self.assertIsNotNone(module.circuit.grad)
        self.assertTrue(bool(torch.isfinite(module.circuit.grad).all()))
        self.assertEqual(tuple(gate.shape), (6,))

    def test_torch_statevector_matches_numpy_statevector(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        config = QARCGConfig(depth=1)
        params = initial_parameters(config)
        params.circuit[0, :, 0] = [0.1, -0.2, 0.15, 0.05]
        params.circuit[0, :, 1] = [-0.15, 0.12, 0.03, -0.08]
        features = build_reader_features(_reader_rows())
        expected = quantum_observables(features, params, config)
        module = make_torch_qarcg(config)
        with torch.no_grad():
            module.circuit.copy_(torch.tensor(params.circuit, dtype=torch.float32))
            actual = module._observables(torch.tensor(features, dtype=torch.float32)).cpu().numpy()
        self.assertTrue(np.allclose(actual, expected, atol=2e-6, rtol=2e-6))

    def test_torch_dpr_reader_adapter_keeps_reader_outputs_and_adds_scores(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        class TinyReader(torch.nn.Module):
            def forward(self, input_ids, attention_mask, return_dict=True):
                batch, length = input_ids.shape
                relevance = input_ids.float().mean(dim=1)
                start = input_ids.float() * 0.01
                end = torch.flip(input_ids.float(), dims=[1]) * 0.01
                return type("ReaderOutput", (), {
                    "relevance_logits": relevance,
                    "start_logits": start,
                    "end_logits": end,
                })()

        batch, length = 6, 8
        input_ids = torch.arange(batch * length).reshape(batch, length)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        adapter = make_torch_dpr_reader_adapter(TinyReader(), QARCGConfig(depth=1))
        output = adapter(input_ids=input_ids, attention_mask=mask, passage_token_mask=mask)
        self.assertIn("reader_outputs", output)
        self.assertEqual(tuple(output["scores"].shape), (batch,))
        self.assertEqual(tuple(output["bounded_residual"].shape), (batch,))
        self.assertTrue(torch.equal(output["scores"], output["reader_outputs"].relevance_logits))

    def test_torch_reader_adapter_derives_span_features_without_labels(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        relevance = torch.linspace(-1.0, 1.0, 6)
        start = torch.zeros((6, 8), dtype=torch.float32)
        end = torch.zeros((6, 8), dtype=torch.float32)
        start[:, 2] = torch.linspace(0.0, 1.0, 6)
        end[:, 4] = torch.linspace(1.0, 0.0, 6)
        passage_mask = torch.zeros((6, 8), dtype=torch.bool)
        passage_mask[:, 1:7] = True
        features, summary = torch_reader_features(relevance, start, end, passage_mask)
        self.assertEqual(tuple(features.shape), (6, 4))
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertEqual(tuple(summary["best_span"].shape), (6,))
        control = make_torch_classical_control(QARCGConfig(depth=1))
        scores, _, residual, _ = control(relevance, features)
        self.assertTrue(torch.equal(scores, relevance))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))


if __name__ == "__main__":
    unittest.main()
