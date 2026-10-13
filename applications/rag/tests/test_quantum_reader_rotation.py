import unittest
import torch

from applications.rag.quantum_reader_rotation import (RotationConfig, make_rotation_head,
    reader_rotation_hook, stable_top5, top5_retention_objective)


def fixture_target():
    roles = ["held"] * 50
    roles[4], roles[7] = "qualified_replaceable_irrelevant", "qualified_missed_direct"
    return {"future_role": "training", "roles": roles, "baseline_top5_indices": [0, 1, 2, 3, 4],
            "immutable_selected_indices": [0, 1, 2, 3], "replaceable_indices": [4],
            "eligible_positive_indices": [7], "draft_pairs": [[7, 4]], "pair_weights": [1.0],
            "maximum_replacement_slots": 1, "missing_selected_direct_qualification": False,
            "scope_qualified": True}


def dense_oracle(head, cls, pooled):
    """Independent full-matrix gate construction from bit meanings, not schedules."""
    import math
    projected = math.sqrt(5 * head.hidden_size) * (torch.nn.functional.normalize(pooled, dim=-1) @ head.projection.T)
    x = projected / torch.sqrt(1 + projected.square())
    angles = torch.cat((torch.ones_like(x[:, :1]), x), 1) @ head.theta.T
    output = torch.nn.functional.pad(cls, (0, head.state_size - head.hidden_size))
    states = [output / cls.norm(dim=1, keepdim=True)]
    for j, (_, control, target) in enumerate(head.descriptors):
        rows = []
        for sample in range(len(cls)):
            gate = torch.eye(head.state_size, dtype=cls.dtype)
            c, s = math.cos(float(angles[sample, j]) / 2), math.sin(float(angles[sample, j]) / 2)
            for a in range(head.state_size):
                if (a >> target) & 1 or (control is not None and not ((a >> control) & 1)):
                    continue
                b = a ^ (1 << target)
                gate[a, a], gate[a, b], gate[b, a], gate[b, b] = c, -s, s, c
            rows.append(gate @ output[sample])
        output = torch.stack(rows)
        states.append(output / cls.norm(dim=1, keepdim=True))
    return output, torch.stack(states, 1)


class RotationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(13)
        self.cls = torch.randn(50, 6, generator=generator, dtype=torch.float64)
        self.pooled = torch.randn(50, 30, generator=generator, dtype=torch.float64)
        self.w = torch.randn(6, generator=generator, dtype=torch.float64)

    def head(self, family="quantum"):
        return make_rotation_head(self.w, torch.tensor(.2, dtype=torch.float64), family=family)

    def test_null_norm_and_signed_readout(self):
        for family in ("quantum", "classical"):
            head = self.head(family)
            result = head(self.cls, self.pooled)
            torch.testing.assert_close(result["rotated"][:, :6], self.cls, rtol=0, atol=0)
            torch.testing.assert_close(result["scores"], self.cls @ self.w + .2)
            torch.testing.assert_close(result["states"].norm(dim=-1), torch.ones(50, 13, dtype=torch.float64))
            with torch.no_grad(): head.theta[:, 0].fill_(.27)
            result = head(self.cls, self.pooled)
            torch.testing.assert_close(result["rotated"].norm(dim=-1), self.cls.norm(dim=-1))
            padded_weight = torch.nn.functional.pad(head.weight, (0, 2))
            overlap = (result["states"][:, -1] * padded_weight).sum(-1) * result["norm"][:, 0] + head.bias
            torch.testing.assert_close(result["scores"], overlap)
            self.assertTrue(bool((result["scores"] < 0).any()))

    def test_quantum_gates_independent_dense_oracle(self):
        head = self.head()
        with torch.no_grad(): head.theta.copy_(torch.linspace(-.35, .4, head.theta.numel()).reshape_as(head.theta))
        result = head(self.cls[:3], self.pooled[:3])
        expected, states = dense_oracle(head, self.cls[:3], self.pooled[:3])
        torch.testing.assert_close(result["rotated"], expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(result["states"], states, rtol=1e-12, atol=1e-12)

    def test_matched_budget_support_and_determinism(self):
        q, c = self.head(), self.head("classical")
        self.assertEqual(q.theta.numel(), c.theta.numel())
        torch.testing.assert_close(q.projection, c.projection, rtol=0, atol=0)
        for j in range(len(q.descriptors)):
            self.assertEqual(len(getattr(q, f"low_{j}")), len(getattr(c, f"low_{j}")))
        for key, value in c.state_dict().items():
            torch.testing.assert_close(value, self.head("classical").state_dict()[key], rtol=0, atol=0)
        real = make_rotation_head(torch.ones(768), torch.tensor(0.))
        self.assertEqual(real.theta.numel(), 200)
        self.assertEqual(real.qubits, 10)
        low = make_rotation_head(torch.ones(768), torch.tensor(0.), config=RotationConfig(layers=1))
        self.assertEqual(low.theta.numel(), 100)

    def test_finite_nonuniform_training_at_null(self):
        for family in ("quantum", "classical"):
            head = self.head(family)
            base = head(self.cls, self.pooled)["scores"].detach()
            optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=0)
            parts = top5_retention_objective(head(self.cls, self.pooled)["scores"], base, fixture_target())
            parts["total"].backward()
            self.assertTrue(bool(head.theta.grad.isfinite().all()))
            self.assertGreater(float(head.theta.grad.norm()), 0)
            optimizer.step()
            delta = head(self.cls, self.pooled)["scores"].detach() - base
            self.assertGreater(float(delta.std()), 1e-7)
            self.assertLess(float(head(self.cls, self.pooled)["conditioning"].abs().max()), 1)

    def test_retention_membership_not_relative_order(self):
        scores = torch.full((50,), -10.)
        scores[:5] = torch.tensor([8., 7., 6., 5., 0.])
        scores[7] = 9.
        row = fixture_target()
        result = top5_retention_objective(scores, scores, row)
        self.assertEqual(float(result["protection"]), 0.)
        self.assertEqual(set(stable_top5(scores)), {0, 1, 2, 3, 7})
        scores[8] = 10.
        self.assertGreater(float(top5_retention_objective(scores, scores, row)["protection"]), 0)
        row["future_role"] = "evaluation"
        with self.assertRaises(ValueError): top5_retention_objective(scores, scores, row)

    def test_fail_closed_and_no_global_rng_mutation(self):
        state = torch.random.get_rng_state().clone()
        head = self.head()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        with self.assertRaises(ValueError): head(self.cls[:, :5], self.pooled)
        with self.assertRaises(ValueError): head(self.cls * 0, self.pooled)
        with self.assertRaises(ValueError): head(self.cls, self.pooled * float("nan"))
        with self.assertRaises(ValueError): make_rotation_head(self.w, .2, family="unregistered")
        self.assertEqual(stable_top5(torch.zeros(50)), [0, 1, 2, 3, 4])

    def test_scoped_hook_changes_only_classifier_and_cleans_on_error(self):
        class Encoder(torch.nn.Module):
            def forward(self, hidden): return (hidden,)
        class Span(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.qa_classifier = torch.nn.Linear(6, 1).double()
        class Reader(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.span_predictor = Span()
            def forward(self, hidden):
                h = self.span_predictor.encoder(hidden)[0]
                return self.span_predictor.qa_classifier(h[:, 0]), h[:, :, 1]
        reader = Reader().eval().requires_grad_(False)
        head = make_rotation_head(reader.span_predictor.qa_classifier.weight, reader.span_predictor.qa_classifier.bias)
        hidden = self.cls[:2, None, :].repeat(1, 4, 1)
        qm = torch.tensor([[False, True, False, False]] * 2)
        pm = torch.tensor([[False, False, True, True]] * 2)
        base, spans = reader(hidden)
        capture = {}
        with reader_rotation_hook(reader, head, qm, pm, capture):
            scores, new_spans = reader(hidden)
        torch.testing.assert_close(scores, base, rtol=0, atol=0)
        torch.testing.assert_close(spans, new_spans, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            with reader_rotation_hook(reader, head, qm, pm, capture): raise RuntimeError("fixture")
        self.assertEqual(len(reader.span_predictor.encoder._forward_hooks), 0)
        self.assertEqual(len(reader.span_predictor.qa_classifier._forward_pre_hooks), 0)


if __name__ == "__main__": unittest.main()
