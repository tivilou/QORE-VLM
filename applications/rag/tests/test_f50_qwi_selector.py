"""Mechanism and boundary tests for the F50-QWI selector."""

from __future__ import annotations

import unittest

import numpy as np

from applications.rag.f50_qwi_selector import (
    F50QWIConfig,
    F50QWIError,
    FULL_TOP50,
    INVALID_STATES,
    REGISTER_DIMENSION,
    build_hamiltonian,
    mechanism_audit,
    select,
)


def _fixture() -> list[dict[str, object]]:
    rows = []
    for index in range(FULL_TOP50):
        vector = [
            float(np.sin((index + 1) * (dimension + 1) * 0.17) + (1.0 if dimension == index % 9 else 0.0))
            for dimension in range(12)
        ]
        rows.append({
            "id": f"passage-{index:02d}",
            "retrieved_rank": index + 1,
            "retrieval_score": float(100 - index + (index % 3) * 0.01),
            "answer_scorer_score": float((index * 11) % 47) / 47.0,
            "dpr_passage_embedding": vector,
        })
    return rows


class F50QWISelectorTests(unittest.TestCase):
    def test_full_register_is_hermitian_and_invalid_states_are_isolated(self) -> None:
        hamiltonian, _, _, _, _ = build_hamiltonian(_fixture())
        self.assertEqual(hamiltonian.shape, (REGISTER_DIMENSION, REGISTER_DIMENSION))
        self.assertEqual(REGISTER_DIMENSION - FULL_TOP50, INVALID_STATES)
        self.assertLessEqual(float(np.max(np.abs(hamiltonian - hamiltonian.conj().T))), 1.0e-12)
        self.assertEqual(float(np.max(np.abs(hamiltonian[:FULL_TOP50, FULL_TOP50:]))), 0.0)
        self.assertEqual(float(np.max(np.abs(hamiltonian[FULL_TOP50:, :FULL_TOP50]))), 0.0)

    def test_born_distribution_is_normalized_and_deterministic(self) -> None:
        first = select(_fixture(), variant="born_exact")
        second = select(_fixture(), variant="born_exact")
        self.assertEqual(first.selected_indices, second.selected_indices)
        self.assertEqual(first.probabilities, second.probabilities)
        self.assertEqual(len(first.selected_indices), 5)
        self.assertEqual(len(set(first.selected_indices)), 5)
        self.assertAlmostEqual(sum(first.probabilities), 1.0, places=12)
        self.assertLessEqual(first.diagnostics["invalid_state_probability_mass"], 1.0e-12)

    def test_tau_zero_is_the_initial_population_identity(self) -> None:
        result = select(_fixture(), config=F50QWIConfig(evolution_time=0.0))
        self.assertTrue(np.allclose(result.probabilities, result.initial_weights, atol=1.0e-12))

    def test_mechanism_audit_runs_all_controls_without_labels(self) -> None:
        audit = mechanism_audit(_fixture())
        self.assertTrue(audit.passed)
        self.assertEqual(set(audit.variants), {
            "born_exact", "real_diffusion_control", "dephased_control", "phase_scramble_control",
        })
        self.assertTrue(audit.checks["tau_zero_identity"])
        self.assertTrue(audit.checks["no_label_input"])

    def test_online_boundary_rejects_evidence_and_wrong_top50_size(self) -> None:
        with_evidence = _fixture()
        with_evidence[0]["evidence"] = {"positive_consensus": True}
        with self.assertRaisesRegex(F50QWIError, "forbidden"):
            select(with_evidence)
        with self.assertRaisesRegex(F50QWIError, "exactly 50"):
            select(_fixture()[:-1])
        wrong_rank = _fixture()
        wrong_rank[-1]["retrieved_rank"] = 51
        with self.assertRaisesRegex(F50QWIError, "ordered sequence"):
            select(wrong_rank)

    def test_ties_break_by_rank_then_passage_identifier(self) -> None:
        rows = _fixture()
        for row in rows:
            row["retrieval_score"] = 1.0
            row["answer_scorer_score"] = 1.0
            row["dpr_passage_embedding"] = [1.0] + [0.0] * 11
        result = select(rows, config=F50QWIConfig(interaction_strength=0.0, evolution_time=0.0))
        self.assertEqual(result.selected_indices, (0, 1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
