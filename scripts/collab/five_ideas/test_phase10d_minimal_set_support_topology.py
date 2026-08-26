#!/usr/bin/env python3
"""Synthetic contract tests for Phase 10D minimal-set support topology."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.collab.five_ideas.phase10d_metrics import Phase10DError, summarize_support_topology
from scripts.collab.five_ideas.run_phase10d_minimal_set_support_topology import (
    Phase10DConfigError,
    _subset_outcomes,
    validate_contract,
)


GATE = {
    "minimum_primary_errors": 1,
    "minimum_copy_control_successes": 1,
    "strict_interaction_fraction_min": 0.40,
    "strict_interaction_bootstrap_ci95_low_min": 0.20,
    "bootstrap_repetitions": 100,
    "bootstrap_seed": 10601,
}


def arm(em: float | None = 0.0, *, attempted: bool = True) -> dict[str, bool | float | None]:
    if not attempted:
        return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}
    assert em is not None
    return {"attempted": True, "em": float(em), "f1": float(em), "generation_time_ms": 1.0}


def subset_outcomes(*, answer_indices: set[int], successful_masks: set[int]) -> list[dict[str, bool | float | int]]:
    rows: list[dict[str, bool | float | int]] = []
    for mask in range(1, 32):
        indices = [index for index in range(5) if mask & (1 << index)]
        cardinality = len(indices)
        matches = sum(index in answer_indices for index in indices)
        rows.append({
            "mask": mask,
            "cardinality": cardinality,
            "answer_match_count": matches,
            "contains_non_answer_literal": matches < cardinality,
            "em": 0.0 if mask == 31 else float(mask in successful_masks),
            "f1": 0.0 if mask == 31 else float(mask in successful_masks),
            "generation_time_ms": 1.0,
            "reused_frozen_baseline": mask == 31,
        })
    return rows


def row(index: int, category: str) -> dict[str, object]:
    answer_indices = {0}
    successes: set[int] = set()
    if category == "singleton_sufficient":
        successes = {1}
    elif category == "strict_non_answer_assisted_interaction":
        successes = {3}
    elif category == "answer_bearing_only_distributed_interaction":
        answer_indices = {0, 1}
        successes = {3}
    elif category == "beyond_selected_set":
        pass
    elif category == "copy_failure":
        return {
            "question_id": f"nq_{index}",
            "retrieval_hit": True,
            "selected_hit": True,
            "selected_answer_match_count": 1,
            "selection_time_ms": 1.0,
            "arms": {"baseline": arm(0), "gold_answer_copy": arm(0)},
            "subset_outcomes": [],
        }
    else:
        raise ValueError(category)
    return {
        "question_id": f"nq_{index}",
        "retrieval_hit": True,
        "selected_hit": True,
        "selected_answer_match_count": len(answer_indices),
        "selection_time_ms": 1.0,
        "arms": {"baseline": arm(0), "gold_answer_copy": arm(1)},
        "subset_outcomes": subset_outcomes(answer_indices=answer_indices, successful_masks=successes),
    }


class Phase10DMetricTests(unittest.TestCase):
    def test_strict_interaction_is_distinct_from_answer_bearing_only_interaction(self):
        payload = {
            "stage": "screen",
            "samples": [
                row(1, "strict_non_answer_assisted_interaction"),
                row(2, "answer_bearing_only_distributed_interaction"),
                row(3, "singleton_sufficient"),
                row(4, "beyond_selected_set"),
            ],
        }
        summary = summarize_support_topology(payload, gate=GATE, stage="screen")
        counts = summary["topology"]["counts"]
        self.assertEqual(counts["strict_non_answer_assisted_interaction"], 1)
        self.assertEqual(counts["answer_bearing_only_distributed_interaction"], 1)
        self.assertEqual(counts["singleton_sufficient"], 1)
        self.assertEqual(counts["beyond_selected_set"], 1)
        self.assertEqual(sum(counts.values()), 4)
        self.assertTrue(summary["decision"]["report_only"])

    def test_strict_gate_passes_when_all_eligible_questions_show_strict_interaction(self):
        payload = {"stage": "screen", "samples": [row(10, "strict_non_answer_assisted_interaction") for _ in range(3)]}
        for index, sample in enumerate(payload["samples"]):
            sample["question_id"] = f"nq_{index}"
        summary = summarize_support_topology(payload, gate=GATE, stage="screen")
        self.assertEqual(summary["decision"]["primary_failure_class"], "pass_phase10d_screen")
        self.assertTrue(summary["gate"]["strict_interaction_bootstrap_ci95_low"])

    def test_copy_failure_is_excluded_from_subset_topology(self):
        summary = summarize_support_topology(
            {"stage": "screen", "samples": [row(20, "copy_failure")]},
            gate=GATE,
            stage="screen",
        )
        self.assertEqual(summary["cohorts"]["copy_control_successes"], 0)
        self.assertEqual(summary["decision"]["primary_failure_class"], "insufficient_copy_control_successes")

    def test_full_mask_must_reuse_exact_baseline_outcome(self):
        sample = row(30, "beyond_selected_set")
        sample["subset_outcomes"][-1]["generation_time_ms"] = 2.0
        with self.assertRaises(Phase10DError):
            summarize_support_topology({"samples": [sample]}, gate=GATE, stage="screen")

    def test_raw_content_field_is_rejected(self):
        payload = {"samples": [row(40, "beyond_selected_set")], "prediction": "forbidden"}
        with self.assertRaises(Phase10DError):
            summarize_support_topology(payload, gate=GATE, stage="screen")


class Phase10DRunnerTests(unittest.TestCase):
    def test_all_masks_preserve_original_order_and_reuse_full_baseline(self):
        class Generator:
            def __init__(self):
                self.calls = []

            def generate(self, question, passages):
                self.calls.append((question, list(passages)))
                return "wrong"

        generator = Generator()
        selected = ["zero", "one", "two", "three", "four"]
        outcomes = _subset_outcomes(
            generator,
            "which",
            selected,
            ["answer"],
            [True, False, False, False, False],
            arm(0),
        )
        self.assertEqual([item["mask"] for item in outcomes], list(range(1, 32)))
        self.assertEqual(len(generator.calls), 30)
        self.assertEqual(generator.calls[0][1], ["zero"])
        self.assertEqual(generator.calls[2][1], ["zero", "one"])
        self.assertTrue(outcomes[-1]["reused_frozen_baseline"])
        self.assertEqual(outcomes[-1]["em"], 0.0)

    def test_repository_contract_validates_without_model_or_wiki_dpr(self):
        root = Path(__file__).resolve().parents[3]
        contract = validate_contract(
            root / "configs/experiments/phase10d_minimal_set_support_topology.yaml",
            root / "configs/experiments/phase10d_minimal_set_support_topology_plan.json",
            root / "configs/experiments/phase10d_minimal_set_support_topology_mechanism_recovery.json",
            root / "configs/experiments/phase10d_minimal_set_support_topology_evidence_packet.json",
        )
        self.assertEqual(contract["status"], "valid")
        self.assertFalse(contract["model_loaded"])
        self.assertFalse(contract["wiki_dpr_started"])

    def test_replication_is_not_executable_without_new_authorization(self):
        root = Path(__file__).resolve().parents[3]
        with self.assertRaises(Phase10DConfigError):
            validate_contract(
                root / "configs/experiments/phase10d_minimal_set_support_topology.yaml",
                root / "configs/experiments/phase10d_minimal_set_support_topology_plan.json",
                root / "configs/experiments/phase10d_minimal_set_support_topology_mechanism_recovery.json",
                root / "configs/experiments/phase10d_minimal_set_support_topology_evidence_packet.json",
                stage="replication",
            )

    def test_mutated_selected_k_contract_is_rejected(self):
        root = Path(__file__).resolve().parents[3]
        source = root / "configs/experiments/phase10d_minimal_set_support_topology.yaml"
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["phase"]["support_topology"]["selected_K"] = 4
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "config.yaml"
            mutated.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(Phase10DConfigError):
                validate_contract(
                    mutated,
                    root / "configs/experiments/phase10d_minimal_set_support_topology_plan.json",
                    root / "configs/experiments/phase10d_minimal_set_support_topology_mechanism_recovery.json",
                    root / "configs/experiments/phase10d_minimal_set_support_topology_evidence_packet.json",
                )


if __name__ == "__main__":
    unittest.main()
