#!/usr/bin/env python3
"""Synthetic contracts for the Phase 9I observation-only diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Exercise prompt and compact-statistics contracts without loading a model or
# the surrounding RAG application.
if "phase9h_probe" not in sys.modules:
    phase9h_stub = types.ModuleType("phase9h_probe")

    @dataclass(frozen=True)
    class ProbeResult:
        prediction: str
        generation_time_ms: float

    phase9h_stub.ProbeResult = ProbeResult
    phase9h_stub.build_extractive_prompt = lambda tokenizer, question, passages: "extractive"
    phase9h_stub.generate_from_prompt = lambda generator, prompt: ProbeResult("", 0.0)
    sys.modules["phase9h_probe"] = phase9h_stub

from phase9i_metrics import Phase9IError, summarize_phase9i, validate_result
from phase9i_probe import (
    ABSTENTION_TOKEN,
    CandidateProbeResult,
    build_evidence_constrained_prompt,
    build_verifier_prompt,
    candidate_pairs,
    candidate_parse_status,
    fixed_candidate_permutation,
    parse_candidate_id,
)


GATE = {
    "formal": {
        "bootstrap_repetitions": 100,
        "bootstrap_seed": 9901,
        "candidate_oracle_em_uplift_min": 0.05,
        "candidate_oracle_ci95_low_min": 0.0,
        "verifier_em_uplift_min": 0.03,
        "verifier_ci95_low_min": 0.0,
        "verifier_oracle_gain_recovery_min": 0.60,
        "verifier_f1_loss_max": 0.01,
    }
}


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


def standard_arm(em: float, f1: float | None = None) -> dict[str, object]:
    return {
        "attempted": True,
        "em": em,
        "f1": em if f1 is None else f1,
        "generation_time_ms": 1.0,
    }


def candidate_arm(em: float | None, status: str = "ok") -> dict[str, object]:
    scored = status in {"ok", "abstain"}
    return {
        "attempted": True,
        "parse_status": status,
        "em": em if scored else None,
        "f1": em if scored else None,
        "generation_time_ms": 1.0 if scored else None,
    }


def sample(index: int, *, verifier_em: float | None = 1.0) -> dict[str, object]:
    verifier_valid = verifier_em is not None
    return {
        "question_id": f"nq_{index}",
        "retrieval_hit": True,
        "selected_hit": True,
        "retrieval_match_count": 2,
        "selected_match_count": 1,
        "primary_error": True,
        "copy_control_success": True,
        "eligible": True,
        "selection_time_ms": 1.0,
        "arms": {
            "baseline": standard_arm(0.0),
            "gold_answer_copy": standard_arm(1.0),
            "extractive_span": candidate_arm(1.0),
            "evidence_constrained": candidate_arm(0.0),
        },
        "candidate_set": {
            "attempted": True,
            "candidate_count": 3,
            "unique_candidate_count": 2,
            "parse_failures": 0,
            "oracle_em": 1.0,
            "oracle_f1": 1.0,
        },
        "verifier": {
            "attempted": True,
            "original_choice_mode": "extractive_span_v1" if verifier_valid else None,
            "permuted_choice_mode": "extractive_span_v1" if verifier_valid else None,
            "order_agreement": True if verifier_valid else None,
            "parse_status": "ok" if verifier_valid else "invalid_choice",
            "em": verifier_em,
            "f1": verifier_em,
            "generation_time_ms": 2.0,
        },
    }


def result(rows: list[dict[str, object]], stage: str = "formal") -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "phase9i_candidate_coverage_grounded_ranking",
        "stage": stage,
        "diagnostic_only": True,
        "selection_mutation": False,
        "report_only": True,
        "config": {},
        "samples": rows,
    }


class Phase9ITests(unittest.TestCase):
    def test_candidate_and_verifier_prompts_do_not_accept_gold(self):
        tokenizer = Tokenizer()
        secret_gold = "secret gold answer"
        candidate_prompt = build_evidence_constrained_prompt(
            tokenizer, "Where?", ["Paris is in France."]
        )
        verifier_prompt = build_verifier_prompt(
            tokenizer,
            "Where?",
            ["Paris is in France."],
            [("baseline_v1", "Paris"), ("extractive_span_v1", "France")],
        )
        self.assertNotIn(secret_gold, candidate_prompt)
        self.assertNotIn(secret_gold, verifier_prompt)
        self.assertIn(ABSTENTION_TOKEN, candidate_prompt)
        self.assertIn("CANDIDATE_0", verifier_prompt)

    def test_candidate_parsing_excludes_empty_and_preserves_abstention(self):
        self.assertEqual(candidate_parse_status("   "), "empty")
        self.assertEqual(candidate_parse_status(ABSTENTION_TOKEN), "abstain")
        pairs = candidate_pairs(
            "   ",
            CandidateProbeResult("extractive_span_v1", "", "empty", 1.0),
            CandidateProbeResult("evidence_constrained_v1", ABSTENTION_TOKEN, "abstain", 1.0),
        )
        self.assertEqual(pairs, [("evidence_constrained_v1", ABSTENTION_TOKEN)])
        self.assertEqual(
            candidate_pairs(
                "Paris",
                CandidateProbeResult("extractive_span_v1", "Paris", "ok", 1.0),
                CandidateProbeResult("evidence_constrained_v1", "France", "ok", 1.0),
            )[0],
            ("baseline_v1", "Paris"),
        )

    def test_verifier_parser_and_fixed_permutation_are_strict(self):
        self.assertEqual(parse_candidate_id("CANDIDATE_1", 3), 1)
        self.assertEqual(parse_candidate_id("candidate 0", 1), 0)
        self.assertIsNone(parse_candidate_id("CANDIDATE_0 or CANDIDATE_1", 2))
        self.assertIsNone(parse_candidate_id("CANDIDATE_3", 3))
        candidates = [
            ("baseline_v1", "A"),
            ("extractive_span_v1", "B"),
            ("evidence_constrained_v1", "C"),
        ]
        self.assertEqual(fixed_candidate_permutation(candidates), list(reversed(candidates)))
        self.assertEqual(fixed_candidate_permutation(candidates), list(reversed(candidates)))

    def test_compact_schema_rejects_raw_text_and_missing_top_level_key(self):
        compact = result([sample(1)])
        self.assertEqual(validate_result(compact)[0]["question_id"], "nq_1")
        compact["samples"][0]["prediction"] = "Paris"
        with self.assertRaises(Phase9IError):
            validate_result(compact)
        incomplete = result([sample(2)])
        del incomplete["config"]
        with self.assertRaises(Phase9IError):
            validate_result(incomplete)

    def test_candidate_oracle_and_verifier_gate_use_configured_bootstrap(self):
        summary = summarize_phase9i(
            result([sample(index) for index in range(10)]), gate=GATE
        )
        self.assertTrue(summary["decision"]["oracle_gate_pass"])
        self.assertTrue(summary["decision"]["verifier_gate_pass"])
        self.assertEqual(
            summary["paired_deltas"]["candidate_oracle_em"]["bootstrap_repetitions"],
            100,
        )
        self.assertEqual(
            summary["decision"]["primary_failure_class"],
            "verifier_gate_passed_replication_only",
        )

    def test_invalid_verifier_cannot_pass_the_terminal_gate(self):
        summary = summarize_phase9i(
            result([sample(index, verifier_em=None) for index in range(10)]),
            gate=GATE,
        )
        self.assertTrue(summary["decision"]["oracle_gate_pass"])
        self.assertFalse(summary["decision"]["verifier_gate_pass"])
        self.assertEqual(
            summary["decision"]["primary_failure_class"],
            "coverage_positive_verifier_negative",
        )


if __name__ == "__main__":
    unittest.main()
