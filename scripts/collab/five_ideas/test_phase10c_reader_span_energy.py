import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from applications.rag.reader_span_energy import (
    SpanLattice,
    decode_with_span_energy,
    lattice_from_reader_hypotheses,
    matched_control_lattice,
)
from scripts.collab.five_ideas.phase10c_metrics import Phase10CError, summarize_phase10c
from scripts.collab.five_ideas.run_phase10c_reader_span_energy import (
    Phase10CConfigError,
    validate_contract,
)


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, value, add_special_tokens=False, **_kwargs):
        values = [len(piece) + 10 for piece in str(value).split()]
        return {"input_ids": values}


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, passages):
        self.calls.append((question, list(passages)))
        return "frozen baseline"


class Phase10CContractTests(unittest.TestCase):
    def test_disabled_lattice_delegates_to_frozen_generator(self):
        generator = FakeGenerator()
        outcome = decode_with_span_energy(generator, "which", ["one", "two"], SpanLattice(()))
        self.assertEqual(outcome.text, "frozen baseline")
        self.assertFalse(outcome.used_energy_decoder)
        self.assertEqual(outcome.generated_token_count, 0)
        self.assertEqual(generator.calls, [("which", ["one", "two"])])

    def test_reader_lattice_is_deterministic_and_normalized(self):
        tokenizer = FakeTokenizer()
        hypotheses = [
            [{"text": "alpha beta", "probability": 0.5}, {"text": "gamma", "probability": 0.2}],
            [{"text": "alpha beta", "probability": 0.4}],
        ]
        left = lattice_from_reader_hypotheses(hypotheses, [0.8, 0.5], tokenizer, max_spans=10)
        right = lattice_from_reader_hypotheses(hypotheses, [0.8, 0.5], tokenizer, max_spans=10)
        self.assertEqual(left, right)
        self.assertEqual(left.span_count, 2)
        self.assertTrue(math.isclose(sum(weight for _, weight in left.spans), 1.0))

    def test_energy_only_follows_span_continuations(self):
        lattice = SpanLattice((((1, 2), 0.4), ((1, 3), 0.6)))
        initial = lattice.next_token_energy(())
        continued = lattice.next_token_energy((1,))
        self.assertEqual(set(initial), {1})
        self.assertEqual(set(continued), {2, 3})
        self.assertTrue(math.isclose(initial[1], math.log1p(1.0)))
        self.assertTrue(math.isclose(continued[2], math.log1p(0.4)))
        self.assertTrue(math.isclose(continued[3], math.log1p(0.6)))

    def test_matched_control_preserves_shape_and_is_deterministic(self):
        tokenizer = FakeTokenizer()
        reader = SpanLattice((((11, 12), 0.4), ((13,), 0.6)))
        passages = ["a bb ccc dddd eeeee", "ffffff ggggggg"]
        left = matched_control_lattice(tokenizer, passages, reader, seed_key="sample-7")
        right = matched_control_lattice(tokenizer, passages, reader, seed_key="sample-7")
        self.assertEqual(left, right)
        self.assertEqual(left.span_count, reader.span_count)
        self.assertEqual(tuple(len(tokens) for tokens, _ in left.spans), tuple(len(tokens) for tokens, _ in reader.spans))

    @staticmethod
    def _arm(name):
        samples = []
        for index in range(10):
            reader = name == "reader_span_energy"
            samples.append({
                "question_id": f"q{index}", "em": 0.4 + (0.1 if reader else 0.0), "f1": 0.5 + (0.1 if reader else 0.0),
                "generation_time_ms": 100.0, "pipeline_time_ms": 110.0 if reader else 100.0,
                "selected_hit": True, "final_context_K": 5, "selected_context_parity": True,
                "lattice_available": name != "baseline_greedy", "lattice_span_count": 2 if name != "baseline_greedy" else 0,
                "lattice_token_count": 3 if name != "baseline_greedy" else 0, "energy_active_steps": 2 if name != "baseline_greedy" else 0,
                "generated_token_count": 2, "decoder_mode": "frozen_generator" if name == "baseline_greedy" else name,
            })
        return {"config": {"arm": name}, "samples": samples}

    def test_compact_metrics_and_gate_arithmetic(self):
        result = {"stage": "screen", "arms": {name: self._arm(name) for name in ("baseline_greedy", "reader_span_energy", "matched_span_control")}}
        summary = summarize_phase10c(result, gate={"bootstrap_repetitions": 100, "bootstrap_seed": 7, "total_pipeline_cost_ratio_max": 1.25, "screen_f1_ci95_low_min": -0.02, "screen_reader_minus_control_mean_min": 0.0, "formal_f1_delta_min": 0.01})
        self.assertEqual(summary["decision"], "clean_screen_inconclusive")
        self.assertTrue(summary["gate"]["cost"])

    def test_compact_forbidden_field_is_rejected(self):
        result = {"stage": "screen", "arms": {name: self._arm(name) for name in ("baseline_greedy", "reader_span_energy", "matched_span_control")}}
        result["question"] = "not allowed"
        with self.assertRaises(Phase10CError):
            summarize_phase10c(result, gate={"bootstrap_repetitions": 100, "bootstrap_seed": 7, "total_pipeline_cost_ratio_max": 1.25, "screen_f1_ci95_low_min": -0.02, "screen_reader_minus_control_mean_min": 0.0, "formal_f1_delta_min": 0.01})

    def test_repository_contract_validates(self):
        root = Path(__file__).resolve().parents[3]
        result = validate_contract(root / "configs/experiments/phase10c_reader_span_energy.yaml", root / "configs/experiments/phase10c_reader_span_energy_plan.json")
        self.assertEqual(result["status"], "valid")
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["wiki_dpr_started"])

    def test_non_screen_stage_is_rejected(self):
        root = Path(__file__).resolve().parents[3]
        with self.assertRaises(Phase10CConfigError):
            validate_contract(root / "configs/experiments/phase10c_reader_span_energy.yaml", root / "configs/experiments/phase10c_reader_span_energy_plan.json", stage="formal")

    def test_mutated_reader_contract_is_rejected(self):
        root = Path(__file__).resolve().parents[3]
        config = yaml.safe_load((root / "configs/experiments/phase10c_reader_span_energy.yaml").read_text())
        config["phase"]["reader_span_energy"]["energy_coefficient"] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(Phase10CConfigError):
                validate_contract(path, root / "configs/experiments/phase10c_reader_span_energy_plan.json")


if __name__ == "__main__":
    unittest.main()
