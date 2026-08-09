"""Contract and 20-question schema tests for the development gate."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gate_manifest import ManifestError, load_manifest
from run_development_gate import append_cli_args
from verify_analyze_diagnostics_pilot import make_result


ROOT = Path(__file__).resolve().parents[3]
FULL_MANIFEST = ROOT / "configs/experiments/five_idea_development_gate.yaml"
SMOKE_MANIFEST = ROOT / "configs/experiments/five_idea_development_gate_smoke.yaml"
ANALYZER = Path(__file__).with_name("analyze_diagnostics_pilot.py")


class DevelopmentGateTest(unittest.TestCase):
    def test_full_and_smoke_manifests_are_strictly_normalized(self):
        full = load_manifest(FULL_MANIFEST)
        smoke = load_manifest(SMOKE_MANIFEST)
        self.assertEqual(full["gate"]["name"], "five_idea_development_gate")
        self.assertEqual(smoke["gate"]["shared_args"]["max_samples"], 20)
        self.assertEqual(
            [item["name"] for item in full["gate"]["configurations"]],
            ["qore_dpr", "qore_as_control", "qore_as_idea6", "topk_as", "mmr_as"],
        )
        self.assertFalse(full["cache_stats"]["available"])
        self.assertIsNone(full["cache_stats"]["hits"])

    def test_duplicate_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                "schema_version: 1\n"
                "gate:\n"
                "  name: bad\n"
                "  shared_args: {dataset: nq_open}\n"
                "  configurations:\n"
                "    - {name: one, kind: baseline}\n"
                "    - {name: one, kind: baseline}\n",
                encoding="utf-8",
            )
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_command_arguments_are_list_based(self):
        command = ["python", "-m", "module"]
        append_cli_args(command, {"method": "qore", "K": 5, "use_answer_scorer": True})
        self.assertEqual(command, ["python", "-m", "module", "--method", "qore", "--K", "5", "--use_answer_scorer"])

    def test_twenty_question_analyzer_schema_smoke(self):
        manifest = load_manifest(SMOKE_MANIFEST)
        qids = tuple(f"q{index}" for index in range(20))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20260809T000000Z"
            for spec in manifest["gate"]["configurations"]:
                data = make_result(spec["name"], qids)
                if spec["kind"] == "qore":
                    for sample in data["samples"]:
                        sample["qubo"]["enhancer_trace"] = [{
                            "name": "baseline",
                            "mode": "replace",
                            "elapsed_ms": 0.25,
                            "input_norm": 0.0,
                            "output_norm": 1.0,
                            "delta_norm": 1.0,
                        }]
                path = run / spec["name"]
                path.mkdir(parents=True)
                (path / "result.json").write_text(json.dumps(data), encoding="utf-8")
            (run / "run_metadata.json").write_text(json.dumps({
                "git": {"commit": "test-commit", "status": "clean"},
                "python": {"version": sys.version},
                "gate_config": {"path": str(SMOKE_MANIFEST)},
            }), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), str(run), "--gate-config", str(SMOKE_MANIFEST), "--bootstrap-reps", "100"],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((run / "analysis/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["validation"]["samples_per_config"], 20)
            self.assertEqual(summary["validation"]["qore_diagnostics"], 60)
            self.assertEqual(len(summary["plugin_timing"]), 3)
            self.assertEqual(summary["cache_stats"], {
                "available": False,
                "hits": None,
                "misses": None,
                "reason": "Answer Scorer cache is not implemented in Phase 1",
            })
            self.assertEqual(summary["reproducibility"]["git"]["commit"], "test-commit")


if __name__ == "__main__":
    unittest.main()
