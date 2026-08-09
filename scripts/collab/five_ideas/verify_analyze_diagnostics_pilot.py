import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CONFIGS = ("qore_dpr", "qore_as_control", "qore_as_idea6", "topk_as", "mmr_as")
SCRIPT = Path(__file__).with_name("analyze_diagnostics_pilot.py")
SECRET_TEXT = "secret passage text"


def make_result(config_name: str, qids=("q0", "q1")) -> dict:
    config = {
        "dataset": "nq_open", "split": "validation", "max_samples": len(qids),
        "corpus_mode": "wiki_dpr", "K": 2, "lam": 2.0, "seed": 42,
        "method": "qore" if config_name.startswith("qore_") else config_name.split("_")[0],
        "gamma": 0.5, "delta": 0.1 if config_name == "qore_as_idea6" else 0.0,
    }
    samples = []
    for qid in qids:
        sample = {
            "question_id": qid, "question": SECRET_TEXT,
            "gold_answers": [SECRET_TEXT], "prediction": SECRET_TEXT,
            "recall": 1.0, "precision": 0.5,
            "redundancy": 0.2, "diversity": 0.8, "em": 1.0,
            "f1": 1.0, "selection_time_ms": 1.0, "generation_time_ms": 2.0,
            "answer_hit_at_retrieved": True,
            "selected_passages": [
                {"retrieved_rank": 0, "text": SECRET_TEXT},
                {"retrieved_rank": 1, "text": SECRET_TEXT},
            ],
            "all_candidates": [
                {"retrieved_rank": 0, "score": 1.0, "is_gold": True, "selected": True, "text": SECRET_TEXT},
                {"retrieved_rank": 1, "score": 0.9, "is_gold": False, "selected": True, "text": SECRET_TEXT},
                {"retrieved_rank": 2, "score": 0.1, "is_gold": False, "selected": False, "text": SECRET_TEXT},
            ],
        }
        if config_name.startswith("qore_"):
            b = [[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]]
            w = [[0.0, 0.05, 0.1], [0.05, 0.0, 0.15], [0.1, 0.15, 0.0]]
            if config_name == "qore_as_idea6":
                w[0][1] = w[1][0] = 0.02
            sample["qubo"] = {
                "a": [0.9, 0.8, 0.1], "b": b, "w": w,
                "x": [1, 1, 0], "pool_ranks": [0, 1, 2],
                "K": 2, "lam": 2.0, "gamma_effective": 0.5,
                "n_candidates": 3, "prefiltered": True,
                "energy": -4.2, "terms": {"total": -4.2},
            }
        samples.append(sample)
    metrics = {"n_samples": len(samples)}
    for key in ("recall", "f1", "em", "redundancy", "selection_time_ms", "generation_time_ms"):
        metrics[f"mean_{key}"] = samples[0][key]
    metrics["n_with_gold"] = len(samples)
    return {"config": config, "metrics": metrics, "samples": samples}


class DiagnosticsAnalysisTest(unittest.TestCase):
    def test_extracts_compact_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20260809T000000"
            for config in CONFIGS:
                path = run / config
                path.mkdir(parents=True)
                (path / "result.json").write_text(json.dumps(make_result(config)), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(run), "--bootstrap-reps", "100"],
                text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            analysis = run / "analysis"
            self.assertTrue((analysis / "paired_effects.csv").exists())
            self.assertTrue((analysis / "qubo_diagnostics.csv").exists())
            with gzip.open(analysis / "qubo_payload.jsonl.gz", "rt", encoding="utf-8") as handle:
                payload = handle.read()
            self.assertIn("\"a\"", payload)
            self.assertNotIn(SECRET_TEXT, payload)
            for artifact in analysis.iterdir():
                if artifact.suffix != ".gz":
                    self.assertNotIn(SECRET_TEXT, artifact.read_text(encoding="utf-8"))
            summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["validation"]["qore_diagnostics_complete"])
            with (analysis / "qubo_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            idea6 = next(row for row in rows if row["configuration"] == "qore_as_idea6")
            self.assertAlmostEqual(float(idea6["objective_without_constant"]), -1.68)
            self.assertAlmostEqual(float(idea6["residual_pair_sum_selected"]), 0.03)
            self.assertNotEqual(float(idea6["qubo_energy_recomputed"]), float(idea6["recorded_energy"]))

    def test_rejects_mismatched_question_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20260809T000001"
            for config in CONFIGS:
                path = run / config
                path.mkdir(parents=True)
                qids = ("q0", "other") if config == "mmr_as" else ("q0", "q1")
                (path / "result.json").write_text(json.dumps(make_result(config, qids)), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(run), "--bootstrap-reps", "100"],
                text=True, capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("question IDs", completed.stderr)


if __name__ == "__main__":
    unittest.main()
