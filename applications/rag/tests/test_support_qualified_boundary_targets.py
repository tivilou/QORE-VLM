"""Synthetic qualification tests, not semantic correctness or utility evidence."""
import copy
import hashlib
import json
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

from applications.rag.support_qualified_boundary_targets import (
    binding, compile_targets, digest, pending_overlay, validate_protected_selection,
)


def fixture(labels=None):
    labels = labels or {0: "direct", 1: "partial", 2: "irrelevant", 3: "uncertain", 4: "irrelevant", 5: "direct"}
    question = "synthetic scope-qualified question"
    qid = hashlib.sha256(question.encode()).hexdigest()
    candidates, judgments = [], []
    for i in range(50):
        text = "synthetic passage " + str(i)
        h = hashlib.sha256(text.encode()).hexdigest()
        rid = hashlib.sha256((qid + h).encode()).hexdigest()
        candidates.append({"text": text, "text_sha256": h, "question_passage_review_id": rid,
                           "reader_score": 50.0 - i, "weak_positive": i == 2})
        label = labels.get(i, "irrelevant")
        judgments.append({"review_case_id": qid, "review_item_id": rid, "support_label": label,
                          "support_quote": text if label in {"direct", "partial", "contradictory"} else None,
                          "rationale": "synthetic label witness", "candidate_index": i})
    trace = {"cases": [{"question": question, "question_sha256": qid,
                        "lane": "previous_training_diagnostic_not_heldout", "candidates": candidates,
                        "methods": {"frozen_reader_topk": {"selected_indices": list(range(5))}}}]}
    review = {"judgments": judgments, "question_target_flags": []}
    return trace, review


def qualified(trace, review):
    overlay = pending_overlay(trace, review)
    for row in overlay["questions"] + overlay["items"]:
        row.update(status="qualified", confidence="independently_adjudicated", evidence=[
            {"role": "primary_review", "reviewer": "synthetic-A", "artifact_ref": "fixture-A",
             "sha256": digest("fixture-A")},
            {"role": "independent_review", "reviewer": "synthetic-B", "artifact_ref": "fixture-B",
             "sha256": digest("fixture-B")}])
        if "scope" in row:
            row["scope"] = {k: "not applicable: synthetic scoped fixture" for k in row["scope"]}
    return overlay


class TargetsTest(unittest.TestCase):
    def setUp(self):
        self.trace, self.review = fixture()
        self.overlay = qualified(self.trace, self.review)

    def run_target(self, overlay=None):
        return compile_targets(self.trace, self.review, self.overlay if overlay is None else overlay)

    def test_pending_is_not_training(self):
        result = compile_targets(self.trace, self.review)
        self.assertEqual(result["summary"]["draft_pairs"], 0)
        self.assertEqual(result["summary"]["proposal_pairs"], 2)
        self.assertFalse(result["training_consumable"])

    def test_qualified_binary_pairs_only(self):
        row = self.run_target()["cases"][0]
        self.assertEqual([(p["positive_index"], p["negative_index"]) for p in row["draft_pairs"]], [(5, 2), (5, 4)])
        self.assertAlmostEqual(sum(p["weight"] for p in row["draft_pairs"]), 1)
        self.assertEqual(row["qualified_replacement_budget"], 1)
        self.assertEqual(row["protected_direct_indices"], [0])

    def test_inputs_unchanged(self):
        before = copy.deepcopy((self.trace, self.review, self.overlay))
        self.run_target()
        self.assertEqual(before, (self.trace, self.review, self.overlay))

    def test_deterministic_and_json_replay(self):
        a = self.run_target()
        self.assertEqual(a, self.run_target())
        self.assertEqual(a, json.loads(json.dumps(a)))

    def test_flag_is_never_overridden(self):
        self.review["question_target_flags"] = [{"review_case_id": self.trace["cases"][0]["question_sha256"], "flag": "time"}]
        self.overlay = qualified(self.trace, self.review)
        self.assertEqual(self.run_target()["summary"]["draft_pairs"], 0)
        self.assertEqual(self.run_target()["summary"]["proposal_pairs"], 0)

    def test_missing_review_not_negative(self):
        del self.review["judgments"][2]
        self.overlay = qualified(self.trace, self.review)
        row = self.run_target()["cases"][0]
        self.assertEqual(row["missing_review_count"], 1)
        self.assertEqual([p["negative_index"] for p in row["draft_pairs"]], [4])
        self.assertEqual(row, json.loads(json.dumps(row)))

    def test_missing_scope_blocks(self):
        self.overlay["questions"] = []
        self.assertEqual(self.run_target()["summary"]["draft_pairs"], 0)

    def test_retention_unqualified_blocks_question(self):
        self.overlay["items"][0]["status"] = "pending"
        self.assertIn("selected_direct_retention_not_qualified", self.run_target()["cases"][0]["blockers"])

    def test_boundary_unqualified_does_not_manufacture_pair(self):
        for r in self.overlay["items"]:
            if r["review_item_id"] == self.trace["cases"][0]["candidates"][5]["question_passage_review_id"]:
                r["status"] = "pending"
        self.assertEqual(self.run_target()["summary"]["draft_pairs"], 0)

    def test_protected_selection_accepts_valid_replacement(self):
        validate_protected_selection(self.run_target()["cases"][0], [0, 1, 3, 4, 5])

    def test_protected_selection_rejects_drop(self):
        with self.assertRaises(ValueError):
            validate_protected_selection(self.run_target()["cases"][0], [1, 2, 3, 4, 5])

    def test_selection_shape(self):
        for indices in ([0, 1, 2, 3], [0, 0, 1, 2, 3], [0, 1, 2, 3, 50], [False, 1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                validate_protected_selection(self.run_target()["cases"][0], indices)

    def test_all_direct_no_forced_preferences(self):
        self.trace, self.review = fixture({i: "direct" for i in range(8)})
        self.overlay = qualified(self.trace, self.review)
        row = self.run_target()["cases"][0]
        self.assertEqual(row["draft_pairs"], [])
        self.assertEqual(row["provisional_direct_slot_gap"], 0)
        self.assertEqual(row["protected_direct_indices"], list(range(5)))

    def test_partial_only_not_negative(self):
        self.trace, self.review = fixture({i: "partial" for i in range(5)} | {5: "direct"})
        self.overlay = qualified(self.trace, self.review)
        self.assertEqual(self.run_target()["summary"]["proposal_pairs"], 0)

    def test_zero_direct_no_target(self):
        self.trace, self.review = fixture({0: "partial"})
        self.overlay = qualified(self.trace, self.review)
        self.assertEqual(self.run_target()["summary"]["draft_pairs"], 0)

    def test_no_weak_label_inference(self):
        # In this fixture index2 is a weak positive but independently irrelevant.
        row = self.run_target()["cases"][0]
        self.assertIn(2, [p["negative_index"] for p in row["draft_pairs"]])
        self.assertIn(5, [p["positive_index"] for p in row["draft_pairs"]])

    def test_weight_not_pair_count_strength(self):
        self.trace, self.review = fixture({0: "direct", 5: "direct", 6: "direct", 7: "direct"})
        self.overlay = qualified(self.trace, self.review)
        row = self.run_target()["cases"][0]
        self.assertEqual(len(row["draft_pairs"]), 12)
        self.assertAlmostEqual(sum(p["weight"] for p in row["draft_pairs"]), 1)
        self.assertEqual(row["qualified_replacement_budget"], 3)

    def test_adjudication_contract_not_proof(self):
        result = self.run_target()
        self.assertFalse(result["witness_integrity_and_independence_verified"])
        self.assertFalse(result["training_authorized"])

    def test_unknown_overlay_identity(self):
        self.overlay["items"][0]["review_item_id"] = "unknown"
        with self.assertRaises(ValueError): self.run_target()

    def test_overlay_schema(self):
        self.overlay["schema_version"] = "unknown"
        with self.assertRaises(ValueError): self.run_target()

    def test_duplicate_overlay(self):
        self.overlay["items"].append(self.overlay["items"][0])
        with self.assertRaises(ValueError): self.run_target()

    def test_qualified_scope_complete(self):
        self.overlay["questions"][0]["scope"]["time"] = ""
        with self.assertRaises(ValueError): self.run_target()

    def test_single_review_never_qualified(self):
        self.overlay["items"][0]["evidence"] = self.overlay["items"][0]["evidence"][:1]
        with self.assertRaises(ValueError): self.run_target()

    def test_distinct_reviewers_required(self):
        self.overlay["questions"][0]["evidence"][1]["reviewer"] = "synthetic-A"
        with self.assertRaises(ValueError): self.run_target()

    def test_provisional_confidence_rejected(self):
        self.overlay["items"][0]["confidence"] = "provisional_single_reviewer"
        with self.assertRaises(ValueError): self.run_target()

    def test_overlay_cannot_relabel(self):
        self.overlay["items"][0]["support_label"] = "irrelevant"
        with self.assertRaises(ValueError): self.run_target()

    def test_stale_overlay(self):
        self.review["judgments"][0]["rationale"] += " changed"
        with self.assertRaises(ValueError): self.run_target()

    def test_source_hash(self):
        self.trace["cases"][0]["candidates"][0]["text"] += " changed"
        with self.assertRaises(ValueError): self.run_target()

    def test_reader_order_frozen(self):
        self.trace["cases"][0]["methods"]["frozen_reader_topk"]["selected_indices"] = [4, 3, 2, 1, 0]
        with self.assertRaises(ValueError): self.run_target()

    def test_duplicate_review(self):
        self.review["judgments"].append(self.review["judgments"][0])
        with self.assertRaises(ValueError): self.run_target()

    def test_bad_quote(self):
        self.review["judgments"][0]["support_quote"] = "not in passage"
        with self.assertRaises(ValueError): self.run_target()

    def test_silver_lane_rejected(self):
        self.trace["cases"][0]["lane"] = "silver_evaluation"
        with self.assertRaises(ValueError): self.run_target()

    def test_nan_score(self):
        self.trace["cases"][0]["candidates"][0]["reader_score"] = float("nan")
        with self.assertRaises(ValueError): self.run_target()

    def test_missing_review_cannot_be_qualified(self):
        del self.review["judgments"][0]
        self.overlay["input_binding_sha256"] = binding(self.trace, self.review)
        with self.assertRaises(ValueError): self.run_target()


class CompilerCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trace, self.review = fixture()
        def write(name, value):
            path = self.root / name
            path.write_bytes(json.dumps(value).encode())
            return hashlib.sha256(path.read_bytes()).hexdigest()
        source_hash = write("case_study.json", self.trace)
        blind_hash = write("support_review.json", {"cases": [{"passages": [{"support_label": None}]}]})
        prior_hash = write("prior.json", self.review)
        delta_hash = write("delta.json", {"judgments": []})
        write("manifest.json", {"prior_file": "prior.json", "prior_sha256": prior_hash,
                               "delta_file": "delta.json", "delta_sha256": delta_hash,
                               "source_case_sha256": source_hash, "source_support_review_sha256": blind_hash,
                               "merged_count": 50})
        self.command = [sys.executable, str(Path(__file__).resolve().parents[3] /
                        "scripts/collab/five_ideas/compile_support_qualified_boundary_targets.py"),
                        "--source-dir", str(self.root), "--review-manifest", str(self.root / "manifest.json"),
                        "--output-dir", str(self.root / "output")]

    def run_cli(self, extra=()):
        return subprocess.run(self.command + list(extra), capture_output=True, text=True, encoding="utf-8")

    def test_cli_pending_and_readback(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        target = json.loads((self.root / "output/boundary_target_draft.json").read_text())
        self.assertEqual(target["summary"]["draft_pairs"], 0)
        self.assertFalse(target["training_consumable"])

    def test_cli_does_not_overwrite(self):
        self.assertEqual(self.run_cli().returncode, 0)
        before = (self.root / "output/boundary_target_draft.json").read_bytes()
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual(before, (self.root / "output/boundary_target_draft.json").read_bytes())

    def test_cli_source_hash_reject(self):
        (self.root / "case_study.json").write_bytes(b"{}")
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertFalse((self.root / "output").exists())

    def test_cli_witness_bytes_required(self):
        overlay = qualified(self.trace, self.review)
        (self.root / "qualification.json").write_text(json.dumps(overlay))
        self.assertNotEqual(self.run_cli(["--qualification", str(self.root / "qualification.json")]).returncode, 0)
        self.assertFalse((self.root / "output").exists())

    def test_cli_verified_witnesses_still_not_training(self):
        overlay = qualified(self.trace, self.review)
        for name in ("fixture-A", "fixture-B"):
            (self.root / name).write_bytes(json.dumps(name).encode())
        (self.root / "qualification.json").write_text(json.dumps(overlay))
        result = self.run_cli(["--qualification", str(self.root / "qualification.json")])
        self.assertEqual(result.returncode, 0, result.stderr)
        target = json.loads((self.root / "output/boundary_target_draft.json").read_text())
        self.assertEqual(target["summary"]["draft_pairs"], 2)
        self.assertFalse(target["training_consumable"])

    def test_cli_witness_path_escape_rejected(self):
        overlay = qualified(self.trace, self.review)
        overlay["questions"][0]["evidence"][0]["artifact_ref"] = "../elsewhere"
        (self.root / "qualification.json").write_text(json.dumps(overlay))
        self.assertNotEqual(self.run_cli(["--qualification", str(self.root / "qualification.json")]).returncode, 0)
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
