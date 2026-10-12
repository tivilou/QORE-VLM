"""Synthetic-only exporter/trace/target-boundary regression tests."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from applications.rag import semantic_supervision_audit as a
from scripts.collab.five_ideas import run_semantic_reader_supervision_audit as runner


def cohort_fixture():
    return {"usable_cases": 434, "evaluation_question_sha256": [a.digest_text(f"eval{i}") for i in range(100)],
            "training_identity": [{"question_sha256": a.digest_text(f"train{i}"),
               "candidate_id_sha256": [a.digest_text(f"{i}:{j}") for j in range(50)],
               "positive_mask": [j % 7 == 0 for j in range(50)]} for i in range(434)]}


class Helpers(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"id": str(i), "retrieved_rank": i + 1} for i in range(50)]
        self.base = np.linspace(4., -4., 50)

    def test_target_exact_old_substring_behavior_not_gold(self):
        from scripts.collab.five_ideas.run_qarcg_reader_screen_100 import _weak_positive_mask
        texts = ["Mandela spoke", "a band", "world-wide event", "France"]
        for answers in (["an"], ["world wide"], [""], ["France"], ["missing"]):
            self.assertEqual(a.weak_labels(texts, answers), _weak_positive_mask(texts, answers))
        self.assertTrue(a.weak_labels(["a band", "other"], ["an"])[0][0])
        self.assertFalse(a.containment_witnesses("a band", ["an"])[0]["match_is_support_proof"])

    def test_sample_fixed_exclusion_and_future_not_unseen(self):
        c = cohort_fixture(); first, future = a.sample_cohort(c); again, second = a.sample_cohort(c)
        self.assertEqual(first, again); self.assertEqual(future, second)
        self.assertEqual(len(first), 32); self.assertEqual(len({p["question_sha256"] for p in first}), 32)
        self.assertEqual([r["usable_case_index"] for r in first[:2]], [1, 2])
        self.assertEqual(future["validation_count"], 87)
        self.assertFalse(future["independent_validation_claim"])
        self.assertTrue(future["previous_heads_trained_on_both_roles"])

    def test_source_leakage_rejected(self):
        c = cohort_fixture(); c["evaluation_question_sha256"][0] = c["training_identity"][0]["question_sha256"]
        with self.assertRaises(ValueError): a.sample_cohort(c)

    def test_invalid_cohort_target_rejected(self):
        c = cohort_fixture(); c["training_identity"][0]["positive_mask"][0] = 1
        with self.assertRaises(ValueError): a.validate_cohort(c)

    def test_wrong_sampling_budget_rejected(self):
        with self.assertRaises(ValueError): a.sample_cohort(cohort_fixture(), count=50)

    def test_stable_ties_and_monotonic_null(self):
        self.assertEqual(a.rank_order(np.zeros(50), self.candidates), list(range(50)))
        self.assertEqual(a.rank_order(self.base, self.candidates), a.rank_order(.5 * self.base, self.candidates))
        c = a.compression_diagnostic(self.base, .5 * self.base)
        self.assertAlmostEqual(c["score_on_base_slope"], .5)
        self.assertAlmostEqual(c["non_affine_rmse"], 0)

    def test_nonfinite_and_duplicate_candidates(self):
        with self.assertRaises(ValueError): a.rank_order(np.full(50, np.nan), self.candidates)
        self.candidates[0]["id"] = self.candidates[1]["id"]
        with self.assertRaises(ValueError): a.rank_order(self.base, self.candidates)

    def test_identity_tolerates_float_tie_reorder_only(self):
        prior_mask = [i % 7 == 0 for i in range(50)]
        cands = [{"id": str(i), "retrieved_rank": i + 1, "retrieval_score": float(50 - i)} for i in range(50)]
        identity = {"question_sha256": a.digest_text("q"),
                    "candidate_id_sha256": [a.digest_text(p["id"]) for p in cands],
                    "positive_mask": prior_mask}
        a.check_identity("q", cands, prior_mask, identity)
        # Float32 tie: positions 6/7 swap order with scores equal within tolerance, and the
        # two have different weak labels, so positional comparison alone would false-fail.
        tied = list(cands)
        tied[6] = {**cands[7], "retrieved_rank": 7, "retrieval_score": 43.5}
        tied[7] = {**cands[6], "retrieved_rank": 8, "retrieval_score": 43.5}
        tied_mask = list(prior_mask)
        tied_mask[6], tied_mask[7] = prior_mask[7], prior_mask[6]
        a.check_identity("q", tied, tied_mask, identity)
        # A genuine reorder (distinct scores) is still rejected.
        real = list(cands)
        real[6] = {**cands[7], "retrieved_rank": 7, "retrieval_score": 44.0}
        real[7] = {**cands[6], "retrieved_rank": 8, "retrieval_score": 43.0}
        with self.assertRaises(ValueError):
            a.check_identity("q", real, prior_mask, identity)
        # A dropped/replaced candidate is rejected.
        missing = copy.deepcopy(cands)
        missing[0]["id"] = "absent"
        with self.assertRaises(ValueError):
            a.check_identity("q", missing, prior_mask, identity)

    def test_constant_correction_does_not_discriminate(self):
        mask = [i == 6 for i in range(50)]
        d = a.boundary_diagnostic(self.base, self.base + 1, mask, self.candidates)
        self.assertEqual(d["pair_count"], 2)
        self.assertEqual(d["positive_direction_pairs"], 0)
        self.assertEqual(d["strict_crossings"], 0)

    def test_wrong_outsider_is_matched_control(self):
        mask = [i == 6 for i in range(50)]
        scores = self.base.copy(); scores[5:] += .3
        d = a.boundary_diagnostic(self.base, scores, mask, self.candidates)
        for p in d["pairs"]:
            self.assertAlmostEqual(p["relative_correction"], p["matched_weak_negative_control"]["relative_correction"])
            self.assertTrue(p["labels_are_not_certified_support"])

    def test_no_boundary_pairs_is_not_success(self):
        d = a.boundary_diagnostic(self.base, self.base, [True] * 50, self.candidates)
        self.assertEqual(d["pair_count"], 0); self.assertEqual(d["strict_crossings"], 0)

    def test_cache_offline_forced_not_setdefault(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            runner.force_offline()
            for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
                self.assertEqual(os.environ[k], "1")

    def test_no_training_code_path(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        for forbidden in (".backward(", "optimizer.step(", "train_head(", "add_faiss_index(", "_load_eval_cases("):
            self.assertNotIn(forbidden, source)


class TraceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name) / "fixture"
        cls.summary, cls.cohort, cls.checkpoints = runner.produce_fixture(cls.directory, return_context=True)
        cls.trace = runner.load_json(cls.directory / "case_study.json")
        cls.review = runner.load_json(cls.directory / "support_review.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_fixture_reopen_and_full_readiness(self):
        self.assertTrue(all(v == "pass" for v in self.summary["validation"].values()))
        self.assertEqual(len(self.trace["cases"]), 32)
        with np.load(self.directory / "audit_values.npz", allow_pickle=False) as values:
            self.assertEqual(sum(k.endswith("__pooled") for k in values.files), 32)
            self.assertEqual(sum(k.endswith("__hidden") for k in values.files), 2)

    def test_blind_view_has_all_texts_no_score_or_weak_fields(self):
        self.assertEqual(sum(len(c["passages"]) for c in self.review["cases"]), 1600)
        for c in self.review["cases"]:
            for p in c["passages"]:
                self.assertEqual(set(p), {"review_item_id", "title", "text", "support_label", "support_quote", "rationale"})
                self.assertIsNone(p["support_label"])
        self.assertEqual(self.review, a.blinded_review(self.trace["cases"]))

    def test_pending_review_never_certifies_support(self):
        self.assertEqual(self.summary["support_review_status"], "pending_review")
        self.assertFalse(self.trace["support_certified"])
        self.assertFalse(self.summary["training_called"])
        with self.assertRaises(ValueError): a.validate_reviews(self.review, self.review)

    def test_review_requires_original_quote(self):
        reviewed = copy.deepcopy(self.review)
        for c in reviewed["cases"]:
            for p in c["passages"]:
                p.update(support_label="irrelevant", support_quote=None, rationale="Unrelated to requested relation.")
        self.assertEqual(a.validate_reviews(self.review, reviewed)["irrelevant"], 1600)
        reviewed["cases"][0]["passages"][0].update(support_label="direct", support_quote="fabricated quote")
        with self.assertRaises(ValueError): a.validate_reviews(self.review, reviewed)

    def test_review_input_edit_rejected(self):
        changed = copy.deepcopy(self.review); changed["cases"][0]["passages"][0]["text"] += " changed"
        with self.assertRaises(ValueError): a.validate_reviews(self.review, changed)

    def test_compact_raw_boundary(self):
        from scripts.collab.five_ideas.run_qarcg_reader_screen_100 import _contains_forbidden
        self.assertFalse(_contains_forbidden(self.summary))
        manifest = runner.load_json(self.directory / "upload_manifest.json")
        self.assertEqual(manifest["exchange_only"], list(runner.RAW_FILES))
        for name in runner.COMPACT_FILES:
            self.assertLessEqual((self.directory / name).stat().st_size, 1_048_576)

    def assert_trace_edit_rejected(self, edit):
        path = self.directory / "case_study.json"
        original = path.read_bytes()
        changed = copy.deepcopy(self.trace)
        edit(changed)
        try:
            a.write_json(path, changed)
            with self.assertRaises(ValueError):
                runner.validate_trace(self.directory, self.cohort, self.checkpoints, fixture=True)
        finally:
            path.write_bytes(original)

    def test_reader_rank_tamper_rejected(self):
        self.assert_trace_edit_rejected(lambda trace: trace["cases"][0]["candidates"][0].update(reader_rank=50))

    def test_stage_or_sample_tamper_rejected(self):
        self.assert_trace_edit_rejected(lambda trace: trace["cases"][0].update(stages=["data", "training"]))

    def test_raw_text_tamper_rejected(self):
        self.assert_trace_edit_rejected(lambda trace: trace["cases"][0]["candidates"][0].update(text="modified raw text"))

    def test_selection_tamper_rejected(self):
        self.assert_trace_edit_rejected(lambda trace: trace["cases"][0]["methods"]["quantum_semantic"].update(selected_indices=[49, 48, 47, 46, 45]))

    def test_upload_failure_preserves_pending_files(self):
        module = SimpleNamespace(upload_manifest=lambda *args, **kwargs: (_ for _ in ()).throw(OSError("synthetic network failure")),
                                 _upload_one=lambda *args, **kwargs: {})
        manifest = (self.directory / "upload_manifest.json").read_bytes()
        with patch.dict("sys.modules", {"scripts.collab.lib.exchange_upload": module}):
            with self.assertRaises(OSError):
                runner.upload_outputs(self.directory, SimpleNamespace(exchange_url="http://fixture.invalid", token_env="FIXTURE_TOKEN"))
        self.assertEqual((self.directory / "upload_manifest.json").read_bytes(), manifest)
        self.assertTrue((self.directory / "case_study.json").exists())
        self.assertFalse((self.directory / "upload_receipts.json").exists())

    def test_upload_corrupted_artifact_fails_before_transport(self):
        called = []
        module = SimpleNamespace(upload_manifest=lambda *args, **kwargs: called.append(True), _upload_one=lambda *args, **kwargs: {})
        path = self.directory / "case_study.md"; original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with patch.dict("sys.modules", {"scripts.collab.lib.exchange_upload": module}):
                with self.assertRaises(ValueError):
                    runner.upload_outputs(self.directory, SimpleNamespace(exchange_url="http://fixture.invalid", token_env="FIXTURE_TOKEN"))
            self.assertEqual(called, [])
        finally:
            path.write_bytes(original)


if __name__ == "__main__": unittest.main()
