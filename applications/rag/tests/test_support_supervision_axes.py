import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
import sys

from applications.rag import support_supervision_axes as axes
from scripts.collab.five_ideas import run_support_supervision_cohort_export as runner


def witness(row, scope=False):
    binding = axes.qualification_binding(row, scope=scope)
    return {"status": "qualified", "confidence": "independently_adjudicated", "evidence": [
        {"reviewer": name, "role": role, "artifact_ref": "fixture-only:" + name,
         "sha256": axes.text_hash(name), "annotation_sha256": binding, "prior_review_exposure": False}
        for name, role in (("fixture-primary", "primary_review"), ("fixture-independent", "independent_review"))]}


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.scratch.name) / "fixture"
        cls.summary, cls.cohort = runner.fixture(cls.directory)
        cls.full_trace = runner.load(cls.directory / "case_study.json")
        cls.trace = copy.deepcopy(cls.full_trace)
        cls.trace["cases"] = cls.trace["cases"][:1]
        cls.trace["case_count"] = 1

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def setUp(self):
        self.t = copy.deepcopy(self.trace)
        self.a = axes.pending_template(self.t)

    def item(self, index):
        rid = self.t["cases"][0]["candidates"][index]["review_item_id"]
        return next(x for x in self.a["cases"][0]["items"] if x["review_item_id"] == rid)

    def qualify_scope(self):
        q = self.a["cases"][0]
        q["scope"] = {k: "explicit fixture scope or not applicable" for k in axes.SCOPE_FIELDS}
        q["scope_qualification"] = witness(q, True)

    def qualify_item(self, index, label="direct", reference="exact", inference="explicit"):
        p = self.item(index)
        p.update(question_support=label, reference_coverage=reference, inference_type=inference,
                 source_grounding="single_passage", rationale="Fixture judgment only",
                 support_quotes=[p["text"]] if label in {"direct", "partial", "contradictory"} else [])
        p["qualification"] = witness(p)
        return p

    def draft(self):
        return axes.compile_draft(self.t, self.a)

    def test_full_cohort_readback(self):
        self.assertEqual(runner.validate_outputs(self.directory, self.cohort), self.summary["validation"])
        self.assertEqual(len(self.full_trace["cases"]), 434)
        self.assertEqual(self.summary["annotation_count"], 21700)

    def test_all_pending_no_binary_defaults(self):
        d = self.draft()
        self.assertEqual(d["draft_pair_count"], 0)
        self.assertEqual(d["cases"][0]["roles"], ["held_not_negative"] * 50)
        self.assertIs(d["training_consumable"], False)

    def test_blinded_fields(self):
        s = json.dumps(self.a)
        for banned in ("reader_score", "reader_rank", "retrieved_rank", "reader_top5", "weak_positive", "future_role"):
            self.assertNotIn(banned, s)
        self.assertTrue(all(p[axis] is None for p in self.a["cases"][0]["items"] for axis in axes.AXES))

    def test_valid_alternative_direct_not_negative(self):
        self.qualify_scope(); self.qualify_item(7, reference="valid_alternative"); self.qualify_item(1, "irrelevant", "not_supported", "none")
        d = self.draft()
        self.assertEqual(d["cases"][0]["draft_pairs"], [[7, 1]])
        self.assertFalse(d["training_authorized"])

    def test_reference_mismatch_alone_never_negative(self):
        self.qualify_scope(); self.qualify_item(1, reference="not_supported")
        self.assertEqual(self.draft()["cases"][0]["roles"][1], "protected_direct")

    def test_protected_direct_never_pair_negative(self):
        self.qualify_scope(); self.qualify_item(0); self.qualify_item(7); self.qualify_item(1, "irrelevant", "not_supported", "none")
        r = self.draft()["cases"][0]
        self.assertEqual(r["protected_indices"], [0]); self.assertNotIn([7, 0], r["draft_pairs"])

    def test_partial_and_uncertain_held(self):
        self.qualify_scope(); self.qualify_item(7)
        for index, label in ((1, "partial"), (2, "uncertain"), (3, "contradictory")):
            self.qualify_item(index, label)
        self.assertEqual(self.draft()["draft_pair_count"], 0)

    def test_question_scope_pending_blocks(self):
        self.qualify_item(7); self.qualify_item(1, "irrelevant")
        self.assertEqual(self.draft()["draft_pair_count"], 0)

    def test_held_qualified_opinion_does_not_train(self):
        self.qualify_scope(); p = self.qualify_item(1, "irrelevant"); p["qualification"]["status"] = "held"
        self.qualify_item(7)
        self.assertEqual(self.draft()["draft_pair_count"], 0)

    def test_pair_weights_per_question(self):
        self.qualify_scope(); self.qualify_item(7); self.qualify_item(14)
        self.qualify_item(1, "irrelevant"); self.qualify_item(2, "irrelevant")
        r = self.draft()["cases"][0]
        self.assertEqual(len(r["draft_pairs"]), 4); self.assertEqual(sum(r["pair_weights"]), 1)
        self.assertEqual(r["maximum_replacement_slots"], 2)

    def test_external_context_held(self):
        self.qualify_scope(); p = self.qualify_item(7); p["source_grounding"] = "requires_external_context"; p["qualification"] = witness(p)
        self.qualify_item(1, "irrelevant")
        self.assertEqual(self.draft()["draft_pair_count"], 0)

    def test_arithmetic_inference_allowed(self):
        self.qualify_scope(); self.qualify_item(7, inference="minimal_arithmetic"); self.qualify_item(1, "irrelevant")
        self.assertEqual(self.draft()["draft_pair_count"], 1)

    def test_direct_without_inference_held(self):
        self.qualify_scope(); self.qualify_item(7, inference="none"); self.qualify_item(1, "irrelevant")
        self.assertEqual(self.draft()["draft_pair_count"], 0)

    def test_stale_binding(self):
        self.t["provenance"]["changed"] = True
        with self.assertRaises(ValueError): self.draft()

    def test_mutated_text(self):
        self.t["cases"][0]["candidates"][0]["text"] += " changed"
        with self.assertRaises(ValueError): axes.pending_template(self.t)

    def test_mutated_title(self):
        self.t["cases"][0]["candidates"][0]["title"] += " changed"
        with self.assertRaises(ValueError): axes.pending_template(self.t)

    def test_bad_quote(self):
        p = self.item(0); p.update(question_support="direct", support_quotes=["not in passage"])
        with self.assertRaises(ValueError): self.draft()

    def test_no_quote(self):
        p = self.item(0); p["question_support"] = "direct"
        with self.assertRaises(ValueError): self.draft()

    def test_reordered_blind_items(self):
        self.a["cases"][0]["items"].reverse()
        with self.assertRaises(ValueError): self.draft()

    def test_evaluation_leak(self):
        self.t["evaluation_question_sha256"][0] = self.t["cases"][0]["question_sha256"]
        with self.assertRaises(ValueError): axes.pending_template(self.t)

    def test_wrong_lane(self):
        self.t["cases"][0]["lane"] = "silver_evaluation"
        with self.assertRaises(ValueError): axes.pending_template(self.t)

    def test_changed_top5(self):
        self.t["cases"][0]["reader_top5_indices"] = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError): axes.pending_template(self.t)

    def test_nonfinite_score(self):
        self.t["cases"][0]["candidates"][0]["reader_score"] = float("nan")
        with self.assertRaises(ValueError): axes.pending_template(self.t)

    def test_same_reviewer_not_independent(self):
        p = self.qualify_item(0)
        p["qualification"]["evidence"][1]["reviewer"] = "fixture-primary"
        with self.assertRaises(ValueError): self.draft()

    def test_challenge_exposure_not_independent(self):
        p = self.qualify_item(0)
        p["qualification"]["evidence"][1]["prior_review_exposure"] = True
        with self.assertRaises(ValueError): self.draft()

    def test_stale_witness(self):
        p = self.qualify_item(0); p["rationale"] += " revised"
        with self.assertRaises(ValueError): self.draft()

    def test_missing_scope_dimension(self):
        self.qualify_scope(); self.a["cases"][0]["scope"]["time"] = None
        q = self.a["cases"][0]; q["scope_qualification"] = witness(q, True)
        with self.assertRaises(ValueError): self.draft()

    def test_unqualified_direct_no_protection_claim(self):
        p = self.item(0); p.update(question_support="direct", support_quotes=[p["text"]])
        self.assertEqual(self.draft()["cases"][0]["protected_indices"], [])

    def test_identity_order_checked_before_reader(self):
        c = self.t["cases"][0]; ps = copy.deepcopy(c["candidates"]); ps.reverse()
        with self.assertRaises(ValueError): runner.record_case(1, self.cohort["training_identity"][0],
            {"question": c["question"], "answers": c["answers"]}, ps, [1.] * 50, "training")

    def test_compact_privacy_and_sizes(self):
        for name in runner.COMPACT_FILES:
            path = self.directory / name
            self.assertLessEqual(path.stat().st_size, 1048576)
            for banned in ("Fixture question", "fixture answer explicitly", "reference_answers", "reader_top5_indices"):
                self.assertNotIn(banned, path.read_text(encoding="utf-8"))

    def test_reader_batch_interface(self):
        # Strict API spy catches accidental changes from prior Reader tokenization.
        import torch
        calls = []
        def tokenizer(**kwargs):
            self.assertEqual(set(kwargs), {"questions", "texts", "padding", "truncation", "max_length", "return_tensors"})
            self.assertEqual(kwargs["questions"], ["q"] * len(kwargs["texts"]))
            calls.append(len(kwargs["texts"]))
            return {"input_ids": torch.ones((len(kwargs["texts"]), 3), dtype=torch.long),
                    "attention_mask": torch.ones((len(kwargs["texts"]), 3), dtype=torch.long)}
        class Reader:
            training = False
            def parameters(self): return []
            def __call__(self, **kwargs):
                assert not torch.is_grad_enabled()
                return SimpleNamespace(relevance_logits=torch.ones(kwargs["input_ids"].shape[0]))
        self.assertEqual(runner.reader_scores(torch, tokenizer, Reader(), "cpu", "q", ["p"] * 50,
                         {"batch_size": 8, "max_length": 350}), [1.] * 50)
        self.assertEqual(calls, [8, 8, 8, 8, 8, 8, 2])

    def test_cached_input_never_downloads_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            body = b'{"fixture":"cohort only"}'
            import hashlib
            meta = {"files": [{"name": "cohort.json", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
                              {"name": "head_checkpoints.npz", "bytes": 42, "sha256": "a" * 64}]}
            manifest = json.dumps(meta).encode()
            (d / "input_manifest.json").write_bytes(manifest)
            calls = []
            def download(path, destination, **kwargs):
                calls.append(path)
                self.assertTrue(path.endswith("/cohort.json")); destination.write_bytes(body)
            helper = SimpleNamespace(download_exchange_file=download)
            with mock.patch.object(runner.old, "SOURCE_MANIFEST_SHA", hashlib.sha256(manifest).hexdigest()), \
                 mock.patch.dict(sys.modules, {"scripts.collab.lib.exchange_upload": helper}):
                runner.prepare_input(d, SimpleNamespace(exchange_url="http://fixture", token_env="FIXTURE_TOKEN"))
            self.assertEqual(len(calls), 1)
            self.assertFalse((d / "head_checkpoints.npz").exists())

    def test_upload_uses_existing_manifest_transport_and_hashes(self):
        calls = []
        helper = SimpleNamespace(upload_manifest=lambda path, **kwargs: calls.append(path.name) or [],
                                 _upload_one=lambda *args: calls.append(args[-1]) or {"fixture_receipt": True})
        args = SimpleNamespace(exchange_url="http://fixture", token_env="FIXTURE_TOKEN")
        with mock.patch.dict(sys.modules, {"scripts.collab.lib.exchange_upload": helper}), \
             mock.patch.dict("os.environ", {"FIXTURE_TOKEN": "synthetic-test-value"}):
            receipts = runner.upload_outputs(self.directory, args)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(calls, ["upload_manifest.json", runner.NAMESPACE + "/fixture/upload_manifest.json"])

    def test_upload_rejects_changed_raw_before_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            d = Path(tmp) / "fixture"; shutil.copytree(self.directory, d)
            (d / "case_study.json").write_text("changed", encoding="utf-8")
            helper = SimpleNamespace(upload_manifest=mock.Mock(), _upload_one=mock.Mock())
            with mock.patch.dict(sys.modules, {"scripts.collab.lib.exchange_upload": helper}):
                with self.assertRaises(ValueError):
                    runner.upload_outputs(d, SimpleNamespace(exchange_url="http://fixture", token_env="FIXTURE_TOKEN"))
            helper.upload_manifest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
