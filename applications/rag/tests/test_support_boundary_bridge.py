"""Semantic protection, actual-witness validation and gradient-direction tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from applications.rag import support_boundary_bridge as bridge
from applications.rag import support_supervision_axes as axes
from applications.rag import protected_boundary_learning as learning
from applications.rag.quantum_semantic_reader import make_semantic_head
from applications.rag.semantic_reader_training import make_optimizer, SCALED
from scripts.collab.five_ideas.support_boundary_engineering_fixture import synthetic_trace


def witness(row, scope=False):
    binding = axes.qualification_binding(row, scope=scope)
    return {"status": "qualified", "confidence": "independently_adjudicated", "evidence": [
        {"reviewer": name, "role": role, "artifact_ref": "synthetic-only:" + name,
         "sha256": axes.text_hash(name), "annotation_sha256": binding, "prior_review_exposure": False}
        for name, role in (("fixture-primary", "primary_review"), ("fixture-independent", "independent_review"))]}


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.trace = synthetic_trace()

    def setUp(self):
        self.trace = copy.deepcopy(type(self).trace)
        self.annotation = axes.pending_template(self.trace)
        self.case = self.trace["cases"][0]
        self.row = self.annotation["cases"][0]
        self.row["scope"] = {k: "explicit fixture scope" for k in axes.SCOPE_FIELDS}
        self.by_id = {p["review_item_id"]: p for p in self.row["items"]}
        self.set_item(0, "direct"); self.set_item(1, "irrelevant"); self.set_item(2, "partial")
        self.set_item(7, "direct")
        self.row["scope_qualification"] = witness(self.row, True)
        for index in (0, 1, 7):
            item = self.item(index); item["qualification"] = witness(item)

    def item(self, index):
        return self.by_id[self.case["candidates"][index]["review_item_id"]]

    def set_item(self, index, label):
        p = self.item(index)
        p.update(question_support=label, reference_coverage="exact" if label == "direct" else "not_supported",
                 inference_type="explicit" if label != "irrelevant" else "none", source_grounding="single_passage",
                 rationale="Explicit fixture judgment", support_quotes=[p["text"]] if label in {"direct", "partial"} else [])

    def draft_row(self):
        return bridge.protected_draft(self.trace, self.annotation)[0]["cases"][0]

    def review(self, *, independent=False, subset=False):
        items = [copy.deepcopy(p) for p in self.row["items"]]
        for p in items:
            p.pop("qualification")
        if subset:
            items = [next(p for p in items if p["review_item_id"] == self.item(7)["review_item_id"])]
        reviewer = {"identity": "fixture-independent" if independent else "fixture-primary",
                    "model_id": "fixture-model", "session_ref": "fixture-session",
                    "prior_review_exposure": not independent, "outside_packet_context_exposure": not independent}
        return {"schema_version": "rag.support_axes_independent_review.v1", "training_authorized": False,
                "training_consumable": False, "cases": [{"question_sha256": self.row["question_sha256"],
                "review_case_id": "fixture-case", "scope": copy.deepcopy(self.row["scope"]), "reviewer": reviewer, "items": items}]}

    def test_exact_merge_held_and_no_consensus_promotion(self):
        primary = {"cases": [copy.deepcopy(self.row)], "training_authorized": False, "training_consumable": False}
        review = self.review(subset=True)
        p = review["cases"][0]["items"][0]
        dispositions = {"training_authorized": False, "training_consumable": False, "items": [{
            "review_case_id": "fixture-case", "review_item_id": p["review_item_id"],
            "axes": {k: p[k] for k in axes.AXES}, "disposition": "held"}]}
        merged, held = bridge.merge_opinions(self.trace, primary, review, dispositions)
        self.assertEqual(held, {p["review_item_id"]})
        self.assertFalse(merged["training_authorized"])
        self.assertEqual(bridge.protected_draft(self.trace, merged)[1]["qualified_pairs"], 0)
        self.assertTrue(all(r["qualification"]["status"] != "qualified" for r in merged["cases"][0]["items"]))

    def test_changed_identity_and_missing_disposition_rejected(self):
        primary = {"cases": [copy.deepcopy(self.row)], "training_authorized": False, "training_consumable": False}
        review = self.review(subset=True)
        d = {"training_authorized": False, "training_consumable": False, "items": []}
        with self.assertRaises(ValueError): bridge.merge_opinions(self.trace, primary, review, d)
        primary["cases"][0]["items"][0]["text"] += "changed"
        review["cases"] = []
        with self.assertRaises(ValueError): bridge.merge_opinions(self.trace, primary, review, d)

    def test_unqualified_selected_direct_blocks_entire_question(self):
        self.item(0)["qualification"] = axes.pending_qualification()
        target = self.draft_row()
        self.assertEqual(target["draft_pairs"], [])
        self.assertEqual(target["missing_selected_direct_qualification"], [0])
        self.assertEqual(target["immutable_selected_indices"], [0, 1, 2, 3, 4])

    def test_partial_held_external_cannot_be_pair_negative(self):
        self.item(1)["question_support"] = "partial"
        self.item(1)["support_quotes"] = [self.item(1)["text"]]
        self.item(1)["qualification"] = witness(self.item(1))
        self.assertEqual(self.draft_row()["draft_pairs"], [])
        self.set_item(1, "irrelevant"); self.item(1)["source_grounding"] = "requires_external_context"
        self.item(1)["qualification"] = witness(self.item(1))
        self.assertEqual(self.draft_row()["draft_pairs"], [])
        self.item(1)["source_grounding"] = "single_passage"
        self.item(1)["qualification"] = {"status": "held", "confidence": None, "evidence": []}
        self.assertEqual(self.draft_row()["draft_pairs"], [])

    def test_validation_is_not_training_even_when_qualified(self):
        self.case["future_role"] = "validation"
        self.annotation["trace_binding_sha256"] = axes.digest(self.trace)
        draft = self.draft_row()
        self.assertEqual(draft["draft_pairs"], [])
        with self.assertRaises(ValueError): learning.validate_row(draft)

    def test_valid_pair_weights_budget_and_all_other_slots_protected(self):
        self.set_item(8, "direct"); self.item(8)["qualification"] = witness(self.item(8))
        target = self.draft_row()
        self.assertEqual(target["draft_pairs"], [[7, 1], [8, 1]])
        self.assertEqual(sum(target["pair_weights"]), 1)
        self.assertEqual(target["maximum_replacement_slots"], 1)
        self.assertEqual(target["immutable_selected_indices"], [0, 2, 3, 4])

    def test_actual_witnesses_reopened_and_semantically_addressed(self):
        request = bridge.qualification_request(self.trace, self.annotation)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = {}
            for role, independent in (("primary_review", False), ("independent_review", True)):
                document = self.review(independent=independent)
                path = root / (role + ".json"); path.write_text(json.dumps(document), encoding="utf-8")
                documents[role] = (path, document)
            for scope, entries in ((True, request["scopes"]), (False, request["items"])):
                for entry in entries:
                    if not scope and entry["review_item_id"] not in {self.item(i)["review_item_id"] for i in (0, 1, 7)}:
                        continue
                    entry.update(status="qualified", confidence="independently_adjudicated", evidence=[])
                    for role, (path, document) in documents.items():
                        reviewer = document["cases"][0]["reviewer"]
                        entry["evidence"].append({"reviewer": reviewer["identity"], "role": role,
                          "artifact_ref": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "annotation_sha256": entry["binding_sha256"], "prior_review_exposure": reviewer["prior_review_exposure"]})
            applied = bridge.apply_explicit_qualification(self.trace, self.annotation, request, root)
            self.assertEqual(bridge.protected_draft(self.trace, applied)[1]["qualified_pairs"], 1)
            documents["independent_review"][0].write_text("{}")
            with self.assertRaises(ValueError): bridge.apply_explicit_qualification(self.trace, self.annotation, request, root)

    def test_matching_hash_without_target_opinion_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unrelated.json"
            doc = self.review(independent=True, subset=True)
            doc["cases"][0]["items"][0]["question_support"] = "irrelevant"
            path.write_text(json.dumps(doc))
            w = {"artifact_ref": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "reviewer": "fixture-independent", "role": "independent_review", "prior_review_exposure": False}
            with self.assertRaises(ValueError): bridge.actual_witness(directory, w, self.item(7))

    def test_pending_request_has_zero_fabricated_witnesses_and_stale_rejected(self):
        request = bridge.qualification_request(self.trace, self.annotation)
        self.assertTrue(all(not r["evidence"] for r in request["scopes"] + request["items"]))
        # Fixture starts qualified; request defaults pending except explicitly-held states.
        self.assertTrue(all(r["status"] == "pending" for r in request["items"]))
        request["annotations_sha256"] = "0" * 64
        with self.assertRaises(ValueError): bridge.apply_explicit_qualification(self.trace, self.annotation, request, ".")

    def test_pair_gradient_promotes_direct_demotes_only_explicit_negative(self):
        target = self.draft_row()
        base = torch.linspace(5, -5, 50)
        scores = base.clone().requires_grad_()
        parts = learning.boundary_objective(scores, base, target)
        parts["total"].backward()
        self.assertLess(float(scores.grad[7]), 0); self.assertGreater(float(scores.grad[1]), 0)
        self.assertEqual(float(scores.grad[2]), 0)  # Partial not assigned a negative gradient.

    def test_no_pair_skips_optimizer_entirely_including_weight_decay(self):
        self.item(0)["qualification"] = axes.pending_qualification()
        row = self.draft_row(); h = make_semantic_head(8)
        optimizer = torch.optim.AdamW(h.parameters(), lr=.1, weight_decay=.1)
        before = {k: p.detach().clone() for k, p in h.state_dict().items()}
        result = learning.boundary_training_step(h, optimizer, torch.linspace(5, -5, 50), torch.randn(50, 40), row)
        self.assertFalse(result["updated"]); self.assertEqual(len(optimizer.state), 0)
        for k, p in h.state_dict().items(): torch.testing.assert_close(p, before[k], rtol=0, atol=0)

    def test_selector_retains_partial_unknown_direct_despite_extreme_outside_scores(self):
        row = self.draft_row(); scores = [-10.] * 50; scores[7] = 100; scores[49] = 1000
        selected = learning.protected_selection(scores, row)
        self.assertEqual(set(selected), {0, 2, 3, 4, 7})
        self.assertNotIn(49, selected)

    def test_null_exact_baseline_and_live_quantum_classical_data_gradient(self):
        row = self.draft_row(); base = torch.linspace(5, -5, 50)
        pooled = torch.randn(50, 40)
        for classical in (False, True):
            head = make_semantic_head(8, classical=classical)
            optimizer = make_optimizer(torch, head, {"learning_rate": .003, "weight_decay": .0001}, SCALED)
            torch.testing.assert_close(head(base, pooled)[0], base, rtol=0, atol=0)
            self.assertEqual(learning.protected_selection(base, row), self.case["reader_top5_indices"])
            result = learning.boundary_training_step(head, optimizer, base, pooled, row)
            self.assertTrue(result["updated"])
            result = learning.boundary_training_step(head, optimizer, base, pooled, row)
            self.assertGreater(float(result["data_gradients_before_clip"]["project.weight"].norm()), 0)


if __name__ == "__main__":
    unittest.main()
