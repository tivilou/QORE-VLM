import copy
import hashlib
import unittest

from scripts.collab.five_ideas.validate_support_scope_review import validate_input, validate_review


def fixture():
    cases = []
    for k in range(9):
        question = "synthetic question " + str(k)
        qid = hashlib.sha256(question.encode()).hexdigest()
        items = []
        for i in range(6):
            text = "synthetic passage " + str(k) + "-" + str(i)
            h = hashlib.sha256(text.encode()).hexdigest()
            items.append({"review_item_id": hashlib.sha256((qid + h).encode()).hexdigest(),
                          "source_text_sha256": h, "title": "synthetic", "text": text})
        cases.append({"audit_case_number": k + 1, "review_case_id": qid, "question": question,
                      "reference_answers": ["fixture"], "items": items})
    data = {"schema_version": "rag.blinded_scope_witness_input.v1", "training_authorized": False,
            "scores_and_prior_labels_included": False, "cases": cases,
            "selection": "synthetic fixed fixture", "source_case_sha256": "synthetic-source"}
    result = {"schema_version": "rag.independent_scope_review.v1", "input_sha256": "bound-fixture",
              "reviewer_id": "synthetic-B", "model_id": "synthetic-not-real-worker",
              "prior_review_exposure": False, "training_authorized": False,
              "question_reviews": [], "item_reviews": []}
    for c in cases:
        result["question_reviews"].append({"review_case_id": c["review_case_id"], "scope_status": "conditional",
            "scope": {key: "synthetic resolution" for key in ("time", "location", "version", "entity", "metric", "reference_relation")},
            "rationale": "fixture"})
        for p in c["items"]:
            result["item_reviews"].append({"review_case_id": c["review_case_id"], "review_item_id": p["review_item_id"],
                "source_text_sha256": p["source_text_sha256"], "support_label": "partial",
                "support_quote": p["text"], "rationale": "fixture"})
    return data, result


class ReviewChecks(unittest.TestCase):
    def setUp(self):
        self.data, self.result = fixture()

    def check(self):
        return validate_review(self.data, self.result, "bound-fixture")

    def test_complete_but_not_promoted(self):
        result = self.check()
        self.assertEqual(result["items_checked"], 54)
        self.assertFalse(result["formal_qualification_promoted"])

    def test_readonly(self):
        before = copy.deepcopy((self.data, self.result))
        self.check()
        self.assertEqual(before, (self.data, self.result))

    def test_leaked_score_rejected(self):
        self.data["cases"][0]["items"][0]["reader_score"] = 1
        with self.assertRaises(ValueError): self.check()

    def test_leaked_prior_label_rejected(self):
        self.data["cases"][0]["items"][0]["original_label"] = "direct"
        with self.assertRaises(ValueError): self.check()

    def test_leaked_root_results_rejected(self):
        self.data["old_results"] = {"Reader": "score"}
        with self.assertRaises(ValueError): self.check()

    def test_changed_text_rejected(self):
        self.data["cases"][0]["items"][0]["text"] += " altered"
        with self.assertRaises(ValueError): self.check()

    def test_changed_question_rejected(self):
        self.data["cases"][0]["question"] += " altered"
        with self.assertRaises(ValueError): self.check()

    def test_missing_item_rejected(self):
        self.result["item_reviews"].pop()
        with self.assertRaises(ValueError): self.check()

    def test_missing_question_rejected(self):
        self.result["question_reviews"].pop()
        with self.assertRaises(ValueError): self.check()

    def test_duplicate_item_rejected(self):
        self.result["item_reviews"][0] = self.result["item_reviews"][1]
        with self.assertRaises(ValueError): self.check()

    def test_fabricated_quote_rejected(self):
        self.result["item_reviews"][0]["support_quote"] = "fabricated"
        with self.assertRaises(ValueError): self.check()

    def test_empty_label_rejected(self):
        self.result["item_reviews"][0]["support_label"] = None
        with self.assertRaises(ValueError): self.check()

    def test_unknown_reviewer_rejected(self):
        self.result["reviewer_id"] = "primary_agent_single_model"
        with self.assertRaises(ValueError): self.check()

    def test_exposure_must_be_disclosed(self):
        self.result["prior_review_exposure"] = None
        with self.assertRaises(ValueError): self.check()

    def test_exposed_review_is_not_certified(self):
        self.result["prior_review_exposure"] = True
        self.assertTrue(self.check()["prior_exposure_disclosed"])
        self.assertFalse(self.check()["independence_certified"])

    def test_no_training_authorization(self):
        self.result["training_authorized"] = True
        with self.assertRaises(ValueError): self.check()

    def test_stale_hash_rejected(self):
        self.result["input_sha256"] = "stale"
        with self.assertRaises(ValueError): self.check()

    def test_partial_quote_required(self):
        self.result["item_reviews"][0]["support_quote"] = None
        with self.assertRaises(ValueError): self.check()

    def test_scope_dimension_required(self):
        self.result["question_reviews"][0]["scope"]["time"] = ""
        with self.assertRaises(ValueError): self.check()


if __name__ == "__main__":
    unittest.main()
