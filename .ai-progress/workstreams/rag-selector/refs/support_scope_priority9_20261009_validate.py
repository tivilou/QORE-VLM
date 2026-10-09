"""Validate explicit primary opinions and immutable-source joins, not semantics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from applications.rag.support_qualified_boundary_targets import compile_targets, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--records-dir", type=Path, required=True)
    args = p.parse_args()
    r = args.records_dir
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    m = load(r / "semantic_supervision_20261009T045728Z_complete_review_manifest.json")
    assert sha(args.source_dir / "case_study.json") == m["source_case_sha256"]
    assert sha(args.source_dir / "support_review.json") == m["source_support_review_sha256"]
    assert sha(r / m["prior_file"]) == m["prior_sha256"] and sha(r / m["delta_file"]) == m["delta_sha256"]
    prior, delta = load(r / m["prior_file"]), load(r / m["delta_file"])
    review = {"judgments": prior["judgments"] + delta["judgments"], "question_target_flags": prior["question_target_flags"]}
    trace = load(args.source_dir / "case_study.json")
    prefix = "support_scope_priority9_20261009_"
    notes = load(r / (prefix + "primary_recheck.json"))
    overlay = load(r / (prefix + "qualification_hold_overlay.json"))
    stored = load(r / (prefix + "compiler_readback.json"))
    summary = load(r / (prefix + "summary.json"))
    old = {j["review_item_id"]: j for j in review["judgments"]}
    original = load(r / "support_boundary_targets_20261009_boundary_target_draft.json")
    expected = set()
    for row in original["cases"]:
        if row["audit_case_number"] not in [5, 6, 8, 13, 16, 17, 21, 26, 27]:
            continue
        c = trace["cases"][row["audit_case_number"] - 1]
        indices = set(row["provisional_direct_indices"] + [i for pair in row["proposal_pairs"] for i in pair])
        expected.update(c["candidates"][i]["question_passage_review_id"] for i in indices)
    assert len(notes["item_reviews"]) == len(expected) == 54
    assert {j["review_item_id"] for j in notes["item_reviews"]} == expected
    assert notes["independent_of_prior_reviewer"] is False and notes["reviewer"] == "primary_agent_single_model"
    assert {q["audit_case_number"] for q in notes["question_reviews"]} == {5, 6, 8, 13, 16, 17, 21, 26, 27}
    for n in notes["item_reviews"]:
        c = trace["cases"][n["audit_case_number"] - 1]
        item = c["candidates"][n["candidate_index"]]
        assert n["review_case_id"] == c["question_sha256"]
        assert n["review_item_id"] == item["question_passage_review_id"]
        assert n["source_text_sha256"] == item["text_sha256"]
        assert n["original_judgment_sha256"] == digest(old[n["review_item_id"]])
        assert n["original_label"] == old[n["review_item_id"]]["support_label"]
        assert n["support_quote"] in item["text"] and n["rationale"].strip()
        assert n["proposed_scoped_label"] in {"direct", "partial", "irrelevant", "contradictory", "uncertain"}
    assert compile_targets(trace, review, overlay) == stored
    assert stored["summary"]["draft_pairs"] == 0 and stored["training_consumable"] is False
    assert summary["independent_reviews_received"] == 0 and summary["original_review_labels_and_flags_modified"] is False
    assert summary["primary_pending_cases"] == [5, 13, 27]
    assert summary["primary_scope_hold_cases"] == [6, 8, 16, 17, 21, 26]
    assert summary["primary_proposed_retained_pairs"] == 4 and summary["primary_proposed_replacement_slots"] == 3
    assert len(summary["proposed_label_changes"]) == 3
    overlay_cases = {q["review_case_id"]: q for q in overlay["questions"]}
    overlay_items = {i["review_item_id"]: i for i in overlay["items"]}
    for row in summary["cases"]:
        c = trace["cases"][row["audit_case_number"] - 1]
        qid = c["question_sha256"]
        filtered = [pair for pair in row["original_proposal_pairs"] if overlay_cases[qid]["status"] == "pending" and
                    all(overlay_items[c["candidates"][i]["question_passage_review_id"]]["status"] == "pending" for i in pair)]
        assert filtered == row["primary_proposed_retained_pairs"]
    print(json.dumps({"questions_rechecked": 9, "explicit_items": 54, "quotes_and_original_ids_verified": True,
                      "primary_proposed_pairs": 4, "compiler_qualified_pairs": 0, "original1600_unchanged": True,
                      "semantic_correctness_or_independence_certified": False}))


if __name__ == "__main__":
    main()
