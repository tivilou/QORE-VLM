"""Mechanical integrity/coverage checks, not semantic qualification or promotion."""
import argparse
import hashlib
import json
from pathlib import Path

SCOPE_FIELDS = {"time", "location", "version", "entity", "metric", "reference_relation"}
LABELS = {"direct", "partial", "irrelevant", "contradictory", "uncertain"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_input(data):
    require(set(data) == {"schema_version", "training_authorized", "scores_and_prior_labels_included",
                          "selection", "source_case_sha256", "cases"}, "unexpected root fields/scores/labels")
    require(data.get("schema_version") == "rag.blinded_scope_witness_input.v1", "input schema")
    require(data.get("training_authorized") is False, "training must stay off")
    require(data.get("scores_and_prior_labels_included") is False, "blind declaration")
    questions, items = {}, {}
    for c in data["cases"]:
        require(set(c) == {"audit_case_number", "review_case_id", "question", "reference_answers", "items"}, "unexpected question fields")
        qid = c["review_case_id"]
        require(qid not in questions and hashlib.sha256(c["question"].encode()).hexdigest() == qid, "question identity")
        questions[qid] = c
        for p in c["items"]:
            require(set(p) == {"review_item_id", "source_text_sha256", "title", "text"}, "unexpected item fields/scores/labels")
            rid = p["review_item_id"]
            require(rid not in items, "duplicate item")
            require(hashlib.sha256(p["text"].encode()).hexdigest() == p["source_text_sha256"], "text hash")
            items[rid] = (qid, p)
    require(len(questions) == 9 and len(items) == 54, "expected nine questions and54 witnesses")
    return questions, items


def validate_review(data, result, input_sha256):
    questions, items = validate_input(data)
    require(result.get("schema_version") == "rag.independent_scope_review.v1", "review schema")
    require(result.get("input_sha256") == input_sha256, "stale review input")
    require(bool(result.get("reviewer_id", "").strip()) and result["reviewer_id"] != "primary_agent_single_model", "distinct reviewer identity required")
    require(bool(result.get("model_id", "").strip()), "actual model or human reviewer type required")
    exposure = result.get("prior_review_exposure")
    require(type(exposure) is bool, "prior exposure must be disclosed")
    require(result.get("training_authorized") is False, "review does not authorize training")
    seen_q, seen_i = set(), set()
    statuses = {}
    for row in result["question_reviews"]:
        qid = row["review_case_id"]
        require(qid in questions and qid not in seen_q, "question coverage/duplicate")
        seen_q.add(qid)
        status = row["scope_status"]
        require(status in {"resolved", "conditional", "unresolved", "abstained"}, "scope status")
        require(set(row["scope"]) == SCOPE_FIELDS and all(isinstance(v, str) and v.strip() for v in row["scope"].values()), "scope dimensions must be explained")
        require(bool(row.get("rationale", "").strip()), "question rationale")
        statuses[status] = statuses.get(status, 0) + 1
    for row in result["item_reviews"]:
        rid = row["review_item_id"]
        require(rid in items and rid not in seen_i, "item coverage/duplicate")
        seen_i.add(rid)
        qid, p = items[rid]
        require(row["review_case_id"] == qid and row["source_text_sha256"] == p["source_text_sha256"], "item identity")
        require(row["support_label"] in LABELS, "explicit support label required")
        require(bool(row.get("rationale", "").strip()), "item rationale")
        quote = row.get("support_quote")
        require(quote is None or (isinstance(quote, str) and quote.strip() and quote in p["text"]), "literal source quote required")
        require(row["support_label"] not in {"direct", "partial", "contradictory"} or quote is not None, "support-bearing label requires quote")
    require(seen_q == set(questions) and seen_i == set(items), "review must cover all9/54, no default labels")
    return {"questions_checked": 9, "items_checked": 54, "declared_scope_status_counts": statuses,
            "prior_exposure_disclosed": exposure, "independence_certified": False,
            "semantic_correctness_verified": False, "training_authorized": False,
            "formal_qualification_promoted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("packet", type=Path)
    p.add_argument("--review", type=Path)
    args = p.parse_args()
    packet = args.packet.resolve()
    manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
    row = next(r for r in manifest["inputs"] if r["path"] == "context/evidence.json")
    raw = (packet / row["path"]).read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    require(h == row["sha256"], "packet input bytes changed")
    data = json.loads(raw)
    validate_input(data)
    summary = {"questions_checked": 9, "items_checked": 54, "input_sha256": h,
               "independent_review_received": False, "training_authorized": False}
    if args.review:
        summary.update(validate_review(data, json.loads(args.review.read_text(encoding="utf-8")), h))
        summary["independent_review_received"] = True
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
