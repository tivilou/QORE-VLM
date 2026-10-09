"""Versioned train-only review axes and non-consumable target drafts.

No label inference, online scoring, fitting, or Silver-panel import. Review
witnesses are declarations to be audited, not a semantic truth certificate.
"""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "rag.support_supervision_axes.v1"
TRACE_SCHEMA = "rag.support_cohort_export.v1"
SCOPE_FIELDS = ("time", "location", "version", "entity", "metric", "reference_relation")
AXES = {
    "question_support": ("direct", "partial", "irrelevant", "contradictory", "uncertain"),
    "reference_coverage": ("exact", "equivalent", "valid_alternative", "not_supported", "unresolved"),
    "inference_type": ("explicit", "minimal_arithmetic", "intra_passage_composition", "none", "unresolved"),
    "source_grounding": ("single_passage", "requires_external_context", "unresolved"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def text_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_trace(trace):
    require(trace.get("schema_version") == TRACE_SCHEMA, "unsupported cohort trace")
    require(trace.get("training_authorized") is False, "export is observation only")
    require(trace.get("stages") == ["data", "retrieval", "frozen_reader_scoring", "baseline_selection", "pending_review"]
            and trace.get("hidden_tensor_capture") is False, "export stage/capture contract changed")
    cases = trace["cases"]
    require(len(cases) == trace["case_count"] and bool(cases), "case coverage mismatch")
    qids, rids = set(), set()
    excluded = trace["evaluation_question_sha256"]
    require(len(excluded) == 100 and len(set(excluded)) == 100, "evaluation exclusion mismatch")
    for number, case in enumerate(cases, 1):
        qid = case["question_sha256"]
        require(qid == text_hash(case["question"]) and qid not in qids and qid not in excluded,
                "question identity/duplicate/evaluation leakage")
        qids.add(qid)
        require(case["usable_case_index"] == number and
                case["lane"] == "previous_training_diagnostic_not_heldout", "training lane/order mismatch")
        require(case["future_role"] in {"training", "validation"}, "unknown prospective role")
        ps = case["candidates"]
        require(len(ps) == 50 and [p["retrieved_rank"] for p in ps] == list(range(1, 51)), "Top50 order mismatch")
        require(len({str(p["id"]) for p in ps}) == 50, "duplicate passage IDs")
        for p in ps:
            rid = p["review_item_id"]
            require(rid == text_hash(qid + ":" + str(p["id"])) and rid not in rids, "review identity mismatch")
            rids.add(rid)
            require(p["text_sha256"] == text_hash(p["text"]) and
                    p["title_sha256"] == text_hash(p["title"]), "passage/title hash mismatch")
            require(type(p["reader_score"]) in (int, float) and math.isfinite(p["reader_score"]) and
                    type(p["retrieval_score"]) in (int, float) and math.isfinite(p["retrieval_score"]), "nonfinite score")
        order = sorted(range(50), key=lambda i: (-ps[i]["reader_score"], i))
        require(case["reader_top5_indices"] == order[:5] and
                [ps[i]["reader_rank"] for i in order] == list(range(1, 51)), "Reader selection/rank mismatch")
    return cases


def pending_qualification():
    return {"status": "pending", "confidence": None, "evidence": []}


def pending_template(trace):
    cases = validate_trace(trace)
    rows = []
    for c in cases:
        qid = c["question_sha256"]
        # Score-independent permutation. No ranks, scores, weak labels or selection.
        ps = sorted(c["candidates"], key=lambda p: text_hash("axes-v1:" + p["review_item_id"]))
        rows.append({"question_sha256": qid, "question": c["question"], "reference_answers": c["answers"],
                     "scope": {k: None for k in SCOPE_FIELDS}, "scope_qualification": pending_qualification(),
                     "items": [{"review_item_id": p["review_item_id"], "title": p["title"], "text": p["text"],
                                "text_sha256": p["text_sha256"], "title_sha256": p["title_sha256"],
                                **{k: None for k in AXES}, "rationale": None, "support_quotes": [],
                                "qualification": pending_qualification()} for p in ps]})
    return {"schema_version": SCHEMA, "trace_binding_sha256": digest(trace),
            "training_authorized": False, "cases": rows}


def qualification_binding(row, *, scope=False):
    """Witnesses must bind the exact versioned scope or annotation they review."""
    return digest({k: v for k, v in row.items() if k != ("scope_qualification" if scope else "qualification")
                   and (not scope or k != "items")})


def qualified(record, expected_binding):
    require(set(record) == {"status", "confidence", "evidence"}, "qualification fields mismatch")
    require(record["status"] in {"pending", "held", "qualified"}, "unknown qualification status")
    if record["status"] != "qualified":
        return False
    require(record["confidence"] == "independently_adjudicated", "independent qualification required")
    evidence = record["evidence"]
    require(len(evidence) >= 2, "two actual witnesses required")
    identities, hashes, roles = set(), set(), set()
    for e in evidence:
        require(set(e) == {"reviewer", "role", "artifact_ref", "sha256", "annotation_sha256", "prior_review_exposure"},
                "witness fields mismatch")
        require(e["role"] in {"primary_review", "independent_review"} and
                isinstance(e["reviewer"], str) and bool(e["reviewer"].strip()) and
                isinstance(e["artifact_ref"], str) and bool(e["artifact_ref"].strip()), "witness identity/reference required")
        h = e["sha256"]
        require(isinstance(h, str) and len(h) == 64 and all(x in "0123456789abcdef" for x in h), "witness hash invalid")
        require(e["annotation_sha256"] == expected_binding, "stale witness annotation")
        require(type(e["prior_review_exposure"]) is bool and
                (e["role"] != "independent_review" or e["prior_review_exposure"] is False), "challenge-exposed review is not independent")
        identities.add(e["reviewer"]); hashes.add(h); roles.add(e["role"])
    require(len(identities) == len(evidence) and len(hashes) == len(evidence) and
            roles == {"primary_review", "independent_review"}, "distinct reviewer/artifact identities required")
    return True


def validate_annotations(trace, annotations):
    template = pending_template(trace)
    require(set(annotations) == set(template) and annotations["schema_version"] == SCHEMA and
            annotations["trace_binding_sha256"] == template["trace_binding_sha256"] and
            annotations["training_authorized"] is False, "annotation schema/source binding mismatch")
    require(len(annotations["cases"]) == len(template["cases"]), "annotation coverage mismatch")
    for expected, row in zip(template["cases"], annotations["cases"]):
        require(set(row) == set(expected), "question fields mismatch")
        for k in ("question_sha256", "question", "reference_answers"):
            require(row[k] == expected[k], "immutable question/reference changed")
        require(set(row["scope"]) == set(SCOPE_FIELDS) and
                all(v is None or (isinstance(v, str) and v.strip()) for v in row["scope"].values()), "scope fields invalid")
        qready = qualified(row["scope_qualification"], qualification_binding(row, scope=True))
        require(not qready or all(row["scope"].values()), "all six scopes need explicit resolution or inapplicability")
        require(len(row["items"]) == 50, "annotation item coverage mismatch")
        for ep, p in zip(expected["items"], row["items"]):
            require(set(p) == set(ep), "annotation fields mismatch")
            for k in ("review_item_id", "title", "text", "text_sha256", "title_sha256"):
                require(p[k] == ep[k], "immutable passage/order changed")
            for k, values in AXES.items():
                require(p[k] is None or p[k] in values, "invalid " + k)
            require(p["rationale"] is None or (isinstance(p["rationale"], str) and p["rationale"].strip()), "rationale invalid")
            require(isinstance(p["support_quotes"], list) and
                    all(isinstance(x, str) and x.strip() and x in p["text"] for x in p["support_quotes"]), "quote is not contiguous passage evidence")
            ready = qualified(p["qualification"], qualification_binding(p))
            require(not ready or (all(p[k] is not None for k in AXES) and bool(p["rationale"])), "qualified item axes/rationale incomplete")
            if p["question_support"] in {"direct", "partial", "contradictory"}:
                require(bool(p["support_quotes"]), "support-bearing judgment needs exact quote")
    return annotations


def compile_draft(trace, annotations):
    """Propose protected pair masks; never emit training permission or binary defaults."""
    validate_annotations(trace, annotations)
    drafts = []
    for case, row in zip(trace["cases"], annotations["cases"]):
        qready = qualified(row["scope_qualification"], qualification_binding(row, scope=True))
        by_id = {p["review_item_id"]: p for p in row["items"]}
        usable, labels = [], []
        for p in case["candidates"]:
            a = by_id[p["review_item_id"]]
            ready = qualified(a["qualification"], qualification_binding(a))
            # Reference mismatch alone NEVER makes an item negative.
            usable.append(qready and ready and a["source_grounding"] == "single_passage" and
                          a["inference_type"] != "unresolved" and a["reference_coverage"] != "unresolved" and
                          (a["question_support"] != "direct" or a["inference_type"] != "none"))
            labels.append(a["question_support"])
        selected = set(case["reader_top5_indices"])
        positives = [i for i in range(50) if usable[i] and labels[i] == "direct"]
        negatives = [i for i in range(50) if usable[i] and labels[i] == "irrelevant" and i in selected]
        protected = [i for i in positives if i in selected]
        pairs = [[p, n] for p in positives if p not in selected for n in negatives]
        roles = ["held_not_negative"] * 50
        for i in positives:
            roles[i] = "protected_direct" if i in selected else "qualified_missed_direct"
        for i in negatives:
            roles[i] = "qualified_replaceable_irrelevant"
        drafts.append({"question_sha256": case["question_sha256"], "future_role": case["future_role"], "roles": roles,
                       "protected_indices": protected, "replaceable_indices": negatives,
                       "draft_pairs": pairs, "pair_weights": [1.0 / len(pairs)] * len(pairs) if pairs else [],
                       "maximum_replacement_slots": len(negatives), "scope_qualified": qready})
    return {"schema_version": "rag.support_axes_target_draft.v1", "trace_binding_sha256": digest(trace),
            "annotation_binding_sha256": digest(annotations), "training_authorized": False,
            "training_consumable": False, "cases": drafts,
            "draft_pair_count": sum(len(r["draft_pairs"]) for r in drafts)}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Validate versioned review and write a non-consumable draft; no training.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(args.output.resolve() not in {args.trace.resolve(), args.annotations.resolve()}, "preserve original inputs")
    draft = compile_draft(json.loads(args.trace.read_text(encoding="utf-8")),
                          json.loads(args.annotations.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as sink:
        json.dump(draft, sink, ensure_ascii=False, indent=2, allow_nan=False)
        sink.write("\n")
    print(json.dumps({"draft_pair_count": draft["draft_pair_count"], "training_authorized": False}))
