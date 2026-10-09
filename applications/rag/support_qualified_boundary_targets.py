"""Offline, fail-closed draft targets; never imported by the online scorer.

No semantic inference or training: a qualification overlay is an external review
record, not a consequence of weak containment, an absent flag, or a model score.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

LABELS = {"direct", "partial", "irrelevant", "contradictory", "uncertain"}
SCOPE_FIELDS = {"time", "location", "version", "entity", "metric", "reference_relation"}
SCHEMA = "rag.support_qualification_overlay.v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _index(rows: list[dict], key: str) -> dict:
    result = {r[key]: r for r in rows}
    _require(len(result) == len(rows), "duplicate " + key)
    return result


def _validated(trace: dict, review: dict) -> tuple[dict, dict, dict]:
    cases = _index(trace["cases"], "question_sha256")
    judgments = _index(review["judgments"], "review_item_id")
    flags = _index(review.get("question_target_flags", []), "review_case_id")
    items = {}
    for qid, c in cases.items():
        _require(hashlib.sha256(c["question"].encode("utf-8")).hexdigest() == qid,
                 "question identity mismatch")
        _require(c.get("lane") == "previous_training_diagnostic_not_heldout",
                 "only old-training diagnostic lane accepted; Silver panel excluded")
        _require(len(c["candidates"]) == 50, "expected fixed Top50")
        scores = []
        for i, p in enumerate(c["candidates"]):
            rid = p["question_passage_review_id"]
            _require(rid not in items, "duplicate candidate identity")
            _require(hashlib.sha256(p["text"].encode("utf-8")).hexdigest() == p["text_sha256"],
                     "passage text hash mismatch")
            _require(type(p["reader_score"]) in (int, float) and math.isfinite(p["reader_score"]),
                     "nonfinite/invalid Reader score")
            items[rid] = (qid, i, p)
            scores.append(p["reader_score"])
        expected = sorted(range(50), key=lambda i: (-scores[i], i))[:5]
        _require(c["methods"]["frozen_reader_topk"]["selected_indices"] == expected,
                 "frozen Reader selection changed")
    _require(set(flags) <= set(cases), "unknown flagged question")
    for rid, r in judgments.items():
        _require(rid in items, "unknown reviewed item")
        qid, i, p = items[rid]
        _require(r["review_case_id"] == qid, "review question mismatch")
        _require(r.get("candidate_index", i) == i, "review candidate order mismatch")
        _require(r["support_label"] in LABELS and bool(r.get("rationale", "").strip()),
                 "invalid review label/rationale")
        quote = r.get("support_quote")
        _require(quote is None or (isinstance(quote, str) and quote.strip() and quote in p["text"]),
                 "review quote mismatch")
        _require(r["support_label"] not in {"direct", "partial", "contradictory"} or quote is not None,
                 "support-bearing review requires exact quote")
        _require(r.get("source_text_sha256", p["text_sha256"]) == p["text_sha256"],
                 "review text identity mismatch")
    return cases, judgments, items


def binding(trace: dict, review: dict) -> str:
    """Binds complete inputs, including label metadata and frozen candidate order."""
    _validated(trace, review)
    return digest({"trace": trace, "review": review})


def pending_overlay(trace: dict, review: dict) -> dict:
    """Preparation only: every status remains pending; no confidence upgrade."""
    cases, judgments, _ = _validated(trace, review)
    questions, items = [], []
    for qid, c in cases.items():
        questions.append({"review_case_id": qid, "status": "pending",
                          "scope": {key: "" for key in sorted(SCOPE_FIELDS)}, "evidence": []})
        selected = set(c["methods"]["frozen_reader_topk"]["selected_indices"])
        for i, p in enumerate(c["candidates"]):
            rid = p["question_passage_review_id"]
            r = judgments.get(rid)
            # Partial/unknown remain outside binary targets. Include retention witnesses.
            if r and (r["support_label"] == "direct" or
                      (i in selected and r["support_label"] == "irrelevant")):
                items.append({"review_item_id": rid, "review_case_id": qid,
                              "source_text_sha256": p["text_sha256"],
                              "review_judgment_sha256": digest(r), "support_label": r["support_label"],
                              "status": "pending", "evidence": []})
    return {"schema_version": SCHEMA, "input_binding_sha256": binding(trace, review),
            "questions": questions, "items": items}


def _qualified(row: dict | None) -> bool:
    if row is None or row.get("status") != "qualified":
        return False
    _require(row.get("confidence") == "independently_adjudicated",
             "qualified record requires independent adjudication")
    evidence = row.get("evidence", [])
    _require(len(evidence) >= 2, "qualified record requires two review witnesses")
    reviewers, hashes = set(), set()
    for e in evidence:
        _require(e.get("role") in {"primary_review", "independent_review"}, "invalid witness role")
        _require(bool(e.get("reviewer", "").strip()) and bool(e.get("artifact_ref", "").strip()),
                 "review witness missing identity/reference")
        h = e.get("sha256", "")
        _require(len(h) == 64 and all(ch in "0123456789abcdef" for ch in h), "invalid witness hash")
        reviewers.add(e["reviewer"])
        hashes.add(h)
    _require(len(reviewers) >= 2 and len(hashes) >= 2 and
             {e["role"] for e in evidence} == {"primary_review", "independent_review"},
             "independent witnesses must have distinct identities and artifacts")
    # This checks the contract, NOT the truth/independence of submitted witnesses.
    return True


def compile_targets(trace: dict, review: dict, overlay: dict | None = None) -> dict:
    """Emit proposals separately from qualification-checked drafts; training stays off."""
    cases, judgments, items = _validated(trace, review)
    locked = binding(trace, review)
    overlay = pending_overlay(trace, review) if overlay is None else overlay
    _require(overlay.get("schema_version") == SCHEMA, "qualification schema mismatch")
    _require(overlay.get("input_binding_sha256") == locked, "stale qualification overlay")
    qs = _index(overlay["questions"], "review_case_id")
    its = _index(overlay["items"], "review_item_id")
    _require(set(qs) <= set(cases) and set(its) <= set(items), "unknown qualification identity")
    for qid, row in qs.items():
        _require(row.get("status") in {"pending", "abstained", "qualified"}, "invalid question status")
        if _qualified(row):
            scope = row.get("scope", {})
            _require(set(scope) == SCOPE_FIELDS and all(isinstance(v, str) and v.strip() for v in scope.values()),
                     "all scope dimensions require explicit resolution, including inapplicability")
    for rid, row in its.items():
        qid, _, p = items[rid]
        r = judgments.get(rid)
        _require(r is not None, "missing judgment is not a target")
        _require(row.get("status") in {"pending", "abstained", "qualified"}, "invalid item status")
        _require(row["review_case_id"] == qid and row["source_text_sha256"] == p["text_sha256"] and
                 row["review_judgment_sha256"] == digest(r) and row["support_label"] == r["support_label"],
                 "qualification label/source binding mismatch")
        _qualified(row)
    flags = {r["review_case_id"] for r in review.get("question_target_flags", [])}
    rows = []
    for number, (qid, c) in enumerate(cases.items(), 1):
        selected = c["methods"]["frozen_reader_topk"]["selected_indices"]
        labels = [judgments.get(p["question_passage_review_id"], {}).get("support_label") for p in c["candidates"]]
        direct = [i for i, label in enumerate(labels) if label == "direct"]
        kept = [i for i in selected if labels[i] == "direct"]
        missed = [i for i in direct if i not in selected]
        negatives = [i for i in selected if labels[i] == "irrelevant"]
        proposals = [[p, n] for p in missed for n in negatives] if qid not in flags else []
        def qualified_item(i):
            return _qualified(its.get(c["candidates"][i]["question_passage_review_id"]))
        blockers = []
        if qid in flags:
            blockers.append("existing_question_target_flag")
        if not _qualified(qs.get(qid)):
            blockers.append("question_scope_not_qualified")
        if any(not qualified_item(i) for i in kept):
            blockers.append("selected_direct_retention_not_qualified")
        pairs = [[p, n] for p, n in proposals if qualified_item(p) and qualified_item(n)] if not blockers else []
        if proposals and not pairs and not blockers:
            blockers.append("boundary_items_not_qualified")
        weight = 1.0 / len(pairs) if pairs else 0.0
        rows.append({"audit_case_number": number, "review_case_id": qid,
                     "frozen_selected_indices": selected[:], "reviewed_label_counts": dict(Counter(x or "missing" for x in labels)),
                     "missing_review_count": labels.count(None), "provisional_direct_indices": direct,
                     "provisional_direct_slot_gap": min(5, len(direct)) - len(kept),
                     "proposal_pairs": proposals, "blockers": blockers,
                     "draft_pairs": [{"positive_index": p, "negative_index": n, "weight": weight} for p, n in pairs],
                     "protected_direct_indices": kept if not blockers else [],
                     "max_irrelevant_replacements": min(len(missed), len(negatives)),
                     "qualified_replacement_budget": min(len({p for p, n in pairs}), len({n for p, n in pairs})),
                     "partial_policy": "abstain_not_negative", "direct_vs_direct_policy": "no_forced_preference"})
    return {"schema_version": "rag.support_boundary_target_draft.v1", "input_binding_sha256": locked,
            "qualification_overlay_sha256": digest(overlay), "claim_ceiling": "L0_engineering_diagnostic",
            "training_consumable": False, "training_authorized": False, "full434_qualified": False,
            "witness_integrity_and_independence_verified": False,
            "summary": {"cases": len(rows), "proposal_cases": sum(bool(r["proposal_pairs"]) for r in rows),
                        "proposal_pairs": sum(len(r["proposal_pairs"]) for r in rows),
                        "draft_pair_cases": sum(bool(r["draft_pairs"]) for r in rows),
                        "draft_pairs": sum(len(r["draft_pairs"]) for r in rows),
                        "flagged_cases": len(flags)}, "cases": rows}


def validate_protected_selection(target_case: dict, selected_indices: list[int]) -> None:
    """Prospective set-level constraint; does not change any production selection."""
    _require(len(selected_indices) == 5 and all(type(i) is int and 0 <= i < 50 for i in selected_indices)
             and len(set(selected_indices)) == 5, "expected five unique valid indices")
    _require(set(target_case["protected_direct_indices"]) <= set(selected_indices),
             "confirmed selected direct evidence dropped")
