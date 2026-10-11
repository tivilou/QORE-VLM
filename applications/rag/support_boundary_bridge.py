"""Exact-identity adjudication adapter and explicit witness audit, offline only.

Acceptance is an opinion disposition, never qualification. Scripts do not infer
labels or issue fitting permission. Full original cases remain in all outputs.
"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from applications.rag import support_supervision_axes as axes

IDENTITY = ("review_item_id", "text", "title", "text_sha256", "title_sha256")
OPINION = (*axes.AXES, "rationale", "support_quotes")
REQUEST_SCHEMA = "rag.support_boundary_qualification_request.v1"


def checked(path, expected):
    path = Path(path)
    axes.require(not path.is_symlink(), "source must not be a symlink")
    body = path.read_bytes()
    axes.require(hashlib.sha256(body).hexdigest() == expected, "artifact hash mismatch: " + path.name)
    return json.loads(body)


def unique(rows, key):
    result = {row[key]: row for row in rows}
    axes.require(len(result) == len(rows), "duplicate " + key)
    return result


def merge_opinions(trace, primary, review, dispositions):
    """Reorder by ID only; preserve opinion axes verbatim, no qualification."""
    annotations = axes.pending_template(trace)
    for source in (primary, review, dispositions):
        axes.require(source.get("training_authorized") is False and source.get("training_consumable") is False,
                     "review must not grant training permission")
    train = {c["question_sha256"] for c in trace["cases"] if c["future_role"] == "training"}
    originals = unique(primary["cases"], "question_sha256")
    axes.require(set(originals) == train, "primary must cover exactly the registered training split")
    revised = unique(review["cases"], "question_sha256")
    axes.require(set(revised) <= train, "adjudication crosses training boundary")
    decisions = unique(dispositions["items"], "review_item_id")
    revised_items = {p["review_item_id"] for c in review["cases"] for p in c["items"]}
    axes.require(set(decisions) == revised_items, "disposition coverage mismatch")
    held = set()
    for row in annotations["cases"]:
        qid = row["question_sha256"]
        if qid not in train:
            continue  # Reserved validation remains entirely producer-null.
        old = originals[qid]
        axes.require(all(row[k] == old[k] for k in ("question", "reference_answers")), "question/reference changed")
        source_items = unique(old["items"], "review_item_id")
        axes.require(set(source_items) == {p["review_item_id"] for p in row["items"]}, "primary Top50 mismatch")
        row["scope"] = copy.deepcopy(old["scope"])
        amended = {}
        if qid in revised:
            r = revised[qid]
            row["scope"] = copy.deepcopy(r["scope"])
            amended = unique(r["items"], "review_item_id")
            axes.require(set(amended) <= set(source_items), "adjudicated item outside Top50")
            for rid, item in amended.items():
                d = decisions[rid]
                axes.require(d["review_case_id"] == r["review_case_id"] and
                             d["axes"] == {k: item[k] for k in axes.AXES}, "disposition/opinion mismatch")
                axes.require(d["disposition"] in {"held", "accepted_provisional"}, "unknown disposition")
                if d["disposition"] == "held":
                    held.add(rid)
        for item in row["items"]:
            rid = item["review_item_id"]
            original = source_items[rid]
            source = amended.get(rid, original)
            axes.require(all(item[k] == original[k] == source[k] for k in IDENTITY), "immutable passage identity mismatch")
            for key in OPINION:
                item[key] = copy.deepcopy(source[key])
            if rid in held:
                item["qualification"]["status"] = "held"
    axes.validate_annotations(trace, annotations)
    return annotations, held


def proposed_pairs(case, row):
    """Diagnostic capacity from existing opinions, NOT qualified supervision."""
    labels = {p["review_item_id"]: p for p in row["items"]}
    selected = set(case["reader_top5_indices"])
    usable = {i for i, p in enumerate(case["candidates"])
              if labels[p["review_item_id"]]["qualification"]["status"] != "held" and
              labels[p["review_item_id"]]["source_grounding"] == "single_passage" and
              labels[p["review_item_id"]]["reference_coverage"] != "unresolved" and
              labels[p["review_item_id"]]["inference_type"] != "unresolved"}
    positives = [i for i in sorted(usable - selected)
                 if labels[case["candidates"][i]["review_item_id"]]["question_support"] == "direct" and
                 labels[case["candidates"][i]["review_item_id"]]["inference_type"] != "none"]
    negatives = [i for i in sorted(usable & selected)
                 if labels[case["candidates"][i]["review_item_id"]]["question_support"] == "irrelevant"]
    return [[p, n] for p in positives for n in negatives], min(len(positives), len(negatives))


def qualification_request(trace, annotations):
    """Explicit pending entries, no manufactured witness or consensus promotion."""
    axes.validate_annotations(trace, annotations)
    request = {"schema_version": REQUEST_SCHEMA, "annotations_sha256": axes.digest(annotations),
               "training_authorized": False, "scopes": [], "items": []}
    for case, row in zip(trace["cases"], annotations["cases"]):
        if case["future_role"] != "training":
            continue
        pairs, _ = proposed_pairs(case, row)
        if not pairs:
            continue
        request["scopes"].append({"question_sha256": row["question_sha256"],
                                  "binding_sha256": axes.qualification_binding(row, scope=True),
                                  **axes.pending_qualification()})
        # Every original selection is included for protection review, not just negatives.
        needed = set(case["reader_top5_indices"]) | {i for pair in pairs for i in pair}
        by_id = {p["review_item_id"]: p for p in row["items"]}
        for i in sorted(needed):
            item = by_id[case["candidates"][i]["review_item_id"]]
            request["items"].append({"review_item_id": item["review_item_id"],
                                     "binding_sha256": axes.qualification_binding(item),
                                     **axes.pending_qualification(),
                                     "status": "held" if item["qualification"]["status"] == "held" else "pending"})
    return request


def actual_witness(root, witness, row, *, scope=False):
    """Reopen a real review and check the addressed scope/item, not just a hash."""
    relative = Path(witness["artifact_ref"])
    path = (Path(root) / relative).resolve()
    axes.require(not relative.is_absolute() and path.is_relative_to(Path(root).resolve()), "witness escapes packet")
    doc = checked(path, witness["sha256"])
    axes.require(doc.get("schema_version") == "rag.support_axes_independent_review.v1", "unsupported actual witness")
    if scope:
        cases = [c for c in doc["cases"] if c["question_sha256"] == row["question_sha256"]]
    else:
        cases = [c for c in doc["cases"] if any(p["review_item_id"] == row["review_item_id"] for p in c["items"])]
    axes.require(len(cases) == 1, "witness does not address target")
    case = cases[0]
    reviewer = case["reviewer"]
    axes.require(reviewer["identity"] == witness["reviewer"] and
                 reviewer["prior_review_exposure"] == witness["prior_review_exposure"], "witness identity/exposure mismatch")
    if witness["role"] == "independent_review":
        axes.require(reviewer["outside_packet_context_exposure"] is False, "independent witness outside-exposed")
    if scope:
        axes.require(case["scope"] == row["scope"], "scope witness differs from approved scope")
    else:
        source = next(p for p in case["items"] if p["review_item_id"] == row["review_item_id"])
        axes.require(all(source[k] == row[k] for k in (*IDENTITY, *axes.AXES)), "witness item identity/axes differ")
        axes.require(bool(source["rationale"]) and all(q in row["text"] for q in source["support_quotes"]), "witness rationale/quote invalid")
        if source["question_support"] in {"direct", "partial", "contradictory"}:
            axes.require(bool(source["support_quotes"]), "support witness lacks quote")


def apply_explicit_qualification(trace, annotations, request, witness_root):
    axes.require(request["schema_version"] == REQUEST_SCHEMA and request["training_authorized"] is False and
                 request["annotations_sha256"] == axes.digest(annotations), "stale qualification request")
    result = copy.deepcopy(annotations)
    eligible = {c["question_sha256"] for c in trace["cases"] if c["future_role"] == "training"}
    scopes = unique(request["scopes"], "question_sha256")
    items = unique(request["items"], "review_item_id")
    axes.require(set(scopes) <= eligible, "validation qualification forbidden by train-only adapter")
    used = set()
    for row in result["cases"]:
        records = [(scopes[row["question_sha256"]], row, True)] if row["question_sha256"] in scopes else []
        records += [(items[p["review_item_id"]], p, False) for p in row["items"] if p["review_item_id"] in items]
        for entry, target, scope in records:
            axes.require(row["question_sha256"] in eligible, "validation item qualification forbidden")
            if not scope:
                used.add(target["review_item_id"])
            expected = axes.qualification_binding(target, scope=scope)
            axes.require(entry["binding_sha256"] == expected, "stale target binding")
            qual = {k: copy.deepcopy(entry[k]) for k in ("status", "confidence", "evidence")}
            axes.require(scope or target["qualification"]["status"] != "held" or qual["status"] == "held", "held disposition cannot be promoted")
            if axes.qualified(qual, expected):
                for witness in qual["evidence"]:
                    actual_witness(witness_root, witness, target, scope=scope)
            target["scope_qualification" if scope else "qualification"] = qual
    axes.require(used == set(items), "qualification target missing")
    axes.validate_annotations(trace, result)
    return result


def protected_draft(trace, annotations, *, residual_bound=0.25):
    """Strengthen existing draft with whole-selected-direct and set protection."""
    draft = axes.compile_draft(trace, annotations)
    axes.require(math.isfinite(residual_bound) and residual_bound > 0, "invalid residual bound")
    counts = Counter()
    details = []
    for case, row, target in zip(trace["cases"], annotations["cases"], draft["cases"]):
        by_id = {p["review_item_id"]: p for p in row["items"]}
        selected = set(case["reader_top5_indices"])
        labels = {i: by_id[case["candidates"][i]["review_item_id"]]["question_support"] for i in selected}
        missing_protection = [i for i in sorted(selected) if labels[i] == "direct" and i not in target["protected_indices"]]
        if case["future_role"] != "training" or missing_protection:
            target["draft_pairs"] = []; target["pair_weights"] = []
        pairs = target["draft_pairs"]
        positive = sorted({p for p, _ in pairs}); negative = sorted({n for _, n in pairs})
        target["baseline_top5_indices"] = list(case["reader_top5_indices"])
        target["eligible_positive_indices"] = positive
        target["replaceable_indices"] = negative
        target["immutable_selected_indices"] = sorted(selected - set(negative))
        target["maximum_replacement_slots"] = min(len(positive), len(negative))
        target["missing_selected_direct_qualification"] = missing_protection
        if case["future_role"] == "training":
            proposed, slots = proposed_pairs(case, row)
            scores = [p["reader_score"] for p in case["candidates"]]
            mean = sum(scores) / 50
            std = max(math.sqrt(sum((v - mean) ** 2 for v in scores) / 50), 1e-6)
            upper = 2 * residual_bound * std
            reachable = [p for p in proposed if scores[p[1]] - scores[p[0]] < upper]
            counts["proposed_pairs_within_loose_residual_bound"] += len(reachable)
            counts["proposed_questions"] += bool(proposed); counts["proposed_pairs"] += len(proposed)
            counts["proposed_replacement_slots"] += slots
            counts["qualified_questions"] += bool(pairs); counts["qualified_pairs"] += len(pairs)
            counts["qualified_replacement_slots"] += target["maximum_replacement_slots"]
            counts["questions_with_unqualified_selected_direct"] += bool(missing_protection)
            details.append({"question_sha256": row["question_sha256"], "proposed_pairs": proposed,
                            "qualified_pairs": pairs, "missing_selected_direct_qualification": missing_protection,
                            "reader_gaps": [scores[n] - scores[p] for p, n in proposed],
                            "loose_max_pair_correction": upper, "within_loose_bound_count": len(reachable)})
    draft["schema_version"] = "rag.protected_support_boundary_draft.v1"
    draft["draft_pair_count"] = sum(len(r["draft_pairs"]) for r in draft["cases"])
    summary = {k: int(counts[k]) for k in ("proposed_questions", "proposed_pairs", "proposed_replacement_slots",
               "qualified_questions", "qualified_pairs", "qualified_replacement_slots", "questions_with_unqualified_selected_direct",
               "proposed_pairs_within_loose_residual_bound")}
    summary.update(training_cases=sum(c["future_role"] == "training" for c in trace["cases"]),
                   validation_cases=sum(c["future_role"] == "validation" for c in trace["cases"]),
                   held_items=sum(p["qualification"]["status"] == "held" for r in annotations["cases"] for p in r["items"]),
                   training_authorized=False, training_consumable=False, semantic_truth_certified=False)
    return draft, summary, details
