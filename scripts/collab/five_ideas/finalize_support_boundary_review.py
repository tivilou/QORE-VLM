#!/usr/bin/env python3
"""Apply explicit final decisions and close supervision inclusion/exclusion.

This script never infers semantic labels, fetches data or trains. A final
exclusion closes an audit item; it does not declare an irrelevant passage.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag import support_boundary_bridge as bridge
from applications.rag import support_supervision_axes as axes


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    axes.require(load(path) == value, "output round-trip failed")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_decisions(trace, annotations, inputs, manual):
    """Only explicitly authored decisions can amend an immutable overlay."""
    rows = copy.deepcopy(annotations)
    by_index = bridge.unique(manual["cases"], "case_index")
    expected = {c["usable_case_index"] for c in inputs["cases"]}
    axes.require(set(by_index) == expected, "final decision case coverage mismatch")
    addressed, excluded = [], []
    for c in inputs["cases"]:
        i = c["usable_case_index"] - 1
        case, row, decision = trace["cases"][i], rows["cases"][i], by_index[i + 1]
        axes.require(case["future_role"] == "training" and row["question_sha256"] == c["question_sha256"], "final decisions cross split/identity")
        available = bridge.unique(row["items"], "review_item_id")
        decisions = decision.get("items", [decision.get("all_items")] * len(c["items"]))
        axes.require(len(decisions) == len(c["items"]), "item decision coverage mismatch")
        if decision["close_scope"]:
            excluded.append(row["question_sha256"])
        for old, d in zip(c["items"], decisions):
            target = available[old["review_item_id"]]
            axes.require(target == old, "final decision binds stale opinion")
            names = ("support", "coverage", "inference", "grounding")
            new = dict(zip(axes.AXES, (d[k] for k in names)))
            axes.require(all(new[k] in values and new[k] not in {"uncertain", "unresolved"} for k, values in axes.AXES.items()), "final axes must be resolved")
            quotes = [d["quote"]] if "quote" in d else [q for q in d["quote_candidates"] if q in target["text"]]
            axes.require(quotes and all(q and q in target["text"] for q in quotes), "final quote not in original body")
            axes.require(d["disposition"] in {"accepted_protected", "accepted_supervision_candidate", "excluded_scope_ambiguity", "excluded_insufficient_passage", "excluded_conflicting_source"}, "unknown final disposition")
            target.update(new, rationale=d["reason"], support_quotes=quotes)
            # A new explicitly authored final version, not promotion of v1 held.
            target["qualification"] = axes.pending_qualification()
            if d["disposition"] != "accepted_supervision_candidate":
                target["qualification"]["status"] = "held"
            addressed.append({"review_item_id": target["review_item_id"], "question_sha256": row["question_sha256"],
                              "axes": new, "disposition": d["disposition"], "reason": d["reason"], "support_quotes": quotes})
    axes.validate_annotations(trace, rows)
    return rows, addressed, set(excluded)


def finalize(trace_path, annotations_path, input_path, manual_path, output, independent_paths, prior_dispositions):
    trace, original = load(trace_path), load(annotations_path)
    axes.validate_annotations(trace, original)
    inputs, manual = load(input_path), load(manual_path)
    axes.require(manual["schema_version"] == "rag.support_final_resolution_manual.v1" and
                 manual["input_sha256"] == sha(input_path) and manual["training_authorized"] is False, "manual source binding mismatch")
    expected = {p["review_item_id"] for c, r in zip(trace["cases"], original["cases"]) if c["future_role"] == "training"
                for p in r["items"] if p["qualification"]["status"] == "held" or any(p[k] in {"uncertain", "unresolved"} for k in axes.AXES)}
    received = [p["review_item_id"] for c in inputs["cases"] for p in c["items"]]
    axes.require(set(received) == expected and len(received) == len(expected), "not all current uncertainties covered")
    rows, resolved, excluded = apply_decisions(trace, original, inputs, manual)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    witness_dir = output / "witnesses"
    witness_dir.mkdir()
    sources = []
    for i, source in enumerate(independent_paths):
        dest = witness_dir / f"independent_{i + 1}.json"
        shutil.copyfile(source, dest)
        doc = load(dest)
        axes.require(doc["schema_version"] == "rag.support_axes_independent_review.v1", "independent schema mismatch")
        sources.append((dest, bridge.unique(doc["cases"], "question_sha256")))
    primary_path = witness_dir / "primary_final_composite.json"
    primary_cases = []
    for c, r in zip(trace["cases"], rows["cases"]):
        if c["future_role"] == "training":
            primary_cases.append({"question_sha256": r["question_sha256"], "review_case_id": f"final-{c['usable_case_index']}",
                "reviewer": {"identity": "primary-final-composite", "model_id": "exact_served_id_unavailable",
                             "session": "20261011-final-boundary-resolution", "prior_review_exposure": True,
                             "outside_packet_context_exposure": True},
                "scope": r["scope"], "items": [{k: p[k] for k in (*bridge.IDENTITY, *bridge.OPINION)} for p in r["items"]]})
    write(primary_path, {"schema_version": "rag.support_axes_independent_review.v1", "cases": primary_cases,
        "review_mode": "inherited_full_primary_plus_prior401_adjudications_plus_explicit104_final_decisions",
        "training_authorized": False, "training_consumable": False, "semantic_truth_certified": False})
    request = bridge.qualification_request(trace, rows)
    primary_sha = sha(primary_path)
    primary_ref = primary_path.relative_to(output).as_posix()

    def qualify(target, scope=False):
        binding = axes.qualification_binding(target, scope=scope)
        for path, cases in sources:
            case = cases.get(target["question_sha256"]) if scope else next((c for c in cases.values() if any(p["review_item_id"] == target["review_item_id"] for p in c["items"])), None)
            if not case or case["reviewer"]["prior_review_exposure"] or case["reviewer"]["outside_packet_context_exposure"]:
                continue
            if scope:
                match = case["scope"] == target["scope"]
            else:
                item = next(p for p in case["items"] if p["review_item_id"] == target["review_item_id"])
                match = all(item[k] == target[k] for k in (*bridge.IDENTITY, *axes.AXES))
            if not match:
                continue
            evidence = [{"reviewer": "primary-final-composite", "role": "primary_review", "artifact_ref": primary_ref,
                         "sha256": primary_sha, "annotation_sha256": binding, "prior_review_exposure": True},
                        {"reviewer": case["reviewer"]["identity"], "role": "independent_review", "artifact_ref": path.relative_to(output).as_posix(),
                         "sha256": sha(path), "annotation_sha256": binding, "prior_review_exposure": False}]
            return {"status": "qualified", "confidence": "independently_adjudicated", "evidence": evidence}
        return {"status": "held", "confidence": None, "evidence": []}

    by_q = bridge.unique(rows["cases"], "question_sha256")
    by_item = {p["review_item_id"]: p for r in rows["cases"] for p in r["items"]}
    for entry in request["scopes"]:
        r = by_q[entry["question_sha256"]]
        entry.update(qualify(r, scope=True) if r["question_sha256"] not in excluded else {"status": "held", "confidence": None, "evidence": []})
    for entry in request["items"]:
        p = by_item[entry["review_item_id"]]
        if p["qualification"]["status"] != "held":
            entry.update(qualify(p))
    reviewed = bridge.apply_explicit_qualification(trace, rows, request, output)
    draft, summary, details = bridge.protected_draft(trace, reviewed)
    # Explicit excluded questions never contribute pairs, even if old opinions match.
    closed = []
    for c, r, target in zip(trace["cases"], reviewed["cases"], draft["cases"]):
        if c["future_role"] != "training":
            continue
        if c["question_sha256"] in excluded:
            axes.require(not target["draft_pairs"], "excluded question emitted pairs")
        opportunities, _ = bridge.proposed_pairs(c, r)
        if target["draft_pairs"]:
            status, reason = "included", "Resolved direct/irrelevant endpoints and all selected-direct protection reopen against actual prior independent opinions."
        elif c["question_sha256"] in excluded:
            status, reason = "excluded", "Final question-scope or conflicting-source decision; not pending."
        elif opportunities:
            status, reason = "excluded", "Existing opinions do not jointly establish all target/protection bindings; this run closes exclusion without lowering witness criteria."
        else:
            status, reason = "no_safe_opportunity", "No defensible missed-direct to selected-irrelevant replacement in frozen opinions."
        closed.append({"question_sha256": c["question_sha256"], "status": status, "reason": reason,
                       "pair_count": len(target["draft_pairs"]), "replacement_slots": target["maximum_replacement_slots"]})
    inherited = load(prior_dispositions)["items"]
    final_by_id = {d["review_item_id"]: {**d, "disposition": "accepted_final"} for d in inherited if d["disposition"] == "accepted_provisional"}
    final_by_id.update({d["review_item_id"]: d for d in resolved})
    axes.require(len(final_by_id) == len(inherited) + len(expected - {d['review_item_id'] for d in inherited}), "final dispute coverage mismatch")
    summary.update(resolved_uncertainty_items=len(resolved), resolved_uncertainty_cases=len(inputs["cases"]),
                   final_dispute_and_uncertainty_items=len(final_by_id), excluded_scope_questions=len(excluded),
                   unresolved_audit_dispositions=0, primary_review_exposed=True,
                   independent_review_limitation="same-parent A/B; Claude exact model redacted; independence is declared, not certified")
    outputs = {"annotations.final.json": reviewed, "targets.final.json": draft, "qualification.final.json": request,
               "dispositions.final.json": {"items": list(final_by_id.values()), "training_authorized": False},
               "case_closure.final.json": {"cases": closed}, "summary.json": summary,
               "pair_audit.private.json": {"cases": details}}
    for name, value in outputs.items():
        write(output / name, value)
    manifest = {"schema_version": "rag.support_final_resolution_manifest.v1", "training_authorized": False,
        "training_consumable": False, "semantic_truth_certified": False,
        "input_sha256": {str(p): sha(p) for p in (trace_path, annotations_path, input_path, manual_path, prior_dispositions)},
        "files": [{"path": p.relative_to(output).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
                  for p in sorted(output.rglob("*.json"))]}
    write(output / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "annotations", "input", "manual", "output", "prior-dispositions"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--independent", action="append", required=True, type=Path)
    a = p.parse_args()
    finalize(a.trace, a.annotations, a.input, a.manual, a.output, a.independent, a.prior_dispositions)
