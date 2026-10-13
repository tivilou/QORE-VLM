#!/usr/bin/env python3
"""Post-hoc score geometry of the existing protection objective; no training."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def pair_geometry(base, positive, negative, immutable, amplitude):
    """Necessary feasibility bounds with independently adjustable score boxes.

    This optimistic relaxation does not certify feasibility of a shared head,
    all pairs at once, or unmasked Top-5. Strict pair crossing excludes ties.
    """
    bp, bn = base[positive], base[negative]
    min_kept = min((base[i] for i in immutable), default=math.inf)
    baseline_conflict = bn >= min_kept
    # p>n and p<=every kept. Each score may move anywhere in [-A,+A].
    lower = max(bp - amplitude, bn - amplitude)
    upper = min(bp + amplitude, min_kept + amplitude)
    feasible = upper > lower
    return {
        "positive_index": positive, "negative_index": negative,
        "pair_initial_gap": bn - bp,
        "baseline_min_immutable_minus_negative": None if not immutable else min_kept - bn,
        "baseline_protection_order_conflict": baseline_conflict,
        "pair_box_reachable": bp + amplitude > bn - amplitude,
        "pair_and_zero_protection_box_reachable": feasible,
        "zero_protection_feasible_interval_width": max(0.0, upper - lower),
        "one_sided_amplitude": amplitude,
    }


def audit(document, bound):
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("positive finite residual bound required")
    records = []
    boundary_records = []
    train = [r for r in document["cases"] if r.get("lane") == "training"]
    if len(train) != 59:
        raise ValueError("expected frozen59 training cases")
    for row in train:
        base = row["methods"]["frozen_reader_topk"]["score_vector"]
        if len(base) != 50 or not all(math.isfinite(v) for v in base):
            raise ValueError("finite Top50 required")
        t = row["training_target"]
        amplitude = bound * max(statistics.pstdev(base), 1e-6)
        fifth = min(base[i] for i in t["baseline_top5_indices"])
        for p in t["eligible_positive_indices"]:
            boundary_records.append({
                "question_sha256": t["question_sha256"], "positive_index": p,
                "base_fifth_minus_positive": fifth - base[p],
                "optimistic_boundary_box_reachable": fifth - base[p] < 2 * amplitude,
            })
        for p, n in t["draft_pairs"]:
            r = pair_geometry(base, p, n, t["immutable_selected_indices"], amplitude)
            r["question_sha256"] = t["question_sha256"]
            records.append(r)
    if len(records) != 228:
        raise ValueError("expected frozen228 pairs")
    return {
        "schema_version": "rag.protected_objective_geometry.v1",
        "claim_ceiling": "L0_diagnostic", "residual_bound": bound,
        "summary": {
            "training_cases": len(train), "pairs": len(records),
            "baseline_protection_order_conflicts": sum(r["baseline_protection_order_conflict"] for r in records),
            "pair_box_reachable": sum(r["pair_box_reachable"] for r in records),
            "pair_and_zero_protection_box_reachable": sum(r["pair_and_zero_protection_box_reachable"] for r in records),
            "unique_missed_direct_candidates": len(boundary_records),
            "optimistic_boundary_box_reachable_candidates": sum(r["optimistic_boundary_box_reachable"] for r in boundary_records),
            "questions_with_any_optimistic_boundary_box_reachable_candidate": len({r["question_sha256"] for r in boundary_records if r["optimistic_boundary_box_reachable"]}),
        },
        "limits": [
            "Baseline order conflict is not proof of mutually impossible losses: kept scores may move.",
            "Independent score-box feasibility is a necessary optimistic bound, not shared-head learnability.",
            "All-pair inversions are stronger than filling available Top5 slots; train goals must distinguish them.",
            "No protected loss modification or fitting was executed.",
        ], "pair_records": records, "boundary_records": boundary_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-bound", type=float, default=.25)
    args = parser.parse_args()
    content = args.input.read_bytes()
    result = audit(json.loads(content), args.residual_bound)
    result["input_sha256"] = hashlib.sha256(content).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
