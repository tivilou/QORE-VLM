"""Offline, post-hoc audit of a registered Q-ARCG selector screen."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

METHODS = (
    "frozen_reader_topk",
    "trained_q_arcg",
    "trained_same_budget_classical_control",
)
LABEL_FIELDS = {
    "positive_consensus": "positive_consensus",
    "direct_consensus": "direct_consensus",
    "all_models_positive": "positive_all_models",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def paired(values: list[int], reference: list[int], seed: int) -> dict[str, Any]:
    delta = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
    return {
        "mean_delta": round(float(delta.mean()), 6),
        "ci95": [round(float(x), 6) for x in np.quantile(boot, [0.025, 0.975])],
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
    }


def audit(detail: dict[str, Any], trace: dict[str, Any], reader: dict[str, Any],
          detail_hash: str, expected_cases: int = 100) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if trace.get("input_sha256") != detail_hash or reader.get("input", {}).get("sha256") != detail_hash:
        raise ValueError("input hash mismatch")
    if trace.get("schema_version") != "sample-trace.v2":
        raise ValueError("unexpected selector trace schema")
    dc = detail["cases"]
    tc = trace["cases"]
    if len(dc) != expected_cases or len(tc) != expected_cases:
        raise ValueError("case coverage mismatch")
    by_case = {int(c["case_number"]): c for c in tc}
    if len(by_case) != expected_cases or len({c["case_number"] for c in dc}) != expected_cases:
        raise ValueError("duplicate case number")
    rr = {(int(r["case_number"]), str(r["candidate_id"])): r for r in reader["cases"]}
    if len(rr) != expected_cases * 50 or len(reader["cases"]) != expected_cases * 50:
        raise ValueError("Reader candidate coverage mismatch")
    metrics = {m: {key: [] for key in (*LABEL_FIELDS, "silver_overlap")} for m in METHODS}
    oracle_labels: Counter[str] = Counter()
    available_positive_counts: list[int] = []
    available_direct_counts: list[int] = []
    swaps: list[dict[str, Any]] = []
    private_cases: list[dict[str, Any]] = []
    max_reader_delta = 0.0
    unique_questions: set[str] = set()
    for case in dc:
        number = int(case["case_number"])
        ct = by_case[number]
        question_hash = digest(str(case["question"]))
        if question_hash != ct["question_sha256"] or question_hash in unique_questions:
            raise ValueError("question identity mismatch or duplicate")
        unique_questions.add(question_hash)
        passages = case["top_50"]
        ids = [digest(str(p["id"])) for p in passages]
        if len(ids) != 50 or len(set(ids)) != 50 or ids != ct["candidate_id_sha256"]:
            raise ValueError("candidate order or identity mismatch")
        index = {h: i for i, h in enumerate(ids)}
        oracle = next(s for s in case["selectors"] if s["selector_id"] == "silver_oracle_common_order")
        silver_ids = {digest(str(p["id"])) for p in oracle["selected_top_5"]}
        if len(silver_ids) != 5 or not silver_ids <= set(ids):
            raise ValueError("Silver reference identity mismatch")
        for h in silver_ids:
            oracle_labels[str(passages[index[h]]["evidence"]["consensus_label"])] += 1
        positive_count = sum(p["evidence"]["positive_consensus"] for p in passages)
        direct_count = sum(p["evidence"]["direct_consensus"] for p in passages)
        available_positive_counts.append(positive_count)
        available_direct_counts.append(direct_count)
        for field, count in (("positive_consensus", positive_count), ("direct_consensus", direct_count)):
            retained = sum(passages[index[h]]["evidence"][field] for h in silver_ids)
            if retained != min(5, count):
                raise ValueError("Silver reference does not attain the panel-label ceiling")
        if set(ct["methods"]) != set(METHODS):
            raise ValueError("method coverage mismatch")
        for method in METHODS:
            mt = ct["methods"][method]
            scores = np.asarray(mt["score_vector"], dtype=np.float64)
            selected = mt["selected_id_sha256"]
            if scores.shape != (50,) or not np.isfinite(scores).all():
                raise ValueError("score shape or finiteness failure")
            if len(selected) != 5 or len(set(selected)) != 5 or not set(selected) <= set(ids):
                raise ValueError("invalid selected set")
            overlap = len(set(selected) & silver_ids)
            if overlap != mt["silver_overlap"]:
                raise ValueError("recorded Silver overlap mismatch")
            metrics[method]["silver_overlap"].append(overlap)
            for key, field in LABEL_FIELDS.items():
                labels = [passages[index[h]]["evidence"][field] for h in selected]
                if not all(type(value) is bool for value in labels):
                    raise ValueError("invalid panel label type")
                metrics[method][key].append(sum(labels))
        base = ct["methods"][METHODS[0]]
        quantum = ct["methods"][METHODS[1]]
        reader_rows = [rr[number, str(p["id"])] for p in passages]
        errors = np.abs(np.asarray(base["score_vector"]) - np.asarray([r["relevance_logit"] for r in reader_rows]))
        max_reader_delta = max(max_reader_delta, float(errors.max()))
        if errors.max() > 1e-4:
            raise ValueError("historical Reader relevance drift exceeds tolerance")
        order = sorted(range(50), key=lambda i: (-base["score_vector"][i], passages[i]["retrieved_rank"], str(passages[i]["id"])))
        ranks = {ids[i]: rank + 1 for rank, i in enumerate(order)}
        added = set(quantum["selected_id_sha256"]) - set(base["selected_id_sha256"])
        removed = set(base["selected_id_sha256"]) - set(quantum["selected_id_sha256"])
        if not added:
            continue
        change: dict[str, Any] = {"case_number": number, "added": [], "removed": []}
        private = {"case_number": number, "question": case["question"], "gold_answers": case["gold_answers"], "added": [], "removed": []}
        for role, members in (("added", added), ("removed", removed)):
            for h in sorted(members, key=lambda value: ranks[value]):
                i = index[h]
                p, r = passages[i], reader_rows[i]
                item = {
                    "reader_score_rank": ranks[h],
                    "retrieved_rank": p["retrieved_rank"],
                    "consensus_label": p["evidence"]["consensus_label"],
                    "positive_consensus": p["evidence"]["positive_consensus"],
                    "direct_consensus": p["evidence"]["direct_consensus"],
                    "in_silver_top5": h in silver_ids,
                    "base_score": base["score_vector"][i],
                    "score_correction": round(quantum["score_vector"][i] - base["score_vector"][i], 7),
                    "historical_reader": {key: r[key] for key in ("span_logit", "span_margin", "start_entropy", "end_entropy", "truncated")},
                }
                change[role].append(item)
                private[role].append({**item, "id": p["id"], "title": p["title"], "text": p["text"], "model_labels": p["evidence"]["models"]})
        for key in metrics[METHODS[0]]:
            change[key + "_delta"] = metrics[METHODS[1]][key][-1] - metrics[METHODS[0]][key][-1]
        swaps.append(change)
        private_cases.append(private)
    aggregate = {
        method: {key: {"total": sum(values), "mean": round(float(np.mean(values)), 6)} for key, values in fields.items()}
        for method, fields in metrics.items()
    }
    comparisons = {
        method: {key: paired(values, metrics[METHODS[0]][key], 20261006 + offset) for key, values in metrics[method].items()}
        for offset, method in enumerate(METHODS[1:])
    }
    summary = {
        "schema_version": "rag.qarcg_case_audit.v1",
        "diagnostic_only": True,
        "analysis_selection": "all cases with changed Q-ARCG Top-5 membership; posthoc exploratory",
        "case_count": len(dc),
        "changed_case_count": len(swaps),
        "replacements": sum(len(s["added"]) for s in swaps),
        "methods": aggregate,
        "paired_vs_reader": comparisons,
        "silver_reference_label_counts": dict(sorted(oracle_labels.items())),
        "panel_label_ceiling": {
            "positive_slots": sum(min(5, n) for n in available_positive_counts),
            "direct_slots": sum(min(5, n) for n in available_direct_counts),
            "cases_with_fewer_than_five_positive_candidates": sum(n < 5 for n in available_positive_counts),
            "cases_without_positive_candidates": sum(n == 0 for n in available_positive_counts),
            "reference_attains_count_ceiling": True,
        },
        "historical_reader_max_relevance_delta": max_reader_delta,
        "swaps": swaps,
        "limitations": [
            "Panel labels and reference membership are Silver diagnostics, not official gold or unique optimal sets.",
            "Historical span fields are from a separate frozen Reader trace, not recorded per-candidate Q-ARCG inputs.",
            "Training losses, usable training count, executed code revision and per-candidate gates remain unavailable.",
            "Posthoc supplementary metrics do not replace the original preregistered membership gate.",
        ],
    }
    return summary, private_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reader-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-cases", type=Path, help="Optional raw-text artifact; never publish to GitHub")
    args = parser.parse_args()
    detail, dh = load(args.detail)
    trace, th = load(args.trace)
    reader, rh = load(args.reader_trace)
    summary, private = audit(detail, trace, reader, dh)
    summary["source_hashes"] = {"detail": dh, "selector_trace": th, "historical_reader_trace": rh}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.private_cases:
        args.private_cases.parent.mkdir(parents=True, exist_ok=True)
        args.private_cases.write_text(json.dumps({"exchange_only": True, "source_hashes": summary["source_hashes"], "cases": private}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": summary["case_count"], "changed_cases": summary["changed_case_count"], "methods": summary["methods"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
