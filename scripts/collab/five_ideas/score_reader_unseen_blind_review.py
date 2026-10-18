#!/usr/bin/env python3
"""Merge blind reviews before unblinding, then score the frozen utility gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/experiments/reader_unseen_blind_review_87.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expected_pairs(packet: dict) -> list[tuple[str, str]]:
    return [(case["review_case_id"], item["review_item_id"])
            for case in packet["cases"] for item in case["items"]]


def validate_review(packet_path: Path, packet: dict, review: dict, cfg: dict) -> dict[tuple[str, str], dict]:
    if review.get("schema_version") != "rag.reader_unseen_blind_review.v1" or not review.get("completed"):
        raise ValueError("review is incomplete or has wrong schema")
    if review.get("packet_sha256") != digest(packet_path):
        raise ValueError("review packet hash mismatch")
    support = set(cfg["semantic_codes"]["answer_support"])
    scope = set(cfg["semantic_codes"]["scope_match"])
    indexed: dict[tuple[str, str], dict] = {}
    for row in review.get("judgments", []):
        pair = (row.get("review_case_id"), row.get("review_item_id"))
        if pair in indexed or row.get("answer_support") not in support or row.get("scope_match") not in scope:
            raise ValueError("invalid or duplicate review judgment")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError("every judgment needs a rationale")
        indexed[pair] = row
    if set(indexed) != set(expected_pairs(packet)):
        raise ValueError("review item coverage mismatch")
    return indexed


def merge(packet_path: Path, review_a_path: Path, review_b_path: Path, output: Path, disputes: Path,
          config_path: Path = CONFIG) -> dict:
    packet, cfg = load(packet_path), load(config_path)
    a = validate_review(packet_path, packet, load(review_a_path), cfg)
    b = validate_review(packet_path, packet, load(review_b_path), cfg)
    agreed, unresolved = [], []
    for pair in expected_pairs(packet):
        left, right = a[pair], b[pair]
        if (left["answer_support"], left["scope_match"]) == (right["answer_support"], right["scope_match"]):
            agreed.append({"review_case_id": pair[0], "review_item_id": pair[1],
                           "answer_support": left["answer_support"], "scope_match": left["scope_match"],
                           "reviewer_a": left, "reviewer_b": right})
        else:
            unresolved.append({"review_case_id": pair[0], "review_item_id": pair[1],
                               "reviewer_a": left, "reviewer_b": right,
                               "adjudicated_answer_support": None, "adjudicated_scope_match": None,
                               "adjudication_rationale": ""})
    write(disputes, {"schema_version": "rag.reader_unseen_blind_disputes.v1",
                     "packet_sha256": digest(packet_path), "unresolved": unresolved})
    consensus = {"schema_version": "rag.reader_unseen_blind_consensus.v1", "packet_sha256": digest(packet_path),
                 "complete": not unresolved, "judgments": agreed, "unresolved_count": len(unresolved)}
    write(output, consensus)
    return consensus


def label(row: dict) -> tuple[bool, bool, bool]:
    uncertain = row["answer_support"] == "uncertain" or row["scope_match"] == "uncertain"
    direct = row["answer_support"] == "direct" and row["scope_match"] == "yes"
    useful = row["answer_support"] in {"direct", "partial"} and row["scope_match"] in {"yes", "partial"}
    return direct, useful, uncertain


def score(packet_path: Path, consensus_path: Path, key_path: Path, output: Path, config_path: Path = CONFIG) -> dict:
    cfg, consensus = load(config_path), load(consensus_path)
    if not consensus.get("complete") or consensus.get("unresolved_count") != 0:
        raise ValueError("cannot unblind before complete blind consensus")
    if consensus.get("packet_sha256") != digest(packet_path):
        raise ValueError("consensus packet hash mismatch")
    key = load(key_path)
    if key.get("packet_sha256") != digest(packet_path):
        raise ValueError("sealed key packet hash mismatch")
    judgments = {(r["review_case_id"], r["review_item_id"]): label(r) for r in consensus["judgments"]}
    methods = ("identity_reader", "quantum", "classical")
    totals = {m: {"direct": 0, "useful": 0, "uncertain": 0, "points": 0} for m in methods}
    case_values = {m: {} for m in methods}
    for case in key["cases"]:
        membership = {row["source_index"]: row["review_item_id"] for row in case["items"]}
        for method in methods:
            direct = useful = uncertain = 0
            for index in case["selections"][method]:
                result = judgments[(case["review_case_id"], membership[index])]
                direct += int(result[0]); useful += int(result[1]); uncertain += int(result[2])
            points = useful + direct
            case_values[method][case["review_case_id"]] = points
            totals[method]["direct"] += direct
            totals[method]["useful"] += useful
            totals[method]["uncertain"] += uncertain
            totals[method]["points"] += points

    def comparison(left: str, right: str) -> dict:
        deltas = [case_values[left][case] - case_values[right][case] for case in case_values[left]]
        return {"wins": sum(v > 0 for v in deltas), "losses": sum(v < 0 for v in deltas),
                "ties": sum(v == 0 for v in deltas), "point_delta": sum(deltas)}

    q_i = comparison("quantum", "identity_reader")
    q_c = comparison("quantum", "classical")
    gate_cfg = cfg["prospective_gate"]
    checks = {
        "useful_over_identity": totals["quantum"]["useful"] - totals["identity_reader"]["useful"] >= gate_cfg["min_quantum_useful_gain_over_identity"],
        "direct_not_worse_than_identity": totals["quantum"]["direct"] - totals["identity_reader"]["direct"] >= gate_cfg["min_quantum_direct_gain_over_identity"],
        "useful_over_classical": totals["quantum"]["useful"] - totals["classical"]["useful"] >= gate_cfg["min_quantum_useful_gain_over_classical"],
        "case_win_margin_over_identity": q_i["wins"] - q_i["losses"] >= gate_cfg["min_quantum_case_win_margin_over_identity"],
        "uncertainty_bound": totals["quantum"]["uncertain"] <= gate_cfg["max_uncertain_selected_items"],
    }
    result = {
        "schema_version": "rag.reader_unseen_blind_review_score.v1",
        "claim_ceiling": cfg["claim_ceiling"],
        "population": {"changed_cases": len(key["cases"]), "unchanged_cases": cfg["transport_run"]["expected_validation_cases"] - len(key["cases"])},
        "totals_on_changed_cases": totals,
        "pairwise": {"quantum_vs_identity": q_i, "quantum_vs_classical": q_c},
        "gate_checks": checks,
        "advance_to_larger_untouched_utility_study": all(checks.values()),
        "limitations": ["Unchanged cases cancel across methods", "This is a small L0 semantic audit", "No quantum advantage or L1/L2 claim follows"],
        "provenance": {"packet_sha256": digest(packet_path), "consensus_sha256": digest(consensus_path), "sealed_key_sha256": digest(key_path), "config_sha256": digest(config_path)},
    }
    write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge_p = sub.add_parser("merge")
    merge_p.add_argument("--packet", type=Path, required=True)
    merge_p.add_argument("--review-a", type=Path, required=True)
    merge_p.add_argument("--review-b", type=Path, required=True)
    merge_p.add_argument("--output", type=Path, required=True)
    merge_p.add_argument("--disputes", type=Path, required=True)
    merge_p.add_argument("--config", type=Path, default=CONFIG)
    score_p = sub.add_parser("score")
    score_p.add_argument("--packet", type=Path, required=True)
    score_p.add_argument("--consensus", type=Path, required=True)
    score_p.add_argument("--key", type=Path, required=True)
    score_p.add_argument("--output", type=Path, required=True)
    score_p.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.command == "merge":
        value = merge(args.packet, args.review_a, args.review_b, args.output, args.disputes, args.config)
    else:
        value = score(args.packet, args.consensus, args.key, args.output, args.config)
    print(json.dumps(value, ensure_ascii=False, indent=2))
