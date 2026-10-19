#!/usr/bin/env python3
"""Merge, blind-adjudicate, and prospectively score the 256-question review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/experiments/reader_larger_untouched_blind_review_256.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_reader_unseen_blind_review as base  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def clustered_bootstrap(deltas: list[int], config: dict) -> dict:
    if not deltas:
        raise ValueError("bootstrap requires question clusters")
    resamples = int(config["resamples"])
    confidence = float(config["confidence"])
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    rng = random.Random(int(config["seed"]))
    count = len(deltas)
    statistics = sorted(sum(deltas[rng.randrange(count)] for _ in range(count)) for _ in range(resamples))
    alpha = 1.0 - confidence
    return {
        "clusters": count,
        "resamples": resamples,
        "seed": int(config["seed"]),
        "confidence": confidence,
        "interval": config["interval"],
        "observed_sum": sum(deltas),
        "lower_bound": quantile(statistics, alpha / 2.0),
        "upper_bound": quantile(statistics, 1.0 - alpha / 2.0),
    }


def validate_review(packet_path: Path, packet: dict, review: dict, cfg: dict) -> dict[tuple[str, str], dict]:
    if review.get("schema_version") != "rag.reader_larger_untouched_blind_review.v1":
        raise ValueError("review has wrong schema")
    compatible = dict(review)
    compatible["schema_version"] = "rag.reader_unseen_blind_review.v1"
    return base.validate_review(packet_path, packet, compatible, cfg)


def merge(packet_path: Path, review_a_path: Path, review_b_path: Path, output: Path, disputes: Path,
          config_path: Path = CONFIG) -> dict:
    packet, cfg = load(packet_path), load(config_path)
    raw_a, raw_b = load(review_a_path), load(review_b_path)
    reviewer_a, reviewer_b = raw_a.get("reviewer_id"), raw_b.get("reviewer_id")
    if not isinstance(reviewer_a, str) or not reviewer_a.strip() or not isinstance(reviewer_b, str) or not reviewer_b.strip():
        raise ValueError("both reviews need a non-empty reviewer_id")
    if reviewer_a == reviewer_b:
        raise ValueError("blind reviews must come from distinct reviewers")
    a = validate_review(packet_path, packet, raw_a, cfg)
    b = validate_review(packet_path, packet, raw_b, cfg)
    agreed, unresolved = [], []
    for pair in base.expected_pairs(packet):
        left, right = a[pair], b[pair]
        if (left["answer_support"], left["scope_match"]) == (right["answer_support"], right["scope_match"]):
            agreed.append({
                "review_case_id": pair[0], "review_item_id": pair[1],
                "answer_support": left["answer_support"], "scope_match": left["scope_match"],
                "reviewer_a": left, "reviewer_b": right,
            })
        else:
            unresolved.append({
                "review_case_id": pair[0], "review_item_id": pair[1],
                "reviewer_a": left, "reviewer_b": right,
                "adjudicated_answer_support": None, "adjudicated_scope_match": None,
                "adjudication_rationale": "",
            })
    disputes_value = {
        "schema_version": "rag.reader_larger_untouched_blind_disputes.v1",
        "packet_sha256": base.digest(packet_path),
        "unresolved": unresolved,
    }
    consensus = {
        "schema_version": "rag.reader_larger_untouched_blind_consensus.v1",
        "packet_sha256": base.digest(packet_path),
        "complete": not unresolved,
        "judgments": agreed,
        "unresolved_count": len(unresolved),
    }
    write(disputes, disputes_value)
    write(output, consensus)
    return consensus


def adjudicate(packet_path: Path, partial_consensus_path: Path, disputes_path: Path,
               adjudications_path: Path, output: Path, config_path: Path = CONFIG) -> dict:
    packet, cfg = load(packet_path), load(config_path)
    packet_sha = base.digest(packet_path)
    partial, disputes, adjudications = load(partial_consensus_path), load(disputes_path), load(adjudications_path)
    for value, name in ((partial, "partial consensus"), (disputes, "disputes"), (adjudications, "adjudications")):
        if value.get("packet_sha256") != packet_sha:
            raise ValueError(f"{name} packet hash mismatch")
    if partial.get("complete") or partial.get("unresolved_count", 0) <= 0:
        raise ValueError("adjudication requires an incomplete blind consensus")
    support, scope = set(cfg["semantic_codes"]["answer_support"]), set(cfg["semantic_codes"]["scope_match"])
    unresolved = {(row["review_case_id"], row["review_item_id"]): row for row in disputes.get("unresolved", [])}
    resolved: dict[tuple[str, str], dict] = {}
    for row in adjudications.get("judgments", []):
        pair = (row.get("review_case_id"), row.get("review_item_id"))
        if pair in resolved or row.get("answer_support") not in support or row.get("scope_match") not in scope:
            raise ValueError("invalid or duplicate adjudication")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError("every adjudication needs a rationale")
        resolved[pair] = row
    if set(resolved) != set(unresolved):
        raise ValueError("adjudication coverage must exactly match unresolved disputes")
    agreed = {(row["review_case_id"], row["review_item_id"]): row for row in partial.get("judgments", [])}
    expected = base.expected_pairs(packet)
    if set(agreed) & set(resolved) or set(agreed) | set(resolved) != set(expected):
        raise ValueError("final consensus item coverage mismatch")
    final_rows = []
    for pair in expected:
        if pair in agreed:
            final_rows.append(agreed[pair])
        else:
            row = resolved[pair]
            final_rows.append({
                "review_case_id": pair[0], "review_item_id": pair[1],
                "answer_support": row["answer_support"], "scope_match": row["scope_match"],
                "reviewer_a": unresolved[pair]["reviewer_a"],
                "reviewer_b": unresolved[pair]["reviewer_b"],
                "adjudication": row,
            })
    consensus = {
        "schema_version": "rag.reader_larger_untouched_blind_consensus.v1",
        "packet_sha256": packet_sha,
        "complete": True,
        "judgments": final_rows,
        "unresolved_count": 0,
    }
    write(output, consensus)
    return consensus


def score(packet_path: Path, consensus_path: Path, key_path: Path, output: Path,
          config_path: Path = CONFIG) -> dict:
    cfg, consensus = load(config_path), load(consensus_path)
    if not consensus.get("complete") or consensus.get("unresolved_count") != 0:
        raise ValueError("cannot unblind before complete blind consensus")
    packet_sha = base.digest(packet_path)
    if consensus.get("packet_sha256") != packet_sha:
        raise ValueError("consensus packet hash mismatch")
    key = load(key_path)
    if key.get("packet_sha256") != packet_sha:
        raise ValueError("sealed key packet hash mismatch")
    judgment_rows = consensus.get("judgments", [])
    judgments = {(row["review_case_id"], row["review_item_id"]): base.label(row)
                 for row in judgment_rows}
    expected = set(base.expected_pairs(load(packet_path)))
    if len(judgments) != len(judgment_rows) or set(judgments) != expected:
        raise ValueError("consensus item coverage mismatch")
    if key.get("population_questions") != cfg["cohort"]["expected_questions"]:
        raise ValueError("sealed key population drift")
    if len(key.get("cases", [])) != cfg["transport_run"]["expected_changed_cases"]:
        raise ValueError("sealed key changed-case count drift")
    methods = ("identity_reader", "quantum", "classical")
    totals = {method: {"direct": 0, "useful": 0, "uncertain": 0, "points": 0} for method in methods}
    case_values = {method: {} for method in methods}
    for case in key["cases"]:
        membership = {row["source_index"]: row["review_item_id"] for row in case["items"]}
        for method in methods:
            direct = useful = uncertain = 0
            for index in case["selections"][method]:
                result = judgments[(case["review_case_id"], membership[index])]
                direct += int(result[0])
                useful += int(result[1])
                uncertain += int(result[2])
            values = {"direct": direct, "useful": useful, "uncertain": uncertain, "points": useful + direct}
            case_values[method][case["review_case_id"]] = values
            for name, value in values.items():
                totals[method][name] += value

    population = int(key["population_questions"])
    unchanged = population - len(key["cases"])
    if unchanged < 0:
        raise ValueError("sealed key population is smaller than changed set")

    def comparison(left: str, right: str) -> dict:
        useful_deltas = [case_values[left][case]["useful"] - case_values[right][case]["useful"]
                         for case in case_values[left]]
        point_deltas = [case_values[left][case]["points"] - case_values[right][case]["points"]
                        for case in case_values[left]]
        full_point_deltas = point_deltas + [0] * unchanged
        return {
            "useful_case_wins": sum(value > 0 for value in useful_deltas),
            "useful_case_losses": sum(value < 0 for value in useful_deltas),
            "useful_case_ties": sum(value == 0 for value in useful_deltas) + unchanged,
            "useful_case_win_margin": sum(value > 0 for value in useful_deltas) - sum(value < 0 for value in useful_deltas),
            "useful_total_delta": sum(useful_deltas),
            "direct_total_delta": totals[left]["direct"] - totals[right]["direct"],
            "point_total_delta": sum(point_deltas),
            "point_delta_bootstrap": clustered_bootstrap(full_point_deltas, cfg["prospective_gate"]["bootstrap"]),
        }

    q_i = comparison("quantum", "identity_reader")
    q_c = comparison("quantum", "classical")
    gate = cfg["prospective_gate"]
    checks = {
        "useful_over_identity": q_i["useful_total_delta"] >= gate["min_quantum_useful_gain_over_identity"],
        "direct_not_worse_than_identity": q_i["direct_total_delta"] >= gate["min_quantum_direct_gain_over_identity"],
        "useful_over_classical": q_c["useful_total_delta"] >= gate["min_quantum_useful_gain_over_classical"],
        "useful_case_win_margin_over_identity": q_i["useful_case_win_margin"] >= gate["min_quantum_useful_case_win_margin_over_identity"],
        "useful_case_win_margin_over_classical": q_c["useful_case_win_margin"] >= gate["min_quantum_useful_case_win_margin_over_classical"],
        "bootstrap_lower_bound_over_identity": q_i["point_delta_bootstrap"]["lower_bound"] > 0.0,
        "bootstrap_lower_bound_over_classical": q_c["point_delta_bootstrap"]["lower_bound"] > 0.0,
        "uncertainty_bound": totals["quantum"]["uncertain"] <= gate["max_uncertain_quantum_selected_items"],
    }
    result = {
        "schema_version": "rag.reader_larger_untouched_blind_review_score.v1",
        "claim_ceiling": cfg["claim_ceiling"],
        "evidence_tier": "L0_diagnostic",
        "experiment_stage": "screen",
        "population": {"questions": population, "changed_cases": len(key["cases"]), "unchanged_cases": unchanged},
        "totals_on_changed_cases": totals,
        "pairwise_full_population": {"quantum_vs_identity": q_i, "quantum_vs_classical": q_c},
        "gate_checks": checks,
        "advance_to_full_data_replication_design": all(checks.values()),
        "continuation_status": "promote" if all(checks.values()) else "kill",
        "limitations": [
            "Semantic labels were collected only for the changed-set union; unchanged questions contribute exact zero pairwise deltas.",
            "This is a frozen 256-question semantic utility screen, not a full-data replication.",
            "A pass does not establish L1/L2 or quantum-specific advantage; a failure stops only this frozen head configuration.",
        ],
        "provenance": {
            "packet_sha256": packet_sha,
            "consensus_sha256": base.digest(consensus_path),
            "sealed_key_sha256": base.digest(key_path),
            "config_sha256": base.digest(config_path),
        },
    }
    write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--packet", type=Path, required=True)
    merge_parser.add_argument("--review-a", type=Path, required=True)
    merge_parser.add_argument("--review-b", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--disputes", type=Path, required=True)
    merge_parser.add_argument("--config", type=Path, default=CONFIG)
    adjudicate_parser = sub.add_parser("adjudicate")
    adjudicate_parser.add_argument("--packet", type=Path, required=True)
    adjudicate_parser.add_argument("--partial-consensus", type=Path, required=True)
    adjudicate_parser.add_argument("--disputes", type=Path, required=True)
    adjudicate_parser.add_argument("--adjudications", type=Path, required=True)
    adjudicate_parser.add_argument("--output", type=Path, required=True)
    adjudicate_parser.add_argument("--config", type=Path, default=CONFIG)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--packet", type=Path, required=True)
    score_parser.add_argument("--consensus", type=Path, required=True)
    score_parser.add_argument("--key", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.command == "merge":
        value = merge(args.packet, args.review_a, args.review_b, args.output, args.disputes, args.config)
    elif args.command == "adjudicate":
        value = adjudicate(args.packet, args.partial_consensus, args.disputes, args.adjudications, args.output, args.config)
    else:
        value = score(args.packet, args.consensus, args.key, args.output, args.config)
    print(json.dumps(value, ensure_ascii=False, indent=2))
