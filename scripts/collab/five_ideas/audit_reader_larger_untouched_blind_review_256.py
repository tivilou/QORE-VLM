#!/usr/bin/env python3
"""Independently replay the LU-06 blind merge, adjudication, and utility score."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


ALLOWED_SUPPORT = {"direct", "partial", "none", "uncertain"}
ALLOWED_SCOPE = {"yes", "partial", "no", "uncertain"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def semantic(row: dict) -> tuple[str, str]:
    value = (row.get("answer_support"), row.get("scope_match"))
    if value[0] not in ALLOWED_SUPPORT or value[1] not in ALLOWED_SCOPE:
        raise ValueError("invalid semantic code")
    return value


def indexed(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        pair = (row.get("review_case_id"), row.get("review_item_id"))
        if pair in result:
            raise ValueError("duplicate semantic row")
        result[pair] = row
    return result


def quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def bootstrap(deltas: list[int], cfg: dict) -> dict:
    rng = random.Random(int(cfg["seed"]))
    count, resamples = len(deltas), int(cfg["resamples"])
    values = [sum(deltas[rng.randrange(count)] for _ in range(count)) for _ in range(resamples)]
    alpha = 1.0 - float(cfg["confidence"])
    return {
        "clusters": count,
        "resamples": resamples,
        "seed": int(cfg["seed"]),
        "confidence": float(cfg["confidence"]),
        "interval": cfg["interval"],
        "observed_sum": sum(deltas),
        "lower_bound": quantile(values, alpha / 2.0),
        "upper_bound": quantile(values, 1.0 - alpha / 2.0),
    }


def audit(packet_path: Path, review_a_path: Path, review_b_path: Path, disputes_path: Path,
          adjudications_path: Path, consensus_path: Path, key_path: Path, score_path: Path,
          config_path: Path, output: Path) -> dict:
    packet, review_a, review_b = load(packet_path), load(review_a_path), load(review_b_path)
    disputes, adjudications = load(disputes_path), load(adjudications_path)
    consensus, key, score, cfg = load(consensus_path), load(key_path), load(score_path), load(config_path)
    packet_sha = digest(packet_path)
    bound = (review_a, review_b, disputes, adjudications, consensus, key)
    if any(value.get("packet_sha256") != packet_sha for value in bound):
        raise ValueError("packet hash binding mismatch")
    if review_a.get("reviewer_id") == review_b.get("reviewer_id"):
        raise ValueError("reviewer identities are not distinct")
    if not review_a.get("completed") or not review_b.get("completed"):
        raise ValueError("review incomplete")
    expected = [(case["review_case_id"], item["review_item_id"])
                for case in packet["cases"] for item in case["items"]]
    if len(expected) != len(set(expected)) or len(expected) != int(cfg["transport_run"]["expected_review_items"]):
        raise ValueError("packet item identity drift")
    expected_set = set(expected)
    a, b = indexed(review_a["judgments"]), indexed(review_b["judgments"])
    final, adjudicated = indexed(consensus["judgments"]), indexed(adjudications["judgments"])
    dispute_index = indexed(disputes["unresolved"])
    if set(a) != expected_set or set(b) != expected_set or set(final) != expected_set:
        raise ValueError("review or consensus coverage mismatch")
    agreed_pairs = {pair for pair in expected if semantic(a[pair]) == semantic(b[pair])}
    disputed_pairs = expected_set - agreed_pairs
    if set(dispute_index) != disputed_pairs or set(adjudicated) != disputed_pairs:
        raise ValueError("dispute or adjudication coverage mismatch")
    for pair in expected:
        source = a[pair] if pair in agreed_pairs else adjudicated[pair]
        if semantic(final[pair]) != semantic(source):
            raise ValueError("final consensus semantic replay mismatch")
    if not consensus.get("complete") or consensus.get("unresolved_count") != 0:
        raise ValueError("final consensus is not complete")

    def flags(row: dict) -> tuple[int, int, int]:
        support, scope = semantic(row)
        direct = int(support == "direct" and scope == "yes")
        useful = int(support in {"direct", "partial"} and scope in {"yes", "partial"})
        uncertain = int(support == "uncertain" or scope == "uncertain")
        return direct, useful, uncertain

    judgments = {pair: flags(final[pair]) for pair in expected}
    methods = ("identity_reader", "quantum", "classical")
    totals = {method: {"direct": 0, "useful": 0, "uncertain": 0, "points": 0} for method in methods}
    case_values = {method: {} for method in methods}
    for case in key["cases"]:
        mapping = {row["source_index"]: row["review_item_id"] for row in case["items"]}
        for method in methods:
            direct = useful = uncertain = 0
            for source_index in case["selections"][method]:
                values = judgments[(case["review_case_id"], mapping[source_index])]
                direct += values[0]
                useful += values[1]
                uncertain += values[2]
            values = {"direct": direct, "useful": useful, "uncertain": uncertain, "points": direct + useful}
            case_values[method][case["review_case_id"]] = values
            for name, value in values.items():
                totals[method][name] += value
    population = int(key["population_questions"])
    unchanged = population - len(key["cases"])

    def compare(left: str, right: str) -> dict:
        useful = [case_values[left][case]["useful"] - case_values[right][case]["useful"]
                  for case in case_values[left]]
        points = [case_values[left][case]["points"] - case_values[right][case]["points"]
                  for case in case_values[left]]
        return {
            "useful_case_wins": sum(value > 0 for value in useful),
            "useful_case_losses": sum(value < 0 for value in useful),
            "useful_case_ties": sum(value == 0 for value in useful) + unchanged,
            "useful_case_win_margin": sum(value > 0 for value in useful) - sum(value < 0 for value in useful),
            "useful_total_delta": sum(useful),
            "direct_total_delta": totals[left]["direct"] - totals[right]["direct"],
            "point_total_delta": sum(points),
            "point_delta_bootstrap": bootstrap(points + [0] * unchanged, cfg["prospective_gate"]["bootstrap"]),
        }

    comparisons = {
        "quantum_vs_identity": compare("quantum", "identity_reader"),
        "quantum_vs_classical": compare("quantum", "classical"),
    }
    gate = cfg["prospective_gate"]
    q_i, q_c = comparisons["quantum_vs_identity"], comparisons["quantum_vs_classical"]
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
    if totals != score["totals_on_changed_cases"] or comparisons != score["pairwise_full_population"]:
        raise ValueError("independent metric replay mismatch")
    if checks != score["gate_checks"] or all(checks.values()) != score["advance_to_full_data_replication_design"]:
        raise ValueError("independent gate replay mismatch")
    result = {
        "schema_version": "rag.reader_larger_untouched_blind_review_audit.v1",
        "status": "passed",
        "population": {"questions": population, "changed_cases": len(key["cases"]),
                       "unchanged_cases": unchanged, "review_items": len(expected)},
        "review": {"agreed_items": len(agreed_pairs), "disputed_items": len(disputed_pairs),
                   "raw_exact_agreement": len(agreed_pairs) / len(expected),
                   "adjudicated_items": len(adjudicated), "unresolved_items": 0,
                   "distinct_reviewers": True},
        "totals_on_changed_cases": totals,
        "pairwise_full_population": comparisons,
        "gate_checks": checks,
        "advance_to_full_data_replication_design": all(checks.values()),
        "replay": {"consensus_exact": True, "metrics_exact": True, "bootstrap_exact": True,
                   "gate_exact": True},
        "provenance": {
            "packet_sha256": packet_sha,
            "review_a_sha256": digest(review_a_path),
            "review_b_sha256": digest(review_b_path),
            "disputes_sha256": digest(disputes_path),
            "adjudications_sha256": digest(adjudications_path),
            "consensus_sha256": digest(consensus_path),
            "sealed_key_sha256": digest(key_path),
            "score_sha256": digest(score_path),
            "config_sha256": digest(config_path),
        },
    }
    write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--disputes", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.packet, args.review_a, args.review_b, args.disputes,
                           args.adjudications, args.consensus, args.key, args.score,
                           args.config, args.output), ensure_ascii=False, indent=2))
