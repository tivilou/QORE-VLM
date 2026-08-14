#!/usr/bin/env python3
"""Replay cohesion calibration on the passage-free QUBO payload.

This is an offline selection-level diagnostic. It does not load Wiki-DPR or
regenerate answers. Recall and redundancy are exact for the recorded candidate
pool; generation F1 is intentionally not estimated for newly selected sets.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from qore.qubo import build_qubo_matrix_from_w, energy
from qore.solvers.brute import solve as brute_solve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--eta",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("configuration") != "qore_as_control":
                continue
            rows[str(item["question_id"])] = item
    return rows


def load_gold(path: Path) -> dict[str, set[int]]:
    gold: dict[str, set[int]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["configuration"] != "qore_as_control":
                continue
            raw = row["gold_ranks"].strip()
            gold[row["question_id"]] = {
                int(value) for value in raw.split(";") if value
            }
    return gold


def selected_redundancy(b: np.ndarray, selected: np.ndarray) -> float:
    if len(selected) < 2:
        return 0.0
    values = b[np.ix_(selected, selected)]
    return float(values[np.triu_indices(len(selected), k=1)].mean())


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    analysis_dir = run_dir / "analysis"
    payload = load_payload(analysis_dir / "qubo_payload.jsonl.gz")
    gold = load_gold(analysis_dir / "per_question.csv")
    if not payload:
        raise SystemExit("no qore_as_control payload rows found")

    output = args.output or (analysis_dir / "cohesion_sensitivity.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for question_id, item in sorted(payload.items()):
        a = np.asarray(item["a"], dtype=np.float64)
        b = np.asarray(item["b"], dtype=np.float64)
        w = np.asarray(item["w"], dtype=np.float64)
        pool_ranks = np.asarray(item["pool_ranks"], dtype=np.int64)
        K = int(item["K"])
        lam = float(item["lam"])
        upper = b[np.triu_indices(len(a), k=1)]
        pool_cohesion = float(np.clip(upper, 0.0, None).mean())
        sorted_quality = np.sort(a)[::-1]
        margin = max(float(sorted_quality[K - 1] - sorted_quality[K]), 0.0)

        for eta in args.eta:
            denominator = max((K - 1) * (pool_cohesion + 1e-6), 1e-6)
            delta_q = float(eta * margin / denominator)
            # The recorded control w is the exact root objective. The cohesion
            # plugin's correction is delta_q * b.
            candidate_w = w + delta_q * b
            qubo = build_qubo_matrix_from_w(a, candidate_w, K, lam=lam)
            x = brute_solve(qubo, K)
            selected_pool = np.flatnonzero(x).astype(np.int64)
            selected_ranks = set(int(value) for value in pool_ranks[selected_pool])
            gold_ranks = gold.get(question_id, set())
            hits = len(selected_ranks & gold_ranks)
            recall = hits / len(gold_ranks) if gold_ranks else 0.0
            rows.append({
                "eta": float(eta),
                "question_id": question_id,
                "pool_size": len(a),
                "K": K,
                "relevance_margin": margin,
                "pool_cohesion": pool_cohesion,
                "delta_q": delta_q,
                "selected_ranks": ";".join(str(v) for v in sorted(selected_ranks)),
                "gold_pool": bool(gold_ranks),
                "gold_hits": hits,
                "recall": recall,
                "redundancy": selected_redundancy(b, selected_pool),
                "energy": energy(x, qubo),
            })

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    baseline = {
        row["question_id"]: row["selected_ranks"]
        for row in rows
        if float(row["eta"]) == 0.0
    }
    summary_rows: list[dict[str, object]] = []
    for eta in args.eta:
        subset = [row for row in rows if float(row["eta"]) == eta]
        valid = [row for row in subset if row["gold_pool"]]
        changed = sum(
            row["selected_ranks"] != baseline.get(row["question_id"])
            for row in subset
        )
        summary_rows.append({
            "eta": eta,
            "questions": len(subset),
            "retrieval_hit_questions": len(valid),
            "mean_recall": float(np.mean([row["recall"] for row in valid])) if valid else 0.0,
            "mean_redundancy": float(np.mean([row["redundancy"] for row in subset])),
            "mean_delta_q": float(np.mean([row["delta_q"] for row in subset])),
            "p95_delta_q": float(np.quantile([row["delta_q"] for row in subset], 0.95)),
            "selection_changes_vs_eta0": changed,
            "mean_energy": float(np.mean([row["energy"] for row in subset])),
        })
    summary_path = output.with_name(output.stem + "_aggregate.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
