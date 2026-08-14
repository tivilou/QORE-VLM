"""Offline oracle diagnostic for an adaptive value-of-information signal.

This diagnostic replays the passage-free QUBO payload emitted by a completed
gate.  It measures (1) the recoverable gold headroom in each candidate pool and
(2) whether a bounded uncertainty transform of the existing quality signal can
close that headroom.  It does not load Wiki-DPR, passage text, answers, or
regenerate model predictions, and its tau sweep is an oracle replay rather than
a held-out experiment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from qore.qubo import build_qubo_matrix_from_w
from qore.solvers.brute import solve as brute_solve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tau",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        help="Strength of the replayed uncertainty quality transform",
    )
    return parser.parse_args()


def load_control_payload(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("configuration") == "qore_as_control":
                rows.append(item)
    if not rows:
        raise ValueError("no qore_as_control rows found in QUBO payload")
    return rows


def selected_redundancy(b: np.ndarray, selected: np.ndarray) -> float:
    if len(selected) < 2:
        return 0.0
    values = b[np.ix_(selected, selected)]
    return float(values[np.triu_indices(len(selected), k=1)].mean())


def gold_flags_for_pool(item: dict) -> np.ndarray:
    """Map passage-free retrieved-rank gold flags onto pool-local positions."""
    flags = {
        int(candidate["retrieved_rank"]): bool(candidate["is_gold"])
        for candidate in item.get("candidate_flags", [])
    }
    pool_ranks = [int(value) for value in item["pool_ranks"]]
    return np.asarray([flags.get(rank, False) for rank in pool_ranks], dtype=bool)


def replay_selection(a: np.ndarray, w: np.ndarray, K: int, lam: float) -> np.ndarray:
    qubo = build_qubo_matrix_from_w(a, w, K, lam=lam)
    return np.flatnonzero(brute_solve(qubo, K)).astype(np.int64)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    analysis_dir = run_dir / "analysis"
    payload = load_control_payload(analysis_dir / "qubo_payload.jsonl.gz")
    output = args.output or (analysis_dir / "voi_oracle.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for item in sorted(payload, key=lambda value: str(value["question_id"])):
        a = np.asarray(item["a"], dtype=np.float64)
        b = np.asarray(item["b"], dtype=np.float64)
        w = np.asarray(item["w"], dtype=np.float64)
        K = int(item["K"])
        lam = float(item["lam"])
        if len(a) > 20 or not (1 <= K < len(a)):
            raise ValueError("oracle replay requires 1 <= K < pool_size <= 20")

        gold_flags = gold_flags_for_pool(item)
        control_selected = np.flatnonzero(np.asarray(item["x"], dtype=np.int64))
        control_hits = int(gold_flags[control_selected].sum())
        gold_pool_count = int(gold_flags.sum())
        oracle_hits = min(K, gold_pool_count)
        oracle_gap = oracle_hits - control_hits
        uncertainty = 4.0 * a * (1.0 - a)

        for tau in args.tau:
            if tau < 0.0:
                raise ValueError("tau must be non-negative")
            # ``a`` was normalized over the full retrieved set before the
            # relevance prefilter.  Re-normalizing the 15-item pool would
            # change the recorded control objective, so preserve its scale.
            transformed = np.clip(a + float(tau) * uncertainty, 0.0, 1.0)
            selected = replay_selection(transformed, w, K, lam)
            hits = int(gold_flags[selected].sum())
            rows.append({
                "question_id": str(item["question_id"]),
                "tau": float(tau),
                "pool_size": len(a),
                "K": K,
                "gold_pool_count": gold_pool_count,
                "control_gold_hits": control_hits,
                "oracle_gold_hits": oracle_hits,
                "oracle_gap": oracle_gap,
                "gold_hits": hits,
                "gold_recall": hits / gold_pool_count if gold_pool_count else 0.0,
                "uncertainty_mean": float(np.mean(uncertainty)),
                "uncertainty_gold_mean": float(np.mean(uncertainty[gold_flags])) if gold_pool_count else 0.0,
                "uncertainty_non_gold_mean": float(np.mean(uncertainty[~gold_flags])) if gold_pool_count < len(a) else 0.0,
                "selection_changed": bool(not np.array_equal(selected, control_selected)),
                "selected_gold_gain": hits - control_hits,
                "oracle_gap_closed": min(max(hits - control_hits, 0), oracle_gap),
                "redundancy": selected_redundancy(b, selected),
            })

    fieldnames = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    aggregate: list[dict[str, object]] = []
    for tau in args.tau:
        subset = [row for row in rows if float(row["tau"]) == float(tau)]
        gap_total = sum(int(row["oracle_gap"]) for row in subset)
        closed_total = sum(int(row["oracle_gap_closed"]) for row in subset)
        aggregate.append({
            "tau": float(tau),
            "questions": len(subset),
            "gold_in_pool_questions": sum(int(row["gold_pool_count"]) > 0 for row in subset),
            "mean_control_gold_hits": float(np.mean([row["control_gold_hits"] for row in subset])),
            "mean_oracle_gold_hits": float(np.mean([row["oracle_gold_hits"] for row in subset])),
            "mean_gold_hits": float(np.mean([row["gold_hits"] for row in subset])),
            "mean_gold_recall": float(np.mean([row["gold_recall"] for row in subset])),
            "mean_oracle_gap": float(np.mean([row["oracle_gap"] for row in subset])),
            "oracle_gap_closed_fraction": closed_total / gap_total if gap_total else 0.0,
            "selection_changes": sum(bool(row["selection_changed"]) for row in subset),
            "mean_selected_gold_gain": float(np.mean([row["selected_gold_gain"] for row in subset])),
            "mean_redundancy": float(np.mean([row["redundancy"] for row in subset])),
            "mean_uncertainty_gold": float(np.mean([row["uncertainty_gold_mean"] for row in subset])),
            "mean_uncertainty_non_gold": float(np.mean([row["uncertainty_non_gold_mean"] for row in subset])),
        })

    aggregate_path = output.with_name(output.stem + "_aggregate.csv")
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate)

    readme_path = output.with_name(output.stem + "_README.md")
    readme_path.write_text(
        "# Adaptive VoI Oracle Diagnostic\n\n"
        "This is a passage-free oracle replay over the recorded qore_as_control "
        "candidate pools. `candidate_flags.is_gold` supplies the diagnostic "
        "oracle label; it is not available to a deployed selector. The tau "
        "sweep adds a bounded uncertainty transform `4*a*(1-a)` to the existing "
        "quality signal and re-solves the exact recorded QUBO. It measures headroom "
        "and plausibility only; it is not held-out evidence and does not estimate "
        "new generation F1.\n\n"
        "A VoI implementation should proceed only if the oracle gap is material "
        "and the uncertainty transform closes a reproducible fraction of it without "
        "excessive selection cost.\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
