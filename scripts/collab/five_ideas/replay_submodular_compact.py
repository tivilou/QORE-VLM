#!/usr/bin/env python3
"""Replay the Submodular selector on passage-free recorded QUBO payloads.

This is a selection diagnostic only.  It reuses the recorded normalized quality
vector, redundancy matrix, control selection, and compact gold flags; it does
not load Wiki-DPR, passage text, or regenerate answers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from applications.rag.baselines.submodular import select


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--saturation-alpha", type=float, default=1.0)
    parser.add_argument("--lambda-submodular", type=float, default=0.5)
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


def gold_flags_for_pool(item: dict) -> np.ndarray:
    flags = {
        int(candidate["retrieved_rank"]): bool(candidate["is_gold"])
        for candidate in item.get("candidate_flags", [])
    }
    return np.asarray(
        [flags.get(int(rank), False) for rank in item["pool_ranks"]],
        dtype=bool,
    )


def replay_row(item: dict, saturation_alpha: float, lambda_submodular: float) -> dict:
    a = np.asarray(item["a"], dtype=np.float64)
    b = np.asarray(item["b"], dtype=np.float64)
    K = int(item["K"])
    control = np.flatnonzero(np.asarray(item["x"], dtype=np.int64))
    selected = select(
        a,
        b,
        K,
        saturation_alpha=saturation_alpha,
        lambda_redundancy=lambda_submodular,
    )
    gold = gold_flags_for_pool(item)
    control_hits = int(gold[control].sum())
    submodular_hits = int(gold[selected].sum())
    selected_b = b[np.ix_(selected, selected)]
    redundancy = (
        float(selected_b[np.triu_indices(K, k=1)].mean()) if K > 1 else 0.0
    )
    return {
        "question_id": str(item["question_id"]),
        "pool_size": len(a),
        "K": K,
        "control_gold_hits": control_hits,
        "submodular_gold_hits": submodular_hits,
        "gold_hit_delta": submodular_hits - control_hits,
        "selection_changed": bool(not np.array_equal(selected, control)),
        "submodular_redundancy": redundancy,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    analysis_dir = run_dir / "analysis"
    rows = [
        replay_row(item, args.saturation_alpha, args.lambda_submodular)
        for item in sorted(
            load_control_payload(analysis_dir / "qubo_payload.jsonl.gz"),
            key=lambda value: str(value["question_id"]),
        )
    ]
    output = args.output or (analysis_dir / "submodular_compact.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "questions": len(rows),
        "saturation_alpha": args.saturation_alpha,
        "lambda_submodular": args.lambda_submodular,
        "mean_control_gold_hits": float(np.mean([r["control_gold_hits"] for r in rows])),
        "mean_submodular_gold_hits": float(np.mean([r["submodular_gold_hits"] for r in rows])),
        "mean_gold_hit_delta": float(np.mean([r["gold_hit_delta"] for r in rows])),
        "selection_changes": sum(bool(r["selection_changed"]) for r in rows),
        "mean_submodular_redundancy": float(np.mean([r["submodular_redundancy"] for r in rows])),
        "source": "passage-free qore_as_control qubo_payload.jsonl.gz",
        "interpretation": "selection diagnostic; not a held-out generation or F1 result",
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
