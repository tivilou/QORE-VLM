#!/usr/bin/env python3
"""Passage-free replay diagnostic for the Spectral/DPP strategy.

The historical QUBO payload stores clipped cosine redundancy rather than raw
embeddings. This replay restores a unit diagonal and applies the strategy's
deterministic PSD projection, so it is a screening diagnostic rather than a
formal reproduction of the embedding-kernel path.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from applications.rag.baselines.spectral_dpp import select_from_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quality-scale", type=float, default=2.0)
    parser.add_argument("--jitter", type=float, default=1e-8)
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


def replay_row(item: dict, quality_scale: float, jitter: float) -> dict:
    quality = np.asarray(item["a"], dtype=np.float64)
    redundancy = np.asarray(item["b"], dtype=np.float64)
    K = int(item["K"])
    similarity = redundancy.copy()
    np.fill_diagonal(similarity, 1.0)
    selected = select_from_similarity(
        quality,
        similarity,
        K,
        quality_scale=quality_scale,
        jitter=jitter,
    )
    control = np.flatnonzero(np.asarray(item["x"], dtype=np.int64))
    gold = gold_flags_for_pool(item)
    selected_redundancy = redundancy[np.ix_(selected, selected)]
    mean_redundancy = (
        float(selected_redundancy[np.triu_indices(K, k=1)].mean()) if K > 1 else 0.0
    )
    return {
        "question_id": str(item["question_id"]),
        "pool_size": len(quality),
        "K": K,
        "control_gold_hits": int(gold[control].sum()),
        "spectral_dpp_gold_hits": int(gold[selected].sum()),
        "gold_hit_delta": int(gold[selected].sum() - gold[control].sum()),
        "selection_changed": bool(not np.array_equal(selected, control)),
        "spectral_dpp_redundancy": mean_redundancy,
    }


def main() -> int:
    args = parse_args()
    analysis_dir = args.run_dir.resolve() / "analysis"
    rows = [
        replay_row(item, args.quality_scale, args.jitter)
        for item in sorted(
            load_control_payload(analysis_dir / "qubo_payload.jsonl.gz"),
            key=lambda value: str(value["question_id"]),
        )
    ]
    output = args.output or (analysis_dir / "spectral_dpp_compact.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "questions": len(rows),
        "quality_scale": args.quality_scale,
        "jitter": args.jitter,
        "mean_control_gold_hits": float(np.mean([r["control_gold_hits"] for r in rows])),
        "mean_spectral_dpp_gold_hits": float(np.mean([r["spectral_dpp_gold_hits"] for r in rows])),
        "mean_gold_hit_delta": float(np.mean([r["gold_hit_delta"] for r in rows])),
        "selection_changes": sum(bool(r["selection_changed"]) for r in rows),
        "mean_spectral_dpp_redundancy": float(np.mean([r["spectral_dpp_redundancy"] for r in rows])),
        "source": "passage-free qore_as_control qubo_payload.jsonl.gz",
        "interpretation": "PSD-repaired clipped-cosine diagnostic; not held-out generation or F1 evidence",
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
