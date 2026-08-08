#!/usr/bin/env python3
"""Summarize the matched QORE/Answer-Scorer attribution matrix."""

from __future__ import annotations

import json
import math
from numbers import Real
import random
import sys
from pathlib import Path

CONFIGS = ["qore_dpr", "qore_as_control", "qore_as_idea6", "topk_as", "mmr_as"]
PAIRED = [
    ("qore_dpr", "qore_as_control"),
    ("qore_as_control", "qore_as_idea6"),
]


def metric(metrics: dict, name: str):
    return metrics.get(f"mean_{name}", metrics.get(name))


def fmt(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def paired_delta(samples, left: str, right: str, key: str):
    common = sorted(set(samples[left]) & set(samples[right]))
    diffs = []
    for qid in common:
        left_value = samples[left][qid].get(key)
        right_value = samples[right][qid].get(key)
        # Some samples legitimately have no answer metric (None), e.g. when
        # generation/evaluation produced no usable prediction.  Exclude those
        # pairs from the paired analysis instead of passing None to isfinite.
        if (
            isinstance(left_value, Real)
            and not isinstance(left_value, bool)
            and isinstance(right_value, Real)
            and not isinstance(right_value, bool)
            and math.isfinite(left_value)
            and math.isfinite(right_value)
        ):
            diffs.append(right_value - left_value)
    if not diffs:
        return None
    mean = sum(diffs) / len(diffs)
    wins = sum(value > 1e-12 for value in diffs)
    losses = sum(value < -1e-12 for value in diffs)
    ties = len(diffs) - wins - losses
    rng = random.Random(20260807)
    boots = [
        sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        for _ in range(1000)
    ]
    boots.sort()
    return mean, wins, losses, ties, boots[25], boots[975]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} RUN_DIR", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1])
    data = {}
    for config in CONFIGS:
        path = run_dir / config / "result.json"
        if not path.exists():
            print(f"Missing result file: {path}", file=sys.stderr)
            return 1
        data[config] = json.loads(path.read_text())

    samples = {
        config: {row["question_id"]: row for row in data[config]["samples"]}
        for config in CONFIGS
    }
    lines = [
        "# Idea 6 Attribution Matrix",
        "",
        f"- Run: {run_dir.name}",
        f"- Samples: {data['qore_dpr']['metrics'].get('n_samples')}",
        f"- Seed: {data['qore_dpr']['config'].get('seed')}",
        "",
        "## Overall Metrics",
        "",
        "| Config | Recall@5 | F1 | EM | Redundancy | Selection ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        m = data[config]["metrics"]
        lines.append(
            f"| {config} | {fmt(metric(m, 'recall'))} | {fmt(metric(m, 'f1'))} | "
            f"{fmt(metric(m, 'em'))} | {fmt(metric(m, 'redundancy'))} | "
            f"{fmt(metric(m, 'selection_time_ms'))} |"
        )

    lines += [
        "",
        "## Paired Effects",
        "",
        "Positive delta means the right-hand configuration is higher.",
        "",
        "| Comparison | Metric | Mean delta | Wins | Losses | Ties | Bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for left, right in PAIRED:
        for key in ["f1", "recall", "em", "redundancy"]:
            result = paired_delta(samples, left, right, key)
            if result is None:
                lines.append(f"| {left} -> {right} | {key} | N/A | | | | |")
                continue
            mean, wins, losses, ties, lo, hi = result
            lines.append(
                f"| {left} -> {right} | {key} | {mean:+.4f} | {wins} | {losses} | "
                f"{ties} | [{lo:+.4f}, {hi:+.4f}] |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- qore_dpr -> qore_as_control estimates the Answer Scorer effect.",
        "- qore_as_control -> qore_as_idea6 is the matched complementarity effect.",
        "- Compare qore_as_idea6 with topk_as and mmr_as under the same Answer Scorer.",
        "- Use question-level bootstrap; the seed is not an independent replicate.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
