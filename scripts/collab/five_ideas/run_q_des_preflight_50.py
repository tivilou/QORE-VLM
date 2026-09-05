#!/usr/bin/env python3
"""CLI for the canonical no-GPU Q-DES preflight."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from applications.rag.qdes_preflight import PreflightError, run_preflight, write_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic Q-DES no-GPU preflight")
    parser.add_argument(
        "--case-study",
        type=Path,
        default=PROJECT_ROOT / "research-web/apps/experiment-results/case-studies/rag-selector-50-detail.json",
        help="Detailed case-study JSON with root field 'cases'",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="Optional panel JSON; schema checked only and never used for selection",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "exchange/five_ideas/qdes_preflight",
        help="Base directory for timestamped compact artifacts",
    )
    args = parser.parse_args()
    output_dir = args.output_dir / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        result = run_preflight(args.case_study, args.panel)
        # Replay is computed from the compact core in the library.  Re-running
        # the same pure analysis here catches accidental nondeterminism before
        # any result is published.
        replay = run_preflight(args.case_study, args.panel)
        if result["replay_digest"] != replay["replay_digest"]:
            raise PreflightError("deterministic replay digest changed between runs")
        write_artifacts(result, output_dir, args.case_study)
    except (OSError, PreflightError) as exc:
        print(f"Q-DES preflight failed: {exc}", file=sys.stderr)
        return 2

    print(f"Q-DES no-GPU preflight: {'PASS' if result['gates']['overall']['pass'] else 'FAIL'}")
    print(f"Cases: {result['metrics']['case_count']} x Top-50: {result['metrics']['top50_count_per_case']}")
    print(f"Parser success: {result['metrics']['parser_success_rate']:.1%}")
    auc = result["metrics"]["auc_micro"]
    print(
        "Micro AUC: typed={typed} answer_scorer={answer} retrieval={retrieval}".format(
            typed="NA" if auc["typed_coverage"] is None else f"{auc['typed_coverage']:.3f}",
            answer="NA" if auc["answer_scorer"] is None else f"{auc['answer_scorer']:.3f}",
            retrieval="NA" if auc["retrieval_score"] is None else f"{auc['retrieval_score']:.3f}",
        )
    )
    print(f"Artifacts: {output_dir}")
    return 0 if result["gates"]["overall"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
