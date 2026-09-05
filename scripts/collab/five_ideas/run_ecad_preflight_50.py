#!/usr/bin/env python3
"""CLI for the canonical no-GPU ECAD conflict preflight."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from applications.rag.ecad_preflight import PreflightError, run_preflight, write_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic ECAD no-GPU conflict preflight")
    parser.add_argument(
        "--case-study",
        type=Path,
        default=PROJECT_ROOT / "research-web/apps/experiment-results/case-studies/rag-selector-50-detail.json",
        help="Detailed case-study JSON with root field 'cases'",
    )
    parser.add_argument(
        "--pairwise-nli",
        type=Path,
        default=None,
        help="Optional ECAD pairwise NLI artifact; absent means lexical negative control and gate failure",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "exchange/five_ideas/ecad_preflight",
        help="Base directory for timestamped compact artifacts",
    )
    args = parser.parse_args()
    output_dir = args.output_dir / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        result = run_preflight(args.case_study, args.pairwise_nli)
        replay = run_preflight(args.case_study, args.pairwise_nli)
        if result["replay_digest"] != replay["replay_digest"]:
            raise PreflightError("deterministic replay digest changed between runs")
        write_artifacts(result, output_dir, args.case_study)
    except (OSError, PreflightError) as exc:
        print(f"ECAD preflight failed: {exc}", file=sys.stderr)
        return 2

    print(f"ECAD no-GPU preflight: {'PASS' if result['gates']['overall']['pass'] else 'FAIL'}")
    print(f"Cases: {result['metrics']['case_count']} x Top-50: {result['metrics']['top50_count_per_case']}")
    print(f"Pair score source: {result['metrics']['score_source']}")
    print(f"Conflict edge rate: {result['metrics']['conflict_edge_rate_mean']:.3f}")
    print(f"Conflict reduction: {result['metrics']['probe_conflict_reduction']}")
    print(f"Artifacts: {output_dir}")
    return 0 if result["gates"]["overall"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
