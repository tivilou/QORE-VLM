#!/usr/bin/env python3
"""Run the fixed-50 support-graph no-GPU preflight."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from applications.rag.support_graph_preflight import (  # noqa: E402
    PreflightError,
    run_preflight,
    write_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic support-graph no-GPU preflight")
    parser.add_argument(
        "--case-study", type=Path,
        default=PROJECT_ROOT / "research-web/apps/experiment-results/case-studies/rag-selector-50-detail.json",
    )
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/support_graph_preflight_50.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "exchange/five_ideas/support_graph_preflight_50",
    )
    args = parser.parse_args()
    output_dir = args.output_dir / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        result = run_preflight(args.case_study, args.config)
        replay = run_preflight(args.case_study, args.config)
        if result["replay_digest"] != replay["replay_digest"]:
            raise PreflightError("deterministic replay digest changed between runs")
        result["gates"]["deterministic_replay_gate"] = {
            "pass": True, "digest": result["replay_digest"],
        }
        result["gates"]["overall"] = {
            "pass": all(value.get("pass", False) for name, value in result["gates"].items() if name != "overall")
        }
        write_artifacts(result, output_dir, args.case_study)
    except (OSError, PreflightError) as exc:
        print(f"Support-graph preflight failed: {exc}", file=sys.stderr)
        return 2

    metrics = result["metrics"]
    candidate = metrics["selectors"]["support_graph"]
    topk = metrics["selectors"]["topk_as"]
    print(f"Support-graph no-GPU preflight: {'PASS' if result['gates']['overall']['pass'] else 'FAIL'}")
    print(f"Cases: {metrics['case_count']} x Top-50: {metrics['top50_count_per_case']}")
    print(f"Broad-positive passages: support_graph={candidate['selected_broad_positive_count']} topk={topk['selected_broad_positive_count']}")
    print(f"Direct-positive passages: support_graph={candidate['selected_direct_positive_count']} topk={topk['selected_direct_positive_count']}")
    print(f"Broad selector misses: {candidate['broad_selector_miss_count']}")
    print(f"Artifacts: {output_dir}")
    return 0 if result["gates"]["overall"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
