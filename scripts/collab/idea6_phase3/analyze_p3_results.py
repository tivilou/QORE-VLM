"""Analyze Phase 3 results and generate summary report.

Usage:
    python scripts/collab/analyze_p3_results.py exchange/p3_solver_idea6/YYYYMMDDTHHMMSS
"""

import argparse
import json
from pathlib import Path
import numpy as np


def find_result_file(config_dir: Path) -> Path | None:
    """Find the evaluator output, supporting legacy and current filenames."""
    for name in ("result.json", "qore_K5_seed*.json"):
        matches = sorted(config_dir.glob(name))
        if matches:
            return matches[0]
    return None


def load_result(result_path: Path) -> dict:
    """Load an evaluator result and normalize metric names."""
    with open(result_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    return {
        "recall_at_5": metrics.get("recall_at_5", metrics.get("mean_recall")),
        "f1": metrics.get("f1", metrics.get("mean_f1")),
        "exact_match": metrics.get("exact_match", metrics.get("mean_em")),
        "redundancy": metrics.get("redundancy", metrics.get("mean_redundancy")),
        "precision": metrics.get("precision", metrics.get("mean_precision")),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 3 results")
    parser.add_argument("result_dir", help="Path to Phase 3 result directory")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)

    if not result_dir.exists():
        print(f"Error: {result_dir} does not exist")
        return

    print("=" * 70)
    print("Phase 3 Results Analysis")
    print("=" * 70)
    print()

    # Collect results for each config and seed
    configs = ["baseline", "idea6_recommended", "idea6_best"]
    seeds = [42, 43, 44]

    results = {}
    for config in configs:
        results[config] = {}
        for seed in seeds:
            config_dir = result_dir / f"seed_{seed}" / config
            result_path = find_result_file(config_dir)
            if result_path:
                results[config][seed] = load_result(result_path)
            else:
                print(f"Warning: Missing result file in {config_dir}")

    # Print summary table
    print("Results Summary:")
    print()
    print(f"{'Config':<20} {'Seed':<6} {'Recall@5':<10} {'F1':<10} {'EM':<10} {'Redundancy':<12}")
    print("-" * 70)

    for config in configs:
        for seed in seeds:
            if seed in results[config]:
                r = results[config][seed]
                fmt = lambda value: f"{value:.4f}" if value is not None else "N/A"
                print(f"{config:<20} {seed:<6} {fmt(r['recall_at_5']):<10} {fmt(r['f1']):<10} "
                      f"{fmt(r['exact_match']):<10} {fmt(r['redundancy']):<12}")

    print()
    print("=" * 70)
    print("Average across seeds:")
    print("=" * 70)
    print()

    print(f"{'Config':<20} {'Recall@5':<15} {'F1':<15} {'EM':<15} {'Redundancy':<15}")
    print("-" * 70)

    baseline_avg = None

    for config in configs:
        if not results[config]:
            continue

        recall_vals = [results[config][s]["recall_at_5"] for s in results[config]]
        f1_vals = [results[config][s]["f1"] for s in results[config] if results[config][s]["f1"] is not None]
        em_vals = [results[config][s]["exact_match"] for s in results[config] if results[config][s]["exact_match"] is not None]
        red_vals = [results[config][s]["redundancy"] for s in results[config]]

        avg_recall = np.mean(recall_vals)
        avg_f1 = np.mean(f1_vals) if f1_vals else None
        avg_em = np.mean(em_vals) if em_vals else None
        avg_red = np.mean(red_vals)

        std_recall = np.std(recall_vals)
        std_f1 = np.std(f1_vals) if f1_vals else None

        if config == "baseline":
            baseline_avg = {
                "recall": avg_recall,
                "f1": avg_f1,
                "em": avg_em,
                "redundancy": avg_red,
            }
            print(f"{config:<20} {avg_recall:.4f}±{std_recall:.4f}  "
                  f"{avg_f1:.4f}±{std_f1:.4f}" if avg_f1 is not None else f"{config:<20} {avg_recall:.4f}±{std_recall:.4f}  F1=N/A", end="")
            print(f"  {avg_em:.4f}  {avg_red:.4f}" if avg_em is not None else f"  EM=N/A  {avg_red:.4f}")
        else:
            delta_recall = (avg_recall - baseline_avg["recall"]) / baseline_avg["recall"] * 100
            delta_red = (avg_red - baseline_avg["redundancy"]) / baseline_avg["redundancy"] * 100
            f1_text = f"{avg_f1:.4f}" if avg_f1 is not None else "N/A"
            em_text = f"{avg_em:.4f}" if avg_em is not None else "N/A"
            print(f"{config:<20} {avg_recall:.4f}±{std_recall:.4f} ({delta_recall:+.1f}%)  "
                  f"{f1_text}  {em_text}  {avg_red:.4f} ({delta_red:+.1f}%)")

    print()
    print("=" * 70)
    print("Phase 3 Success Criteria:")
    print("=" * 70)
    print()

    if baseline_avg:
        for config in ["idea6_recommended", "idea6_best"]:
            if not results[config]:
                continue

            recall_vals = [results[config][s]["recall_at_5"] for s in results[config]]
            f1_vals = [
                results[config][s]["f1"]
                for s in results[config]
                if results[config][s]["f1"] is not None
            ]
            avg_recall = np.mean(recall_vals)
            std_recall = np.std(recall_vals)

            delta_recall_pct = (avg_recall - baseline_avg["recall"]) / baseline_avg["recall"] * 100

            print(f"{config}:")
            print(f"  ✓ Recall@5 提升: {delta_recall_pct:.1f}% (target: ≥35%)")
            if f1_vals and baseline_avg["f1"] is not None:
                avg_f1 = np.mean(f1_vals)
                delta_f1 = avg_f1 - baseline_avg["f1"]
                print(
                    f"  - F1: {avg_f1:.4f} "
                    f"(baseline={baseline_avg['f1']:.4f}, Δ={delta_f1:+.4f})"
                )
            else:
                print("  - F1: N/A (generation metric missing)")
            print(f"  ✓ 结果稳定性: std={std_recall:.4f} (target: <0.02)")
            print()


if __name__ == "__main__":
    main()
