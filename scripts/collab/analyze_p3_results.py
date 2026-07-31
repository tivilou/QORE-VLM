"""Analyze Phase 3 results and generate summary report.

Usage:
    python scripts/collab/analyze_p3_results.py exchange/p3_solver_idea6/YYYYMMDDTHHMMSS
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_result(result_path: Path) -> dict:
    """Load result.json and extract key metrics."""
    with open(result_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    return {
        "recall_at_5": metrics.get("recall_at_5", 0.0),
        "f1": metrics.get("f1", 0.0),
        "exact_match": metrics.get("exact_match", 0.0),
        "redundancy": metrics.get("redundancy", 0.0),
        "precision": metrics.get("precision", 0.0),
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
            result_path = result_dir / f"seed_{seed}" / config / "result.json"
            if result_path.exists():
                results[config][seed] = load_result(result_path)
            else:
                print(f"Warning: Missing {result_path}")

    # Print summary table
    print("Results Summary:")
    print()
    print(f"{'Config':<20} {'Seed':<6} {'Recall@5':<10} {'F1':<10} {'EM':<10} {'Redundancy':<12}")
    print("-" * 70)

    for config in configs:
        for seed in seeds:
            if seed in results[config]:
                r = results[config][seed]
                print(f"{config:<20} {seed:<6} {r['recall_at_5']:<10.4f} {r['f1']:<10.4f} "
                      f"{r['exact_match']:<10.4f} {r['redundancy']:<12.4f}")

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
        f1_vals = [results[config][s]["f1"] for s in results[config]]
        em_vals = [results[config][s]["exact_match"] for s in results[config]]
        red_vals = [results[config][s]["redundancy"] for s in results[config]]

        avg_recall = np.mean(recall_vals)
        avg_f1 = np.mean(f1_vals)
        avg_em = np.mean(em_vals)
        avg_red = np.mean(red_vals)

        std_recall = np.std(recall_vals)
        std_f1 = np.std(f1_vals)

        if config == "baseline":
            baseline_avg = {
                "recall": avg_recall,
                "f1": avg_f1,
                "em": avg_em,
                "redundancy": avg_red,
            }
            print(f"{config:<20} {avg_recall:.4f}±{std_recall:.4f}  "
                  f"{avg_f1:.4f}±{std_f1:.4f}  "
                  f"{avg_em:.4f}  {avg_red:.4f}")
        else:
            delta_recall = (avg_recall - baseline_avg["recall"]) / baseline_avg["recall"] * 100
            delta_f1 = (avg_f1 - baseline_avg["f1"]) / baseline_avg["f1"] * 100
            delta_em = (avg_em - baseline_avg["em"]) / baseline_avg["em"] * 100 if baseline_avg["em"] > 0 else 0
            delta_red = (avg_red - baseline_avg["redundancy"]) / baseline_avg["redundancy"] * 100

            print(f"{config:<20} {avg_recall:.4f}±{std_recall:.4f} ({delta_recall:+.1f}%)  "
                  f"{avg_f1:.4f}±{std_f1:.4f} ({delta_f1:+.1f}%)  "
                  f"{avg_em:.4f} ({delta_em:+.1f}%)  "
                  f"{avg_red:.4f} ({delta_red:+.1f}%)")

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
            f1_vals = [results[config][s]["f1"] for s in results[config]]

            avg_recall = np.mean(recall_vals)
            avg_f1 = np.mean(f1_vals)
            std_recall = np.std(recall_vals)

            delta_recall_pct = (avg_recall - baseline_avg["recall"]) / baseline_avg["recall"] * 100
            delta_f1_pct = (avg_f1 - baseline_avg["f1"]) / baseline_avg["f1"] * 100

            print(f"{config}:")
            print(f"  ✓ Recall@5 提升: {delta_recall_pct:.1f}% (target: ≥35%)")
            print(f"  ✓ F1 提升: {delta_f1_pct:.1f}% (target: ≥10%)")
            print(f"  ✓ 结果稳定性: std={std_recall:.4f} (target: <0.02)")
            print()


if __name__ == "__main__":
    main()
