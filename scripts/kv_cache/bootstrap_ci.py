#!/usr/bin/env python3
"""
Statistical analysis for KV-cache evaluation results with bootstrap confidence intervals.

Usage:
    # Run eval multiple times with different seeds:
    python scripts/kv_cache/eval_kv_cache.py --policy qore --seed 42 --max_samples 30 --output_file qore_run1.json
    python scripts/kv_cache/eval_kv_cache.py --policy qore --seed 43 --max_samples 30 --output_file qore_run2.json
    python scripts/kv_cache/eval_kv_cache.py --policy qore --seed 44 --max_samples 30 --output_file qore_run3.json

    # Then aggregate with bootstrap CI:
    python scripts/kv_cache/bootstrap_ci.py results/kv_cache/longbench/qore_run*.json

Outputs:
    - Mean ± 95% CI for F1, latency, throughput, memory
    - Per-run breakdown
    - Combined summary CSV
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats


def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95, stat_fn=np.mean):
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        n_bootstrap: Number of bootstrap resamples
        confidence: CI level (0.95 = 95%)
        stat_fn: Statistic to compute (default: mean)

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    rng = np.random.RandomState(0)  # fixed seed for reproducibility

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(stat_fn(sample))

    boot_stats = np.array(boot_stats)
    point = stat_fn(data)
    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return point, lower, upper


def load_results(paths):
    """Load multiple result JSON files."""
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def aggregate_runs(runs, n_bootstrap=10000):
    """Aggregate metrics across multiple runs with bootstrap CI."""

    # Collect per-sample metrics across all runs
    all_f1 = []
    all_latency = []
    all_throughput = []
    all_cache_len = []
    all_peak_mem = []
    all_resident_mem = []

    for run in runs:
        # Macro F1 per run (already aggregated in the JSON)
        if "metrics" in run and "macro_f1" in run["metrics"]:
            all_f1.append(run["metrics"]["macro_f1"])

        # Performance metrics
        perf = run.get("performance", {})
        if "avg_time_ms" in perf:
            all_latency.append(perf["avg_time_ms"])
        if "avg_tokens_per_sec" in perf:
            all_throughput.append(perf["avg_tokens_per_sec"])

        # Compression metrics
        comp = run.get("compression", {})
        measured = run.get("measured", {})

        if "avg_final_cache_len" in comp:
            all_cache_len.append(comp["avg_final_cache_len"])
        if "avg_peak_MB" in measured:
            all_peak_mem.append(measured["avg_peak_MB"])
        if "avg_resident_MB" in measured:
            all_resident_mem.append(measured["avg_resident_MB"])

    # Compute bootstrap CI for each metric
    summary = {}

    if all_f1:
        f1_mean, f1_low, f1_high = bootstrap_ci(all_f1, n_bootstrap)
        summary["f1"] = {
            "mean": round(f1_mean, 4),
            "ci_lower": round(f1_low, 4),
            "ci_upper": round(f1_high, 4),
            "n_runs": len(all_f1)
        }

    if all_latency:
        lat_mean, lat_low, lat_high = bootstrap_ci(all_latency, n_bootstrap)
        summary["latency_ms"] = {
            "mean": round(lat_mean, 1),
            "ci_lower": round(lat_low, 1),
            "ci_upper": round(lat_high, 1),
            "n_runs": len(all_latency)
        }

    if all_throughput:
        thr_mean, thr_low, thr_high = bootstrap_ci(all_throughput, n_bootstrap)
        summary["throughput_tok_s"] = {
            "mean": round(thr_mean, 2),
            "ci_lower": round(thr_low, 2),
            "ci_upper": round(thr_high, 2),
            "n_runs": len(all_throughput)
        }

    if all_cache_len:
        cache_mean, cache_low, cache_high = bootstrap_ci(all_cache_len, n_bootstrap)
        summary["avg_cache_len"] = {
            "mean": round(cache_mean, 1),
            "ci_lower": round(cache_low, 1),
            "ci_upper": round(cache_high, 1),
            "n_runs": len(all_cache_len)
        }

    if all_peak_mem:
        peak_mean, peak_low, peak_high = bootstrap_ci(all_peak_mem, n_bootstrap)
        summary["peak_mem_MB"] = {
            "mean": round(peak_mean, 1),
            "ci_lower": round(peak_low, 1),
            "ci_upper": round(peak_high, 1),
            "n_runs": len(all_peak_mem)
        }

    if all_resident_mem:
        res_mean, res_low, res_high = bootstrap_ci(all_resident_mem, n_bootstrap)
        summary["resident_mem_MB"] = {
            "mean": round(res_mean, 1),
            "ci_lower": round(res_low, 1),
            "ci_upper": round(res_high, 1),
            "n_runs": len(all_resident_mem)
        }

    return summary


def print_summary(summary, policy_name):
    """Pretty-print the aggregated summary."""
    print(f"\n{'='*60}")
    print(f"Policy: {policy_name}")
    print(f"{'='*60}\n")

    for metric, stats in summary.items():
        mean = stats["mean"]
        low = stats["ci_lower"]
        high = stats["ci_upper"]
        n = stats["n_runs"]

        # Format metric name
        name_map = {
            "f1": "F1 Score",
            "latency_ms": "Latency (ms)",
            "throughput_tok_s": "Throughput (tok/s)",
            "avg_cache_len": "Avg Cache Length",
            "peak_mem_MB": "Peak Memory (MB)",
            "resident_mem_MB": "Resident Memory (MB)"
        }
        display_name = name_map.get(metric, metric)

        print(f"{display_name:25} {mean:8.2f}  [95% CI: {low:8.2f} - {high:8.2f}]  (n={n})")

    print()


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI analysis for KV-cache eval")
    parser.add_argument("result_files", nargs="+", help="JSON result files from multiple runs")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Number of bootstrap samples")
    parser.add_argument("--output", type=str, help="Optional: save summary to JSON")
    args = parser.parse_args()

    print(f"Loading {len(args.result_files)} result files...")
    runs = load_results(args.result_files)

    # Extract policy name from first run
    policy_name = runs[0].get("policy", "unknown") if runs else "unknown"

    print(f"Aggregating {len(runs)} runs with {args.n_bootstrap} bootstrap samples...")
    summary = aggregate_runs(runs, args.n_bootstrap)

    print_summary(summary, policy_name)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"policy": policy_name, "summary": summary, "n_runs": len(runs)}, f, indent=2)
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
