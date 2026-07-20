"""Batch evaluation suite: run multiple seeds × methods automatically.

Orchestrates N methods × M seeds, saves checkpoints, handles failures gracefully,
and aggregates results with statistical analysis (mean, std, significance tests).

Usage:
    python -m scripts.rag.eval_suite \\
        --dataset nq_open \\
        --corpus_mode aligned \\
        --corpus_output_dir data/nq_corpus \\
        --methods qore,mmr,topk \\
        --seeds 42,123,456 \\
        --K 5 \\
        --max_samples 100

This runs 9 configs (3 methods × 3 seeds), saves each result, then aggregates
with paired t-tests and confidence intervals.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser(description="Batch RAG evaluation suite")

    # Dataset (passed to eval_rag_refactored.py)
    p.add_argument("--dataset", default="nq_open")
    p.add_argument("--split", default="validation")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--custom_path", help="Path for jsonl dataset")

    # Corpus mode
    p.add_argument("--corpus_mode", default="aligned",
                   choices=["aligned", "precomputed", "faiss", "wiki_dpr"])
    p.add_argument("--corpus_output_dir", help="Aligned corpus cache dir")
    p.add_argument("--n_distractors", type=int, default=36000)
    # FAISS mode
    p.add_argument("--faiss_embeddings_path", help="Embeddings .npy for faiss mode")
    p.add_argument("--faiss_passages_path", help="Passages list for faiss mode")
    p.add_argument("--faiss_mmap", action="store_true",
                   help="Load embeddings as memmap for faiss mode")
    # wiki_dpr mode
    p.add_argument("--wiki_dpr_config", default="psgs_w100.nq.compressed",
                   help="facebook/wiki_dpr config for wiki_dpr mode")
    p.add_argument("--wiki_dpr_cache_dir", default=None,
                   help="HuggingFace cache dir for wiki_dpr dataset")
    p.add_argument("--wiki_dpr_nprobe", type=int, default=64,
                   help="IVFPQ search breadth for wiki_dpr mode")

    # Selection
    p.add_argument("--methods", default="qore,mmr,topk",
                   help="Comma-separated methods")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--num_reads", type=int, default=100)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--qore_prefilter_size", type=int, default=None,
                   help="QORE relevance-first candidate pool size")
    p.add_argument("--direct_solve_max_n", type=int, default=20,
                   help="QORE: max N for direct QUBO solve without prefilter")
    p.add_argument("--lambda_mmr", type=float, default=0.7)

    # Generation
    p.add_argument("--model_path",
                   default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--skip_generation", action="store_true")

    # Batch control
    p.add_argument("--seeds", default="42,123,456",
                   help="Comma-separated seeds")
    p.add_argument("--output_dir", default="results/rag/suite")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed runs")

    return p.parse_args()


def run_single(args, method: str, seed: int, output_dir: Path) -> dict:
    """Run one config via subprocess, return result dict or None on failure."""
    output_file = f"{method}_K{args.K}_seed{seed}.json"
    output_path = output_dir / output_file

    if args.resume and output_path.exists():
        print(f"  → Skipping {method} seed {seed} (already exists)")
        with open(output_path) as f:
            return json.load(f)

    # Build command
    cmd = [
        sys.executable, "-m", "scripts.rag.eval_rag_refactored",
        "--dataset", args.dataset,
        "--split", args.split,
        "--max_samples", str(args.max_samples),
        "--corpus_mode", args.corpus_mode,
        "--method", method,
        "--K", str(args.K),
        "--num_reads", str(args.num_reads),
        "--lam", str(args.lam),
        "--lambda_mmr", str(args.lambda_mmr),
        "--model_path", args.model_path,
        "--seed", str(seed),
        "--output_dir", str(output_dir),
        "--output_file", output_file,
    ]
    if args.gamma is not None:
        cmd += ["--gamma", str(args.gamma)]
    if args.qore_prefilter_size is not None:
        cmd += ["--qore_prefilter_size", str(args.qore_prefilter_size)]
    if args.direct_solve_max_n != 20:
        cmd += ["--direct_solve_max_n", str(args.direct_solve_max_n)]
    if args.corpus_output_dir:
        cmd += ["--corpus_output_dir", args.corpus_output_dir]
    if args.skip_generation:
        cmd += ["--skip_generation"]
    if args.custom_path:
        cmd += ["--custom_path", args.custom_path]

    # FAISS mode parameters
    if args.corpus_mode == "faiss":
        if hasattr(args, 'faiss_embeddings_path') and args.faiss_embeddings_path:
            cmd += ["--faiss_embeddings_path", args.faiss_embeddings_path]
        if hasattr(args, 'faiss_passages_path') and args.faiss_passages_path:
            cmd += ["--faiss_passages_path", args.faiss_passages_path]
        if hasattr(args, 'faiss_mmap') and args.faiss_mmap:
            cmd += ["--faiss_mmap"]

    # wiki_dpr mode parameters
    if args.corpus_mode == "wiki_dpr":
        if hasattr(args, 'wiki_dpr_config') and args.wiki_dpr_config:
            cmd += ["--wiki_dpr_config", args.wiki_dpr_config]
        if hasattr(args, 'wiki_dpr_cache_dir') and args.wiki_dpr_cache_dir:
            cmd += ["--wiki_dpr_cache_dir", args.wiki_dpr_cache_dir]
        if hasattr(args, 'wiki_dpr_nprobe') and args.wiki_dpr_nprobe:
            cmd += ["--wiki_dpr_nprobe", str(args.wiki_dpr_nprobe)]

    print(f"  → Running {method} seed {seed}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        with open(output_path) as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        print(e.stderr)
        return None


def aggregate_results(results_by_method: dict, output_dir: Path):
    """Compute mean/std/CI and significance tests, save summary."""
    summary = {"methods": {}, "significance": {}}

    # Collect metrics per method
    for method, runs in results_by_method.items():
        if not runs:
            continue
        metrics_lists = {}
        for run in runs:
            for k, v in run["metrics"].items():
                if k.startswith("mean_"):
                    metric = k[5:]  # strip "mean_"
                    metrics_lists.setdefault(metric, []).append(v)

        method_summary = {}
        for metric, vals in metrics_lists.items():
            method_summary[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }
        summary["methods"][method] = method_summary

    # Significance tests (QORE vs others)
    if "qore" in summary["methods"]:
        for other in ["mmr", "topk"]:
            if other not in summary["methods"]:
                continue
            for metric in summary["methods"]["qore"]:
                qore_vals = summary["methods"]["qore"][metric]["values"]
                other_vals = summary["methods"][other][metric]["values"]
                if len(qore_vals) == len(other_vals) and len(qore_vals) > 1:
                    t, p = stats.ttest_rel(qore_vals, other_vals)
                    sig_key = f"qore_vs_{other}_{metric}"
                    summary["significance"][sig_key] = {"t": float(t), "p": float(p)}

    # Save
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print("\n" + "=" * 70)
    print("Aggregated Results")
    print("=" * 70)
    print(f"Method       Recall@K         Redundancy       EM               F1")
    print("-" * 70)
    for method in ["qore", "mmr", "topk"]:
        if method not in summary["methods"]:
            continue
        m = summary["methods"][method]
        recall = m.get("recall", {})
        redun = m.get("redundancy", {})
        em = m.get("em", {})
        f1 = m.get("f1", {})
        print(
            f"{method:12s} "
            f"{recall.get('mean', 0):.4f}±{recall.get('std', 0):.4f}  "
            f"{redun.get('mean', 0):.4f}±{redun.get('std', 0):.4f}  "
            f"{em.get('mean', 0):.4f}±{em.get('std', 0):.4f}  "
            f"{f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}"
        )

    print("\nStatistical Significance (paired t-test):")
    print("-" * 70)
    for key, val in summary["significance"].items():
        p = val["p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"{key:40s} p={p:.4f} {sig}")

    print(f"\nSummary saved to: {summary_path}")


def main():
    args = parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RAG Batch Evaluation Suite")
    print("=" * 70)
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(methods) * len(seeds)}")
    print(f"Output: {output_dir}")
    print()

    # Run all configs
    results_by_method = {m: [] for m in methods}
    for method in methods:
        for seed in seeds:
            result = run_single(args, method, seed, output_dir)
            if result:
                results_by_method[method].append(result)

    # Aggregate
    aggregate_results(results_by_method, output_dir)
    print("\n✓ Batch evaluation complete")


if __name__ == "__main__":
    main()
