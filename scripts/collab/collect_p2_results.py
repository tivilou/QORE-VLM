#!/usr/bin/env python3
"""Collect a P2 experiment round into exchange/p2_solver_idea6/<timestamp>/.

Run after run_p2_experiments.sh. Creates the round directory, copies committable
artifacts, generates README from result.json files, and prepares for commit.

Usage:
    python scripts/collab/collect_p2_results.py <timestamp>
    python scripts/collab/collect_p2_results.py 20260728T120000

where <timestamp> is the YYYYMMDDTHHMMSS directory created by run_p2_experiments.sh
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_lib import (
    CollectError, fmt_duration, git_provenance, gpu_info, load_json,
)

EXPERIMENT = "p2_solver_idea6"
EXPECTED_CONFIGS = [
    ("0.5", "0.0", False),  # Baseline: solver fix only
    ("0.3", "0.1", True),
    ("0.3", "0.3", True),
    ("0.3", "0.5", True),
    ("0.5", "0.1", True),
    ("0.5", "0.3", True),
    ("0.5", "0.5", True),
    ("0.7", "0.1", True),
    ("0.7", "0.3", True),
    ("0.7", "0.5", True),
]


def check_experiments(round_dir: Path) -> tuple[list[dict], list[str]]:
    """Load all 10 result.json files and validate."""
    results, warnings = [], []

    for gamma, delta, use_comp in EXPECTED_CONFIGS:
        config_name = f"gamma{gamma}_delta{delta}"
        exp_dir = round_dir / config_name
        result_path = exp_dir / "result.json"

        if not result_path.exists():
            raise CollectError(
                f"缺少 {config_name}/result.json。\n"
                f"  检查 {exp_dir} 是否存在，实验是否跑完。"
            )

        result = load_json(result_path)
        cfg = result.get("config", {})
        metrics = result.get("metrics", {})

        # 验证配置
        actual_gamma = cfg.get("gamma")
        actual_delta = cfg.get("delta", 0.0)
        if abs(actual_gamma - float(gamma)) > 1e-6:
            warnings.append(f"{config_name}: gamma 不匹配 (期望 {gamma}, 实际 {actual_gamma})")
        if abs(actual_delta - float(delta)) > 1e-6:
            warnings.append(f"{config_name}: delta 不匹配 (期望 {delta}, 实际 {actual_delta})")

        # 验证关键指标存在
        required_metrics = ["mean_recall", "mean_redundancy", "mean_f1"]
        missing = [m for m in required_metrics if metrics.get(m) is None]
        if missing:
            raise CollectError(
                f"{config_name} 缺少关键指标: {missing}\n"
                f"  检查实验是否正常完成。"
            )

        results.append({
            "config_name": config_name,
            "gamma": float(gamma),
            "delta": float(delta),
            "use_complementarity": use_comp,
            "config": cfg,
            "metrics": metrics,
            "path": exp_dir,
        })

    return results, warnings


def generate_readme(
    round_dir: Path,
    timestamp: str,
    results: list[dict],
    warnings: list[str],
    git_state: str,
) -> str:
    """Generate README.md content."""
    lines = [f"# Phase 2 实验结果 - {timestamp}\n"]

    # Git 状态
    lines.append("## Git 状态\n")
    lines.append("```")
    lines.append(git_state.strip())
    lines.append("```\n")

    # 警告
    if warnings:
        lines.append("## ⚠️ 警告\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    # 实验配置总览
    lines.append("## 实验配置\n")
    lines.append(f"- **数据集**: {results[0]['config'].get('dataset', 'nq_open')}")
    lines.append(f"- **样本数**: {results[0]['config'].get('max_samples', 200)}")
    lines.append(f"- **方法**: {results[0]['config'].get('method', 'qore')}")
    lines.append(f"- **K**: {results[0]['config'].get('K', 5)}")
    lines.append(f"- **λ**: {results[0]['config'].get('lam', 2.0)}")
    lines.append(f"- **种子**: {results[0]['config'].get('seed', 42)}\n")

    # 结果表格
    lines.append("## 实验结果\n")
    lines.append("| Config | γ | δ | Complementarity | Recall@5 | Redundancy | F1 | EM | Precision |")
    lines.append("|--------|---|---|-----------------|----------|------------|----|----|-----------|")

    for r in results:
        m = r["metrics"]
        comp_str = "dpr" if r["use_complementarity"] else "-"
        baseline_marker = " **[baseline]**" if r["delta"] == 0.0 else ""

        lines.append(
            f"| {r['config_name']}{baseline_marker} | "
            f"{r['gamma']:.1f} | {r['delta']:.1f} | {comp_str} | "
            f"{m.get('mean_recall', 0.0):.4f} | "
            f"{m.get('mean_redundancy', 0.0):.4f} | "
            f"{m.get('mean_f1', 0.0):.4f} | "
            f"{m.get('mean_em', 0.0):.4f} | "
            f"{m.get('mean_precision', 0.0):.4f} |"
        )

    lines.append("")

    # Baseline 对比
    baseline = next(r for r in results if r["delta"] == 0.0)
    lines.append("## 与 Baseline 对比\n")
    lines.append(f"**Baseline** (solver fix, γ=0.5, δ=0.0):")
    bm = baseline["metrics"]
    lines.append(f"- Recall@5: {bm.get('mean_recall', 0.0):.4f}")
    lines.append(f"- Redundancy: {bm.get('mean_redundancy', 0.0):.4f}")
    lines.append(f"- F1: {bm.get('mean_f1', 0.0):.4f}")
    lines.append(f"- EM: {bm.get('mean_em', 0.0):.4f}\n")

    lines.append("**Idea 6 最佳配置**（按目标：冗余度最低且 F1 ≥ baseline）:\n")

    # 找符合约束的最佳配置
    candidates = [
        r for r in results
        if r["use_complementarity"]
        and r["metrics"].get("mean_f1", 0.0) >= bm.get("mean_f1", 0.0)
    ]

    if candidates:
        best = min(candidates, key=lambda r: r["metrics"].get("mean_redundancy", 1.0))
        bst_m = best["metrics"]
        lines.append(f"- **Config**: {best['config_name']} (γ={best['gamma']}, δ={best['delta']})")
        lines.append(f"- Recall@5: {bst_m.get('mean_recall', 0.0):.4f} (Δ={bst_m.get('mean_recall', 0.0) - bm.get('mean_recall', 0.0):+.4f})")
        lines.append(f"- Redundancy: {bst_m.get('mean_redundancy', 0.0):.4f} (Δ={bst_m.get('mean_redundancy', 0.0) - bm.get('mean_redundancy', 0.0):+.4f})")
        lines.append(f"- F1: {bst_m.get('mean_f1', 0.0):.4f} (Δ={bst_m.get('mean_f1', 0.0) - bm.get('mean_f1', 0.0):+.4f})")
        lines.append(f"- EM: {bst_m.get('mean_em', 0.0):.4f} (Δ={bst_m.get('mean_em', 0.0) - bm.get('mean_em', 0.0):+.4f})")
    else:
        lines.append("*无配置满足约束（F1 ≥ baseline）*")

    lines.append("")

    # 系统信息
    lines.append("## 系统信息\n")
    lines.append("```")
    lines.append(gpu_info())
    lines.append("```\n")

    # 产物清单
    lines.append("## 产物清单\n")
    lines.append("```")
    for r in results:
        lines.append(f"{r['config_name']}/")
        lines.append(f"  result.json  # 完整结果（不提交，已打包到 .zip）")
        if (r["path"] / "config.yaml").exists():
            lines.append(f"  config.yaml  # 实验配置")
    lines.append("```\n")

    lines.append("---\n")
    lines.append(f"*生成于 {timestamp} by scripts/collab/collect_p2_results.py*")

    return "\n".join(lines)


def package_results(round_dir: Path, timestamp: str):
    """Package all result.json files into a zip."""
    zip_path = round_dir / f"results_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for result_file in round_dir.rglob("result.json"):
            rel_path = result_file.relative_to(round_dir)
            zf.write(result_file, arcname=rel_path)
    print(f"✓ 打包结果到 {zip_path.relative_to(Path.cwd())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timestamp", help="实验批次时间戳 (YYYYMMDDTHHMMSS)")
    parser.add_argument("--who", default="Q", help="提交者名字（默认: Q）")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    exchange_base = repo / "exchange" / EXPERIMENT
    round_dir = exchange_base / args.timestamp

    if not round_dir.exists():
        print(f"ERROR: 目录不存在: {round_dir}")
        print(f"  检查时间戳是否正确，或先运行 run_p2_experiments.sh")
        sys.exit(1)

    print(f"=== 收集 Phase 2 实验结果: {args.timestamp} ===\n")

    # 检查实验
    print(">>> 检查实验完整性...")
    try:
        results, warnings = check_experiments(round_dir)
        print(f"✓ 找到 {len(results)} 个配置的结果")
        if warnings:
            print(f"⚠️  {len(warnings)} 个警告")
    except CollectError as e:
        print(f"✗ {e}")
        sys.exit(1)

    # Git 状态
    print("\n>>> 收集 Git 状态...")
    git_state = git_provenance(repo)
    (round_dir / "meta").mkdir(exist_ok=True)
    (round_dir / "meta" / "git_state.txt").write_text(git_state)
    print("✓ 写入 meta/git_state.txt")

    # 生成 README
    print("\n>>> 生成 README...")
    readme_content = generate_readme(round_dir, args.timestamp, results, warnings, git_state)
    (round_dir / "README.md").write_text(readme_content)
    print(f"✓ 写入 {round_dir.relative_to(repo)}/README.md")

    # 打包结果
    print("\n>>> 打包 result.json...")
    package_results(round_dir, args.timestamp)

    # 完成
    print("\n" + "=" * 60)
    print("✓ 收集完成！")
    print("\n下一步:")
    print(f"  1. 查看 {round_dir.relative_to(repo)}/README.md")
    print(f"  2. git add exchange/{EXPERIMENT}/{args.timestamp}/")
    print(f"  3. git commit -m 'experiment(p2): solver+idea6 results {args.timestamp}'")
    print(f"  4. git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
