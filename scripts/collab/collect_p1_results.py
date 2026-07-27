#!/usr/bin/env python3
"""Collect a P1 diagnosis round into exchange/p1_diagnosis/<timestamp>/.

Run after run_all_diagnosis.sh. Creates the round directory, copies the
committable artifacts, generates the README from what can be read off the
artifacts, appends a round-table row, packages the raw result.json, and commits.

Everything it writes is derived from the artifacts. It does not ask the runner
to write up conclusions or caveats — those get settled in conversation after
reading the reports, and a template that prompts for them just produces filler.

Deliberately separate from run_all_diagnosis.sh rather than chained onto it.
The diagnoses are pure-CPU and re-runnable, exit 2 on gamma saturation, and can
partially fail — a failed diagnosis should not quietly produce a deliverable,
and re-running analysis should not keep rewriting the exchange directory.

Two gates that would have caught the 2026-07-27 round at packaging time:

  --skip_generation / missing F1   that round's whole point was idea 1, whose
                                   criterion is whether raising gamma hurts QA
                                   quality; without F1 it cannot be judged
  dirty working tree               that round had a hand-edited yaml and two
                                   stale diagnosis scripts

check_dump_fields.py does not cover either: it checks dump fields, and that
round's dump fields were complete. F1 was the missing piece.

Usage:
    python scripts/collab/collect_p1_results.py
    python scripts/collab/collect_p1_results.py --who 师弟
    python scripts/collab/collect_p1_results.py --allow-no-f1   # 只在明知故犯时
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_lib import (  # noqa: E402
    CollectError, append_round_row, fmt_duration, git_provenance,
    gpu_info, load_json, prepare_round_dir, timestamp_from_start_time,
)

EXPERIMENT = "p1_diagnosis"
GAMMAS = ["0.0", "0.5", "1.0"]
IDEA_MAP = {
    "gamma_sweep.md": "idea 1 两阶段 QUBO",
    "context_dependency.md": "idea 4 上下文完整性",
    "complementarity.md": "idea 6 互补性矩阵",
    "qubo_objective.md": "idea 7 Soft QUBO",
}


def check_experiments(results_dir: Path, allow_no_f1: bool) -> tuple[list[dict], list[str]]:
    """Load the three status.json + result.json, refusing on the known traps."""
    per_gamma, warnings = [], []

    for g in GAMMAS:
        exp = results_dir / f"gamma_{g}"
        status = load_json(exp / "status.json")
        if status.get("status") != "success":
            raise CollectError(
                f"gamma_{g} 的 status = {status.get('status')}（不是 success）。\n"
                f"  先看 {exp}/stderr.log。\n"
                "  跑挂的实验不该产出交付物 —— 那些 result.json 可能是上一轮的陈数据。"
            )

        result = load_json(exp / "result.json")
        cfg, metrics = result.get("config", {}), result.get("metrics", {})

        if cfg.get("skip_generation"):
            msg = (f"gamma_{g} 是带 --skip_generation 跑的 → 没有 F1。\n"
                   "  这正是 2026-07-27 那趟的问题: idea 1 的判据(γ 增大在降冗余的\n"
                   "  同时是否伤了 QA 质量)缺一半，判不了。\n"
                   "  仓库的 phase1_diagnosis.yaml 没有这个 flag，是被手改进去的。\n"
                   "  确实要收集这种结果就加 --allow-no-f1。")
            if not allow_no_f1:
                raise CollectError(msg)
            warnings.append(f"gamma_{g}: skip_generation=True，无 F1")

        elif metrics.get("mean_f1") is None:
            msg = (f"gamma_{g} 没开 skip_generation 但 mean_f1 是 None。\n"
                   "  生成可能整段失败了(模型没加载上/OOM)，看 stderr.log。")
            if not allow_no_f1:
                raise CollectError(msg)
            warnings.append(f"gamma_{g}: mean_f1 缺失")

        per_gamma.append({
            "gamma": g, "status": status, "config": cfg, "metrics": metrics,
            "command": status.get("command", "(未记录)"),
            "elapsed": status.get("elapsed_seconds", 0.0),
        })

    return per_gamma, warnings


def copy_artifacts(dest: Path, results_dir: Path, analysis_dir: Path,
                   base_dir: Path) -> list[str]:
    copied = []

    for md in sorted(analysis_dir.glob("*.md")):
        shutil.copy2(md, dest / "analysis" / md.name)
        copied.append(f"analysis/{md.name}")

    for g in GAMMAS:
        src = results_dir / f"gamma_{g}" / "status.json"
        shutil.copy2(src, dest / "meta" / f"status_gamma_{g}.json")
        copied.append(f"meta/status_gamma_{g}.json")

    run_summary = base_dir / "run_summary.json"
    if run_summary.exists():
        shutil.copy2(run_summary, dest / "meta" / "run_summary.json")
        copied.append("meta/run_summary.json")

    # 真实命令行 —— 排查时最有用的东西。2026-07-27 那趟的手改就是从这里发现的。
    cmds = dest / "config" / "actual_commands.txt"
    cmds.write_text("".join(
        load_json(results_dir / f"gamma_{g}" / "status.json")["command"] + "\n"
        for g in GAMMAS
    ))
    copied.append("config/actual_commands.txt")

    # 手改过配置就把实际用的那份一起收进来
    for yml in (Path("scripts/tuning/config/phase1_diagnosis.yaml"),):
        if yml.exists():
            shutil.copy2(yml, dest / "config" / yml.name)
            copied.append(f"config/{yml.name}")

    return copied


def metrics_table(per_gamma: list[dict]) -> str:
    rows = ["| 配置 | Recall@5 | Precision@5 | 冗余度↓ | 多样性↑ | F1 |",
            "|---|---|---|---|---|---|"]

    def cell(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    for e in per_gamma:
        m = e["metrics"]
        rows.append(
            f"| γ={e['gamma']} | {cell(m.get('mean_recall'))} | "
            f"{cell(m.get('mean_precision'))} | {cell(m.get('mean_redundancy'))} | "
            f"{cell(m.get('mean_diversity'))} | {cell(m.get('mean_f1'))} |"
        )
    return "\n".join(rows)


def build_readme(per_gamma: list[dict], prov: dict, who: str, ts: str,
                 warnings: list[str], copied: list[str],
                 zip_path: Path | None) -> str:
    first = per_gamma[0]
    cfg = first["config"]
    start = first["status"].get("start_time", "")[:16].replace("T", " ")
    total = sum(e["elapsed"] for e in per_gamma)
    m0 = first["metrics"]

    reports = "\n".join(
        f"- `analysis/{f}` — {desc}"
        for f, desc in IDEA_MAP.items()
        if f"analysis/{f}" in copied
    )

    clean = "是" if not prov["dirty"] else "**否**"
    dirty_detail = ""
    if prov["dirty"]:
        lines = "\n".join(f"  - `{ln}`" for ln in prov["status_lines"][:20])
        dirty_detail = (
            f"\n\n  ⚠️ **工作区有未提交改动**，跑的代码与仓库 `{prov['head'].split()[0]}` 不一致：\n"
            f"{lines}\n\n"
            "  2026-07-27 那趟就是手改了 yaml 加 `--skip_generation`、"
            "外加两个诊断脚本是旧版，而四份报告里完全看不出来 —— 所以这里自动记下来。"
        )

    warn_block = ""
    if warnings:
        warn_block = ("\n> ⚠️ **收集时的告警**（用 `--allow-no-f1` 强制收集的）：\n"
                      + "\n".join(f"> - {w}" for w in warnings) + "\n")

    raw = "大产物没提交（见 `exchange/README.md`）。\n\n"
    if zip_path:
        size = zip_path.stat().st_size / 1048576
        raw += (f"- 文件: 3 × `result.json`（带 `--dump_passages`）\n"
                f"- 已打包: `{zip_path}` ({size:.1f} MB)，需要时单独发\n")
    else:
        raw += "- 未打包（`--no-zip`）\n"

    return f"""# Phase 1 诊断（4 个 idea 前提验证）— {who}, {start} 北京时间
{warn_block}
## 跑了什么

- **配置**: `phase1_diagnosis.yaml`，{cfg.get('max_samples')} 题 × {len(per_gamma)} 个 γ \
({'/'.join(e['gamma'] for e in per_gamma)})，`{cfg.get('corpus_mode')}`\
{'，answer scorer 开' if cfg.get('use_answer_scorer') else ''}\
{'，**生成关**' if cfg.get('skip_generation') else '，生成开'}
- **真实命令**: `config/actual_commands.txt`（从 `meta/status_*.json` 的 `command` 抽的）
- **耗时**: 每配置 {' / '.join(fmt_duration(e['elapsed']) for e in per_gamma)}，\
共 {fmt_duration(total)}
- **检索**: {m0.get('n_samples')} 题中 {m0.get('n_with_gold')} 题命中 gold\
（{m0.get('n_retrieval_failure')} 题检索阶段失败）

## 结果

{metrics_table(per_gamma)}

## 诊断报告

{reports or '(没有报告文件被收集到 —— run_all_diagnosis.sh 跑了吗？)'}

## 环境

- GPU: {gpu_info()}
- git HEAD: `{prov['head']}`
- 工作区干净: {clean}{dirty_detail}

## 原始数据

{raw}"""


def package_raw(results_dir: Path, ts: str) -> Path | None:
    out = Path.home() / f"P1_{ts}.zip"
    files = [(results_dir / f"gamma_{g}" / "result.json", g) for g in GAMMAS]
    if not all(p.exists() for p, _ in files):
        return None
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, g in files:
            z.write(p, f"P1_diagnosis/experiments/gamma_{g}/result.json")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="把一趟 P1 诊断收集进 exchange/p1_diagnosis/<时间戳>/")
    ap.add_argument("--base_dir", type=Path,
                    default=Path("scratch/research/P1_diagnosis"))
    ap.add_argument("--who", default=None, help="谁跑的（默认问）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的趟次目录")
    ap.add_argument("--allow-no-f1", action="store_true",
                    dest="allow_no_f1",
                    help="缺 F1 也收集（只在明知故犯时用）")
    ap.add_argument("--no-zip", action="store_true", dest="no_zip",
                    help="不打包原始 result.json")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    exchange_root = repo / "exchange"
    results_dir = args.base_dir / "experiments"
    analysis_dir = args.base_dir / "analysis"

    print("=== 收集 P1 诊断结果 ===")
    print(f"来源: {args.base_dir}")

    if not results_dir.is_dir():
        raise CollectError(f"实验目录不存在: {results_dir}\n"
                           "  先跑 bash scripts/collab/run_phase1_full.sh")
    if not analysis_dir.is_dir() or not list(analysis_dir.glob("*.md")):
        raise CollectError(
            f"没有诊断报告: {analysis_dir}\n"
            "  先跑 bash scripts/diagnosis/run_all_diagnosis.sh")

    per_gamma, warnings = check_experiments(results_dir, args.allow_no_f1)
    print(f"✅ 三个实验都是 success"
          + ("，F1 齐全" if not warnings else f"，但有 {len(warnings)} 条告警"))

    ts = timestamp_from_start_time(per_gamma[0]["status"]["start_time"])
    print(f"趟次时间戳: {ts}（取自 gamma_0.0 的 start_time，北京时间）")

    who = args.who
    if not who:
        try:
            who = input("谁跑的？(直接回车 = 师弟): ").strip() or "师弟"
        except EOFError:
            who = "师弟"

    prov = git_provenance(repo)
    if prov["dirty"]:
        print(f"⚠️  工作区有未提交改动（{len(prov['status_lines'])} 个文件）"
              "—— 会记进 README，请在「改动」栏说明")

    dest = prepare_round_dir(exchange_root, EXPERIMENT, ts, force=args.force)
    copied = copy_artifacts(dest, results_dir, analysis_dir, args.base_dir)
    print(f"✅ 拷贝 {len(copied)} 个文件到 {dest.relative_to(repo)}")

    zip_path = None if args.no_zip else package_raw(results_dir, ts)
    if zip_path:
        size = zip_path.stat().st_size / 1048576
        print(f"✅ 原始数据打包: {zip_path} ({size:.1f} MB) —— 这个要单独发")

    (dest / "README.md").write_text(
        build_readme(per_gamma, prov, who, ts, warnings, copied, zip_path))
    print("✅ README.md 已生成")

    if append_round_row(exchange_root / EXPERIMENT / "README.md", ts, who):
        print("✅ 趟次表已加一行")

    rel = dest.relative_to(repo)
    readme_rel = (exchange_root / EXPERIMENT / "README.md").relative_to(repo)
    subprocess.run(["git", "add", str(rel), str(readme_rel)], cwd=repo)
    subprocess.run(["git", "commit", "-q", "-m", f"exchange: P1 诊断 {ts}"],
                   cwd=repo)
    print(f"✅ 已提交: exchange: P1 诊断 {ts}")

    print("\n推上去：\n  git push\n")
    if zip_path:
        print(f"那份原始数据需要时单独发：{zip_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CollectError as e:
        # stdout 先 flush，不然报错会出现在进度行之前，读起来像是一开始就挂了。
        sys.stdout.flush()
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
