#!/usr/bin/env python3
"""Pre-flight check: does a result.json carry what the diagnoses need?

Run this after the quick test and before committing 1.5-2h to the full run.

Why this exists as a script rather than the inline `python -c` it replaces:
the inline version could not tell three very different failures apart, because
all three print the same wall of `False`:

  1. the file is STALE — left over from a run predating --dump_passages
  2. the run FAILED or timed out, leaving a partial/old file in place
  3. --dump_passages genuinely did not reach the eval

It also crashed rather than reported: `d['samples'][0]` raises IndexError on an
empty sample list, and KeyError on a legacy file whose key is 'results'. And it
always exited 0, so it could not gate anything.

So this checks provenance (mtime, git HEAD, config flags, sample count) before
checking fields, and exits non-zero with a concrete cause.

Exit codes:
  0  all required fields present
  1  fields missing, or the file is unusable/unreadable
  2  file looks stale or partial — rerun before trusting anything
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# (sample field, why it is needed) — keep in sync with the dump written by
# eval_rag_refactored's --dump_passages branch.
SAMPLE_FIELDS = [
    ("question", "上下文依赖 / 互补性诊断需要问题文本"),
    ("selected_passages", "选中的 K 条 (text/score/is_gold/matched_answers)"),
    ("all_candidates", "区分 gold 未进池 vs 进池未选中"),
    ("qubo", "solver 真实看到的目标函数"),
]

# Nested under sample["qubo"]. Absent legitimately only in the degenerate
# K >= pool_size - 1 case, where _record_qubo_diagnostics writes "skipped".
QUBO_FIELDS = [
    ("a", "归一化质量向量 (在全部 N 个候选上归一化，下游无法重建)"),
    ("b", "余弦冗余矩阵 (需要 embedding，JSON 不带)"),
    ("pool_ranks", "池内位置 → 原候选 index 的映射"),
]


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="检查 result.json 是否带齐诊断所需字段")
    p.add_argument(
        "--results",
        type=Path,
        default=Path("scratch/research/quick_test/experiments/gamma_0.5/result.json"),
        help="要检查的 result.json",
    )
    p.add_argument(
        "--expect_samples",
        type=int,
        default=0,
        help="期望的样本数 (0=不检查)。对不上说明读的不是这次的产物",
    )
    args = p.parse_args()

    path: Path = args.results
    print(f"检查文件: {path}")

    if not path.exists():
        print(f"\n❌ 文件不存在。Step 2 没跑成功，或者输出目录不是这个。")
        print(f"   先跑: bash scripts/collab/run_phase1_quick.sh")
        return 1

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"\n❌ 不是合法 JSON: {e}")
        print("   通常意味着评测被中途杀掉，写了一半。删掉重跑。")
        return 2

    # ── 来源信息：先确认这文件是这次跑的 ────────────────────────────
    mtime = path.stat().st_mtime
    age_h = (time.time() - mtime) / 3600
    print(f"  写入时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))} "
          f"({age_h:.1f} 小时前)")
    print(f"  当前 HEAD: {git_head()}")

    cfg = data.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    # KEY ABSENT vs False 是两种不同的病:
    #   缺键   = 这文件由 --dump_passages 存在之前的代码写的 (陈旧产物)
    #   False = 拉到了新代码但跑的时候没带上这个 flag
    if "dump_passages" not in cfg:
        print("\n❌ config 里没有 'dump_passages' 这个键。")
        print("   说明该文件是**旧代码**写的（早于 --dump_passages 的引入），")
        print("   不是这次跑出来的。清掉重跑：")
        print("     git pull")
        print(f"     rm -rf {path.parent.parent.parent}")
        print("     bash scripts/collab/run_phase1_quick.sh")
        return 2
    if not cfg.get("dump_passages"):
        print("\n❌ config.dump_passages = False —— 评测没带这个 flag。")
        print("   检查用的是仓库里的 yaml（应有 dump_passages: true）：")
        print("     grep -n dump_passages scripts/tuning/config/quick_test.yaml")
        return 1

    samples = data.get("samples")
    if samples is None:
        samples = data.get("results")
        if samples is None:
            print(f"\n❌ 没有 'samples' 键。顶层键: {sorted(data)}")
            return 1
        print("\n⚠️  用的是旧键名 'results' —— 陈旧产物，删掉重跑。")
        return 2
    if not samples:
        print("\n❌ 'samples' 是空的：评测一题都没跑完。看 stderr.log。")
        return 2

    n = len(samples)
    print(f"  样本数: {n}")
    if args.expect_samples and n != args.expect_samples:
        print(f"\n⚠️  期望 {args.expect_samples} 题，实际 {n} 题。")
        print("   读到的可能不是这次的产物，或评测中途被杀。")
        return 2

    # ── 字段检查 ────────────────────────────────────────────────────
    print()
    s = samples[0]
    missing: list[str] = []
    for key, why in SAMPLE_FIELDS:
        ok = key in s and s[key] is not None
        print(f"  {'✅' if ok else '❌'} {key:<20} {why}")
        if not ok:
            missing.append(key)

    qubo = s.get("qubo") or {}
    if qubo.get("skipped"):
        # 合法缺失：K >= pool_size-1 时 QUBO 没构建。
        print(f"\n⚠️  qubo.skipped: {qubo['skipped']}")
        print("   这是退化选择（K 太接近池子大小），不是 bug，")
        print("   但 idea 7 的诊断需要真实构建过的 QUBO。检查 K / direct_solve_max_n。")
        return 2
    for key, why in QUBO_FIELDS:
        ok = key in qubo
        print(f"  {'✅' if ok else '❌'} qubo.{key:<15} {why}")
        if not ok:
            missing.append(f"qubo.{key}")

    print()
    if missing:
        print(f"❌ 缺 {len(missing)} 个字段: {missing}")
        print("   停在这里，把上面整段输出发我 —— 不要往 Step 4 走。")
        return 1

    print("✅ 七项齐全，可以进 Step 4。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
