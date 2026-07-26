#!/usr/bin/env python3
"""互补性诊断 — 验证 idea 6 的前提。

**假设**: 当前目标只**避免冗余**（负向惩罚相似），没有**主动寻找互补**
（正向奖励信息互补）。所以选出的段落彼此不重复，但合起来的信息覆盖不全。

**为什么这个假设在 NQ 上可测。**
"互补"需要一个不依赖多答案标注的定义。本诊断用的定义是：

    一组段落的覆盖度 = 该组命中的 gold 段落数 / 池子里 gold 段落总数

关键区分（这是 idea 6 与 idea 2 的差别，也是它在 NQ 上可测的原因）：
idea 2 需要"多个不同答案"（NQ 没有，gold_answers 是别名集合）；
idea 6 只需要"多条互补的证据段落"—— NQ 的 gold 段落可以有多条，
每条提供答案的不同侧面，这个是有的。

**核心测量**：把「低冗余」和「高覆盖」分开，看两者是否等价。
    如果 argmin(冗余) == argmax(覆盖) —— 避免冗余已经隐含了互补，idea 6 无增益
    如果两者分离 —— 存在只靠降冗余拿不到的覆盖，idea 6 有空间

用法:
    python scripts/diagnosis/complementarity_diagnosis.py \\
        --results <dir>/gamma_0.5/result.json \\
        --output  <dir>/analysis/complementarity.md

数据要求: --dump_passages（需要 qubo 的真实 a/b 和 all_candidates 的 gold）。

⚠️ 限制：覆盖度用 gold 段落数定义，是"信息互补"的代理而非本体。
两条 gold 段落可能讲同一件事（真冗余），也可能讲互补的两面 —— 本诊断
区分不了。要区分需要段落级的语义标注，NQ 没有。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnosis_io import DiagnosisInputError, load_samples

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def analyze_sample(sample: Dict, pool_cap: int = 12) -> Dict | None:
    """比较「最低冗余子集」与「最高覆盖子集」是否一致。"""
    qd = sample.get('qubo')
    if not qd or qd.get('skipped') or 'a' not in qd:
        return None

    a_full = np.asarray(qd['a'], dtype=np.float64)
    b_full = np.asarray(qd['b'], dtype=np.float64)
    pool_ranks = qd.get('pool_ranks')
    if pool_ranks is None or len(pool_ranks) != len(a_full):
        return None

    K = int(qd.get('K') or 0)
    if not (1 <= K <= len(a_full) - 1):
        return None

    gold_by_rank = {
        c['retrieved_rank']: bool(c.get('is_gold'))
        for c in (sample.get('all_candidates') or [])
    }
    gold_full = np.array([gold_by_rank.get(int(r), False) for r in pool_ranks])

    order = np.argsort(a_full)[::-1][:pool_cap]
    a = a_full[order]
    b = b_full[np.ix_(order, order)]
    gold = gold_full[order]

    n_gold = int(gold.sum())
    if n_gold == 0:
        return None
    n = len(a)
    if K > n - 1:
        return None

    best_red, red_cov = np.inf, 0.0
    best_cov, cov_red = -1.0, 0.0
    for combo in itertools.combinations(range(n), K):
        idx = list(combo)
        # 组内平均两两冗余
        pairs = [b[i, j] for i, j in itertools.combinations(idx, 2)]
        red = float(np.mean(pairs)) if pairs else 0.0
        cov = float(gold[idx].sum()) / n_gold
        if red < best_red:
            best_red, red_cov = red, cov
        if cov > best_cov or (cov == best_cov and red < cov_red):
            best_cov, cov_red = cov, red

    return {
        'min_red_coverage': red_cov,     # 最低冗余子集达到的覆盖
        'max_coverage': best_cov,        # 可达的最高覆盖
        'coverage_gap': best_cov - red_cov,
        'aligned': abs(best_cov - red_cov) < 1e-9,
        'min_redundancy': best_red,
        'max_cov_redundancy': cov_red,
        'n_gold': n_gold,
        'pool_size': n,
    }


def generate_report(stats: List[Dict], output_path: Path) -> bool:
    valid = [s for s in stats if s is not None]
    if len(valid) < 10:
        print(f"⚠️ 有效样本仅 {len(valid)} 条，不足以判断", file=sys.stderr)
        return False

    gaps = np.array([s['coverage_gap'] for s in valid])
    n_aligned = sum(1 for s in valid if s['aligned'])
    red_cov = np.array([s['min_red_coverage'] for s in valid])
    max_cov = np.array([s['max_coverage'] for s in valid])
    red_of_mincov = np.array([s['min_redundancy'] for s in valid])
    red_of_maxcov = np.array([s['max_cov_redundancy'] for s in valid])

    out = []
    out.append("# 互补性诊断报告（idea 6）\n")
    out.append(f"\n**样本数**: {len(valid)}\n")
    out.append("\n---\n")

    out.append("\n## 「降冗余」是否等价于「提覆盖」\n")
    out.append("\n对每道题枚举子集，分别找最低冗余的和最高覆盖的：\n")
    out.append(f"\n| | 覆盖度 | 组内冗余 |")
    out.append(f"\n|---|---|---|")
    out.append(f"\n| 最低冗余子集 | {red_cov.mean():.4f} | {red_of_mincov.mean():.4f} |")
    out.append(f"\n| 最高覆盖子集 | {max_cov.mean():.4f} | {red_of_maxcov.mean():.4f} |")
    out.append(f"\n\n- **平均覆盖差距**: {gaps.mean():.4f}")
    out.append(f"\n- **两者一致的题数**: {n_aligned}/{len(valid)} "
               f"({n_aligned/len(valid)*100:.1f}%)\n")

    out.append("\n### 假设验证\n")
    out.append("\n**假设**: 只避免冗余不足以获得最优信息覆盖\n")

    sep_rate = 1 - n_aligned / len(valid)
    if sep_rate > 0.3 and gaps.mean() > 0.05:
        verdict = 'holds'
        out.append(f"\n**结论**: ✅ **假设成立**\n")
        out.append(f"\n{sep_rate*100:.1f}% 的题里，最低冗余的子集并不是覆盖最好的，"
                   f"平均少 {gaps.mean():.4f} 覆盖度。")
        out.append("\n说明「降冗余」和「提互补」是两个不同的方向 —— "
                   "支持 idea 6（在目标里加正向互补项）。\n")
        out.append(f"\n注意最高覆盖子集的冗余（{red_of_maxcov.mean():.4f}）"
                   f"高于最低冗余子集（{red_of_mincov.mean():.4f}）—— "
                   "这正是「一味降冗余会牺牲覆盖」的直接证据。\n")
    elif sep_rate > 0.1:
        verdict = 'partial'
        out.append(f"\n**结论**: ⚠️ **假设部分成立**\n")
        out.append(f"\n分离存在但不大（{sep_rate*100:.1f}% 的题）。\n")
    else:
        verdict = 'rejected'
        out.append(f"\n**结论**: ❌ **假设不成立**\n")
        out.append(f"\n{n_aligned/len(valid)*100:.1f}% 的题里最低冗余子集就是覆盖最优的，"
                   "「避免冗余」已经隐含了互补。idea 6 无额外空间，优先级应下调。\n")

    out.append("\n---\n")
    out.append("\n## 方法学限制\n")
    out.append("\n1. **覆盖度用 gold 段落数定义**，是「信息互补」的代理。"
               "两条 gold 段落可能讲同一件事，本诊断区分不了。")
    out.append("\n2. 候选池截断到前 12 条，报出的差距是**下界**。")
    out.append("\n3. 与 idea 2 的区别：idea 2 需要多个不同答案（NQ 没有），"
               "idea 6 只需要多条 gold 证据段落（NQ 有）—— 这是它可测的原因。\n")
    out.append("\n**脚本**: `scripts/diagnosis/complementarity_diagnosis.py`\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(out))
    print(f"✅ 报告生成: {output_path}")
    print(f"\n=== 摘要 ===")
    print(f"最低冗余子集覆盖 {red_cov.mean():.4f} vs 最优覆盖 {max_cov.mean():.4f}")
    print(f"一致: {n_aligned}/{len(valid)} ({n_aligned/len(valid)*100:.1f}%)")
    print(f"结论: {verdict}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description='互补性诊断 (idea 6)')
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pool_cap', type=int, default=12)
    args = ap.parse_args()

    print("=== 互补性诊断 (idea 6) ===")
    print(f"输入: {args.results}")

    try:
        samples = load_samples(args.results, require=('qubo',))
    except DiagnosisInputError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print(f"加载 {len(samples)} 个样本，开始枚举…")
    stats = []
    for i, s in enumerate(samples):
        if i and i % 50 == 0:
            print(f"  进度 {i}/{len(samples)}")
        stats.append(analyze_sample(s, pool_cap=args.pool_cap))

    return 0 if generate_report(stats, args.output) else 1


if __name__ == '__main__':
    sys.exit(main())
