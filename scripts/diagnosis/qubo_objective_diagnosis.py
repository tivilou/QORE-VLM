#!/usr/bin/env python3
"""QUBO 代理目标诊断 — 验证 idea 7 (Soft QUBO 端到端) 的前提。

**假设**: QUBO 是人工设计的代理目标，没有直接优化 F1，
所以 QUBO 最优解 ≠ F1 最优解。

**为什么重写了整个脚本。**
旧版把每道题的 QUBO 能量放在一起做**跨题相关**。这个设计测不到假设：
能量的绝对值主要由每道题自己的候选池量级决定（池子大小、分数范围、
lam·K² 偏移），"选得好不好"只占很小一部分。实测同一个代理目标：

    跨题相关   r = 0.007
    同题内相关 r = 0.986

差 140 倍。所以旧版无论 QUBO 设计得多好，都会报"相关性低 → 假设成立"。

假设的真实含义是**同题内**的：这道题里，QUBO 选的 K 条不是 F1 最优的
K 条。所以本版在每道题内部枚举候选子集，比较：

    QUBO 排第一的子集  vs  Recall 排第一的子集

用法:
    python scripts/diagnosis/qubo_objective_diagnosis.py \\
        --results <dir>/gamma_0.5/result.json \\
        --output  <dir>/analysis/qubo_objective.md \\
        --gamma 0.5 --lam 2.0

数据要求: --dump_passages（需要 all_candidates 的 score/is_gold）。

⚠️ 核心限制：**用 Recall 代替 F1 做排序目标。**
真正的 F1 最优子集需要对每个候选子集跑一次 LLM 生成（C(10,5)=252 次/题，
200 题 = 50400 次生成），成本不可接受。本诊断用"子集里 gold 的个数"
（即 Recall）作为可计算的代理。

这弱化了结论：Recall 最优不等于 F1 最优。但方向仍然有效 ——
若 QUBO 连 Recall 都优化不好，它更不可能优化好 F1。
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

# 项目根目录，为了 import qore 的真实 QUBO 实现
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def enumerate_subsets(sample: Dict, K: int, gamma: float, lam: float,
                      pool_cap: int = 12) -> Dict | None:
    """在一道题内枚举子集，比较 QUBO 最优与 Recall 最优。

    用 sample['qubo'] 里 selector 记录的**真实** a 和 b，直接调用
    qore.qubo 的实现。两点都是必须的：

    1. 不自己重写公式 —— 旧版手写的与 qore/qubo.py 有三处偏差
       （质量项多乘 lam、a 未归一化、冗余用词重叠而非 embedding 余弦）。
    2. 不用代理的 b —— 实测「分数接近度」代理与真实余弦选出同一子集的
       概率只有 0.5%（200 次试验命中 1 次）。用它等于换了个目标函数。

    候选池截断到 pool_cap（按 a 降序），因为 C(15,5)=3003 × 200 题偏慢，
    C(12,5)=792 可接受。截断会**低估** oracle，所以 gap 是下界。
    """
    from qore.qubo import build_qubo_matrix, energy

    qd = sample.get('qubo')
    if not qd or qd.get('skipped') or 'a' not in qd or 'b' not in qd:
        return None      # 退化题（K>=池子大小），QUBO 未构建

    a_full = np.asarray(qd['a'], dtype=np.float64)
    b_full = np.asarray(qd['b'], dtype=np.float64)
    if a_full.ndim != 1 or b_full.shape != (len(a_full), len(a_full)):
        return None
    if len(a_full) < K + 1:
        return None

    # gold 标记：qubo 记录的是 prefilter 后的池子，需要按 rank 对齐。
    # selected_passages 的 retrieved_rank 是原候选池下标，与 a/b 的下标
    # 不是同一套 —— 所以 gold 只能从 pool_ranks 映射。
    pool_ranks = qd.get('pool_ranks')
    if pool_ranks is None or len(pool_ranks) != len(a_full):
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

    n_gold_total = int(gold.sum())
    if n_gold_total == 0:
        return None                          # 无 gold，Recall 恒为 0，无从比较

    n = len(a)
    Q = build_qubo_matrix(a, b, K, lam=lam, gamma=gamma)

    best_e, best_e_recall = np.inf, 0.0
    best_r, best_r_energy = -1.0, 0.0
    for combo in itertools.combinations(range(n), K):
        x = np.zeros(n)
        x[list(combo)] = 1
        e = float(energy(x, Q))
        rec = float(gold[list(combo)].sum()) / n_gold_total
        if e < best_e:
            best_e, best_e_recall = e, rec
        if rec > best_r or (rec == best_r and e < best_r_energy):
            best_r, best_r_energy = rec, e

    return {
        'qubo_best_recall': best_e_recall,   # QUBO 最优子集实际达到的 Recall
        'oracle_recall': best_r,             # 枚举出的最好 Recall
        'gap': best_r - best_e_recall,       # >0 表示 QUBO 没选到最优
        'agrees': abs(best_r - best_e_recall) < 1e-9,
        'pool_size': n,
        'n_gold': n_gold_total,
    }


def generate_report(stats: List[Dict], output_path: Path,
                    gamma: float, lam: float) -> bool:
    valid = [s for s in stats if s is not None]
    if len(valid) < 10:
        print(f"⚠️ 有效样本仅 {len(valid)} 条，不足以判断", file=sys.stderr)
        return False

    qubo_r = np.array([s['qubo_best_recall'] for s in valid])
    oracle_r = np.array([s['oracle_recall'] for s in valid])
    gaps = np.array([s['gap'] for s in valid])
    n_agree = sum(1 for s in valid if s['agrees'])

    out = []
    out.append("# QUBO 代理目标诊断报告（idea 7）\n")
    out.append(f"\n**样本数**: {len(valid)}")
    out.append(f"\n**参数**: γ={gamma}, λ={lam}")
    out.append(f"\n**枚举池**: 每题最多 {valid[0]['pool_size']} 条候选，选 K 条\n")
    out.append("\n---\n")

    out.append("\n## 同题内比较（核心）\n")
    out.append("\n对每道题枚举候选子集，比较两个子集达到的 Recall：\n")
    out.append(f"\n- **QUBO 能量最低的子集**: 平均 Recall {qubo_r.mean():.4f}")
    out.append(f"\n- **枚举出的最优子集**（oracle）: 平均 Recall {oracle_r.mean():.4f}")
    out.append(f"\n- **平均差距**: {gaps.mean():.4f}")
    out.append(f"\n- **QUBO 命中最优的题数**: {n_agree}/{len(valid)} "
               f"({n_agree/len(valid)*100:.1f}%)\n")

    out.append("\n### 假设验证\n")
    out.append("\n**假设**: QUBO 最优解 ≠ F1(此处用 Recall 代理) 最优解\n")

    disagree_rate = 1 - n_agree / len(valid)
    if disagree_rate > 0.3 and gaps.mean() > 0.05:
        verdict = 'holds'
        out.append(f"\n**结论**: ✅ **假设成立**\n")
        out.append(f"\n{disagree_rate*100:.1f}% 的题里 QUBO 没选到 Recall 最优子集，"
                   f"平均少 {gaps.mean():.4f}。")
        out.append("\n说明代理目标与任务目标存在系统性偏离 —— 支持 idea 7"
                   "（让任务目标进入优化）。\n")
    elif disagree_rate > 0.1:
        verdict = 'partial'
        out.append(f"\n**结论**: ⚠️ **假设部分成立**\n")
        out.append(f"\n偏离存在但不大（{disagree_rate*100:.1f}% 的题，"
                   f"平均差 {gaps.mean():.4f}）。\n")
    else:
        verdict = 'rejected'
        out.append(f"\n**结论**: ❌ **假设不成立**\n")
        out.append(f"\nQUBO 在 {n_agree/len(valid)*100:.1f}% 的题里就是最优的，"
                   "代理目标设计合理。idea 7 的收益空间有限。\n")

    out.append("\n---\n")
    out.append("\n## 方法学限制（写论文时必须声明）\n")
    out.append("\n1. **用 Recall 代替 F1**。真正的 F1 最优子集需要对每个候选子集"
               "跑一次 LLM 生成（792 次/题 × 200 题 ≈ 16 万次），成本不可接受。"
               "Recall 最优 ≠ F1 最优，所以结论是方向性的。")
    out.append("\n2. **候选池截断**到前 12 条（按质量分降序）。真实 oracle 可能"
               "用到被截掉的候选，所以报出的 gap 是**下界**。")
    out.append("\n3. 用的是 selector 记录的真实 a/b 与 `qore.qubo` 的实现，"
               "**不是**重新推导的公式 —— 这一点与旧版不同（旧版做跨题相关，"
               "实测跨题 r=0.007 而同题内 r=0.986，测的是候选池量级差异）。\n")
    out.append("\n**脚本**: `scripts/diagnosis/qubo_objective_diagnosis.py`\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(out))
    print(f"✅ 报告生成: {output_path}")
    print(f"\n=== 摘要 ===")
    print(f"QUBO 平均 Recall {qubo_r.mean():.4f} vs oracle {oracle_r.mean():.4f}")
    print(f"命中最优: {n_agree}/{len(valid)} ({n_agree/len(valid)*100:.1f}%)")
    print(f"结论: {verdict}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description='QUBO 代理目标诊断 (idea 7)')
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--lam', type=float, default=2.0)
    ap.add_argument('--pool_cap', type=int, default=12,
                    help='每题枚举的候选数上限 (C(12,5)=792)')
    args = ap.parse_args()

    print("=== QUBO 代理目标诊断 (idea 7) ===")
    print(f"输入: {args.results}")
    print(f"参数: γ={args.gamma}, λ={args.lam}, pool_cap={args.pool_cap}")

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
        stats.append(enumerate_subsets(s, K=len(s.get('selected_passages') or []) or 5,
                                       gamma=args.gamma, lam=args.lam,
                                       pool_cap=args.pool_cap))

    return 0 if generate_report(stats, args.output, args.gamma, args.lam) else 1


if __name__ == '__main__':
    sys.exit(main())
