#!/usr/bin/env python3
"""γ sweep 诊断 — Phase 1 五个诊断中的第 1 个。

验证两阶段 QUBO 的假设：质量与多样性在单目标 QUBO 中冲突。

读取 run_tuning_suite 产出的三个 γ 配置结果，对比 Recall/冗余度随 γ 的
变化，判断属于哪种场景，并给出 Phase 2 的参数起点。

使用方法:
    python scripts/diagnosis/gamma_sweep_diagnosis.py \
        --results_dir scratch/research/P1_diagnosis/experiments \
        --output scratch/research/P1_diagnosis/analysis/gamma_sweep.md

⚠️ 两个已知限制（写论文时不要越过）:

1. **差距大 ≠ 冲突**。本诊断能证明 γ 有效、能给出权衡曲线上的三个点，
   但「两个目标无法同时最优」需要 Pareto 前沿，三点 sweep 给不出。
2. **F1 需要生成**。若评测带了 --skip_generation，结果里没有 mean_f1，
   本诊断只能基于 Recall/冗余度判断，不能声称对下游 QA 的影响。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GAMMA_CONFIGS = ['gamma_0.0', 'gamma_0.5', 'gamma_1.0']


def load_results(results_dir: Path) -> dict:
    """加载三个 γ 配置的聚合指标。

    目录布局由 scripts/tuning/config/phase1_diagnosis.yaml 决定：
        <results_dir>/gamma_<value>/result.json

    早前这里读的是 <results_dir>/gamma_<value>.json（少了一层目录），
    永远 0/3 命中，analysis.md 因此一直是空的。
    """
    results = {}

    for config in GAMMA_CONFIGS:
        file_path = Path(results_dir) / config / "result.json"
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            continue

        with open(file_path) as f:
            data = json.load(f)

        metrics = data.get('metrics') or {}
        if 'mean_recall' not in metrics:
            print(f"❌ {file_path} 的 metrics 缺 mean_recall（评测是否失败？）")
            continue

        results[config] = {
            'recall': metrics['mean_recall'],
            'precision': metrics.get('mean_precision', 0.0),
            'redundancy': metrics['mean_redundancy'],
            'diversity': 1 - metrics['mean_redundancy'],
            # F1 只在评测未加 --skip_generation 时存在
            'f1': metrics.get('mean_f1'),
            'n_samples': metrics.get('n_samples', 0),
            'n_with_gold': metrics.get('n_with_gold', 0),
        }

    return results


def selection_overlap(results_dir: Path) -> dict | None:
    """三个 γ 选出段落的两两 Jaccard 重合度。

    这是 γ sweep 的**前置门控**。聚合指标接近不足以区分两种情况：
      (a) γ 生效了，但多样性对 Recall 影响本来就小
      (b) γ 没生效 —— QUBO 非对角项 Q_ij = γ·b_ij + 2·lam，若 b 的量级
          远小于 2·lam，放大 γ 只是等比放大一个被压死的项，argmin 不变
    两者的结论完全相反：(a) 说明多样性无用，(b) 说明参数需要重标定。
    只看 mean_recall 无法分辨，必须比对逐题的选择集合。

    Returns None 时表示缺 selected_passages（评测未加 --dump_passages），
    此时调用方应把 sweep 结论标为"未经饱和检查"。
    """
    per_gamma = {}
    for config in GAMMA_CONFIGS:
        path = Path(results_dir) / config / "result.json"
        if not path.exists():
            return None
        samples = json.loads(path.read_text()).get("samples") or []
        picks = {}
        for s in samples:
            sp = s.get("selected_passages")
            if not sp:
                return None          # 缺 dump，无法做这项检查
            ranks = [p.get("retrieved_rank") for p in sp]
            if any(r is None for r in ranks):
                return None
            picks[s.get("question_id")] = frozenset(ranks)
        per_gamma[config] = picks

    common = set.intersection(*(set(p) for p in per_gamma.values()))
    if not common:
        return None

    pairs = {}
    for i, ga in enumerate(GAMMA_CONFIGS):
        for gb in GAMMA_CONFIGS[i + 1:]:
            js = []
            for qid in common:
                A, B = per_gamma[ga][qid], per_gamma[gb][qid]
                union = A | B
                js.append(len(A & B) / len(union) if union else 1.0)
            pairs[f"{ga} vs {gb}"] = sum(js) / len(js)

    mean_j = sum(pairs.values()) / len(pairs)
    return {
        "pairs": pairs,
        "mean_jaccard": mean_j,
        "n_questions": len(common),
        # >0.95 = 三个 γ 几乎选出同一批段落 -> γ 未生效
        "saturated": mean_j > 0.95,
    }


def analyze_results(results: dict, overlap: dict | None = None) -> dict:
    """判断场景并给出 Phase 2 建议。"""
    r0, r5, r10 = (results[c] for c in GAMMA_CONFIGS)

    recall_gain = r10['recall'] - r0['recall']

    best_recall_config = max(results, key=lambda k: results[k]['recall'])
    best_precision_config = max(results, key=lambda k: results[k]['precision'])

    if r0['recall'] > r5['recall'] + 0.01 and r0['recall'] > r10['recall'] + 0.01:
        scenario = 'pure_quality'
        recommendation = (
            "纯质量（γ=0.0）显著最优。\n"
            "建议：不实现两阶段 QUBO，直接用 Top-K on Answer Scorer，"
            "或研究为什么多样性有害。"
        )
    elif abs(r0['recall'] - r5['recall']) < 0.01 and abs(r5['recall'] - r10['recall']) < 0.01:
        scenario = 'no_effect'
        recommendation = (
            "γ 对 Recall 几乎无影响（差距 <1%）。\n"
            "但**先看饱和检查那一节再下结论** —— 若 Jaccard 接近 1，\n"
            "说明 γ 根本没改变选择（QUBO 里被 2·lam 压住），\n"
            "那是参数标定问题，不是「多样性无用」。"
        )
    elif r10['recall'] > r5['recall'] + 0.01 and r10['recall'] > r0['recall'] + 0.01:
        scenario = 'pure_diversity'
        recommendation = (
            "纯多样性（γ=1.0）显著最优（罕见情况）。\n"
            "建议：重新审视设计假设，可能 Answer Scorer 质量信号有问题。"
        )
    else:
        scenario = 'balanced'
        recommendation = (
            "质量和多样性都重要，需要权衡。\n"
            "建议：实现两阶段 QUBO（Phase 2）。\n"
            f"  - Stage 1: γ=0 或 0.1（Recall={r0['recall']:.3f}，偏质量）\n"
            f"  - Stage 2: γ=0.6-0.7（在 Stage 1 基础上增加多样性）"
        )

    # 饱和时覆盖场景判断：选择集合几乎相同，聚合指标的差异是噪声，
    # 任何「多样性有用/无用」的结论都不成立。
    if overlap is not None and overlap['saturated']:
        scenario = 'gamma_saturated'
        recommendation = (
            f"γ 未生效（选择重合度 {overlap['mean_jaccard']:.3f} > 0.95）。\n"
            "本次 sweep 无法判断多样性的作用 —— 先重标定 QUBO 系数\n"
            "（降 lam，或归一化 b 使其与 a 量级相当），再重跑本诊断。\n"
            "**不要据此判断 idea 1 的去留。**"
        )

    return {
        'scenario': scenario,
        'recommendation': recommendation,
        'recall_gain_from_diversity': recall_gain,
        'best_recall_config': best_recall_config,
        'best_precision_config': best_precision_config,
        'has_f1': all(results[c]['f1'] is not None for c in GAMMA_CONFIGS),
        'overlap': overlap,
    }


def generate_report(results: dict, analysis: dict, output_path: Path) -> None:
    """生成 γ sweep 报告。"""
    r0, r5, r10 = (results[c] for c in GAMMA_CONFIGS)
    has_f1 = analysis['has_f1']

    out = []
    out.append("# γ Sweep 诊断报告\n")
    out.append(f"\n**样本数**: {r5['n_samples']} 题")
    out.append(f"\n**检索命中 gold**: {r5['n_with_gold']}/{r5['n_samples']}\n")
    out.append("\n---\n")

    out.append("\n## 实验结果对比\n")
    header = "\n| 配置 | Recall@5 | Precision@5 | 冗余度↓ | 多样性↑ |"
    sep = "\n|------|----------|-------------|---------|---------|"
    if has_f1:
        header += " F1 |"
        sep += "----|"
    out.append(header)
    out.append(sep)

    for config in GAMMA_CONFIGS:
        r = results[config]
        row = (f"\n| γ={config.split('_')[1]} | {r['recall']:.4f} | "
               f"{r['precision']:.4f} | {r['redundancy']:.4f} | {r['diversity']:.4f} |")
        if has_f1:
            row += f" {r['f1']:.4f} |"
        out.append(row)

    if not has_f1:
        out.append("\n\n⚠️ **本次评测没有 F1**（跑了 --skip_generation）。")
        out.append("下面的结论只覆盖检索侧的 Recall/冗余度权衡，")
        out.append("**不能推断对下游 QA 质量的影响**。")
        out.append("要谈 F1，需去掉 --skip_generation 重跑。\n")

    # ── 饱和检查：必须在任何 γ 结论之前 ──────────────────────────────
    ov = analysis.get('overlap')
    out.append("\n\n## 0. 饱和检查（先看这节）\n")
    if ov is None:
        out.append("\n⚠️ **未做** —— 缺 `selected_passages`（评测需加 `--dump_passages`）。")
        out.append("\n下面所有 γ 结论都**未经饱和检查**：如果 γ 实际没生效，")
        out.append("「多样性影响小」这个读数是假的。\n")
    else:
        out.append(f"\n三个 γ 选出段落的平均 Jaccard 重合度（{ov['n_questions']} 题）：\n")
        for k, v in ov['pairs'].items():
            out.append(f"\n- {k}: {v:.3f}")
        out.append(f"\n\n**平均: {ov['mean_jaccard']:.3f}**\n")
        if ov['saturated']:
            out.append("\n### ❌ γ 未生效，本次 sweep 不可解读\n")
            out.append(f"\n重合度 {ov['mean_jaccard']:.3f} > 0.95 —— 三个 γ 几乎选出同一批段落。")
            out.append("\n\n原因在 QUBO 的非对角项 `Q_ij = γ·b_ij + 2·lam`：")
            out.append("若 `b`（段落间余弦相似度）的量级远小于 `2·lam`，")
            out.append("放大 γ 只是等比放大一个被压死的项，argmin 不变。")
            out.append("\n\n**下面的 Recall/冗余度对比反映的不是多样性效应，**")
            out.append("**而是随机波动。不要据此判断 idea 1 的去留。**")
            out.append("\n\n下一步：降低 `lam`，或对 `b` 做归一化使其与 `a` 量级相当，")
            out.append("重跑本 sweep 直到重合度明显低于 0.95。\n")
        else:
            out.append(f"\n✅ γ 生效（重合度 {ov['mean_jaccard']:.3f} < 0.95），下面的对比可解读。\n")

    out.append("\n\n## 关键发现\n")

    dr5 = (r5['recall'] - r0['recall']) * 100
    dr10 = (r10['recall'] - r0['recall']) * 100

    out.append("\n### 1. Recall 随 γ 的变化\n")
    out.append(f"- γ: 0.0 → 0.5: {dr5:+.1f} 个百分点")
    out.append(f"\n- γ: 0.0 → 1.0: {dr10:+.1f} 个百分点\n")

    if dr10 < -3:
        out.append("\n**读数**: 多样性显著损害 Recall（>3 个百分点）\n")
    elif dr10 < 0:
        out.append("\n**读数**: 多样性略微损害 Recall\n")
    elif dr10 > 3:
        out.append("\n**读数**: 多样性显著提升 Recall（罕见）\n")
    else:
        out.append("\n**读数**: 多样性对 Recall 影响很小（<3 个百分点）\n")

    out.append("\n### 2. 冗余度随 γ 的变化\n")
    dred = (r10['redundancy'] - r0['redundancy']) * 100
    out.append(f"- γ: 0.0 → 1.0: {dred:+.1f} 个百分点\n")
    if dred < 0:
        out.append(f"\nγ 确实在降冗余（降了 {abs(dred):.1f} 个百分点），说明多样性项生效。\n")
    else:
        out.append(f"\n⚠️ γ 升高但冗余没降 —— 先查 selector 是否真的用上了 γ。\n")

    out.append("\n### 3. 最优配置\n")
    br, bp = analysis['best_recall_config'], analysis['best_precision_config']
    out.append(f"- Recall 最高: **{br}** ({results[br]['recall']:.4f})")
    out.append(f"\n- Precision 最高: **{bp}** ({results[bp]['precision']:.4f})\n")

    out.append("\n## 场景判断\n")
    scenario_names = {
        'pure_quality': '纯质量最优',
        'balanced': '质量与多样性权衡',
        'pure_diversity': '纯多样性最优（罕见）',
        'no_effect': 'γ 无明显作用',
    }
    out.append(f"\n**场景**: `{analysis['scenario']}` — "
               f"{scenario_names.get(analysis['scenario'], '未知')}\n")

    out.append("\n## 下一步建议\n")
    out.append(f"\n{analysis['recommendation']}\n")

    out.append("\n---\n")
    out.append("\n## 这个诊断能证明什么、不能证明什么\n")
    out.append("\n**能**: γ 是否生效、Recall-冗余度权衡曲线上的三个点、"
               "Phase 2 两阶段的参数起点。\n")
    out.append("\n**不能**: 「质量与多样性冲突」。冲突的判据是 Pareto 前沿上"
               "无法同时最优，三点 sweep 给不出前沿。"
               "论文里若要主张冲突，需要更密的 γ 网格 + 双指标同时呈现。\n")
    out.append("\n**脚本**: `scripts/diagnosis/gamma_sweep_diagnosis.py`\n")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(out))


def main() -> int:
    parser = argparse.ArgumentParser(description='γ sweep 诊断')
    parser.add_argument('--results_dir', type=Path, required=True,
                        help='包含 gamma_*/result.json 的目录')
    parser.add_argument('--output', type=Path, required=True,
                        help='输出报告路径')
    args = parser.parse_args()

    print("=== γ Sweep 诊断 ===")
    print(f"输入目录: {args.results_dir}")

    results = load_results(args.results_dir)

    if len(results) < len(GAMMA_CONFIGS):
        print(f"❌ 只找到 {len(results)}/{len(GAMMA_CONFIGS)} 个结果",
              file=sys.stderr)
        print(f"   期望: {args.results_dir}/{{{','.join(GAMMA_CONFIGS)}}}/result.json",
              file=sys.stderr)
        return 1

    print(f"✅ 加载 {len(results)} 个 γ 配置")

    overlap = selection_overlap(args.results_dir)
    if overlap is None:
        print("⚠️ 饱和检查跳过（缺 selected_passages，需 --dump_passages）")
    else:
        print(f"选择重合度: {overlap['mean_jaccard']:.3f} "
              f"({'饱和' if overlap['saturated'] else '正常'})")

    analysis = analyze_results(results, overlap)
    generate_report(results, analysis, args.output)

    print(f"✅ 报告生成: {args.output}")
    print(f"\n场景: {analysis['scenario']}")
    print(f"建议:\n{analysis['recommendation']}")
    if not analysis['has_f1']:
        print("\n⚠️ 无 F1 数据（--skip_generation），结论仅覆盖检索侧")

    # γ 饱和时以非零退出，让 driver 能把它标成"需处理"而不是"已完成"
    return 2 if analysis['scenario'] == 'gamma_saturated' else 0


if __name__ == '__main__':
    sys.exit(main())
