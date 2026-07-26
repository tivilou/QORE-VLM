#!/usr/bin/env python3
"""
答案多样性诊断脚本

目的: 验证"文本多样但答案重复"的假设
假设: 当前方法只考虑文本层面多样性，导致选出的段落虽然文本不同，但答案重复

输出:
1. 文本多样性 vs 答案多样性对比
2. 答案重复率统计
3. 具体案例分析
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnosis_io import DiagnosisInputError, load_samples, passage_texts


def compute_text_similarity(passages: List[str]) -> float:
    """
    计算段落间的文本相似度（简化版，基于词重叠）

    实际实现应该用更好的方法（如 sentence-transformers）
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(passages) < 2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(passages)
        sim_matrix = cosine_similarity(vectors)

        # 计算平均相似度（排除对角线）
        n = len(passages)
        pairwise_sims = []
        for i in range(n):
            for j in range(i+1, n):
                pairwise_sims.append(sim_matrix[i, j])

        return np.mean(pairwise_sims) if pairwise_sims else 0.0
    except:
        # Fallback: 词重叠
        return compute_word_overlap(passages)


def compute_word_overlap(passages: List[str]) -> float:
    """基于词重叠的简单相似度"""
    if len(passages) < 2:
        return 0.0

    # 分词
    words_list = [set(p.lower().split()) for p in passages]

    # 计算两两重叠
    overlaps = []
    n = len(passages)
    for i in range(n):
        for j in range(i+1, n):
            overlap = len(words_list[i] & words_list[j])
            union = len(words_list[i] | words_list[j])
            if union > 0:
                overlaps.append(overlap / union)

    return np.mean(overlaps) if overlaps else 0.0


def compute_answer_stats(answers_per_passage: List[List[str]]) -> Dict:
    """答案层面的冗余统计。

    输入是每个已选段落命中的 gold answer 字符串列表（由 eval 的
    `--dump_passages` 写出的 `matched_answers`，与评测的 gold 判定同源）。

    ⚠️ NQ-open 的 gold_answers 是**同一个答案的别名集合**（"Obama" /
    "Barack Obama"），不是多个不同答案。所以这里的"distinct"是别名级，
    不代表语义上的多个答案。真正可解释的量是:

    - answer_bearing_ratio: 含答案的段落占比 → 证据冗余程度
    - duplicate_ratio: 超出第一个含答案段落之外的那些占比 → 被"重复证据"
      占掉的槽位比例。这才是答案多样性约束想回收的东西。
    """
    n_passages = len(answers_per_passage)
    if n_passages == 0:
        return {}

    bearing = [a for a in answers_per_passage if a]
    n_bearing = len(bearing)

    distinct = set()
    for spans in bearing:
        distinct.update(s.lower().strip() for s in spans if s)

    # 第一个含答案段落是必要的，其余是重复证据
    duplicates = max(0, n_bearing - 1)

    return {
        'num_passages': n_passages,
        'n_answer_bearing': n_bearing,
        'answer_bearing_ratio': n_bearing / n_passages,
        'n_distinct_answer_strings': len(distinct),
        'duplicate_ratio': duplicates / n_passages,
    }


def analyze_single_query(sample: Dict) -> Dict:
    """分析单个查询的文本多样性 vs 答案冗余"""
    passages = sample.get('selected_passages') or []
    if not passages:
        return None

    text_similarity = compute_text_similarity(passage_texts(sample))
    text_diversity = 1 - text_similarity

    # eval 已经记录了每个段落命中哪些 gold answer，这里不再自己做匹配，
    # 避免与评测的 gold 判定产生分歧。
    answers_per_passage = [p.get('matched_answers') or [] for p in passages]
    stats = compute_answer_stats(answers_per_passage)
    if not stats:
        return None

    stats.update({
        'text_diversity': text_diversity,
        'text_similarity': text_similarity,
        'f1': sample.get('f1'),
    })
    return stats


def generate_report(stats: List[Dict], output_file: Path):
    """生成诊断报告"""
    # 过滤有效数据
    valid_stats = [s for s in stats if s is not None]

    if not valid_stats:
        print("⚠️ 没有有效数据")
        return False

    # 计算统计指标
    text_divs = [s['text_diversity'] for s in valid_stats]
    bearing_ratios = [s['answer_bearing_ratio'] for s in valid_stats]
    dup_ratios = [s['duplicate_ratio'] for s in valid_stats]
    distinct_counts = [s['n_distinct_answer_strings'] for s in valid_stats]

    avg_text_div = np.mean(text_divs)
    avg_bearing = np.mean(bearing_ratios)
    avg_dup = np.mean(dup_ratios)
    avg_distinct = np.mean(distinct_counts)

    # 问题案例：文本上看起来很多样，但槽位被重复证据占掉
    problem_cases = []
    for i, s in enumerate(valid_stats):
        if s['text_diversity'] > 0.6 and s['duplicate_ratio'] >= 0.4:
            problem_cases.append((i, s))

    # 生成报告
    report = []
    report.append("# 答案多样性诊断报告\n")
    report.append(f"**分析样本数**: {len(valid_stats)}\n")
    report.append("\n---\n")

    report.append("\n## 读数须知\n")
    report.append("\nNQ-open 的 `gold_answers` 是**同一答案的别名集合**，不是多个不同答案。")
    report.append("所以「答案多样性」在本数据集上不可直接测量；本报告改测**证据冗余**：")
    report.append("\n\n- `answer_bearing_ratio`: 含答案段落占比")
    report.append("\n- `duplicate_ratio`: 除第一个含答案段落外的占比 = 被重复证据占掉的槽位")
    report.append("\n\n答案多样性约束想回收的正是 `duplicate_ratio` 这部分槽位。\n")

    report.append("\n## 关键发现\n")
    report.append(f"\n### 整体统计\n")
    report.append(f"- **文本多样性**: {avg_text_div:.3f}")
    report.append(f"\n- **含答案段落占比**: {avg_bearing:.3f}")
    report.append(f"\n- **重复证据占比**: {avg_dup:.3f}")
    report.append(f"\n- **平均别名命中数**: {avg_distinct:.2f}\n")

    num_passages = [s['num_passages'] for s in valid_stats]

    # ── 先算完两条证据，再给唯一一次结论 ──────────────────────────────
    # 证据 A（可回收空间有多大）: 重复证据占比
    # 证据 B（回收了能否提升 F1）: 占比与 F1 的相关性 ← 更直接，冲突时以它为准
    has_room = avg_text_div > 0.5 and avg_dup >= 0.3
    some_room = avg_dup >= 0.15

    corr_note = ''
    r = None
    f1_pairs = [(s['duplicate_ratio'], s['f1']) for s in valid_stats
                if s.get('f1') is not None]
    if len(f1_pairs) >= 10:
        xs = np.array([p[0] for p in f1_pairs])
        ys = np.array([p[1] for p in f1_pairs])
        if xs.std() > 1e-9 and ys.std() > 1e-9:
            r = float(np.corrcoef(xs, ys)[0, 1])

    if r is None:
        # 没有 F1 就只能靠占比，此时结论必须标注为未经 F1 验证
        hypothesis_holds = has_room or some_room
    elif r < -0.1:
        hypothesis_holds = True
        corr_note = f"重复证据与 F1 负相关 (r={r:+.3f}) — 支持回收槽位"
    elif r > 0.1:
        hypothesis_holds = False
        corr_note = f"重复证据与 F1 正相关 (r={r:+.3f}) — 与假设相反"
    else:
        hypothesis_holds = False
        corr_note = f"重复证据与 F1 几乎无关 (r={r:+.3f})"

    report.append(f"\n### 证据 A：可回收空间\n")
    report.append(f"\n**文本多样性 {avg_text_div:.3f}，重复证据占比 {avg_dup:.3f}**"
                  f"（平均 {avg_dup * float(np.mean(num_passages)):.1f}/"
                  f"{np.mean(num_passages):.0f} 个槽位）\n")
    if has_room:
        report.append(f"\n段落文本层面确实多样，且有 {avg_dup*100:.1f}% 的槽位被重复证据占用。\n")
    elif some_room:
        report.append(f"\n存在一定的证据冗余，但不算严重。\n")
    else:
        report.append(f"\n重复证据占比很低，可回收的槽位不多。\n")

    report.append(f"\n### 证据 B：回收槽位能否提升 F1\n")
    if len(f1_pairs) < len(valid_stats):
        report.append(f"\n⚠️ 仅 {len(f1_pairs)}/{len(valid_stats)} 个样本有 F1"
                      f"（评测加了 --skip_generation？）。\n")
    if r is None:
        report.append(f"\n**无法判断** —— 有 F1 的样本不足 10 条或方差过小。")
        report.append(f"\n下面的结论只基于证据 A，**未经 F1 验证**，"
                      f"不能作为实施答案多样性约束的依据。\n")
    else:
        report.append(f"\n**Pearson r(重复证据占比, F1) = {r:+.3f}**（n={len(f1_pairs)}）\n")
        if r < -0.1:
            report.append(f"\n负相关：槽位被重复证据占得越多，F1 越低。"
                          f"**这是答案多样性约束最直接的支撑证据。**\n")
        elif r > 0.1:
            report.append(f"\n⚠️ 正相关：重复证据多的样本 F1 反而更高。"
                          f"这与假设方向相反 —— 可能多份证据本身有助于生成。\n")
        else:
            report.append(f"\n几乎无关：回收槽位对 F1 的影响不可预期。\n")

    # ── 唯一一次结论 ──────────────────────────────────────────────────
    report.append(f"\n### 结论\n")
    report.append(f"\n**假设**: 段落文本多样，但答案层面重复（槽位被重复证据占用）\n")
    if hypothesis_holds and r is not None:
        report.append(f"\n**✅ 假设成立** —— 有 {avg_dup*100:.1f}% 的槽位可回收，"
                      f"且回收方向与 F1 一致 (r={r:+.3f})。\n")
    elif hypothesis_holds:
        report.append(f"\n**⚠️ 假设待验证** —— 有 {avg_dup*100:.1f}% 的槽位可回收，"
                      f"但缺 F1 数据，无法确认回收能否提升质量。\n")
    elif r is not None and (has_room or some_room):
        report.append(f"\n**❌ 假设不成立** —— 占比 {avg_dup:.3f} 看似有可回收空间，"
                      f"但它与 F1 的相关性只有 {r:+.3f}。"
                      f"证据 B 比证据 A 更直接，以它为准。\n")
    else:
        report.append(f"\n**❌ 假设不成立** —— 重复证据占比 {avg_dup:.3f} 本身就很低。\n")

    # 问题案例：文本上很分散，却把多个槽位花在重复证据上
    if problem_cases:
        report.append(f"\n### 典型问题案例\n")
        report.append(f"\n{len(problem_cases)} 个样本文本多样 >0.6 且重复证据占比 >0.4：\n")

        for idx, (case_id, case_stat) in enumerate(problem_cases[:5], 1):
            report.append(f"\n**案例 {idx}** (#{case_id}):")
            report.append(f"\n- 文本多样性: {case_stat['text_diversity']:.3f}")
            report.append(f"\n- 含答案段落: {case_stat['n_answer_bearing']}/{case_stat['num_passages']}")
            report.append(f"\n- 重复证据占比: {case_stat['duplicate_ratio']:.3f}")
            if case_stat.get('f1') is not None:
                report.append(f"\n- F1: {case_stat['f1']:.3f}")
            report.append("\n")

    # 分布分析
    def _dist(vals, name):
        report.append(f"\n**{name}分布**:")
        for label, lo, hi in [("低 (<0.4)", -1e9, 0.4),
                              ("中 (0.4-0.7)", 0.4, 0.7),
                              ("高 (≥0.7)", 0.7, 1e9)]:
            n = sum(1 for d in vals if lo <= d < hi)
            report.append(f"\n- {label}: {n} ({n/len(vals)*100:.1f}%)")
        report.append("\n")

    report.append(f"\n### 分布分析\n")
    _dist(text_divs, "文本多样性")
    _dist(dup_ratios, "重复证据占比")

    # 建议
    report.append(f"\n## 改进建议\n")

    if hypothesis_holds:
        report.append(f"\n基于诊断结果，建议：\n")
        report.append(f"\n1. ✅ **实施答案多样性约束**（高优先级）")
        report.append(f"\n   - 在 QUBO 目标中惩罚「同时选中多个含相同答案的段落」")
        report.append(f"\n   - 可直接复用 answer scorer 的证据信号，无需新模型\n")
        report.append(f"\n2. ⚠️ **先在 200 题上验证**")
        report.append(f"\n   - 重复证据占比 {avg_dup:.3f} 说明平均有 "
                      f"{avg_dup * float(np.mean(num_passages)):.1f} 个槽位可回收，")
        report.append(f"\n     但回收的槽位是否真能提升 F1 需要实测\n")
    elif r is not None and (has_room or some_room):
        report.append(f"\n❌ **不建议现在实施答案多样性约束**。\n")
        report.append(f"\n槽位确实有 {avg_dup*100:.1f}% 可回收，但回收方向与 F1 无关"
                      f"（r={r:+.3f}）—— 换掉这些槽位大概率不会提升答案质量。\n")
        report.append(f"\n若仍想推进，先查清为什么「选到证据」不等于「答得对」："
                      f"瓶颈可能在生成阶段而非选择阶段。这与全量评测的观察一致"
                      f"（Recall +13.4% 只换来 F1 +3.6%）。\n")
    else:
        report.append(f"\n重复证据占比 {avg_dup:.3f} 本身很低，可回收空间有限，"
                      f"优先级应低于 Phase 2（两阶段 QUBO）。\n")

    report.append(f"\n---\n")
    report.append("\n**注意**: NQ-open 的 gold_answers 是同一答案的别名集合，"
                  "因此「distinct 答案数」是别名级而非语义级。"
                  "结论应以「重复证据占比」为准。\n")
    report.append(f"\n**脚本**: `scripts/diagnosis/answer_diversity_diagnosis.py`\n")

    # 写入文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(''.join(report))

    print(f"✅ 报告生成: {output_file}")

    # 打印摘要
    print(f"\n=== 摘要 ===")
    print(f"样本数: {len(valid_stats)}")
    print(f"文本多样性: {avg_text_div:.3f}")
    print(f"含答案段落占比: {avg_bearing:.3f}")
    print(f"重复证据占比: {avg_dup:.3f}")
    if corr_note:
        print(corr_note)

    if hypothesis_holds and r is not None:
        print(f"\n✅ 假设成立：建议实施答案多样性约束")
    elif hypothesis_holds:
        print(f"\n⚠️ 假设待验证：有可回收槽位，但缺 F1 无法确认收益")
        print(f"   去掉 --skip_generation 重跑才能定论")
    elif r is not None and avg_dup >= 0.15:
        print(f"\n❌ 假设不成立：占比 {avg_dup:.3f} 有空间，但与 F1 相关性仅 {r:+.3f}")
    else:
        print(f"\n❌ 假设不成立：重复证据占比 {avg_dup:.3f} 本身很低")

    return True


def main():
    parser = argparse.ArgumentParser(description='答案多样性诊断')
    parser.add_argument('--results', type=Path, required=True,
                       help='实验结果文件 (JSON)')
    parser.add_argument('--output', type=Path, required=True,
                       help='输出报告路径')

    args = parser.parse_args()

    print("=== 答案多样性诊断 ===")
    print(f"输入: {args.results}")

    # 缺字段就报错退出，不生成"假设不成立"的空报告
    try:
        samples = load_samples(args.results, require=('selected_passages',))
    except DiagnosisInputError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    print(f"加载 {len(samples)} 个样本")

    stats = []
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(samples)}")
        stats.append(analyze_single_query(sample))

    return 0 if generate_report(stats, args.output) else 1


if __name__ == '__main__':
    sys.exit(main())
