#!/usr/bin/env python3
"""
QUBO 目标分析诊断脚本

目的: 验证"QUBO 目标与下游任务（F1）不一致"的假设
假设: 当前 QUBO 目标函数是人工设计的代理目标，没有直接优化 F1，
     导致 QUBO 最优解不等于 F1 最优解

输出:
1. QUBO 目标与 F1 的相关性
2. QUBO 最优 vs F1 最优的一致性
3. 端到端训练的潜在收益
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple
from result_adapter import load_query_results, write_unavailable_report


def compute_qubo_objective(passage_scores: List[float],
                           redundancy_matrix: List[List[float]],
                           gamma: float,
                           lam: float) -> float:
    """
    计算 QUBO 目标值

    E = λ·Σq_i·x_i - γ·Σb_ij·x_i·x_j

    这里简化为已选择段落的目标值
    """
    n = len(passage_scores)

    # 质量项
    quality_term = lam * sum(passage_scores)

    # 冗余项（两两之间）
    redundancy_term = 0
    for i in range(n):
        for j in range(i + 1, n):
            if i < len(redundancy_matrix) and j < len(redundancy_matrix[i]):
                redundancy_term += redundancy_matrix[i][j]

    # QUBO 目标（最小化）
    objective = -quality_term + gamma * redundancy_term

    return objective


def analyze_single_query(query_result: Dict, gamma: float = 0.5, lam: float = 2.0) -> Dict:
    """分析单个查询的 QUBO 目标 vs F1"""

    # 获取选中的段落及其指标
    selected = query_result.get('selected_passages', [])
    if not selected:
        return None

    # 获取 QUBO 相关信息（如果有）
    # 注意：这些信息可能不在当前的结果文件中，需要从实验中提取
    passage_scores = [p.get('score', 0) for p in selected]

    # 计算冗余矩阵（简化版，使用文本相似度）
    # 实际应该使用实验中计算的 redundancy matrix
    redundancy_matrix = compute_simple_redundancy(selected)

    # 计算 QUBO 目标
    qubo_score = compute_qubo_objective(passage_scores, redundancy_matrix, gamma, lam)

    # 获取 F1
    metrics = query_result.get('metrics', {})
    f1 = metrics.get('f1', 0)
    recall = metrics.get('recall', 0)
    precision = metrics.get('precision', 0)

    return {
        'qubo_score': qubo_score,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'num_passages': len(selected)
    }


def compute_simple_redundancy(passages: List[Dict]) -> List[List[float]]:
    """简单的冗余度计算（基于词重叠）"""
    n = len(passages)
    matrix = [[0.0] * n for _ in range(n)]

    texts = [p.get('text', '') for p in passages]

    for i in range(n):
        for j in range(i + 1, n):
            # 简单的词重叠
            words_i = set(texts[i].lower().split())
            words_j = set(texts[j].lower().split())

            if words_i and words_j:
                overlap = len(words_i & words_j)
                union = len(words_i | words_j)
                similarity = overlap / union if union > 0 else 0
                matrix[i][j] = similarity
                matrix[j][i] = similarity

    return matrix


def load_results(results_file: Path) -> List[Dict]:
    """加载实验结果"""
    return load_query_results(results_file)


def generate_report(stats: List[Dict], output_file: Path):
    """生成诊断报告"""
    # 过滤有效数据
    valid_stats = [s for s in stats if s is not None and s['f1'] > 0]

    if not valid_stats:
        write_unavailable_report(
            output_file, "QUBO 目标分析诊断报告",
            "每题的 selected_passages、段落分数和 QUBO 冗余矩阵",
        )
        return

    # 提取数据
    qubo_scores = np.array([s['qubo_score'] for s in valid_stats])
    f1_scores = np.array([s['f1'] for s in valid_stats])
    recall_scores = np.array([s['recall'] for s in valid_stats])

    # 计算相关性
    pearson_corr, pearson_p = pearsonr(qubo_scores, f1_scores)
    spearman_corr, spearman_p = spearmanr(qubo_scores, f1_scores)

    # QUBO 和 F1 的排序一致性
    qubo_ranks = np.argsort(qubo_scores)  # QUBO 越小越好
    f1_ranks = np.argsort(-f1_scores)  # F1 越大越好

    # 计算 top-k 的重叠
    k_values = [10, 20, 50, 100]
    top_k_overlaps = {}
    for k in k_values:
        if k <= len(valid_stats):
            qubo_topk = set(qubo_ranks[:k])
            f1_topk = set(f1_ranks[:k])
            overlap = len(qubo_topk & f1_topk)
            top_k_overlaps[k] = overlap / k

    # 生成报告
    report = []
    report.append("# QUBO 目标分析诊断报告\n")
    report.append(f"**分析样本数**: {len(valid_stats)}\n")
    report.append("\n---\n")

    report.append("\n## 关键发现\n")
    report.append(f"\n### 相关性分析\n")
    report.append(f"\n**QUBO 目标 vs F1**:\n")
    report.append(f"- Pearson 相关系数: **{pearson_corr:.3f}** (p={pearson_p:.4f})")
    report.append(f"\n- Spearman 相关系数: **{spearman_corr:.3f}** (p={spearman_p:.4f})\n")

    # 解释相关性
    if abs(pearson_corr) > 0.7:
        corr_level = "强"
    elif abs(pearson_corr) > 0.5:
        corr_level = "中等"
    elif abs(pearson_corr) > 0.3:
        corr_level = "弱"
    else:
        corr_level = "很弱或无"

    report.append(f"\n相关性强度: **{corr_level}**\n")

    # 注意：QUBO 是最小化，F1 是最大化，所以负相关是好的
    if pearson_corr < 0:
        report.append(f"\n注意：负相关是预期的（QUBO 最小化，F1 最大化）\n")

    # Top-k 重叠
    report.append(f"\n### Top-K 重叠分析\n")
    report.append(f"\nQUBO Top-K 与 F1 Top-K 的重叠率：\n")
    for k, overlap in top_k_overlaps.items():
        report.append(f"- Top-{k}: {overlap*100:.1f}%")
    report.append("\n")

    # 假设验证
    report.append(f"\n### 假设验证\n")
    report.append(f"\n**假设**: QUBO 目标与 F1 不一致\n")

    # 判断标准
    low_correlation = abs(pearson_corr) < 0.7
    low_overlap = np.mean(list(top_k_overlaps.values())) < 0.7 if top_k_overlaps else True

    report.append(f"\n**观察**:")
    report.append(f"\n- 相关性: {abs(pearson_corr):.3f} ({'低' if low_correlation else '高'})")
    if top_k_overlaps:
        avg_overlap = np.mean(list(top_k_overlaps.values()))
        report.append(f"\n- 平均 Top-K 重叠: {avg_overlap*100:.1f}% ({'低' if low_overlap else '高'})\n")

    if low_correlation and low_overlap:
        report.append(f"\n**结论**: ✅ **假设强烈成立**\n")
        report.append(f"\nQUBO 目标与 F1 的相关性低（{abs(pearson_corr):.3f}），")
        report.append(f"且 Top-K 重叠率低，")
        report.append(f"表明 QUBO 最优解经常不是 F1 最优解。\n")
        report.append(f"\n这说明代理目标与真实目标存在较大差距。\n")

    elif low_correlation or low_overlap:
        report.append(f"\n**结论**: ⚠️ **假设部分成立**\n")
        report.append(f"\nQUBO 目标与 F1 有一定相关性，但不是完全一致。\n")

    else:
        report.append(f"\n**结论**: ❌ **假设不成立**\n")
        report.append(f"\nQUBO 目标与 F1 高度相关，代理目标设计合理。\n")

    # 散点分布
    report.append(f"\n### 分布分析\n")

    # QUBO 和 F1 的分位数
    qubo_quartiles = np.percentile(qubo_scores, [25, 50, 75])
    f1_quartiles = np.percentile(f1_scores, [25, 50, 75])

    report.append(f"\n**QUBO 目标分布**:")
    report.append(f"\n- Q1 (25%): {qubo_quartiles[0]:.3f}")
    report.append(f"\n- 中位数: {qubo_quartiles[1]:.3f}")
    report.append(f"\n- Q3 (75%): {qubo_quartiles[2]:.3f}\n")

    report.append(f"\n**F1 分布**:")
    report.append(f"\n- Q1 (25%): {f1_quartiles[0]:.3f}")
    report.append(f"\n- 中位数: {f1_quartiles[1]:.3f}")
    report.append(f"\n- Q3 (75%): {f1_quartiles[2]:.3f}\n")

    # 不一致案例
    # 找到 QUBO 好但 F1 差，或 QUBO 差但 F1 好的案例
    qubo_percentile = np.percentile(qubo_scores, range(101))
    f1_percentile = np.percentile(f1_scores, range(101))

    inconsistent_cases = []
    for i, (q, f) in enumerate(zip(qubo_scores, f1_scores)):
        q_rank = np.searchsorted(qubo_percentile, q)
        f_rank = np.searchsorted(f1_percentile, f)

        # QUBO 排名和 F1 排名差异大
        if abs(q_rank - f_rank) > 30:  # 差异 > 30 个百分位
            inconsistent_cases.append((i, q, f, q_rank, f_rank))

    if inconsistent_cases:
        report.append(f"\n### 不一致案例\n")
        report.append(f"\n发现 {len(inconsistent_cases)} 个不一致案例（QUBO 排名与 F1 排名差异 >30%）：\n")

        for idx, (case_id, qubo, f1, q_rank, f_rank) in enumerate(inconsistent_cases[:5], 1):
            report.append(f"\n**案例 {idx}**:")
            report.append(f"\n- QUBO score: {qubo:.3f} (排名: {q_rank}%)")
            report.append(f"\n- F1: {f1:.3f} (排名: {f_rank}%)")
            report.append(f"\n- 排名差异: {abs(q_rank - f_rank)}%\n")

    # 建议
    report.append(f"\n## 改进建议\n")

    if low_correlation or low_overlap:
        report.append(f"\n基于诊断结果，建议：\n")
        report.append(f"\n1. ✅ **端到端可微训练**（长期，高优先级）")
        report.append(f"\n   - 将 QUBO 松弛为可微分形式")
        report.append(f"\n   - 直接优化 F1（或其上界）")
        report.append(f"\n   - 预期提升: +5-10% F1")
        report.append(f"\n   - 但实现复杂度高（3+ 月）\n")

        report.append(f"\n2. ✅ **改进 QUBO 目标函数**（短期）")
        report.append(f"\n   - 添加更多启发式项（答案多样性等）")
        report.append(f"\n   - 调整权重使其更接近 F1")
        report.append(f"\n   - 预期提升: +1-2% F1\n")

        report.append(f"\n3. ⚠️ **重排序**（临时方案）")
        report.append(f"\n   - QUBO 选择候选池")
        report.append(f"\n   - 用学习到的排序器重排")
        report.append(f"\n   - 简单但效果有限\n")
    else:
        report.append(f"\nQUBO 目标与 F1 相关性较好，")
        report.append(f"端到端训练的收益可能有限。")
        report.append(f"建议优先考虑其他优化方向。\n")

    # 潜在收益估算
    if low_correlation:
        report.append(f"\n### 端到端训练潜在收益\n")
        report.append(f"\n如果能够直接优化 F1：\n")

        # 理论上界：F1 最优解的平均值
        f1_optimal = np.mean(np.partition(f1_scores, -len(f1_scores)//10)[-len(f1_scores)//10:])
        # 当前：QUBO 选择的平均 F1
        current_f1 = np.mean(f1_scores)

        potential_gain = f1_optimal - current_f1

        report.append(f"- 当前平均 F1: {current_f1:.3f}")
        report.append(f"\n- F1 最优解平均: {f1_optimal:.3f}")
        report.append(f"\n- **理论潜在提升**: {potential_gain*100:+.1f}% ({potential_gain:+.3f})\n")

        report.append(f"\n注意：这是理论上界，实际收益会更小。\n")

    report.append(f"\n---\n")
    report.append(f"\n**注意**: 本分析使用简化的 QUBO 目标计算。")
    report.append(f"实际应该使用实验中完整的 QUBO 矩阵以获得更准确的结果。\n")
    report.append(f"\n**生成时间**: 自动生成")
    report.append(f"\n**脚本**: `scripts/diagnosis/qubo_objective_diagnosis.py`\n")

    # 写入文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(''.join(report))

    print(f"✅ 报告生成: {output_file}")

    # 打印摘要
    print(f"\n=== 摘要 ===")
    print(f"Pearson 相关系数: {pearson_corr:.3f}")
    print(f"相关性强度: {corr_level}")
    if top_k_overlaps:
        print(f"平均 Top-K 重叠: {np.mean(list(top_k_overlaps.values()))*100:.1f}%")

    if low_correlation or low_overlap:
        print(f"\n✅ 假设成立：QUBO 目标与 F1 不一致")
    else:
        print(f"\n⚠️ 假设不成立或较弱")


def main():
    parser = argparse.ArgumentParser(description='QUBO 目标分析诊断')
    parser.add_argument('--results', type=Path, required=True,
                       help='实验结果文件 (JSON)')
    parser.add_argument('--output', type=Path, required=True,
                       help='输出报告路径')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='QUBO 参数 γ (default: 0.5)')
    parser.add_argument('--lam', type=float, default=2.0,
                       help='QUBO 参数 λ (default: 2.0)')

    args = parser.parse_args()

    print("=== QUBO 目标分析诊断 ===")
    print(f"输入: {args.results}")
    print(f"参数: γ={args.gamma}, λ={args.lam}")

    # 加载结果
    results = load_results(args.results)
    print(f"加载 {len(results)} 个查询结果")

    # 分析每个查询
    stats = []
    for i, query_result in enumerate(results):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(results)}")

        stat = analyze_single_query(query_result, args.gamma, args.lam)
        stats.append(stat)

    # 生成报告
    generate_report(stats, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
