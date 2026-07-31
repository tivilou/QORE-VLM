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

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
from result_adapter import load_query_results, write_unavailable_report


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


def extract_answer_spans(passages: List[str], query: str, answer_spans: List[List[str]]) -> Set[str]:
    """
    提取段落中的答案

    这里使用预先提取的 answer_spans（如果有）
    否则使用简单的 n-gram 匹配
    """
    if answer_spans:
        # 使用预提取的答案
        all_answers = set()
        for spans in answer_spans:
            all_answers.update([s.lower().strip() for s in spans])
        return all_answers

    # Fallback: 简单的 n-gram 提取（不准确，仅作示例）
    # 实际应该用 QA 模型或预先提取的答案
    return set()


def compute_answer_diversity(answer_spans_list: List[List[str]]) -> Tuple[float, float]:
    """
    计算答案多样性

    Returns:
        (unique_ratio, overlap_ratio)
    """
    if not answer_spans_list:
        return 0.0, 1.0

    # 收集所有答案（小写，去空格）
    all_answers = []
    for spans in answer_spans_list:
        normalized = [s.lower().strip() for s in spans]
        all_answers.extend(normalized)

    if not all_answers:
        return 0.0, 1.0

    # 计算唯一答案比例
    unique_answers = len(set(all_answers))
    total_answers = len(all_answers)
    unique_ratio = unique_answers / total_answers

    # 计算答案重叠率
    answer_counts = Counter(all_answers)
    duplicates = sum(c - 1 for c in answer_counts.values() if c > 1)
    overlap_ratio = duplicates / total_answers if total_answers > 0 else 0.0

    return unique_ratio, overlap_ratio


def analyze_single_query(query_result: Dict) -> Dict:
    """分析单个查询的答案多样性"""
    passages = query_result.get('selected_passages', [])

    if not passages:
        return None

    # 提取文本
    passage_texts = [p.get('text', '') for p in passages]

    # 1. 计算文本多样性
    text_similarity = compute_text_similarity(passage_texts)
    text_diversity = 1 - text_similarity

    # 2. 提取答案
    answer_spans_list = []
    for p in passages:
        # 假设每个 passage 有 'answer_spans' 字段
        # 如果没有，需要用 QA 模型提取
        spans = p.get('answer_spans', [])
        answer_spans_list.append(spans)

    # 3. 计算答案多样性
    answer_unique_ratio, answer_overlap_ratio = compute_answer_diversity(answer_spans_list)

    return {
        'text_diversity': text_diversity,
        'text_similarity': text_similarity,
        'answer_unique_ratio': answer_unique_ratio,
        'answer_overlap_ratio': answer_overlap_ratio,
        'num_passages': len(passages)
    }


def load_results(results_file: Path) -> List[Dict]:
    """加载实验结果"""
    return load_query_results(results_file)


def generate_report(stats: List[Dict], output_file: Path):
    """生成诊断报告"""
    # 过滤有效数据
    valid_stats = [s for s in stats if s is not None]

    if not valid_stats:
        write_unavailable_report(
            output_file, "答案多样性诊断报告",
            "每题的 selected_passages 文本及 answer_spans",
        )
        return

    # 计算统计指标
    text_divs = [s['text_diversity'] for s in valid_stats]
    answer_unique = [s['answer_unique_ratio'] for s in valid_stats]
    answer_overlap = [s['answer_overlap_ratio'] for s in valid_stats]

    avg_text_div = np.mean(text_divs)
    avg_answer_unique = np.mean(answer_unique)
    avg_answer_overlap = np.mean(answer_overlap)

    # 找出问题案例（文本多样但答案重复）
    problem_cases = []
    for i, s in enumerate(valid_stats):
        if s['text_diversity'] > 0.6 and s['answer_unique_ratio'] < 0.5:
            problem_cases.append((i, s))

    # 生成报告
    report = []
    report.append("# 答案多样性诊断报告\n")
    report.append(f"**分析样本数**: {len(valid_stats)}\n")
    report.append("\n---\n")

    report.append("\n## 关键发现\n")
    report.append(f"\n### 整体统计\n")
    report.append(f"- **文本多样性**: {avg_text_div:.3f}")
    report.append(f"\n- **答案唯一比例**: {avg_answer_unique:.3f}")
    report.append(f"\n- **答案重叠率**: {avg_answer_overlap:.3f}\n")

    # 判断假设是否成立
    report.append(f"\n### 假设验证\n")
    report.append(f"\n**假设**: 文本多样但答案重复\n")

    ratio = avg_answer_unique / avg_text_div if avg_text_div > 0 else 0
    report.append(f"\n**答案多样性 / 文本多样性**: {ratio:.3f}\n")

    if ratio < 0.7 and avg_answer_overlap > 0.3:
        report.append(f"\n**结论**: ✅ **假设成立**\n")
        report.append(f"\n答案唯一比例（{avg_answer_unique:.3f}）显著低于文本多样性（{avg_text_div:.3f}），")
        report.append(f"且答案重叠率较高（{avg_answer_overlap:.3f}）。")
        report.append(f"\n这表明当前方法虽然选出了文本不同的段落，但答案层面存在大量重复。\n")
    elif ratio < 0.85:
        report.append(f"\n**结论**: ⚠️ **假设部分成立**\n")
        report.append(f"\n答案多样性（{avg_answer_unique:.3f}）略低于文本多样性（{avg_text_div:.3f}），")
        report.append(f"有一定的改进空间。\n")
    else:
        report.append(f"\n**结论**: ❌ **假设不成立**\n")
        report.append(f"\n答案多样性与文本多样性接近，答案重复不是主要问题。\n")

    # 问题案例
    if problem_cases:
        report.append(f"\n### 典型问题案例\n")
        report.append(f"\n发现 {len(problem_cases)} 个问题案例（文本多样 > 0.6 但答案唯一 < 0.5）：\n")

        for idx, (case_id, case_stat) in enumerate(problem_cases[:5], 1):
            report.append(f"\n**案例 {idx}**:")
            report.append(f"\n- 文本多样性: {case_stat['text_diversity']:.3f}")
            report.append(f"\n- 答案唯一比例: {case_stat['answer_unique_ratio']:.3f}")
            report.append(f"\n- 答案重叠率: {case_stat['answer_overlap_ratio']:.3f}\n")

    # 分布分析
    report.append(f"\n### 分布分析\n")
    report.append(f"\n**文本多样性分布**:")
    report.append(f"\n- 低 (<0.4): {sum(1 for d in text_divs if d < 0.4)} ({sum(1 for d in text_divs if d < 0.4)/len(text_divs)*100:.1f}%)")
    report.append(f"\n- 中 (0.4-0.7): {sum(1 for d in text_divs if 0.4 <= d < 0.7)} ({sum(1 for d in text_divs if 0.4 <= d < 0.7)/len(text_divs)*100:.1f}%)")
    report.append(f"\n- 高 (≥0.7): {sum(1 for d in text_divs if d >= 0.7)} ({sum(1 for d in text_divs if d >= 0.7)/len(text_divs)*100:.1f}%)\n")

    report.append(f"\n**答案唯一比例分布**:")
    report.append(f"\n- 低 (<0.4): {sum(1 for d in answer_unique if d < 0.4)} ({sum(1 for d in answer_unique if d < 0.4)/len(answer_unique)*100:.1f}%)")
    report.append(f"\n- 中 (0.4-0.7): {sum(1 for d in answer_unique if 0.4 <= d < 0.7)} ({sum(1 for d in answer_unique if 0.4 <= d < 0.7)/len(answer_unique)*100:.1f}%)")
    report.append(f"\n- 高 (≥0.7): {sum(1 for d in answer_unique if d >= 0.7)} ({sum(1 for d in answer_unique if d >= 0.7)/len(answer_unique)*100:.1f}%)\n")

    # 建议
    report.append(f"\n## 改进建议\n")

    if ratio < 0.7:
        report.append(f"\n基于诊断结果，建议：\n")
        report.append(f"\n1. ✅ **实施答案多样性约束**（高优先级）")
        report.append(f"\n   - 在 QUBO 目标函数中添加答案层面的多样性项")
        report.append(f"\n   - 预期提升：+1-2% F1\n")
        report.append(f"\n2. ✅ **答案去重机制**")
        report.append(f"\n   - 如果两个段落包含相同答案，惩罚其同时被选中\n")
    else:
        report.append(f"\n答案多样性问题不严重，可以考虑其他优化方向。\n")

    report.append(f"\n---\n")
    report.append(f"\n**生成时间**: 自动生成")
    report.append(f"\n**脚本**: `scripts/diagnosis/answer_diversity_diagnosis.py`\n")

    # 写入文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(''.join(report))

    print(f"✅ 报告生成: {output_file}")

    # 打印摘要
    print(f"\n=== 摘要 ===")
    print(f"文本多样性: {avg_text_div:.3f}")
    print(f"答案唯一比例: {avg_answer_unique:.3f}")
    print(f"答案重叠率: {avg_answer_overlap:.3f}")
    print(f"比例: {ratio:.3f}")

    if ratio < 0.7:
        print(f"\n✅ 假设成立：建议实施答案多样性约束")
    else:
        print(f"\n⚠️ 假设不成立或较弱")


def main():
    parser = argparse.ArgumentParser(description='答案多样性诊断')
    parser.add_argument('--results', type=Path, required=True,
                       help='实验结果文件 (JSON)')
    parser.add_argument('--output', type=Path, required=True,
                       help='输出报告路径')

    args = parser.parse_args()

    print("=== 答案多样性诊断 ===")
    print(f"输入: {args.results}")

    # 加载结果
    results = load_results(args.results)
    print(f"加载 {len(results)} 个查询结果")

    # 分析每个查询
    stats = []
    for i, query_result in enumerate(results):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(results)}")

        stat = analyze_single_query(query_result)
        stats.append(stat)

    # 生成报告
    generate_report(stats, args.output)


if __name__ == '__main__':
    main()
