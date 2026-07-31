#!/usr/bin/env python3
"""
上下文依赖分析诊断脚本

目的: 验证"独立选择破坏段落间依赖关系"的假设
假设: 当前方法独立选择段落，没有考虑段落间的依赖（共指、实体连贯等），
     导致选出的段落虽然各自相关，但组合起来信息不完整

输出:
1. 共指链完整性分析
2. 实体连贯性分析
3. 跨段落引用统计
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
import re
from result_adapter import load_query_results, write_unavailable_report


def extract_entities_simple(text: str) -> Set[str]:
    """
    简单的实体提取（基于大写词）

    实际应该用 NER 模型（如 spaCy）
    """
    # 简单方法：提取首字母大写的连续词
    words = text.split()
    entities = set()

    current_entity = []
    for word in words:
        # 去除标点
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
            current_entity.append(clean_word)
        else:
            if current_entity:
                entities.add(' '.join(current_entity))
                current_entity = []

    if current_entity:
        entities.add(' '.join(current_entity))

    return entities


def extract_pronouns(text: str) -> List[str]:
    """提取代词"""
    pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their']
    text_lower = text.lower()
    found = []

    for pronoun in pronouns:
        if re.search(r'\b' + pronoun + r'\b', text_lower):
            found.append(pronoun)

    return found


def check_coreference_simple(passages: List[str]) -> Dict:
    """
    简单的共指检查

    实际应该用共指消解模型（如 AllenNLP Coref）
    """
    # 检查是否有代词但缺少先行词
    issues = []

    for i, passage in enumerate(passages):
        pronouns = extract_pronouns(passage)

        if pronouns:
            # 检查前面的段落是否有实体
            if i == 0:
                # 第一个段落有代词但没有先行词
                entities = extract_entities_simple(passage)
                if not entities:
                    issues.append({
                        'passage_idx': i,
                        'issue': 'pronoun_without_antecedent',
                        'pronouns': pronouns
                    })
            else:
                # 检查前面的段落
                prev_entities = set()
                for j in range(i):
                    prev_entities.update(extract_entities_simple(passages[j]))

                if not prev_entities:
                    issues.append({
                        'passage_idx': i,
                        'issue': 'pronoun_without_prior_entity',
                        'pronouns': pronouns
                    })

    return {
        'total_issues': len(issues),
        'issues': issues
    }


def compute_entity_coherence(passages: List[str]) -> Dict:
    """
    计算实体连贯性

    检查段落间是否有共同实体（表示连贯）
    """
    if len(passages) < 2:
        return {'coherence': 1.0, 'shared_entities': 0}

    # 提取每个段落的实体
    passage_entities = [extract_entities_simple(p) for p in passages]

    # 计算段落间的实体重叠
    overlaps = []
    for i in range(len(passages)):
        for j in range(i + 1, len(passages)):
            shared = passage_entities[i] & passage_entities[j]
            union = passage_entities[i] | passage_entities[j]

            if union:
                overlap = len(shared) / len(union)
                overlaps.append(overlap)

    avg_overlap = np.mean(overlaps) if overlaps else 0

    # 计算全局共享实体
    all_entities = set()
    for entities in passage_entities:
        all_entities.update(entities)

    shared_count = 0
    for entity in all_entities:
        count = sum(1 for entities in passage_entities if entity in entities)
        if count > 1:
            shared_count += 1

    return {
        'coherence': avg_overlap,
        'shared_entities': shared_count,
        'total_entities': len(all_entities)
    }


def check_cross_references(passages: List[str]) -> Dict:
    """
    检查跨段落引用

    检查是否有指示性词语（"as mentioned", "above", "previously"等）
    """
    reference_patterns = [
        r'\bas mentioned\b',
        r'\bas stated\b',
        r'\bas described\b',
        r'\babove\b',
        r'\bpreviously\b',
        r'\bearlier\b',
        r'\bthis\b',
        r'\bthat\b',
        r'\bthese\b',
        r'\bthose\b'
    ]

    references = []

    for i, passage in enumerate(passages):
        passage_lower = passage.lower()
        for pattern in reference_patterns:
            if re.search(pattern, passage_lower):
                references.append({
                    'passage_idx': i,
                    'pattern': pattern
                })
                break  # 只记录一次

    return {
        'total_references': len(references),
        'references': references
    }


def analyze_single_query(query_result: Dict) -> Dict:
    """分析单个查询的上下文依赖"""
    passages = query_result.get('selected_passages', [])

    if not passages or len(passages) < 2:
        return None

    # 提取文本
    passage_texts = [p.get('text', '') for p in passages]

    # 1. 共指检查
    coref_result = check_coreference_simple(passage_texts)

    # 2. 实体连贯性
    entity_result = compute_entity_coherence(passage_texts)

    # 3. 跨段落引用
    cross_ref_result = check_cross_references(passage_texts)

    return {
        'num_passages': len(passages),
        'coref_issues': coref_result['total_issues'],
        'entity_coherence': entity_result['coherence'],
        'shared_entities': entity_result['shared_entities'],
        'total_entities': entity_result['total_entities'],
        'cross_references': cross_ref_result['total_references']
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
            output_file, "上下文依赖分析诊断报告",
            "每题的 selected_passages 文本",
        )
        return

    # 计算统计指标
    coref_issues = [s['coref_issues'] for s in valid_stats]
    entity_coherence = [s['entity_coherence'] for s in valid_stats]
    shared_entities = [s['shared_entities'] for s in valid_stats]
    cross_refs = [s['cross_references'] for s in valid_stats]

    avg_coref_issues = np.mean(coref_issues)
    avg_entity_coherence = np.mean(entity_coherence)
    avg_shared_entities = np.mean(shared_entities)
    avg_cross_refs = np.mean(cross_refs)

    # 问题案例
    problem_cases = []
    for i, s in enumerate(valid_stats):
        if s['coref_issues'] > 0 or s['entity_coherence'] < 0.2:
            problem_cases.append((i, s))

    # 生成报告
    report = []
    report.append("# 上下文依赖分析诊断报告\n")
    report.append(f"**分析样本数**: {len(valid_stats)}\n")
    report.append("\n---\n")

    report.append("\n## 关键发现\n")
    report.append(f"\n### 整体统计\n")
    report.append(f"- **平均共指问题**: {avg_coref_issues:.2f} 个/查询")
    report.append(f"\n- **平均实体连贯性**: {avg_entity_coherence:.3f}")
    report.append(f"\n- **平均共享实体**: {avg_shared_entities:.2f} 个/查询")
    report.append(f"\n- **平均跨段落引用**: {avg_cross_refs:.2f} 个/查询\n")

    # 假设验证
    report.append(f"\n### 假设验证\n")
    report.append(f"\n**假设**: 独立选择破坏段落间的依赖关系\n")

    # 判断标准
    has_coref_issues = avg_coref_issues > 0.5
    low_coherence = avg_entity_coherence < 0.3
    few_shared = avg_shared_entities < 2

    if has_coref_issues and low_coherence:
        report.append(f"\n**结论**: ✅ **假设强烈成立**\n")
        report.append(f"\n观察到明显的上下文依赖问题：")
        report.append(f"\n- 共指问题频繁（平均 {avg_coref_issues:.2f} 个）")
        report.append(f"\n- 实体连贯性低（{avg_entity_coherence:.3f}）")
        report.append(f"\n- 共享实体少（{avg_shared_entities:.2f} 个）\n")
        report.append(f"\n这表明独立选择确实破坏了段落间的依赖关系。\n")

    elif has_coref_issues or low_coherence:
        report.append(f"\n**结论**: ⚠️ **假设部分成立**\n")
        report.append(f"\n存在一定的上下文依赖问题，但不是特别严重。\n")

        if has_coref_issues:
            report.append(f"- 有一些共指问题（{avg_coref_issues:.2f} 个）\n")
        if low_coherence:
            report.append(f"- 实体连贯性较低（{avg_entity_coherence:.3f}）\n")

    else:
        report.append(f"\n**结论**: ❌ **假设不成立**\n")
        report.append(f"\n段落间的依赖关系保持较好，独立选择没有明显破坏上下文。\n")

    # 详细分析
    report.append(f"\n### 详细分析\n")

    report.append(f"\n#### 共指问题分布\n")
    no_issues = sum(1 for c in coref_issues if c == 0)
    some_issues = sum(1 for c in coref_issues if 0 < c <= 2)
    many_issues = sum(1 for c in coref_issues if c > 2)

    report.append(f"- 无问题: {no_issues} ({no_issues/len(coref_issues)*100:.1f}%)")
    report.append(f"\n- 少量问题 (1-2): {some_issues} ({some_issues/len(coref_issues)*100:.1f}%)")
    report.append(f"\n- 严重问题 (>2): {many_issues} ({many_issues/len(coref_issues)*100:.1f}%)\n")

    report.append(f"\n#### 实体连贯性分布\n")
    low_coh = sum(1 for c in entity_coherence if c < 0.2)
    med_coh = sum(1 for c in entity_coherence if 0.2 <= c < 0.5)
    high_coh = sum(1 for c in entity_coherence if c >= 0.5)

    report.append(f"- 低连贯 (<0.2): {low_coh} ({low_coh/len(entity_coherence)*100:.1f}%)")
    report.append(f"\n- 中连贯 (0.2-0.5): {med_coh} ({med_coh/len(entity_coherence)*100:.1f}%)")
    report.append(f"\n- 高连贯 (≥0.5): {high_coh} ({high_coh/len(entity_coherence)*100:.1f}%)\n")

    # 问题案例
    if problem_cases:
        report.append(f"\n### 典型问题案例\n")
        report.append(f"\n发现 {len(problem_cases)} 个问题案例：\n")

        for idx, (case_id, case_stat) in enumerate(problem_cases[:5], 1):
            report.append(f"\n**案例 {idx}**:")
            report.append(f"\n- 共指问题: {case_stat['coref_issues']}")
            report.append(f"\n- 实体连贯性: {case_stat['entity_coherence']:.3f}")
            report.append(f"\n- 共享实体: {case_stat['shared_entities']}/{case_stat['total_entities']}\n")

    # 建议
    report.append(f"\n## 改进建议\n")

    if has_coref_issues or low_coherence:
        report.append(f"\n基于诊断结果，建议：\n")
        report.append(f"\n1. ✅ **实施上下文完整性建模**（中优先级）")
        report.append(f"\n   - 在 QUBO 中添加段落间依赖项")
        report.append(f"\n   - 考虑共指链和实体连贯性")
        report.append(f"\n   - 预期提升: +1-2% F1\n")

        report.append(f"\n2. ✅ **图结构建模**")
        report.append(f"\n   - 将段落建模为图节点")
        report.append(f"\n   - 边表示依赖关系（共指、实体共现）")
        report.append(f"\n   - 选择时考虑子图连通性\n")

        report.append(f"\n3. ⚠️ **后处理调整**（临时方案）")
        report.append(f"\n   - 选择后检查共指完整性")
        report.append(f"\n   - 必要时调整选择\n")
    else:
        report.append(f"\n上下文依赖问题不严重，可以优先考虑其他优化方向。\n")

    report.append(f"\n---\n")
    report.append(f"\n**注意**: 本分析使用简单的启发式方法。")
    report.append(f"实际应该使用 NER 和共指消解模型以获得更准确的结果。\n")
    report.append(f"\n**生成时间**: 自动生成")
    report.append(f"\n**脚本**: `scripts/diagnosis/context_dependency_diagnosis.py`\n")

    # 写入文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(''.join(report))

    print(f"✅ 报告生成: {output_file}")

    # 打印摘要
    print(f"\n=== 摘要 ===")
    print(f"共指问题: {avg_coref_issues:.2f}")
    print(f"实体连贯性: {avg_entity_coherence:.3f}")
    print(f"共享实体: {avg_shared_entities:.2f}")

    if has_coref_issues or low_coherence:
        print(f"\n✅ 假设成立：建议实施上下文完整性建模")
    else:
        print(f"\n⚠️ 假设不成立或较弱")


def main():
    parser = argparse.ArgumentParser(description='上下文依赖分析诊断')
    parser.add_argument('--results', type=Path, required=True,
                       help='实验结果文件 (JSON)')
    parser.add_argument('--output', type=Path, required=True,
                       help='输出报告路径')

    args = parser.parse_args()

    print("=== 上下文依赖分析诊断 ===")
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

    return 0


if __name__ == '__main__':
    exit(main())
