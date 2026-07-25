#!/usr/bin/env python3
"""
查询类型分析诊断脚本

目的: 验证"不同类型查询需要不同多样性策略"的假设
假设: 简单事实查询需要高质量，复杂推理查询需要高多样性

输出:
1. 按查询类型分析不同 γ 的效果
2. 各类型查询的最优 γ
3. Query-adaptive 策略的潜在收益
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


def classify_query_simple(query: str) -> str:
    """
    简单的查询分类（基于长度和复杂度）

    实际应该用更好的方法：
    - NER 识别实体数量
    - 依存句法分析
    - 问题类型分类器
    """
    query_lower = query.lower().strip()
    words = query_lower.split()

    # 特征提取
    length = len(words)

    # 复杂查询的指标
    has_multi_hop = any(w in query_lower for w in ['and', 'also', 'besides', 'moreover'])
    has_comparison = any(w in query_lower for w in ['compare', 'difference', 'versus', 'vs'])
    has_complex_wh = any(w in query_lower for w in ['why', 'how', 'explain'])

    # 简单查询的指标
    has_simple_wh = query_lower.startswith(('who', 'what', 'when', 'where'))

    # 分类
    if length <= 5 and has_simple_wh and not has_complex_wh:
        return 'simple'
    elif length > 10 or has_multi_hop or has_comparison or has_complex_wh:
        return 'complex'
    else:
        return 'medium'


def load_results_all_gamma(results_dir: Path) -> Dict[float, List[Dict]]:
    """加载所有 γ 配置的结果"""
    gamma_results = {}

    for gamma_file in results_dir.glob('gamma_*/result.json'):
        gamma_str = gamma_file.parent.name.split('_')[1]
        gamma = float(gamma_str)

        with open(gamma_file, 'r') as f:
            data = json.load(f)

        gamma_results[gamma] = data.get('results', [])

    return gamma_results


def compute_metrics(query_results: List[Dict]) -> Dict[str, float]:
    """计算指标"""
    if not query_results:
        return {}

    recalls = []
    precisions = []
    f1s = []
    redundancies = []

    for result in query_results:
        metrics = result.get('metrics', {})
        if metrics:
            recalls.append(metrics.get('recall', 0))
            precisions.append(metrics.get('precision', 0))
            f1s.append(metrics.get('f1', 0))
            redundancies.append(metrics.get('redundancy', 0))

    return {
        'recall': np.mean(recalls) if recalls else 0,
        'precision': np.mean(precisions) if precisions else 0,
        'f1': np.mean(f1s) if f1s else 0,
        'redundancy': np.mean(redundancies) if redundancies else 0,
        'count': len(query_results)
    }


def analyze_by_query_type(gamma_results: Dict[float, List[Dict]]) -> Dict:
    """按查询类型分析"""
    # 分类查询
    query_types = defaultdict(lambda: defaultdict(list))

    # 假设所有 γ 的查询顺序一致
    base_gamma = min(gamma_results.keys())
    base_results = gamma_results[base_gamma]

    # 对每个查询分类
    for idx, result in enumerate(base_results):
        query = result.get('query', '')
        qtype = classify_query_simple(query)

        # 收集所有 γ 下该查询的结果
        for gamma, results in gamma_results.items():
            if idx < len(results):
                query_types[qtype][gamma].append(results[idx])

    # 计算每种类型在每个 γ 下的指标
    type_gamma_metrics = {}
    for qtype in query_types:
        type_gamma_metrics[qtype] = {}
        for gamma in sorted(gamma_results.keys()):
            metrics = compute_metrics(query_types[qtype][gamma])
            type_gamma_metrics[qtype][gamma] = metrics

    return type_gamma_metrics


def find_best_gamma(type_metrics: Dict[float, Dict]) -> Tuple[float, str]:
    """找到最优 γ"""
    best_gamma = None
    best_f1 = -1

    for gamma, metrics in type_metrics.items():
        f1 = metrics.get('f1', 0)
        if f1 > best_f1:
            best_f1 = f1
            best_gamma = gamma

    return best_gamma, f1


def generate_report(type_gamma_metrics: Dict, output_file: Path):
    """生成诊断报告"""
    report = []
    report.append("# 查询类型分析诊断报告\n")
    report.append("\n---\n")

    report.append("\n## 关键发现\n")

    # 统计各类型数量
    report.append("\n### 查询类型分布\n")
    for qtype in ['simple', 'medium', 'complex']:
        if qtype in type_gamma_metrics:
            count = type_gamma_metrics[qtype][list(type_gamma_metrics[qtype].keys())[0]].get('count', 0)
            report.append(f"\n- **{qtype.capitalize()}**: {count} 个查询")

    # 各类型的最优 γ
    report.append("\n\n### 各类型查询的最优 γ\n")

    best_gammas = {}
    for qtype in ['simple', 'medium', 'complex']:
        if qtype not in type_gamma_metrics:
            continue

        best_gamma, best_f1 = find_best_gamma(type_gamma_metrics[qtype])
        best_gammas[qtype] = best_gamma

        report.append(f"\n**{qtype.capitalize()} 查询**:")
        report.append(f"\n- 最优 γ: **{best_gamma}**")
        report.append(f"\n- 最优 F1: {best_f1:.3f}\n")

    # 详细对比表
    report.append("\n### 详细对比\n")

    for qtype in ['simple', 'medium', 'complex']:
        if qtype not in type_gamma_metrics:
            continue

        report.append(f"\n#### {qtype.capitalize()} 查询\n")
        report.append("\n| γ | Recall | Precision | F1 | 冗余度 |")
        report.append("\n|---|--------|-----------|----|---------| ")

        for gamma in sorted(type_gamma_metrics[qtype].keys()):
            metrics = type_gamma_metrics[qtype][gamma]
            report.append(f"\n| {gamma:.1f} | {metrics['recall']:.3f} | {metrics['precision']:.3f} | {metrics['f1']:.3f} | {metrics['redundancy']:.3f} |")

        report.append("\n")

    # 假设验证
    report.append("\n## 假设验证\n")
    report.append("\n**假设**: 不同类型查询需要不同的多样性策略\n")

    # 检查是否有明显差异
    if best_gammas:
        simple_gamma = best_gammas.get('simple', None)
        complex_gamma = best_gammas.get('complex', None)

        if simple_gamma is not None and complex_gamma is not None:
            gamma_diff = abs(simple_gamma - complex_gamma)

            report.append(f"\n**观察**:")
            report.append(f"\n- Simple 查询最优 γ: {simple_gamma}")
            report.append(f"\n- Complex 查询最优 γ: {complex_gamma}")
            report.append(f"\n- 差异: {gamma_diff}\n")

            if gamma_diff >= 0.3:
                report.append(f"\n**结论**: ✅ **假设强烈成立**\n")
                report.append(f"\n简单查询和复杂查询的最优 γ 相差 {gamma_diff}，")
                report.append(f"表明不同类型查询确实需要不同的多样性策略。\n")

                if simple_gamma < complex_gamma:
                    report.append(f"\n- Simple 查询偏好**低多样性**（高质量）")
                    report.append(f"\n- Complex 查询偏好**高多样性**（广覆盖）\n")
                else:
                    report.append(f"\n- Simple 查询偏好**高多样性**")
                    report.append(f"\n- Complex 查询偏好**低多样性**\n")

            elif gamma_diff >= 0.2:
                report.append(f"\n**结论**: ⚠️ **假设成立**\n")
                report.append(f"\n存在一定差异（{gamma_diff}），但不是特别显著。\n")
            else:
                report.append(f"\n**结论**: ❌ **假设不成立**\n")
                report.append(f"\n最优 γ 差异很小（{gamma_diff}），")
                report.append(f"不同类型查询的多样性需求相似。\n")

    # Query-adaptive 潜在收益
    report.append("\n## Query-Adaptive 策略潜在收益\n")

    # 计算如果使用 adaptive γ 的理论收益
    adaptive_f1s = []
    uniform_f1s = []

    for qtype in ['simple', 'medium', 'complex']:
        if qtype not in type_gamma_metrics:
            continue

        # Adaptive: 使用最优 γ
        best_gamma = best_gammas.get(qtype)
        if best_gamma is not None:
            adaptive_f1 = type_gamma_metrics[qtype][best_gamma]['f1']
            adaptive_f1s.append(adaptive_f1)

        # Uniform: 使用固定 γ=0.5
        if 0.5 in type_gamma_metrics[qtype]:
            uniform_f1 = type_gamma_metrics[qtype][0.5]['f1']
            uniform_f1s.append(uniform_f1)

    if adaptive_f1s and uniform_f1s:
        avg_adaptive = np.mean(adaptive_f1s)
        avg_uniform = np.mean(uniform_f1s)
        potential_gain = avg_adaptive - avg_uniform

        report.append(f"\n**理论收益估算**:\n")
        report.append(f"\n- 固定 γ=0.5 (当前): F1 = {avg_uniform:.3f}")
        report.append(f"\n- Adaptive γ (理想): F1 = {avg_adaptive:.3f}")
        report.append(f"\n- **潜在提升**: {potential_gain*100:+.1f}% ({potential_gain:+.3f})\n")

        if potential_gain > 0.01:
            report.append(f"\n✅ Query-adaptive 策略有明显收益\n")
        else:
            report.append(f"\n⚠️ Query-adaptive 策略收益有限\n")

    # 建议
    report.append("\n## 改进建议\n")

    if best_gammas and len(set(best_gammas.values())) > 1:
        report.append("\n基于诊断结果，建议：\n")
        report.append("\n1. ✅ **实施 Query-Adaptive γ 策略**")
        report.append("\n   - 根据查询类型动态调整 γ")
        report.append("\n   - 特征: 查询长度、复杂度、问题类型")
        report.append("\n   - 预期提升: +1-2% F1\n")
        report.append("\n2. ✅ **查询分类器**")
        report.append("\n   - 训练简单的查询复杂度分类器")
        report.append("\n   - 输入: 查询文本")
        report.append("\n   - 输出: 推荐的 γ 值\n")
    else:
        report.append("\n不同类型查询的最优 γ 相似，")
        report.append("Query-adaptive 策略优先级较低。\n")

    report.append("\n---\n")
    report.append("\n**生成时间**: 自动生成")
    report.append("\n**脚本**: `scripts/diagnosis/query_type_diagnosis.py`\n")

    # 写入文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(''.join(report))

    print(f"✅ 报告生成: {output_file}")

    # 打印摘要
    print(f"\n=== 摘要 ===")
    for qtype, gamma in best_gammas.items():
        print(f"{qtype}: 最优 γ = {gamma}")

    if best_gammas and len(set(best_gammas.values())) > 1:
        print(f"\n✅ 假设成立：建议实施 Query-adaptive 策略")
    else:
        print(f"\n⚠️ 假设不成立或较弱")


def main():
    parser = argparse.ArgumentParser(description='查询类型分析诊断')
    parser.add_argument('--results_dir', type=Path, required=True,
                       help='实验结果目录（包含 gamma_*/ 子目录）')
    parser.add_argument('--output', type=Path, required=True,
                       help='输出报告路径')

    args = parser.parse_args()

    print("=== 查询类型分析诊断 ===")
    print(f"输入目录: {args.results_dir}")

    # 加载所有 γ 的结果
    gamma_results = load_results_all_gamma(args.results_dir)
    print(f"加载 {len(gamma_results)} 个 γ 配置")

    if not gamma_results:
        print("❌ 未找到结果文件")
        return 1

    # 按查询类型分析
    type_gamma_metrics = analyze_by_query_type(gamma_results)
    print(f"分析 {len(type_gamma_metrics)} 种查询类型")

    # 生成报告
    generate_report(type_gamma_metrics, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
