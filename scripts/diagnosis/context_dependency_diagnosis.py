#!/usr/bin/env python3
"""上下文完整性诊断 — 验证 idea 4 的前提。

**假设**: 独立打分选段落会破坏段落间的依赖（共指、实体连贯），
导致选出的段落各自相关、合起来信息不完整。

**关键设计：必须有 gold 对照。**
旧版只统计选中段落的共指问题数和实体连贯性，输出「实体连贯性 0.017」
这类数字 —— 没有分母，无法判断这是高还是低。Wikipedia 段落里首字母大写
的词遍地都是，绝对值本身没有意义。

本版对每道题算两组同样的指标：
    selected  = 模型选中的 K 条
    gold      = 候选池里含答案的那些条
然后看 selected 是否**系统性差于** gold。只有出现差距，才说明
「独立选择破坏了依赖」；若两者相当，说明依赖损伤不是选择造成的。

用法:
    python scripts/diagnosis/context_dependency_diagnosis.py \
        --results <dir>/gamma_0.5/result.json \
        --output  <dir>/analysis/context_dependency.md

数据要求: 评测时加 --dump_passages（需要 selected_passages 的文本）。

⚠️ 已知偏差（写论文时必须声明）:
gold 段落的文本只有在它**被选中**时才会出现在 dump 里。所以 gold 对照组
偏向"选对了"的情形，会低估真实差距。要完全消除这个偏差，需要 dump 所有
候选的文本（JSON 体积 ~5-10x）。当前实现是折中。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnosis_io import DiagnosisInputError, load_samples

PRONOUNS = ['he', 'she', 'it', 'they', 'him', 'her', 'them',
            'his', 'hers', 'its', 'their']

# 指示性引用词：出现这些说明段落依赖前文
REFERENCE_PATTERNS = [
    r'\bas mentioned\b', r'\bas stated\b', r'\bas described\b',
    r'\bthe former\b', r'\bthe latter\b',
    r'\bpreviously\b', r'\bearlier\b', r'\babove\b',
]


def extract_entities(text: str) -> Set[str]:
    """粗略实体抽取：连续的首字母大写词。

    不是 NER。句首单词会被误判，Wikipedia 里这类噪声很多 —— 这是为什么
    本诊断只看 selected 与 gold 的**相对差距**，不看绝对值。
    """
    entities = set()
    current: List[str] = []
    for word in text.split():
        clean = re.sub(r'[^\w]', '', word)
        if clean and clean[0].isupper() and len(clean) > 1:
            current.append(clean)
        else:
            if current:
                entities.add(' '.join(current))
                current = []
    if current:
        entities.add(' '.join(current))
    return entities


def dangling_pronoun_count(texts: List[str]) -> int:
    """有代词但整组段落里找不到任何实体先行词的段落数。

    这是"共指被切断"的代理信号：段落以 "He was born in..." 开头，
    而 He 指的是谁在别的段落里。
    """
    all_entities: Set[str] = set()
    for t in texts:
        all_entities |= extract_entities(t)

    count = 0
    for t in texts:
        low = t.lower()
        has_pronoun = any(re.search(r'\b' + p + r'\b', low) for p in PRONOUNS)
        if has_pronoun and not extract_entities(t):
            # 本段有代词却没有自己的实体 -> 依赖别处
            count += 1
    return count


def entity_coherence(texts: List[str]) -> float:
    """段落两两之间实体集合的平均 Jaccard。高 = 讲同一批实体 = 连贯。"""
    if len(texts) < 2:
        return 0.0
    ents = [extract_entities(t) for t in texts]
    vals = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            union = ents[i] | ents[j]
            if union:
                vals.append(len(ents[i] & ents[j]) / len(union))
    return float(np.mean(vals)) if vals else 0.0


def cross_reference_count(texts: List[str]) -> int:
    """含指示性引用词的段落数（依赖未包含的前文）。"""
    n = 0
    for t in texts:
        low = t.lower()
        if any(re.search(p, low) for p in REFERENCE_PATTERNS):
            n += 1
    return n


def group_metrics(texts: List[str]) -> Dict | None:
    """一组段落的三项完整性指标。"""
    if len(texts) < 2:
        return None
    return {
        'n': len(texts),
        'dangling_pronouns': dangling_pronoun_count(texts),
        'entity_coherence': entity_coherence(texts),
        'cross_references': cross_reference_count(texts),
    }


def analyze_sample(sample: Dict) -> Dict | None:
    """对一道题分别算 selected 组和 gold 组的指标。"""
    passages = sample.get('selected_passages') or []
    if len(passages) < 2:
        return None

    sel_texts = [p.get('text', '') for p in passages]
    # gold 组：选中段落里 is_gold 为真的（见模块 docstring 的偏差说明）
    gold_texts = [p.get('text', '') for p in passages if p.get('is_gold')]

    sel = group_metrics(sel_texts)
    if sel is None:
        return None

    return {
        'selected': sel,
        'gold': group_metrics(gold_texts),   # 可能是 None（gold < 2 条）
        'f1': sample.get('f1'),
        'recall': sample.get('recall'),
    }


def generate_report(stats: List[Dict], output_path: Path) -> bool:
    valid = [s for s in stats if s is not None]
    if len(valid) < 10:
        print(f"⚠️ 有效样本仅 {len(valid)} 条，不足以判断", file=sys.stderr)
        return False

    def col(group: str, key: str):
        return [s[group][key] for s in valid
                if s.get(group) is not None and s[group].get(key) is not None]

    sel_dang = col('selected', 'dangling_pronouns')
    sel_coh = col('selected', 'entity_coherence')
    sel_xref = col('selected', 'cross_references')

    paired = [s for s in valid if s.get('gold') is not None]
    n_paired = len(paired)

    out = []
    out.append("# 上下文完整性诊断报告（idea 4）\n")
    out.append(f"\n**样本数**: {len(valid)}")
    out.append(f"\n**可做 gold 对照的样本**: {n_paired}\n")
    out.append("\n---\n")

    out.append("\n## selected 组的绝对值\n")
    out.append(f"\n- 悬空代词段落数: {np.mean(sel_dang):.2f} / 题")
    out.append(f"\n- 实体连贯性: {np.mean(sel_coh):.3f}")
    out.append(f"\n- 跨段引用段落数: {np.mean(sel_xref):.2f} / 题\n")
    out.append("\n⚠️ 这些**绝对值本身不能判断好坏** —— 实体抽取是粗启发式，"
               "句首大写词会被误判。结论只看下面的 selected vs gold 差距。\n")

    out.append("\n## selected vs gold 对照（核心）\n")

    if n_paired < 10:
        out.append(f"\n❌ **无法判断** —— 只有 {n_paired} 题的选中段落里含 ≥2 条 gold。")
        out.append("\n\n这本身是个信号：说明模型很少同时选中多条 gold。")
        out.append("但它也让本诊断的对照组失效 —— 假设**未能验证**，")
        out.append("不要据此判断 idea 4 的去留。\n")
        verdict = 'inconclusive'
    else:
        ps_dang = [s['selected']['dangling_pronouns'] for s in paired]
        pg_dang = [s['gold']['dangling_pronouns'] for s in paired]
        ps_coh = [s['selected']['entity_coherence'] for s in paired]
        pg_coh = [s['gold']['entity_coherence'] for s in paired]

        d_dang = np.mean(ps_dang) - np.mean(pg_dang)
        d_coh = np.mean(ps_coh) - np.mean(pg_coh)

        out.append(f"\n| 指标 | selected | gold | 差距 |")
        out.append(f"\n|---|---|---|---|")
        out.append(f"\n| 悬空代词 / 题 | {np.mean(ps_dang):.2f} | "
                   f"{np.mean(pg_dang):.2f} | {d_dang:+.2f} |")
        out.append(f"\n| 实体连贯性 | {np.mean(ps_coh):.3f} | "
                   f"{np.mean(pg_coh):.3f} | {d_coh:+.3f} |\n")

        # 配对检验：同一道题内比较，消除题目难度差异
        try:
            from scipy.stats import wilcoxon
            if len(set(np.array(ps_coh) - np.array(pg_coh))) > 1:
                _, p_coh = wilcoxon(ps_coh, pg_coh)
                out.append(f"\n实体连贯性配对检验: p={p_coh:.4f}\n")
            else:
                p_coh = 1.0
        except Exception:
            p_coh = 1.0

        # 判据：selected 的连贯性显著低于 gold，且悬空代词更多
        worse_coh = d_coh < -0.05 and p_coh < 0.05
        worse_dang = d_dang > 0.3

        if worse_coh and worse_dang:
            verdict = 'holds'
            out.append("\n### ✅ 假设成立\n")
            out.append(f"\nselected 的实体连贯性比 gold 低 {abs(d_coh):.3f}"
                       f"（p={p_coh:.4f}），且悬空代词多 {d_dang:.2f} 个/题。")
            out.append("\n说明独立打分选出的段落，依赖完整性确实差于 gold 组合。\n")
        elif worse_coh or worse_dang:
            verdict = 'partial'
            out.append("\n### ⚠️ 假设部分成立\n")
            out.append("\n两项指标只有一项显示 selected 更差，证据不够强。\n")
        else:
            verdict = 'rejected'
            out.append("\n### ❌ 假设不成立\n")
            out.append("\nselected 的依赖完整性与 gold 相当，"
                       "「独立选择破坏依赖」在 NQ 上没有得到支持。")
            out.append("\nidea 4 的优先级应下调。\n")

    out.append("\n---\n")
    out.append("\n## 方法学限制\n")
    out.append("\n1. **gold 对照组有偏**：gold 段落的文本只在被选中时才进 dump，"
               "所以对照组偏向「选对了」的情形，会**低估**真实差距。")
    out.append("\n2. 实体抽取是首字母大写启发式，非 NER。仅用于相对比较。")
    out.append("\n3. 悬空代词是共指断裂的代理信号，不等于真实共指错误。\n")
    out.append("\n**脚本**: `scripts/diagnosis/context_dependency_diagnosis.py`\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(out))
    print(f"✅ 报告生成: {output_path}")
    print(f"\n=== 摘要 ===")
    print(f"样本 {len(valid)}，可对照 {n_paired}")
    print(f"结论: {verdict}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description='上下文完整性诊断 (idea 4)')
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    print("=== 上下文完整性诊断 (idea 4) ===")
    print(f"输入: {args.results}")

    try:
        samples = load_samples(args.results, require=('selected_passages',))
    except DiagnosisInputError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print(f"加载 {len(samples)} 个样本")
    stats = [analyze_sample(s) for s in samples]
    return 0 if generate_report(stats, args.output) else 1


if __name__ == '__main__':
    sys.exit(main())
