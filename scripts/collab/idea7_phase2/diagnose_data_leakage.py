#!/usr/bin/env python3
"""
诊断 Idea 7 Phase 2 数据泄露问题

检查点：
1. result.json 是否包含 query_embedding
2. query_embedding 是否等于 gold passages 的均值（数据泄露）
3. 训练数据加载是否正确
"""

import json
import numpy as np
import sys
from pathlib import Path

def diagnose_result_json(result_path: str):
    """诊断 result.json 文件"""
    print(f"\n{'='*70}")
    print(f"📁 检查文件: {result_path}")
    print(f"{'='*70}\n")

    if not Path(result_path).exists():
        print(f"❌ 文件不存在: {result_path}")
        return False

    with open(result_path) as f:
        data = json.load(f)

    # Handle wrapped format
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
        print(f"✅ 检测到包装格式 ({{samples: [...]}}) - {len(samples)} 个样本")
    elif isinstance(data, list):
        samples = data
        print(f"✅ 检测到列表格式 - {len(samples)} 个样本")
    else:
        print(f"❌ 未知格式: {type(data)}")
        return False

    # 检查关键字段
    issues = []
    valid_count = 0

    for i, sample in enumerate(samples[:10]):  # 只检查前10个
        has_query_emb = 'query_embedding' in sample and sample['query_embedding'] is not None
        has_passages = 'selected_passages' in sample or 'retrieved' in sample

        if has_passages:
            passages_key = 'selected_passages' if 'selected_passages' in sample else 'retrieved'
            passages = sample[passages_key]

            gold_passages = [p for p in passages if p.get('is_gold', False)]
            has_embeddings = all('embedding' in p and p['embedding'] is not None for p in passages)

            if has_query_emb and has_embeddings and len(gold_passages) > 0:
                valid_count += 1

                # 检查数据泄露：query_emb 是否等于 gold passages 均值
                query_emb = np.array(sample['query_embedding'])
                gold_embs = [np.array(p['embedding']) for p in gold_passages]
                gold_mean = np.mean(gold_embs, axis=0)

                # 计算余弦相似度
                cos_sim = np.dot(query_emb, gold_mean) / (np.linalg.norm(query_emb) * np.linalg.norm(gold_mean))

                if i < 3:  # 打印前3个样本的详细信息
                    print(f"\n  样本 {i}:")
                    print(f"    - query_embedding: ✅ (shape: {query_emb.shape})")
                    print(f"    - passages with embeddings: ✅ ({len(passages)} 个)")
                    print(f"    - gold passages: {len(gold_passages)} 个")
                    print(f"    - query vs gold_mean 相似度: {cos_sim:.4f}")

                    if cos_sim > 0.999:
                        print(f"      🚨 极高相似度！可能是数据泄露")
                        issues.append(f"样本 {i}: query_emb ≈ gold_mean (相似度 {cos_sim:.4f})")
                    elif cos_sim > 0.95:
                        print(f"      ⚠️  高相似度，需要关注")
                    else:
                        print(f"      ✅ 正常相似度")
            else:
                if not has_query_emb:
                    issues.append(f"样本 {i}: 缺少 query_embedding")
                if not has_embeddings:
                    issues.append(f"样本 {i}: passages 缺少 embedding 字段")
                if len(gold_passages) == 0:
                    issues.append(f"样本 {i}: 没有 gold passages (检索失败)")

    print(f"\n{'='*70}")
    print(f"📊 检查结果")
    print(f"{'='*70}")
    print(f"有效样本数: {valid_count} / {len(samples[:10])} (前10个)")

    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个问题:")
        for issue in issues[:5]:  # 只显示前5个
            print(f"  - {issue}")
    else:
        print(f"\n✅ 前10个样本检查通过")

    return valid_count > 0 and len(issues) == 0


def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_data_leakage.py <result.json路径>")
        print("\n示例:")
        print("  python diagnose_data_leakage.py /path/to/result.json")
        sys.exit(1)

    result_path = sys.argv[1]

    print("\n" + "="*70)
    print("🔍 Idea 7 Phase 2 数据泄露诊断")
    print("="*70)

    success = diagnose_result_json(result_path)

    print("\n" + "="*70)
    print("💡 建议")
    print("="*70)

    if success:
        print("\n✅ 数据文件看起来正常！")
        print("\n如果训练时仍然出现 Recall=1.0 从 epoch 1 开始，检查:")
        print("  1. 训练脚本是否使用了正确的 result.json 路径")
        print("  2. 训练脚本是否正确加载 query_embedding")
        print("  3. 运行: grep 'query_emb.*gold' scripts/rag/train/train_soft_qubo_simple.py")
    else:
        print("\n❌ 发现问题！")
        print("\n修复步骤:")
        print("  1. 确保 eval_rag_refactored.py 包含 query_embedding 输出")
        print("  2. 确保使用 --dump_passages 标志运行 eval")
        print("  3. 重新生成 result.json:")
        print(f"     bash scripts/collab/idea7_phase2/run_idea7_phase2.sh")


if __name__ == "__main__":
    main()
