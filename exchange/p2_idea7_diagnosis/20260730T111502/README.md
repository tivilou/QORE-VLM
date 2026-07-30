# Idea 7 重新评估 - 20260730T111502

## 背景

在 Phase 1 诊断中，Idea 7 (QUBO 代理目标) 显示：
- QUBO 最优解平均 Recall: 0.5909
- Oracle 最优解平均 Recall: 0.8913
- **Gap**: 0.3004 (命中率 34.7%)

**问题**: Idea 6 和 Idea 7 之前共享同一个 oracle。现在 Idea 6 已实现并取得显著效果：
- Recall@5: 0.3196 → 0.4454 (+37.9%)
- F1: 0.4406 → 0.5092 (+13.6%)

本实验目的：测量实现 Idea 6 后，Idea 7 的 oracle gap 是否仍然显著。

## 实验配置

- **γ**: 0.5 (Idea 6 推荐值)
- **δ**: 0.1 (Idea 6 推荐值)
- **λ**: 2.0
- **样本数**: 200
- **种子**: 42
- **Complementarity**: DPR answer scorer

## 结果

详见 `qubo_diagnosis.md`

## 对比分析

| 指标 | Phase 1 (无 Idea 6) | Phase 2 (含 Idea 6) | 变化 |
|------|---------------------|---------------------|------|
| QUBO 最优 Recall | 0.5909 | 0.5909 | TODO |
| Oracle Recall | 0.8913 | 0.8913 | TODO |
| Gap | 0.3004 | 0.3004 | TODO |
| 命中率 | 34.7% | 34.7% | TODO |

## 决策建议

- [ ] **Gap 仍显著 (>0.15)**: 强烈建议实现 Idea 7
- [ ] **Gap 中等 (0.08-0.15)**: 考虑实现 Idea 7（权衡成本）
- [ ] **Gap 较小 (<0.08)**: Idea 6 已充分优化，暂不做 Idea 7

## 文件清单

- `result.json` - RAG 评估完整结果 (4210455 bytes, 200 样本)
- `qubo_diagnosis.md` - QUBO 诊断报告
- `run.log` - 运行日志

---

*生成于 20260730T111502*
