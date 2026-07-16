# RAG Passage Selection with QORE

## Goal

实现并验证 QORE 在 RAG 场景下的 passage selection,相比 baseline(Top-K/MMR)在质量-多样性权衡上取得优势。

## Current State

**阶段**: P0(gamma 调优)修复完成,但仍存在稳定性和性能问题

**完成的工作**:
- ✅ 核心代码实现(selector.py, signals_rag.py, baselines/)
- ✅ 22个单元测试全过
- ✅ Synthetic demo 可运行
- ✅ P0修复:gamma 自动调优改为 top-3K median策略(`2e4685f`)

**调试发现的问题**:
1. **Gamma 自动调优缺陷**(已修复):
   - 原版基于 top-K mean,被 near-duplicates 污染
   - 修复:改用 top-3K median(更robust)
   
2. **QORE 在对抗性场景下表现差**:
   - K=8: recall 20-40%(不稳定,seed敏感) vs MMR 60%
   - K=5: recall 44% vs MMR 40%(QORE 稍优)
   - **K 越大表现越差**(违反直觉)
   
3. **模拟退火稳定性差**:
   - 不同 seed: recall 0-40%(标准差 12.6%)
   - 同一问题收敛到不同局部最优

**根本原因分析**:
- Synthetic scenario 是对抗性的:near-dup 的 relevance(0.77-1.0)接近 gold(0.80-1.0)
- Gold 之间也有中等冗余(0.1-0.3)
- QORE 无法正确权衡"高质量+中等冗余"的 gold vs "高质量+高冗余"的 near-dup

## In Progress

无(暂停,待决定下一步方向)

## Next Actions

### P1: 调查 K 越大越差的问题
1. Profile K=5 vs K=8 的 QUBO 能量地形
2. 检查约束惩罚 lambda 是否需要随 K 调整
3. 可能需要改进 solver(更好的初始化/退火schedule)

### P2: 真实数据集验证
1. 在 HotpotQA/NQ 等真实数据上测试
2. 看 QORE 是否只在 synthetic scenario 失效
3. 如果真实数据也差,需要重新考虑算法设计

### P3(可选): 算法改进
- 多阶段选择(先质量筛选,再冗余优化)
- 改进 QUBO 构造(非线性权重?)
- 尝试其他 solver(遗传算法/贪心后优化?)

## Blockers

需要决定优先级:继续优化 RAG 还是先完成其他模块?

## Validation

- 22测试全过
- Synthetic demo: K=5 时 QORE ≈ MMR,K=8 时 QORE << MMR
- 缺乏真实数据集验证

## Relevant Decisions

无
