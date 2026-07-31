# Baseline 异常调查结果

## 问题

在 `COMPLETE_PROJECT_STATUS.md` 中提到：

> P2 baseline (0.3196) 比之前 baseline (0.4454) 低 28.2%

这导致对 Idea 6 效果的怀疑。

## 调查结果

### ✅ **真相：0.4454 不是 baseline，而是 Idea 6 的配置！**

查看 `exchange/p2_solver_idea6/20260729T131723/README.md`：

| Config | γ | δ | Complementarity | Recall@5 |
|--------|---|---|-----------------|----------|
| gamma0.5_delta0.0 **[baseline]** | 0.5 | 0.0 | - | **0.3196** |
| gamma0.5_delta0.1 | 0.5 | 0.1 | dpr | **0.4454** |

**关键发现**：
- 0.3196 = baseline (γ=0.5, δ=0.0, 无互补性)
- 0.4454 = Idea 6 (γ=0.5, δ=0.1, **有互补性**)

### 误解来源

在之前的文档中，将 0.4454 错误地标记为 "baseline"，导致混淆：

1. `exchange/idea7_phase2/20260731T120040/DETAILED_ANALYSIS.md`:
   - "对比 Phase 2 baseline (Recall=0.4454)" ❌ **错误**

2. `exchange/COMPLETE_PROJECT_STATUS.md`:
   - "P2 baseline: 0.3196 vs 之前: 0.4454" ❌ **错误**

实际上：
- **P2 真实 baseline**: 0.3196 (γ=0.5, δ=0.0)
- **P2 Idea 6 最佳**: 0.4598 (γ=0.3, δ=0.1)
- **0.4454**: Idea 6 的一个配置 (γ=0.5, δ=0.1)

## 正确的对比

### Idea 6 Phase 2 真实效果

| 指标 | Baseline (δ=0) | Idea 6 最佳 (γ=0.3, δ=0.1) | 提升 |
|------|---------------|--------------------------|------|
| **Recall@5** | 0.3196 | 0.4598 | **+43.9%** ✅ |
| **F1** | 0.4406 | 0.5107 | **+15.9%** ✅ |
| **EM** | 0.2500 | 0.3000 | **+20.0%** ✅ |
| **Redundancy** | 0.8090 | 0.7982 | **-1.3%** ✅ |

**结论**：✅ **Idea 6 显著有效**，远超 +5% 目标

### Idea 7 Phase 2 真实效果

| 指标 | Baseline | Idea 7 | 改进 |
|------|---------|--------|------|
| **Recall@5** | 0.3196 | 0.3224 | **+0.9%** ❌ |

**注**：Idea 7 的对比也有误，之前文档说 -27.6% 是相对于 0.4454，但那是 Idea 6 的结果。
实际上 Idea 7 相对真实 baseline (0.3196) 只有微小提升 +0.9%，仍然远低于目标。

## 正确的项目状态

### ✅ Idea 6: 成功

- **Baseline**: Recall@5 = 0.3196 (γ=0.5, δ=0.0)
- **最佳配置**: Recall@5 = 0.4598 (γ=0.3, δ=0.1)
- **提升**: +43.9% (远超 +5% 目标)
- **推荐**: γ=0.5, δ=0.1 (平衡最好)

### ❌ Idea 7: 失败

- **Baseline**: Recall@5 = 0.3196
- **Idea 7**: Recall@5 = 0.3224
- **提升**: +0.9% (远低于 +5% 目标)
- **状态**: 暂停

## 下一步

1. **✅ 无需担心 baseline 异常** - 这是误解，真实 baseline 一直是 0.3196
2. **✅ Idea 6 Phase 3 可以继续** - 效果已确认显著
3. **✅ 推荐配置**: γ=0.5, δ=0.1 (Recall=0.4454, F1=0.5092)

## 需要修正的文档

1. `exchange/COMPLETE_PROJECT_STATUS.md` - 移除 baseline 异常部分
2. `exchange/idea7_phase2/20260731T120040/DETAILED_ANALYSIS.md` - 修正 baseline 数值
3. `/root/.claude/projects/-home-Q-DUET-VLM/memory/p1-diagnosis-f1-verdict.md` - 确认使用正确 baseline

---

**结论**: ✅ **没有 baseline 异常，Idea 6 效果真实有效，可以继续 Phase 3！**

**致歉**: 之前的混淆导致了不必要的担忧。0.4454 是 Idea 6 的成果，不是 baseline。
