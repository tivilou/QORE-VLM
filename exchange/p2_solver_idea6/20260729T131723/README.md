# Phase 2 实验结果 - 20260729T131723

## Git 状态

```
HEAD: e5230c0 fix(collab): use repo root for relative path in package_results

⚠️  Working tree is dirty:
  M scripts/rag/eval_rag_refactored.py
   M scripts/rag/eval_suite.py
  ?? docs/rag_e2e_10_30_analysis.md
  ?? docs/rag_e2e_200_result_analysis.md
  ?? models/
  ?? scripts/diagnosis/result_adapter.py
```

## 实验配置

- **数据集**: nq_open
- **样本数**: 200
- **方法**: qore
- **K**: 5
- **λ**: 2.0
- **种子**: 42

## 实验结果

| Config | γ | δ | Complementarity | Recall@5 | Redundancy | F1 | EM | Precision |
|--------|---|---|-----------------|----------|------------|----|----|-----------|
| gamma0.5_delta0.0 **[baseline]** | 0.5 | 0.0 | - | 0.3196 | 0.8090 | 0.4406 | 0.2500 | 0.2680 |
| gamma0.3_delta0.1 | 0.3 | 0.1 | dpr | 0.4598 | 0.7982 | 0.5101 | 0.2950 | 0.3640 |
| gamma0.3_delta0.3 | 0.3 | 0.3 | dpr | 0.4566 | 0.8022 | 0.5107 | 0.3000 | 0.3620 |
| gamma0.3_delta0.5 | 0.3 | 0.5 | dpr | 0.4193 | 0.7993 | 0.4964 | 0.2850 | 0.3380 |
| gamma0.5_delta0.1 | 0.5 | 0.1 | dpr | 0.4454 | 0.7876 | 0.5092 | 0.2950 | 0.3570 |
| gamma0.5_delta0.3 | 0.5 | 0.3 | dpr | 0.4502 | 0.7943 | 0.5092 | 0.2950 | 0.3550 |
| gamma0.5_delta0.5 | 0.5 | 0.5 | dpr | 0.4052 | 0.7916 | 0.5029 | 0.2850 | 0.3320 |
| gamma0.7_delta0.1 | 0.7 | 0.1 | dpr | 0.4408 | 0.7811 | 0.5004 | 0.2900 | 0.3550 |
| gamma0.7_delta0.3 | 0.7 | 0.3 | dpr | 0.4387 | 0.7886 | 0.5046 | 0.2900 | 0.3490 |
| gamma0.7_delta0.5 | 0.7 | 0.5 | dpr | 0.3817 | 0.7865 | 0.4894 | 0.2750 | 0.3200 |

## 与 Baseline 对比

**Baseline** (solver fix, γ=0.5, δ=0.0):
- Recall@5: 0.3196
- Redundancy: 0.8090
- F1: 0.4406
- EM: 0.2500

**Idea 6 最佳配置**（按目标：冗余度最低且 F1 ≥ baseline）:

- **Config**: gamma0.7_delta0.1 (γ=0.7, δ=0.1)
- Recall@5: 0.4408 (Δ=+0.1212)
- Redundancy: 0.7811 (Δ=-0.0279)
- F1: 0.5004 (Δ=+0.0598)
- EM: 0.2900 (Δ=+0.0400)

## 系统信息

```
NVIDIA GeForce RTX 4090, 49140 MiB
```

## 产物清单

```
gamma0.5_delta0.0/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.3_delta0.1/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.3_delta0.3/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.3_delta0.5/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.5_delta0.1/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.5_delta0.3/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.5_delta0.5/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.7_delta0.1/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.7_delta0.3/
  result.json  # 完整结果（不提交，已打包到 .zip）
gamma0.7_delta0.5/
  result.json  # 完整结果（不提交，已打包到 .zip）
```

---

*生成于 20260729T131723 by scripts/collab/collect_p2_results.py*