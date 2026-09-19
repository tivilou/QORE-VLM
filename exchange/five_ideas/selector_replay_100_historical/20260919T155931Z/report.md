# 100题历史 selector Top-50 -> Top-5 replay

这是固定 Top-50 上的 L0 selector-only replay。选择阶段只使用问题、候选身份/文本、DPR embedding、检索分数和 Answer Scorer 分数；Silver evidence 只在选择完成后用于诊断。没有调用 Generator，也不能单独升级为 L1/L2。

- 输入：`silver-oracle-top5-100-20260915T120525Z-detail.json`，SHA-256 `669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731`
- 规模：100 题 × 50 候选，K=5
- 重放配置：`case_study_compatible`
- 代码 revision：`558ce4f686b2ad7ad6d9b788e330859439bd61a3`

## 结果

| 方法 | 状态 | Silver overlap 均值 | set-F1 均值 | broad 选中均值 | selector miss | embedding 冗余 | 中位耗时 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| qore_as | registered_baseline | 2.590 | 0.518 | 0.000 | 0 | 117.188 | 0.000 |
| topk_as | registered_baseline | 3.550 | 0.710 | 0.000 | 0 | 125.231 | 0.000 |
| mmr_as | replayed | 2.950 | 0.590 | 0.000 | 0 | 120.147 | 0.793 |
| submodular_as | replayed | 2.600 | 0.520 | 0.000 | 0 | 115.302 | 0.668 |
| spectral_dpp_as | replayed | 2.230 | 0.446 | 0.000 | 0 | 112.526 | 4.329 |
| qore_mobius_pairwise | replayed | 2.810 | 0.562 | 0.000 | 0 | 119.511 | 838.487 |
| qore_cohesion_fixed | replayed | 2.780 | 0.556 | 0.000 | 0 | 118.764 | 11.131 |
| qore_cohesion_calibrated | replayed | 2.820 | 0.564 | 0.000 | 0 | 119.080 | 11.072 |

## 相对基线的逐题胜负

| 方法 | 相对 QORE：胜/平/负 | 相对 Top-k：胜/平/负 |
|---|---:|---:|
| mmr_as | 39/54/7 | 0/63/37 |
| submodular_as | 12/79/9 | 1/39/60 |
| spectral_dpp_as | 9/56/35 | 0/26/74 |
| qore_mobius_pairwise | 23/73/4 | 1/44/55 |
| qore_cohesion_fixed | 21/77/2 | 2/45/53 |
| qore_cohesion_calibrated | 25/73/2 | 2/48/50 |

## 输入与边界

- 重新取得的 Wiki-DPR Top-50 通过逐题 ID、title、正文和检索分数校验；最大分数绝对误差：`0`。
- retrieval miss：100/100；这些题不归因于 selector。
- QORE/Mobius/Cohesion 是明确命名的 QUBO enhancer arms；原始 QORE 本身已经是 QUBO selector。
- Differentiable-QUBO/Idea 7 未纳入：checkpoint/training provenance 不在本 replay contract 内，且历史 real-data 结果为负。
- 完整 selector trace 只通过 18083 exchange 保存；GitHub 只保存 compact 文件。
