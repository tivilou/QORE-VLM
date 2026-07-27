# Phase 1 诊断（4 个 idea 前提验证）— 师弟, 2026-07-27 20:13 北京时间

## 跑了什么

- **配置**: `phase1_diagnosis.yaml`，200 题 × 3 个 γ (0.0/0.5/1.0)，`wiki_dpr`，answer scorer 开，生成开
- **真实命令**: `config/actual_commands.txt`（从 `meta/status_*.json` 的 `command` 抽的）
- **耗时**: 每配置 2分38秒 / 2分26秒 / 2分26秒，共 7分30秒
- **检索**: 200 题中 156 题命中 gold（44 题检索阶段失败）

## 结果

| 配置 | Recall@5 | Precision@5 | 冗余度↓ | 多样性↑ | F1 |
|---|---|---|---|---|---|
| γ=0.0 | 0.4989 | 0.3910 | 0.8218 | 0.1782 | 0.5018 |
| γ=0.5 | 0.4253 | 0.3510 | 0.7860 | 0.2140 | 0.4903 |
| γ=1.0 | 0.3909 | 0.3270 | 0.7759 | 0.2241 | 0.4625 |

## 诊断报告

- `analysis/gamma_sweep.md` — idea 1 两阶段 QUBO
- `analysis/context_dependency.md` — idea 4 上下文完整性
- `analysis/complementarity.md` — idea 6 互补性矩阵
- `analysis/qubo_objective.md` — idea 7 Soft QUBO

## 环境

- GPU: NVIDIA GeForce RTX 4090, 49140 MiB
- git HEAD: `8054f94 fix(tuning): don't crash printing metrics that generation never produced`
- 工作区干净: **否**

  ⚠️ **工作区有未提交改动**，跑的代码与仓库 `8054f94` 不一致：
  - `M scripts/rag/eval_rag_refactored.py`
  - ` M scripts/rag/eval_suite.py`
  - `?? docs/rag_e2e_10_30_analysis.md`
  - `?? docs/rag_e2e_200_result_analysis.md`
  - `?? models/`
  - `?? scripts/diagnosis/result_adapter.py`
  - `?? "tatus -sb"`

  2026-07-27 那趟就是手改了 yaml 加 `--skip_generation`、外加两个诊断脚本是旧版，而四份报告里完全看不出来 —— 所以这里自动记下来。

## 原始数据

大产物没提交（见 `exchange/README.md`）。

- 文件: 3 × `result.json`（带 `--dump_passages`）
- 已打包: `/root/P1_20260727T201307.zip` (2.4 MB)，需要时单独发
