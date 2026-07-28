# Phase 2: Solver Fix + Idea 6 Complementarity Matrix

## 实验目标

1. **验证 solver 修复效果** — 对比修复前后，idea 7 的 gap 缩小多少
2. **调参 idea 6** — 网格搜索 (gamma, delta)，目标：冗余度 < baseline 且 F1/Recall ≥ baseline

## 实验配置

### Baseline（solver fix only）
- **gamma**: 0.5
- **delta**: 0.0（无互补项）
- **用途**: 验证 solver 修复效果

### Idea 6 调参网格

gamma × delta:
- 0.3 × {0.1, 0.3, 0.5}
- 0.5 × {0.1, 0.3, 0.5}
- 0.7 × {0.1, 0.3, 0.5}

共 9 组 + 1 组 baseline (delta=0.0) = 10 组配置

### 共同配置
- **数据集**: NQ-open aligned corpus
- **样本数**: 200
- **方法**: qore
- **K**: 5
- **λ**: 2.0
- **种子**: 42

## 评测指标

对比 baseline（MMR 冗余 0.796、F1 47.15%）：

- **主目标**: 冗余度显著 < 0.796
- **约束**: F1 ≥ 47.15%、Recall@5 不低于 baseline
- **次要**: Precision、EM

## 提交结果格式

每趟实验建一个 `YYYYMMDDTHHMMSS/` 目录（北京时间），包含：

```
YYYYMMDDTHHMMSS/
├── README.md                 # 自动生成（结果表格 + 最佳配置）
├── meta/git_state.txt        # Git 状态
├── gamma0.5_delta0.0/        # Baseline 配置目录
├── gamma0.3_delta0.1/        # Idea 6 配置目录（9 组）
├── ...
└── results_<timestamp>.zip   # 所有 result.json 打包
```

**不提交的大文件**（已在 .gitignore）：
- result.json（~4.1 MB/个）
- *.samples.json

## 外部依赖

- [ ] `eval_rag_refactored.py` 支持 `--delta` 和 `--complementarity_method` 参数
- [ ] `--use_answer_scorer` 的 DPR 模型路径配置正确
- [ ] 计算资源：200 题 × 10 配置，每题 DPR 前向 ~66 次（N=12）

## 已知限制

1. **DPR 成对打分成本**: N=12 时每题 66 次 scorer 前向，200 题共 13.2k 次，按 batch_size=16 约 825 次前向
2. **Prefilter M>20 会 fallback anneal**: 建议保持默认 M=15
3. **Complementarity 只在 qore 方法下工作**

## 如何运行实验

详见 `scripts/collab/p2_solver_idea6/README.md`（工作流程说明）。

简要流程（3 步）：
```bash
# 1. 进入脚本目录
cd scripts/collab/p2_solver_idea6

# 2. 运行实验（自动汇总）
bash run_p2_experiments.sh

# 3. 提交
git add ../../exchange/p2_solver_idea6/<timestamp>
git commit && git push
```

## 相关文档

- **工作流程**: `scripts/collab/p2_solver_idea6/README.md`（给执行者看）
- **实现细节**: `.ai-progress/workstreams/rag-selector/sessions/20260728T060000Z-solver-idea6.md`（gitignore，本地可见）
- **Idea 6 塌缩警告**: `.ai-progress/.../refs/p2-plan-inputs-20260728.md` 第 3 节
- **Commit**: `61cc37d` feat(rag): solver fix + idea 6 complementarity matrix
