# Phase 2: Solver Fix + Idea 6 Complementarity Matrix

## 实验目标

1. **验证 solver 修复效果** — 对比修复前后，idea 7 的 gap 缩小多少
2. **调参 idea 6** — 网格搜索 (gamma, delta)，目标：冗余度 < baseline 且 F1/Recall ≥ baseline

## 实验配置

### Solver 修复验证

```bash
# 修复后的 baseline（delta=0.0，即无互补项）
python -m scripts.rag.eval_rag_refactored \
  --corpus_mode aligned \
  --dataset nq_open \
  --max_samples 200 \
  --method qore \
  --K 5 \
  --gamma 0.5 \
  --delta 0.0 \
  --lam 2.0 \
  --seed 42 \
  --output_dir exchange/p2_solver_idea6/YYYYMMDDTHHMMSS \
  --output_file result.json
```

### Idea 6 调参网格

gamma × delta:
- 0.3 × {0.1, 0.3, 0.5}
- 0.5 × {0.1, 0.3, 0.5}
- 0.7 × {0.1, 0.3, 0.5}

共 9 组 + 1 组 baseline (delta=0.0) = 10 组配置

```bash
for gamma in 0.3 0.5 0.7; do
  for delta in 0.1 0.3 0.5; do
    python -m scripts.rag.eval_rag_refactored \
      --corpus_mode aligned \
      --dataset nq_open \
      --max_samples 200 \
      --method qore \
      --K 5 \
      --gamma $gamma \
      --delta $delta \
      --complementarity_method dpr \
      --use_answer_scorer \
      --lam 2.0 \
      --seed 42 \
      --output_dir exchange/p2_solver_idea6/YYYYMMDDTHHMMSS \
      --output_file result.json
  done
done
```

## 评测指标

对比 baseline（MMR 冗余 0.796、F1 47.15%）：

- **主目标**：冗余度显著 < 0.796
- **约束**：F1 ≥ 47.15%、Recall@5 不低于 baseline
- **次要**：Precision、EM

## 提交要求

每趟实验建一个 `YYYYMMDDTHHMMSS/` 目录（北京时间，取第一个实验的 start_time），包含：

```
YYYYMMDDTHHMMSS/
├── README.md          # 用 scripts/collab/collect_p2_results.py 生成（需新建脚本）
├── config/
│   └── *.yaml         # 配置文件
├── meta/
│   └── git_state.txt  # git log -1, git diff, git status
└── analysis/          # （可选）人工分析、可视化
```

**不提交的大文件**（已在 .gitignore）：
- result.json（4.1 MB/个）
- *.zip
- *.samples.json

## 外部依赖确认

- [ ] `eval_rag_refactored.py` 支持 `--delta` 和 `--complementarity_method` 参数
- [ ] `--use_answer_scorer` 的 DPR 模型路径配置正确
- [ ] 计算资源：200 题 × 10 配置，每题 DPR 前向 ~66 次（N=12）

## 已知限制

1. **DPR 成对打分成本**：N=12 时每题 66 次 scorer 前向，200 题共 13.2k 次，按 batch_size=16 约 825 次前向
2. **Prefilter M>20 会 fallback anneal**：建议保持默认 M=15
3. **Complementarity 只在 qore 方法下工作**

## 相关文档

- 实现细节：`.ai-progress/workstreams/rag-selector/sessions/20260728T060000Z-solver-idea6.md`（gitignore，本地可见）
- Idea 6 塌缩警告：`.ai-progress/.../refs/p2-plan-inputs-20260728.md` 第 3 节
- Commit: `61cc37d` feat(rag): solver fix + idea 6 complementarity matrix
