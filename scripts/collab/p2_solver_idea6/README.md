# Phase 2: Solver Fix + Idea 6

## 目标

1. **验证 solver 修复效果** — anneal → brute，命中率从 3-7% → 100%
2. **调参 idea 6 互补性矩阵** — gamma × delta 网格搜索
3. **优化目标** — 冗余度 < baseline 且 F1/Recall ≥ baseline（多目标）

## 工作流程（3 步）

### 1. 拉取最新代码
```bash
git pull origin main
```

### 2. 运行实验（自动汇总）
```bash
cd scripts/collab/p2_solver_idea6
bash run_p2_experiments.sh
```
- 自动跑 10 组配置（1 baseline + 9 idea6 grid）
- 每组 ~200 样本，总耗时约数小时
- **完成后自动汇总结果**，无需手动操作

### 3. 提交到 GitHub
```bash
# 查看生成的结果摘要
cat ../../exchange/p2_solver_idea6/<timestamp>/README.md

# 提交（<timestamp> 在上一步输出中显示）
git add ../../exchange/p2_solver_idea6/<timestamp>/
git commit -m "experiment(p2): solver+idea6 results <timestamp>"
git push
```

**提示**: 如果需要单独重新汇总（如手动修改了某个结果），可运行：
```bash
python collect_p2_results.py  # 不传参数，自动使用最新批次
# 或
python collect_p2_results.py <timestamp>  # 指定批次
```

## 实验配置

### Baseline（solver fix only）
- **gamma**: 0.5
- **delta**: 0.0（无互补项）
- **用途**: 验证 solver 修复效果

### Idea 6 调参网格
```
gamma × delta:
  0.3 × {0.1, 0.3, 0.5}
  0.5 × {0.1, 0.3, 0.5}
  0.7 × {0.1, 0.3, 0.5}
= 9 组配置
```

### 共同配置
- **数据集**: NQ-open
- **Corpus 模式**: wiki_dpr (全量 21M Wikipedia)
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

## 输出产物

运行后生成：
```
exchange/p2_solver_idea6/<timestamp>/
├── README.md                 # 自动生成的结果汇总
├── meta/git_state.txt        # Git 状态
├── gamma0.5_delta0.0/        # Baseline
│   └── result.json           # （打包到 .zip，不提交）
├── gamma0.3_delta0.1/        # Idea 6 配置 1
├── ...                       # 其他 8 组
└── results_<timestamp>.zip   # 所有 result.json 打包
```

**不提交的大文件**（已在 .gitignore）：
- result.json（~4.1 MB/个）
- *.samples.json

## 外部依赖确认

在运行前确认：

- [ ] `eval_rag_refactored.py` 支持 `--delta` 参数
- [ ] `eval_rag_refactored.py` 支持 `--complementarity_method` 参数
- [ ] DPR answer scorer 模型已加载（`--use_answer_scorer`）
- [ ] 计算资源：200 题 × 10 配置，DPR 前向 ~66 次/题

`run_p2_experiments.sh` 会在运行前自动检查前两项。

## 已知限制

1. **DPR 成对打分成本**  
   N=12 时每题 66 次 scorer 前向，200 题共 13.2k 次。  
   按 batch_size=16 约 825 次前向，比生成便宜但仍需实测吞吐。

2. **Prefilter M>20 会 fallback anneal**  
   若 `qore_prefilter_size > 20`，预筛路径会退回 anneal（brute 上限 N≤20）。  
   建议保持默认 M=15。

3. **Complementarity 只在 qore 方法下工作**  
   topk/mmr/vqc 方法不支持。

## 故障排查

### 常见问题
详见 `SETUP.md` 的故障排除部分，或查看项目级文档：
- `../../../docs/rag/troubleshooting.md` - 全面的故障排查指南

### 快速检查
```bash
# 1. 验证参数支持
python -m scripts.rag.eval_rag_refactored --help | grep -E "delta|complementarity"

# 2. 测试单个配置（10题）
# 编辑 run_p2_experiments.sh，临时设置 MAX_SAMPLES=10

# 3. 查看 CHANGELOG
cat CHANGELOG.md
```

## 相关文档

### 本目录文档
- `SETUP.md` - 详细设置指南（环境、依赖、验证）
- `CHANGELOG.md` - 脚本修改历史
- `docs/P2_FIX_REPORT.md` - 参数修复技术报告
- `docs/ALIGNED_FIX_SUMMARY.md` - Aligned 模式修复说明

### 项目文档
- `../../../docs/rag/corpus_modes.md` - Corpus 模式技术指南
- `../../../docs/rag/troubleshooting.md` - 常见问题排查

### 实验结果
- `../../../exchange/p2_solver_idea6/` - 实验结果目录
- `../../../exchange/p2_solver_idea6/README.md` - 实验需求和结果总览

---

**有问题随时联系！**
