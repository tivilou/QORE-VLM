# Phase 2: Solver Fix + Idea 6 - Experiment Results

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

---

## 实验结果

结果按时间戳组织，每个目录包含一趟完整的 10 组实验：

```
exchange/p2_solver_idea6/
├── README.md                    # 本文件：实验说明
├── 20260728T212216/             # 第一趟实验
│   ├── README.md                # 自动生成的结果汇总
│   ├── meta/git_state.txt
│   └── ...
├── 20260728T212432/             # 第二趟实验
│   └── ...
└── [更多时间戳目录]/
```

每趟实验的详细结果见对应时间戳目录下的 `README.md`。

---

## 如何运行新实验

详见 `scripts/collab/p2_solver_idea6/README.md` 和 `SETUP.md`。

简要流程：
```bash
# 1. 拉取最新代码
git pull origin main

# 2. 运行实验
cd scripts/collab/p2_solver_idea6
bash run_p2_experiments.sh

# 3. 提交结果
git add ../../exchange/p2_solver_idea6/<timestamp>/
git commit -m "experiment(p2): solver+idea6 results <timestamp>"
git push
```

---

## 相关文档

### 实验脚本
- `scripts/collab/p2_solver_idea6/README.md` - 脚本使用说明
- `scripts/collab/p2_solver_idea6/SETUP.md` - 详细设置指南
- `scripts/collab/p2_solver_idea6/CHANGELOG.md` - 脚本修改历史

### 技术文档
- `docs/rag/corpus_modes.md` - Corpus 模式技术指南
- `docs/rag/troubleshooting.md` - 常见问题排查

---

## 提交格式说明

每趟实验自动生成一个时间戳目录（北京时间），包含：

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
