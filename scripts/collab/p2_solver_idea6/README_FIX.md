# Phase 2 实验脚本修复说明

## 问题
1. ~~`eval_rag_refactored.py` 不支持 `--delta` 参数~~ ✅ 已修复
2. ~~`eval_rag_refactored.py` 不支持 `--complementarity_method` 参数~~ ✅ 已修复
3. ~~模块导入错误 `No module named 'scripts'`~~ ✅ 已修复

## 已完成的修复

### 1. 添加命令行参数
在 `scripts/rag/eval_rag_refactored.py` 中添加了：
- `--delta`: 互补性权重参数
- `--complementarity_method`: 互补性计算方法（支持 'dpr' 或 None）

### 2. 传递参数到 select_passages
修改了 `select_passages()` 调用，传递以下参数：
- `delta`
- `complementarity_method`
- `answer_scorer` (当 complementarity_method='dpr' 时)
- `passage_texts` (当 complementarity_method='dpr' 时)
- `question` (当 complementarity_method='dpr' 时)

### 3. 修复模块导入问题
在 `run_p2_experiments.sh` 中添加了：
```bash
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
```

## 使用方法

从 QORE-VLM 仓库根目录运行：

```bash
bash scripts/collab/p2_solver_idea6/run_p2_experiments.sh
```

**重要**: 不要从 `scripts/collab/p2_solver_idea6/` 目录内运行脚本！

脚本会：
1. 自动检查参数支持
2. 运行 10 个实验配置（1 个 baseline + 9 个 idea 6 配置）
3. 自动汇总结果
4. 生成 README.md

## 验证修复

可以先验证参数是否可用：

```bash
cd /home/Q-DUET-VLM/QORE-VLM
python -m scripts.rag.eval_rag_refactored --help | grep -E "delta|complementarity"
```

应该能看到：
- `--delta DELTA`
- `--complementarity_method {dpr,None}`

## 实验配置

### Baseline (1个)
- gamma=0.5, delta=0.0, complementarity=none

### Idea 6 调参网格 (9个)
- gamma ∈ {0.3, 0.5, 0.7}
- delta ∈ {0.1, 0.3, 0.5}
- complementarity_method=dpr
- use_answer_scorer=true

共 10 个实验配置。

## 提交结果

实验完成后：

```bash
git add exchange/p2_solver_idea6/<TIMESTAMP>/
git commit -m 'experiment(p2): solver+idea6 results <TIMESTAMP>'
git push
```
