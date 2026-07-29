# Phase 2 实验脚本修复完成报告

## 问题总结

师弟遇到的错误：
```
/root/miniconda3/bin/python: Error while finding module specification for 'scripts.rag.eval_rag_refactored' 
(ModuleNotFoundError: No module named 'scripts')
ERROR: eval_rag_refactored.py 不支持 --delta 参数，请先添加
```

## 修复内容

### ✅ 修复 1: 添加 `--delta` 参数
**文件**: `scripts/rag/eval_rag_refactored.py`

在参数解析部分添加：
```python
p.add_argument("--delta", type=float, default=0.0,
               help="Complementarity weight (only for method=qore with complementarity_method set). "
                    "Positive delta rewards selecting complementary passage pairs.")
```

### ✅ 修复 2: 添加 `--complementarity_method` 参数
**文件**: `scripts/rag/eval_rag_refactored.py`

在参数解析部分添加：
```python
p.add_argument("--complementarity_method", type=str, default=None,
               choices=["dpr", None],
               help="How to compute complementarity c_ij (only for method=qore). "
                    "None (default): no complementarity term. "
                    "'dpr': use DPR answer scorer pairwise signals (requires --use_answer_scorer).")
```

### ✅ 修复 3: 更新 `select_passages` 调用
**文件**: `scripts/rag/eval_rag_refactored.py`

在调用 `select_passages()` 时添加新参数：
```python
selected_local = select_passages(
    query_emb,
    retrieved_embs,
    K=args.K,
    method=args.method,
    num_reads=args.num_reads,
    lam=args.lam,
    gamma=args.gamma,
    delta=args.delta,  # 新增
    complementarity_method=args.complementarity_method,  # 新增
    answer_scorer=answer_scorer if args.complementarity_method == 'dpr' else None,  # 新增
    passage_texts=retrieved_texts if args.complementarity_method == 'dpr' else None,  # 新增
    question=question if args.complementarity_method == 'dpr' else None,  # 新增
    lambda_mmr=args.lambda_mmr,
    seed=args.seed,
    relevance_scores=retrieval_scores,
    qore_prefilter_size=args.qore_prefilter_size,
    direct_solve_max_n=args.direct_solve_max_n,
    diagnostics=qubo_diag,
)
```

### ✅ 修复 4: 确保 passage_texts 可用
**文件**: `scripts/rag/eval_rag_refactored.py`

在 answer_scorer 逻辑后添加：
```python
# Ensure retrieved_texts is available if complementarity_method='dpr'
if args.complementarity_method == 'dpr' and retrieved_texts is None:
    if args.corpus_mode == "precomputed":
        retrieved_texts = [candidates[idx]["text"] for idx in retrieved_idx]
    elif args.corpus_mode == "wiki_dpr":
        pass  # already set above
    else:
        retrieved_texts = [corpus.passages[idx] for idx in retrieved_idx]
```

### ✅ 修复 5: 解决模块导入问题
**文件**: `scripts/collab/p2_solver_idea6/run_p2_experiments.sh`

在脚本开始处添加 PYTHONPATH 设置：
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# 设置 PYTHONPATH 使 Python 能找到 scripts 模块
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
```

## 验证结果

所有 7 项验证测试全部通过：

1. ✅ `--delta` 参数存在且可用
2. ✅ `--complementarity_method` 参数存在且可用
3. ✅ `select_passages` 函数包含所有必需参数
4. ✅ `scripts.rag.eval_rag_refactored` 模块可正常导入
5. ✅ 参数解析正确（delta=0.3, complementarity_method='dpr'）
6. ✅ `run_p2_experiments.sh` 的参数检查逻辑正常
7. ✅ PYTHONPATH 设置正确，模块导入无问题

## 使用说明

### 运行完整实验（10 个配置）

```bash
cd /home/Q-DUET-VLM/QORE-VLM
bash scripts/collab/p2_solver_idea6/run_p2_experiments.sh
```

### 实验配置详情

**Baseline (1个)**
- gamma=0.5, delta=0.0, complementarity=none

**Idea 6 调参网格 (9个)**
- gamma ∈ {0.3, 0.5, 0.7}
- delta ∈ {0.1, 0.3, 0.5}
- complementarity_method=dpr
- use_answer_scorer=true

共 10 个实验，每个 200 题。

### 提交结果

实验完成后会自动生成 README.md，然后：

```bash
git add exchange/p2_solver_idea6/<TIMESTAMP>/
git commit -m 'experiment(p2): solver+idea6 results <TIMESTAMP>'
git push
```

## 技术细节

### 参数传递链路

1. **命令行** → `--delta` 和 `--complementarity_method`
2. **parse_args()** → `args.delta` 和 `args.complementarity_method`
3. **select_passages()** → 函数参数 `delta`, `complementarity_method`, `answer_scorer`, `passage_texts`, `question`
4. **QORE selector** → 计算互补性矩阵并优化 QUBO

### 互补性方法工作原理

当 `complementarity_method='dpr'` 时：
- 使用 DPR answer scorer 计算 passage 对之间的互补性
- QUBO 目标函数变为: `w = gamma * b - delta * c`
  - `b`: redundancy matrix（冗余度，越高越相似）
  - `c`: complementarity matrix（互补性，越高越互补）
  - `delta > 0`: 奖励选择互补的 passage 对

### 依赖关系

使用 `complementarity_method='dpr'` 时必须：
- 设置 `--use_answer_scorer`（加载 DPR reader）
- 确保 `passage_texts` 可用（需要文本内容计算互补性）
- 传递 `question` 给 selector（DPR 需要 query）

所有这些依赖都已在修复中自动处理。

## 故障排除

### 如果仍然遇到模块导入错误

```bash
# 手动设置 PYTHONPATH
export PYTHONPATH="/home/Q-DUET-VLM/QORE-VLM:$PYTHONPATH"

# 验证模块可导入
python -c "import scripts.rag.eval_rag_refactored; print('OK')"
```

### 如果参数不识别

```bash
# 验证参数存在
python -m scripts.rag.eval_rag_refactored --help | grep -E "delta|complementarity"
```

应该看到：
```
--delta DELTA
--complementarity_method {dpr,None}
```

### 手动测试单个实验

```bash
cd /home/Q-DUET-VLM/QORE-VLM
python -m scripts.rag.eval_rag_refactored \
    --corpus_mode aligned \
    --dataset nq_open \
    --max_samples 10 \
    --method qore \
    --K 5 \
    --gamma 0.5 \
    --delta 0.3 \
    --complementarity_method dpr \
    --use_answer_scorer \
    --skip_generation \
    --output_dir /tmp/test_output
```

## 修改文件清单

1. `scripts/rag/eval_rag_refactored.py` - 添加参数和更新调用
2. `scripts/collab/p2_solver_idea6/run_p2_experiments.sh` - 设置 PYTHONPATH

## 验证时间

- 修复完成: 2026-07-28
- 验证测试: 7/7 通过
- 状态: ✅ 可以交付给师弟使用

---

**注意**: 运行完整 10 个实验需要较长时间（约 200 题 × 10 配置），建议在有足够计算资源和时间的情况下运行。
