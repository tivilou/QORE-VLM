# RAG QORE 修复计划：暴露 gamma 参数恢复证据覆盖率

## 背景

师弟 200 样本试跑 #2（报告：`/home/Q-DUET-VLM/analysis(1).md`）发现：
- ✅ QORE 冗余度最低（0.764 vs MMR 0.832）
- ❌ QORE Recall@5 低 11 个百分点（24.4% vs MMR 34.9%）
- ❌ QORE F1 低 6 个百分点（40.9% vs MMR 47.1%）

**根因**（已对照源码验证）：
1. **lam 不是多样性旋钮**：lam 只进基数约束，对所有 K=5 的解是常数，不影响排序
2. **gamma 才是真正的旋钮**：gamma=1.0 固定，10 个冗余项累加压过质量项 → 过度分散
3. **上轮 lam 2.0→1.0 的"修复"无效**，需回滚并修正误导性帮助文本

## 目标

让 QORE 在保持低冗余优势的同时，恢复证据覆盖率至接近或不低于 MMR/Top-K。

## 修复计划（分 4 阶段）

---

### 阶段 1：代码修复（暴露 gamma + 回滚 lam）

**目标**：让用户能通过 CLI 调整 gamma，回滚错误的 lam 修改

#### 1.1 暴露 gamma 参数到 selector

**文件**：`applications/rag/selector.py`

**改动**：
```python
def select_passages(
    query_embedding: np.ndarray,
    passage_embeddings: np.ndarray,
    K: int,
    method: str = "qore",
    relevance_scores: np.ndarray | None = None,
    redundancy_method: str = "cosine",
    lam: float = 2.0,
    gamma: float | None = None,  # ← 新增参数
    num_reads: int = 50,
    ...
):
```

**传递到 qore_solve**：
```python
# 第 91 行和 104 行，两处调用
x = qore_solve(a, b, K, lam=lam, gamma=gamma, method="anneal", **kwargs)
```

**预期**：selector 支持 gamma 参数，默认 None（自动调参为 1.0）

---

#### 1.2 添加 gamma 到 eval_rag_refactored.py CLI

**文件**：`scripts/rag/eval_rag_refactored.py`

**改动 1**：参数定义
```python
# 在第 95-98 行附近
p.add_argument("--lam", type=float, default=2.0,  # ← 回滚默认值
               help="QUBO cardinality penalty weight (keep at ~2.0)")
p.add_argument("--gamma", type=float, default=None,
               help="QUBO redundancy weight (None=auto-tune to 1.0; "
                    "lower=favor relevance, higher=favor diversity; "
                    "try 0.05-0.5 for better answer coverage)")
```

**改动 2**：传递到 selector
```python
# 第 238-248 行附近
selected_local = select_passages(
    query_emb,
    retrieved_embs,
    K=args.K,
    method=args.method,
    num_reads=args.num_reads,
    lam=args.lam,
    gamma=args.gamma,  # ← 新增
    lambda_mmr=args.lambda_mmr,
    seed=args.seed,
    relevance_scores=retrieval_scores,
)
```

**预期**：CLI 支持 `--gamma` 参数

---

#### 1.3 添加 gamma 到 eval_suite.py CLI

**文件**：`scripts/rag/eval_suite.py`

**改动 1**：参数定义
```python
# 第 61-64 行附近
p.add_argument("--K", type=int, default=5)
p.add_argument("--num_reads", type=int, default=100)
p.add_argument("--lam", type=float, default=2.0)  # ← 回滚默认值
p.add_argument("--gamma", type=float, default=None)  # ← 新增
p.add_argument("--lambda_mmr", type=float, default=0.7)
```

**改动 2**：传递到 eval_rag_refactored
```python
# 第 80-95 行附近，构建 cmd 的地方
cmd = [
    sys.executable, "-m", "scripts.rag.eval_rag_refactored",
    ...
    "--lam", str(args.lam),
    "--gamma", str(args.gamma) if args.gamma is not None else "None",  # ← 新增
    "--lambda_mmr", str(args.lambda_mmr),
    ...
]
```

**注意**：需要特殊处理 None 值传递

**预期**：eval_suite 支持 `--gamma` 参数

---

#### 1.4 单元测试验证

**测试内容**：
1. selector 接受 gamma 参数并正确传递
2. gamma=0 时只考虑相关性（退化为 Top-K）
3. gamma=10 时极度偏向多样性
4. CLI 参数解析正确

**验证方法**：
```python
# 测试 1：gamma=0 应该接近 Top-K
selected_gamma0 = select_passages(..., gamma=0.0, method='qore')
selected_topk = select_passages(..., method='topk')
# 预期：两者高度重合

# 测试 2：gamma=10 应该极度分散
selected_gamma10 = select_passages(..., gamma=10.0, method='qore')
# 预期：冗余度极低，但 recall 也低

# 测试 3：CLI 参数解析
args = parse_args(['--gamma', '0.5', ...])
assert args.gamma == 0.5
```

**交付**：
- [ ] 所有测试通过
- [ ] commit: "功能:暴露gamma参数+回滚lam默认值"

---

### 阶段 2：gamma 网格搜索（快速筛选）

**目标**：在 200 样本上快速找到最优 gamma 值

#### 2.1 设计实验

**数据**：NQ-open validation 前 200 题（与师弟试跑一致）

**gamma 候选值**：0, 0.05, 0.1, 0.2, 0.5, 1.0

**配置**：
- `--corpus_mode wiki_dpr`
- `--max_samples 200`
- `--method qore`
- `--seed 42`
- `--skip_generation`（跳过生成，只评估选择质量）
- `--K 5`
- `--lam 2.0`

**评估指标**：
1. **Recall@5**（主要）：证据覆盖率
2. **Redundancy**（次要）：保持低冗余
3. 权衡：选择 Recall 最高且 Redundancy 仍明显低于 MMR 的 gamma

---

#### 2.2 执行脚本

```bash
# 在 QORE-VLM 根目录
for gamma in 0 0.05 0.1 0.2 0.5 1.0; do
  echo "Testing gamma=$gamma"
  python -m scripts.rag.eval_rag_refactored \
    --corpus_mode wiki_dpr \
    --dataset nq_open \
    --max_samples 200 \
    --method qore \
    --seed 42 \
    --K 5 \
    --lam 2.0 \
    --gamma $gamma \
    --skip_generation \
    --output_dir results/rag/gamma_search \
    --output_file qore_gamma${gamma}_seed42.json
done
```

**预计时间**：6 × (200 题 × 0.45s) ≈ 9 分钟

---

#### 2.3 分析结果

**提取关键指标**：
```python
import json
import pandas as pd

results = []
for gamma in [0, 0.05, 0.1, 0.2, 0.5, 1.0]:
    with open(f'results/rag/gamma_search/qore_gamma{gamma}_seed42.json') as f:
        data = json.load(f)
        results.append({
            'gamma': gamma,
            'recall': data['metrics']['mean_recall'],
            'redundancy': data['metrics']['mean_redundancy'],
            'diversity': data['metrics']['mean_diversity'],
        })

df = pd.DataFrame(results)
print(df)
```

**选择标准**：
1. Recall@5 > 30%（目标：接近 MMR 的 34.9%）
2. Redundancy < 0.80（保持优势：明显低于 MMR 的 0.832）
3. 在满足以上条件的 gamma 中，选择 Recall 最高的

**预期最优范围**：gamma ∈ [0.05, 0.2]

**交付**：
- [ ] 6 个 JSON 结果文件
- [ ] gamma 选择报告（表格 + 推荐值）
- [ ] commit: "实验:gamma网格搜索(200样本)"

---

### 阶段 3：最优 gamma 的完整验证（200 样本 + 生成）

**目标**：用最优 gamma 跑完整端到端（包含生成），验证 EM/F1 提升

#### 3.1 执行完整评测

假设阶段 2 选出 `gamma=0.1`（示例）

```bash
# 三个方法完整对比
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 200 \
  --methods qore mmr topk \
  --seeds 42 \
  --K 5 \
  --lam 2.0 \
  --gamma 0.1 \
  --output_dir results/rag/gamma_optimized_200
```

**预计时间**：3 × (200 题 × (0.45s + 0.24s)) ≈ 7 分钟

---

#### 3.2 对比分析

**对比基准**：师弟试跑 #2 结果（`/home/Q-DUET-VLM/analysis(1).md`）

| 指标 | QORE (gamma=1.0) | QORE (gamma=0.1) | MMR | Top-K |
|------|------------------|------------------|-----|-------|
| Recall@5 | 24.4% | **目标 >30%** | 34.9% | 35.8% |
| Redundancy | 0.764 | **保持 <0.80** | 0.832 | 0.841 |
| F1 | 40.9% | **目标 >45%** | 47.1% | 46.6% |
| EM | 24.5% | **目标 >26%** | 28.0% | 27.0% |

**成功标准**：
- ✅ Recall 提升 ≥5 个百分点
- ✅ F1 提升 ≥4 个百分点
- ✅ Redundancy 仍明显低于 MMR（保持优势）

**如果失败**：
- 如果 Recall 仍 <28%，考虑更激进的 gamma=0.05 或 gamma=0
- 如果 Redundancy 接近 MMR，考虑稍微提高 gamma

**交付**：
- [ ] 3 个方法的 JSON 结果
- [ ] summary.json 汇总
- [ ] 对比报告（markdown 表格）
- [ ] commit: "实验:gamma=0.1完整验证(200样本)"

---

### 阶段 4：全量评测（可选，取决于阶段 3 结果）

**前置条件**：阶段 3 成功（QORE 性能接近或超越 MMR）

#### 4.1 多 seed 验证（推荐）

```bash
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 0 \
  --methods qore mmr topk \
  --seeds 42,123,456 \
  --K 5 \
  --lam 2.0 \
  --gamma 0.1 \
  --output_dir results/rag/full_eval_gamma_optimized
```

**预计时间**：9 × (~3600 题 × 0.7s) ≈ 6.3 小时

**交付**：
- [ ] 9 个 JSON 结果（3 methods × 3 seeds）
- [ ] summary.json（含均值、标准差、配对 t-test）
- [ ] 最终实验报告
- [ ] commit: "实验:gamma优化后全量评测(3seeds)"

---

## 风险与备选方案

### 风险 1：gamma 搜索后仍无改善

**症状**：所有 gamma 值下 Recall 都 <28%

**原因**：50 候选空间太大，SA 容易陷入局部最优

**备选方案**：
1. **relevance-first 候选池**：
   - 修改 selector.py，QUBO 前先按 DPR 分数取 Top-15
   - 基线在同一候选池上公平比较
   - 预期：Recall 提升，速度加快

2. **降低 direct_solve_max_n**：
   - 从 64 降到 30，强制走两阶段（Top-M prefilter + QUBO）
   - selector.py 第 86 行

---

### 风险 2：gamma 调优后冗余度优势消失

**症状**：gamma=0.05 时 Redundancy 接近 MMR (0.82)

**诊断**：gamma 过低，冗余惩罚失效

**对策**：
- 接受适度的冗余度（如 0.80 vs MMR 0.832，仍有优势）
- 或选择稍高的 gamma（如 0.1），平衡 Recall 和 Redundancy

---

### 风险 3：速度仍然过慢

**症状**：gamma 优化后选择仍 >400 ms/题

**对策**（P2 优先级）：
- 减少 num_reads（100 → 50）
- 使用 relevance-first 小候选池（50 → 15）
- 或明确 QORE 定位为离线重排

---

## 检查点与决策点

### Checkpoint 1：阶段 1 完成后

**检查**：
- [ ] 代码改动正确（gamma 暴露、lam 回滚）
- [ ] 单元测试通过
- [ ] CLI 参数解析正确

**决策**：通过后进入阶段 2

---

### Checkpoint 2：阶段 2 完成后

**检查**：
- [ ] 6 个 gamma 值的 Recall 曲线
- [ ] 找到最优 gamma（Recall >30%, Redundancy <0.80）

**决策 A**：找到最优值 → 进入阶段 3  
**决策 B**：所有 gamma 都不理想 → 触发风险 1 备选方案

---

### Checkpoint 3：阶段 3 完成后

**检查**：
- [ ] QORE F1 ≥ 45%（vs 基线 47.1%）
- [ ] QORE Recall ≥ 30%（vs 基线 34.9%）
- [ ] Redundancy 仍 <0.80（vs 基线 0.832）

**决策 A**：成功 → 进入阶段 4（全量评测）  
**决策 B**：部分成功（Recall 提升但仍低 2-3 个点）→ 考虑 relevance-first 候选池  
**决策 C**：失败 → 触发风险 1 备选方案

---

## 时间估算

| 阶段 | 工作量 | 预计时间 |
|------|--------|----------|
| 阶段 1：代码修复 | 4 个文件改动 + 测试 | 30-60 分钟 |
| 阶段 2：gamma 搜索 | 6 次运行 × 1.5 分钟 | 10 分钟 |
| 阶段 3：完整验证 | 3 methods × 7 分钟 | 10 分钟 |
| 阶段 4：全量评测 | 9 runs × 42 分钟 | 6.3 小时（后台） |
| **总计** | - | **1-2 小时（交互）+ 6 小时（后台）** |

---

## 交付清单

### 代码改动
- [ ] `applications/rag/selector.py`（暴露 gamma）
- [ ] `scripts/rag/eval_rag_refactored.py`（CLI + 传递）
- [ ] `scripts/rag/eval_suite.py`（CLI + 传递）

### 实验结果
- [ ] 阶段 2：6 个 gamma 的 JSON + 分析报告
- [ ] 阶段 3：3 methods 的 JSON + 对比报告
- [ ] 阶段 4：9 runs 的 JSON + summary + 最终报告

### 文档
- [ ] 更新 `docs/rag_full_eval_guide.md`（推荐 gamma 值）
- [ ] 更新 `.ai-progress/workstreams/rag-selector/state.md`

---

## 下一步

**立即行动**：开始阶段 1（代码修复）

**第一个 commit**：
```
功能:暴露gamma参数+回滚lam默认值

根因分析:
- lam只进基数约束,对所有K=5的解是常数,不影响排序
- gamma才是真正控制多样性的旋钮
- gamma=1.0固定导致10个冗余项累加压过质量项

改动:
- selector.py: 添加gamma参数并传递给qore_solve
- eval_rag_refactored.py: 添加--gamma CLI, 回滚lam默认值2.0
- eval_suite.py: 添加--gamma CLI传递

测试:
- gamma=0时接近Top-K(只考虑相关性)
- gamma=10时极度分散(只考虑多样性)
- CLI参数解析正确

参考: 师弟200样本报告 /home/Q-DUET-VLM/analysis(1).md
```

---

是否开始执行阶段 1？
