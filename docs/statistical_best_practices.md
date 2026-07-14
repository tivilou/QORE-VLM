# Statistical Best Practices for KV-Cache Evaluation

## Problem

师弟的 `experiment_analysis.md` 指出:当前评测是**单次运行无置信区间**,样本是**顺序取前 N 个**(非随机),结果可信度低,无法用于论文级结论。

## Solution

本文档说明如何进行**统计严格的评测**:随机抽样、多种子重复、bootstrap 置信区间。

---

## 1. 随机分层抽样(已支持)

`load_longbench` 现在支持 `sample_seed` 参数:

```bash
# Sequential (legacy, biased toward first samples)
python scripts/kv_cache/eval_kv_cache.py --policy qore --max_samples 30 --seed 42 ...

# Random stratified sampling (unbiased, reproducible with seed)
python scripts/kv_cache/eval_kv_cache.py --policy qore --max_samples 30 --seed 42 ...
```

`--seed` 控制:
- **QORE 求解器的随机性**(anneal/QAOA)
- **样本抽样的随机性**(从每个 task 的 pool 里随机选)

**分层抽样**:配额在 6 个 task 间均匀分配(每个 task ~5 样本 for max_samples=30),避免单一 task 主导。

---

## 2. 多次运行(不同 seed)

单次运行的指标受样本选择和求解器随机性影响。**论文级评测必须多次运行**:

```bash
# Run 1
python scripts/kv_cache/eval_kv_cache.py \
  --policy qore --seed 42 --max_samples 30 \
  --output_file qore_run1.json

# Run 2
python scripts/kv_cache/eval_kv_cache.py \
  --policy qore --seed 43 --max_samples 30 \
  --output_file qore_run2.json

# Run 3
python scripts/kv_cache/eval_kv_cache.py \
  --policy qore --seed 44 --max_samples 30 \
  --output_file qore_run3.json
```

**推荐**:至少 3 次,最好 5-10 次(用于 bootstrap 的稳定性)。

---

## 3. Bootstrap 置信区间

用 `scripts/kv_cache/bootstrap_ci.py` 汇总多次运行,计算 95% CI:

```bash
python scripts/kv_cache/bootstrap_ci.py \
  results/kv_cache/longbench/qore_run*.json
```

**输出示例**:
```
============================================================
Policy: qore
============================================================

F1 Score                     0.4523  [95% CI:   0.4401 -   0.4645]  (n=5)
Latency (ms)              5234.12  [95% CI: 5102.34 - 5365.90]  (n=5)
Throughput (tok/s)          24.56  [95% CI:   23.98 -   25.14]  (n=5)
Avg Cache Length          1024.30  [95% CI: 1019.20 - 1029.40]  (n=5)
Peak Memory (MB)         18234.50  [95% CI:18102.30 -18366.70]  (n=5)
Resident Memory (MB)     15123.40  [95% CI:14998.10 -15248.70]  (n=5)
```

**解读**:
- **Mean**:跨 5 次运行的平均值
- **95% CI**:bootstrap 置信区间(10,000 次重采样)
- 如果两个方法的 CI 不重叠 → 差异统计显著

---

## 4. 对比多个策略

对每个策略分别跑多次,然后汇总对比:

```bash
# QORE (5 runs)
for seed in 42 43 44 45 46; do
  python scripts/kv_cache/eval_kv_cache.py \
    --policy qore --seed $seed --max_samples 30 \
    --output_file qore_run${seed}.json
done

# H2O (5 runs)
for seed in 42 43 44 45 46; do
  python scripts/kv_cache/eval_kv_cache.py \
    --policy h2o --seed $seed --max_samples 30 \
    --output_file h2o_run${seed}.json
done

# Aggregate each
python scripts/kv_cache/bootstrap_ci.py results/*/qore_run*.json --output qore_summary.json
python scripts/kv_cache/bootstrap_ci.py results/*/h2o_run*.json --output h2o_summary.json
```

**论文表格**用 summary 的 mean ± CI。

---

## 5. 样本量建议

| 目标 | max_samples | 运行次数 | 总样本数(次×样本) |
|------|-------------|----------|-------------------|
| **快速验证** | 10-20 | 3 | 30-60 |
| **论文初稿** | 30-50 | 5 | 150-250 |
| **最终版** | 50-100 | 5-10 | 250-1000 |

**权衡**:
- `max_samples` 太小 → 单次方差大,需要更多运行次数
- `max_samples` 太大 → 单次耗时长(LongBench 输入≥3K tokens,Llama-3-8B 单样本 ~30s)
- **推荐**:max_samples=30,运行 5 次(总 150 样本,每次 ~15 分钟)

---

## 6. 分阶段计时(TODO)

当前 `time_ms` 是端到端延迟。**更细粒度的分解**有助于定位瓶颈:
- **Prefill 时间**(处理完整输入,一次性)
- **Decode 时间**(逐 token 生成,累积)
- **Eviction 时间**(淘汰决策 + cache 重组,嵌入在 decode 里)

**实现方式**(未来改进):在 `generate_with_eviction` 里插桩记录 prefill vs decode 计时,写入 `per_sample` 字段。

---

## 7. 论文中如何报告

**错误** ❌:
```
QORE F1: 0.452
H2O F1: 0.431
```
(单次运行,无法判断差异是否显著)

**正确** ✅:
```
QORE F1: 0.452 ± 0.012 (95% CI: [0.440, 0.465], n=5)
H2O F1: 0.431 ± 0.015 (95% CI: [0.416, 0.446], n=5)
```
(多次运行均值 + CI,CI 不重叠 → 显著)

**图表**:用 error bar(CI 区间)或 violin plot(分布)展示不确定性。

---

## 8. 完整示例脚本

```bash
#!/bin/bash
# run_longbench_with_ci.sh
# 对 QORE/H2O/SnapKV 各跑 5 次,汇总 CI

POLICIES="qore h2o snapkv"
SEEDS="42 43 44 45 46"
MAX_SAMPLES=30
OUTPUT_DIR=results/kv_cache/longbench_ci

mkdir -p $OUTPUT_DIR

for policy in $POLICIES; do
  echo "Running $policy with 5 seeds..."
  for seed in $SEEDS; do
    python scripts/kv_cache/eval_kv_cache.py \
      --model_path NousResearch/Meta-Llama-3-8B-Instruct \
      --policy $policy --seed $seed --max_samples $MAX_SAMPLES \
      --max_capacity 1024 --trigger_every 128 \
      --output_dir $OUTPUT_DIR --output_file ${policy}_seed${seed}.json
  done

  # Aggregate
  python scripts/kv_cache/bootstrap_ci.py \
    $OUTPUT_DIR/${policy}_seed*.json \
    --output $OUTPUT_DIR/${policy}_summary.json
done

echo "Done. Summaries in $OUTPUT_DIR/*_summary.json"
```

运行:
```bash
bash scripts/kv_cache/run_longbench_with_ci.sh
```

---

## Summary

| 改进 | 状态 | 工具/参数 |
|------|------|-----------|
| 随机分层抽样 | ✅ | `--seed` in eval_kv_cache.py |
| 多次运行 | ✅ 手动 | 用不同 seed 多次调用 |
| Bootstrap CI | ✅ | `bootstrap_ci.py` |
| 分阶段计时 | 🔜 TODO | 需插桩 generate_with_eviction |

**立即可用**:多次运行 eval + bootstrap_ci 汇总,已可满足论文统计要求。
