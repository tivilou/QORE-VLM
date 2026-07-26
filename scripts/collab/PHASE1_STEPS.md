# Phase 1 执行步骤

按顺序执行，每步确认通过再进入下一步。

---

## Step 0 拉取代码

```bash
cd QORE-VLM
git pull
ls scripts/diagnosis/
```

应看到这些文件：

```
__init__.py
answer_diversity_diagnosis.py
complementarity_diagnosis.py
context_dependency_diagnosis.py
diagnosis_io.py
gamma_sweep_diagnosis.py
qubo_objective_diagnosis.py
query_type_diagnosis.py
run_all_diagnosis.sh
```

---

## Step 1 环境检查

```bash
bash scripts/collab/setup_env.sh
```

---

## Step 2 快速测试（约 5 分钟）

```bash
bash scripts/collab/run_phase1_quick.sh
```

---

## Step 3 检查字段（关键，不要跳过）

```bash
python -c "
import json
d = json.load(open('scratch/research/quick_test/experiments/gamma_0.5/result.json'))
s = d['samples'][0]
for k in ['question', 'selected_passages', 'all_candidates', 'qubo']:
    print(k, '=', k in s)
q = s.get('qubo') or {}
print('qubo.a  =', 'a' in q)
print('qubo.b  =', 'b' in q)
print('pool_ranks =', 'pool_ranks' in q)
"
```

七项必须全部为 `True`。任何一项是 `False`，停止，把输出发我。

---

## Step 4 完整实验（约 1.5-2 小时）

```bash
bash scripts/collab/run_phase1_full.sh
```

---

## Step 5 诊断分析（约 15 分钟，仅 CPU）

```bash
bash scripts/diagnosis/run_all_diagnosis.sh
```

---

## Step 6 打包交付

```bash
cd scratch/research
zip -r P1_diagnosis_$(date +%Y%m%d).zip \
    P1_diagnosis/analysis \
    P1_diagnosis/experiments/*/result.json \
    P1_diagnosis/run_summary.json
```

把 zip 发我。

---

## 预期行为（不是故障）

**Step 2 的诊断 4 报错缺 `f1`** — 快速测试不跑 LLM 生成，没有 F1。正式实验（Step 4）开了生成。

**Step 5 的诊断 1 提示「γ 未生效（饱和）」并退出码 2** — 这是脚本主动拦下不可靠的结论，不是崩溃。其余三个诊断不受影响。把报告发我。

**报错信息里写明缺哪个字段** — 例如 `samples lack required field(s) ['qubo']`。这类报错是设计好的，会告诉你该在评测时加哪个参数。

---

## 出错时

```bash
cat scratch/research/P1_diagnosis/experiments/gamma_0.5/stderr.log
cat scratch/research/P1_diagnosis/experiments/gamma_0.5/status.json
```

发这两个文件给我。

**实验超时被杀**（单配置超过 75 分钟）不要直接重跑 —— 超时会触发 2 次重试，白跑三遍要 3.75 小时。先看 stderr.log。

---

## 这次诊断验的是哪几个 idea

| 报告文件 | idea |
|---|---|
| `gamma_sweep.md` | 1 两阶段 QUBO |
| `context_dependency.md` | 4 上下文完整性建模 |
| `complementarity.md` | 6 互补性矩阵 |
| `qubo_objective.md` | 7 Soft QUBO 端到端 |

idea 2（答案多样性）和 idea 3（Query-adaptive γ）在 NQ 上原理上测不了，
脚本保留但不在本次运行 —— 前者需要多答案数据集（NQ 的 gold_answers 是
同一答案的别名集合），后者需要问句类型有真实分布的数据集。
