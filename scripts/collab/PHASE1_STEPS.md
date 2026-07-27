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
check_dump_fields.py
complementarity_diagnosis.py
context_dependency_diagnosis.py
diagnosis_io.py
gamma_sweep_diagnosis.py
qubo_objective_diagnosis.py
query_type_diagnosis.py
run_all_diagnosis.sh
```

少文件说明没拉到，`git log --oneline -1` 看一下 HEAD。

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
python scripts/diagnosis/check_dump_fields.py --expect_samples 10
```

七项必须全部 ✅，最后一行是「✅ 七项齐全，可以进 Step 4」。

退出码的含义：

| 码 | 含义 | 怎么办 |
|---|---|---|
| 0 | 字段齐全 | 进 Step 4 |
| 1 | 字段缺失 / 文件不可用 | 停下，把整段输出发我 |
| 2 | 文件是陈的或不完整 | 按提示清目录重跑 Step 2 |

**退出码 2 最常见**，意思是你读到的不是这次跑出来的文件。脚本会先报来源信息
（写入时间、当前 HEAD、`config.dump_passages`、样本数），再看字段 —— 因为
「文件是上一轮的」和「dump 真的坏了」在只看字段时输出一模一样，分不出来。

遇到 `config 里没有 'dump_passages' 这个键`，就是这个情况：

```bash
git pull
rm -rf scratch/research/quick_test
bash scripts/collab/run_phase1_quick.sh
```

---

## Step 4 完整实验（约 1.5-2 小时）

```bash
bash scripts/collab/run_phase1_full.sh
```

跑完再确认一次字段（这次是 200 题的产物）：

```bash
python scripts/diagnosis/check_dump_fields.py \
    --results scratch/research/P1_diagnosis/experiments/gamma_0.5/result.json \
    --expect_samples 200
```

---

## Step 5 诊断分析（约 15 分钟，仅 CPU）

```bash
bash scripts/diagnosis/run_all_diagnosis.sh
```

开跑前它会检查每个 `gamma_*/status.json`。有实验是 timeout/failed 就直接停 ——
那些目录里的 `result.json` 可能是上一轮的，拿它出的报告数字看着合理但无效。

---

## Step 6 打包交付

Step 4 的 `post_process` 已经自动打好包了：

```bash
ls scratch/research/P1_diagnosis/package/
```

但那个包在 Step 5 之前生成，**不含四份诊断报告**。Step 5 跑完后补一个：

```bash
cd scratch/research
zip -r P1_diagnosis_$(date +%Y%m%d).zip \
    P1_diagnosis/analysis \
    P1_diagnosis/experiments/*/result.json \
    P1_diagnosis/experiments/*/status.json \
    P1_diagnosis/run_summary.json
cd ../..
```

把 zip 发我。`status.json` 一起带上，我这边能确认每个实验是真跑完的。

---

## 预期行为（不是故障）

**Step 2 打印「⚠️ 无 F1 数据（--skip_generation），结论仅覆盖检索侧」** —— 快速测试不跑
LLM 生成，没有 F1。Step 2 只跑 γ sweep 这一个分析（不是完整的四个诊断），它容忍缺
F1，照样退出 0。正式实验（Step 4）开了生成，四个诊断都有 F1。

**Step 5 的 `[1/4] γ sweep` 提示「γ 未生效（饱和）」并退出码 2** —— 这是脚本主动拦下
不可靠的结论，不是崩溃。其余三个诊断不受影响。把报告发我。

**Step 5 里 idea 2 和 idea 3 不出现** —— 这两个在 NQ 上原理上测不了，见文末表格。
driver 只跑四个，标签是 `[1/4]` 到 `[4/4]`。

**报错信息里写明缺哪个字段** —— 例如 `samples lack required field(s) ['qubo']`。这类
报错是设计好的，会告诉你该在评测时加哪个参数。

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

driver 里的标签与报告文件的对应：

| driver 标签 | 报告文件 | idea |
|---|---|---|
| `[1/4]` | `gamma_sweep.md` | 1 两阶段 QUBO |
| `[2/4]` | `context_dependency.md` | 4 上下文完整性建模 |
| `[3/4]` | `complementarity.md` | 6 互补性矩阵 |
| `[4/4]` | `qubo_objective.md` | 7 Soft QUBO 端到端 |

idea 2（答案多样性）和 idea 3（Query-adaptive γ）在 NQ 上原理上测不了，
脚本保留但不在本次运行 —— 前者需要多答案数据集（NQ 的 gold_answers 是
同一答案的别名集合），后者需要问句类型有真实分布的数据集。
