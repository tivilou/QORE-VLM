# 合作者使用指南

**对象**: 团队内部成员  
**状态**: 🔒 内部协作使用（暂时公开）

本目录包含团队成员使用的快捷脚本。论文发表后可能归档。

---

## 快速开始

### 师弟：Phase 1 诊断实验

**Step 1: 环境检查**
```bash
bash scripts/collab/setup_env.sh
```

**Step 2: 快速测试**（5分钟，验证环境）
```bash
bash scripts/collab/run_phase1_quick.sh
```

**Step 3: 完整实验**（1.5-2 小时，含 LLM 生成）
```bash
bash scripts/collab/run_phase1_full.sh
```

**Step 4: 跑 5 个诊断分析**（约 15 分钟，仅需 CPU）
```bash
bash scripts/diagnosis/run_all_diagnosis.sh
```

**Step 5: 查看结果**
```bash
ls scratch/research/P1_diagnosis/analysis/
#   gamma_sweep.md         γ 对 Recall/冗余度的影响
#   answer_diversity.md    重复证据占掉多少槽位
#   query_type.md          不同查询类型的最优 γ
#   context_dependency.md  段落间依赖是否被破坏
#   qubo_objective.md      QUBO 目标与 F1 的一致性
```

### 两份配置的差异（Step 2 用哪份、Step 3 用哪份）

| | quick_test.yaml | phase1_diagnosis.yaml |
|---|---|---|
| 用途 | Step 2 验流程跑通 | Step 3 拿真结论 |
| 题数 | 10 | 200 |
| LLM 生成 | ❌ 关（省时间） | ✅ 开（诊断需要 F1） |
| 单配置耗时 | ~2 分钟 | ~30-35 分钟 |
| 5 个诊断 | 3 和 5 会**报错**、2 给「待验证」 | 5 个都出结论 |

**Step 2 跑完看到诊断 3、5 报错是正常的**——quick_test 不跑生成所以没有 F1，
那两个诊断需要 F1 做对照。报错信息里会写明缺哪个字段。
**不要因此以为脚本有 bug**，Step 3 用的配置开了生成，5 个都能跑。

### 遇到问题时

诊断脚本缺字段会直接报错并写明缺什么，例如：

```
❌ samples lack required field(s) ['selected_passages'].
   'question'/'selected_passages' 需要评测时加 --dump_passages
   'f1'/'em' 需要评测时不加 --skip_generation
```

**实验超时被杀**（单配置超过 75 分钟）时不要直接重跑——
先看 `scratch/research/P1_diagnosis/experiments/<name>/stderr.log`。
超时会触发 2 次重试，白跑三遍要 3.75 小时。

---

## 高级使用

如果需要自定义参数，直接使用调参框架：

```bash
# 只跑 50 题
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml \
    --override max_samples=50

# 修改其他参数
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml \
    --override max_samples=100 seed=123
```

详见 `scripts/tuning/QUICK_START.md`

---

## 输出位置

所有结果存储在 `scratch/research/`（不推送到 GitHub）：

```
scratch/research/P1_diagnosis/
├── experiments/              # 各实验结果
│   ├── gamma_0.0/
│   ├── gamma_0.5/
│   └── gamma_1.0/
├── analysis/                 # 分析报告
│   ├── gamma_sweep.md        ⭐ 5 份诊断报告
│   ├── answer_diversity.md
│   ├── query_type.md
│   ├── context_dependency.md
│   └── qubo_objective.md
├── package/                  # 打包文件
│   └── P1_diagnosis_*.zip
└── run_summary.json         ⭐ 运行摘要
```

---

## 故障排查

### 问题 1: GPU 不可用
```bash
nvidia-smi
```
如果看到 GPU 信息就没问题，可能是临时问题，重试即可。

### 问题 2: 实验失败
查看错误日志：
```bash
cat scratch/research/P1_diagnosis/experiments/gamma_0.5/stderr.log
```

查看状态：
```bash
cat scratch/research/P1_diagnosis/experiments/gamma_0.5/status.json
```

### 问题 3: 想快速测试
只跑 50 题（15-20分钟）：
```bash
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml \
    --override max_samples=50
```

---

## 注意事项

1. ✅ 所有实验结果存储在 `scratch/research/`（不推送到 GitHub）
2. ✅ 每次实验会生成打包的 zip 文件
3. ✅ 如有问题，发送 stderr.log 或 status.json 给我
4. ✅ 中途可以按 Ctrl+C 停止，已完成的结果会保留

---

## 完成后交付

实验完成后，发给我：
1. **分析报告**: `scratch/research/P1_diagnosis/analysis/` 下全部 .md
2. **打包文件**: `scratch/research/P1_diagnosis/package/*.zip`
3. **你的决策**: 根据报告，你认为下一步应该做什么？

---

**有问题随时联系！**
