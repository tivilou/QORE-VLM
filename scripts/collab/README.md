# 合作者使用指南

**对象**: 团队内部成员  
**状态**: 🔒 内部协作使用（暂时公开）

本目录包含团队成员使用的快捷脚本。论文发表后可能归档。

---

## 快速开始

### 师弟：Phase 1 诊断实验

逐步的完整流程看 **`PHASE1_STEPS.md`**（含字段检查、预期行为、出错怎么办）。
这里只列命令：

```bash
bash scripts/collab/setup_env.sh          # 环境检查
bash scripts/collab/run_phase1_quick.sh   # 快速测试（10 题，验流程）
python scripts/diagnosis/check_dump_fields.py --expect_samples 10   # 必须过
bash scripts/collab/run_phase1_full.sh    # 完整实验（200 题，1.5-2h）
bash scripts/diagnosis/run_all_diagnosis.sh                        # 四个诊断（~15min, CPU）
```

产出的四份报告：

```bash
ls scratch/research/P1_diagnosis/analysis/
#   gamma_sweep.md         idea 1  γ 对 Recall/冗余度的影响
#   context_dependency.md  idea 4  段落间依赖是否被破坏
#   complementarity.md     idea 6  互补性矩阵
#   qubo_objective.md      idea 7  QUBO 目标与 F1 的一致性
```

idea 2（答案多样性）和 idea 3（Query-adaptive γ）在 NQ 上原理上测不了，
脚本保留但不在 driver 里跑。原因见 `run_all_diagnosis.sh` 头部注释。

### 两份配置的差异

| | quick_test.yaml | phase1_diagnosis.yaml |
|---|---|---|
| 用途 | 验流程跑通 | 拿真结论 |
| 题数 | 10 | 200 |
| LLM 生成 | ❌ 关（省时间） | ✅ 开（诊断需要 F1） |
| 单配置耗时 | ~2 分钟计算 + 索引加载 | ~30-35 分钟 |
| post_process | 只跑 γ sweep 一个分析 | 只跑 γ sweep（四个诊断靠 driver） |

**quick_test 跑完会打印「⚠️ 无 F1 数据（--skip_generation）」—— 这是正常的。**
它不跑生成所以没 F1，而 γ sweep 容忍这个，照样退出 0。
10 题 + 无 F1 **不足以支撑任何诊断结论**，只能证明流程通。

### 遇到问题时

诊断脚本缺字段会直接报错并写明缺什么，例如：

```
❌ samples lack required field(s) ['selected_passages'].
   'question'/'selected_passages' 需要评测时加 --dump_passages
   'f1'/'em' 需要评测时不加 --skip_generation
```

**字段全缺 / 看着像什么都没 dump 出来** —— 先跑这个，它会先拥源再看字段：

```bash
python scripts/diagnosis/check_dump_fields.py \
    --results scratch/research/P1_diagnosis/experiments/gamma_0.5/result.json \
    --expect_samples 200
```

退出码 2 = 文件是陈的或不完整（不是这次跑出来的），按提示清目录重跑。

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
│   ├── gamma_0.0/            #   result.json + status.json + std*.log
│   ├── gamma_0.5/
│   └── gamma_1.0/
├── analysis/                 # 四份诊断报告
│   ├── gamma_sweep.md        ⭐ idea 1
│   ├── context_dependency.md    idea 4
│   ├── complementarity.md       idea 6
│   └── qubo_objective.md        idea 7
├── package/                  # 打包文件（post_process 自动生成，不含诊断报告）
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
4. ⚠️ 中途 Ctrl+C 停掉后，已完成的结果会保留，但**被打断的那个实验不会留下
   result.json**（runner 开跑前就删掉旧产物了）。重跑时它会重新生成。

---

## 完成后交付

实验完成后，发给我：
1. **分析报告**: `scratch/research/P1_diagnosis/analysis/` 下全部 .md
2. **实验产物**: 各 `experiments/*/result.json` 和 `status.json`
   （status.json 一定带上，我这边靠它确认每个实验是真跑完的）
3. **你的决策**: 根据报告，你认为下一步应该做什么？

---

**有问题随时联系！**
