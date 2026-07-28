# Phase 1: 诊断实验

## 目标

验证 7 个优化 idea 在 NQ-open 数据集上的可行性：
- Idea 1: 两阶段 QUBO（γ sweep 验证）
- Idea 4: 上下文完整性（段落间依赖）
- Idea 6: 互补性矩阵
- Idea 7: QUBO 目标与 F1 一致性

Idea 2（答案多样性）和 Idea 3（Query-adaptive γ）在 NQ 上原理上不可测。

## 工作流程（5 步）

### 1. 环境检查
```bash
cd scripts/collab/p1_diagnosis
bash ../setup_env.sh
```

### 2. 快速验证（可选，验证流程）
```bash
bash run_phase1_quick.sh
# 10 题，~2 分钟，无 LLM 生成
# 检查字段
python ../../diagnosis/check_dump_fields.py --expect_samples 10
```

### 3. 完整实验
```bash
bash run_phase1_full.sh
# 200 题 × 3 gamma，~1.5-2 小时
```

### 4. 运行诊断分析
```bash
bash ../../diagnosis/run_all_diagnosis.sh
# ~15 分钟，纯 CPU，生成 4 份报告
```

### 5. 收集结果
```bash
# 查看时间戳
ls ../../exchange/p1_diagnosis/
# 汇总（<timestamp> 是运行时间）
python collect_p1_results.py <timestamp>
# 提交
git add ../../exchange/p1_diagnosis/<timestamp>
git commit -m "experiment(p1): diagnosis results <timestamp>"
git push
```

## 实验配置

### 快速测试（run_phase1_quick.sh）
- **样本数**: 10
- **LLM 生成**: 关闭（--skip_generation）
- **用途**: 验证流程，不产生诊断结论
- **耗时**: ~2 分钟

### 完整实验（run_phase1_full.sh）
- **样本数**: 200
- **LLM 生成**: 开启（需要 F1）
- **配置**: gamma ∈ {0.0, 0.5, 1.0}
- **耗时**: 单配置 ~30-35 分钟，总计 ~1.5-2 小时

## 输出产物

```
scratch/research/P1_diagnosis/
├── experiments/              # 各实验结果
│   ├── gamma_0.0/result.json + status.json + stderr.log
│   ├── gamma_0.5/
│   └── gamma_1.0/
├── analysis/                 # 四份诊断报告
│   ├── gamma_sweep.md        # Idea 1
│   ├── context_dependency.md # Idea 4
│   ├── complementarity.md    # Idea 6
│   └── qubo_objective.md     # Idea 7
└── run_summary.json
```

收集后提交到：
```
exchange/p1_diagnosis/<timestamp>/
├── README.md                 # 自动生成的结果汇总
├── meta/git_state.txt
├── config/*.yaml
└── analysis/*.md             # 诊断报告（复制自 scratch/）
```

## 故障排查

### 问题 1: 缺少字段错误
```
❌ samples lack required field(s) ['selected_passages']
```
**解决**: 先运行字段检查脚本
```bash
python ../../diagnosis/check_dump_fields.py \
    --results scratch/research/P1_diagnosis/experiments/gamma_0.5/result.json \
    --expect_samples 200
```
退出码 2 = 文件陈旧或不完整，清目录重跑。

### 问题 2: GPU 不可用
```bash
nvidia-smi  # 检查 GPU 状态
```

### 问题 3: 实验超时
单配置超过 75 分钟会被杀。先查看日志：
```bash
cat scratch/research/P1_diagnosis/experiments/gamma_0.5/stderr.log
```
超时会触发 2 次重试，避免白跑 3 遍（3.75 小时）。

### 问题 4: 只想跑少量样本测试
```bash
python ../../tuning/run_tuning_suite.py \
    --config ../../tuning/config/phase1_diagnosis.yaml \
    --override max_samples=50
```

## 高级：自定义参数

直接使用调参框架：
```bash
python ../../tuning/run_tuning_suite.py \
    --config ../../tuning/config/phase1_diagnosis.yaml \
    --override max_samples=100 seed=123
```

详见 `../../tuning/QUICK_START.md`

## 相关文档

- 详细步骤: [PHASE1_STEPS.md](PHASE1_STEPS.md)（历史文档，含更多细节）
- Exchange 说明: `../../exchange/p1_diagnosis/README.md`
- 诊断脚本: `../../diagnosis/run_all_diagnosis.sh`

## 注意事项

1. ⚠️ `run_phase1_quick.sh` 的「无 F1 数据」警告是正常的（skip_generation）
2. ✅ 所有结果存储在 `scratch/research/`（gitignore）
3. ⚠️ Ctrl+C 中断后，被打断的实验不会留下 result.json（会被重跑时清掉）
4. ✅ status.json 必须保留，用于确认实验真跑完

---

**有问题随时联系！**
