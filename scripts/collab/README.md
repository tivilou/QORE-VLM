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

**Step 3: 完整实验**（1-2小时）
```bash
bash scripts/collab/run_phase1_full.sh
```

**Step 4: 查看结果**
```bash
cat scratch/research/P1_diagnosis/analysis/analysis.md
```

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
│   └── analysis.md          ⭐ 重点查看
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
1. **分析报告**: `scratch/research/P1_diagnosis/analysis/analysis.md`
2. **打包文件**: `scratch/research/P1_diagnosis/package/*.zip`
3. **你的决策**: 根据报告，你认为下一步应该做什么？

---

**有问题随时联系！**
