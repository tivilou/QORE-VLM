# Idea 6 Phase 3 实验指南

**时间**: 2026-07-31  
**负责人**: 师弟  
**状态**: 准备就绪，等待执行

---

## 📋 实验目标

在完整 validation 集 (3610 samples) 上验证 Idea 6 的效果，使用 3 个随机种子确保结果稳定性。

---

## ✅ Phase 2 验证结果回顾

| 指标 | Baseline | Idea 6 推荐 (γ=0.5, δ=0.1) | Idea 6 最佳 (γ=0.3, δ=0.1) |
|------|----------|--------------------------|--------------------------|
| **Recall@5** | 0.3196 | 0.4454 (+39.4%) | 0.4598 (+43.9%) |
| **F1** | 0.4406 | 0.5092 (+15.6%) | 0.5101 (+15.8%) |
| **EM** | 0.2500 | 0.2950 (+18.0%) | 0.2950 (+18.0%) |
| **Redundancy** | 0.8090 | 0.7876 (-2.6%) | 0.7982 (-1.3%) |

**Phase 2 样本数**: 200  
**Phase 3 样本数**: 3610 (18倍)

---

## 🎯 实验配置

### 配置 1: Baseline
- **文件**: `configs/experiments/baseline_phase3.yaml`
- **参数**: γ=0.5, δ=0 (无互补性)
- **预期**: Recall@5 ≈ 0.32

### 配置 2: Idea 6 推荐 (⭐ 推荐)
- **文件**: `configs/experiments/idea6_phase3_recommended.yaml`
- **参数**: γ=0.5, δ=0.1
- **预期**: Recall@5 ≈ 0.44, F1 ≈ 0.51

### 配置 3: Idea 6 最佳
- **文件**: `configs/experiments/idea6_phase3_best.yaml`
- **参数**: γ=0.3, δ=0.1
- **预期**: Recall@5 ≈ 0.46, F1 ≈ 0.51

### 随机种子
- seed=42, 43, 44

---

## 🚀 运行方法

### 方法 1: 使用一键脚本 (推荐)

```bash
cd /home/Q-DUET-VLM/QORE-VLM

# 运行全部实验 (3 configs × 3 seeds = 9 runs)
bash scripts/collab/run_p3_experiments.sh
```

**预计时间**: 
- 单次运行: ~30-60 分钟 (取决于硬件)
- 总计: ~4.5-9 小时

**输出**: `exchange/p3_solver_idea6/YYYYMMDDTHHMMSS/`

---

### 方法 2: 手动运行单个配置

```bash
cd /home/Q-DUET-VLM/QORE-VLM

# Baseline
python -m scripts.rag.eval.eval_rag_refactored \
    --config configs/experiments/baseline_phase3.yaml \
    --seed 42 \
    --output_dir exchange/p3_manual/baseline_seed42

# Idea 6 推荐
python -m scripts.rag.eval.eval_rag_refactored \
    --config configs/experiments/idea6_phase3_recommended.yaml \
    --seed 42 \
    --output_dir exchange/p3_manual/idea6_recommended_seed42

# Idea 6 最佳
python -m scripts.rag.eval.eval_rag_refactored \
    --config configs/experiments/idea6_phase3_best.yaml \
    --seed 42 \
    --output_dir exchange/p3_manual/idea6_best_seed42
```

---

## 📊 结果分析

### 查看结果

```bash
# 查看输出目录
ls -la exchange/p3_solver_idea6/YYYYMMDDTHHMMSS/seed_*/

# 每个目录包含:
# - baseline/result.json
# - idea6_recommended/result.json
# - idea6_best/result.json
```

### 提取关键指标

```python
import json

with open("exchange/p3_solver_idea6/.../seed_42/idea6_recommended/result.json") as f:
    data = json.load(f)
    
print(f"Recall@5: {data['metrics']['recall_at_5']}")
print(f"F1: {data['metrics']['f1']}")
print(f"EM: {data['metrics']['exact_match']}")
print(f"Redundancy: {data['metrics']['redundancy']}")
```

---

## ✅ 验收标准

### Phase 3 成功标准

1. **Recall@5 提升 ≥ 35%**
   - Baseline: ~0.32
   - Target: ≥ 0.43

2. **F1 提升 ≥ 10%**
   - Baseline: ~0.44
   - Target: ≥ 0.48

3. **结果稳定性**
   - 3 个种子的标准差 < 0.02

4. **冗余度下降**
   - Redundancy 下降或持平

---

## 📦 提交结果

### 需要提交的文件

```
exchange/p3_solver_idea6/YYYYMMDDTHHMMSS/
├── README.md                    # 实验报告
├── git_commit.txt               # Git commit hash
├── git_status.txt               # Git status
├── seed_42/
│   ├── baseline/result.json
│   ├── idea6_recommended/result.json
│   └── idea6_best/result.json
├── seed_43/
│   └── ...
└── seed_44/
    └── ...
```

### 不提交的文件

- `result.json` 的详细内容（太大）
- 改为打包为 `.zip` 或只提交汇总

---

## ⚠️ 注意事项

1. **确保 GPU 可用**
   - 检查: `nvidia-smi`
   - 需要至少 24GB 显存

2. **确保磁盘空间充足**
   - 每个 result.json 约 200-500MB
   - 总计需要约 5-10GB

3. **建议使用 tmux/screen**
   ```bash
   tmux new -s idea6_p3
   bash scripts/collab/run_p3_experiments.sh
   # Ctrl+B, D to detach
   ```

4. **监控进度**
   ```bash
   tmux attach -t idea6_p3
   # 或查看日志
   tail -f exchange/p3_solver_idea6/.../seed_42/baseline/log.txt
   ```

---

## 🐛 故障排查

### 问题 1: CUDA out of memory
**解决**: 减少 batch size 或使用 CPU

### 问题 2: DPR model 加载失败
**解决**: 检查 `/root/QORE-VLM/models/dpr-reader` 是否存在

### 问题 3: 运行时间过长
**解决**: 先用 `max_samples: 100` 测试，确认无误后再跑全量

---

## 📞 联系方式

如有问题，请在 GitHub 提 issue 或联系项目负责人。

---

**祝实验顺利！** 🎉
