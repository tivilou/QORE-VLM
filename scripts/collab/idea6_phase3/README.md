# Idea 6 Phase 3 - 全量验证实验

**目标**: 在完整 validation 集 (3610 samples) 上验证 Idea 6 (Complementarity) 的效果

**日期**: 2026-07-31

---

## 📁 目录内容

### 脚本

- **`run_p3_experiments.sh`** - 一键运行所有 Phase 3 实验
  - 3 个配置（baseline + 2 idea6 变体）
  - 3 个随机种子（42, 43, 44）
  - 总计 9 次运行
  - 预计时间：4.5-9 小时

- **`analyze_p3_results.py`** - 分析实验结果并生成汇总报告
  - 计算每个配置的平均指标
  - 计算相对 baseline 的提升百分比
  - 检查验收标准
  - 评估结果稳定性

- **`package_p3_results.py`** - 打包结果用于 GitHub 提交
  - 压缩 result.json 文件
  - 生成汇总 README
  - 排除大文件

- **`quick_test.sh`** - 快速测试脚本（10 samples）
  - 验证配置文件正确性
  - 验证环境设置
  - 预计时间：5-10 分钟

---

## 🚀 使用方法

### 1. 运行完整实验（推荐）

```bash
cd /home/Q-DUET-VLM/QORE-VLM

# 使用 tmux 运行（推荐，防止断开）
tmux new -s idea6_p3
bash scripts/collab/idea6_phase3/run_p3_experiments.sh

# 分离会话：Ctrl+B, 然后按 D
# 重新连接：tmux attach -t idea6_p3
```

### 2. 快速测试（可选）

```bash
# 先用少量样本测试配置是否正确
bash scripts/collab/idea6_phase3/quick_test.sh
```

### 3. 分析结果

```bash
# 实验完成后分析结果
python scripts/collab/idea6_phase3/analyze_p3_results.py \
    exchange/p3_solver_idea6/YYYYMMDDTHHMMSS
```

### 4. 打包结果

```bash
# 打包用于 GitHub 提交
python scripts/collab/idea6_phase3/package_p3_results.py \
    exchange/p3_solver_idea6/YYYYMMDDTHHMMSS
```

---

## 📊 实验配置

### Baseline
- **配置**: `configs/experiments/baseline_phase3.yaml`
- **参数**: γ=0.5, δ=0.0 (无互补性)
- **期望**: Recall@5 ≈ 0.32

### Idea 6 推荐配置
- **配置**: `configs/experiments/idea6_phase3_recommended.yaml`
- **参数**: γ=0.5, δ=0.1
- **期望**: Recall@5 ≈ 0.44, F1 ≈ 0.51

### Idea 6 最佳配置
- **配置**: `configs/experiments/idea6_phase3_best.yaml`
- **参数**: γ=0.3, δ=0.1
- **期望**: Recall@5 ≈ 0.46, F1 ≈ 0.51

---

## 📂 输出结构

```
exchange/p3_solver_idea6/YYYYMMDDTHHMMSS/
├── README.md                         # 实验报告（由 package 脚本生成）
├── git_commit.txt                    # Git commit hash
├── git_status.txt                    # Git working tree status
├── git_diff.patch                    # 未提交的修改
├── seed_42/
│   ├── baseline/
│   │   ├── result.json               # 完整结果（~300MB）
│   │   └── log.txt                   # 运行日志
│   ├── idea6_recommended/
│   │   ├── result.json
│   │   └── log.txt
│   └── idea6_best/
│       ├── result.json
│       └── log.txt
├── seed_43/
│   └── ...
├── seed_44/
│   └── ...
└── results.zip                       # 打包的结果（由 package 脚本生成）
```

---

## ✅ 验收标准

Phase 3 成功标准：

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

## ⚠️ 注意事项

### 系统要求

- **GPU**: 至少 24GB 显存
- **磁盘**: 至少 10GB 可用空间
- **时间**: 4.5-9 小时（完整实验）

### 运行建议

1. **使用 tmux/screen**
   - 防止 SSH 断开导致实验中断
   - `tmux new -s idea6_p3`

2. **监控进度**
   - 查看日志：`tail -f exchange/p3_solver_idea6/.../seed_42/baseline/log.txt`
   - 检查 GPU：`watch -n 1 nvidia-smi`

3. **磁盘空间**
   - 每个 result.json 约 300MB
   - 总计约 2.7GB（9个文件）
   - 打包后约 500MB

---

## 🐛 故障排查

### 问题 1: CUDA out of memory
**原因**: GPU 显存不足  
**解决**: 
```bash
# 关闭其他占用 GPU 的进程
nvidia-smi
kill -9 <PID>
```

### 问题 2: DPR model 未找到
**原因**: DPR answer scorer 模型未下载  
**解决**:
```bash
# 检查模型是否存在
ls -la /root/QORE-VLM/models/dpr-reader
```

### 问题 3: 运行时间过长
**原因**: 系统负载过高或配置错误  
**解决**:
```bash
# 先用 quick_test.sh 验证
bash scripts/collab/idea6_phase3/quick_test.sh
```

### 问题 4: 脚本权限错误
**原因**: 脚本没有执行权限  
**解决**:
```bash
chmod +x scripts/collab/idea6_phase3/*.sh
```

---

## 📈 Phase 2 回顾（200 samples）

作为参考，Phase 2 的结果：

| 配置 | Recall@5 | F1 | EM | Redundancy |
|------|----------|-----|-----|------------|
| Baseline | 0.3196 | 0.4406 | 0.2500 | 0.8090 |
| Idea 6 推荐 | 0.4454 | 0.5092 | 0.2950 | 0.7876 |
| Idea 6 最佳 | 0.4598 | 0.5101 | 0.3000 | 0.7982 |

**提升**:
- Recall: +39.4% ~ +43.9%
- F1: +15.6% ~ +15.8%
- EM: +18.0% ~ +20.0%

Phase 3 预期保持类似的提升幅度。

---

## 📚 相关文档

- **实验指南**: `../../exchange/IDEA6_PHASE3_GUIDE.md`
- **项目状态**: `../../exchange/PROJECT_STATUS_20260731.md`
- **插件系统**: `../../docs/ENHANCER_PLUGIN_SYSTEM.md`
- **配置文件**: `../../configs/experiments/idea6_phase3_*.yaml`

---

## 🔄 工作流程

```
1. 准备 → 2. 运行 → 3. 监控 → 4. 分析 → 5. 打包 → 6. 提交
   ↓         ↓         ↓         ↓         ↓         ↓
 quick    run_p3    watch     analyze   package    git
 test     exp.sh    logs      results   results    push
```

---

## 📞 支持

如有问题：
1. 查看 `../../exchange/IDEA6_PHASE3_GUIDE.md`
2. 查看实验日志：`*.log.txt`
3. 在 GitHub 提 issue

---

**创建日期**: 2026-07-31  
**负责人**: 师弟  
**状态**: 准备就绪，等待执行

**预祝实验成功！** 🎉
