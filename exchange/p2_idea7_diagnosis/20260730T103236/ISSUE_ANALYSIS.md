# Idea 7 重新评估结果分析

## ⚠️ 发现的问题

**结果数据与 Phase 1 完全相同，这不正常！**

### 对比数据

| 指标 | Phase 1 (2026-07-27) | Phase 2 (2026-07-30) | 状态 |
|------|---------------------|---------------------|------|
| 样本数 | 150 | 150 | ⚠️ 应该是 200 |
| QUBO 最优 Recall | 0.5909 | 0.5909 | ❌ 完全相同 |
| Oracle Recall | 0.8913 | 0.8913 | ❌ 完全相同 |
| Gap | 0.3004 | 0.3004 | ❌ 完全相同 |
| 命中率 | 52/150 (34.7%) | 52/150 (34.7%) | ❌ 完全相同 |

### 问题诊断

检查发现 **result.json 文件缺失**：

```bash
$ ls exchange/p2_idea7_diagnosis/20260730T103236/
README.md  qubo_diagnosis.md
# ❌ 缺少 result.json！
```

这说明：
1. **步骤 1（RAG 评估）失败了** - 没有生成 result.json
2. 步骤 2（诊断）可能：
   - 使用了旧的 Phase 1 结果文件（来自其他路径）
   - 或者脚本有 bug 导致错误处理不当

## 需要重新运行

### 步骤 1: 检查评估是否真的运行了

```bash
# 查看运行日志（如果有保存）
# 或者检查 GPU 使用记录

# 检查是否有其他时间戳的目录
ls -la exchange/p2_idea7_diagnosis/
```

### 步骤 2: 重新运行完整实验

```bash
cd ~/QORE-VLM
git pull

# 清理旧的不完整结果
rm -rf exchange/p2_idea7_diagnosis/20260730T103236

# 重新运行（确保有 GPU 和足够的磁盘空间）
bash scripts/collab/p2_idea7_diagnosis/run_idea7_diagnosis.sh
```

### 步骤 3: 验证结果完整性

运行完成后，检查：

```bash
RESULT_DIR=exchange/p2_idea7_diagnosis/<new_timestamp>

# 1. 确认三个文件都存在
ls -lh $RESULT_DIR/
# 应该看到:
#   README.md
#   qubo_diagnosis.md
#   result.json  ← 重要！约 4MB

# 2. 检查 result.json 的样本数
python3 -c "import json; data=json.load(open('$RESULT_DIR/result.json')); print(f'样本数: {len(data[\"samples\"])}')"
# 应该输出: 样本数: 200

# 3. 检查诊断报告的数值是否变化
head -20 $RESULT_DIR/qubo_diagnosis.md | grep "样本数\|平均 Recall\|平均差距"
# 样本数应该是 200（不是 150）
# 数值应该与 Phase 1 不同
```

## 常见失败原因

### 1. GPU 内存不足
```bash
# 症状：CUDA out of memory
# 解决：减少样本数
nano scripts/collab/p2_idea7_diagnosis/run_idea7_diagnosis.sh
# 修改: SAMPLES=100
```

### 2. 磁盘空间不足
```bash
# 检查磁盘空间
df -h
# result.json 约 4-5 MB，需要足够空间
```

### 3. DPR 模型缺失
```bash
# 检查模型是否存在
ls -la ~/models/dpr/
# 或检查 Hugging Face 缓存
ls -la ~/.cache/huggingface/
```

### 4. Corpus 数据问题
```bash
# 检查 wiki_dpr corpus 是否可用
ls -la ~/.cache/huggingface/datasets/
```

## 预期的正确结果

实现 Idea 6 后，**理论上应该看到**：

- **QUBO 最优 Recall 提升**（因为互补性改善了选择质量）
- **Oracle Recall 可能也提升**（候选池质量更好）
- **Gap 可能缩小**（如果 QUBO 提升 > Oracle 提升）
- **样本数应该是 200**（不是 150）

如果重新运行后仍然是相同数值，那可能意味着：
- Idea 6 的互补性改进主要体现在最终 F1 上，而不是 QUBO 层面的选择
- 需要更深入的分析来理解这个现象

---

**请师弟重新运行实验，并确保 result.json 文件正确生成！**
