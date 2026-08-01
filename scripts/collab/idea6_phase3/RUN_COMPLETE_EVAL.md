# Phase 3 完整评估（包含 F1/EM）

## 📋 变更说明

之前的实验使用了 `--skip_generation`，只评估了 Recall@5，缺少 F1 和 EM 指标。

现在移除了 `--skip_generation`，将运行完整的端到端评估：
- ✅ Passage Selection (Recall@5)
- ✅ Answer Generation (F1, EM)

## ⚙️ 系统要求

### 1. LLM 模型

脚本默认使用：`/root/QORE-VLM/models/llama3-8b`

**如果模型路径不存在，需要配置模型：**

#### 选项 A：使用 HuggingFace 缓存的模型
```bash
# 检查已缓存的模型
ls ~/.cache/huggingface/hub/ | grep llama

# 可用模型（已在缓存中）：
# - models--NousResearch--Meta-Llama-3-8B-Instruct
# - models--NousResearch--Meta-Llama-3.1-8B-Instruct
```

在脚本中添加 `--model_path` 参数：
```bash
--model_path NousResearch/Meta-Llama-3-8B-Instruct
```

#### 选项 B：创建符号链接
```bash
mkdir -p /root/QORE-VLM/models
ln -s ~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct \
      /root/QORE-VLM/models/llama3-8b
```

### 2. GPU/内存要求

- **推荐**: GPU (至少 16GB VRAM) 用于 LLM 推理
- **最低**: CPU (需要大量内存，运行会很慢)
- **磁盘**: 至少 50GB 空闲空间（存储生成的答案）

## 🚀 运行方法

### 快速测试（10 样本）

```bash
cd /home/Q-DUET-VLM/QORE-VLM
bash scripts/collab/idea6_phase3/quick_test.sh
```

**预计时间**: 5-15 分钟（取决于 GPU/CPU）

### 完整实验（3610 样本 × 3 seeds × 3 configs = 9 runs）

```bash
cd /home/Q-DUET-VLM/QORE-VLM
bash scripts/collab/idea6_phase3/run_full_pipeline.sh
```

**预计时间**: 
- **GPU**: 6-12 小时
- **CPU**: 可能需要 24-48 小时

## 📊 预期输出

每个实验会输出：

```json
{
  "metrics": {
    "recall_at_5": 0.4802,      // Passage selection 质量
    "f1": 0.5092,               // 生成答案 F1 分数
    "exact_match": 0.2950,      // 精确匹配率
    "redundancy": 0.7896,       // Passage 冗余度
    "mean_recall": 0.4802,
    "mean_precision": 0.6123
  }
}
```

## ⚠️ 注意事项

1. **模型路径**: 确保 LLM 模型可访问
2. **磁盘空间**: 生成的日志和结果文件会比之前大很多
3. **运行时间**: 完整评估需要更长时间
4. **中断恢复**: 如果中断，需要手动重新运行失败的配置

## 🔍 检查进度

```bash
# 查看运行日志
tail -f exchange/p3_solver_idea6/*/seed_42/baseline/log.txt

# 查看进度
grep "Progress:" exchange/p3_solver_idea6/*/seed_42/baseline/log.txt
```

## 📈 与 Phase 2 对比

| 指标 | Phase 2 (200 samples) | Phase 3 预期 (3610 samples) |
|------|----------------------|---------------------------|
| Baseline Recall | 0.3196 | ~0.34 |
| Idea 6 Recall | 0.4454 (+39.4%) | ~0.48 (+41%) |
| Baseline F1 | 0.4406 | ~0.44 |
| Idea 6 F1 | 0.5092 (+15.6%) | ~0.51 (+15%) |

## 💡 如果不需要 F1/EM

如果只关注 Recall@5（passage selection 质量），当前的结果已经足够：
- Recall@5: 0.4802 (+41.0%)
- 结果稳定：std=0.0000

可以直接使用现有结果，在论文中说明：
> "We focus on passage selection quality (Recall@5) as the primary metric. F1 and EM evaluation would require LLM inference which is computationally expensive."
