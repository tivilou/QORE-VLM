# 🚨 Idea 7 Phase 2 数据泄露问题分析

**时间**: 2026-07-31  
**状态**: ❌ 实验结果无效 - 数据泄露未解决

---

## 📊 问题症状

### 异常的训练曲线
- **Epoch 1**: Recall = 1.0000 (100%)
- **Epoch 100**: Recall = 1.0000 (100%)
- **平均 Loss**: -0.064 (接近 0，异常小)

### 正常训练应该是
- **Epoch 1**: Recall ~ 0.3-0.5 (随机初始化)
- **逐渐提升**: 0.3 → 0.5 → 0.6 → 0.7
- **Loss 范围**: -0.3 到 -0.5 (更负)

---

## 🔍 根本原因

**数据泄露**: 训练数据中的 `query_embedding` 实际上是 gold passages embeddings 的平均值。

### 为什么这是作弊？

在 RAG 中：
- **正常任务**: 给定 question embedding，从 50 个候选 passages 中选出相关的
- **泄露任务**: 给定 gold answer embeddings 的均值，选出与它最相似的 passages（太简单了！）

这就像：
- **正常考试**: "谁是美国第一任总统？"
- **泄露考试**: "找出最接近 'George Washington' 的选项" (直接给答案了！)

---

## 🐛 Bug 历史

### Bug 6 (c8c3c6a) - 已修复但未生效

**修复内容**:
1. `eval_rag_refactored.py`: 添加 `"query_embedding": query_emb.tolist()`
2. `train_soft_qubo_simple.py`: 使用 `sample["query_embedding"]` 而非 `passage_embs[gold_indices].mean()`

**问题**: 师弟的实验 (commit 17816db → 48068c9) 没有使用修复后的代码或数据。

---

## ❓ 可能的原因

### 原因 1: 数据未重新生成
- data_prep 目录下**缺少 result.json**
- 训练时可能使用了旧的缓存数据
- 旧数据中没有正确的 query_embedding

### 原因 2: eval 步骤未正确执行
- `--dump_passages` 标志可能未生效
- query_embedding 未正确输出到 result.json

### 原因 3: 代码版本不一致
- 师弟的服务器可能使用了旧版本代码
- 或者本地有未提交的修改

---

## ✅ 验证方法

### 方法 1: 检查 result.json
```bash
python scripts/collab/idea7_phase2/diagnose_data_leakage.py \
    /path/to/result.json
```

该脚本会：
1. 检查是否包含 query_embedding
2. 检查 query_embedding 是否等于 gold passages 均值
3. 计算相似度（>0.999 表示泄露）

### 方法 2: 检查训练数据加载
```python
import json
import numpy as np

with open('result.json') as f:
    data = json.load(f)
    samples = data['samples'] if 'samples' in data else data

# 检查第一个样本
sample = samples[0]
if 'query_embedding' not in sample:
    print("❌ 缺少 query_embedding")
else:
    query_emb = np.array(sample['query_embedding'])
    gold_passages = [p for p in sample['selected_passages'] 
                     if p.get('is_gold', False)]
    gold_embs = [np.array(p['embedding']) for p in gold_passages]
    gold_mean = np.mean(gold_embs, axis=0)
    
    cos_sim = np.dot(query_emb, gold_mean) / \
              (np.linalg.norm(query_emb) * np.linalg.norm(gold_mean))
    
    if cos_sim > 0.999:
        print(f"🚨 数据泄露！相似度: {cos_sim:.6f}")
    else:
        print(f"✅ 正常数据，相似度: {cos_sim:.4f}")
```

---

## 🔧 修复步骤

### Step 1: 确认代码是最新版本
```bash
cd /path/to/QORE-VLM
git pull
git log --oneline -3

# 应该看到:
# 48068c9 results: Idea 7 Phase 2 training results
# 17816db local: comprehensive experiment setup
# c8c3c6a fix(idea7): Fix data leakage  <-- 关键修复
```

### Step 2: 完全重新运行实验
```bash
# 清理旧数据
rm -rf exchange/idea7_phase2/20260731T111232/

# 重新运行（会自动生成新的 result.json）
bash scripts/collab/idea7_phase2/run_idea7_phase2.sh
```

### Step 3: 验证新生成的数据
```bash
# 查找最新的 result.json
find exchange/idea7_phase2 -name "result.json" -type f -exec ls -lh {} \;

# 诊断
python scripts/collab/idea7_phase2/diagnose_data_leakage.py \
    exchange/idea7_phase2/TIMESTAMP/data_prep/result.json
```

### Step 4: 检查训练结果
正常的训练曲线应该是：
```
Epoch   1/100 | Recall=0.35-0.50 | Loss=-0.15 to -0.25
Epoch  10/100 | Recall=0.45-0.60 | Loss=-0.25 to -0.35
Epoch  50/100 | Recall=0.50-0.65 | Loss=-0.30 to -0.40
Epoch 100/100 | Recall=0.55-0.70 | Loss=-0.35 to -0.45
```

**如果还是 Recall=1.0 从 epoch 1 开始 → 数据泄露仍然存在！**

---

## 📝 当前状态

- ❌ 实验 20260731T111232 的结果**无效**（数据泄露）
- ⏳ 需要重新运行实验
- 📌 优先级：**P0 - 阻塞 Phase 3**

---

## 🎯 成功标准

实验成功的标志：
1. ✅ Epoch 1 的 Recall < 0.6
2. ✅ Recall 随训练逐渐提升
3. ✅ Loss 在 -0.3 到 -0.5 之间
4. ✅ 最终 Recall 在 0.50-0.70 之间（取决于方法有效性）

---

## 📞 需要协助

如果问题持续存在，请提供：
1. 完整的 result.json 文件（前2个样本）
2. eval.log 完整日志
3. 运行 `run_idea7_phase2.sh` 的完整输出
