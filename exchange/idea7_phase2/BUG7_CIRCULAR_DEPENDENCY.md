# 🐛 Bug 7: Circular Dependency - Training on QORE's Output

**发现时间**: 2026-07-31  
**严重程度**: P0 - Critical  
**状态**: ✅ 已修复

---

## 📋 问题描述

### 症状
- Epoch 1 就达到 Recall = 1.0000
- 100 个 epoch 都保持 Recall = 1.0000
- 训练 Loss 非常小（-0.06 左右）

### 初步诊断误判
最初怀疑是数据泄露（query_embedding = gold passages 均值），但诊断脚本显示：
- ✅ query_embedding 存在
- ✅ 相似度正常（0.69-0.76，不是 0.999）
- ✅ 没有数据泄露

---

## 🔍 真正的根本原因

**用 QORE 优化后的结果来训练 Idea 7！**

### 错误的流程

```
1. 检索 top-50 候选 passages
   └─> [P1, P2, P3, ..., P50] (包含一些 gold)

2. QORE 方法优化选择 K=5 个
   └─> [P1*, P7*, P12*, P23*, P45*] (QORE 已经优化过，gold 排在前面)

3. 用这 5 个训练 Idea 7 ❌
   └─> 学习目标：从这 5 个"完美"的结果中选择
   └─> 结果：Recall=1.0 因为数据已经被优化过了
```

### 为什么这是循环依赖？

**Idea 7 的目标**：学习如何从原始检索结果中选择最优子集  
**实际训练数据**：QORE 已经选择好的最优子集

这就像：
- ❌ 用高考状元的试卷训练"如何提高成绩"
- ✅ 应该用普通学生的试卷，让模型学习如何改进

### 数据分析证据

检查 `result(1).json` 前10个样本：

```
样本 0: Gold 排名 [0, 1] - 前2个都是 gold！
样本 1: Gold 排名 [0, 3] - 前4个有2个 gold
样本 2: Gold 排名 [0, 1] - 前2个都是 gold！
样本 3: Gold 排名 [0, 1, 2, 3, 4] - 全部5个都是 gold！
...
```

**所有 gold passages 都排在 top-5**，因为这是 QORE 优化后的结果！

---

## ✅ 修复方案

### 正确的流程

```
1. 检索 top-50 候选 passages
   └─> [P1, P2, P3, ..., P50] (原始检索结果)

2. 记录 gold 在这 50 个中的位置
   └─> gold_indices = [3, 7, 15, 23, 41]

3. 用这 50 个训练 Idea 7 ✅
   └─> 学习目标：从 50 个候选中学习选出 gold
   └─> 结果：模型真正学习选择策略
```

### 代码修改

#### 1. `eval_rag_refactored.py` - 添加 `retrieved_passages` 字段

**位置**: Line ~453

```python
# For Idea 7 training: dump ALL retrieved passages with embeddings
# This is expensive (~50x larger JSON) but necessary for training a
# learnable selector that needs to learn from the full candidate pool.
if retrieved_texts is not None and retrieved_embs is not None:
    retrieved_passages_full = [
        {
            "text": retrieved_texts[j],
            "score": float(cand_scores[j]) if cand_scores is not None else None,
            "is_gold": j in gold_in_retrieved,
            "selected": j in sel_set,
            "embedding": retrieved_embs[j].tolist(),
            "retrieved_rank": int(j),
        }
        for j in range(n_cand)
    ]
```

**输出字段**:
```json
{
  "query_embedding": [...],
  "selected_passages": [...],        // QORE 选择的 5 个
  "retrieved_passages": [...],       // 原始检索的 50 个 (新增)
  "all_candidates": [...]            // 元数据
}
```

#### 2. `train_soft_qubo_simple.py` - 优先使用 `retrieved_passages`

**位置**: Line ~95

```python
# Priority: use "retrieved_passages" (all candidates) over "selected_passages"
passages_key = None
if "retrieved_passages" in item and item["retrieved_passages"]:
    passages_key = "retrieved_passages"  # 优先使用原始检索结果
elif "retrieved" in item:
    passages_key = "retrieved"
elif "selected_passages" in item:
    passages_key = "selected_passages"   # 降级到 QORE 输出（不推荐）
```

---

## 📊 预期效果

### 修复前（Bug 7）
```
Epoch   1: Recall=1.0000, Loss=-0.05
Epoch  50: Recall=1.0000, Loss=-0.06
Epoch 100: Recall=1.0000, Loss=-0.05
```
→ 数据太简单，模型没有学到东西

### 修复后（正确训练）
```
Epoch   1: Recall=0.35-0.50, Loss=-0.20
Epoch  50: Recall=0.50-0.65, Loss=-0.35
Epoch 100: Recall=0.55-0.70, Loss=-0.40
```
→ 逐渐学习，真正优化

### 关键指标变化

| 指标 | Bug 7 (错误) | 修复后 (预期) |
|------|-------------|--------------|
| 训练数据大小 | 5 passages | 50 passages |
| Epoch 1 Recall | 1.0 | 0.35-0.50 |
| 最终 Recall | 1.0 (假象) | 0.55-0.70 (真实) |
| Loss 范围 | -0.05~-0.08 | -0.20~-0.40 |
| Gold 在 top-5 率 | 100% | 40-50% (取决于方法有效性) |

---

## 🔧 如何重新运行实验

### Step 1: 拉取修复
```bash
cd /path/to/QORE-VLM
git pull  # 应该包含 Bug 7 修复
```

### Step 2: 清理旧数据
```bash
rm -rf exchange/idea7_phase2/*
```

### Step 3: 重新运行
```bash
bash scripts/collab/idea7_phase2/run_idea7_phase2.sh
```

### Step 4: 验证修复
```bash
# 检查新生成的 result.json
python3 << 'EOF'
import json
with open('exchange/idea7_phase2/TIMESTAMP/data_prep/result.json') as f:
    data = json.load(f)
    sample = data['samples'][0]
    
    print(f"✅ retrieved_passages: {len(sample.get('retrieved_passages', []))} 个")
    print(f"   selected_passages: {len(sample.get('selected_passages', []))} 个")
    
    if len(sample.get('retrieved_passages', [])) >= 40:
        print("\n✅ 修复成功！使用了完整检索结果")
    else:
        print("\n❌ 还是只有少量 passages")
EOF
```

### Step 5: 检查训练曲线
正常训练应该看到：
- ✅ Epoch 1: Recall < 0.6
- ✅ 逐渐提升
- ✅ Loss 在 -0.20 到 -0.40

---

## 📝 相关 Bug

- **Bug 6**: 数据泄露（query_emb = gold mean）→ 已修复 c8c3c6a
- **Bug 7**: 循环依赖（训练 QORE 输出）→ 本次修复

这两个 bug 症状相同（Recall=1.0 from epoch 1），但根本原因不同：
- Bug 6: 作弊的 query
- Bug 7: 作弊的 passages

---

## ✅ 验收标准

实验成功的标志：
1. ✅ result.json 包含 `retrieved_passages` 字段（~50 个）
2. ✅ 训练日志显示加载了 50 个候选（不是 5 个）
3. ✅ Epoch 1 的 Recall < 0.6
4. ✅ Recall 随训练逐渐提升
5. ✅ Loss 在 -0.20 到 -0.40 之间

---

## 💡 教训

**数据准备的质量直接决定训练结果的有效性。**

在构建训练数据时，必须确保：
1. 训练数据反映真实任务难度
2. 不能用优化后的结果训练优化算法
3. 诊断工具要覆盖数据质量检查，不仅仅是格式检查
