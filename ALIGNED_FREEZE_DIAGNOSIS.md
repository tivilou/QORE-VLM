# Aligned 模式卡死问题诊断报告

## 问题定位

### 根本原因
**aligned 模式在 `_sample_distractors` 方法中使用流式迭代加载 wiki_dpr 数据集，导致看起来像卡住。**

### 代码分析

**文件**: `applications/rag/data/aligned.py:136-174`

```python
def _sample_distractors(self, n: int, exclude_texts: set):
    """Stream wiki_dpr and randomly keep n passages not in exclude_texts."""
    
    cfg = self.config.get("wiki_dpr_config", "psgs_w100.nq.exact")  # 默认配置
    seed = int(self.config.get("seed", 42))
    oversample = float(self.config.get("distractor_oversample", 3.0))  # 默认 3.0
    window = int(n * oversample)  # 36000 * 3.0 = 108000
    
    stream = load_dataset(
        "facebook/wiki_dpr", cfg, split="train",
        streaming=True, trust_remote_code=True,  # 流式加载
    )
    
    texts, embs = [], []
    for item in stream:
        if len(texts) >= window:  # 需要迭代 108000 条
            break
        if item["text"] in exclude_texts:
            continue
        texts.append(item["text"])
        embs.append(item["embeddings"])
    # ... 无进度日志
```

### 问题点

1. **默认配置**: `psgs_w100.nq.exact` 而不是 `psgs_w100.nq.compressed`
   - `exact` 配置可能不存在或下载失败
   - 师弟的缓存中只有 `compressed` 版本

2. **流式迭代慢**: 需要迭代 108,000 条数据
   - `n_distractors=36000` (默认)
   - `oversample=3.0` (默认)
   - `window = 36000 * 3.0 = 108,000`
   - 从 21M 数据集流式读取 108K 条非常慢

3. **无进度日志**: 代码中没有任何进度输出
   - 用户看到的最后输出是 "Building corpus (mode=aligned)..."
   - 之后完全静默，看起来像卡住

4. **配置不匹配**: 
   - 代码默认 `psgs_w100.nq.exact`
   - 师弟缓存中是 `psgs_w100.nq.compressed`
   - 可能触发重新下载或找不到数据

---

## 验证证据

### 师弟的验证结果

✅ **wiki_dpr 模式工作正常**:
```bash
--corpus_mode wiki_dpr \
--wiki_dpr_config psgs_w100.nq.compressed \
--wiki_dpr_cache_dir /root/.cache/huggingface/datasets
```
输出：
```
Loading dataset shards: 100%|████| 161/161
Corpus ready: corpus_size: 21015300
Evaluating 1 questions...
Retrieval: 1/1 hit gold in Top-50
```

❌ **aligned 模式卡住**:
```bash
--corpus_mode aligned
```
输出：
```
Building corpus (mode=aligned)...
[卡住，无后续输出]
```

进程状态：
- CPU: ~0%
- GPU: 0%
- wchan: `futex_wait_queue_me` (等待状态)

### 代码对比

**wiki_dpr 模式** (`wiki_dpr_corpus.py`):
- 使用完整加载（非流式）
- 指定正确配置 `psgs_w100.nq.compressed`
- 有详细进度日志
- ✅ 工作正常

**aligned 模式** (`aligned.py`):
- 使用流式加载
- 默认配置 `psgs_w100.nq.exact` (错误)
- 无进度日志
- ❌ 卡住

---

## 解决方案

### 方案 1: 修复 aligned 模式默认配置（推荐）

**修改**: `applications/rag/data/aligned.py:149`

```python
# 修改前
cfg = self.config.get("wiki_dpr_config", "psgs_w100.nq.exact")

# 修改后
cfg = self.config.get("wiki_dpr_config", "psgs_w100.nq.compressed")
```

### 方案 2: 添加进度日志

**修改**: `applications/rag/data/aligned.py:158-166`

```python
texts, embs = [], []
print(f"  Sampling {window} distractors from wiki_dpr (config={cfg})...")
last_report = time.time()
for item in stream:
    if len(texts) >= window:
        break
    if item["text"] in exclude_texts:
        continue
    texts.append(item["text"])
    embs.append(item["embeddings"])
    
    # 每 1000 条或每秒报告一次进度
    if len(texts) % 1000 == 0 or time.time() - last_report > 1.0:
        print(f"    Sampled {len(texts)}/{window} distractors...")
        last_report = time.time()

print(f"  ✓ Sampled {len(texts)} distractors")
```

### 方案 3: 改用非流式加载（最快）

**优点**: 
- 更快（一次性加载，内存映射）
- 可以真正随机采样
- 与 wiki_dpr 模式一致

**缺点**: 
- 需要更多内存（但数据已在缓存中）

```python
def _sample_distractors(self, n: int, exclude_texts: set):
    """Sample n random distractors from wiki_dpr."""
    if n <= 0:
        return [], np.empty((0, self.DEFAULT_EMBED_DIM), np.float32)
    
    from datasets import load_dataset
    
    cfg = self.config.get("wiki_dpr_config", "psgs_w100.nq.compressed")
    seed = int(self.config.get("seed", 42))
    cache_dir = self.config.get("cache_dir")
    
    # 完整加载（使用缓存）
    print(f"  Loading wiki_dpr dataset (config={cfg})...")
    ds = load_dataset(
        "facebook/wiki_dpr", cfg, split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    print(f"  ✓ Loaded {len(ds)} passages")
    
    # 随机采样
    print(f"  Sampling {n} distractors...")
    rng = np.random.default_rng(seed)
    indices = []
    for idx in rng.permutation(len(ds)):
        if len(indices) >= n:
            break
        if ds[int(idx)]["text"] not in exclude_texts:
            indices.append(int(idx))
    
    print(f"  ✓ Sampled {len(indices)} distractors")
    texts = [ds[i]["text"] for i in indices]
    embs = np.array([ds[i]["embeddings"] for i in indices], dtype=np.float32)
    
    return texts, embs
```

### 方案 4: 临时方案 - 切换到 wiki_dpr 模式

**最简单，立即可用**:

修改 `scripts/collab/p2_solver_idea6/run_p2_experiments.sh`:

```bash
# 修改前
CORPUS_MODE="aligned"

# 修改后  
CORPUS_MODE="wiki_dpr"
```

添加必要参数：
```bash
cmd="$cmd --wiki_dpr_config psgs_w100.nq.compressed"
cmd="$cmd --wiki_dpr_cache_dir /root/.cache/huggingface/datasets"
```

---

## 推荐行动方案

### 立即执行（让师弟跑起来）:

**选项 A**: 修改实验脚本改用 wiki_dpr 模式
- ✅ 无需改代码
- ✅ 立即可用
- ✅ 师弟已验证工作
- ⚠️ 行为略有不同（全量检索 vs 对齐语料）

**选项 B**: 修复 aligned 模式默认配置
- ✅ 只改一行代码
- ✅ 保持原有行为
- ⏳ 需要测试验证

### 长期修复（提高代码质量）:

1. 修复默认配置 `exact` → `compressed`
2. 添加详细进度日志
3. 考虑改用非流式加载（更快更简单）

---

## 总结

| 方面 | 结论 |
|------|------|
| **根本原因** | aligned 模式默认配置错误 + 流式加载慢 + 无进度日志 |
| **表现** | 看起来卡住，实际在缓慢迭代或等待下载 |
| **验证** | wiki_dpr 模式正常，aligned 模式卡死 |
| **临时方案** | 切换到 wiki_dpr 模式（1分钟） |
| **永久修复** | 修改默认配置 + 添加日志（30分钟） |

**下一步**: 你想执行哪个方案？
