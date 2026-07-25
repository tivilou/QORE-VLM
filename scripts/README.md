# Scripts 目录

实验和分析脚本。

---

## 目录结构

### ⭐ 正式脚本（永久公开）

- **rag/** - RAG 评测脚本
  - `eval_rag_refactored.py` - 主评测脚本
  - `eval_suite.py` - 批量评测
  - `build_corpus.py` - 数据准备
  - `build_faiss_corpus.py` - FAISS 索引构建

- **kv_cache/** - KV Cache 评测脚本
  - `eval_kv_cache.py` - 主评测脚本
  - `summarize.py` - 结果分析
  - `bootstrap_ci.py` - 统计分析

- **ablations/** - 消融实验

- **overhead/** - Overhead 实验

---

### 🔒 内部协作（暂时公开）

- **tuning/** - 调参自动化框架
  - 用于团队内部的调参实验自动化
  - 论文发表后可能归档
  - 详见 `tuning/README_COLLAB.md`

- **collab/** - 合作者快捷脚本
  - `run_phase1_quick.sh` - 快速测试
  - `run_phase1_full.sh` - 完整实验
  - `setup_env.sh` - 环境检查
  - 详见 `collab/README.md`

---

### ❌ 本地开发（不推送到 GitHub）

- **dev/** - 测试和开发脚本
- **legacy/** - 废弃代码

这些目录通过 `.gitignore` 排除，仅存在于本地。

---

## 使用说明

### 运行 RAG 评测

```bash
# 单次评测
python scripts/rag/eval_rag_refactored.py \
    --corpus_mode wiki_dpr \
    --dataset nq_open \
    --method qore

# 批量评测
python scripts/rag/eval_suite.py \
    --corpus_mode wiki_dpr \
    --dataset nq_open \
    --methods qore,mmr,topk
```

### 合作者：运行调参实验

```bash
# 快速开始
bash scripts/collab/run_phase1_quick.sh

# 详见
cat scripts/collab/README.md
```

---

## 论文发表后

- ⭐ **正式脚本** 将永久保留
- 🔒 **内部工具** (`tuning/`, `collab/`) 将归档到 `archive/` 分支

---

**详细文档**: 见各目录下的 README.md
