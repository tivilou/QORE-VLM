# RAG Documentation

RAG 模块完整文档索引。

---

## 📖 使用指南 (Guides)

### 核心文档
- [Corpus Modes](guides/corpus_modes.md) - 四种 corpus 模式技术指南（aligned / wiki_dpr / faiss / precomputed）
- [Troubleshooting](guides/troubleshooting.md) - 常见问题排查指南

### 评测与使用
- [Evaluation Guide](guides/evaluation_guide.md) - RAG 评测指南（给协作者）
- [Full Evaluation Guide](guides/full_eval_guide.md) - 全量评测实验指引
- [Refactor Guide](guides/refactor_guide.md) - RAG 模块重构后使用指南
- [Answer Scorer Guide](guides/answer_scorer_guide.md) - Answer Scorer 优化使用指南

---

## 📊 实验报告 (Reports)

### 最新报告
- [Final Report (3-Seed)](reports/final_report_3seeds.md) - RAG 全量评测最终报告（2026-07-21）
- [Answer Scorer Full (3-Seed)](reports/answer_scorer_full_3seeds.md) - Answer Scorer 全量 3-Seed 最终报告（2026-07-24）

### 归档报告
- [Archive](reports/archive/) - 历史报告归档
  - `answer_scorer_200_report.md` - 200题验证报告（2026-07-22）
  - `rag_refactor_summary.md` - RAG 重构完成总结（2024-07-16）

---

## 📋 实验指令 (Instructions)

- [Answer Scorer Full Instructions](instructions/answer_scorer_full_instructions.md) - Answer Scorer 全量 3-Seed 实验指令

---

## 🔗 相关资源

### 脚本与工具
- `scripts/rag/eval_rag_refactored.py` - RAG 评测主脚本
- `scripts/collab/` - 协作实验脚本

### 代码模块
- `applications/rag/data/` - Corpus 管理器实现
- `applications/rag/retrieval/` - 检索模块
- `applications/rag/selection/` - 选择算法（QORE / MMR / VQC）
- `applications/rag/generation/` - 生成模块

### 实验结果
- `exchange/p1_diagnosis/` - Phase 1 诊断实验结果
- `exchange/p2_solver_idea6/` - Phase 2 实验结果

---

## 📂 目录结构

```
rag/
├── README.md              # 本文件
├── guides/                # 使用指南（6个）
├── reports/               # 实验报告（2个 + archive/）
│   └── archive/          # 归档的旧报告
└── instructions/          # 实验指令（1个）
```

---

## 🚀 快速开始

### 运行 RAG 评测
```bash
python -m scripts.rag.eval_rag_refactored \
    --corpus_mode wiki_dpr \
    --dataset nq_open \
    --max_samples 10 \
    --method qore \
    --K 5
```

### 查看文档
- **新手**: 从 [Evaluation Guide](guides/evaluation_guide.md) 开始
- **遇到问题**: 查看 [Troubleshooting](guides/troubleshooting.md)
- **理解 corpus**: 阅读 [Corpus Modes](guides/corpus_modes.md)

---

**有问题？查看 [Troubleshooting](guides/troubleshooting.md) 或联系团队。**
