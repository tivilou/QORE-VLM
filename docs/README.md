# Documentation

QORE 项目文档导航。

---

## 📚 项目文档 (Project)

- [项目说明](project/项目说明.md) - QORE 项目概述
- [Technical Roadmap](project/technical_roadmap.md) - 技术路线图与方法设计
- [Development Plan](project/development_plan.md) - 开发计划
- [Codebase Analysis](project/codebase_analysis.md) - DUET-VLM 代码库解读

---

## 🔍 RAG 模块 (RAG Module)

详见 [rag/README.md](rag/README.md) 获取 RAG 相关完整文档。

### 快速链接

**使用指南**:
- [Corpus Modes](rag/guides/corpus_modes.md) - 四种 corpus 模式技术指南
- [Troubleshooting](rag/guides/troubleshooting.md) - 常见问题排查
- [Evaluation Guide](rag/guides/evaluation_guide.md) - 评测指南（给协作者）

**实验报告**:
- [Final Report (3-Seed)](rag/reports/final_report_3seeds.md) - RAG 全量评测最终报告
- [Answer Scorer Full (3-Seed)](rag/reports/answer_scorer_full_3seeds.md) - Answer Scorer 全量 3-Seed 报告

---

## 🧪 实验方法论 (Experiments)

- [Experiment Guide](experiments/experiment_guide.md) - Bootrear 实验指南
- [Statistical Best Practices](experiments/statistical_best_practices.md) - KV-Cache 评测统计最佳实践

---

## ⚡ 性能优化 (Performance)

- [KV Cache Tuning](performance/kv_cache_tuning.md) - KV Cache 性能调优指南
- [HuggingFace Cache API](performance/hf_cache_api.md) - HF Cache API 技术笔记

---

## 📂 目录结构

```
docs/
├── README.md              # 本文件
├── project/               # 项目级文档
├── rag/                   # RAG 模块文档
│   ├── guides/           # 使用指南
│   ├── reports/          # 实验报告
│   │   └── archive/     # 归档的旧报告
│   └── instructions/     # 实验指令
├── experiments/          # 实验方法论
└── performance/          # 性能优化
```

---

## 🔗 其他文档

- **脚本文档**: `scripts/collab/` - 协作实验脚本使用说明
- **实验结果**: `exchange/` - 实验结果交换区
- **根目录**: `README.md` - 项目主 README
