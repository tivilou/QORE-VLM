# 调参自动化框架

**状态**: 🔒 内部协作使用（暂时公开）

本框架用于团队内部的调参实验自动化。论文发表后可能归档或删除。

---

## 快速开始

### 合作者快捷方式（推荐）

```bash
# 环境检查
bash scripts/collab/setup_env.sh

# 快速测试（5分钟）
bash scripts/collab/run_phase1_quick.sh

# 完整实验（1-2小时）
bash scripts/collab/run_phase1_full.sh
```

### 直接使用

```bash
# Phase 1 诊断（完整版，200题）
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml

# 快速测试（10题）
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/quick_test.yaml

# 自定义参数
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml \
    --override max_samples=50
```

详见 `QUICK_START.md`

---

## 功能特性

- ✅ 自动运行多个实验
- ✅ 失败自动重试
- ✅ 自动分析结果
- ✅ 自动打包成 zip
- ✅ 生成详细报告

---

## 注意事项

⚠️ **本框架为内部工具**，代码质量可能不如正式脚本。

⚠️ **论文发表后**，本目录可能归档或删除。

✅ **如需复现论文结果**，请使用正式脚本：`scripts/rag/eval_rag_refactored.py`

---

## 文档

- **快速开始**: `QUICK_START.md`
- **完整文档**: `README.md`（原有的详细文档）
- **框架总结**: `FRAMEWORK_SUMMARY.md`

---

**仅供内部协作使用**
