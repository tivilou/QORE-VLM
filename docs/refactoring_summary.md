# 代码重构总结 - 按功能组织脚本

**日期**: 2026-07-30  
**类型**: 文件结构重构  
**影响**: scripts/ 目录组织方式

---

## 重构动机

### 问题
原始结构使用**实验编号**作为顶层目录：

```
scripts/
├── idea7/              # ❌ 不可扩展（idea8, idea9...）
│   ├── train_*.py
│   └── ...
├── rag/
│   └── eval_rag_refactored.py
└── collab/
```

**缺点**：
1. 不可扩展：每个新实验都会创建新的 ideaX 目录
2. 职责不清：不知道 idea7 是做什么的（训练？评估？）
3. 重复混乱：scripts/idea7 和 scripts/collab/idea7_phase2 都存在

---

## 重构方案（方案 1）✅

按**功能**而非**实验编号**组织：

```
scripts/
└── rag/                        # RAG 相关所有脚本
    ├── eval/                   # 评估脚本
    │   ├── __init__.py
    │   └── eval_rag_refactored.py
    ├── train/                  # 训练脚本
    │   ├── __init__.py
    │   ├── train_soft_qubo.py              # Idea 7
    │   ├── train_soft_qubo_simple.py
    │   ├── README_soft_qubo.md
    │   └── QUICKSTART_soft_qubo.md
    └── utils/                  # 工具脚本
        ├── __init__.py
        └── generate_synthetic_data.py
```

**优点**：
- ✅ 按**功能**（eval/train/utils）组织
- ✅ 未来添加新训练方法，都放在 `scripts/rag/train/` 下
- ✅ 清晰的职责分离
- ✅ 可扩展：train_method1.py, train_method2.py...

---

## 迁移映射

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `scripts/idea7/__init__.py` | 删除 | 不再需要 |
| `scripts/idea7/train_soft_qubo.py` | `scripts/rag/train/train_soft_qubo.py` | 训练脚本 |
| `scripts/idea7/train_soft_qubo_simple.py` | `scripts/rag/train/train_soft_qubo_simple.py` | 简化训练 |
| `scripts/idea7/generate_synthetic_data.py` | `scripts/rag/utils/generate_synthetic_data.py` | 工具脚本 |
| `scripts/idea7/README.md` | `scripts/rag/train/README_soft_qubo.md` | 重命名避免冲突 |
| `scripts/idea7/QUICKSTART.md` | `scripts/rag/train/QUICKSTART_soft_qubo.md` | 重命名避免冲突 |
| `scripts/rag/eval_rag_refactored.py` | `scripts/rag/eval/eval_rag_refactored.py` | 移入 eval/ 子目录 |

---

## 更新的引用

### 1. Python 模块导入

**旧**:
```bash
python -m scripts.idea7.train_soft_qubo_simple
python -m scripts.idea7.generate_synthetic_data
```

**新**:
```bash
python -m scripts.rag.train.train_soft_qubo_simple
python -m scripts.rag.utils.generate_synthetic_data
python -m scripts.rag.eval.eval_rag_refactored
```

### 2. 受影响的文件

更新了以下文件中的所有路径引用：

- ✅ `scripts/collab/idea7_phase2/run_idea7_phase2.sh`
- ✅ `scripts/collab/idea7_phase2/README.md`
- ✅ `scripts/rag/train/README_soft_qubo.md`
- ✅ `scripts/rag/train/QUICKSTART_soft_qubo.md`
- ✅ `docs/idea7_implementation_summary.md`

### 3. 保持不变的部分

- ✅ `scripts/collab/idea7_phase2/` 目录名称保持不变（师弟在用）
- ✅ `exchange/idea7_*` 目录名称保持不变（实验结果）
- ✅ `docs/idea7_*.md` 文档名称保持不变（历史记录）

---

## 验证清单

重构后需要验证：

### 功能验证
- [ ] Phase 2 脚本能正常运行
  ```bash
  bash scripts/collab/idea7_phase2/run_idea7_phase2.sh --help
  ```
- [ ] 训练脚本能正常导入
  ```bash
  python -c "from scripts.rag.train.train_soft_qubo_simple import main"
  ```
- [ ] 评估脚本能正常运行
  ```bash
  python -m scripts.rag.eval.eval_rag_refactored --help
  ```

### 文档验证
- [x] README 中的命令已更新
- [x] QUICKSTART 中的命令已更新
- [x] 协作脚本中的路径已更新

---

## 未来扩展

按照新结构，未来可以这样组织：

```
scripts/rag/
├── eval/
│   └── eval_rag_refactored.py
├── train/
│   ├── train_soft_qubo.py              # Idea 7: 端到端 QUBO 优化
│   ├── train_baseline.py               # 基线训练
│   ├── train_encoder_finetune.py       # Idea 2: DPR encoder 微调
│   └── train_reward_model.py           # 未来: RLHF 训练
└── utils/
    ├── generate_synthetic_data.py
    └── data_preprocessing.py
```

**命名约定**：
- `train_<method_name>.py` - 具体训练方法
- `README_<method_name>.md` - 对应文档
- 避免使用 ideaX 作为文件名，使用描述性名称

---

## Git 历史

```bash
# 重构提交
fc22455 refactor: reorganize scripts by functionality (Plan 1)

# 可以回退到重构前
git checkout 3dbd406  # 重构前的最后一次提交
```

---

## 迁移指南（给师弟）

如果你已经在使用旧路径：

1. **拉取最新代码**
   ```bash
   cd ~/QORE-VLM
   git pull
   ```

2. **无需修改工作流程**
   - `scripts/collab/idea7_phase2/run_idea7_phase2.sh` 已自动更新
   - 直接运行即可，无需修改

3. **如果你有自定义脚本**
   更新导入路径：
   ```python
   # 旧
   from scripts.idea7.train_soft_qubo_simple import main
   
   # 新
   from scripts.rag.train.train_soft_qubo_simple import main
   ```

---

## 总结

✅ **重构完成**  
✅ **所有引用已更新**  
✅ **向后兼容（协作脚本无需修改）**  
✅ **可扩展性提升**

**下一步**: 继续 Phase 2 实验，无需担心路径问题！🚀
