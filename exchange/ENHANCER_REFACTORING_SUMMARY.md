# Enhancer Plugin System - Refactoring Summary

## 问题背景

之前的实现中，每个新 Idea（如 Idea 6、Idea 7）都需要修改核心代码（`selector.py`）。如果 Idea 失败需要移除，又要撤回修改，代码会越来越臃肿且难以维护。

## 解决方案

重构为**插件化架构**，采用以下设计模式：

1. **策略模式** - 每个 Idea 实现统一接口
2. **组合模式** - 多个 Idea 可以组合
3. **注册表模式** - 插件自动发现
4. **配置驱动** - 实验由 YAML 配置

## 核心设计

### QUBOEnhancer 基类

```python
class QUBOEnhancer(ABC):
    @abstractmethod
    def enhance(self, w, a, b, context) -> np.ndarray:
        """修改交互矩阵 w"""
        pass
```

### 插件注册

```python
@register_enhancer("idea6")
class ComplementarityEnhancer(QUBOEnhancer):
    def enhance(self, w, a, b, context):
        return gamma * b - delta * c
```

### 组合使用

```python
# 方式 1: Python API
indices = select_passages(
    ...,
    enhancers=["baseline", "idea6", "idea4"],
    enhancer_configs={
        "baseline": {"gamma": 0.5},
        "idea6": {"delta": 0.1},
        "idea4": {"alpha": 0.05},
    }
)

# 方式 2: 配置文件
enhancers:
  - name: "idea6"
    config: {gamma: 0.5, delta: 0.1}
  - name: "idea4"
    config: {alpha: 0.05}
```

## 优势

### ✅ 易于添加新 Idea

```bash
# 1. 创建插件文件
vim qore/enhancers/idea8_diversity.py

# 2. 实现 QUBOEnhancer
# 3. 注册 @register_enhancer("idea8")
# 4. 导入到 __init__.py

# 完成！无需修改核心代码
```

### ✅ 易于移除失败 Idea

```bash
# 删除插件文件
rm qore/enhancers/idea7_*.py

# 从 __init__.py 移除导入
# 完全不影响其他代码
```

### ✅ 易于组合 Idea

```python
# 单独使用
enhancers=["idea6"]

# 组合使用
enhancers=["baseline", "idea4", "idea6"]

# 结果：w = gamma*b + alpha*integrity - delta*c
```

### ✅ 易于 A/B 测试

```bash
# 测试 3 种配置
python eval.py --config baseline.yaml
python eval.py --config idea6.yaml
python eval.py --config idea6+idea4.yaml
```

### ✅ 向后兼容

```python
# 旧代码仍然工作
select_passages(..., gamma=0.5, delta=0.1, complementarity_method="dpr")

# 自动转换为新 API
# enhancers=["idea6"], enhancer_configs={"idea6": {...}}
```

## 文件结构

```
qore/enhancers/                      # 新增：插件系统
├── __init__.py                      # 导出和注册
├── base.py                          # 基类定义
├── registry.py                      # 注册表
├── pipeline.py                      # 组合器
├── baseline.py                      # ✅ Baseline 插件
├── idea6_complementarity.py         # ✅ Idea 6 插件
└── idea4_context_integrity.py       # ✅ Idea 4 插件（占位）

applications/rag/
├── selector.py                      # ✅ 重构：支持插件
└── selector_legacy.py               # ✅ 备份：原始版本

configs/experiments/                 # 新增：实验配置
├── baseline.yaml                    # ✅ 标准 QUBO
├── idea6_p2_best.yaml              # ✅ Idea 6 最佳配置
├── idea6_grid_search.yaml          # ✅ Idea 6 网格搜索
└── idea6_plus_idea4.yaml           # ✅ 组合实验

docs/
├── ENHANCER_PLUGIN_SYSTEM.md       # ✅ 用户文档
└── ENHANCER_DEVELOPER_GUIDE.md     # ✅ 开发者指南

scripts/
└── test_enhancers.py                # ✅ 插件系统测试
```

## 现有插件

| 插件 | 描述 | 状态 | 配置 |
|------|------|------|------|
| `baseline` | w = gamma * b | ✅ 完成 | gamma |
| `idea6` | w = gamma*b - delta*c | ✅ 完成 | gamma, delta |
| `idea4` | 上下文完整性 | 🚧 占位 | alpha |

## 测试结果

```bash
$ PYTHONPATH=. python scripts/test_enhancers.py

============================================================
Test 1: Available enhancers
============================================================
Registered enhancers: ['baseline', 'idea6', 'idea4']

============================================================
Test 2: Create individual enhancers
============================================================
Baseline: baseline - Baseline (γ=0.5)
Idea 6: idea6 - Idea 6 Complementarity (γ=0.5, δ=0.1)
Idea 4: idea4 - Idea 4 Context Integrity (α=0.05)

============================================================
Test 3: Create pipeline
============================================================
Pipeline: Baseline (γ=0.5) → Idea 4 Context Integrity (α=0.05)
Enhancers: ['baseline', 'idea4']

============================================================
All tests passed! ✅
============================================================
```

## 使用示例

### 1. 使用现有插件

```python
from applications.rag.selector import select_passages

# Idea 6（新 API）
indices = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=5,
    method="qore",
    enhancers=["idea6"],
    enhancer_configs={"idea6": {"gamma": 0.5, "delta": 0.1}},
    question=question,
    passage_texts=texts,
    answer_scorer=scorer,
)

# 组合多个 Idea
indices = select_passages(
    ...,
    enhancers=["idea6", "idea4"],
    enhancer_configs={
        "idea6": {"gamma": 0.5, "delta": 0.1},
        "idea4": {"alpha": 0.05},
    },
)
```

### 2. 添加新 Idea

```python
# qore/enhancers/idea8_diversity.py

@register_enhancer("idea8")
class DiversityEnhancer(QUBOEnhancer):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = self.config.get("alpha", 0.1)
    
    def enhance(self, w, a, b, context):
        # w_ij = -alpha * (1 - b_ij)
        # 奖励选择语义距离远的段落
        diversity = 1.0 - b
        np.fill_diagonal(diversity, 0.0)
        return w - self.alpha * diversity
    
    @property
    def name(self) -> str:
        return "idea8"
    
    def description(self) -> str:
        return f"Idea 8 Diversity (α={self.alpha})"
```

```python
# qore/enhancers/__init__.py
from . import idea8_diversity
```

```python
# 使用
indices = select_passages(..., enhancers=["idea8"])
```

### 3. 移除失败的 Idea

```bash
# Idea 失败了
rm qore/enhancers/idea8_diversity.py
# 从 __init__.py 移除导入

# 完成！不影响其他代码
```

## 下一步

1. **✅ 完成** - 插件框架和基础插件
2. **待办** - 完善 Idea 4 实现
3. **待办** - 添加配置文件解析到评估脚本
4. **待办** - Idea 6 Phase 3 全量实验
5. **待办** - 探索新的 Idea 并快速迭代

## 文档

- **用户文档**: `docs/ENHANCER_PLUGIN_SYSTEM.md`
- **开发者指南**: `docs/ENHANCER_DEVELOPER_GUIDE.md`
- **测试脚本**: `scripts/test_enhancers.py`
- **配置示例**: `configs/experiments/*.yaml`

## 向后兼容性

所有现有代码保持兼容：

```python
# 旧代码（legacy API）
select_passages(..., gamma=0.5, delta=0.1, complementarity_method="dpr")

# 新代码（plugin API）
select_passages(..., enhancers=["idea6"], enhancer_configs={"idea6": {...}})

# 两者等价，自动转换
```

## 总结

通过插件化架构，我们实现了：

- 🎯 **核心稳定** - QUBO 引擎和求解器不变
- 🔌 **插件灵活** - Idea 以插件形式独立存在
- 🧩 **组合自由** - 插件可任意组合
- ⚙️ **配置驱动** - 实验由 YAML 配置管理
- 🚀 **快速迭代** - 添加/删除 Idea 不影响核心代码

这让研究迭代更快速、代码更清晰、实验更可重复。
