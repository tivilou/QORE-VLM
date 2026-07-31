# QORE Enhancer Plugin System

## 概述

QORE 重构为插件化架构，让每个 Idea 以独立插件形式存在。这样设计的好处：

- ✅ **易于添加新 Idea** - 实现一个类即可，无需修改核心代码
- ✅ **易于移除失败 Idea** - 删除插件文件，不影响其他代码
- ✅ **易于组合 Idea** - 通过配置文件组合多个增强器
- ✅ **易于 A/B 测试** - 配置驱动的实验管理
- ✅ **向后兼容** - 保留 legacy API，旧代码仍可运行

---

## 架构

```
核心 QUBO 引擎（不变）
    ↓
信号构建管道（可插拔）
    ├─ 质量信号 a（基础）
    ├─ 冗余矩阵 b（基础）
    └─ 交互矩阵 w 构建器（插件化）
        ├─ baseline: w = gamma * b
        ├─ idea6: w = gamma * b - delta * c（互补性）
        ├─ idea4: w 添加上下文完整性项
        └─ [未来的 idea...]
```

---

## 快速开始

### 1. 使用现有插件

```python
from applications.rag.selector import select_passages

# 使用 Idea 6（新 API）
indices = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=5,
    method="qore",
    enhancers=["idea6"],
    enhancer_configs={
        "idea6": {"gamma": 0.5, "delta": 0.1}
    },
    question=question,
    passage_texts=texts,
    answer_scorer=scorer,
)

# 组合多个 Idea
indices = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=5,
    method="qore",
    enhancers=["idea6", "idea4"],
    enhancer_configs={
        "idea6": {"gamma": 0.5, "delta": 0.1},
        "idea4": {"alpha": 0.05},
    },
    # ... context ...
)

# 使用 Baseline
indices = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=5,
    method="qore",
    enhancers=["baseline"],
    enhancer_configs={"baseline": {"gamma": 1.0}},
)

# Legacy API（向后兼容）
indices = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=5,
    method="qore",
    gamma=0.5,
    delta=0.1,
    complementarity_method="dpr",
    # ... 自动推断为 idea6 ...
)
```

### 2. 使用配置文件

```yaml
# configs/experiments/my_experiment.yaml
experiment:
  name: "my_experiment"
  dataset: "nq_open"
  max_samples: 200

selection:
  method: "qore"
  K: 5
  lam: 2.0
  
  enhancers:
    - name: "idea6"
      config:
        gamma: 0.5
        delta: 0.1
        method: "dpr"
```

```bash
# 运行实验
python -m scripts.rag.eval_rag --config configs/experiments/my_experiment.yaml
```

---

## 如何添加新 Idea

### 步骤 1：实现 Enhancer 类

创建 `qore/enhancers/ideaX_description.py`:

```python
"""Idea X: Description of your idea."""

from typing import Any
import numpy as np
from .base import QUBOEnhancer
from .registry import register_enhancer

@register_enhancer("ideaX")
class IdeaXEnhancer(QUBOEnhancer):
    """
    Idea X: Your idea description.
    
    Config:
        param1 (float): Description. Default X.
        param2 (str): Description. Default Y.
    
    Context requirements:
        - key1: description
        - key2: description
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.param1 = self.config.get("param1", default_value)
        self.param2 = self.config.get("param2", default_value)
    
    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Modify w according to your idea.
        
        Args:
            w: (N, N) current interaction matrix from previous enhancers
            a: (N,) quality scores
            b: (N, N) redundancy matrix
            context: Additional context dict
        
        Returns:
            w_new: (N, N) modified interaction matrix
        """
        # 验证必需的 context
        self.validate_context(context, ["key1", "key2"])
        
        # 你的逻辑
        # ...
        
        return w_modified
    
    @property
    def name(self) -> str:
        return "ideaX"
    
    def description(self) -> str:
        return f"Idea X Description (param1={self.param1})"
```

### 步骤 2：注册插件

在 `qore/enhancers/__init__.py` 中添加：

```python
from . import ideaX_description
```

### 步骤 3：测试

```python
from qore.enhancers import get_enhancer

# 创建增强器
enhancer = get_enhancer("ideaX", {"param1": 0.5})

# 测试
w = enhancer.enhance(w_in, a, b, context)
```

### 步骤 4：使用

```python
# 单独使用
indices = select_passages(..., enhancers=["ideaX"])

# 组合使用
indices = select_passages(..., enhancers=["baseline", "ideaX", "idea6"])
```

---

## 现有插件

### baseline

标准 QUBO：w = gamma * b

**Config:**
- `gamma` (float): 冗余权重。Default 1.0

**Context:** 无要求

**用途:** 默认的 QORE 配置

---

### idea6

互补性矩阵：w = gamma * b - delta * c

**Config:**
- `gamma` (float): 冗余权重。Default 1.0
- `delta` (float): 互补性权重。Default 0.0
- `method` (str): 互补性计算方法。Default "dpr"

**Context:**
- `question` (str): 查询文本
- `passages` (list[str]): 段落文本
- `answer_scorer`: DPR answer scorer 实例

**用途:** P2 验证成功，Recall +39.4%

**最佳配置:** gamma=0.5, delta=0.1

---

### idea4

上下文完整性：奖励选择同一文档的连续段落

**Config:**
- `alpha` (float): 完整性奖励权重。Default 0.1

**Context:**
- `passages_meta` (list[dict]): 段落元数据，需包含 "doc_id" 和 "rank"

**用途:** 占位实现，需完整规范后完成

---

## 组合原理

增强器按顺序应用，每个增强器接收前一个的输出：

```python
# 示例：baseline + idea4
w = np.zeros((N, N))          # 初始
w = baseline.enhance(w, ...)   # w = gamma * b
w = idea4.enhance(w, ...)      # w = gamma * b + alpha * integrity
```

最终 w 用于构建 QUBO：

```
Q_ij = w_ij + 2*lam  (for i < j)
```

---

## 配置文件示例

查看 `configs/experiments/` 目录：

- `baseline.yaml` - 标准 QUBO
- `idea6_p2_best.yaml` - Idea 6 最佳配置
- `idea6_grid_search.yaml` - Idea 6 网格搜索
- `idea6_plus_idea4.yaml` - 组合两个 Idea

---

## 实验工作流

### 1. 创建配置文件

```yaml
# configs/experiments/new_idea_test.yaml
experiment:
  name: "new_idea_test"

selection:
  enhancers:
    - name: "ideaX"
      config:
        param1: 0.5
```

### 2. 运行实验

```bash
python -m scripts.rag.eval_rag --config configs/experiments/new_idea_test.yaml
```

### 3. 分析结果

结果保存在 `exchange/ideaX/` 目录

### 4. 成功 → Phase 3，失败 → 移除插件

```bash
# 失败的话，删除插件文件
rm qore/enhancers/ideaX_*.py

# 从 __init__.py 移除 import
# 完全不影响其他代码
```

---

## 向后兼容

旧代码使用 `gamma`/`delta`/`complementarity_method` 参数仍然有效：

```python
# 旧代码
indices = select_passages(
    ...,
    gamma=0.5,
    delta=0.1,
    complementarity_method="dpr",
)

# 自动转换为：
# enhancers=["idea6"]
# enhancer_configs={"idea6": {"gamma": 0.5, "delta": 0.1, "method": "dpr"}}
```

---

## 测试

运行测试验证插件系统：

```bash
cd /home/Q-DUET-VLM/QORE-VLM
PYTHONPATH=. python scripts/test_enhancers.py
```

---

## 下一步

1. **完善 Idea 4** - 实现完整的上下文完整性逻辑
2. **添加新 Idea** - 按照上述步骤添加 Idea 7、Idea 8 等
3. **全量实验** - Idea 6 Phase 3 (3610 samples, 3 seeds)
4. **组合实验** - 测试 Idea 6 + Idea 4 等组合

---

## 文件结构

```
qore/
  enhancers/
    __init__.py          # 导出和注册
    base.py              # 基类定义
    registry.py          # 注册表
    pipeline.py          # 组合器
    baseline.py          # 标准 QUBO
    idea6_complementarity.py   # Idea 6
    idea4_context_integrity.py # Idea 4
    [future ideas...]

applications/rag/
  selector.py            # 重构后的选择器（支持插件）
  selector_legacy.py     # 备份的原始版本

configs/
  experiments/
    baseline.yaml
    idea6_p2_best.yaml
    idea6_grid_search.yaml
    idea6_plus_idea4.yaml
    [future configs...]

scripts/
  test_enhancers.py      # 插件系统测试
```
