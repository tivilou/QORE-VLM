# Enhancer Plugin System - Developer Guide

## 设计模式

插件系统采用以下设计模式的组合：

### 1. 策略模式 (Strategy Pattern)

每个 Idea 是一个策略，实现相同的接口 `QUBOEnhancer`：

```python
class QUBOEnhancer(ABC):
    @abstractmethod
    def enhance(self, w, a, b, context) -> np.ndarray:
        pass
```

### 2. 组合模式 (Composite Pattern)

多个增强器通过 `EnhancerPipeline` 组合：

```python
pipeline = EnhancerPipeline([enhancer1, enhancer2, enhancer3])
w = pipeline.enhance(a, b, context)
```

### 3. 注册表模式 (Registry Pattern)

增强器通过装饰器自动注册：

```python
@register_enhancer("my_idea")
class MyIdeaEnhancer(QUBOEnhancer):
    ...
```

### 4. 配置驱动 (Configuration-Driven)

实验通过 YAML 配置文件驱动，无需修改代码。

---

## 核心概念

### 交互矩阵 w

`w[i, j]` 表示选择段落 i 和 j 的成对交互：

- **正值** = 惩罚一起选择（冗余）
- **负值** = 奖励一起选择（互补）
- **零** = 无交互

QUBO 目标函数：

```
E(x) = -sum_i a_i x_i + sum_{i<j} w_ij x_i x_j + lam*(sum_i x_i - K)^2
```

其中：
- `a_i`: 段落质量
- `w_ij`: 成对交互（增强器修改的部分）
- `lam`: 基数约束惩罚

### 增强器职责

增强器负责构建或修改 `w` 矩阵：

```python
def enhance(self, w, a, b, context):
    # w: 前一个增强器的输出（初始为零矩阵）
    # a: 质量信号（归一化到 [0, 1]）
    # b: 冗余矩阵（余弦相似度）
    # context: 额外数据（文本、scorer 等）
    
    # 你的逻辑：修改 w
    w_new = ...
    
    return w_new
```

### Pipeline 执行流程

```python
# 初始化
w = np.zeros((N, N))

# 依次应用增强器
for enhancer in pipeline.enhancers:
    w = enhancer.enhance(w, a, b, context)

# 最终 w 用于构建 QUBO
Q = build_qubo_matrix_from_w(a, w, K, lam)
```

---

## 实现新 Idea 的清单

### □ 步骤 1：设计

- [ ] 明确你的 Idea 如何修改 w 矩阵
- [ ] 确定需要哪些配置参数
- [ ] 确定需要哪些 context 数据
- [ ] 考虑计算复杂度（O(N^2) 是上限）

### □ 步骤 2：实现

- [ ] 创建 `qore/enhancers/ideaX_description.py`
- [ ] 继承 `QUBOEnhancer`
- [ ] 实现 `__init__`, `enhance`, `name`, `description`
- [ ] 用 `@register_enhancer("ideaX")` 装饰
- [ ] 在 `qore/enhancers/__init__.py` 中导入

### □ 步骤 3：测试

- [ ] 单元测试：在 `scripts/test_enhancers.py` 中添加测试
- [ ] 功能测试：创建小样本配置文件
- [ ] 验证 w 矩阵的形状和对称性
- [ ] 验证与其他增强器的组合

### □ 步骤 4：实验

- [ ] 创建 `configs/experiments/ideaX_pilot.yaml`
- [ ] 运行小规模实验（max_samples: 100）
- [ ] 分析结果，调整参数
- [ ] 如果成功，运行 Phase 2（200 samples）

### □ 步骤 5：文档

- [ ] 在 `docs/ENHANCER_PLUGIN_SYSTEM.md` 中添加插件说明
- [ ] 记录最佳配置和适用场景
- [ ] 更新 `configs/experiments/` 中的示例

### □ 步骤 6：清理

- [ ] 如果失败，删除插件文件
- [ ] 从 `__init__.py` 移除导入
- [ ] 删除配置文件
- [ ] 记录失败原因和教训

---

## 常见模式

### 模式 1：基于信号的增强器

直接使用 a 或 b 构建 w：

```python
def enhance(self, w, a, b, context):
    return self.gamma * b  # 基于冗余
```

### 模式 2：基于文本的增强器

从段落文本计算交互：

```python
def enhance(self, w, a, b, context):
    passages = context["passages"]
    question = context["question"]
    
    # 计算成对信号
    pairwise_signal = self._compute_pairwise(passages, question)
    
    return w + self.alpha * pairwise_signal
```

### 模式 3：基于元数据的增强器

从段落元数据计算交互：

```python
def enhance(self, w, a, b, context):
    meta = context["passages_meta"]
    
    # 奖励来自同一文档的段落
    for i in range(N):
        for j in range(i+1, N):
            if meta[i]["doc_id"] == meta[j]["doc_id"]:
                w[i, j] -= self.bonus  # 负值 = 奖励
                w[j, i] -= self.bonus
    
    return w
```

### 模式 4：组合增强器

叠加多个信号：

```python
def enhance(self, w, a, b, context):
    # w 可能已经包含其他增强器的贡献
    # 添加你的贡献
    return w + self.weight * self._compute_signal(context)
```

---

## 最佳实践

### 1. 保持 w 的对称性

```python
# 确保 w[i, j] == w[j, i]
for i in range(N):
    for j in range(i+1, N):
        value = compute_value(i, j)
        w[i, j] = value
        w[j, i] = value  # 对称
```

### 2. 保持对角线为零

```python
# w[i, i] = 0（不与自己交互）
np.fill_diagonal(w, 0.0)
```

### 3. 处理缺失的 context

```python
def enhance(self, w, a, b, context):
    if context.get("passages") is None:
        # 无法计算，返回原 w
        return w
    
    # 或者抛出错误
    self.validate_context(context, ["passages", "question"])
```

### 4. 考虑计算成本

```python
# 避免 O(N^3) 或更高复杂度
# 对于大 N，考虑近似或采样

def enhance(self, w, a, b, context):
    N = len(a)
    
    if N > 100:
        # 采样或近似
        pass
    else:
        # 精确计算
        pass
```

### 5. 使用有意义的默认值

```python
def __init__(self, config):
    super().__init__(config)
    # 选择合理的默认值，让插件开箱即用
    self.alpha = self.config.get("alpha", 0.1)
```

---

## 调试技巧

### 1. 打印 w 矩阵

```python
def enhance(self, w, a, b, context):
    w_new = self._compute_w(...)
    
    # 调试输出
    print(f"[{self.name}] w range: [{w_new.min():.3f}, {w_new.max():.3f}]")
    print(f"[{self.name}] w mean: {w_new.mean():.3f}")
    
    return w_new
```

### 2. 验证对称性

```python
assert np.allclose(w, w.T), "w must be symmetric"
assert np.allclose(np.diag(w), 0), "w diagonal must be zero"
```

### 3. 单元测试

```python
def test_my_enhancer():
    enhancer = get_enhancer("ideaX", {"param": 0.5})
    
    N = 5
    a = np.random.rand(N)
    b = np.random.rand(N, N)
    np.fill_diagonal(b, 0)
    b = (b + b.T) / 2  # 对称
    
    w = enhancer.enhance(np.zeros((N, N)), a, b, {})
    
    assert w.shape == (N, N)
    assert np.allclose(w, w.T)
    assert np.allclose(np.diag(w), 0)
```

---

## 常见陷阱

### ❌ 忘记对称性

```python
# 错误
w[i, j] = value
# w[j, i] 未设置！
```

### ❌ 覆盖而非叠加

```python
# 错误：丢弃了前面增强器的贡献
def enhance(self, w, a, b, context):
    return self.gamma * b  # 覆盖了 w

# 正确：叠加
def enhance(self, w, a, b, context):
    return w + self.gamma * b  # 叠加到 w
```

但也有例外：baseline 增强器通常是第一个，可以直接返回新的 w。

### ❌ 未验证 context

```python
# 错误：假设 context 总是有效
passages = context["passages"]  # 可能 KeyError

# 正确：验证
self.validate_context(context, ["passages"])
passages = context["passages"]
```

### ❌ 复杂度过高

```python
# 错误：O(N^3)
for i in range(N):
    for j in range(N):
        for k in range(N):
            w[i, j] += compute(i, j, k)  # 太慢
```

---

## 示例：完整实现

```python
"""Idea 8: Passage diversity reward based on semantic distance."""

from typing import Any
import numpy as np
from .base import QUBOEnhancer
from .registry import register_enhancer

@register_enhancer("idea8")
class DiversityEnhancer(QUBOEnhancer):
    """
    Idea 8: Reward selecting semantically distant passages.
    
    This is the opposite of redundancy penalty: we want diverse coverage.
    Uses w_ij = -alpha * (1 - b_ij), where b_ij is cosine similarity.
    
    Config:
        alpha (float): Diversity reward weight. Default 0.1.
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.alpha = self.config.get("alpha", 0.1)
    
    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Add diversity reward: w_ij -= alpha * (1 - b_ij).
        
        Low similarity (distant passages) → negative w_ij → reward.
        """
        N = len(b)
        
        # Diversity matrix: 1 - similarity
        diversity = 1.0 - b
        np.fill_diagonal(diversity, 0.0)
        
        # Negative weight = reward
        return w - self.alpha * diversity
    
    @property
    def name(self) -> str:
        return "idea8"
    
    def description(self) -> str:
        return f"Idea 8 Diversity Reward (α={self.alpha})"
```

使用：

```python
# 单独使用
indices = select_passages(..., enhancers=["idea8"])

# 与 baseline 组合
indices = select_passages(..., enhancers=["baseline", "idea8"])
# 结果：w = gamma * b - alpha * (1 - b)
#     = (gamma + alpha) * b - alpha
```

---

## 总结

插件系统的核心思想：

1. **核心不变** - QUBO 引擎和求解器不变
2. **插件可变** - 每个 Idea 是独立插件
3. **组合灵活** - 插件可任意组合
4. **配置驱动** - 实验由 YAML 配置
5. **易于迭代** - 失败的 Idea 删除即可

这样设计让研究迭代更快速、代码更清晰、实验更可重复。
