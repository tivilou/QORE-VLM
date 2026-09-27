# Q-GSRR 100题 feature-only audit

本报告只审计固定 Top-50 中的多粒度文本信号；没有运行 selector、Generator 或 evaluator，且没有读取 Evidence/Silver 标签。

- 输入：`669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731`
- 规模：100 题 × 50 候选
- 视图：title、full passage、最高单句窗口、最高相邻双句窗口
- 边界：rank 5 对 rank 6--10，最多局部替换第 5 位

## 结果

| 触发定义 | 触发题数 | 触发候选对数 | 门槛 15 题 |
|---|---:|---:|---:|
| full 反转 + 单句/双句局部反转 | 2 | 3 | 未通过 |
| 上述条件 + title 不下降（canonical） | 2 | 2 | 未通过 |

## 决定

- 当前 canonical gate：`all_view_inversion`。
- 结果：`stop_before_selector_implementation`。
- 该结果仍是 L0 mechanism diagnostic，不代表 Silver utility、最终答案正确率或量子优势。
- 若继续，下一步只能是冻结量子/经典匹配控制和 identity/leakage contract；不能调阈值追样本。
