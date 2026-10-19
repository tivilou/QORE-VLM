# 256题未触碰盲评：机制恢复 dossier

- Outcome: candidate
- Evidence stage: screen
- Claim ceiling: diagnostic

## 已测事实

- 固定256题中有67题的三臂 Top-5 至少两组不同；两名隔离 reviewer 审完404段，352项语义码一致、52项在方法身份封存时由第三名 adjudicator 裁决，最终未决0。
- 在67个变化题上，Reader / quantum / classical 的 useful 段总数为 `141 / 151 / 142`，direct 段总数为 `79 / 85 / 80`。
- quantum 相对 Reader 的 useful 胜/负/平为 `12 / 2 / 242`（含189个精确零差值不变题），useful 总差 `+10`，direct 总差 `+6`，points 差 `+16`，完整256题问题簇 bootstrap 95%区间 `[5, 28]`。
- quantum 相对 matched classical 的 useful 胜/负/平为 `10 / 1 / 245`，useful 总差 `+9`，direct 总差 `+5`，points 差 `+14`，bootstrap 95%区间 `[4, 25]`。
- 预注册8项效用门全部通过；独立审计对共识、总量、bootstrap和门槛逐项重放一致。

## 仍然成立的替代解释

1. 这是一个256题 screen，而不是完整 NQ/Wiki-DPR 正式评估；189题因三臂选择完全相同而只提供精确零差值。
2. 52/404项初始语义码有分歧，说明判定仍有测量噪声；盲裁决消除了未决项，但不是不同数据集复制。
3. matched classical 也优于 Reader，quantum 比 classical 多9个 useful 段是积极信号，但尚未经过独立 split/seed 和 finite-shot 实现复核。
4. 结果直接支持“Top-5 中保留了更多语义支持证据”，不等同于 Generator 最终答案提升，也不能单独证明量子组件的因果优势。

## 已关闭的解释

- 不是“量子头没有实际改变选择”：quantum 在61/256题改变 Reader Top-5，并在47/256题与 classical 不同。
- 不是一次评分实现错误：768条分数向量已独立重放，盲审后的总量、case级胜负、bootstrap和8项门槛又由第二套审计逻辑完全重放。
- 不是方法身份诱导 reviewer：公开盲包不含答案、候选身份、排名、分数、方法或选择成员关系；密钥直到404项共识完成后才打开。

## 下一步区分性测量

冻结一个完整、未触碰的自然数据正式评估人口，并在运行前固定两个独立确认单元、同一 Reader/quantum/classical checkpoint、语义判定协议、问题簇统计和数据暴露策略。若 quantum 对 Reader 和 matched classical 的正向 useful/points 效果不能在完整人口及独立确认中复现，则停止这套 frozen head；若复现，再单独进入 finite-shot、资源与严格量子归因审计。

## 结论

当前应保留该候选并进入“完整真实数据复制设计”，但证据等级仍锁在 `L0_diagnostic`，不得提前声称 L1/L2 或量子优势。
