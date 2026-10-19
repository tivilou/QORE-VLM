# 20261018T-141409Z | reader-larger-untouched-cohort-freeze | 256题未触碰 cohort 冻结验收

## 0. 项目工作流总览

本项目只研究固定检索结果中的 `Top-50 -> Top-5` 筛选。输入是一道 Natural Questions 问题及 Wiki-DPR 已检索出的50个候选段；DPR Reader 产生基础相关分，冻结的量子头和同预算经典头分别调整候选得分，再各自选5段。答案、Silver/support 标签和生成答案只允许在选段完成后的盲评阶段使用，不能参与本次问题抽样、检索或在线打分。

本检查点只完成数据阶段：从 `nq_open/validation[512:3610]` 中按预注册 SHA-256 顺序选256题，排除已接触的434题和历史100题，并固定每题的 DPR/Wiki-DPR Top-50。没有训练、Reader推理、量子/经典打分、Top-5选择或Generator调用，因此本条目没有方法效果指标。

## 一句话结论

- [已测事实] 真实256题×50段 cohort 已完整上传并通过独立哈希、结构、历史重叠和目标字段审计；它可以进入冻结 scoring manifest 阶段，但尚不能说明量子方法是否提升选段效果。

## 1. 这次想法是什么

- 通俗原理：先把一批从未用于训练、调参或旧100题分析的新问题及其Top-50完全锁死，再运行同一套Reader、量子头和经典头，避免看到结果后换题。
- 为什么值得试：此前validation87只有15题发生集合变化，虽然盲评方向为正，但样本太小，仍为L0。
- 预期机制：[机制解释] 如果量子头学习到的不是已暴露问题上的偶然修补，它应在这批身份不重叠的问题上产生足够多、且优于Reader和经典头的有效变化。

| 对比项 | 上一检查点 | 本检查点 | 目的 |
| --- | --- | --- | --- |
| 数据范围 | validation87，已完成一次transport/盲评 | 新256题，每题固定Top-50 | 扩大未触碰验证范围 |
| 允许观察的信息 | transport后仅盲评变化并集 | 仅问题身份与检索候选 | 先冻结样本，阻断结果条件化抽样 |
| 方法输出 | Reader/量子/经典Top-5已有 | 尚未产生 | 保持真正的前瞻性检验 |

## 2. 怎么验证

- 数据：`nq_open/validation[512:3610]`，seed `20261018`，确定性哈希排序取256题。
- 排除：保护边界434题与历史100题的精确问题文本 SHA-256，并集534个唯一身份。
- 检索：DPR query encoder；Wiki-DPR `psgs_w100.nq.compressed`；`nprobe=64`；每题Top-50。
- trace：`sample-trace.v2`，`cohort_lock.json`，256个样本；覆盖 `data/preprocess/retrieval/diagnosis`。`scoring/selection` 缺失是因为冻结前明确未授权，而不是运行失败。
- 完整性：重新计算cohort身份、问题/段落身份、rank覆盖、有限检索分数、历史重叠、目标字段、manifest/receipt、源码归档与Git revision。
- 证据：18083目录 `five_ideas/reader_larger_untouched_cohort_256/20261018T141409477054Z`；cohort SHA-256 `49c57f20...455a02`；身份 SHA-256 `25a71785...25397`。

## 3. Case Study：固定顺序第一题

- case选择规则：预注册哈希排序后的第1题，不按结果挑选。
- dataset index：`2509`；问题身份：`8b57ed94...c9a47`。
- 候选：50个唯一段，rank完整覆盖1—50；检索分数范围 `73.30968475341797—77.58512878417969`。
- target-access：未读取答案、Silver、support标签、方法得分或生成结果。
- 训练/优化：不适用。
- 推理/选择：尚未运行；不能从检索分数推断未来三个方法的Top-5。

## 4. 结果

| 评估单元 | 主指标 | 诊断指标 | 状态 |
| --- | ---: | --- | --- |
| 问题 | 256/256唯一 | dataset index 556—3605 | 通过 |
| 候选段 | 12,800/12,800 | 每题50个唯一段、rank 1—50、分数均有限 | 通过 |
| 历史隔离 | 0/534重叠 | 434保护题+100历史题 | 通过 |
| 目标边界 | 0个禁用字段 | answer/gold/Silver/support/prediction/generation均未出现 | 通过 |
| 完整性 | 4/4上传文件 | manifest、receipt、服务端字节数和SHA一致 | 通过 |
| 方法效用 | 未提供 | scoring/selection尚未运行 | 不适用 |

## 5. 从结果学到了什么

- 正向经验：[已测事实] 数据冻结流程已经从“依赖合作者口头说明”提升为可独立重算的身份、源码和传输完整性链。
- 负向教训：[已测事实] GitHub没有出现 `5815c74` 之后的新精简结果提交；这不影响已验证的18083原始artifact，但说明“上传大文件”和“提交compact manifest”仍是两个步骤。
- 机制判断：[机制解释] 当前设计已经排除了最明显的题目复用和结果后换题风险，使后续变化更能代表冻结头的迁移行为。
- 尚未证明：[尚未证明] 尚未证明量子头能改变足够多的Top-5、能优于Reader或匹配经典头，更没有L1/L2或量子优势证据。

## 6. 当前证据判断

- evidence tier：`L0_data_freeze`。
- claim ceiling：仅能声明“256题未触碰 cohort 冻结与数据边界验收通过”。
- 原因：没有任何Reader/量子/经典评分和语义效用结果。

## 7. 下一步方向

1. 绑定cohort SHA、Reader revision、顺序量子/经典最终checkpoint及源码，生成不可变scoring manifest。
2. 在已绑定且实时检查容量的GPU主机上运行一次Reader特征抽取与三臂transport。
3. 仅当变化数门槛通过，才构建方法盲化的语义审阅包。

推荐：方向1——冻结scoring manifest

- hypothesis：精确绑定输入和checkpoint后，后续transport可以做到无换题、无换头、无标签访问的一次性可回放执行。
- minimum experiment：只生成并验证manifest与下载/回放预检，不运行正式256题评分。
- success gate：cohort/Reader/checkpoint/source哈希全部精确匹配，三臂定义唯一，synthetic或小fixture同样本回放通过。
- kill gate：任一checkpoint来源不清、哈希漂移、Reader parity失败或需要重选问题时停止。
- resource estimate：CPU级manifest与哈希核验；正式Reader运行另需GPU。
- uncertainty：历史顺序参数的完整归档和在Titan运行环境中的可加载性仍需确认。
- authorization：由既有预注册计划授权进入LU-04；正式GPU评分需在LU-04通过后才进入LU-05。

## 8. 证据索引

- Progress：`.ai-progress/workstreams/rag-selector/state.md`
- Receipt：`.ai-progress/workstreams/rag-selector/refs/reader_larger_untouched_cohort_20261018_receipt.json`
- Contract：`configs/experiments/reader_larger_untouched_utility_256.json`
- Plan：`.ai-progress/workstreams/rag-selector/plans/reader-larger-untouched-utility/v001.json`
- Server artifact：`/srv/qore-collab-uploads/five_ideas/reader_larger_untouched_cohort_256/20261018T141409477054Z`
