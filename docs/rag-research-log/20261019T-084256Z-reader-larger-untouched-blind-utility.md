# 20261019T-084256Z | LU-06 | 256题未触碰双盲语义效用审计

## 0. 项目工作流总览

本项目只优化 RAG 中固定 `Top-50 -> Top-5` 的段落筛选。每道 Natural Questions 问题先由冻结的 Wiki-DPR 检索器给出50段；冻结的 DPR Reader 为50段打分。原始 Reader 直接取最高5段，实验量子头和参数预算匹配的经典头分别在同一 Reader 信号上做冻结的分数修正，再各取5段。三种方法使用完全相同的问题和候选段；检索器、Generator、答案生成和最终问答 evaluator 均不改变。

量子/经典头的训练发生在先前锁定的训练材料上，本轮不训练、不调参，只在身份不重叠的256题上做一次推理。运行阶段先在不知道答案或语义标签的情况下保存三臂分数和 Top-5；只有三臂选择确实不同的67题进入语义审阅。两名 reviewer 只看问题、标题和段落文本，不看答案、分数、排名、方法身份或选择成员关系。两人有分歧的52段由第三名 reviewer 在方法密钥仍封存时裁决，404段全部达成共识后才把语义标签与三种方法的 Top-5 对齐。

主评估量是 Top-5 中 `useful` 段数量：段落必须对回答问题有实质帮助，且实体、时间、关系和答案类型范围匹配。`direct` 表示段落本身直接给出回答证据，是保护指标；`points=useful+direct` 用于问题级 bootstrap。没有发生选择变化的189题在三种方法间严格相同，成对差值为0并进入完整256题的问题簇统计。Generator 生成质量、EM、F1 本轮不适用，因为它们不属于此次 selector-only 门槛。

## 一句话结论

- [已测事实] 冻结量子头在256题盲审 screen 中相对 Reader 多保留10个 useful、6个 direct 段，相对 matched classical 多9个 useful、5个 direct 段；8项预注册门全部通过且独立重放一致，但当前仍严格记为 `L0_diagnostic`，只推进完整真实数据复制设计。

## 1. 这次想法是什么

- 通俗原理：不是继续看量子头“有没有换段”，而是在不知道方法身份时直接判断换进来的段是否更能回答问题。
- 为什么值得试：上一阶段已证明量子头会在61/256题改变 Reader Top-5，并在47/256题与经典头不同；但“发生变化”本身不说明变化有用。
- 预期机制：[机制解释] 如果顺序保护训练学到的分数修正确实保留了更有用的 Reader 候选，量子头在盲审后应同时超过原 Reader 和 matched classical，且问题级正向差值的 bootstrap 下界大于0。

| 对比点 | 上一阶段 LU-05 | 本阶段 LU-06 |
| --- | --- | --- |
| 看什么 | Top-5 是否变化、量子/经典是否分离 | 变化后的段是否真正支持问题 |
| 可见信息 | 分数、Top-5索引；语义标签关闭 | reviewer 只见问题和段落；方法密钥封存 |
| 结论上限 | label-blind transport | method-blind semantic utility screen |

## 2. 怎么验证

- 实验单元：256个身份不重叠问题；语义审阅覆盖其中67个变化题的404个 Top-5 union 段。
- 配置：`configs/experiments/reader_larger_untouched_blind_review_256.json`，SHA-256 `3ed753bbaab66dc792641afe38663b88d1b091d1bafd01318e7e477844df1a6c`。
- transport：run `20261019T014046Z`；trace SHA-256 `5c419bbf8a975cef65260d1a7c80b0348659436f0172e13a5bec32a627ef18f1`。
- 盲包：run `20261019T084256Z`；public packet SHA-256 `60691dd6e8fee7c61ac5da68ce2e381ac9c00ebbd80652d1e58bada8680a7109`。
- 对照：冻结 Reader 原始 Top-5、冻结量子头、冻结 matched classical 头；同一问题、同一50段、同一次 Reader 特征计算。
- 数据边界：packet 构建不读取答案、Silver、支持标注或生成结果；reviewer 看不到候选身份、数据位置、排名、分数、方法和选择成员关系。
- 预注册门：quantum useful 至少 Reader `+5`、classical `+3`；direct 不低于 Reader；useful case-win margin 至少 `+3/+1`；完整256题 points bootstrap 两个下界均 `>0`；quantum 不确定选中项为0。
- trace：`sample-trace.v2` transport 覆盖 data/retrieval/scoring/selection；404段盲审覆盖 evaluation/diagnosis。训练在本轮不适用，context/generation 不适用。

## 3. Case Study：按项目契约选择的真实案例

### 3.1 训练、拟合或优化中的真实案例

- 不适用。本轮使用已冻结 checkpoint，没有新的训练、loss、optimizer step 或 checkpoint 选择。

### 3.2 推理与评估中的真实案例

- 固定选择规则不是事后挑最好案例，而是纳入三臂 Top-5 至少两组不同的全部67题和全部404个 union 段。
- 每个真实问题均保存 question hash、50个 candidate hash、Reader/quantum/classical Top-5 和完整分数向量；公开 review 视图用 opaque case/item ID，仅暴露问题、标题和段落文本。
- 为避免解封后挑案例，本条目不从67题中另选一个“典型成功题”作故事化展示；完整逐题轨迹由 packet、sealed key、consensus 的哈希共同约束。

### 3.3 正向经验与负向教训的批量机制案例

- `batch_id=20261019T084256Z-all-changed-union`，样本为全部67变化题/404段，选择规则在语义结果前已固定。
- [已测事实] 两名隔离 reviewer 的语义码352/404完全一致；52项分歧全部盲裁，未决0，原始一致率87.13%。
- [已测事实] 量子相对 Reader 的 useful case 为12胜/2负/242平；相对 classical 为10胜/1负/245平。这里的平局包含189个三臂 Top-5 完全相同的问题。
- [机制解释] 优势不是由少数未决项残留或事后删题形成；但仍可能受同一数据源、一次 checkpoint 和语义测量噪声影响。

## 4. 结果

| 方法（67个变化题） | useful | direct | uncertain | points |
| --- | ---: | ---: | ---: | ---: |
| Reader | 141 | 79 | 0 | 220 |
| Quantum | **151** | **85** | 0 | **236** |
| Matched classical | 142 | 80 | 0 | 222 |

| 完整256题成对比较 | useful差 | direct差 | useful胜/负/平 | points差 | 95%问题簇bootstrap |
| --- | ---: | ---: | --- | ---: | --- |
| Quantum - Reader | **+10** | **+6** | **12 / 2 / 242** | **+16** | **[5, 28]** |
| Quantum - Classical | **+9** | **+5** | **10 / 1 / 245** | **+14** | **[4, 25]** |

- [已测事实] 8/8预注册检查通过，`advance_to_full_data_replication_design=true`。
- [已测事实] 第二套审计代码重新合并 review/adjudication，并独立计算总量、case胜负、10000次 bootstrap 和 gate，全部逐项一致。
- 数值/运行缺陷：未发现。完整原始盲审材料当前保留在本地 `.codex-tmp/lu06-blind-review/20261019T084256Z`；紧凑哈希回执已进入进度体系，18083归档等待本机交换凭据，不影响已提交的紧凑证据。

## 5. 从结果学到了什么

- 正向经验：[已测事实] 顺序保护量子头第一次在身份不重叠的自然问题 screen 上，同时超过 Reader 和 matched classical，而且不是只提升模糊 useful：direct 也增加。
- 正向经验：[已测事实] validation87 的小幅优势没有在更大的256题上消失；量子相对经典的 useful 差从 `+1` 扩大为 `+9`。
- 负向教训：[已测事实] 52/404项需要裁决，说明语义评估不能只依赖单 reviewer；双审加盲裁仍应保留为正式协议。
- 机制判断：[机制解释] 当前最值得投入的不是再改头或调阈值，而是验证同一个 frozen head 的效果能否覆盖完整未触碰人口并独立复现。
- 尚未证明：[尚未证明] 该优势是否在完整 NQ 人口、独立 split、不同数据源或 finite-shot 量子执行中持续；也未证明 Generator 最终答案改善或量子特有优势。

## 6. 当前证据判断

- evidence tier: `L0_diagnostic`
- claim ceiling: `diagnostic`
- why: 本轮是预注册的256题 screen，虽有真实 label-free selection、正向语义指标和统计下界，但项目的 `full_real_data_l2` 覆盖层要求完整正式人口和至少两个独立确认；finite-shot 与资源审计也未完成。

## 7. 下一步方向

1. 冻结完整未触碰 NQ/Wiki-DPR 正式评估人口，并按 question hash 预先切成两个独立确认单元。
2. 复用完全相同的 Reader、quantum、classical checkpoint 和盲审协议，不在本轮结果上重训或调门槛。
3. 完整复制通过后，再单独做 finite-shot、资源和严格量子归因审计；这些不能与普通效用复制混在一次实验里。

推荐：方向1——完整真实数据复制设计

- hypothesis: 当前 `+10/+9 useful` 和正 bootstrap 下界来自可迁移的 frozen head，而不是256题偶然性。
- minimum experiment: 在运行前冻结完整未触碰自然数据人口、两个独立确认单元、身份排除、三臂 checkpoint、changed-set 双盲审阅和问题簇统计；本阶段只完成设计和预注册。
- success gate: 完整人口及两个独立确认单元都满足预注册的正向 useful/points 门；matched classical 结果独立报告。
- kill gate: 完整正式人口或任一必须的独立确认未过门，或出现身份/标签泄漏、来源漂移、无法回放，即停止这套 frozen head。
- resource estimate: 设计为低成本 CPU 工作；以后正式执行需要一次完整 Reader GPU 打分和所有变化题的双盲语义审阅。
- uncertainty: 剩余未触碰人口的变化率、review 工作量及效果大小未知。
- authorization: 已由前一预注册 gate 授权“设计与预注册”，尚未授权启动正式全量运行。

## 8. 证据索引

- 紧凑回执：`.ai-progress/workstreams/rag-selector/refs/reader_larger_untouched_blind_review_20261019_receipt.json`
- 证据卡：`.ai-progress/workstreams/rag-selector/refs/reader_larger_untouched_blind_review_20261019_evidence_card.json`
- 机制恢复：`.ai-progress/workstreams/rag-selector/refs/reader_larger_untouched_blind_review_20261019_mechanism_recovery.md`
- transport 回执：`.ai-progress/workstreams/rag-selector/refs/reader_larger_untouched_transport_20261019_receipt.json`
- 评分与独立审计代码：`scripts/collab/five_ideas/score_reader_larger_untouched_blind_review_256.py`、`scripts/collab/five_ideas/audit_reader_larger_untouched_blind_review_256.py`
