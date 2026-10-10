# 全训练队列支持监督导出（不训练）

## 合作者运行

已配置 `QORE_EXCHANGE_TOKEN` 后，在项目根目录执行：

```bash
bash scripts/collab/five_ideas/run_support_supervision_cohort_export.sh
```

脚本默认配置已内置；不指定 Python 绝对路径或项目位置。它使用合作者
已有的 NQ train、Wiki-DPR 索引和 DPR 模型缓存；缓存缺失就报错，
不下载语料/模型、不重建索引。必要时只从 18083 取 hash 锁定的
`input_manifest.json` 和 `cohort.json`，不取旧训练头的检查点。
`--wiki-dpr-cache` 为非默认缓存位置的可选参数。

## 这次保存什么

- 上一轮 512 题中可用的全部 434 题，不重新挑题或替换失败样本。
- 每题原问题、参考答案、原检索顺序的 50 段全文/标题/ID/检索分。
  候选身份按 50 段的多重集合核对：压缩 Wiki-DPR 内积在 float32 分辨率下会
  出现仅差约 1 ULP 的真实并列，此时顺序在重跑间不稳定；导出只在并列分数
  容差 `1e-3` 内接受顺序互换，缺段或真正重排仍会失败，弱标签按候选身份对齐。
- 固定 DPR Reader 的全部 50 个 relevance 分数、排名及 Top-5。
- 无分数、无弱标签、无排名、无选中信息的独立审阅模板。
- 未来训练/验证分区 347/87；旧模型看过两边，**不算旧模型的独立验证**。
- 完整 JSON 和可读 Markdown；不运行 Generator，故没有新的生成答案。

阶段链为 data -> retrieval -> frozen_reader_scoring -> baseline_selection ->
pending_review。保留完整标量值和输入文本，不宣称记录了隐藏张量，
也不宣称这是新量子模型实验。新锁定的全文不能回溯证明历史 432 题文本一致。

## 新的判断轴

`support_axes_review.json` 全部待填，不搬运或覆盖此前 1600 条标签。

| 字段 | 意义 |
|---|---|
| question_support | 在明确问题范围下，单段是 direct / partial / irrelevant / contradictory / uncertain |
| reference_coverage | 对参考答案是 exact / equivalent / valid_alternative / not_supported / unresolved |
| inference_type | explicit / minimal_arithmetic / intra_passage_composition / none / unresolved |
| source_grounding | single_passage / requires_external_context / unresolved |
| scope | 明确 time、location、version、entity、metric、reference_relation；不适用也要明写 |
| qualification | pending / held / qualified，独立保存审阅者、来源哈希、绑定的标注版本和先前意见暴露 |

题目不限定日期时不得猜测日期；有效的替代答案不因缺少参考字符串而
被判负。简单算术或同段组合推理可单独记录；靠别段补全名字/实体则暂存待审。
判断轴不是新的真值：审阅者还需实际读段落并检查范围、引用和独立性。

## 非训练目标草稿

每个 qualified 题目/段落要求至少两个真实、不同身份/来源的审阅证据，
包含 primary_review 和 independent_review。独立证据不能是看过
先前挑战意见的续轮；每条 witness 绑定当前完整 annotation 哈希。
这些声明通过机械校验也不自动证明内容正确或审阅真正独立。

仅 qualified missed direct 与 Top-5 中 qualified irrelevant 构成候选对；
Top-5 中 qualified direct 受保护；partial、uncertain、contradictory、
外部上下文依赖、未解决范围和未确认段落绝不默认二值负例。每题对权重
归一化；同时记录可替换槽位数量。草稿始终 `training_authorized=false`
且 `training_consumable=false`，后续训练需要另行决定与独立数据门。

审阅后可离线运行纯标准库编译器（输出文件必须是新文件）：

```bash
python -m applications.rag.support_supervision_axes TRACE_JSON REVIEW_JSON NEW_DRAFT_JSON
```

现有 `support_qualified_boundary_targets.py` 未改，不降低旧版资格要求。
Silver 100 题仅以问题 hash 排除训练，全文、标签、三模型意见不进入导出/训练。

## 上传、失败恢复和提交

输出：`exchange/five_ideas/support_supervision_cohort_export/<UTC时间戳>/`。
脚本自动在 18083 创建同名目录、上传全部正式输出，回执校验字节数/hash。
JSON/Markdown 全文、审阅、目标和分区始终 exchange-only，与大小无关。
仅 `summary.json`、`report.md`、`run_metadata.json`、`upload_manifest.json`
可提交 GitHub，分别不超过 1 MiB；不要强制加入被忽略的 raw 文件。

导出中断时保留时间戳 `.partial.jsonl`，不会生成成功 summary 或自动上传
不完整数据；它供错误调查，不支持跨版本自动续跑。上传失败则保留完整
本地输出，无须重跑 Reader：

```bash
bash scripts/collab/five_ideas/run_support_supervision_cohort_export.sh --upload-only 已完成的输出目录
```

本地/服务器只做 synthetic fixture 和 schema/replay 测试。真实 434 题导出
由合作者执行；导出完成不是完成语义标注，更不是量子优势或筛选效果证据。
