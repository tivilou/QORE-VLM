# RAG 研究日志

## 当前研究总览

- 最新实现：[训练监督与选段边界审计](20261009T-date-only-semantic-reader-supervision-audit.md)。固定原训练32题×50段，不训练、重放旧头，保存全文/数值与独立盲审材料；36项合成回归和上传预检通过，等待合作者导出与支持审阅。

- 最新结果：[语义头修复后的 100 题实验](20261009T-date-only-semantic-reader-repair-result.md)。训练通路已生效；量子语义有用/直接证据 `204/139`，Reader `203/138`，仅一题改善、区间触零，仍未过门。下一步先审查训练监督和选段边界判别，不直接放大残差或加深线路。

- 最新修复：[语义评分头训练通路修复与预检](20261008T-date-only-semantic-reader-training-repair.md)。解耦衰减与按维度缩小投影学习率后，两条保存的训练样本上量子/经典头均保留候选差异；本地、服务器 44 项测试通过。已准备相同 100 题的一键重跑脚本，真实筛选收益尚未验证，仍为 L0。

- 最新结果：[量子语义头训练退化分析](20261008T-101111Z-quantum-semantic-reader-training-collapse.md)。有效/直接证据数与 Reader 都为 `203/138`；两种语义头权重和残差退化，100 题全部回到原分数。下一步先修训练通路与预检，不直接放大实验；仍为 L0。

- 最新实现：[量子语义 Reader 头初测](20261008T-date-only-quantum-semantic-reader-implementation.md)。保留相关分锚点、引入问题条件化隐藏表示，对比量子语义/经典语义/量子标量；主指标为有效证据命中。合成验证通过，真实实验待合作者运行，仍为 L0。

- 最新更正：[Q-ARCG全部换位题语义分析](20261008T-date-only-qarcg-paired-case-analysis.md)。固定Silver集合含填充段，集合下降不等于证据下降；Q-ARCG/Reader有用证据`204/203`、直接证据`138/138`，仍无可靠提升。下一假设是保留锚点、引入问题条件化语义表示，训练来源待补齐。

- 最新部分证据审计：[Q-ARCG 100题门控残差初测](20261008T-date-only-qarcg-screen-trace-audit.md)。Reader / Q-ARCG / 经典控制 Silver交集为 `3.55 / 3.49 / 3.53`，本轮未提升；先补齐训练与运行来源文件，再归因，不启动扩大实验。

- 当前目标：在完整真实数据上提出并验证至少达到 `L2_reproducible_utility` 的 RAG 改进机制，并保留通往 L4 严格机制归因的路径。
- 当前证据上限：在线 RAG 新机制仍未超过 `L0_diagnostic`；已知稳定正向结果是 Answer Scorer 相对 QORE-DPR 的提升，Phase 9J 只证明 gold-defined 条件下的 reader-span headroom。
- 最新结论：100题 silver-oracle 冻结 Generator ceiling 显示，silver 正例成员数虽从 QORE 的 broad/direct `177/119` 提高到 `307/222`，最终 F1 只提高 `+0.014143`（95% CI `[-0.041715,+0.071531]`），EM 下降 `-0.04`，未通过预注册 gate。Top-k 的 silver 成员数高于 QORE 但最终 F1 更低，说明 evidence retention 不是可靠的任务效用代理。三模型标签仍是 L0 silver 诊断，不是 gold 或可部署排名。
- 当前研究边界：只优化固定 Top-50 -> Top-5 selector；Generator、检索、evaluator 和答案实现不属于本轮候选。下一候选必须提供独立、可追溯且 gold-free 的证据关系信号；若只能继续依赖 token overlap、Answer Scorer 融合或 generic diversity，应停止该 selector 家族并先处理 provenance/calibration 阻塞。

## 时间线

- [2026-10-09 | 先验证训练标签是否真能支持答案，再设计量子边界目标](20261009T-date-only-semantic-reader-supervision-audit.md)

- [2026-10-09 | 修复后只多救回一个证据，定位分数压缩与边界限制](20261009T-date-only-semantic-reader-repair-result.md)

- [2026-10-08 | 修复语义评分头的衰减与饱和，先检查是否学起来](20261008T-date-only-semantic-reader-training-repair.md)

- [2026-10-08 | 量子语义 Reader 初测结果：训练退化，先修学习通路](20261008T-101111Z-quantum-semantic-reader-training-collapse.md)

- [2026-10-08 | 量子语义 Reader 头实现与回放验证](20261008T-date-only-quantum-semantic-reader-implementation.md)

- [2026-10-08 | Q-ARCG逐题分析及指标更正：集合交集与证据命中分开](20261008T-date-only-qarcg-paired-case-analysis.md)

- [2026-10-08 | Q-ARCG 100题 trace审计：选择门槛未通过，训练来源待补齐](20261008T-date-only-qarcg-screen-trace-audit.md)

- [2026-08-29 | Gold evidence alignment v1：解析修复成功，但标注源覆盖不足](20260829T-082824Z-gold-evidence-alignment-v1.md)
- [2026-08-29 | DPR positive gold alignment v2：问题身份预检失败](20260829T-124911Z-dpr-positive-gold-alignment-v2.md)
- [2026-08-30 | DPR question-join preflight：确认零连接并增加确定性诊断](20260830T-044609Z-dpr-question-join-preflight.md)
- [2026-08-30 | DPR NQ-test gold source repair：修正 dev/test population 混用](20260830T-date-only-dpr-nq-test-source-repair.md)
- [2026-08-30 | DPR NQ-test gold preflight：问题已全连接但正例上下文覆盖不足](20260830T-112303Z-dpr-nq-test-preflight-coverage.md)
- [2026-08-30 | DPR gold-info 字段诊断：确认 23 条空 context](20260830T-134402Z-dpr-gold-field-diagnostics.md)
- [2026-08-30 | DPR gold-info 结构诊断增强：嵌套与候选正例字段](20260830T-160411Z-dpr-gold-structure-diagnostics-enhanced.md)
- [2026-08-31 | DPR context-only eligible universe：全量 source preflight 通过](20260831T-083058Z-dpr-eligible-universe-preflight.md)
- [2026-09-01 | DPR gold context 与当前 Wiki-DPR 严格身份映射失败及来源语义核验](20260901T-031929Z-dpr-eligible-universe-mapping-failure.md)
- [2026-09-01 | 全量官方标题与 Wiki-DPR 标题存在性诊断](20260901T-084025Z-dpr-title-presence-audit.md)
- [2026-09-02 | RAG selector 五题 case study 机制诊断](20260902T-case-study-selector-mechanism-diagnostic.md)
- [2026-09-02 | RAG selector 50题 case study 机制诊断](20260902T-121037Z-rag-selector-case-study-50-analysis.md)
- [2026-09-03 | 三模型证据盲评与答案实现分层](20260903T-042356Z-rag-evidence-panel-50-analysis.md)
- [2026-09-04 | 50 题详细 case study 改进方向分析](20260904T-162256Z-rag-case-study-improvement-analysis.md)
- [2026-09-05 | selector-only QORE-QES 改进方向](20260905T-selector-only-qore-qes.md)
- [2026-09-08 | Anchor--Residual 固定 50 题预检失败](20260908T-044401Z-anchor-residual-preflight.md)
- [2026-09-15 | Silver-oracle 100题冻结 Generator ceiling：证据命中增加未传递为稳定回答收益](20260915T-date-only-silver-oracle-top5-ceiling.md)
