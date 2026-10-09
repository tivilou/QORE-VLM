# 32题×50段全部审阅完成：监督错误和真正可救的边界分开

时间精度：仅记录2026-10-09日期。本轮是现有训练材料的语义审计，不是新训练实验。

## 已完成什么

按原题序逐段阅读剩余第13—32题的817个段落，给出明确标签、简短判断理由和支持原句。此前783条标签及元数据保持不变，合计1600/1600，32题各50/50，未读为0。脚本只负责身份、引用、连接分数和统计，没有按标题/关键词自动分类，也没有把未读文本补成负例。

新增817条：直接支持33、部分支持105、不相关673、不确定6。累计：直接97、部分223、不相关1259、与目标矛盾3、不确定18。既有3条矛盾是参考/事件顺序冲突，不表示历史段落虚假。

原始case_study/support_review、参考答案、producer的1600个空label、原9个目标风险标志没有改动。既有旧头见过这些训练题；阅读时隐藏分数，但此前已看过原结果，不宣称独立盲评、三模型共识或official gold。证据仍为L0。

## 完整32题一览

下表direct等标签是本次单模型暂定支持判断；槽位差是min(5,候选direct数)−Reader已选direct数，不是答案正确率或生成增益。flag仅沿用已有9个，未flag也不等于已通过资格。

|题号|问题简述|D/P/I/C/U|Reader选D/P|暂定D槽位差|已有目标风险|
|---|---|---|---|---|---|
|1|Hot Tub拍摄地点|1/2/47/0/0|1/2|0|未flag，待资格|
|2|海上相遇优先权|0/5/45/0/0|0/2|0|conditional_scenario_missing|
|3|马cannon bone位置|2/7/41/0/0|1/4|1|未flag，待资格|
|4|CaSO4元素|0/3/47/0/0|0/2|0|serialized_formula_subscript_missing|
|5|Anarkali演员|3/2/45/0/0|2/1|1|未flag，待资格|
|6|热刺新球场位置|2/21/27/0/0|0/4|2|未flag，待资格|
|7|RuPaul第8季冠军|3/0/47/0/0|3/0|0|未flag，待资格|
|8|Days中Susan演员|5/0/45/0/0|4/0|1|未flag，待资格|
|9|土星环主要材料|6/13/31/0/0|3/2|2|未flag，待资格|
|10|最早迁往Madinah者|0/8/38/3/1|0/1|0|reference_order_conflict|
|11|总决赛单场三分纪录|0/1/49/0/0|0/1|0|未flag，待资格|
|12|NBA最长连胜|1/2/47/0/0|1/1|0|未flag，待资格|
|13|奥运两年交替开始|3/5/42/0/0|2/2|1|未flag，待资格|
|14|Andy兄弟演员|3/1/46/0/0|3/1|0|未flag，待资格|
|15|2016美国总统|4/17/28/0/1|0/3|4|election_vs_office_ambiguous|
|16|美国众议院主持者|6/3/41/0/0|2/1|3|未flag，待资格|
|17|吠陀时期发展|3/15/32/0/0|2/1|1|未flag，待资格|
|18|Oompa-Loompas演员|7/1/41/0/1|5/0|0|未flag，待资格|
|19|Midnight in Paris主题|4/7/39/0/0|3/2|1|未flag，待资格|
|20|Kuch Tum演员|3/1/46/0/0|3/1|0|未flag，待资格|
|21|美国最大树|9/7/33/0/1|3/1|2|未flag，待资格|
|22|1966世界杯West Ham球员|0/6/44/0/0|0/3|0|未flag，待资格|
|23|原版Willy Wonka演员|6/0/44/0/0|5/0|0|未flag，待资格|
|24|Pledge开始时间|0/20/26/0/4|0/1|0|event_start_ambiguous|
|25|Vivo首次印度发布|0/1/49/0/0|0/1|0|未flag，待资格|
|26|英国王位第一继承者|6/6/38/0/0|2/2|3|未flag，待资格|
|27|Alf妻子|5/3/42/0/0|3/1|2|未flag，待资格|
|28|FA Cup半决赛何时|0/1/41/0/8|0/1|0|question_reference_type_mismatch|
|29|海湾战争伊拉克领导者|8/12/30/0/0|0/5|5|未flag，待资格|
|30|Inherit the Wind牧师演员|0/3/46/0/1|0/1|0|adaptation_version_unspecified|
|31|Puerto Rico加入美国|7/25/17/0/1|2/3|3|acquisition_vs_constitution_conflict|
|32|联邦制度最终权威|0/25/25/0/0|0/4|0|jurisdiction_and_authority_scope_unspecified|

## 关键结论一：字串弱监督确实与支持关系不一致

在这批选定32题、1600段中，原weak-positive213段包括direct65、partial81、irrelevant58、contradictory2、uncertain7；原weak-negative1387段包括direct32、partial142、irrelevant1201、contradictory1、uncertain11。这是此审计样本的交叉表，不推广成434题或整个NQ的噪声率。

原答案字串命中既会把错人物、错事件、错统计口径当正例，也会漏掉别名、同义表达、跨句关系、隐式位置和对比句中的正确对象。量子/经典语义头目前仍沿用这类监督，可能把正确关系压低、把主题词推高；本轮支持这个监督问题的存在，但未通过重训因果实验证明它是唯一失败原因。

## 关键结论二：不是每题都该被要求换段

剔除已有9个风险题后，还有23题仅“暂未flag”，并未正式资格：
- 13题存在直接支持槽位差，共25槽；Reader当前48个direct，暂定容量上限73，差25。这只是段落证据容量，绝不是答对48/73题。
- 3题(11/22/25)的50段没有完整直接答案，仍有partial。不能要求selector挖出候选池没有的直接证据，也不能把partial全部作坏段。
- 7题(1/7/12/14/18/20/23)已保留min(5,direct总数)。其中18有7个direct、23有6个direct，Top5已全direct；漏掉其他direct不表示应该换集合。
- 9个更清晰的候选(5/6/8/13/16/17/21/26/27)同时有未选direct与已选irrelevant；资格确认后可优先研究direct-vs-irrelevant边界。
- 4题(3/9/19/29)只剩direct-vs-partial比较，Reader已选5段均有部分或完整支持。不能强制partial为负，也未证明换入更多direct会提升生成。

## 关键结论三：漏选具有不同机制，不能统一增幅解决

全部32题(含风险题)97个direct中，Reader第1—5为50，第6—10为17，第11—20为13，第21—50为17。因此47个未选direct不等于47个可替换槽：有的题已选5个direct，有的目标冲突。量子语义/经典语义/单调控制依然保持原选择；量子标量也未增加直接支持。

### 本轮具体case的新线索

1. **第13题奥运交替**：新段说明同年冬夏安排持续至1992，随后冬奥在夏奥两年后，可推得1994；Reader排名21、weak-negative。推理链与日期字串不是一回事，且“两年交替”不表示每一种奥运自己改成两年一次。
2. **第16题议长**：Speaker Ryan在2018主持相关职务以及辞任2019的时间信息出现在低排名段，另有泛讲议长权力的高分段。关系/任期比标题更具体，但“现任”只能按材料时点资格，不引用为2026事实。
3. **第17题吠陀**：新段把state formations与Ganges Valley urbanization明确连接，Reader14、weak-negative。很多同标题段只谈宗教、仪式或社会等级；对象属性关系比全篇主题相关性重要。中期缺城市也不直接否定后期城市化。
4. **第19题电影主题**：剧情明确“不同人向往不同golden ages”，Reader7、weak-negative；怀旧可由情节关系表达，不必含nostalgia字串。但当前已选5段都有支持，不应直接打压有用partial。
5. **第21题最大树**：General Sherman的“largest”在其他树的对比介绍中出现，部分排名16/19/27/36/39。必须区分最大体积、最高、最老、单物种最大、现存与历史已倒，具体比较可支持而标题名称不同不应一票否定。9个direct也不强迫塞进5槽。
6. **第26题继承者**：当前Prince Charles + Prince of Wales是heir apparent构成完整身份链，漏选direct在6/7/12/15名且均weak-negative；Charles, Prince of Wales这一参考字符串的格式会漏掉等价关系。结论仅限Elizabeth II仍在位的原材料时点。
7. **第27题Alf**：演员Judy Nunn的履历明确角色Ailsa与Alf结婚；Alf条目也说明婚礼，未选direct在11/16名且weak-negative。真实演员配偶、角色配偶、Alf Garnett其他剧与Home and Away不能混。
8. **第29题战争领导者**：Reader选5个partial，完整总统任期1979—2003或明确1991战争期间Saddam掌权的段反而低分。未选direct在6/7/9/14/23/26/32/40；高分政策/早期权力背景并非无用，但不能完整连接所问战争时点。任务关系+时间范围是关键，不单独靠Saddam姓名。
9. **第15/24/28/30/31/32题目标风险**：当选vs在任、初创vs正式采用、whenvs地点、影视版本、领土转移vs制宪、最终司法裁决vs笼统全部权力必须先限定。第31题1898领土取得可直接回答字面问法，但与1950参考冲突；保留原参考并隔离，绝不悄悄改标准答案后训练。

## 旧方法对照与正确下一步

|旧方法|160选段中的direct|partial|irrelevant|contradictory|uncertain|
|---|---|---|---|---|---|
|Reader Top5|50|54|48|2|6|
|量子语义头|50|54|48|2|6|
|经典语义头|50|54|48|2|6|
|单调压缩控制|50|54|48|2|6|
|量子标量控制|50|53|48|2|7|

它们是支持段计数，不是最终答案正确率。本轮没有新的selector收益，也没有证明量子优势。旧语义头不换Top5的事实未改变。

结合此前固定融合失败、经典/量子语义训练坍缩修复、非选择性分数压缩、近/深排名救援的差异和这次完整文本审计，**唯一推荐下一步是支持资格感知的边界监督，而不是再加大残差或直接扩大训练**：

- 先显式资格化问题条件、时间/版本/指标口径、支持置信度，再设计可审计target compiler与合成测试。
- 从9个“漏direct且已选irrelevant”的候选验证目标构造；保护原Reader已选direct，既有有用partial保留或弃权，不强制direct-vs-direct排序；其他4题单独考察分级而不硬设负类。
- 缺失、uncertain、冲突题不产生默认负标签；没有direct的题不造正例；最多5的容量限制和多正确段不设唯一gold集合。
- 量子主线不变：仍从Reader问题条件化语义表示进入可训练量子头，并与同预算经典语义、标量量子、单调压缩/零残差比较。能经典模拟不构成否决，效用和理想实现资源优势须分别验证。
- 成功门：身份/范围/引用/目标资格可追溯，正例保留与缺失/partial隔离测试全过，干净可救边界足够。若目标或对照资格仍不足，停在资格审计；不靠已见100题Silver调参。
- 后续434题要完整监督资格，并预注册fresh347/87及独立100题评估；当前32题审完不代表434题完成。旧头见过这批，不当heldout。

本轮只完成用户指定的剩余817段及汇总，未启动target实现、重训、Reader/Generator推理，也未下载模型或Wiki-DPR。

## 可复核记录与验证

[全1600入口manifest](../../.ai-progress/workstreams/rag-selector/refs/semantic_supervision_20261009T045728Z_complete_review_manifest.json)引用不变的783条旧文件与817条增量，并锁定实际SHA。两个文件各小于1MiB，不把约1.5MiB合并文件塞入Git。每条新判断包含原case/盲视图ordinal/候选ID/原句/理由。

[817条显式标注](../../.ai-progress/workstreams/rag-selector/refs/semantic_supervision_20261009T045728Z_complete_annotations.json) · [增量审阅](../../.ai-progress/workstreams/rag-selector/refs/semantic_supervision_20261009T045728Z_complete_review_delta.json) · [每题原排名/分差/全部方法连接](../../.ai-progress/workstreams/rag-selector/refs/semantic_supervision_20261009T045728Z_complete_join.json) · [机械汇总](../../.ai-progress/workstreams/rag-selector/refs/semantic_supervision_20261009T045728Z_complete_diagnostics.json) · [验证与回滚](../../.ai-progress/workstreams/rag-selector/refs/semantic_supervision_20261009T045728Z_complete_verification.md)。

检查覆盖、引用、源哈希、旧783元数据、9标志、所有160选段、分片manifest和恶意篡改共24项。本地首次测试拦截Windows文本换行导致的delta哈希不一致；改为显式UTF-8/LF字节序列写入，添加字节保留属性，再重跑通过。原文件未改，标签未因此重判。
