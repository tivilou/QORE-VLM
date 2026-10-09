# 9题范围复核：先区别真正漏选与不够完整的支持

时间精度：2026-10-09，仅日期。本轮是主审再检查，不是独立复核已完成。

## 工作本身和本次改变

我们固定每题召回50段，只研究如何选5段。Reader相关分仍是锚点；量子语义
头的模型与线路都没改。旧的答案字串弱监督会错标，优化路径修好也不保证
真正识别证据。所以本轮先看上一轮9题28个边界候选是否真的适合作为监督。
输入只用既有旧训练审计原文与标签；不运行训练、Reader或Generator，不下
数据。100题Silver是评估材料，不进这次监督审查。

|上一步|本次|
|---|---|
|候选对机械编译28对/最多10个无关替换槽位|主模型逐题读54个边界/保留见证，提出范围与标签异议|
|原标签暂定、资格全部pending|另存主审意见与abstention覆盖层；原1600标签/9 flags不改|
|独立复核未准备|9题54段完整原文紧凑包已备好，未启动第二审阅者|

## 完整9题观察（均是主审意见，不是gold）

|题号|问题简述|发现|当前建议|
|---|---|---|---|
|5|电影Anarkali演员|最初Nargis选角和音乐剧演员不是1960已发行电影的Madhubala|2对保留待独立复核；版本范围显式记录|
|6|新Tottenham球场在哪|Tottenham/旧球场选址有证据，但与Haringey行政区参考的粒度关系未认证|暂缓，不按行政区字符串强判|
|8|Susan扮演者|Eileen/Brynn/Stacy有不同明确历史时段，问句没有日期|暂缓，不把2018新演员段直接当唯一正确目标|
|13|奥运两年交替何时开始|1994对应冬夏交替，每类仍四年；漏选段可由1992+2推断|1对保留待独立复核；推断方式需认可|
|16|House主持官|Ryan任职证据是2015—2018等时点，原问句缺国家/日期|暂缓，不从参考反推采题时间|
|17|Vedic时期发展|漏选段只说国家形成和城市化开始，未给large，且有后继时期边界|原direct提出partial异议，暂缓两对|
|21|美国最大树|largest未明说体积、现存、单干，历史更大树/种类vs个体也需区分|暂缓，不混高度/寿命/体积|
|26|王位下一继承人|Charles是文中Elizabeth II在位时期状态；一段只给Prince of Wales职位没给姓名|暂缓；职位段提出partial异议|
|27|Alf配偶|角色婚姻和演员真人婚姻不同；Martha也有明确角色婚姻；一段只说编剧计划|1对保留；计划段提出partial，保护有效另类答案|

主审再读54段均留下明确判断、连续原句及简短依据，但不能因为读了第二遍
就把同一个模型算成独立第二审阅者。已看过原标签/过去结果；本次隐藏分数，
不宣称独立盲评或三模型共识。证据仍L0。

## 逐样本追踪与计量边界

按预定9题及原boundary/retention并集审读全部54段，不事后只挑成功样本。
以下异议说明具体变化，完整记录在JSON。candidate_index是0基索引，不是排名。

1. 题17/index11：原句“a new wave of state formations, linked to the beginning
   of urbanization in the Ganges Valley”缺少large，且后文说Vedic之后的
   Mahajanapada period。不能把有两项相关信号直接等同完整“大型城市化国家”。
2. 题26/index37：原文明确Prince of Wales是heir apparent，但在“The current
   Prince”处截断，没有Charles。职位正确不等于已经完整给出具体人的答案。
3. 题27/index32：“Writers decided that they pair would marry early on.”是
   编剧计划；附近Nunn自己的婚礼又涉及演员真人。计划不是已发生剧情事件。
4. 题27/index6明确Alf married Martha；即便参考只给Ailsa也不能把它自动
   变成负例。我们保持先前direct暂定，不强迫唯一Silver/参考答案集合。

[已测事实] 原始28候选对记录仍在；新覆盖层不改原label，仅新增abstained/
pending。主审暂缓6题并排除计划段后，剩3题4个暂定对、3个替换槽位。
编译器重新打开全部数据仍输出0资格通过draft；没有第二审阅者记录。
这个“4”是审阅意见导出的候选量，不是验证集成绩或模型上限的证明。

[已测事实] 本地新验证器19项+旧编译器40项共59测试通过；旧1600审计24项
保持通过。私有包51,195字节，9题54段，文本和引用哈希核对通过。测试只
验机械完整性，不替代语义判断。新训练/隐藏特征变化/梯度/生成答案阶段
不适用，因为本次没有执行；不存在新的sample-trace.v2模型实验。

## 累积经验：为什么这一步比马上重训重要

此前稳定carryover是Answer Scorer。固定span融合、广泛集合重排及多种历史
方法未稳定胜过它；优化器修复后语义头仍可能只压缩分数。完整1600初审
发现弱正负例噪声，但本轮又提醒：模型审阅标签本身也会把“部分正确”误
当“完整支持”。因此不能把修正字串标签当成监督问题已经解决。

[机制解释] 训练目标需要同时校验问题条件、段落完整支持和已有正确证据
保留；否则量子头学得更强也可能更准确地优化了错误目标。
[尚未证明] 这些异议是否获独立审阅认同、是否让量子头优于匹配经典头、
是否推广到新题、是否有理想量子资源优势，都没有得到实验确认。

## 唯一推荐下一步与门槛

用新的第二审阅者会话审完整9题54段，包括暂缓/有争议案例，而不只审3题
“好看”的边界。任务包只给原题、参考、标题、全文和ID；不提供分数、排名、
方法选择、旧标签或主审观点。由用户手动启动Claude Code，不调用不稳定API。
上下文约51KB正文加短任务/模板，避免让昂贵模型重新读整个项目。

成功门：全部9/54显式判断、准确引用、真实身份/模型与已有暴露披露、逐项
解释六个范围；再由主agent比较分歧并裁决，不因validator通过就自动晋升。
暂缓门：时间/粒度/指标仍无根据或支持不完整就保留abstention，不扩大残差。
完整434监督、新347/87协议与保护损失设计仍是后续训练前的独立门，不能用
这54个见证代替全队列；100Silver不调参。已有资料审查/整理已获开始授权，
第二worker没有启动；真实拟合仍不执行。

## 证据
- [主审54条意见](../../.ai-progress/workstreams/rag-selector/refs/support_scope_priority9_20261009_primary_recheck.json)
- [版本化暂缓覆盖层](../../.ai-progress/workstreams/rag-selector/refs/support_scope_priority9_20261009_qualification_hold_overlay.json)
- [范围/候选汇总](../../.ai-progress/workstreams/rag-selector/refs/support_scope_priority9_20261009_summary.json)
- [数据清单](../../.ai-progress/workstreams/rag-selector/refs/support_scope_priority9_20261009_manifest.json)
- [手动独立任务](../../scripts/collab/five_ideas/support_scope_review_packets/20261009-priority9-independent/TASK.md)
