# 9题/54证据独立范围复核（手动启动）

## 唯一任务
从完整原文判断问题范围及单段是否支持答案，挑战而非附和前一个审阅者。
这是旧训练诊断材料的质量审计，不是100题Silver测试或模型收益实验。
只读本包；原文完整，但不是每题全部50段，不能宣称全池覆盖或唯一oracle。
54段为预先冻结的边界/保护见证集合，由先前暂定判断取样，有选择偏差。
你看不到原标签、候选排名、Reader分数、是否被某方法选中或主审结论。

## 逐题/逐段规则
1. 各题显式解释time/location/version/entity/metric/reference_relation六个范围。
   区分question本身确定的范围与reference才暗示的范围。参考可能是过时/不全
   答案，不以参考字符串作为段落标签规则，不补造题目时间、国家或比较指标。
2. 对每段独立给direct/partial/irrelevant/contradictory/uncertain。direct要求
   在明确或清楚声明的条件范围内给足实质支持；记录隐式推断。主题相关、
   仅给职位/计划/部分属性不自动是完整支持。partial不是负例。原句若缺少
   必要上下文就abstain/uncertain，不读相邻段补全。引用必须是段内连续原句。
3. 允许不同有效答案和已有正确段保留，不强制唯一参考Top5或direct-vs-direct。
   时间/演员/任职/同名改编/种类vs个体/指标/真人vs角色需特别留意，但不要
   因这些类别就用批量规则给标签。conditional不等于正式scope resolved。
4. 原记录、问题和参考均保持原样；所有纠正只是你的版本化审查结果。

## 输出及停点
使用context/review_template.json的结构填写全部9题、54段，不留null或批量
默认标签。写output/review.json、output/result.md、rounds/0001/result.md。
每段给简短明确理由，支持性判断附准确原句；每题说明条件/歧义，不需
复述整段，也不用生成论文或新idea。reviewer_id/model_id记录实际身份；
prior_review_exposure如实披露，不能因使用新目录就宣称独立盲评已证明。
只写rounds/**、output/**和state.json，不启动新轮、不改任何输入/项目/进度。
遇到证据缺口记录unresolved/uncertain，不请求数据下载或API调用。
产出后停下；主agent另做身份/引用/差异检查和语义复核，不自动晋升训练资格。
