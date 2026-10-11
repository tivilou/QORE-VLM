# 受保护边界监督：映射、审计与训练接口

此交付完成离线映射和独立训练接口，未实现或授权真实训练runner。现有Reader、弱标签trainer、Generator、评测器、残差幅度均不改。

## 实际数据与结果

固定434题，每题50段；347训练题的原主审意见与94题439段新裁决精确按问题及段落ID合并，87验证题所有标签保持null。使用正文/标题哈希和不可变标准答案验证，不以答案字符串推断标签。38held不能通过请求升级。

真实映射审计：90题、464候选对、129最大替换槽位；真实资格对0。这里的候选对只是已有临时意见所示的机会，不能进入训练。现有残差上界0.25×题内Reader分数标准差下，仅95对小于两个候选相对纠正的宽松极限；369对仅靠当前有界头无法翻转，不据此放大幅度。

## 三个独立入口

- `support_boundary_bridge.merge_opinions`：原意见与裁决的无损映射，全部资格默认pending/held。
- `apply_explicit_qualification`：要求单独手工资格请求绑定确切scope/item版本；重新打开真实review证据，核对身份、四轴、引用、声明暴露和哈希。相同文件哈希、同模型一致或暂定接受均不会自动资格化。请求不是独立性的认证。
- `protected_boundary_learning`：只接受显式的漏选direct→已选irrelevant对，题内权重和为1；配对softplus损失＋原分数锚点＋保留槽位保护损失。量子和经典语义头复用同一接口。没有合格对就跳过整个optimizer step，避免纯weight decay。

选段接口仅允许替换被明确资格化为无关的原Top5槽位，原Top5其余槽位全部保留，包括partial、held、未知和direct。正例不能作为负例；没有对时严格回到原Top5。此保护mask仅是训练/诊断接口，含标签信息，不能作为正式在线selector或无标签泛化能力的证据。正式评估必须另行冻结不依赖标注的选择规则，比较无保护选择与保护诊断，并报告真实损伤率。

## 研究人员本地审计

有本次私有输入的项目根目录运行：

```bash
bash scripts/collab/five_ideas/run_support_boundary_bridge_audit.sh
```

没有网络/模型调用或自动下载。默认寻找本地 `.cache/support_boundary_bridge/20261011-source/case_study.json`、原主审和裁决包；路径或哈希不符即停。输出按UTC时间戳写入 `.cache/support_boundary_bridge/`，不覆盖旧文件。文件中的 `summary.json` 可公开前审查，其余原文/身份/资格材料为私有附件。

`qualification_request.json` 包括候选题scope、所有原Top5项和拟纠正项的精确绑定，默认不含任何资格见证。资格记录只承认独立意见实际覆盖的内容；没有覆盖的已选direct仍会阻断整题。不得仅因与主审一致就自动提升其他591项；应有明确正文/scope审核及对原有支持的保护记录。补齐请求后可显式传入 `--qualification` 编译复核，仍不授予训练消费权限。

## 合成验证与限制

13项聚焦测试验证实际证据回读、held/外部/partial排除、验证隔离、保护检查、梯度方向及null。另有一个固定人工合成样本，两种头各2步，保存266个完整detached数组（scores、pooled、encoding、observable、gate、residual、梯度、更新前后参数、optimizer moments），重开并重放4次更新。合成资格仅是接口测试，不是实际见证。

fixture schema为 `rag.protected_boundary_engineering_fixture.v1`，不是正式 `sample-trace.v2`。真实Reader hidden states未运行，故formal实验架构状态仅approved；正式训练/验证runner和trace readiness仍待设计。当前不能把87题验证补标、100Silver调参或保护mask oracle回放悄悄纳入训练。

下一步只需针对候选边界及原Top5保护项完成显式资格记录，优先核实范围干净且现有残差可达的候选。若没有安全且可达的足量对，报告瓶颈；不降标签门槛、不强制每题五个正例、不自动加大残差。
