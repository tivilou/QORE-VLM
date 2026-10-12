# Frozen support boundary Reader screen：训练有效，但没有改善 Top-50 → Top-5

- 日期：2026-10-12
- 实验目录：`five_ideas/protected_boundary_reader_screen_100/20261012T151854474980Z/`
- 精简证据：本地缓存 `.cache/protected_boundary_result/20261012T151854474980Z/summary.json`
- 完整 case study、逐题分数和张量轨迹：18083 同目录，上传回执已逐文件复核。
- 证据等级：L0，固定100题暴露面板；不能据此声明 L1/L2 或量子优势。

## 结果

| 方法 | 有用段/题均 | 直接证据/题均 | 三模型全同意/题均 | 固定 Silver 重合/题均 |
|---|---:|---:|---:|---:|
| Frozen Reader Top-k | 203 / 2.03 | 138 / 1.38 | 171 / 1.71 | 355 / 3.55 |
| Quantum semantic | 203 / 2.03 | 137 / 1.37 | 170 / 1.70 | 352 / 3.52 |
| Classical semantic | 203 / 2.03 | 138 / 1.38 | 171 / 1.71 | 353 / 3.53 |

预注册门槛没有形成正向证据：量子相对 Reader 的有用段差值为 `0.00`（100 平局），直接证据 `-0.01`，Silver `-0.03`；相对同预算经典头的有用段和直接证据也为 `0.00/-0.01`。`selection_diagnostic=false`，`quantum_attribution_diagnostic=false`。

实现与实验边界有效：固定 DPR Reader revision、代码无脏路径；训练只用了冻结的59题/228对，评估是未加目标 mask 的100题；Reader、检索、Generator 和 evaluator 均未改变或调用。完整轨迹验证通过：3254 个数组、318 次 head 输出回放、12 次优化器更新回放，`validation.status=passed`。

## 训练发生了什么

这不是“梯度没走”或“量子线路完全没动”。量子头的平均 pair loss 从 `3.9783` 降到 `3.8535`，经典头从 `3.9787` 降到 `3.8564`；228 个训练对中量子头有182个相对初始差距改善，经典头有226个改善。但量子头和经典头的 pair gap 仍分别没有一次变成正值（pair inversion 为 `0/228`），平均 gap 仍约为 `-3.79`，所以训练后的修正无法把被保护的正例真正推过错误候选或 Top-5 边界。

修正不是全体相同平移：最后一轮量子/经典最大有效修正约 `0.51`，逐题最大变化约 `0.67`；但 Top-5 第5与第6名的中位分差约 `0.31`，训练正例与替代负例的平均初始差距约 `-3.97`。这说明少数近边界题可以发生换位，远离边界的低分正例仍不可达。量子头第3轮编码饱和比例达到 `0.915`，经典头为 `0.280`；量子线路没有表现出相对经典头的可归因优势。

## 逐题机制证据

评估集量子头只改变4/100题，经典头改变5/100题。变化几乎都是边界交换，而不是命中新证据：

- `what was the religion in the virginia colony`：量子头以 rank15 的主题相关段替换 rank43 的直接教会证据，直接证据 `4→3`。
- `where did the book small steps take place`：rank13 的 Silver 段被 rank15 的同主题段替换，Silver `5→4`，有用段不变。
- `who wrote the song to make you feel my love`：rank38 的 Silver 段被 rank29 的 `I Feel Love` 主题段替换，Silver `5→4`，有用段不变。
- `what episode does caroline get turned into a vampire`：rank15 与 rank3 互换，所有诊断指标不变，属于近似并列噪声。

经典头还发生了两次同指标交换（Cool Hand Luke、SIP）。因此本轮暴露出的是“训练目标能够产生小幅候选相关修正，但没有学到可靠的直接支持判别”，而不是量子模块单独失效。

## 累积经验与下一步

结合此前语义头坍缩、修复后只救回一个证据、Q-ARCG 无净增益以及本轮的0次 pair inversion，当前家族不能继续靠增大残差、增加 epoch 或重新跑同一面板来破局。下一步先做一个确定性的 **pair-feasibility audit**：逐对计算“正例要进入 Top-5 所需的最小修正”、现有残差上限、初始 Top-5 边界距离和跨题泛化余量，区分可达但未学会、可达但会误伤、以及结构上不可达三类。这个审计完成前不实现新 selector、不调 Silver、不把本轮结果升级为正向 idea；之后只有能改变“不可达/主题相关误换”因果链的量子 Reader-integrated 机制才进入候选。
