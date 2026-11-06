# 双数据集启动被数据发布检查拦住：处理说明

## 当前报错是什么

`DATA_RELEASE_NOT_READY`表示项目负责人尚未交付完整正式数据release。它发生在模型运行、依赖安装和脚本自动下载之前，不是DPR/BGE计算错误、GPU错误，也不是凭此就能判断私服离线。

目前交付完成的是八项单模型增强加两项双模型融合的一键实验代码。正式NQ-open、TriviaQA的合格输入及段落标签包尚缺；代码交付不能当作正式数据交付。师弟无需替项目负责人自行重建划分、候选或标签，也不要改`status: ready`绕过检查。

## 现在运行什么

保留当前下载和已有实验目录。同步最新`codex/top8-dual-scorer`代码后，先用不联网、不写实验目录的诊断：

```bash
bash scripts/run_two_datasets.sh --diagnose
```

如两个模型已在当前项目的`models/`下，只读逐文件核验固定清单：

```bash
bash scripts/run_two_datasets.sh --diagnose --check-local-models .
```

输出包含`MATCH/MISSING/SIZE_MISMATCH/SHA256_MISMATCH`。此步骤会顺序读模型文件做哈希，不导入模型、不下载。ModelScope的`Snapshot ready`或`master`不证明与固定Hugging Face revision字节相同；匹配说明已下载文件在字节上可复用，当前整包安装器仍按固定归档工作，不会仅凭同名本地目录跳过校验或自动拼包。

诊断返回0表示诊断完成，不表示正式release就绪；看`category/ready/releases`。`--plan-only`也只是配置检查，不探测私服。正式数据缺失时普通启动仍退出2。

若只是准备已曝光B1复现输入，可显式执行：

```bash
bash scripts/run_two_datasets.sh --datasets triviaqa --population reproduction --prepare-only --workspace /root/qduet-assets-b1
```

这会通过已安装的私有访问配置下载并核验B1包，不训练；它不修复正式全量缺口。去掉`--prepare-only`才是B1复现实验，不是正式NQ-open＋TriviaQA。私有`junior.json`由项目负责人经私有渠道安装，不放Git、报错文档或聊天公开内容。

启动器现在优先使用已激活conda/venv的Python，或显式`QDUET_BOOTSTRAP_PYTHON`，然后才回退系统Python。该Python运行标准库协调器；科学运行环境仍由既有私有配置／固定依赖流程选择。`py310`与系统`python3.11`的选择不是此次release报错的根因。

## 历史资产在哪里

以下是负责人现有3080资产服务器的路径，不是师弟机器的路径，也不要求师弟SSH访问：

| 资产 | 已记录路径与用途 |
|---|---|
| TriviaQA冻结身份、角色 | `/srv/qore-research-assets/joint-witness-top50-20261021/questions/`；冻结计数61689/7891/7606，512训练、128曝光开发、61049训练保留 |
| 原Top50桥 | `/srv/qore-research-assets/joint-witness-top50-20261021/export-v2/`；8531题、426550候选，含512训练、128开发及全部7891验证 |
| 桥与回放收据 | `/srv/qore-research-assets/joint-witness-top50-20261021/verification/`；`artifact_manifest.private.json`、`bridge_receipt.json`、`full_replay_receipt.json` |
| 固定原始TriviaQA镜像 | `/srv/qore-research-assets/triviaqa-pinned/`；train/validation/test Parquet |
| 可安装B1两模型归档 | `/home/Q-DUET-VLM/top8-private-inputs-20261103-v001/top8-private-inputs-two-models.tar` |
| 私服B1对象 | `five_ideas/top8-junior-inputs/20261103-v001/top8-private-inputs-two-models.tar`；2753105920B，SHA256 `29a4d5808a330b45f5dcb307ffca59de498d020ec7d70409e1716eedf3566a32` |

2026-11-06负责人只读检查：上述桥、原始数据目录、归档均存在；已有凭据的包HEAD请求返回200，Content-Length与清单一致。没有重新下载／重核整包SHA，也没有因此认证师弟机器到服务的连通性。身份清单、原始文件、候选桥和正式语义标签release是不同资产。

问题冻结和历史桥的项目收据见`reader_joint_witness_question_freeze_20261021_receipt.json`及`reader_joint_witness_bridge_completion_20261021_receipt.json`；历史5条截断候选事实保留，不能借打包删题或改文本。原冻结题数不是原始下载的总题数；不同镜像须逐题身份、角色映射，不能按下载文件顺序套划分。

## 对师弟八个问题的答复

1. **应跑哪个范围？** 目标仍是两套正式实验。现有默认拦截符合当前缺包事实；可选B1仅用于复现或联调，不能用其结果宣称全量完成。
2. **TriviaQA候选与划分？** 上表给出源目录。现有8531题桥不覆盖61049训练保留及7606测试的全部候选，不等于61689/7891/7606全套可训练release。负责人需恢复／发布具体成员与SHA清单，不让师弟重猜。
3. **NQ-open能否复用？** 历史2742题来自validation[512:3610]再做冻结曝光排除，已被使用；其reader为single-nq，完整原候选及数值文件恢复不足。87,925/3,610原始问答不能直接按2742套用，旧审阅和结果也不能冒充当前multiset的新正式输入。仍需恢复原件或另立明确的新版本重建；不混成旧重放。
4. **0/1/2标签是否已有？** 可运行B1包具有已审训练／开发标签及规范；正式两套扩展人口均没有已完成并登记的合格标签包。0=none、1=partial、2=direct，针对实际共同可见文本。官方答案字符串、FAISS相关性和未审段落不能当该标签。新正式范围应先冻结身份与候选，再按隔离盲审／多数裁决发布训练监督；评估标签独立封存，预测联合seal后才读。不能把现有稀疏标签扩成全Top50完整真值。
5. **wiki_dpr索引能直接用吗？** 大小2.88GB或76GB不能证明版本／行序匹配。历史检索协议是`facebook/dpr-question_encoder-single-nq-base` revision `d04a52f6d2f96c60117a925e8c24c4043a75f265`，`psgs_w100.nq.compressed`、nprobe64、Top50；冻结wiki_dpr revision `0ae2454140a2d6864475c83f26e6dc9cd4ab9ce4`。这是检索器，不是实验的multiset reader。优先复用历史桥，不现场换检索器或用题的标签挑候选。现成协议见`configs/experiments/reader_joint_witness_retrieval_assets.json`及`reader_full_data_replication.json`；仅原始索引不能生成段落语义标签。
6. **私服是否开着？** 11-06包HEAD已实测200；此前“可能没开”不是已证实结论。如师弟仍连不上，应提供去掉凭据的状态码／连接错误，仅检查既有服务，不另开服务或关闭TLS。访问配置须私下安装。
7. **ModelScope模型可否复用？** 可按上述命令核所有所需文件的size/SHA。全部MATCH才证明与固定资产一致；文件名、master、下载成功均不足。不匹配需保留原目录并定位差异，不盲目重下整个模型。
8. **最终谁生成登记？** 项目负责人负责将合格release发布私服并登记`junior.json`，师弟负责运行冻结实验。当前尚无可直接用来启动两套正式训练的完整release配置。schema与成员格式见`docs/two-dataset-junior-run-instructions.md`；不能只补三个路径或改ready标志。

## 负责人尚需完成的交付

按版本准备两数据集完整原Top50输入及角色／组件清单、训练标签、独立评估标签；记录模型可见文本审查、问题泄漏与oracle/headroom证据；核归档／成员size/SHA、资源需求，并私下登记release。完成后原命令`bash scripts/run_two_datasets.sh`才会进入正式准备和训练。此次启动诊断修复不等于上述数据工程已经完成。
