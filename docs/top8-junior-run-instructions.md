# 师弟一键实验入口

新增 [NQ-open＋TriviaQA统一入口](two-dataset-junior-run-instructions.md)，现包含八项单模型增强＋CROSS_SCORE／CLASS_FUSION两项融合，两数据集独立训练／评估，共786学习单元。下面的原入口仍为单独TriviaQA八项范围；要运行十项请使用`bash scripts/run_two_datasets.sh`。不能以此声称NQ正式数据已经恢复。

一条命令负责获取私有数据与两个固定模型、准备环境、运行八方案及全部经典对照/三个固定种子、联合封存预测、评估和回传结果。正式全量由师弟运行，owner 不代跑。

已有代码目录中，入口是：

```bash
bash scripts/run_all_top8.sh
```

首次从 GitHub 启动可用下列命令，代码自动克隆到服务器的 `qduet-junior/code`，资产和实验放到 `qduet-junior/work`：

```bash
curl -fsSL https://raw.githubusercontent.com/tivilou/QORE-VLM/codex/top8-dual-scorer/scripts/bootstrap_top8.sh | bash
```

默认请求 **formal 正式人口**。当前历史审计没有找到已经冻结、独立且带所需段落语义标签的 TriviaQA 全量发布，因此默认命令会明确报出数据缺口，停止于安装、GB 下载与 GPU 使用之前；这不是一个已经可启动的正式全量数据包。历史 2742 题全量属于 NQ-open，不能替换这次 TriviaQA。

现已就绪的是原曝光 B1：512 训练、128 开发。若明确复现原实验，仍只需一条命令：

```bash
curl -fsSL https://raw.githubusercontent.com/tivilou/QORE-VLM/codex/top8-dual-scorer/scripts/bootstrap_top8.sh | bash -s -- --population reproduction
```

此命令会完整跑所提供 B1 人口的四轮、八项双模型及所有对照、三个固定种子，共 348 个学习单元；不能称独立正式确认。它自动取得现有 2,753,105,920 字节/17 文件包，校验整包与逐成员 SHA，拒绝路径逃逸、重复成员及链接。原生 DPR/BGE 分词、最大长度、Top50→Top5、同输入经典对照、失败报告与未审上下界保持。

## 一次性私有访问配置

运行服务器需有 Python3.11、Git、CUDA GPU，建议至少 7 GiB 可用显存/16 GiB RAM/46 GiB 可用项目盘。大文件仅在服务器之间传输。脚本自行创建 Python 环境并安装固定依赖；已有完全匹配环境可在私有配置设置 `python` 路径。

访问令牌不在 GitHub、不放进命令行。使用项目既有私有交付配置 `~/.config/qduet/junior.json`，或复用交付服务器 `/etc/qore-upload/client.json`。配置只需 `url`、`token`；通过既有安全渠道安装一次，以后入口自动使用。不能声称任意陌生服务器无需私有访问凭据就能下载这些数据。直接交互执行入口也支持首次隐藏输入并保存令牌。

正式人口一旦由 owner 完成冻结与审查，将其版本化 release 放入同一私有配置的 `populations.formal`，采用 `junior-campaign.v001.json` 中 release 格式，含：整包/所有成员大小 SHA、固定三个输入文件、两个模型目录、精确人口计数、独立性及标签资格证据。师弟不必再输入数据路径。不能仅改 `ready` 标志把 B1 冒充正式发布。

## 恢复与回传

同一条命令重跑会复用已校验资产和同配置训练恢复记录。已结束的成功或失败单元不会自动重试；结束后的重跑仅补传结果。Linux 子进程继承工作目录锁，防止启动器意外退出后重复训练。

逐项预测、训练检查点、联合封存、评估、失败记录及大小/SHA 清单自动推送到私有项目命名空间 `five_ideas/top8-junior-results/<run-id>/`；所有上传都核对服务器回执。成功回传产生 `upload-complete.json`。回传失败保留 `pending-upload.json`，入口返回非零，不能把本地训练成功冒充交付成功。GitHub只发布代码和不含逐题内容的审计，不公开模型、数据或标签。

## 方案名单

八项双模型：MIXED_DEPTH、FIDELITY、VALUE_GEOMETRY、LOCAL_REUPLOAD、NATIVE_BORN、FEATURE_GATE、LOCAL_FFN、ORDINAL。

按用户最新要求，新增仅 DPR 方法须严格超过原八项中第二名 FIDELITY 的历史 DPR 平均 useful 下界 **373.5/640**；必须保留全部原定种子。已恢复的终态记录中没有额外单 DPR 方法达到此线，当前不虚构新增项。Q-CROSS-SCORE、Q-CLASS-FUSION 成绩更高，但依赖两个打分器，不能算仅 DPR；未审上界也不是已测高分。详细记录见 `dpr-additions-audit.v001.json`。这次规则是“八项共同方案加合格 DPR 专用项”，前次“总数最多八项”限制被替代；当前合格新增为空，实际仍是八项。

数据历史依据见 `formal-dataset-history-audit.v001.json`。这些流程的标准库 fixture/传输测试不是新模型实验或正式规模运行认证。
