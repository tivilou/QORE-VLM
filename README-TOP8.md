# Q-DUET 八项单模型增强＋两项双模型融合

NQ-open＋TriviaQA 的统一入口现为：

```bash
curl -fsSL https://raw.githubusercontent.com/tivilou/QORE-VLM/codex/top8-dual-scorer/scripts/bootstrap_two_datasets.sh | bash
```

两数据集分别训练和评估；八方案 × DPR/BGE × 三种子，再加Q-CROSS-SCORE和Q-CLASS-FUSION两项双模型融合及其全部经典控制，共786个学习单元。融合使用两路原生分数，单列逐种子结果，不混作单DPR方法。**正式输入／标签发布尚未齐全，默认会在下载和训练前报出缺口。** [十项运行说明与数据状态](docs/two-dataset-junior-run-instructions.md)。

以下保留原TriviaQA单数据集入口。

统一入口自动完成私有资产获取、环境准备、全部实验、预测封存、评估及结果回传：

```bash
bash scripts/run_all_top8.sh
```

首次克隆也可一条命令：

```bash
curl -fsSL https://raw.githubusercontent.com/tivilou/QORE-VLM/codex/top8-dual-scorer/scripts/bootstrap_top8.sh | bash
```

**当前正式独立 TriviaQA 全量人口未就绪，默认 formal 会在下载/安装/GPU 前明确报缺口。** 现有私有包仅为曝光 B1 512/128；完整复现它需明确追加 `--population reproduction`。旧 NQ-open 全量不是此 TriviaQA 数据。

[一次配置、运行和自动回传](docs/top8-junior-run-instructions.md) · [模型及实验机制](docs/top8-dual-scorer-experiments.md) · [现有结果](docs/top8-dual-scorer-results.md)

八项双模型保持；新增 DPR 专用项按历史平均 useful 严格大于373.5筛选，已审终态记录没有额外合格项，跨模型融合不混作 DPR-only。记录在 `configs/experiments/idea_search_20261103/dpr-additions-audit.v001.json`。

代码和审计公开；数据、权重、逐题输出及访问凭据通过项目私有交换服务传送。正常失败和未审区间如实保留，量子方法超过 native 不自动证明超过匹配经典控制。
