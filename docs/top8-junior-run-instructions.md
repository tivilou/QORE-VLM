# 给师弟：八个方案的 DPR / BGE 统一实验

代码分支：`codex/top8-dual-scorer`。主入口 `scripts/run_top8.py`，默认两个原生模型、八个方案、每个方案全部经典控制、三个固定种子。代码和汇总通过 GitHub；输入、模型与逐段输出保留在实验服务器。

先准备一次环境及私有文件：

1. 从项目已授权的 3080 交付目录取得 `top8-private-inputs-two-models.tar`。包大小 2,753,105,920 字节，SHA256 为 `29a4d5808a330b45f5dcb307ffca59de498d020ec7d70409e1716eedf3566a32`，含 17 个文件：原 DPR、BGE 两模型、B1 的 512 训练 / 128 评估输入及分离标签。通过服务器间传输渠道取得，不上传 GitHub，不经项目持有人的本机。
2. 解包后两个模型分别位于 `models/dpr-reader-multiset-base/`、`models/bge-reranker-v2-m3/`。BGE revision、逐文件大小和 SHA 在 `configs/experiments/reader_joint_witness_b1_model_assets.json`；程序拒绝缺失或不同版本的文件，不自动下载。旧 DPR 单模型包仍可保留作原交付复现。
3. 运行服务器应有 CUDA、至少约 7 GiB 可用显存、16 GiB RAM；B1 双模型套件建议至少 40 GiB 可用磁盘，完整研究人口增大时相应扩容。大文件无需经过项目持有人的本机。

该私有包在 3080 的确切路径为 `/home/Q-DUET-VLM/top8-private-inputs-20261103-v001/top8-private-inputs-two-models.tar`。以下解包命令假设已通过既有私有传输渠道，将它放到克隆后的仓库根目录。JSONL 字段、50 段身份及分离标签格式见 [输入格式](qvalue-handoff.md)。

```bash
git clone --single-branch --branch codex/top8-dual-scorer https://github.com/tivilou/QORE-VLM.git
cd QORE-VLM
git rev-parse HEAD
echo '29a4d5808a330b45f5dcb307ffca59de498d020ec7d70409e1716eedf3566a32  top8-private-inputs-two-models.tar' | sha256sum -c -
tar -xf top8-private-inputs-two-models.tar
python -m pip install -r requirements-qvalue.txt
python scripts/run_top8.py --dpr-model models/dpr-reader-multiset-base --bge-model models/bge-reranker-v2-m3 --train data/qvalue/train.jsonl --eval-inputs data/qvalue/eval.inputs.jsonl --eval-labels data/qvalue/eval.labels.json --output outputs/top8-full
```

最后一行是一键实验指令。默认种子 20261102、20261103、20261104，四个完整 epoch；按提供的人口自动增加训练步数，不硬编码 256 步。保留输出中的运行来源、缓存清单、训练记录、最终参数、联合预测封存与 evaluation.json。失败臂必须报告，不从排名中悄悄删除。

训练中断时原命令追加 `--resume`，代码和输入保持相同。恢复范围与实际验证见 [完整说明](top8-dual-scorer-experiments.md)。不能为提高开发分数改变种子、臂、阈值或训练轮数。

`evaluation.json` 给出每个模型 native、每臂 useful/direct 的已审下界与未审上界，以及双模型等权 native 改进排序。未知区间重叠时不能下确定胜负结论。超过 native 只说明完整方法有用；是否有量子机制的独立贡献，要另看匹配经典控制。

已有私有包复现的是曝光 B1 探索。正式全量研究使用同一输入格式的完整训练与独立确认人口，在评分、参考读取前冻结组件和角色；不能把旧 B1 改名当独立确认。正式实验由师弟执行，owner 的 BGE 单种子和既有 DPR 筛选结果只作前期证据。

对事先冻结的正式全量输入，保留默认全部模型、方案、对照及三个种子，只替换三个输入文件和新输出目录：

```bash
python scripts/run_top8.py --dpr-model models/dpr-reader-multiset-base --bge-model models/bge-reranker-v2-m3 --train data/full/train.jsonl --eval-inputs data/full/confirmation.inputs.jsonl --eval-labels data/full/confirmation.labels.json --output outputs/top8-formal-full
```

这里的 `data/full/` 是师弟实际准备并冻结的全量人口，不在 B1 私有包中。程序按实际问题数运行四轮，不把 512 / 128 写死；存储与内存需求也随人口增长，不能将 B1 的空间估计直接当作全量估计。

## 允许仅 DPR 的方案

入选不再要求同时兼容 BGE。DPR 专用方案可以根据自己的 DPR 成绩入选；双模型适用性是运行范围字段，不是成绩门槛。最多仍选八个不同方案，同信息经典对照全部保留。跨模型融合另标为同时需要两个模型，不冒充 DPR 单模型增强。

入口支持 `--dpr-only-ideas` 和 `--bge-only-ideas`。这些参数控制实际调度，未指定的其余注册方案在提供的两个模型上运行。也可用 `--ideas` 显式指定共同运行的方案；共同列表与专用列表必须互不重叠，总数不得超过八。专用列表缺少对应模型时立即报错。

例如，下列命令**演示把 FIDELITY 只安排在 DPR**，其余七项仍跑双模型；这不是声称 FIDELITY 无法适配 BGE，也不改变它已有的 BGE 成绩：

```bash
python scripts/run_top8.py --dpr-model models/dpr-reader-multiset-base --bge-model models/bge-reranker-v2-m3 --dpr-only-ideas FIDELITY --train data/full/train.jsonl --eval-inputs data/full/confirmation.inputs.jsonl --eval-labels data/full/confirmation.labels.json --output outputs/top8-mixed-scope
```

追加 `--plan-only` 可先打印精确模型/方案安排及学习单元数；不加载模型、不启动训练、不创建输出目录。这个示例三个种子共333个学习单元，默认全双模型仍为348个。

结果分别列出 `dual`、`dpr_only`、`bge_only` 排名。DPR 专用方案仅相对 DPR native 统计，BGE 记 `NOT_SCHEDULED`，不计零分、不除以二；原定双模型却失败或缺少种子的方案仍是不完整结果，不能自动降格成 DPR 专用来隐藏失败。

当前八项都有 BGE 实现，现有默认名单保持。DPR 专用候选重新参加选拔，不因为缺 BGE 被排除；未实现移植的旧候选不得仅添加名称就当作可运行，须先接入其确切机制和对照。当前复核的 DPR 跨度方案尚未按原排序超过入选线，见 [范围修订记录](../configs/experiments/idea_search_20261103/top8-scope-policy.v002.json)。
