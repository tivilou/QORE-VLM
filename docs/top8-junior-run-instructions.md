# 给师弟：八个方案的 DPR / BGE 统一实验

代码分支：`codex/top8-dual-scorer`。主入口 `scripts/run_top8.py`，默认两个原生模型、八个方案、每个方案全部经典控制、三个固定种子。代码和汇总通过 GitHub；输入、模型与逐段输出保留在实验服务器。

先准备一次环境及私有文件：

1. 沿用已有 `qvalue-private-inputs-model.tar`，SHA256 为 `27cca24fee6508cb39742f2badce0ba74e3149eabef93c8df85bc3854bfa348d`。它含原 DPR 模型和 B1 的 512 训练 / 128 评估输入及分离标签，位于项目既有 3080 交付目录。通过服务器间传输渠道取得，不上传 GitHub。
2. 取得现有 `BAAI/bge-reranker-v2-m3` 六个模型/分词文件，放在 `models/bge-reranker-v2-m3/`。其 revision、逐文件大小和 SHA 在 `configs/experiments/reader_joint_witness_b1_model_assets.json`；程序拒绝缺失或不同版本的文件，不自动下载。现有 P4 上已验证这六个文件，模型权重为 2,271,071,852 字节。
3. 运行服务器应有 CUDA、至少约 7 GiB 可用显存、16 GiB RAM；B1 双模型套件建议至少 40 GiB 可用磁盘，完整研究人口增大时相应扩容。大文件无需经过项目持有人的本机。

```bash
git clone --single-branch --branch codex/top8-dual-scorer https://github.com/tivilou/QORE-VLM.git
cd QORE-VLM
git rev-parse HEAD
echo '27cca24fee6508cb39742f2badce0ba74e3149eabef93c8df85bc3854bfa348d  qvalue-private-inputs-model.tar' | sha256sum -c -
tar -xf qvalue-private-inputs-model.tar
python -m pip install -r requirements-qvalue.txt
python scripts/run_top8.py --dpr-model models/dpr-reader-multiset-base --bge-model models/bge-reranker-v2-m3 --train data/qvalue/train.jsonl --eval-inputs data/qvalue/eval.inputs.jsonl --eval-labels data/qvalue/eval.labels.json --output outputs/top8-full
```

最后一行是一键实验指令。默认种子 20261102、20261103、20261104，四个完整 epoch；按提供的人口自动增加训练步数，不硬编码 256 步。保留输出中的运行来源、缓存清单、训练记录、最终参数、联合预测封存与 evaluation.json。失败臂必须报告，不从排名中悄悄删除。

训练中断时原命令追加 `--resume`，代码和输入保持相同。恢复范围与实际验证见 [完整说明](top8-dual-scorer-experiments.md)。不能为提高开发分数改变种子、臂、阈值或训练轮数。

`evaluation.json` 给出每个模型 native、每臂 useful/direct 的已审下界与未审上界，以及双模型等权 native 改进排序。未知区间重叠时不能下确定胜负结论。超过 native 只说明完整方法有用；是否有量子机制的独立贡献，要另看匹配经典控制。

已有私有包复现的是曝光 B1 探索。正式全量研究使用同一输入格式的完整训练与独立确认人口，在评分、参考读取前冻结组件和角色；不能把旧 B1 改名当独立确认。正式实验由师弟执行，owner 的 BGE 单种子和既有 DPR 筛选结果只作前期证据。
