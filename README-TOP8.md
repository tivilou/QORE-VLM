# 八方案 × DPR / BGE 实验套件

固定八方案：MIXED_DEPTH、FIDELITY、VALUE_GEOMETRY、LOCAL_REUPLOAD、NATIVE_BORN、FEATURE_GATE、LOCAL_FFN、ORDINAL。保留各自完整经典对照，统一比较每个模型自己的原生打分，不将 DPR 与 BGE 原始分数直接混排。

依赖准备并把私有输入和模型放在 GPU 服务器后，一条命令运行全部双模型三种子矩阵：

```bash
python scripts/run_top8.py --dpr-model models/dpr-reader-multiset-base --bge-model models/bge-reranker-v2-m3 --train data/qvalue/train.jsonl --eval-inputs data/qvalue/eval.inputs.jsonl --eval-labels data/qvalue/eval.labels.json --output outputs/top8-full
```

训练中断后在原命令末尾加 `--resume`。不自动下载、不改系统网络，不把缺标签当 none；全部预测联合封存后才读评估标签。输出含逐臂结果、未知界限、失败和双模型排序。数据、模型与原始预测属于私有实验材料，不提交公开 GitHub。

环境、八种机制、原生接口、固定优化设置、结果解释、恢复限制和已验证范围见 [完整说明](docs/top8-dual-scorer-experiments.md)。正式全量由师弟运行；已有 B1 是曝光开发人口，不是独立确认集。量子方法相对 native 的提升与相对匹配经典控制的独立贡献分别报告。
