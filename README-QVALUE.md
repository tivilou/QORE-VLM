# Q-VALUE 实验交付入口

完整运行说明、公式、对照与结果：[docs/qvalue-handoff.md](docs/qvalue-handoff.md)。

- 证据等级：普通方法相对冻结DPR的限定L2；量子独立贡献尚未成立。
- 原曝光B1：512训练/128评估，Q-VALUE useful为373/373/373，冻结DPR为368，HEAD为374/373/373。
- 已实际完成九臂小型端到端canary。正式全量由接收者运行。
- 原题文、标签和精确模型通过私有包提供，不包含在GitHub代码中。

准备Python环境并解包私有输入/模型后，从仓库根目录运行：

```bash
python -m pip install -r requirements-qvalue.txt
python scripts/run_qvalue.py --model models/dpr-reader-multiset-base --train data/qvalue/train.jsonl --eval-inputs data/qvalue/eval.inputs.jsonl --eval-labels data/qvalue/eval.labels.json --output outputs/qvalue-full
```

默认三种子、九个学习臂及冻结native，全部预测先封存再读取评估标签。不要添加`--canary`来运行正式实验。输出目录必须不存在；原始输出保持私有。
