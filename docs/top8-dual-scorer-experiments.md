# 前八名 idea：DPR / BGE 一键实验

这批方案按已归档 DPR 探索结果筛选，不把双模型融合计作单模型改造。首要排序是 useful 下界的种子均值，同分比较 direct 下界均值，再按候选名称排序。开发集已曝光；历史优势和新 BGE 结果均不代表独立确认。

2026-11-03范围修订：**同时兼容BGE不再是入选要求**。DPR专用方案可以按自己的成绩入选，在同一脚本中只调度DPR；当前八项都有双模型实现，默认名单暂不变。`--dpr-only-ideas`、单模型分组排名及示例见[运行说明](top8-junior-run-instructions.md#允许仅-dpr-的方案)。历史筛选、已有模型成绩和失败记录保持，缺少BGE不计零分，也不把双模型融合改称DPR专用。

| 初选排名 | 方案 | DPR useful 历史值 | 主要插入点 |
|---|---|---|---|
| 1 | MIXED_DEPTH | 375..377 / 374 / 373 | 三个原生深度的混合密度与 Uhlmann fidelity |
| 2 | FIDELITY | 374 / 373 | 最后层 CLS 对原生 token key 的量子 fidelity 偏置 |
| 3 | VALUE_GEOMETRY | 373 / 373 / 373 | 最后层每个原生 64 维 value/context 子空间 |
| 4 | LOCAL_REUPLOAD | 372 / 372 / 373 | 原生 FFN 的四坐标局部重上传线路 |
| 5 | NATIVE_BORN | 372 / 372 / 373 | 将原生 log-odds 编入量子读出 |
| 6 | FEATURE_GATE | 372 / 372 / 373 | 原生最终线性头输入的乘性特征门 |
| 7 | LOCAL_FFN | 372 / 372 / 373 | 原生 FFN 四坐标的 Born 非线性 |
| 8 | ORDINAL | 372 / 372 / 373 | useful/direct 有序投影测量 |

原生比较锚点为 DPR 368 useful、BGE 361 useful。新矩阵会重新生成各自原生预测；历史值只用于初选。控制是否超过量子方案与是否超过原生模型分开分析。四坐标线路、量子态 fidelity 和正交 value 线路均为经典精确模拟，不主张量子计算加速。

## 环境与输入

使用 CUDA、FP32、PyTorch **2.5.1**、Transformers **4.44.2**、safetensors、tokenizers；依赖版本见 `requirements-qvalue.txt`。本轮服务器实际使用 Python 3.10.10、Torch 2.5.1+cu121、NumPy 1.26.4、safetensors 0.7.0、tokenizers 0.19.1。按运行服务器的 CUDA 环境安装兼容的 Torch 轮子。原始模型版本及逐文件 SHA 在 `configs/experiments/reader_joint_witness_b1_model_assets.json`，程序只加载既有文件，不下载模型。

输入沿用 [输入格式](qvalue-handoff.md) 的 JSONL 定义：训练行包含问题、原文 50 段、组件身份及**已审训练项**的稀疏标签；评估输入没有标签。评估标签单独保存。缺标签不能作为 none。正式研究的人口、角色、模型训练曝光及独立确认须另行冻结；不能把 owner 的 512/128 探索样本换个名称当独立确认。

BGE 保留原生 `dense → tanh → out_proj`，通用读出方案位于 out_proj 的原生输入处；上游改造仍穿过冻结 dense/tanh。DPR 对应桥接为恒等。两模型共用问题和原文、无标题、原顺序每 8 段动态右 padding；DPR 原生上限 350，BGE 512。实际截断计数和每个原生块的 readout/tail 重放误差写入缓存。模型原生长度差异不伪装为相同词元序列。

固定 Transformers 4.44.2 中，DPR 的 BERT 层支持原生 SDPA，BGE 的 XLM-R 层使用原生 eager 注意力；重建缓存层时分别保留这两种实现。FIDELITY 的偏置在各自原生层中注入，BGE 不宣称执行了 SDPA 内核。owner 的完整 BGE 缓存已通过 32,000 候选的原生读出/尾层重放，零截断；这项检查本身不是训练收益或双模型量子归因证据。

## 一次运行完整矩阵

大文件和输出放在实验服务器，建议至少 40 GiB **可用**空间；缓存量随完整研究人口增长。下面命令运行两个模型、八个方案、各自全部控制和三个预先固定种子；正式全量由师弟运行。

```bash
python scripts/run_top8.py \
  --dpr-model models/dpr-reader-multiset-base \
  --bge-model models/bge-reranker-v2-m3 \
  --train data/top8/train.jsonl \
  --eval-inputs data/top8/eval.inputs.jsonl \
  --eval-labels data/top8/eval.labels.json \
  --output outputs/top8-full
```

默认固定种子为 20261102、20261103、20261104；各臂四轮，完整问题批次 8，AdamW lr=0.001、weight_decay=0、clip_norm=1。无有序对的问题仍占分母。禁止根据开发结果调参或追加种子。owner 新 BGE 探索只使用其中种子 20261103，不可据此称 BGE 已通过多种子 L2。

每个模型只提取一次原生缓存。大部分方案复用 CLS/context/FFN 数值；FIDELITY 复用完整最后层序列前缀。五折混合态类别库只用训练标签，训练样本排除自身组件所在折，评估用全部训练库；全局 OOF 标准化的弱依赖沿用历史方案并明确披露。

已有完整缓存可直接用 `--cache outputs/old/cache-dpr/cache.json --cache outputs/old/cache-bge/cache.json` 代替模型和原文输入。首次运行的输出目录必须不存在。旧失败和中间输出保留。

## 结果与恢复限制

`matrix/joint-prediction-seal.json` 在读评估标签之前联合封存所有已完成预测和检查点，列出失败臂。最终每个 50 段预测逐项重建检查点并按位重放。`matrix/evaluation.json` 包含 useful/direct 已审下界与未审上界、原生结果、每个控制及综合排名。只有两模型全部请求种子齐全的方案才进入双模型排序；排名用两模型各自 native 改进的等权均值，不能按原始总分混排两个模型。未审项导致重叠区间时不判确定胜负。

一键入口使用 v2 恢复层。每 32 步和每轮结束保存完整训练参数、AdamW 状态、CPU/CUDA 随机状态、问题排列、排列生成器和训练记录；二进制文件落盘并读回 SHA 后才发布步骤收据。中断后在**同一代码、输入、配置、输出目录**下重跑原命令并追加 `--resume`。已完成单元验证文件 SHA 后复用，明确失败的单元保留失败，不自动重试；尚未完成的单元恢复最近已封存步骤，旧部分输出保留在旧 generation 目录。不能借恢复更换参数、种子、模型或数据。

恢复只支持已完整生成的原生缓存及训练阶段；若原生缓存提取中断且 `cache.json` 尚不存在，该根保留，新建输出根重新提取。混合深度类别库会从同一训练缓存确定性重算，代价计入本次执行。评估结果已写出后不继续训练。恢复层的实际 GPU 等价检查结果以对应收据为准，静态检查不认证数值等价。

`--canary` 只验证固定前两训练/评估题的一步通路，五折混合态拟合不会伪造通过。完整 sample-trace.v2 和量子归因审查仍独立标记，不以脚本可运行冒充完成。

本轮 owner 已完成 BGE 的全部58个学习单元，各256步、512训练/128曝光开发，全部最终Top50预测检查点重放误差为0；176份联合封存文件逐大小/SHA核验通过。八方案BGE useful下界均超过原生361。具体区间、全部经典控制与历史DPR综合探索排序见 [实测结果](top8-dual-scorer-results.md)。

owner队列在58臂及评估完成后，因旧扁平摘要写入器遇到正常子目录而exit1；该失败保持，不伪造exit0。独立纯标准库元数据收尾已在新目录完成，未重训、未重新评估、未修改旧根。这里的一键入口直接调用 `top8_suite_v002`，不经过owner专用的扁平摘要写入器。上述结果不替代双模型同代码同种子的正式全量矩阵、完整sample-trace或独立确认。
