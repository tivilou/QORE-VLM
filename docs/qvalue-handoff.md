# Q-VALUE：L2 方法复现与正式全量实验入口

这是待师弟验证的量子参数化打分方案。现有证据是 **L2：完整方法相对未优化 DPR 的多种子收益**。目前没有证据证明收益来自量子模块，也没有量子硬件或计算加速结论。正式全量实验由师弟运行。

## 现有结果与结论范围

数据为 TriviaQA B1：512 个训练问题、128 个开发问题；每题固定原始 Top50，从中取 Top5。共同原文、不加标题。只使用7339项已审训练监督。开发集多次曝光，标签是三人多数 Silver，不能当作独立确认集或官方 gold。已确定该候选池的 useful 理想上限为435/640。

| 方法 | 顺序种子20261102 | 20261103 | 20261104 | 相对冻结DPR的平均变化 |
|---|---:|---:|---:|---:|
| 冻结 DPR multiset | 368 | 368 | 368 | 0 |
| Q-VALUE完整方法 | 373 | 373 | 373 | +5 useful/640 |
| 仅训练普通HEAD | 374 | 373 | 373 | +5.33 useful/640 |

Q-VALUE三次 direct 为169/170/170，含direct的问题数均为66；原DPR为164、64。三次训练顺序预先固定、全部纳入，量子角均从恒等变换初始化；这些种子不是三个不同的量子初始化。完整方法较原DPR平均提高0.78125个百分点；它没有超过普通HEAD，原更严格的量子归因筛选结论仍为NO_GO。CONCAT、MLP及其他控制的结果也须保留，不能只展示上表前三行作为量子贡献证明。

证据卡：`configs/experiments/idea_search_20261103/qvalue-native-comparison-l2-card.v001.json`。原始完整对照与限制：`qvalue-geometry-terminal-result.v002.json`。旧单题有限差分0/0证书保持NOT_PASSED；新的双题核对单独记录在`qvalue-l2-closeout-terminal-result.v001.json`，没有重训或改写旧记录。

## 方法

冻结`facebook/dpr-reader-multiset-base`原始权重，revision为`ab1974d6023bc4dd27cd713c9ac2b6247509244d`。它有TriviaQA预训练曝光。保持原始Q/K注意力、token、FP32数值政策和相关性头的初始值。

最后一层的12个注意力头各产生64维CLS上下文$C_h$。每头使用6比特、两层RY与受控RY的实数线路，24个角度，共288角；门的端序及方向由受审源码固定。它可精确经典模拟。

$$C'_h=U_h(\theta_h)C_h,\qquad
z'=z_{\rm native}+T(C',r)-T(C,r),\qquad
s=w^\top z'+b.$$

$T$是原模型冻结的注意力输出投影、残差、LayerNorm和FFN；$r$是最后层的原输入CLS。这样$\theta=0$时恢复原相关性函数。训练量子角和768维相关性头权重，头偏置冻结。没有使用答案字符串生成标签。

所有方法共享完整原生表示、同一监督、同一问题顺序与预算。必须同时运行HEAD、CONCAT、MLP、DENSE、LORA、TANGENT、CAYLEY、LOCAL、QUANTUM九臂，另报告冻结native。LOCAL是局部旋转控制，不称为严格不纠缠控制；名义角数不等于可辨识容量。

## 已完成的运行验证

2026-11-03，独立队列 canary v004 实际 complete / exit0 / quiescent：前2训练题与前2评估题、九臂各一步、十组预测联合封存、检查点重放、稀疏标签评估及完整512/128私有输入导出均完成。新提取原生表示与原记录最大误差0；34份科学文件逐文件大小/SHA/EOF核验。它验证运行链路，不新增方法收益或替代正式全量。原三个技术失败及其成本仍保留。证据：`qvalue-portable-canary-terminal-result.v004.json`。

## 环境与一键运行

先从仓库根目录解包项目单独提供的私有文件 `qvalue-private-inputs-model.tar`，其中已有原始512/128输入、独立评估标签和上述精确revision的五个模型文件；不要上传此包到公开仓库。SHA256为 `27cca24fee6508cb39742f2badce0ba74e3149eabef93c8df85bc3854bfa348d`，大小459857920字节。

```bash
sha256sum qvalue-private-inputs-model.tar
tar -xf qvalue-private-inputs-model.tar
```

解包后的`PRIVATE-BUNDLE-MANIFEST.json`列出全部成员SHA。完整512/128包用于复现已曝光B1；新的正式全量人口须由运行者固定并提供同格式文件。

从仓库根目录使用自己的Python环境。建议Python3.11、CUDA版PyTorch2.5.1、Transformers4.44.2；依赖见`requirements-qvalue.txt`。安装依赖与模型下载由运行者在自己的机器上准备，脚本本身只读本地文件，不自动联网或修改环境。

```bash
python -m pip install -r requirements-qvalue.txt
python scripts/run_qvalue.py --model models/dpr-reader-multiset-base --train data/qvalue/train.jsonl --eval-inputs data/qvalue/eval.inputs.jsonl --eval-labels data/qvalue/eval.labels.json --output outputs/qvalue-full
```

模型必须是上述精确revision的原始文件。脚本逐文件验证大小和SHA，并核对实际加载权重。也可设置`QVALUE_MODEL_DIR`，或使用标准HF缓存中的该revision；HF缓存的符号链接会复制到本次输出的私有模型视图，不改共享缓存。`QVALUE_DATA_DIR`可指定数据目录。没有绝对服务器路径、SSH配置、调度器依赖或凭据。

已有受验证的私有原生表示缓存时，可跳过重复的冻结模型提取：

```bash
python scripts/run_qvalue.py --cache data/qvalue-cache/cache.json --eval-labels data/qvalue/eval.labels.json --output outputs/qvalue-full
```

缓存是含训练标签、原生表示和模型片段的私有文件，禁止上传GitHub。缓存模式信任供应者的来源清单；首次新数据正式实验应从原始输入执行提取，保留提取清单供核验。

默认固定三种子`[20261102,20261103,20261104]`，每种子4个完整训练epoch，每批8个完整问题；AdamW学习率0.001、betas0.9/0.999、eps1e-8、weight_decay0，梯度裁剪1。每题按所有有序标签对的logistic损失取平均，再按批中问题数取平均。无可比较标签对的问题保留在分母。512题时每臂256步；新全量人口会按完整4epoch自然增加步数，不能仍冒称是旧256步实验。

仅验证安装和接口时使用`--canary`。它只取前2训练题、前2开发题、1个种子、1epoch，输出明确标记CANARY，不能作为方法效果或全量实验结果。完整运行不带该开关。

## 数据文件

训练文件为UTF-8 JSONL，每行字段必须恰好为`id,component_id,question,paragraphs,labels`。`paragraphs`包含固定原始顺序的50项，每项仅有`id,text`。`labels`把已审段落ID映射为0=none、1=partial、2=direct；未审项不出现，不能填0。组件ID和问题身份不得跨训练与评估角色。

评估条件文件字段为`id,component_id,question,paragraphs`，**不能包含labels或答案**。评估标签单独存成JSON，格式为`{"问题ID":{"段落ID":1}}`。脚本在全部种子、全部对照的预测与模型联合封存后才打开该文件。缺标签的选位报告上下界和unknown，不报虚假的完整precision。

原分词为慢版DPRReaderTokenizer，原始8候选动态右padding、最大350 token、无标题；先处理全部原始候选块，再过滤已审训练行。长输入的截断数记录在缓存清单，不能据此声称新人口的输入完整性已通过；新正式人口需独立审查任务匹配、参考和泄漏。

输出含缓存、检查点、逐候选预测、联合seal与汇总。只把代码、汇总、清单和SHA放GitHub；原题文、标签、表示、权重及逐项结果走项目的私有交付渠道。脚本拒绝覆盖已有输出。

## 师弟应验证的主张

先固定全量人口、角色、参考和独立组件，再运行上述所有对照与三种子。主要指标为useful Top5计数，另报告direct及含direct题数；完整保留负差、失败及未知覆盖。先检验完整方法相对冻结DPR的L2收益能否复现，再检验相对HEAD和匹配经典控制是否仍有增量。后者才关系到L3量子机制归因，不能把前者替代成后者。BGE上的通用性尚未得到本方案验证，应作为后续独立实验。新颖性审查也尚未支持首创声明；尤其需要核对现有正交微调与量子参数化适配工作。
