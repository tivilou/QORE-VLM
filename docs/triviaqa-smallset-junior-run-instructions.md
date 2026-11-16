# TriviaQA 8000／1000／1000 小集实验

当前目标是固定的小规模 TriviaQA：8000 训练题、1000 验证题、1000 测试题，
每题保留原 Top50，共 500000 个段落位置。DeepSeek／Gemini 两票一致项保留；
66331 分歧已由第三票裁决。标签仅为 `0=无用／1=有用`，不是人工 gold，
没有 `partial/direct` 真值。大规模 NQ-open／TriviaQA 的旧 formal 入口仍保持未发布。

## 当前发布状态

输入及标签已完成核验，**当前小集数据归档已发布到已有私有共享服务**。
2026-11-15 数据服务器恢复后，190976000 字节归档经服务器直传，上传回执及
另一台服务器的完整 HTTP 下载、大小、SHA 和 EOF 核对通过。
归档 SHA256 为 `1a53cf1da9099b0b432b0e6b7b30625cde674cf6de3e7c19414b0da7ffc827af`。
发布清单为 `configs/experiments/triviaqa_smallset_20261112/junior-release.v001.json`，
不会下载旧 B1 冒充新小集，也不改变大规模双数据集 formal 入口的状态。

## 单命令

同步项目的 `codex/top8-dual-scorer` 分支，并复用之前私有配置
`~/.config/qduet/junior.json`，然后执行：

```bash
bash scripts/run_triviaqa_smallset.sh
```

入口依次获取并校验数据、复用精确 DPR/BGE 模型包、转换三角色输入、准备运行环境、
在师弟环境执行小型二分类数值自检、运行固定矩阵、封存预测、分别评估 valid/test、
上传声明结果并核对大小与 SHA 回执。私有配置的 URL／token 不放 Git，也不放命令行。
负责人本轮没有运行正式模型实验。

源码入口会先检查本地启动依赖，不加载模型、不读取数据。如果遇到
`INCOMPLETE_SOURCE_DELIVERY` 或旧版的 `No module named
'applications.rag.model_input_guards'`，先完整更新交付分支：

```bash
git pull --ff-only origin codex/top8-dual-scorer
bash scripts/run_triviaqa_smallset.sh
```

该旧版错误发生在创建实验输出前，更新后使用原命令即可。已下载并通过 SHA 核验的
数据、模型以及已准备环境会复用；无需删除目录或重新下载。可单独运行
`python3 -B scripts/check_top8_delivery.py` 检查源码依赖；通过不代表 GPU 实验通过。

只准备数据和模型、不运行实验：

```bash
bash scripts/run_triviaqa_smallset.sh --prepare-only
```

只查看固定计划、无需凭据或数据：

```bash
bash scripts/run_triviaqa_smallset.sh --plan-only
```

同一代码／输入／配置的训练中断后续作：

```bash
bash scripts/run_triviaqa_smallset.sh --resume
```

已经明确失败的 cell 不会自动重试。未封存的提取过程不是可恢复训练检查点；
先保留失败目录、解决具体问题，再按新运行目录处理，不覆盖旧结果。

## 固定比较与二分类适配

保留八项 DPR/BGE 单模型方法（MIXED_DEPTH、FIDELITY、VALUE_GEOMETRY、
LOCAL_REUPLOAD、NATIVE_BORN、FEATURE_GATE、LOCAL_FFN、ORDINAL），以及
CROSS_SCORE、CLASS_FUSION 两项双模型融合及所有经典控制。
三个固定初始化为 20261102／20261103／20261104，四轮训练，393 个学习单元。
这是新小集的固定验证计划，不把旧 B1 分数当成新数据的结果。

MIXED_DEPTH 只建立实际存在的两类训练统计；CLASS_FUSION 使用两个等维测量子空间、
两类训练先验与二分类似然。ORDINAL 保留内部顺序测量结构，其二分类似然合并两个
“有用”隐状态；内部细分没有 direct 标签含义，不报告 direct 指标。
这些是新二分类版本，尚无本轮真实 GPU 效果或吞吐结果。

所有方法、控制和 seed 的完整 valid/test 预测共同封存后才读取评估标签。
valid 用于方法分析；test 只报告固定比较，禁止据其结果挑方法、调参或重新训练。
两角色分别报告 useful/5000，不混成一个 10000 分母排名。

按完整运营标签在原 Top50 中理想选取，train 的 Top5 useful 上限为
24320／40000，valid 为 3049／5000；它们是标签约定下的可达上限，不是模型成绩。
test 不计算用于选方案的上限。

## 服务器空间

数据和模型只下载到师弟的实验服务器，不经过负责人本机。数据下载只需数百 MiB，
现有双模型包约 2.75 GB。完整 FP32 最后层序列缓存可能需要数百 GiB；
按两模型最大长度计算的保守上界约 1.5 TiB，**不是实测需求**。
入口只要求启动时至少留 40 GiB，并在提取／训练过程中持续保护这 40 GiB 余量；
40 GiB 不是整轮容量承诺。请在实际实验服务器安排充足磁盘与内存，空间不足时扩容，
不删历史、不降低精度、不删控制臂。完整数据的真实吞吐与峰值内存仍待师弟运行记录。
