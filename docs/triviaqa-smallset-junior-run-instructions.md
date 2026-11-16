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

新版提取会逐题提交完成记录，`--resume` 只跳过身份和文件哈希均通过的题。
提取恢复与训练检查点是两件事；已经明确失败的训练 cell 不会自动重试，
旧训练结果仍保留，不混入变更了源码身份的新实验。

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

数据和模型只下载到师弟的实验服务器，不经过负责人本机。数据包约 191 MB，
现有双模型包约 2.75 GB。旧版全序列缓存的 1.5 TiB 估算已由新存储路径替代，
不再据此要求扩容到 2 TB；`junior-release.v001.json` 中旧字段保留用于识别原发布包，
当前运行入口不再使用该字段估算缓存。

新版本保持 FP32，每题提交向量／原始词元及大小、SHA 和完成记录。
所有题的向量静态上限约 43.4 GiB，词元与元数据估计约 9 GiB。
仅 FIDELITY 需要的完整序列使用**每个模型最多 32 GiB**的临时 LRU 缓存，
两个模型合计最多 64 GiB；缺失时由原冻结模型、相同完整词元块重新计算，
完整 hidden、mask、native scores 的哈希必须与首次提取一致才接受。
这控制了占盘，但增加计算时间；真实整轮耗时仍须实际记录。
CPU 端题向量 LRU 限额为 128 MiB，训练统计、模型、当前批次等仍额外占内存。

新持久缓存加上述工作集估计约 120 GiB；训练检查点、输出、既有文件另计。
入口持续保留 40 GiB 余量，不能据此承诺任意旧目录都能在 500 GiB 盘上装下。
临时完整序列缓存可以用 `--sequence-cache-gib 8` 降为每模型 8 GiB，代价是更多重算；
输入、精度、控制臂、种子及学习配置不变。

**已有提取未完成时**，先停止旧进程，更新代码后执行：

```bash
git pull --ff-only origin codex/top8-dual-scorer
bash scripts/run_triviaqa_smallset.sh --resume
```

不要与旧进程同时运行，不删除旧缓存或已下载资产。新版本复用已提交的新逐题记录；
旧版只有 prefix 文件、没有 features／索引的题不能冒充已完成题，需补提取一次紧凑向量。
这些旧 prefix 原样保留，在 FIDELITY 读取时只有与新提取完整哈希匹配才直接复用。
新格式中断后从最近完整题继续，未提交的文件保留并另起文件代次。
已开始旧版本方法训练的矩阵有源码绑定，新版本不会混用其训练检查点；
这种情况需另开运行目录，保留旧结果。截图中的旧提取阶段尚未进入矩阵训练。
