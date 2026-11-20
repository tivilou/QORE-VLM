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
2026-11-18 的按字段读取版本额外生成无损 FP32 列文件，双模型全部方法累计最多
约 40.1 GiB，缓存与工作集总估计相应约 160 GiB；旧文件、输出、文件系统开销仍另计。
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

## 2026-11-18：只优化特征读取和训练统计

本次不改模型机制、FP32、训练顺序、每批题数、4轮训练、3个种子或393个学习单元，
不删除已有缓存，不操作正在运行的服务器进程。只改两处：

1. 每个方法只读取自己用到的字段，再送到GPU。例如只用 `readout` 的方法不再
   反复加载和传输 `depths/z/context` 等整包字段。首次使用时从原题文件生成按字段的
   FP32 列文件；按原始文件SHA绑定，使用CPU文件映射读取，不把整列常驻内存。
   原题分片保留不动；映射前验证列大小/SHA，读取行返回私有副本。
2. 同一模型缓存的标准化统计只计算一次并持久化。严格保留原训练题顺序、
   `torch.cat`、CPU `mean/std(unbiased=False)` 及原阈值，没有改用流式近似，
   不读取验证/测试标签，不混入验证/测试特征。重启先核训练ID/源SHA再复用。

这是上段“变更源码须新开目录”的**狭窄例外**：本次提供精确旧/新运行源码SHA对，
除此之外的缓存SHA/模型/种子/epoch/配置必须全部相同，才允许原目录续作。
旧矩阵绑定、已完成结果及优化器/RNG检查点均原样保留；新增
`matrix/feature-io-continuation.v1.json` 单独记录实际新运行源码及兼容依据。
未知版本或科学配置变化仍拒绝续作，不提供通用的跳过源码校验开关。

**师弟更新时先结束旧进程，确认其退出，再更新；不要在旧Python还运行时拉取代码。**
训练每32步及每轮末保存检查点，更新后从最后完整检查点恢复；刚保存点之后的步数需重做，
已有完整 cell 先核文件SHA后跳过，不从头训练。复用原 workspace 和命令参数：

```bash
git pull --ff-only origin codex/top8-dual-scorer
bash scripts/run_triviaqa_smallset.sh --resume
```

首次建立列文件会有一次读取/写入开销；按需新增的列合计最多约40.1GiB。
持续保留40GiB空闲守卫，空间紧张时保留原文件并扩容，不自动删除旧成果。
合成CPU基准中，同样64题×50段×128维、4遍读取，原整包约1.179秒，列读取约0.0228秒，
另有首次建列约0.300秒；传输载荷统计为85196800→6553600字节。
这是CPU微基准，不是实测GPU传输速度或整个393组实验的耗时预测；FIDELITY完整层重算未优化。

软件回归为52项基础检查与20项CPU张量测试，包括106个非序列方法/臂组合的分数、
损失、梯度和单步AdamW完全一致；FIDELITY原有真实随机小层测试仍通过。
旧结果跳过/原检查点绑定保存通过合成恢复测试；不宣称已经在师弟GPU上做过跨版本恢复。
可独立运行合成读取基准（只用CPU，不取模型/数据）：

```bash
python scripts/benchmark_top8_feature_io.py --mode optimized
```

## 2026-11-18：减少诊断同步，FIDELITY HEAD 跳过完整序列

在上述两项优化上继续减少等待，保持全部393组、FP32、原每批8题、逐题
forward/backward顺序、梯度裁剪和AdamW更新不变：

1. 每题反向前的非有限loss检查保留。loss／监督对数及更新前的梯度有限性、
   活跃性检查合并成一个小的数据包回传CPU，再按原题顺序累加Python浮点数；
   更新后的参数有限性合为一个CPU判断。没有把逐题loss改为批量前向或GPU求和，
   也没有推迟梯度错误到优化器更新之后。
2. FIDELITY的HEAD对照只用缓存的readout及已绑定原生分数，跳过hidden／mask
   的读取或重算。保留原候选位置、原chunk边界和每次线性头调用形状，
   不把整题合成更大的矩阵乘法。FIDELITY其余量子／经典注意力臂继续走原完整层路径。

52项基础检查、27项CPU张量测试通过；另在DPR/BGE随机小模型的FIDELITY HEAD和
FEATURE_GATE QUANTUM四个组合中，对照旧代码运行3步，并从旧第1步检查点恢复：
预测分数／排序、训练记录、参数、优化器、RNG与题顺序均完全相同。
空输入保护另由单测覆盖。这些是合成CPU验证，未运行负责人正式实验，
也不代表已经认证师弟GPU上的逐位等价或实际加速倍数。

诊断微基准的CPU标量提取调用从42次减至1次，另有1次合并数据包回传；
统计范围不含保留的每题反向前loss检查、梯度范数记录和方法内部检查。
这是同步点数量的代码路径证据，不是GPU耗时测量。
本轮两处不增加新的大型持久缓存，上述按字段列文件的空间估计仍适用。

精确接受两个旧源码版本到本版的继续运行：原87e239a版本或前两项优化3035225版本。
`runtime-speed-compatibility.v002.json`记录完整源码SHA映射；其余配置和输入必须相同。
旧矩阵绑定、完成结果和检查点保持原样；若已有v1续作收据也保留原字节，
另记`matrix/runtime-speed-continuation.v2.json`及对v1收据的SHA引用。
未知源码和改动科学配置仍拒绝恢复。

**先在完整检查点后停止旧Python并确认退出，再拉取；复用原workspace及参数：**

```bash
git pull --ff-only origin codex/top8-dual-scorer
bash scripts/run_triviaqa_smallset.sh --resume
```

完整cell核SHA后跳过；未完成cell从最后完整检查点继续。
### 2026-11-20：FIDELITY活动臂优化

本版进一步处理FIDELITY的MLP／GIVENS／PRODUCT／QUANTUM活动臂；仍计算完整原生
最后一层，保留全部token、FP32、原mask、DPR math-SDPA／BGE eager、原读出头、
393个组合及最终2000题的两次独立预测回放。没有采用CLS-only路径或减少检查。

- 完整序列缓存的访问时间先记内存，最多每30秒及正常关闭时保存索引。
  新增缓存正文仍立即写盘并核SHA，损坏仍拒绝使用；崩溃最多丢失最近淘汰顺序。
- 同一次前向中，已经按原矩阵形状计算的冻结Q/K投影供原生最后一层复用。
  仅限无输入梯度、冻结且无自定义hook的原生Linear；其余情况保留旧计算路径。
  无论成功或异常，临时调用替换都在finally中恢复。没有新增Q/K磁盘缓存。
- 冻结参考量子核使用每个模型最多32MiB的设备内存LRU；训练模型与最终回放模型
  合计最多64MiB。键绑定已核原始块、冻结Q/K及参考参数版本、形状、dtype和设备；
  转换设备／精度会清空缓存，参数恢复或变化会失效重算。缓存不进入检查点。
  新增大型磁盘缓存0；旧32GiB／模型的完整序列工作集上限不变。

`fidelity-speed-compatibility.v003.json`仅接纳列出的三个旧源码映射到本版；输入、
模型、seed、训练配置必须完全相同。新增`matrix/fidelity-speed-continuation.v3.json`，
原v1/v2收据先严格核对再按原字节保留，完成cell及旧检查点不覆盖。

DPR／BGE随机小层的CPU软件测试覆盖四个活动臂、多头、padding、五步AdamW、
冷缓存检查点恢复、梯度／参数／优化器／分数／排序逐位相同及异常清理。
它们不是师弟真实预训练GPU上的吞吐或逐位恢复认证。缓存命中率依赖工作集；
不据此承诺把10天缩短到固定天数。更新仍使用上面的pull和原目录resume命令，
不要在运行中的Python进程尚未退出时拉取。

### 2026-11-20：第四项——完整序列缓存命中优化

`--sequence-cache-gib`默认从32改为`auto`；仍支持显式非负整数，例如
`--sequence-cache-gib 32`或`0`（不落盘序列正文，CPU热缓存仍可用）。auto在特征／token
提取完成后，按每个实际缓存文件系统的可用空间分配；加入已有受管scratch字节避免resume
重复扣减，减去40GiB可用余量和48GiB后续列文件／结果增长估计，再在同盘模型间均分，
每模型最多256GiB。这是容量上限，按需写入，不预分配、不要求1.5TiB，也不保证完整工作集能放下。
写入时另检查实际缓存盘仍留40GiB；空间紧时跳过可再生缓存写入，训练计算照原路径进行。
手动更小容量仍可用，代价是更多冻结backbone重算；旧cache.json不改。

缓存满后优先保留已接纳的训练块，先淘汰低优先级块；新训练块也不会反复挤走已有训练块，
避免顺序遍历造成每个epoch循环淘汰。验证／测试选位不影响这一策略或训练监督。
预测和独立检查点回放之间使用每个活跃reader最多256MiB的CPU序列化热缓存；保存不可变
字节，读取时重新反序列化并验证原完整hidden／mask／native分数SHA，调用方改张量不污染缓存。
超上限块照原路径重算。没有新GPU张量缓存，没有减少token、臂、seed、epoch、步骤或预测回放。

每32步的训练JSON加入`sequence_cache`；缓存每60秒及关闭时也输出命中／重算次数、命中率、
host端读取和重算耗时、驻留字节／上限与淘汰统计。host时间包括加载与等待，不是CUDA kernel计时。
16项标准库和32项合成CPU张量测试通过，393cell计划不变；真实预训练GPU吞吐需在师弟机器测量。
`cache-policy-compatibility.v004.json`精确接纳原四版源码到本版；新增v4续作收据，旧v1/v2/v3
收据／原绑定／完成cell／检查点保持原字节。更新仍须在完整检查点后退出旧Python，再pull并原目录resume。

### 当前进度显示与CLI dashboard标准

目前脚本直接追加JSON：训练每32步显示idea／arm／seed／step，本版另带缓存命中与耗时；
提取、缓存和结果阶段另输出事件。外层shell直接exec Python，没有统一终端仪表盘，
也没有自动部署Web监控。当前不满足multi-gpu-experiment-orchestration完整CLI标准：
尚缺全局完成／运行／待跑／暂停／失败表、阶段和durable checkpoint／heartbeat分离、
滚动及全attempt速率、只覆盖已测阶段的ETA，以及TTY原地刷新／重定向追加和日志隔离。
未测的BGE或fusion耗时应保留unknown，不用已完成组合数量线性外推总时长。
此处记录核对结论；本次未把UI改造纳入缓存优化。

### 2026-11-20：统一终端进度界面（默认开启）

原命令默认启用统一CLI界面，无需另装UI依赖或另起服务：

```bash
git pull --ff-only origin codex/top8-dual-scorer
bash scripts/run_triviaqa_smallset.sh --resume
```

更新前先让旧Python退出，使用同一workspace和原参数。本版还修复上一版第32步缓存日志
误用`root`变量的问题，正确读取`cache_root`；不改任何训练计算。

界面每5秒刷新，显示393个学习组合的完成／运行／待跑／失败／暂停及当前任务、PID、
实际worker报告的设备／参数dtype（未测kernel仍标待测）。准备和提取、初始化、训练、
完整恢复检查点保存、2000题预测与逐题回放、联合封存、评估、结果回传分别显示。
训练4000/4000只表示训练计数完成，cell在预测／回放／文件提交完成后才记完成。
续跑的既有cell在worker核SHA后更新计数；重新检查期间不会把未核旧状态当完成。

速率来自本次实际计数事件，恢复起点使用已加载step，不把旧工作计入本进程速度。
切换训练与预测阶段会重置单位和计时范围；只估算当前计数阶段，不包含后续阶段或排队任务。
未测BGE／fusion和全套剩余时间保留待测，不以163/393等组合数量线性外推总天数。
阶段计数长时间未更新时暂停ETA，父心跳仍可保持在线，二者不等于失败。

TTY／tmux中原地刷新；重定向或TERM=dumb自动追加快照。可显式选择：

```bash
bash scripts/run_triviaqa_smallset.sh --resume --display append --refresh 10
bash scripts/run_triviaqa_smallset.sh --watch
```

自定义过workspace时，watch传入同一个`--workspace`。watch只读，不启动／恢复／停止任务，
退出watch不会影响训练。`--display off`保留旧原始输出模式；`--plan-only`仍不写目录。

显示状态在`<workspace>/progress/latest.json`原子替换，outer heartbeat、worker事件、
训练计数与本地检查点提交各自记录。完整stdout/stderr在
`<workspace>/progress/<attempt>/runner.log`，逐cell日志在同目录`jobs/`。
这些运行日志不因本次UI改动自动上传。父进程与worker实际退出状态分开记录；
worker返回0但仍有未完成计划项目时，界面入口返回2，不宣称实验完整完成。

Ctrl-C或TERM/HUP清理本次启动的进程组，返回中断状态；只认证已经提交的检查点，
不宣称正在执行的batch已安全保存。恢复继续使用原`--resume`规则；不自动重试失败cell。
`progress-ui-compatibility.v005.json`接纳列出的五版精确旧源码，追加v5收据，旧v1..v4不改。

软件验证28项标准库、33项合成CPU张量检查、52源码依赖、393cell计划和shell语法通过。
32步微型训练中，显示开／关的最终分数、参数和AdamW状态相同；中断、损坏显示管道、
reader故障、只读watch和未完成时exit0均有覆盖。这些不是师弟真实GPU／tmux的运行认证。
