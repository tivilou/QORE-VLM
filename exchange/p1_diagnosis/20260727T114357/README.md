# Phase 1 诊断（4 个 idea 前提验证）— 师弟, 2026-07-27 11:43 北京时间

## 这轮想回答什么问题

四个 idea 的前提假设各自成不成立？

- idea 1 两阶段 QUBO：质量与多样性在单目标里是不是冲突？
- idea 4 上下文完整性：独立选择是不是破坏了段落间依赖？
- idea 6 互补性矩阵：只避冗余够不够拿到最优信息覆盖？
- idea 7 Soft QUBO：QUBO 代理目标与任务目标偏离多少？

目的是**数据驱动定 Phase 2/3 的优先级**，不是出论文数字。

## 一句话结论

idea 6 和 idea 7 的假设**都成立、可用于排优先级**；idea 4 部分成立；
**idea 1 判不了** —— 这轮手加了 `--skip_generation`，没有 F1。

## 跑了什么

- **配置**: `phase1_diagnosis.yaml`，200 题 × 3 个 γ (0.0/0.5/1.0)，wiki_dpr + answer scorer
- **真实命令**: `config/actual_commands.txt`（从 `meta/status_*.json` 的 `command` 字段抽的）
- **耗时**: 每配置约 97-103 秒，共约 5 分钟（**因为没跑生成**；开生成是 30-35 分钟/配置）
- **时间线**（北京时间）: 实验 11:43:57 → 11:48:54，诊断报告 11:57:58 → 11:58:02
- **改动**: ⚠️ **手改了 yaml，加了 `--skip_generation`**。仓库当前版本没有这个 flag
  （`6f03fdb` 专门去掉的，因为诊断依赖 F1）。另外 checkout 里有两个诊断脚本是旧版。

## 结果

| 配置 | Recall@5 | Precision@5 | 冗余度↓ | 多样性↑ | F1 |
|---|---|---|---|---|---|
| γ=0.0 | **0.4989** | **0.3910** | 0.8218 | 0.1782 | — |
| γ=0.5 | 0.4253 | 0.3510 | 0.7860 | 0.2140 | — |
| γ=1.0 | 0.3909 | 0.3270 | 0.7759 | 0.2241 | — |

200 题，156 题检索命中 gold（44 题检索阶段就失败）。
池子 50 候选 → prefilter 到 15 → 选 5。γ 未饱和（Jaccard 0.564 < 0.95）。

## 结论可信度

**这栏比上面的数字重要。**

- ✅ **idea 6 互补性矩阵 — 假设成立，证据最干净**
  最低冗余子集覆盖 0.3145 vs 最高覆盖子集 0.8913，只有 14/150 (9.3%) 两者一致。
  且最高覆盖子集的冗余（0.7930）**高于**最低冗余子集（0.7529）—— 一味降冗余会牺牲覆盖的直接证据。
  不依赖 F1，所以这轮缺生成不影响。

- ✅ **idea 7 Soft QUBO — 假设成立**
  同题内枚举：QUBO 能量最低子集平均 Recall 0.5909 vs oracle 0.8913，
  命中最优 52/150 (34.7%)。用的是 selector 记录的真实 a/b 和 `qore.qubo` 的实现。
  局限：用 Recall 代理 F1（真算 F1 oracle 要每题 792 次生成），候选池截到 12 条所以 gap 是**下界**。

- ⚠️ **idea 4 上下文完整性 — 部分成立**
  实体连贯性 selected 0.099 vs gold 0.165（差 -0.067，配对检验 p=0.0000），
  但悬空代词两组都是 0.00 —— 两项指标只有一项显示 selected 更差，证据不够强。
  gold 对照组有偏（gold 文本只在被选中时才进 dump），会**低估**真实差距。

- ❌ **idea 1 两阶段 QUBO — 判不了**
  γ 生效了，但缺 F1。现在只能说 γ=0.0 的 **Recall** 最高。
  `gamma_sweep.md` 里那句「建议不实现两阶段 QUBO」**不要采信** —— 判据缺一半。
  冗余度确实随 γ 降了 4.6 个百分点，多样性是不是真伤了 QA 质量，得看 F1。

- ❌ **`analysis/*.OLD-SCRIPT.md` 两份作废**
  见下节。

## 遇到的问题

**1. `--skip_generation` 把 idea 1 的判定废掉了**

`status.json` 的命令行里同时有 `--skip_generation` 和 `--dump_passages`，
这个组合不匹配仓库任何一个版本。samples 里一个 f1 都没有，`prediction` 全是 `None`。

**2. 两份报告是旧版脚本产出的，数字不可信**

四份报告 mtime 是 11:57:58 → 11:58:00 → 11:58:00 → 11:58:02（北京时间），正好 driver 的
`[1/4]`~`[4/4]` 顺序，同一次运行。但格式新旧混杂：

| 报告 | 版本 | 用当前脚本复跑 |
|---|---|---|
| `gamma_sweep.md` | ✅ 新版 | 逐字一致 |
| `complementarity.md` | ✅ 新版 | 数字逐个吻合 |
| `context_dependency.OLD-SCRIPT.md` | ❌ 旧版 | 完全不同 → `analysis_corrected/` |
| `qubo_objective.OLD-SCRIPT.md` | ❌ 旧版 | 完全不同 → `analysis_corrected/` |

**`qubo_objective.OLD-SCRIPT.md` 尤其要作废。** 它报了「样本数 139」「平均 F1 0.423」
「Pearson -0.215」「✅ 假设强烈成立」，可这份 result.json **没有任何 f1 字段**。
把 `0bfa8ec` 版旧脚本拿同一份 result.json 实跑，输出是「加载 0 个查询结果 / 没有有效数据」，
静默退出 0 且不写文件 —— 那 139 个样本和那些 F1 数字**不是从这份数据来的**，来源未确定。

而且旧版设计本身是失效的（`6f03fdb` 已说明）：它做**跨题**相关性，
实测跨题 r=0.007、同题内 r=0.986 —— 能量量级被每题自己的候选池尺度主导，
所以无论 QUBO 多好都会报「相关性低 → 假设成立」。那个 ✅ 是设计缺陷的产物，不是发现。

**3. `quick_summary.md` 的小 bug**

写着「最佳 F1: gamma_1.0」但三行 F1 全是 `0.0000` —— 全零上取 argmax 挑了个任意赢家。
`result_analyzer.py` 的问题，不影响结论。

## 环境

- git HEAD: 未提供（checkout 状态混杂：`complementarity_diagnosis.py` /
  `gamma_sweep_diagnosis.py` 是新的，`context_dependency_diagnosis.py` /
  `qubo_objective_diagnosis.py` 是旧的）
- `git status` 干净吗: 否（至少 yaml 被手改过）
- 推测：对那两个脚本有本地改动，`git pull` 拒绝覆盖

## 原始数据

`result.json` 没提交（3 个 × 4.1MB，见 `exchange/README.md`）。

- 来源: `P1_diagnosis_20260727.zip`（2.4MB），师弟 2026-07-27 直传
- 本地路径: `/home/Q-DUET-VLM/P1_diagnosis_20260727.zip`

## 下一步

重跑只为拿一样东西：**idea 1 的 F1 判定**。idea 6 / 7 的结论已经可用。

建议干净 clone 再跑（`wiki_dpr` 缓存在 `~/.cache/huggingface`，仓库外，不会重下）：

```bash
git clone https://github.com/tivilou/QORE-VLM.git QORE-VLM-clean
cd QORE-VLM-clean && git log --oneline -1     # 应为 b4aa6ab 或更新
bash scripts/collab/run_phase1_quick.sh
python scripts/diagnosis/check_dump_fields.py --expect_samples 10
```

**yaml 一个字都别改**，特别是别加回 `--skip_generation`。

跑之前先确认生成能用（默认 `meta-llama/Meta-Llama-3-8B-Instruct`，gated 模型要 HF token）：

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct'); print('OK')"
```
