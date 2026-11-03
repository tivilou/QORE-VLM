# NQ-open＋TriviaQA：师弟一键实验

默认入口运行两套独立训练／评估，不混合训练。每套都使用同版本 DPR multiset、BGE-v2-m3，八项方案、全部匹配经典控制、原模型和三个固定种子。每套348个学习单元，合计696个；正式规模实验由师弟执行。

## 一条启动命令

```bash
curl -fsSL https://raw.githubusercontent.com/tivilou/QORE-VLM/codex/top8-dual-scorer/scripts/bootstrap_two_datasets.sh | bash
```

已有最新代码目录也可直接运行：

```bash
bash scripts/run_two_datasets.sh
```

代码、数据、环境、共享资产和实验结果均写入运行服务器的 `qduet-two-datasets/`，不经过owner电脑。首次克隆后固定该代码；相同命令重入不自动pull变更。若已有旧检出，保留其工作目录，使用新的 `QDUET_RUN_ROOT`；不要在旧实验中途换版本。

**当前入口代码已具备，但默认正式人口尚不能开始训练。** 两个正式数据发布包仍未就绪。默认命令会同时报告缺口，在依赖安装、大文件下载、模型加载前退出2，不静默只跑一个数据集，也不以曝光小集代替正式实验。

只查看完整矩阵与就绪情况，不联网、不写输出、不需要凭据：

```bash
bash scripts/run_two_datasets.sh --plan-only
```

当前可实际复现的是TriviaQA曝光B1，512训练／128开发：

```bash
curl -fsSL https://raw.githubusercontent.com/tivilou/QORE-VLM/codex/top8-dual-scorer/scripts/bootstrap_two_datasets.sh | bash -s -- --datasets triviaqa --population reproduction
```

该命令是B1复现，不代表NQ已运行、正式验证成功或跨数据集迁移成功。

## 数据状态与补齐责任

| 数据集 | 已存在的依据 | 尚缺 |
|---|---|---|
| TriviaQA | 61,689／7,891／7,606题身份于2026-10-21冻结；8,531题Top50桥包含全部7,891验证题，历史回放完成 | 扩展训练输入、合格的模型可见文本0/1/2标签发布、最终测试候选／标签；当前候选文件尚未全量重新核SHA |
| NQ-open | 历史2,742题全量结果及配置保留 | 可恢复、当前schema完整输入／训练监督／评估标签发布；旧人口已曝光，不能改称新独立测试 |

2026-11-03只读服务器元数据确认TriviaQA桥目录和备份仍存在；目录存在不替代内容或标签审计。此前“TriviaQA只有规划、身份未冻结”的概括不准确，本页按原冻结和桥完成收据纠正；旧审计保留历史。

61,689含512训练、128已曝光开发、61,049训练保留。保留开发用途时不能把128题也拟合，扩大拟合人口最多61,561；具体正式划分须按完整组件重新发布。官方问答答案不自动成为段落0/1/2标签，不用答案字符串构造真值。新正式数据先核问题泄漏、模型可见文本标签和可提升上限，后运行实验。

## 自动获取与私有配置

复用 `~/.config/qduet/junior.json` 或 `QDUET_HANDOFF_CONFIG`，其中 `url`、`token` 仅通过项目私有渠道安装，不放GitHub或命令行。owner准备release后在同一私有配置登记：

```json
{
  "url": "项目私有服务地址",
  "token": "仅私下安装的令牌",
  "datasets": {
    "nq_open": {"populations": {"formal": "完整release对象"}},
    "triviaqa": {"populations": {"formal": "完整release对象"}}
  }
}
```

上面是结构示意，两个release不能填字符串。每份release沿用 `junior-campaign.v001.json` 中的对象格式：

- `dataset`分别为`nq_open`、`triviaqa`；`population=formal`、`status=ready`，精确`counts.train/evaluation`。
- `archive`：项目私有服务的`exchange_path`、`size`、`sha256`；`files`：每个成员的`path/size/sha256`。
- `inputs.train/eval_inputs/eval_labels`为包内相对路径，评估标签与输入分离；每题原Top50全保留。
- `models`固定为`models/dpr-reader-multiset-base`与`models/bge-reranker-v2-m3`，两个数据集必须具有相同模型逐文件SHA；实际科学加载还会检查原固定模型清单。
- `exposed_population=false`、`independent_confirmation=true`、`label_semantics=reviewed_none_partial_direct`及`qualification_evidence`。
- `qualification.cross_dataset_question_audit`、`model_visible_label_audit`、`oracle_or_headroom_evidence`分别指向实际审计证据。标志或非空路径本身不是科学资格证明，owner必须先完成其所指的实际审查。

师弟无需输入训练／评估／标签三个路径。脚本自动获取整包，核整包及成员SHA，按完整归档和清单缓存，共用环境。相同包只安装一次；不同包即使含相同模型也可能占用额外空间。跨数据集训练／评估题目做NFKC、大小写和空白规范化身份检查；原文输入不修改，这不是语义去重证明。共享Wikipedia段落不被自动删除，测试间问题重叠单独记录。

## 输出、排序与恢复

独立缓存和训练放在`work/datasets/nq_open/`与`work/datasets/triviaqa/`；每套全部预测封存后才读本套评估标签。两个训练配置在运行前固定，前一套结果不驱动后一套调参。当前不安排“TriviaQA训练→NQ测试”的零样本迁移，也不把两套独立训练结果叫作迁移结果。

保留原模型和全部经典控制，不以胜原模型自动认证量子归因。每个数据集／模型分别报告原始计数、完整分母与未审上下界；综合排序按四个组合的“每题useful相对原模型增量”取等权均值。任何缺失种子、失败矩阵或缺失数据集均不参与伪完整排名；同时报告四个组合是否全部改善及direct是否全受保护。区间重叠仍未解决，不以保守下界排序冒称显著性。

`work/two-dataset-summary.json`保存综合表；每套的预测、训练记录、检查点、seal和评估自动回传私有服务，综合表也单独回传，产生`two-dataset-upload.json`。GitHub只发布代码。上传失败返回非零，重入仅补传已终态实验；未完成训练沿原恢复机制接续，不自动重试失败单元。总工作锁与子实验锁均由科学子进程继承。

只读计划和标准库合成测试不证明正式规模吞吐／内存通过。原生完整序列FIDELITY缓存可能很大；40GiB只是最低保留空间，不能当全量所需磁盘估计。正式release应附实际缓存／内存规划，空间不足应扩容服务器，不占owner本机、不删历史结果，也不偷偷删臂、降低精度或改输入。
