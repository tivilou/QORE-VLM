# 仅回传旧对照：不重跑实验

适用：已经完成并上传 `triviaqa-smallset-20261122T011753Z` 的师弟工作区。此入口只补齐已有对照的回传，不训练、不推理、不安装环境、不重算标签，也不修改旧 `matrix`、`matrix-primary54`、checkpoint或缓存。

## 操作

在原项目、原分支 `codex/top8-dual-scorer` 同步后执行：

```bash
git pull --ff-only
bash scripts/run_triviaqa_smallset.sh --upload-existing-controls
```

默认沿用原工作区 `~/qduet-triviaqa-smallset` 和已有私有 `junior.json`。若原来指定了其他工作区，只沿用原来的 `--workspace` 值。不要加 `--resume`、`--prepare-only` 或 `--experiment-profile full` 来补跑。

## 收集范围和保护

- 先匹配本轮primary联合seal、原run-binding、两个cache manifest SHA、角色、种子及训练配置；不同小集、旧B1、canary或不同源绑定直接停止。
- 按预先固定的8方法×2模型×3种子非QUANTUM臂检查旧 `run-smallset/matrix`，不是按分数挑结果。
- 仅已有 `complete`、4000步、8000训练/2000评估且全部声明文件SHA匹配的单元，其旧预测和disposition才被回传。checkpoint只核SHA，不加载或复制其张量。
- 缺失、失败、只有checkpoint分别报告，不填零、不启动补跑。全局预算上限2GiB私有回传，单预测32MiB；超限在任何上传前停止。
- 原文件只读；小型清单写入工作区 `control-receipts/controls-*`，大预测直接从原文件流式上传到既有私有服务，不新建大压缩包、不经过师兄本机。
- 每个上传收据核对path/size/SHA，最后回传完整清单。网络失败保留pending记录，不自动循环重试。密钥沿用私有配置，不进入输出或Git。

成功输出的 `stage` 为 `existing_controls_uploaded`，同时给出 `complete`、`failed`、`checkpoint_only`、`not_present` 数量。这里的成功仅表示旧文件收集上传成功，不表示300个对照全部跑过或量子归因通过。

本次恢复清单是在主结果已报告后形成，不把它冒充历史“所有对照预测先于标签”的联合seal。师兄仍须核对完整角色/候选身份、原先完成证据及数值，再进行描述性比较；没有新生成预测或评估分数。
