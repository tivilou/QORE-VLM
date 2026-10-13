# Quantum Reader feature-rotation screen contract

日期：2026-10-14  
工作流：rag-selector

## 本次决定

固定 Top-50 → Top-5、DPR Reader、检索候选、Generator 和 evaluator，使用冻结的
59 个训练问题/228 个 witness pair，以及固定 100 题评估面板。新机制是把问题条件化的
真实酉旋转放在 Reader 最终 CLS 表征与原 relevance classifier 之间；零参数是严格恒等，
10 个数据 qubit、2 层、200 个角参数。量子臂与相同预算的经典坐标旋转臂成对比较。

为了区分机制和损失函数，正式 screen 固定六个臂：Reader Top-k、旧 head+旧 loss、旧
head+新 Top-5 retention loss、新量子 head+旧 loss、新量子 head+新 loss（主候选）、
新经典 head+新 loss（量子归因控制）。旧语义 head 的投影参数量与 200 参数旋转头不同，
因此只把量子/经典旋转臂作为参数匹配的归因对；六个臂共享训练顺序、seed、epoch、优化器
预算和梯度裁剪。

## 证据边界

labels 只进入训练 loss，评估标签、Silver 和三模型判断在所有臂完成完整 50 候选排序后才
做 posthoc 统计。主门槛是新量子 head+新 loss 相对冻结 Reader 的正证据 consensus 配对
均值与 bootstrap 下界都为正，同时 direct support 不下降；量子归因门槛是相对匹配经典臂
同样成立。任一门槛失败、Reader/span/null/trace 完整性失败都停止，不增加 epoch、残差界或
Silver 调参。

## 实现与验证

- 配置：`configs/experiments/quantum_reader_rotation_screen_100.json`
- 插件计划：`configs/experiments/quantum_reader_rotation_screen_100_plan.json`
- runner：`scripts/collab/five_ideas/run_quantum_reader_rotation_screen_100.py`
- 一键脚本：`scripts/collab/five_ideas/run_quantum_reader_rotation_screen_100.sh`
- 本地 lambeq 环境通过 14 个 focused unittest；Python 编译、JSON 和 6-plugin plan 校验通过。
- 固定 Reader 工程预检已在 RTX3060 通过；本次没有下载 Wiki-DPR，也没有运行真实 cohort fitting。

## 下一步

把项目根目录的一键命令交给师弟执行。收到输出后重新核验 upload manifest、文件 SHA、完整
trace 和每个臂的 Top-5 统计，再决定保留、转向或关闭该候选。
