# Experiment Results

本目录存放论文的正式实验结果。每位实验者一个子目录。

## 提交方式

1. 复制 `_TEMPLATE/` 并改名为 `<姓名>_<日期>/`
2. 按模板格式填写结果
3. commit 并 push

## 目录结构

```
results/
├── _TEMPLATE/          # 模板，复制后改名使用
└── Bootrear_2026-XX/   # 示例
    ├── README.md       # 概述：跑了什么、结论
    ├── env.md          # 硬件和软件环境
    ├── config/         # 实验配置/脚本
    ├── logs/           # 原始日志
    ├── results/        # 指标文件 (csv/json)
    └── notes.md        # 备注
```
