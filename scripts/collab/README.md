# 协作脚本

按实验阶段组织的自动化脚本，用于运行实验和提交结果到 `exchange/`。

## 阶段

- **[p1_diagnosis/](p1_diagnosis/)** — Phase 1 诊断（γ sweep、idea 6/7 验证）
- **[p2_solver_idea6/](p2_solver_idea6/)** — Phase 2 solver 修复 + idea 6 互补性矩阵

## 通用工具

- **[lib/collect_lib.py](lib/collect_lib.py)** — 共享收集函数（git 状态、GPU 信息、JSON 加载等）
- **[setup_env.sh](setup_env.sh)** — 环境配置检查

## 工作流程

每个阶段子目录包含：
- `README.md` — 完整工作流程说明（给执行者看）
- `run_*.sh` — 自动运行实验脚本
- `collect_*.py` — 汇总结果到 `exchange/<stage>/` 脚本

典型流程：
```bash
# 1. 进入对应阶段子目录
cd scripts/collab/p2_solver_idea6  # 例如 P2

# 2. 阅读 README.md，了解实验配置和依赖

# 3. 运行实验（自动汇总）
bash run_p2_experiments.sh

# 4. 提交到 GitHub
git add ../../exchange/p2_solver_idea6/<timestamp>
git commit -m "experiment(p2): solver+idea6 results <timestamp>"
git push
```

## 目录结构

```
scripts/collab/
├── README.md                 # 本文件
├── setup_env.sh              # 环境配置（通用）
├── lib/
│   └── collect_lib.py        # 共享工具函数
├── p1_diagnosis/
│   ├── README.md             # P1 工作流程说明
│   ├── PHASE1_STEPS.md       # P1 详细步骤（历史文档）
│   ├── run_phase1_full.sh    # 跑完整 P1 诊断
│   ├── run_phase1_quick.sh   # 快速验证（少量样本）
│   └── collect_p1_results.py # 汇总 P1 结果
└── p2_solver_idea6/
    ├── README.md             # P2 工作流程说明
    ├── run_p2_experiments.sh # 跑 10 组实验（solver fix + idea 6 grid）
    └── collect_p2_results.py # 汇总 P2 结果
```

## 设计原则

1. **阶段隔离** — 每个 Phase 独立子目录，互不干扰
2. **文档就近** — 工作流程说明在脚本旁边，不用跳到 `exchange/` 找
3. **可复用** — `lib/` 放共享函数，各阶段 import
4. **可扩展** — 未来 Phase 3/4 直接加新子目录

## 与 exchange/ 的关系

- **`scripts/collab/<stage>/README.md`** — **如何跑实验**（给执行者看）
- **`exchange/<stage>/README.md`** — **实验需求说明**（给读者看结果用）

执行者看前者，评审者看后者。

---

**对象**: 团队内部成员  
**状态**: 🔒 内部协作使用（暂时公开）
