# Exchange — 中间实验结果交换区

团队内部交换**中间测试与诊断结果**的地方。与 `results/` 的分工：

| | `results/` | `exchange/`（本目录） |
|---|---|---|
| 放什么 | 论文正式实验结果 | 中间测试、诊断报告、调参记录 |
| 生命周期 | 长期保留 | 论文发表后归档删除 |
| 要求 | 完整可复现（env/config/logs） | 够对方看懂就行 |
| 谁看 | 外部读者也可能看 | 只有我们几个 |

---

## ⚠️ 不要提交大文件

**只交换报告和元数据。** 具体地：

| 文件 | 大小 | 提交？ |
|---|---|---|
| `analysis/*.md` 诊断报告 | ~5KB/份 | ✅ 提交 |
| `status.json` / `run_summary.json` | <1KB | ✅ 提交 |
| `config/*.yaml` 实际用的配置 | ~3KB | ✅ 提交 |
| `experiments/*/result.json` | **4.1MB/个** | ❌ 不提交 |
| 打包的 `*.zip` | **2.4MB/轮** | ❌ 不提交 |

原因：git 删不掉历史。一轮 2.4MB，跑十轮就是 24MB 永久留在每个人的 clone 里，
而真正要读的只有那 20KB 报告。仓库 `.git` 现在已经 156MB 了。

**`result.json` 怎么给对方**：还是打包直传（微信/网盘/scp）。需要它的场合只有一个 ——
对方要用**别的脚本**重新分析你的原始数据。那种时候单独发。

`.gitignore` 已经拦了 `exchange/**/result.json` 和 `exchange/**/*.zip`，
误 `git add` 会被挡住。真要提交某个大文件得 `git add -f`，那时请先说一声。

---

## 怎么用

```bash
cp -r exchange/_TEMPLATE exchange/p1_diagnosis_20260727
cd exchange/p1_diagnosis_20260727
# 填 README.md，把 analysis/*.md 和 status.json 拷进来
git add exchange/p1_diagnosis_20260727
git commit -m "exchange: Phase 1 诊断结果（200题，缺F1）"
git push
```

命名：`<做了什么>_<YYYYMMDD>`。同一天跑两轮就 `_a` / `_b`。

## 目录结构

```
exchange/
├── _TEMPLATE/                      # 复制这个
└── p1_diagnosis_20260727/
    ├── README.md                   # 谁跑的、跑了什么、结论、有什么问题
    ├── analysis/                   # 诊断报告 .md
    ├── meta/                       # status.json / run_summary.json
    └── config/                     # 实际用的 yaml（不是仓库里那份，是你真跑的那份）
```

`config/` 那条容易被忽略但很重要 —— **手改过 yaml 一定要放进来**。
2026-07-27 那轮就是因为手加了 `--skip_generation`，直到我看 `status.json`
里的命令行才发现，白跑一轮。
