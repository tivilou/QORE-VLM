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
| `status.json` / `run_summary.json` / `summary.csv` | <1KB | ✅ 提交 |
| 实际用的配置 + 真实命令行 | ~3KB | ✅ 提交 |
| RAG `result.json`（带 `--dump_passages`） | **4.1MB/个** | ❌ 不提交 |
| KV `*.samples.json` | **百 KB～MB 级** | ❌ 不提交 |
| 打包的 `*.zip` | **2.4MB/趟** | ❌ 不提交 |

原因：git 删不掉历史。一趟 2.4MB，跑十趟就是 24MB 永久留在每个人的 clone 里，
而真正要读的只有那 20KB 报告。仓库 `.git` 现在已经 156MB 了。

**大产物怎么给对方**：还是打包直传（微信/网盘/scp）。需要它的场合只有一个 ——
对方要用**别的脚本**重新分析你的原始数据。那种时候单独发。

`.gitignore` 已经拦了 `exchange/**/result.json`、`exchange/**/*.zip`、
`exchange/**/*.samples.json`、`exchange/**/experiments/`，误 `git add` 会被挡住。
真要提交某个大文件得 `git add -f`，那时请先说一声。

---

## 怎么用

**两层结构**：实验类型 / 趟次。同一个实验跑多趟是常态（修了 bug 重跑、
换参数重跑），所以趟次归在实验类型下面，而不是平铺在 `exchange/` 里。

**新实验类型**（第一次跑某个实验）：

```bash
cp -r exchange/_TEMPLATE_EXPERIMENT exchange/kv_balanced_rerun
# 填好 exchange/kv_balanced_rerun/README.md 的趟次表
```

**新趟次**（已有的实验又跑了一次）：

```bash
TS=$(date +%Y%m%dT%H%M%S)          # 跑实验前先记下开始时刻
cp -r exchange/_TEMPLATE exchange/p1_diagnosis/$TS
# 把 analysis/*.md 、status.json 、实际用的配置拷进去，填 README.md
# 然后到 exchange/p1_diagnosis/README.md 的趟次表里加一行
git add exchange/p1_diagnosis
git commit -m "exchange: P1 诊断 $TS（开生成重跑）"
git push
```

## 命名与时区

- **实验类型层**: 描述性名字，不带日期、不带 idea 编号
  （`p1_diagnosis`、`kv_balanced_rerun`。idea 编号会变 —— `6f03fdb` 已经把 idea 2/3
  从 driver 里摘掉过一次）
- **趟次层**: `<YYYYMMDD>T<HHMMSS>`，**北京时间**，取**第一个实验的开始时刻**

```bash
date +%Y%m%dT%H%M%S      # 不加 -u，我们用北京时间
```

取开始时刻而非打包时刻，是为了让目录名能和 `meta/status_*.json` 里的
`start_time` 直接对上 —— 排查时要拿目录名去 grep 日志、对时间线。

⚠️ `status.json` 里是 `datetime.now().isoformat()`，**裸本地时间没有时区标记**。
两台机器的裸时间戳无法互相排序 —— 分析 2026-07-27 那个 zip 时就踩过：
zip 内部记 `11:57:58`，`unzip -l` 显示 `03:57`，差 8 小时。
**本目录统一北京时间，别混 UTC。**
（`.ai-progress/` 用的是 UTC 带 `Z` 后缀，是另一套，别看错。）

## 目录结构

```
exchange/
├── README.md                        # 本文件
├── _TEMPLATE/                       # 单趟的模板
├── _TEMPLATE_EXPERIMENT/            # 实验类型层的模板（趟次表）
└── p1_diagnosis/
    ├── README.md                    # 趟次表：哪趟有效、废的为什么废
    └── 20260727T114357/
        ├── README.md                # 想答什么问题、结论、可信度、改动
        ├── analysis/                # 诊断报告 .md
        ├── meta/                    # status.json / run_summary.json
        └── config/                  # 实际用的配置 + 真实命令行
```

**实验类型层的 `README.md` 是关键。** 有了同级的多趟，必须有个地方回答
「哪趟是当前有效的」。这和 `results/` 不同 —— 那里每个目录是独立的正式结果，
而这里**后一趟往往是前一趟的修正**。废的趟次一定要写清为什么废。

`config/` 那条容易被忽略但很重要 —— **手改过配置一定要放进来**。
2026-07-27 那趟就是因为手加了 `--skip_generation`，直到我看 `status.json`
里的命令行才发现，白跑一轮。
