# P1 诊断 — 趟次记录

验证 4 个 idea 的前提假设（idea 1 两阶段 QUBO / 4 上下文完整性 /
6 互补性矩阵 / 7 Soft QUBO）。

**执行**: `bash scripts/collab/run_phase1_full.sh` →
`bash scripts/diagnosis/run_all_diagnosis.sh`

---

## 趟次

| 趟次（北京时间） | 谁 | 结论 | 状态 |
|---|---|---|---|
| [`20260727T114357`](20260727T114357/) | 师弟 | idea 6/7 假设成立；idea 4 部分成立；idea 1 判不了 | ⚠️ **部分作废** |
| [`20260727T201307`](20260727T201307/) | 师弟 | | |

### `20260727T114357` 为什么部分作废

两个独立问题，都不是数据本身的问题（三个实验 `status` 全 `success`，dump 字段齐全）：

1. **手改 yaml 加了 `--skip_generation`** → samples 里一个 f1 都没有，
   `prediction` 全 `None`。idea 1 的判据（γ 增大在降冗余的同时是否伤了 QA 质量）
   缺一半，`gamma_sweep.md` 里「建议不实现两阶段 QUBO」那句**不要采信**。
2. **两份报告是旧版脚本产出的** → `analysis/*.OLD-SCRIPT.md`。
   其中 `qubo_objective` 那份报了「平均 F1 0.423」，而这份 result.json 没有任何 f1 字段；
   旧版设计本身也是失效的（做跨题相关，跨题 r=0.007 vs 同题内 r=0.986）。
   已用当前脚本复跑，放在 `analysis_corrected/`。

**仍然可用的**: idea 6 和 idea 7 的结论。两者都用 Recall 代理，不依赖 F1，
数字已和当前脚本复跑逐个核对过（`complementarity.md` 与 `gamma_sweep.md` 逐字一致）。

---

## 命名

`<YYYYMMDD>T<HHMMSS>`，**北京时间**，取**第一个实验的 `start_time`**。

取实验开始时刻而非打包时刻，是为了让目录名能和 `meta/status_*.json` 里的
`start_time` 直接对上 —— 排查时要拿目录名去 grep 日志、对时间线。
这轮实验 11:43 开始、报告 11:57 生成、打包更晚，用打包时刻会让目录名和内容里的
时间戳错开。

跑之前先记下开始时刻：

```bash
date +%Y%m%dT%H%M%S      # 注意：不加 -u，我们用北京时间
```

⚠️ `status.json` 里的时间戳是 `datetime.now().isoformat()`，**裸本地时间没有时区标记**。
两台机器的裸时间戳无法互相排序 —— 分析 2026-07-27 那个 zip 时就踩过：
zip 内部记的是 `11:57:58`，而 `unzip -l` 显示 `03:57`，差 8 小时。
所以本目录统一按北京时间，别混 UTC。
（`.ai-progress/` 用的是 UTC 带 `Z` 后缀，是另一套，别看错。）
