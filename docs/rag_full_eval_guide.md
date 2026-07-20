# RAG 全量评测实验指引

## 更新（2026-07-20）

**gamma 参数修复**（commit `f6672f2`），本轮命令必须显式加 `--gamma 0.5`：

1. **新增 `--gamma`（多样性权重，本轮的关键参数）**
   - 根因：冗余惩罚权重此前固定 1.0，K=5 时 10 个成对冗余项累加压过质量项，
     导致 QORE 丢失含答案段落（上轮 Recall@5 比基线低约 11 个百分点）
   - 50k 子集网格搜索（200 题）：**gamma=0.5 最优** — Recall 20.23%（gamma=1.0 仅
     12.69%），冗余度 0.675 仍显著低于基线
2. **纠正 2026-07-19 版第 2 条**：`lam` 不是质量-多样性权衡参数，它只控制"恰好选
   K 段"的基数约束，对选哪 K 段没有影响；默认值已回滚为 `lam=2.0`，无需调整
3. 上轮修复保持有效（commit `2fc455f`）：DPR 检索分数传递、简短答案 prompt
   （EM 已从 0 恢复到 ~25%）

**评测改进**（commit `df972e7`），本轮自动生效：

4. **Gold 判定改进**：从简单字符串包含改为 token 边界匹配，避免误命中（如 "2012" 
   匹配 "20122"），Recall/Precision 数值更准确
5. **评测标注改进**：输出新增 Recall@Retrieved（检索上界，如 Recall@50）和
   retrieval failure 统计，分离检索失败和重排失败
6. **可选优化**：新增 `--qore_prefilter_size` 参数（默认 max(K*3,15)），若 gamma=0.5
   仍不达标可尝试 `--qore_prefilter_size 15` 进一步限制候选池（减少低相关段风险）

**参考**：200 样本实验报告（`/home/Q-DUET-VLM/analysis(1).md`）的根因分析。

---

## 概述

RAG 模块代码已完成（模块化重构 + 内存优化 + gamma 参数 + 评测改进）。本轮分两步执行：
先跑 200 题验证 gamma=0.5 的效果，达标后再跑全量 NQ 评测。

**机器要求**：
- RAM: 32GB+（推荐 64GB）
- GPU: 24GB 显存（用于生成答案）
- 磁盘: 100GB+ 可用空间（HuggingFace 缓存）
- 网络: 能访问 HuggingFace（首次运行会下载 ~80GB wiki_dpr 数据集；上轮已运行过的
  机器会直接命中缓存）

---

## 实验步骤

### 1. 拉取最新代码

```bash
cd /path/to/QORE-VLM
git pull origin main
```

### 2. 第一步：200 题验证（gamma=0.5）

```bash
# 在 QORE-VLM 根目录下运行
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 200 \
  --methods qore,mmr,topk \
  --seeds 42 \
  --K 5 \
  --lam 2.0 \
  --gamma 0.5 \
  --output_dir results/rag/gamma05_validation
```

**注意**：
- `--methods` 用**逗号分隔**（用空格会报 unrecognized arguments）
- 本地已有模型的话加 `--model_path models/llama3-8b`（按实际路径）
- 旧版文档中的 `--num_workers` 参数不存在，不要加

**通过标准**（对比上轮 gamma=1.0 的 200 题结果）：
- [ ] QORE Recall@5 ≥ 30%（上轮 24.4%；MMR 34.9%）
- [ ] QORE F1 ≥ 45%（上轮 40.9%；MMR 47.1%）
- [ ] QORE 冗余度 < 0.80（保持明显低于 MMR 的 0.832）

三项达标 → 执行第二步全量评测；任何一项不达标 → 按交付方式推送 JSON 后停止，
等分析后再定。

### 3. 第二步：全量评测

**建议在 tmux 中运行**（避免 SSH 断开中断）：
```bash
tmux new -s rag_eval
```

```bash
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 0 \
  --methods qore,mmr,topk \
  --seeds 42 \
  --K 5 \
  --lam 2.0 \
  --gamma 0.5 \
  --output_dir results/rag/full_eval_21M
```

- `--max_samples 0`：使用全部样本（NQ validation 约 3600 题）
- Ctrl+B, D 分离 tmux；`tmux attach -t rag_eval` 重新连接查看进度

### 4. 检查输出

每步运行结束后，输出目录下应该有：
```
qore_K5_seed42.json
mmr_K5_seed42.json
topk_K5_seed42.json
summary.json          # 自动生成的汇总结果
```

**检查点**：
- [ ] 3 个 JSON 文件都生成了
- [ ] QORE 的 Redundancy 明显低于 MMR/Top-K
- 注：单 seed 下 summary.json 不含 t-test（跨 seed 显著性检验需要多 seed）

---

## 交付方式

**将结果提交到 GitHub**（`results/` 在 .gitignore 中，需要加 `-f`）：

```bash
git add -f results/rag/gamma05_validation/*.json
git commit -m "实验结果:RAG 200题验证(gamma=0.5)"
git push origin main
```

全量评测完成后同样操作（目录换成 `results/rag/full_eval_21M`）。

**关键指标**（在 summary.json 中）：
1. **Recall@5**：QORE vs MMR vs Top-K（越高越好）
2. **Redundancy**：QORE 应该最低
3. **EM / F1**：最终答案质量
