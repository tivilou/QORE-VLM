# Answer Scorer 全量 3-Seed 实验指令

## ✅ Smoke Test 结果确认

**40题 × 3 seeds 验证通过**（2026-07-23）：
- ✅ F1: 59.9% (vs MMR 55.7%, +4.2 个百分点, p<0.001 ***)
- ✅ Recall@5: 44.3% (vs baseline 33.76%, +10.5 个百分点)
- ✅ 冗余度: 0.793 (vs MMR 0.805, p<0.01 **)
- ✅ EM: 34.2% (vs MMR 30.0%, +4.2 个百分点, p<0.05 *)
- ✅ 配置正确: use_answer_scorer=True, skip_generation=False
- ✅ 3-seed 方差极小 (std < 0.01)，结果稳定

**结论**：Answer Scorer 在端到端评测中显著提升 QA 性能，同时保持冗余度优势。

---

## 🚀 全量实验命令

```bash
cd /path/to/QORE-VLM

# 确保最新代码
git pull origin main  # 最新 commit: cd8745b

# 创建 tmux 会话（推荐）
tmux new -s rag_answer_scorer_full

# 全量 3-seed 实验
python -m scripts.rag.eval_suite \
  --corpus_mode wiki_dpr \
  --dataset nq_open \
  --max_samples 0 \
  --methods qore,mmr,topk \
  --seeds 42,123,456 \
  --K 5 \
  --lam 2.0 \
  --gamma 0.5 \
  --direct_solve_max_n 20 \
  --use_answer_scorer \
  --answer_scorer_backend dpr \
  --output_dir results/rag/answer_scorer_full_3seeds

# 分离 tmux: Ctrl+B, D
# 重新连接: tmux attach -t rag_answer_scorer_full
```

---

## ⏱️ 预期耗时

- **样本数**: 3610 题 × 3 seeds × 3 methods = 32,490 次推理
- **每题耗时**: ~77ms (选择) + ~235ms (生成) ≈ 312ms
- **单个方法 × 3 seeds**: 3610 × 3 × 0.312s ≈ **56 分钟**
- **三个方法总计**: 56 × 3 ≈ **2.8 小时**

**实际可能更长**：
- DPR reader 首次推理有缓存加载
- 生成时间可能波动（LLM 负载）
- 建议预留 **3-4 小时**

---

## 📊 预期结果（基于 40 题外推）

| 方法 | Recall@5 | F1 | 冗余度 | EM |
|------|----------|----|---------|----|
| **QORE + Answer Scorer** | **~43-45%** | **~58-60%** | **~0.79** | **~33-35%** |
| MMR | 36.24% | 43.95% | 0.834 | 27.12% |
| Top-K | 36.92% | 44.01% | 0.841 | 26.93% |

**vs Baseline (QORE 无 Answer Scorer, 3-seed)**:
- Recall@5: 33.76% → ~44% (+10 个百分点)
- F1: 43.46% → ~59% (+15 个百分点)
- 冗余度: 0.810 → ~0.79 (-2%)
- EM: 26.76% → ~34% (+7 个百分点)

---

## ✅ 输出检查清单

实验完成后，检查 `results/rag/answer_scorer_full_3seeds/` 目录：

### 1. 文件完整性
```bash
ls -lh results/rag/answer_scorer_full_3seeds/
```

应该有：
- [ ] `qore_K5_seed42.json`
- [ ] `qore_K5_seed123.json`
- [ ] `qore_K5_seed456.json`
- [ ] `mmr_K5_seed42.json`
- [ ] `mmr_K5_seed123.json`
- [ ] `mmr_K5_seed456.json`
- [ ] `topk_K5_seed42.json`
- [ ] `topk_K5_seed123.json`
- [ ] `topk_K5_seed456.json`
- [ ] `summary.json`

### 2. 配置验证
```bash
python3 << 'EOF'
import json
with open('results/rag/answer_scorer_full_3seeds/qore_K5_seed42.json') as f:
    data = json.load(f)
config = data['config']
print(f"use_answer_scorer: {config.get('use_answer_scorer')}")
print(f"answer_scorer_backend: {config.get('answer_scorer_backend')}")
print(f"gamma: {config.get('gamma')}")
print(f"skip_generation: {config.get('skip_generation')}")
print(f"n_samples: {data['metrics']['n_samples']}")
EOF
```

应该输出：
```
use_answer_scorer: True
answer_scorer_backend: dpr
gamma: 0.5
skip_generation: False
n_samples: 3610
```

### 3. 结果合理性
```bash
python3 << 'EOF'
import json
with open('results/rag/answer_scorer_full_3seeds/summary.json') as f:
    data = json.load(f)
qm = data['methods']['qore']
print(f"QORE Recall@5: {qm['recall']['mean']:.4f} ± {qm['recall']['std']:.4f}")
print(f"QORE F1:       {qm['f1']['mean']:.4f} ± {qm['f1']['std']:.4f}")
print(f"QORE 冗余度:    {qm['redundancy']['mean']:.4f} ± {qm['redundancy']['std']:.4f}")
print(f"QORE EM:       {qm['em']['mean']:.4f} ± {qm['em']['std']:.4f}")
EOF
```

**通过标准**：
- [ ] Recall@5 ≥ 42%
- [ ] F1 ≥ 56%
- [ ] 冗余度 < 0.81
- [ ] EM ≥ 32%
- [ ] 标准差 < 0.01（3-seed 稳定性）

### 4. 统计显著性
```bash
grep -A 10 '"significance"' results/rag/answer_scorer_full_3seeds/summary.json
```

**关注**：
- [ ] `qore_vs_mmr_f1`: p < 0.05 (QORE F1 显著更高)
- [ ] `qore_vs_mmr_redundancy`: p < 0.05 (QORE 冗余度显著更低)

---

## 🎯 成功标准

实验成功需要满足：

**必要条件**（全部满足才算成功）：
1. ✅ 所有 JSON 文件生成（9 个方法 × seed + summary.json）
2. ✅ QORE Recall@5 ≥ 42%（vs baseline 33.76%）
3. ✅ QORE F1 ≥ 56%（vs baseline 43.46%）
4. ✅ QORE 冗余度 < 0.81（vs baseline 0.810）
5. ✅ F1 vs MMR 统计显著（p < 0.05）

**理想结果**（锦上添花）：
- QORE F1 > MMR F1（59% vs 44%，差距 >15%）
- QORE EM > MMR EM（34% vs 27%）
- 3-seed 标准差 < 0.01（结果稳定）

---

## 📦 交付

实验完成后，打包结果：

```bash
cd results/rag
zip -r answer_scorer_full_3seeds.zip answer_scorer_full_3seeds/
```

**发送**：
- `answer_scorer_full_3seeds.zip`
- 终端输出截图（显示完成时间和任何错误）

---

## 🔍 故障排查

### 问题 1：磁盘空间不足
```
OSError: Not enough disk space
```
**解决**：清理缓存或换磁盘更大的机器。
```bash
df -h /root/.cache/huggingface
du -sh /root/.cache/huggingface/*
```

### 问题 2：GPU OOM
```
CUDA out of memory
```
**解决**：DPR reader batch_size 太大。
编辑 `applications/rag/answer_scorer.py:23`，改为 `batch_size=8`。

### 问题 3：进程被杀
```
Killed
```
**解决**：内存不足。在 tmux 中运行，检查 `dmesg | tail`。

### 问题 4：卡在某题很久
**现象**：进度条长时间不动。
**检查**：
```bash
# 查看进程
ps aux | grep eval_suite

# 查看 GPU
nvidia-smi

# 查看日志
tail -f results/rag/answer_scorer_full_3seeds/*.log
```

---

## 📞 联系

如有问题，提供：
1. 完整错误信息（截图或复制）
2. `git log -1`（确认代码版本）
3. 运行的完整命令
4. 系统信息：`nvidia-smi`, `df -h`, `free -h`

---

## 时间线

- **启动时间**：记录开始时间
- **检查点 1**：30 分钟后，确认至少完成 1 个 seed 的 1 个方法
- **检查点 2**：1.5 小时后，确认完成 ~50%
- **预期完成**：2.8-4 小时

如果超过 6 小时仍未完成，检查是否卡住。

---

## ✅ 最终确认

在运行前，确认：
- [ ] 代码已 `git pull` 到最新（commit cd8745b+）
- [ ] 在 tmux 中运行（防止断线）
- [ ] 磁盘空间 > 50GB
- [ ] GPU 可用 (`nvidia-smi`)
- [ ] 网络稳定（访问 HuggingFace）

**准备好了就开始吧！** 🚀
