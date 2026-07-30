# Idea 7 重新评估 - 快速开始

## 一行命令运行

```bash
cd ~/QORE-VLM && git pull && bash scripts/collab/p2_idea7_diagnosis/run_idea7_diagnosis.sh
```

**预计时间**：40-70 分钟（200 样本 × DPR 打分 + 生成 + 诊断）

## 运行环境要求

- ✅ GPU（需要 DPR answer scorer）
- ✅ 21M Wikipedia corpus（wiki_dpr 模式）
- ✅ 预训练模型文件（DPR）

## 运行后产物

```
exchange/p2_idea7_diagnosis/<timestamp>/
├── README.md              # 实验说明和对比框架
├── result.json            # RAG 评估完整结果（~4MB）
└── qubo_diagnosis.md      # 诊断报告（核心结果）⭐
```

## 查看结果

**关键指标**（在 `qubo_diagnosis.md` 中）：

```bash
# 快速查看结果摘要
cat exchange/p2_idea7_diagnosis/<timestamp>/qubo_diagnosis.md | grep -A5 "同题内比较"
```

重点看：
- **QUBO 能量最低的子集**: 平均 Recall X.XXXX
- **枚举出的最优子集**: 平均 Recall X.XXXX
- **平均差距** (Gap): X.XXXX
- **QUBO 命中最优的题数**: XX/200 (XX.X%)

## 对比 Phase 1

| 指标 | Phase 1 (无 Idea 6) | Phase 2 (含 Idea 6) |
|------|---------------------|---------------------|
| Gap | 0.3004 | **待测** |
| 命中率 | 34.7% | **待测** |

**决策标准**：
- Gap > 0.15 → ✅ 强烈建议实现 Idea 7
- Gap 0.08-0.15 → ⚠️ 考虑实现
- Gap < 0.08 → ❌ 暂不实现（Idea 6 已充分优化）

## 提交结果

```bash
cd ~/QORE-VLM
git add exchange/p2_idea7_diagnosis/<timestamp>/
git commit -m "experiment(idea7): re-evaluation after idea6 implementation"
git push
```

## 故障排查

### GPU 内存不足
```bash
# 编辑脚本，减少样本数
nano scripts/collab/p2_idea7_diagnosis/run_idea7_diagnosis.sh
# 修改: SAMPLES=100  # 原来是 200
```

### 找不到模型
```bash
ls ~/models/dpr/  # 检查 DPR 模型是否存在
```

### 其他问题
详见 `README.md` 故障排查章节

---

**有问题随时联系！**
