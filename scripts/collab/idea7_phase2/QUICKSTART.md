# Idea 7 Phase 2 快速开始

## 一行命令

```bash
cd ~/QORE-VLM && git pull && bash scripts/collab/idea7_phase2/run_idea7_phase2.sh
```

## 等待完成

- ⏱️ **GPU**: ~1 小时
- ⏱️ **CPU**: ~3 小时

## 查看结果

```bash
# 找到最新的结果目录
ls -lt exchange/idea7_phase2/ | head -5

# 查看报告（替换 <timestamp> 为实际时间戳）
cat exchange/idea7_phase2/<timestamp>/RESULTS.md
```

## 关键指标

报告会告诉你：

✅ **成功**: Recall 提升 ≥ 5%（继续 Phase 3）  
⚠️ **部分成功**: Recall 提升 < 5%（需调试）  
❌ **失败**: Recall 无改进（Pivot 到 Idea 2）

## 提交结果

如果实验成功：

```bash
cd ~/QORE-VLM
git add exchange/idea7_phase2/<timestamp>
git commit -m "results(idea7): Phase 2 training complete"
git push
```

然后告诉导师查看结果！🎉
