# KV Cache Evaluation & Optimization

## Goal

修复并优化 QORE 的 KV cache 评测代码,完成与 baseline(H2O/SnapKV/PyramidKV)的公平对比,验证 QORE 在长上下文场景下的质量优势。论文级实验要求统计严格、测量公平、延迟可接受。

## Current State

**阶段**: 核心修复完成,首次完整评测完成(1150样本),待延迟优化

**完成的工作**(11个提交,~2300行新代码):
1. 师弟反馈的7个问题全修复:RoPE重旋转、hook映射、backend统一、官方baseline移植、容量口径、显存时点、随机抽样
2. B方案官方范式(809行):H2O/SnapKV/PyramidKV逐行移植+QORE官方范式+双范式支持
3. 统计设计(479行):随机抽样+bootstrap CI+文档
4. 进度显示:tqdm+智能fallback

**师弟最新实验**(2026-07-16,修复后代码):
- 1150样本,输入6698 tokens,容量1024,Llama-3-8B
- ✅ QORE F1=0.1949(压缩策略第1,比snapkv高38%,比H2O高73%)
- ✅ 保留full的53.1%(激进压缩下质量优势明显)
- ❌ 延迟18023ms(是full的6.2×,最慢)

**对比旧报告(30样本/4K)**:相对优势从4.5%→38%(更长输入→更高压缩→QORE优势体现)

## In Progress

无(当前暂停,转RAG模块)

## Next Actions

### P0: 延迟优化(论文必需,2-3小时)
1. 降`num_reads`(30→10):预期-30%延迟
2. 淘汰后关attention capture:预期-20%延迟
3. 目标:6.2×→3-4×
4. 重跑1150样本验证

### P1: 多seed+CI(1-2天)
1. 3-5个seed各跑1150样本
2. bootstrap_ci汇总CI
3. 证明统计显著性

### P2: 可选
- 范式对比(decode_evict vs prefill_compress)
- 分阶段计时

## Blockers

无。延迟优化需权衡质量风险(需实验验证)。

## Validation

- 72测试全过
- 1150样本完整评测完成
- F1结果论文可用(质量优势明确)
- 延迟结果不可用(6.2×太高,需优化)

## Relevant Decisions

无需decision文件(所有修复都是bug修复,不涉及设计选择)。
