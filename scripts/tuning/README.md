# 实验自动化框架使用指南

**创建日期**: 2026-07-24  
**版本**: 1.0  
**目的**: 自动运行、监控、分析和打包调参实验

---

## 快速开始

### Phase 1 诊断实验（推荐第一次使用）

```bash
cd /home/Q-DUET-VLM/QORE-VLM

# 运行 Phase 1 诊断（γ sweep, 3 个实验）
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml
```

**预期耗时**: 1 小时  
**输出位置**: `scratch/research/P1_diagnosis/`

---

## 框架功能

### ✅ 自动化功能

1. **环境检查**
   - GPU 可用性
   - 磁盘空间
   - Python 包依赖

2. **实验运行**
   - 串行/并行执行
   - 进度显示
   - 失败重试
   - 超时处理

3. **结果收集**
   - 自动提取指标
   - 状态跟踪
   - 日志记录

4. **结果分析**
   - 自动对比
   - 生成报告
   - 运行自定义分析脚本

5. **结果打包**
   - 自动打包成 zip
   - 包含所有结果和报告

---

## 配置文件格式

配置文件使用 YAML 格式，位于 `scripts/tuning/config/`

### 基本结构

```yaml
name: "实验名称"
description: "实验描述"
output_dir: "输出目录"

experiments:
  - name: "实验1"
    description: "实验1描述"
    command: "python -m scripts.rag.eval_rag_refactored"
    args:
      corpus_mode: "wiki_dpr"
      max_samples: 200
      gamma: 0.0
      # ... 其他参数
    expected_time_minutes: 20

  - name: "实验2"
    # ...

post_process:
  analyze:
    enabled: true
    script: "分析脚本路径"
  
  package:
    enabled: true
    output: "打包文件路径"

execution:
  parallel: 1              # 并行数（1=串行）
  retry_on_failure: 2      # 失败重试次数
  timeout_minutes: 30      # 超时时间
```

---

## 高级用法

### 1. 覆盖配置参数

```bash
# 快速测试：只跑 50 题
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml \
    --override max_samples=50

# 多个覆盖
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/phase1_diagnosis.yaml \
    --override max_samples=50 seed=123
```

### 2. 并行运行（如果有多个 GPU）

修改配置文件中的 `execution.parallel`:
```yaml
execution:
  parallel: 3  # 同时运行 3 个实验
```

### 3. 创建自定义实验配置

复制 `phase1_diagnosis.yaml` 并修改：

```bash
cp scripts/tuning/config/phase1_diagnosis.yaml \
   scripts/tuning/config/my_experiment.yaml

# 编辑配置
vim scripts/tuning/config/my_experiment.yaml

# 运行
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/my_experiment.yaml
```

---

## 输出结构

运行完成后，输出目录结构如下：

```
scratch/research/P1_diagnosis/
├── experiments/              # 各实验结果
│   ├── gamma_0.0/
│   │   ├── result.json      # 实验结果
│   │   ├── stdout.log       # 标准输出
│   │   ├── stderr.log       # 错误输出
│   │   └── status.json      # 状态信息
│   ├── gamma_0.5/
│   └── gamma_1.0/
│
├── analysis/                 # 分析结果
│   ├── analysis.md          # 详细分析报告
│   └── quick_summary.md     # 快速摘要
│
├── package/                  # 打包文件
│   └── P1_diagnosis_20260724_143052.zip
│
├── run_summary.json         # 运行摘要
└── run.log                  # 完整日志
```

---

## 查看结果

### 1. 查看运行摘要

```bash
cat scratch/research/P1_diagnosis/run_summary.json
```

### 2. 查看分析报告

```bash
cat scratch/research/P1_diagnosis/analysis/gamma_sweep.md
```

### 3. 查看单个实验结果

```bash
cat scratch/research/P1_diagnosis/experiments/gamma_0.0/result.json
```

### 4. 查看日志

```bash
# 标准输出
cat scratch/research/P1_diagnosis/experiments/gamma_0.0/stdout.log

# 错误输出
cat scratch/research/P1_diagnosis/experiments/gamma_0.0/stderr.log
```

---

## 故障排查

### 问题 1: 环境检查失败

```
❌ GPU 不可用
```

**解决**: 
- 检查 GPU: `nvidia-smi`
- 如果不需要 GPU，修改配置文件中的 `pre_checks`

---

### 问题 2: 实验失败

```
❌ 失败: ModuleNotFoundError
```

**解决**:
- 查看 `stderr.log` 获取详细错误
- 检查 Python 包是否安装
- 检查命令是否正确

---

### 问题 3: 超时

```
❌ 失败: Timeout after 30 minutes
```

**解决**:
- 增加 `timeout_minutes`
- 或减少 `max_samples`

---

### 问题 4: 分析脚本失败

```
⚠️  分析脚本不存在
```

**解决**:
- 检查分析脚本路径是否正确
- 确保分析脚本可执行: `chmod +x script.py`

---

## 框架文件说明

```
scripts/tuning/
├── run_tuning_suite.py          # 主控脚本 ⭐
├── config/                       # 配置文件
│   └── phase1_diagnosis.yaml    # Phase 1 配置
├── utils/                        # 工具模块
│   ├── experiment_runner.py     # 实验运行器
│   ├── result_analyzer.py       # 结果分析器
│   └── packager.py              # 结果打包器
└── sweeps/                       # 预定义实验脚本（可选）
```

---

## 扩展框架

### 添加新的实验类型

1. 创建配置文件:
```bash
vim scripts/tuning/config/new_experiment.yaml
```

2. 定义实验:
```yaml
name: "New Experiment"
experiments:
  - name: "exp1"
    command: "python -m your.script"
    args:
      param1: value1
```

3. 运行:
```bash
python scripts/tuning/run_tuning_suite.py \
    --config scripts/tuning/config/new_experiment.yaml
```

---

### 添加自定义分析脚本

1. 创建分析脚本:
```python
# scripts/tuning/analysis/custom_analyzer.py
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    # 分析逻辑
    # ...
    
    # 输出报告
    with open(args.output, 'w') as f:
        f.write("# Custom Analysis\n")

if __name__ == '__main__':
    main()
```

2. 在配置中引用:
```yaml
post_process:
  analyze:
    enabled: true
    script: "scripts/tuning/analysis/custom_analyzer.py"
    args:
      results_dir: "{output_dir}/experiments"
      output: "{output_dir}/analysis/custom.md"
```

---

## 常见实验配置示例

### 示例 1: γ Sweep

```yaml
name: "Gamma Sweep"
experiments:
  - name: "gamma_0.0"
    args:
      gamma: 0.0
  - name: "gamma_0.3"
    args:
      gamma: 0.3
  - name: "gamma_0.5"
    args:
      gamma: 0.5
  - name: "gamma_0.7"
    args:
      gamma: 0.7
  - name: "gamma_1.0"
    args:
      gamma: 1.0
```

### 示例 2: Prefilter Size Sweep

```yaml
name: "Prefilter Size Sweep"
experiments:
  - name: "size_10"
    args:
      qore_prefilter_size: 10
  - name: "size_15"
    args:
      qore_prefilter_size: 15
  - name: "size_20"
    args:
      qore_prefilter_size: 20
  - name: "size_25"
    args:
      qore_prefilter_size: 25
```

### 示例 3: 多维度网格搜索

```yaml
name: "Grid Search"
experiments:
  - name: "gamma0.3_lam2.0"
    args:
      gamma: 0.3
      lam: 2.0
  - name: "gamma0.3_lam2.5"
    args:
      gamma: 0.3
      lam: 2.5
  - name: "gamma0.5_lam2.0"
    args:
      gamma: 0.5
      lam: 2.0
  - name: "gamma0.5_lam2.5"
    args:
      gamma: 0.5
      lam: 2.5
```

---

## 性能优化建议

### 1. 使用并行运行（如果有多 GPU）

```yaml
execution:
  parallel: 3  # GPU 数量
```

### 2. 先小样本快速验证

```bash
# 先跑 50 题
python scripts/tuning/run_tuning_suite.py \
    --config config.yaml --override max_samples=50

# 确认没问题后跑全量
python scripts/tuning/run_tuning_suite.py \
    --config config.yaml
```

### 3. 使用 skip_generation

对于调参实验（不需要生成答案），使用 `skip_generation=true` 节省时间。

---

## 最佳实践

1. **先测试**: 用小样本快速测试配置是否正确
2. **命名规范**: 实验名称清晰描述参数（如 `gamma_0.5_lam_2.0`）
3. **保存配置**: 每次实验保存配置文件，便于复现
4. **检查结果**: 运行完立即查看 `run_summary.json`
5. **备份重要结果**: 将打包的 zip 文件备份到安全位置

---

## 技术支持

如果遇到问题：
1. 查看 `stderr.log` 获取详细错误
2. 检查 `status.json` 了解实验状态
3. 参考本文档的"故障排查"部分

---

**文档版本**: 1.0  
**最后更新**: 2026-07-24
