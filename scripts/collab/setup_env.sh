#!/bin/bash
# 环境配置检查脚本

set -e

echo "=== 环境配置检查 ==="
echo ""

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装"
    exit 1
fi
echo "✅ Python: $(python --version)"

# 检查 GPU
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU 可用:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠️  GPU 不可用（nvidia-smi 未找到）"
fi

# 检查磁盘空间
echo ""
df -h . | awk 'NR==1 || /\/$/ {print}'

# 检查 Python 包
echo ""
echo "检查 Python 依赖..."
python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch 未安装"
python -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>/dev/null || echo "❌ transformers 未安装"
python -c "import numpy; print('✅ numpy')" 2>/dev/null || echo "❌ numpy 未安装"

echo ""
echo "✅ 环境检查完成"
