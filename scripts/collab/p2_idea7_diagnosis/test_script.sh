#!/bin/bash
# 测试 run_idea7_diagnosis.sh 的路径和参数配置

set -e

echo "=== 测试 Idea 7 诊断脚本 ==="
echo ""

# 1. 测试路径解析
echo "1. 测试路径解析..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
echo "  ✓ SCRIPT_DIR: $SCRIPT_DIR"
echo "  ✓ REPO_ROOT: $REPO_ROOT"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
echo "  ✓ 已切换到 REPO_ROOT 并设置 PYTHONPATH"
echo ""

# 2. 测试输出目录创建
echo "2. 测试输出目录创建..."
TIMESTAMP="test_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$REPO_ROOT/exchange/p2_idea7_diagnosis/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "  ✓ 输出目录创建成功: ${OUTPUT_DIR}"
echo ""

# 3. 测试 eval_rag_refactored.py 参数
echo "3. 测试 eval_rag_refactored.py 参数..."
if python -m scripts.rag.eval_rag_refactored --help > /dev/null 2>&1; then
    echo "  ✓ eval_rag_refactored.py 可以导入"
else
    echo "  ✗ eval_rag_refactored.py 导入失败"
    exit 1
fi

# 检查关键参数
for param in "--K" "--lam" "--gamma" "--delta" "--dump_passages" "--complementarity_method" "--use_answer_scorer"; do
    if python -m scripts.rag.eval_rag_refactored --help 2>&1 | grep -q "$param"; then
        echo "  ✓ 参数 $param 存在"
    else
        echo "  ✗ 参数 $param 不存在"
        exit 1
    fi
done
echo ""

# 4. 测试诊断脚本存在性
echo "4. 测试诊断脚本..."
if [ -f "$REPO_ROOT/scripts/diagnosis/qubo_objective_diagnosis.py" ]; then
    echo "  ✓ 诊断脚本存在: scripts/diagnosis/qubo_objective_diagnosis.py"
else
    echo "  ✗ 诊断脚本不存在"
    exit 1
fi

# 检查诊断脚本参数
if python "$REPO_ROOT/scripts/diagnosis/qubo_objective_diagnosis.py" --help > /dev/null 2>&1; then
    echo "  ✓ 诊断脚本可以运行"
else
    echo "  ✗ 诊断脚本运行失败"
    exit 1
fi
echo ""

# 5. 测试 README 生成
echo "5. 测试 README 生成..."
cat > "${OUTPUT_DIR}/README.md" <<EOF
# 测试 README
时间戳: ${TIMESTAMP}
EOF
if [ -f "${OUTPUT_DIR}/README.md" ]; then
    echo "  ✓ README 生成成功"
else
    echo "  ✗ README 生成失败"
    exit 1
fi
echo ""

# 6. 清理测试目录
echo "6. 清理测试目录..."
rm -rf "${OUTPUT_DIR}"
echo "  ✓ 测试目录已清理"
echo ""

echo "=========================================="
echo "✓ 所有测试通过！脚本应该可以正常运行"
echo "=========================================="
