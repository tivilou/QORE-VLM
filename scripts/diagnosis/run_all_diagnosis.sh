#!/bin/bash
# Phase 1 诊断 —— 验证 4 个 idea 的前提假设
#
#   idea 1  两阶段 QUBO      gamma_sweep_diagnosis.py
#   idea 4  上下文完整性      context_dependency_diagnosis.py
#   idea 6  互补性矩阵        complementarity_diagnosis.py
#   idea 7  Soft QUBO 端到端  qubo_objective_diagnosis.py
#
# 另外两个 idea 在 NQ 上原理上测不了，脚本保留但不在此运行：
#   idea 2  答案多样性 —— NQ 的 gold_answers 是同一答案的别名集合
#                        （"December 1972" / "14 December 1972 UTC"），
#                        不是多个不同答案。需 AmbigQA / WebQSP。
#   idea 3  Query-adaptive γ —— 实测 3610 题，分类器 simple 桶只有 2 题
#                        (0.1%)，NQ 问句同质性太高。需 HotpotQA 混样。
#
# 前置：先跑完实验
#   bash scripts/collab/run_phase1_full.sh
#
# 数据要求：评测必须带 --dump_passages（配置里已开）。
#   缺字段时各脚本会明确报错并写明要加哪个参数，不会静默产出空报告。

set -uo pipefail
cd "$(dirname "$0")/../.."

RESULTS_DIR="scratch/research/P1_diagnosis/experiments"
ANALYSIS_DIR="scratch/research/P1_diagnosis/analysis"
MAIN_RESULT="$RESULTS_DIR/gamma_0.5/result.json"

mkdir -p "$ANALYSIS_DIR"

# 实验是否真的跑成功了 —— 光看 result.json 在不在不够。
# run_tuning_suite 超时会 kill 进程、失败返回非零，两种情况下评测都没重写
# result.json。早期的 runner 不清旧产物，所以一份陈文件能让四个诊断照常出
# 报告，数字看着还是对的 —— 那比读不到更危险。这里拿 status.json 卡住。
check_status() {
    local exp_dir="$1" status_file="$1/status.json"
    [ -f "$status_file" ] || return 0   # 没有 status 就不下结论（可能是手跑的）
    local st
    st=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('status',''))" \
         "$status_file" 2>/dev/null) || return 0
    if [ -n "$st" ] && [ "$st" != "success" ]; then
        echo "   ⚠️  $(basename "$exp_dir") 的 status = $st"
        return 1
    fi
    return 0
}

BAD_STATUS=0
for d in "$RESULTS_DIR"/gamma_*; do
    [ -d "$d" ] || continue
    check_status "$d" || BAD_STATUS=1
done
if [ "$BAD_STATUS" -eq 1 ]; then
    echo ""
    echo "❌ 有实验没跑成功（超时/失败）。这些目录里的 result.json 可能是上一轮的陈数据，"
    echo "   拿它跑诊断会得到看着合理但完全无效的结论。"
    echo ""
    echo "   先看: cat $RESULTS_DIR/gamma_0.5/stderr.log"
    echo "   确认后重跑: bash scripts/collab/run_phase1_full.sh"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Phase 1 诊断 —— 4 个 idea 的前提验证                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "实验结果: $RESULTS_DIR"
echo "分析输出: $ANALYSIS_DIR"
echo ""

FAILED=()
SKIPPED=()
SATURATED=0

# 目录存在但没有 result.json 也算缺输入 —— run_tuning_suite 一启动就会
# 建 gamma_*/ 目录写日志，跑挂了目录也在。
HAVE_SWEEP=0
if [ -d "$RESULTS_DIR" ] && \
   [ -n "$(find "$RESULTS_DIR" -mindepth 2 -name result.json -print -quit 2>/dev/null)" ]; then
    HAVE_SWEEP=1
fi

# ── idea 1: γ sweep ──────────────────────────────────────────────────
# 退出码 2 = γ 饱和（三个 γ 选出几乎同一批段落），不是失败，
# 但意味着后面所有关于「多样性有无用」的读数都不可解读。
echo "────────────────────────────────────────────────────────────"
echo "[1/4] idea 1 两阶段 QUBO —— γ sweep"
if [ "$HAVE_SWEEP" -eq 1 ]; then
    python scripts/diagnosis/gamma_sweep_diagnosis.py \
        --results_dir "$RESULTS_DIR" \
        --output "$ANALYSIS_DIR/gamma_sweep.md"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "   ✅ 完成"
    elif [ $rc -eq 2 ]; then
        echo "   ⚠️  γ 未生效（饱和）—— 报告已生成但结论不可用"
        SATURATED=1
    else
        echo "   ❌ 失败"
        FAILED+=("[1/4] γ sweep")
    fi
else
    echo "   ⏭️  跳过：$RESULTS_DIR 下没有 gamma_*/result.json"
    SKIPPED+=("[1/4] γ sweep")
fi
echo ""

# ── 其余三个都读 gamma_0.5 的单份结果 ────────────────────────────────
run_single() {
    local label="$1" script="$2" outfile="$3"; shift 3
    echo "────────────────────────────────────────────────────────────"
    echo "$label"
    if [ ! -f "$MAIN_RESULT" ]; then
        echo "   ⏭️  跳过：$MAIN_RESULT 不存在"
        SKIPPED+=("$label")
        echo ""
        return
    fi
    if python "$script" --results "$MAIN_RESULT" \
            --output "$ANALYSIS_DIR/$outfile" "$@"; then
        echo "   ✅ 完成"
    else
        echo "   ❌ 失败（见上方报错）"
        FAILED+=("$label")
    fi
    echo ""
}

run_single "[2/4] idea 4 上下文完整性" \
    scripts/diagnosis/context_dependency_diagnosis.py \
    context_dependency.md

run_single "[3/4] idea 6 互补性矩阵" \
    scripts/diagnosis/complementarity_diagnosis.py \
    complementarity.md

run_single "[4/4] idea 7 Soft QUBO 端到端" \
    scripts/diagnosis/qubo_objective_diagnosis.py \
    qubo_objective.md --gamma 0.5 --lam 2.0

# ── 汇总 ─────────────────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                      诊断汇总                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 报告目录: $ANALYSIS_DIR/"
ls -1 "$ANALYSIS_DIR" 2>/dev/null | sed 's/^/   /' || echo "   (空)"
echo ""

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "⏭️  跳过 ${#SKIPPED[@]} 个（缺输入）:"
    printf '   %s\n' "${SKIPPED[@]}"
    echo ""
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "❌ 失败 ${#FAILED[@]} 个:"
    printf '   %s\n' "${FAILED[@]}"
    echo ""
    echo "常见原因：评测未加 --dump_passages。报错信息里会写明缺哪个字段。"
    exit 1
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "先跑 bash scripts/collab/run_phase1_full.sh 再回来。"
    exit 1
fi

if [ "$SATURATED" -eq 1 ]; then
    echo "⚠️  γ 饱和：idea 1 的结论不可用，需先重标定 QUBO 系数。"
    echo "    其余三个诊断的结论不受影响。"
    echo ""
fi

echo "✅ 4 个诊断全部完成"
echo ""
echo "下一步：逐份看报告，按假设强度决定先做哪个 idea。"
echo ""
echo "交结果（自动建趟次目录、拷文件、生成 README）："
echo "  python scripts/collab/collect_p1_results.py"
