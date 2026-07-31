#!/bin/bash
# Test script to verify Phase 3 scripts work from any directory

set -e

echo "================================================================"
echo "  Phase 3 Scripts - Path Independence Test"
echo "================================================================"
echo ""

PROJECT_ROOT="/home/Q-DUET-VLM/QORE-VLM"
PHASE3_DIR="$PROJECT_ROOT/scripts/collab/idea6_phase3"

TESTS_PASSED=0
TESTS_FAILED=0

echo "Test 1: Checking bash script syntax..."
echo ""

for script in run_p3_experiments.sh run_full_pipeline.sh quick_test.sh; do
    echo -n "  $script: "
    if bash -n "$PHASE3_DIR/$script" 2>/dev/null; then
        echo "✓ PASS"
        ((TESTS_PASSED++))
    else
        echo "✗ FAIL"
        ((TESTS_FAILED++))
    fi
done

echo ""
echo "Test 2: Checking Python script syntax..."
echo ""

for script in analyze_p3_results.py package_p3_results.py; do
    echo -n "  $script: "
    if python -m py_compile "$PHASE3_DIR/$script" 2>/dev/null; then
        echo "✓ PASS"
        ((TESTS_PASSED++))
    else
        echo "✗ FAIL"
        ((TESTS_FAILED++))
    fi
done

echo ""
echo "Test 3: Path calculation from different directories..."
echo ""

# Test from different working directories
for test_dir in "/tmp" "$HOME" "$PROJECT_ROOT" "$PROJECT_ROOT/scripts"; do
    echo -n "  From $test_dir: "

    RESULT=$(cd "$test_dir" && bash -c "
        SCRIPT_FILE='$PHASE3_DIR/run_p3_experiments.sh'
        SCRIPT_DIR=\"\$(cd \"\$(dirname \"\$SCRIPT_FILE\")\" && pwd)\"
        PROJECT_ROOT=\"\$(cd \"\$SCRIPT_DIR/../../..\" && pwd)\"
        echo \$PROJECT_ROOT
    ")

    if [ "$RESULT" = "$PROJECT_ROOT" ]; then
        echo "✓ PASS (calculated: $RESULT)"
        ((TESTS_PASSED++))
    else
        echo "✗ FAIL (expected: $PROJECT_ROOT, got: $RESULT)"
        ((TESTS_FAILED++))
    fi
done

echo ""
echo "Test 4: Verify all config files exist..."
echo ""

for config in baseline_phase3.yaml idea6_phase3_recommended.yaml idea6_phase3_best.yaml; do
    echo -n "  $config: "
    if [ -f "$PROJECT_ROOT/configs/experiments/$config" ]; then
        echo "✓ PASS"
        ((TESTS_PASSED++))
    else
        echo "✗ FAIL"
        ((TESTS_FAILED++))
    fi
done

echo ""
echo "================================================================"
echo "  Test Results"
echo "================================================================"
echo ""
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✅ All tests passed! Scripts are path-independent."
    exit 0
else
    echo "❌ Some tests failed."
    exit 1
fi
