#!/usr/bin/env python3
"""
Test script to verify speed_mode functionality.
Checks that speed_mode correctly overrides num_reads.
"""

import sys
import subprocess


def test_speed_mode_parsing():
    """Test that speed_mode correctly sets num_reads."""

    tests = [
        # (args, expected_num_reads, description)
        (["--speed_mode", "quality"], 100, "Quality mode should set num_reads=100"),
        (["--speed_mode", "balanced"], 50, "Balanced mode should set num_reads=50"),
        (["--speed_mode", "fast"], 30, "Fast mode should set num_reads=30"),
        (["--num_reads", "75"], 75, "Direct num_reads should work"),
        (["--speed_mode", "balanced", "--num_reads", "75"], 50, "Speed_mode should override num_reads"),
    ]

    print("Testing speed_mode functionality...")
    print("=" * 60)

    all_passed = True

    for args, expected, description in tests:
        # Construct command to test argument parsing
        cmd = [
            sys.executable, "-c",
            """
import sys
sys.path.insert(0, '.')
from scripts.kv_cache.eval_kv_cache import parse_args

# Mock sys.argv
sys.argv = ['test', '--model_path', 'dummy'] + sys.argv[1:]
args = parse_args()
print(args.num_reads)
"""
        ] + args

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            actual = int(result.stdout.strip().split('\n')[-1])

            if actual == expected:
                print(f"✅ PASS: {description}")
                print(f"   Args: {' '.join(args)}")
                print(f"   Result: num_reads={actual}")
            else:
                print(f"❌ FAIL: {description}")
                print(f"   Args: {' '.join(args)}")
                print(f"   Expected: {expected}, Got: {actual}")
                all_passed = False

        except Exception as e:
            print(f"❌ ERROR: {description}")
            print(f"   Args: {' '.join(args)}")
            print(f"   Error: {e}")
            all_passed = False

        print()

    print("=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(test_speed_mode_parsing())
