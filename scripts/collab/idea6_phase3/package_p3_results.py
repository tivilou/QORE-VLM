"""Package Phase 3 results for GitHub submission.

This script:
1. Compresses result.json files (they're large, ~300MB each)
2. Generates a summary README
3. Creates a .zip archive for easy sharing
4. Keeps logs and small metadata files uncompressed

Usage:
    python scripts/collab/idea6_phase3/package_p3_results.py \
        exchange/p3_solver_idea6/YYYYMMDDTHHMMSS
"""

import argparse
import json
import zipfile
from pathlib import Path
import shutil


def find_result_file(config_dir: Path) -> Path | None:
    for name in ("result.json", "qore_K5_seed*.json"):
        matches = sorted(config_dir.glob(name))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Package Phase 3 results")
    parser.add_argument("result_dir", help="Path to Phase 3 result directory")
    parser.add_argument("--output", help="Output zip filename (default: results.zip)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: {result_dir} does not exist")
        return

    output_zip = args.output or (result_dir / "results.zip")

    print("=" * 70)
    print("Packaging Phase 3 Results")
    print("=" * 70)
    print()
    print(f"Input:  {result_dir}")
    print(f"Output: {output_zip}")
    print()

    # Create README summary
    print("Generating summary README...")
    readme_path = result_dir / "README.md"

    # Collect metrics
    configs = ["baseline", "idea6_recommended", "idea6_best"]
    seeds = [42, 43, 44]

    with open(readme_path, "w") as f:
        f.write("# Idea 6 Phase 3 Results\n\n")
        f.write(f"**Timestamp**: {result_dir.name}\n\n")
        f.write("---\n\n")
        f.write("## Quick Summary\n\n")

        # Extract metrics
        for config in configs:
            f.write(f"### {config}\n\n")
            f.write("| Seed | Recall@5 | F1 | EM | Redundancy |\n")
            f.write("|------|----------|----|----|------------|\n")

            for seed in seeds:
                result_file = find_result_file(result_dir / f"seed_{seed}" / config)
                if result_file:
                    with open(result_file) as rf:
                        data = json.load(rf)
                        metrics = data.get("metrics", {})
                        recall = metrics.get("recall_at_5", metrics.get("mean_recall"))
                        f1 = metrics.get("f1", metrics.get("mean_f1"))
                        em = metrics.get("exact_match", metrics.get("mean_em"))
                        red = metrics.get("redundancy", metrics.get("mean_redundancy"))
                        fmt = lambda value: f"{value:.4f}" if value is not None else "N/A"
                        f.write(f"| {seed} | {fmt(recall)} | {fmt(f1)} | {fmt(em)} | {fmt(red)} |\n")

            f.write("\n")

        f.write("---\n\n")
        f.write("## Files\n\n")
        f.write("- `results.zip`: Compressed evaluator result JSON files\n")
        f.write("- `seed_XX/CONFIG/log.txt`: Execution logs (uncompressed)\n")
        f.write("- `git_*.txt`: Git status information\n")
        f.write("\n")
        f.write("For detailed analysis, run:\n")
        f.write("```bash\n")
        f.write(f"python scripts/collab/idea6_phase3/analyze_p3_results.py {result_dir}\n")
        f.write("```\n")

    print(f"  ✓ Created {readme_path}")
    print()

    # Create zip archive
    print("Creating zip archive...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add result.json files
        for result_file in result_dir.rglob("*.json"):
            if result_file.name in {"git_commit.txt", "git_status.txt"}:
                continue
            arcname = result_file.relative_to(result_dir)
            zf.write(result_file, arcname)
            print(f"  + {arcname}")

    print()
    print(f"  ✓ Created {output_zip}")

    # Get zip size
    zip_size_mb = Path(output_zip).stat().st_size / (1024 * 1024)
    print(f"  Size: {zip_size_mb:.1f} MB")
    print()

    print("=" * 70)
    print("✅ Packaging complete!")
    print("=" * 70)
    print()
    print("Files for GitHub submission:")
    print(f"  - {readme_path}")
    print(f"  - {output_zip}")
    print(f"  - git_commit.txt, git_status.txt")
    print(f"  - seed_XX/CONFIG/log.txt (optional)")
    print()


if __name__ == "__main__":
    main()
