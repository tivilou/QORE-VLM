"""Summarize experiment results into a CSV comparison table."""

import argparse
import json
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    results = []

    for json_file in sorted(input_dir.glob("*.json")):
        # Skip per-sample detail files (e.g. qore.samples.json) — those are the
        # eval's raw prediction dumps, not method result summaries.
        if json_file.name.endswith(".samples.json"):
            continue
        with open(json_file) as f:
            data = json.load(f)
        # A method result is a dict with metrics/status; anything else (e.g. a
        # stray list) isn't a comparable method row.
        if not isinstance(data, dict) or not ("metrics" in data or "status" in data):
            continue
        row = {"method": json_file.stem}
        if "metrics" in data:
            # Flatten scalar metrics; skip nested structures (e.g. f1_per_task)
            # so the CSV stays one-row-per-method. Per-task detail lives in JSON.
            for k, v in data["metrics"].items():
                if isinstance(v, (dict, list)):
                    continue
                row[k] = v
            # Surface real (post-eviction) cache footprint from measured stats.
            measured = data.get("measured", {})
            if measured.get("avg_cache_MB") is not None:
                row["cache_MB"] = measured["avg_cache_MB"]
            if measured.get("avg_final_cache_len") is not None:
                row["final_cache_len"] = measured["avg_final_cache_len"]
        elif "status" in data:
            row["status"] = data["status"]
        results.append(row)

    if not results:
        print(f"No JSON results found in {input_dir}")
        return

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())
    for r in results[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Summary written to {output_path}")
    print(f"  {len(results)} methods compared")


if __name__ == "__main__":
    main()
