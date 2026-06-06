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
        with open(json_file) as f:
            data = json.load(f)
        row = {"method": json_file.stem}
        if "metrics" in data:
            row.update(data["metrics"])
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
