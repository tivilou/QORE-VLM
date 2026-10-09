"""Read-only source compilation. No network, model loading or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from applications.rag.support_qualified_boundary_targets import compile_targets, pending_overlay


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("source hash mismatch: " + path.name)
    return load(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--review-manifest", type=Path, required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    manifest = load(args.review_manifest)
    root = args.review_manifest.parent
    def review_part(key):
        name = manifest[key + "_file"]
        if Path(name).name != name:
            raise ValueError("review path must be a sibling file")
        return checked(root / name, manifest[key + "_sha256"])
    prior, delta = review_part("prior"), review_part("delta")
    trace = checked(args.source_dir / "case_study.json", manifest["source_case_sha256"])
    blind = checked(args.source_dir / "support_review.json", manifest["source_support_review_sha256"])
    if any(p["support_label"] is not None for c in blind["cases"] for p in c["passages"]):
        raise ValueError("producer-null labels changed")
    review = {"judgments": prior["judgments"] + delta["judgments"],
              "question_target_flags": prior["question_target_flags"]}
    if len(review["judgments"]) != manifest["merged_count"]:
        raise ValueError("review coverage mismatch")
    overlay = load(args.qualification) if args.qualification else pending_overlay(trace, review)
    if args.qualification:
        witness_root = args.qualification.resolve().parent
        for row in overlay["questions"] + overlay["items"]:
            if row.get("status") != "qualified":
                continue
            for witness in row.get("evidence", []):
                relative = Path(witness["artifact_ref"])
                witness_path = (witness_root / relative).resolve()
                if relative.is_absolute() or not witness_path.is_relative_to(witness_root):
                    raise ValueError("witness must be a file within the qualification packet")
                if hashlib.sha256(witness_path.read_bytes()).hexdigest() != witness["sha256"]:
                    raise ValueError("review witness hash mismatch")
    targets = compile_targets(trace, review, overlay)
    # Exclusive creation avoids overwriting any immutable review or earlier run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("qualification_overlay.json", overlay), ("boundary_target_draft.json", targets)):
        (args.output_dir / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2,
                                                       allow_nan=False) + "\n").encode("utf-8"))
        if load(args.output_dir / name) != value:
            raise ValueError("output round-trip failed")
    print(json.dumps(targets["summary"], sort_keys=True))
    print("training_consumable=false; no models, fitting or source changes")


if __name__ == "__main__":
    main()
