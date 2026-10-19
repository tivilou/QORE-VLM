#!/usr/bin/env python3
"""Verify and freeze all label-blind inputs for the 256-case Reader score run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from applications.rag.reader_larger_untouched_manifest import (
    build_scoring_manifest,
    file_sha256,
    validate_scoring_manifest,
)
from scripts.collab.lib.source_snapshot import archive_sources, verify_sources


SPEC = ROOT / "configs/experiments/reader_larger_untouched_scoring_spec.json"
SOURCES = (
    "applications/rag/quantum_reader_rotation.py",
    "applications/rag/quantum_semantic_reader.py",
    "applications/rag/reader_larger_untouched.py",
    "applications/rag/reader_larger_untouched_manifest.py",
    "applications/rag/reader_unseen_transport.py",
    "configs/experiments/reader_larger_untouched_scoring_spec.json",
    "configs/experiments/reader_larger_untouched_utility_256.json",
    "configs/experiments/reader_larger_untouched_utility_256_plan.json",
    "scripts/collab/five_ideas/freeze_reader_larger_untouched_scoring_manifest.py",
    "scripts/collab/lib/source_snapshot.py",
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def run(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    snapshot = archive_sources(ROOT, SOURCES, args.output / "scoring_manifest_sources.zip")
    manifest = build_scoring_manifest(
        spec,
        root=ROOT,
        cohort_dir=args.cohort_dir,
        sequential_values=args.sequential_values,
        reader_snapshot=args.reader_snapshot,
        source_snapshot=snapshot,
        git_revision=git_revision(),
    )
    manifest_path = args.output / "scoring_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = validate_scoring_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    verify_sources(ROOT, snapshot)
    receipt = {
        "schema_version": "rag.reader_larger_untouched_scoring_manifest_receipt.v1",
        "status": "passed",
        "manifest_file": {
            "name": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": file_sha256(manifest_path),
        },
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "source_snapshot": snapshot,
        "checks": checks,
        "scoring_started": False,
    }
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "receipt": receipt}, ensure_ascii=False, indent=2))
    return args.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--sequential-values", type=Path, required=True)
    parser.add_argument("--reader-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
