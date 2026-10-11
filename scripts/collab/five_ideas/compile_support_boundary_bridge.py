#!/usr/bin/env python3
"""Offline exact mapping + qualification audit. Never loads models or trains."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag import support_boundary_bridge as bridge
from applications.rag import support_supervision_axes as axes

CONFIG = ROOT / "configs/experiments/support_boundary_bridge.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compile_run(trace_path, primary_path, adjudication_root, output, *, config_path=CONFIG, qualification=None):
    cfg = load(config_path)
    axes.require(cfg == load(CONFIG), "use the frozen bridge configuration")
    trace = bridge.checked(trace_path, cfg["source_sha256"]["trace"])
    primary = bridge.checked(primary_path, cfg["source_sha256"]["primary"])
    root = Path(adjudication_root) / "root/output"
    pointer = load(root / "FINAL.json")
    axes.require(pointer["status"] == "provisional_adjudication_complete_qualification_pending", "unsupported root final status")
    for name in ("review_path", "dispositions_path"):
        axes.require(Path(pointer[name]).name == pointer[name], "final artifact must be sibling")
    review = bridge.checked(root / pointer["review_path"], cfg["source_sha256"]["root_review"])
    dispositions = bridge.checked(root / pointer["dispositions_path"], cfg["source_sha256"]["dispositions"])
    axes.require(pointer["review_sha256"] == cfg["source_sha256"]["root_review"] and
                 pointer["dispositions_sha256"] == cfg["source_sha256"]["dispositions"], "final pointer binding changed")
    merged, held = bridge.merge_opinions(trace, primary, review, dispositions)
    axes.require(len(trace["cases"]) == 434 and len(primary["cases"]) == 347 and
                 len(review["cases"]) == 94 and len(dispositions["items"]) == 439 and len(held) == 38,
                 "frozen real input coverage mismatch")
    pending = bridge.qualification_request(trace, merged)
    if qualification is not None:
        merged = bridge.apply_explicit_qualification(trace, merged, load(qualification), Path(qualification).resolve().parent)
    draft, summary, details = bridge.protected_draft(trace, merged, residual_bound=cfg["residual_bound"])
    summary.update(schema_version="rag.support_boundary_bridge_summary.v1", mapped_adjudicated_items=439,
                   accepted_provisional_items=401, source_sha256=cfg["source_sha256"],
                   validation_semantic_labels_added=0, engineering_interface_only=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    values = {"mapped_annotations.json": merged, "qualification_request.json": pending,
              "protected_target_draft.json": draft, "pair_audit.private.json": {"cases": details}, "summary.json": summary}
    files = []
    for name, value in values.items():
        path = output / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        axes.require(load(path) == value, "output round-trip failed")
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / "manifest.json").write_text(json.dumps({"schema_version": "rag.support_boundary_bridge_manifest.v1",
        "files": files, "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "training_authorized": False, "training_consumable": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / ".cache/support_boundary_bridge/20261011-source/case_study.json")
    parser.add_argument("--primary", type=Path, default=ROOT / ".cache/support_axes_review/20261010-train-full-primary/primary_annotations.json")
    parser.add_argument("--adjudication", type=Path, default=ROOT / ".cache/support_axes_adjudication/20261011-two-agent-v1")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / ".cache/support_boundary_bridge" / stamp
    compile_run(args.trace, args.primary, args.adjudication, output, qualification=args.qualification)


if __name__ == "__main__":
    main()
