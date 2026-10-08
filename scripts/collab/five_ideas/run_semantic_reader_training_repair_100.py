#!/usr/bin/env python3
"""Same fixed screen with a separately frozen optimizer-repair configuration."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen

CONFIG = ROOT / "configs/experiments/semantic_reader_training_repair_100.json"
PLAN = ROOT / "configs/experiments/semantic_reader_training_repair_100_plan.json"


def validate_repair():
    from applications.rag.semantic_reader_training import SCALED
    cfg = screen.validate_config(CONFIG, config_reference=CONFIG, plan_path=PLAN)
    contract = screen.old._load_json(PLAN)["repair_contract"]
    path = ROOT / contract["qualification_report"]
    report_digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if report_digest != contract["qualification_report_sha256"]:
        raise ValueError("qualification report identity mismatch")
    report = screen.old._load_json(path)
    if not report["passed"] or cfg["training"]["optimizer_policy"] != SCALED:
        raise ValueError("repair has not passed train-only qualification")
    for source, expected in report["canonical_source_hashes"].items():
        actual = hashlib.sha256((ROOT / source).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError("qualified source changed: " + source)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--input", type=Path)
    p.add_argument("--output-root", type=Path, default=ROOT / "exchange/five_ideas/semantic_reader_training_repair_100")
    p.add_argument("--device")
    p.add_argument("--exchange-url", default=screen.old.DEFAULT_EXCHANGE_URL)
    p.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--preflight", type=Path)
    args = p.parse_args()
    try:
        validate_repair()
        screen.run(args, config_reference=CONFIG, plan_path=PLAN)
    except Exception as exc:
        print(f"Training-repair screen failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
