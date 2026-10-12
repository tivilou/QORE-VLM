#!/usr/bin/env python3
"""Package the frozen private audit; output contains texts and stays out of Git."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag.protected_boundary_inputs import build_bundle, load_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_bundle(args.source, args.final, args.output)
    _, audit = load_bundle(args.output)
    print(json.dumps({**receipt, **audit}))
