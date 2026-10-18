#!/usr/bin/env python3
"""Independently replay the label-blind validation87 transport metrics and gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from applications.rag.reader_unseen_transport import evaluate_transport_gate, stable_top5, transport_metrics

CONFIG = ROOT / "configs/experiments/reader_unseen_transport_gate_87.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(run: Path) -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    trace = json.loads((run / "selector_trace.json").read_text(encoding="utf-8"))
    values_path = run / "transport_values.npz"
    if trace["schema_version"] != "sample-trace.v2" or trace["population"] != {"validation": 87, "evaluation": 0, "training": 0}:
        raise ValueError("trace identity drift")
    if trace["values_sha256"] != digest(values_path):
        raise ValueError("values archive drift")
    if any(trace["label_boundary"].values()):
        raise ValueError("label boundary crossed")
    with np.load(values_path, allow_pickle=False) as values:
        if set(values.files) != {"identity_scores", "quantum_scores", "classical_scores"}:
            raise ValueError("numeric allowlist drift")
        for name, spec in trace["values"].items():
            value = values[name]
            if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"] or hashlib.sha256(value.tobytes()).hexdigest() != spec["sha256"]:
                raise ValueError("numeric coordinate drift: " + name)
        computed = transport_metrics(values["identity_scores"], values["quantum_scores"], values["classical_scores"])
        if computed != trace["transport_metrics"]:
            raise ValueError("transport metric replay drift")
        gate = evaluate_transport_gate(computed, cfg)
        if gate != trace["gate"]:
            raise ValueError("prospective gate replay drift")
        for i, sample in enumerate(trace["samples"]):
            expected = {
                "identity_top5": stable_top5(values["identity_scores"][i]),
                "quantum_top5": stable_top5(values["quantum_scores"][i]),
                "classical_top5": stable_top5(values["classical_scores"][i]),
            }
            if any(sample[key] != value for key, value in expected.items()):
                raise ValueError(f"Top5 replay drift for sample {i}")
    return {"schema_version": "rag.reader_unseen_transport_audit.v1", "run_id": trace["run_id"], "synthetic_only": bool(trace.get("synthetic_only")), "status": "passed", "cases_replayed": 87,
            "score_vectors_replayed": 261, "label_boundary_verified": True, "transport_metrics": computed, "gate": gate,
            "trace_sha256": digest(run / "selector_trace.json"), "values_sha256": digest(values_path), "claim_ceiling": "L0_label_blind_transport"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    audit_path = args.run / "independent_audit.json"
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = args.run / "upload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exchange_files"] = [{"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                                  for path in sorted(args.run.iterdir()) if path.is_file() and path.name != "upload_manifest.json"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
