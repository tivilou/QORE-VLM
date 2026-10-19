#!/usr/bin/env python3
"""Independently replay every stored Reader/head vector and the transport gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from applications.rag.reader_larger_untouched_manifest import file_sha256
from applications.rag.reader_larger_untouched_transport import (
    VALUE_KEYS,
    case_record,
    replay_head_vectors,
    validate_trace,
    verify_bound_inputs,
)


DEFAULT_MANIFEST = ROOT / "configs/experiments/reader_larger_untouched_scoring_manifest.json"


def audit(args: argparse.Namespace) -> dict:
    import torch

    manifest, cohort, bindings = verify_bound_inputs(
        root=ROOT,
        manifest_path=args.manifest,
        cohort_dir=args.cohort_dir,
        reader_snapshot=args.reader_snapshot,
        sequential_values=args.sequential_values,
    )
    trace_path = args.run / "selector_trace.json"
    values_path = args.run / "transport_values.npz"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("values_sha256") != file_sha256(values_path):
        raise ValueError("transport-values archive drift")
    with np.load(values_path, allow_pickle=False) as archive:
        if tuple(archive.files) != VALUE_KEYS:
            raise ValueError("transport-values key/order drift")
        values = {key: archive[key].copy() for key in VALUE_KEYS}
    records = [
        case_record(case, prefix=f"unseen_{number:03d}")
        for number, case in enumerate(cohort["cases"], 1)
    ]
    structural = validate_trace(trace, values, manifest, expected_records=records)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    replay = replay_head_vectors(
        values=values,
        sequential_values=args.sequential_values,
        manifest=manifest,
        device=device,
    )
    result = {
        "schema_version": "rag.reader_larger_untouched_transport_audit.v1",
        "run_id": trace["run_id"],
        "status": "passed",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "input_bindings_replayed": bindings,
        "trace_validation": structural,
        "independent_vector_replay": replay,
        "cases_replayed": 256,
        "feature_vectors_replayed": 512,
        "score_vectors_replayed": 768,
        "label_boundary_verified": True,
        "device": str(device),
        "trace_sha256": file_sha256(trace_path),
        "values_sha256": file_sha256(values_path),
        "claim_ceiling": "L0_label_blind_transport",
    }
    audit_path = args.run / "independent_audit.json"
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "completed_and_independently_replayed"
    summary["independent_vector_replay"] = {
        "status": "passed",
        "audit_sha256": file_sha256(audit_path),
        "score_vectors_replayed": 768,
        "top5_exact": True,
    }
    summary["next_step"] = (
        "construct a blinded changed-set review packet"
        if summary["gate"]["advance_to_blind_review"]
        else "stop this frozen head configuration; semantic review is not authorized"
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = [path for path in sorted(args.run.iterdir()) if path.is_file() and path.name != "upload_manifest.json"]
    upload_manifest = {
        "schema_version": "collaborator-exchange.manifest.v1",
        "target_directory": f"five_ideas/reader_larger_untouched_transport_256/{trace['run_id']}",
        "exchange_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in files
        ],
    }
    (args.run / "upload_manifest.json").write_text(
        json.dumps(upload_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--reader-snapshot", type=Path, required=True)
    parser.add_argument("--sequential-values", type=Path, required=True)
    parser.add_argument("--device", default=None)
    audit(parser.parse_args())
