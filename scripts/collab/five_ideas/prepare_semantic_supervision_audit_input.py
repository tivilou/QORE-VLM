#!/usr/bin/env python3
"""Extract a small hash-locked audit input from the verified repaired run."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag.semantic_supervision_audit import TRAINABLE, digest_file, validate_cohort, write_json

EXPECTED_SOURCE = {
    "summary.json": "b0423f31195f8628a34286b5e0f7fbff579cd7c8678e13a5decaa7e64831d24e",
    "selector_trace.json": "7e69b3efe0f733f15c68e2ea62bdd7c633e7f8122718bf0d6af90cb7b0fff258",
    "semantic_values.npz": "573209aa882b4d8122ffae26b2dc37bff214fb7882744e5dae4d3d3e22d76843",
}


def prepare(source, target):
    import json
    for name, expected in EXPECTED_SOURCE.items():
        if digest_file(source / name) != expected:
            raise ValueError("original repaired input hash mismatch: " + name)
    summary = json.loads((source / "summary.json").read_text())
    trace = json.loads((source / "selector_trace.json").read_text())
    cohort = {"schema_version": "rag.semantic_training_audit_cohort.v1",
              "source_run": "five_ideas/semantic_reader_training_repair_100/20261008T145652Z",
              "source_artifact_sha256": EXPECTED_SOURCE,
              "source_code_revision": summary["provenance"]["code_revision"],
              "reader": summary["provenance"]["reader"],
              "source_config": summary["provenance"]["effective_config"],
              "requested_cases": summary["training"]["requested"],
              "usable_cases": summary["training"]["usable"],
              "training_identity": trace["training_identity"],
              "evaluation_question_sha256": [r["question_sha256"] for r in trace["cases"]],
              "original_full_training_text_hashes_available": False}
    validate_cohort(cohort)
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / "cohort.json", cohort)
    with np.load(source / "semantic_values.npz", allow_pickle=False) as values:
        keys = [key for key in values.files if any(key.startswith(f"checkpoint__{m}__{epoch}__")
                for m in TRAINABLE for epoch in ("initial", "3")) or
                any(key == f"training_{i:03d}__{field}" for i in (1, 2) for field in ("base", "pooled", "scalar", "weak_mask"))]
        if len(keys) < 8 or not all(np.isfinite(values[k]).all() for k in keys):
            raise ValueError("checkpoint/witness extraction failed")
        np.savez_compressed(target / "head_checkpoints.npz", **{k: values[k] for k in keys})
        with np.load(target / "head_checkpoints.npz", allow_pickle=False) as reopened:
            for key in keys:
                np.testing.assert_array_equal(reopened[key], values[key])
    records = [{"name": name, "bytes": (target / name).stat().st_size, "sha256": digest_file(target / name)}
               for name in ("cohort.json", "head_checkpoints.npz")]
    write_json(target / "input_manifest.json", {"schema_version": "rag.semantic_training_audit_input.v1",
               "source_run": cohort["source_run"], "source_artifact_sha256": EXPECTED_SOURCE,
               "files": records, "checkpoint_epochs": ["initial", "3"], "target_boundary": "train-only; evaluation question hashes are exclusion-only"})
    print(json.dumps({"target": str(target), "files": records,
                      "input_manifest_sha256": digest_file(target / "input_manifest.json")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.output)
