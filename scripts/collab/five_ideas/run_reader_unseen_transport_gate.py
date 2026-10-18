#!/usr/bin/env python3
"""Run the frozen, label-blind transport gate on validation87."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from applications.rag.quantum_reader_rotation import RotationConfig, make_rotation_head
from applications.rag.reader_unseen_transport import evaluate_transport_gate, transport_metrics, stable_top5
from scripts.collab.lib.source_snapshot import archive_sources, verify_sources

CONFIG = ROOT / "configs/experiments/reader_unseen_transport_gate_87.json"
PLAN = ROOT / "configs/experiments/reader_unseen_transport_gate_87_plan.json"
SOURCES = (
    "applications/rag/reader_unseen_transport.py",
    "applications/rag/quantum_reader_rotation.py",
    "scripts/collab/five_ideas/run_reader_unseen_transport_gate.py",
    "scripts/collab/five_ideas/audit_reader_unseen_transport_gate.py",
    "scripts/collab/lib/source_snapshot.py",
    "configs/experiments/reader_unseen_transport_gate_87.json",
    "configs/experiments/reader_unseen_transport_gate_87_plan.json",
    ".ai-progress/workstreams/rag-selector/refs/reader_sequential_protected_updates_20261018_recovery.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_features(directory: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = json.loads((directory / "cache_index.json").read_text(encoding="utf-8"))
    if index["label_boundary"]["answers_read"] or index["label_boundary"]["support_annotations_read"]:
        raise ValueError("feature cache crossed the label boundary")
    with np.load(directory / "reader_features.npz", allow_pickle=False) as arrays:
        records = index["records"]
        base = np.stack([arrays[r["prefix"] + "__base"] for r in records]).astype(np.float32)
        cls = np.stack([arrays[r["prefix"] + "__cls"] for r in records]).astype(np.float32)
        pooled = np.stack([arrays[r["prefix"] + "__pooled"] for r in records]).astype(np.float32)
    weight = np.load(directory / "relevance_weight.npy", allow_pickle=False)
    bias = np.load(directory / "relevance_bias.npy", allow_pickle=False)
    if base.shape != (87, 50) or cls.shape[0] != 87 or pooled.shape[0] != 87:
        raise ValueError("validation feature shape drift")
    return index, base, cls, pooled, (weight, bias)


def restore(values: np.lib.npyio.NpzFile, family: str, cfg: dict, expected_weight: np.ndarray, expected_bias: np.ndarray) -> torch.nn.Module:
    weight = torch.from_numpy(values[family + "__head__weight"].copy())
    bias = torch.from_numpy(values[family + "__head__bias"].copy())
    np.testing.assert_array_equal(weight.numpy().reshape(-1), expected_weight.reshape(-1))
    np.testing.assert_array_equal(bias.numpy().reshape(()), expected_bias.reshape(()))
    head = make_rotation_head(weight, bias, family=family, config=RotationConfig(**cfg["source_rotation"]))
    state = {name: torch.from_numpy(values[family + "__head__" + name].copy()) for name in head.state_dict()}
    head.load_state_dict(state)
    with torch.no_grad():
        head.theta.copy_(torch.from_numpy(values[family + "__final_theta"].copy()).reshape_as(head.theta))
    return head.eval()


def run(args: argparse.Namespace) -> Path:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan["authorization"] != "implemented" or cfg["population"] != {"validation_cases": 87, "top50": 50, "top5": 5}:
        raise ValueError("transport contract drift")
    index, base, cls, pooled, (weight, bias) = load_features(args.features)
    expected_sequential_sha = cfg["sequential_source"]["sha256"]
    if not args.synthetic and digest(args.sequential_values) != expected_sequential_sha:
        raise ValueError("sequential values hash mismatch")
    with np.load(args.sequential_values, allow_pickle=False) as values:
        head_cfg = {"source_rotation": cfg["head"]}
        quantum = restore(values, "quantum", head_cfg, weight, bias)
        classical = restore(values, "classical", head_cfg, weight, bias)
    with torch.no_grad():
        q_scores = np.stack([quantum(torch.from_numpy(cls[i]), torch.from_numpy(pooled[i]))["scores"].numpy() for i in range(87)])
        c_scores = np.stack([classical(torch.from_numpy(cls[i]), torch.from_numpy(pooled[i]))["scores"].numpy() for i in range(87)])
    metrics = transport_metrics(base, q_scores, c_scores)
    gate = evaluate_transport_gate(metrics, cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output / run_id
    out.mkdir(parents=True, exist_ok=False)
    snapshot = archive_sources(ROOT, SOURCES, out / "executed_sources.zip")
    values_path = out / "transport_values.npz"
    np.savez_compressed(values_path, identity_scores=base, quantum_scores=q_scores, classical_scores=c_scores)
    trace = {
        "schema_version": "sample-trace.v2", "artifact_type": "reader_unseen_transport_gate", "run_id": run_id, "synthetic_only": bool(args.synthetic),
        "population": {"validation": 87, "evaluation": 0, "training": 0},
        "stages": ["data", "preprocess", "scoring", "selection", "diagnosis"],
        "samples": [{"question_sha256": r["question_sha256"], "candidate_id_sha256": r["candidate_id_sha256"], "baseline_top5_indices": r["baseline_top5_indices"],
                     "identity_top5": stable_top5(base[i]), "quantum_top5": stable_top5(q_scores[i]), "classical_top5": stable_top5(c_scores[i])} for i, r in enumerate(index["records"])],
        "values": {"identity_scores": {"shape": list(base.shape), "dtype": str(base.dtype), "sha256": hashlib.sha256(base.tobytes()).hexdigest()}, "quantum_scores": {"shape": list(q_scores.shape), "dtype": str(q_scores.dtype), "sha256": hashlib.sha256(q_scores.tobytes()).hexdigest()}, "classical_scores": {"shape": list(c_scores.shape), "dtype": str(c_scores.dtype), "sha256": hashlib.sha256(c_scores.tobytes()).hexdigest()}},
        "values_sha256": digest(values_path),
        "label_boundary": cfg["label_boundary"], "transport_metrics": metrics, "gate": gate,
        "provenance": {"feature_cache_sha256": digest(args.features / "reader_features.npz"), "sequential_values_sha256": digest(args.sequential_values), "plan_sha256": digest(PLAN), "source_snapshot": snapshot, "offline_model": True, "answers_read": False, "support_annotations_read": False},
    }
    verify_sources(ROOT, snapshot)
    (out / "selector_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"schema_version": "rag.reader_unseen_transport_result.v1", "run_id": run_id, "synthetic_only": bool(args.synthetic), "status": "completed", "metrics": metrics, "gate": gate,
               "claim_ceiling": "fixture_only" if args.synthetic else "L0_label_blind_transport", "next_step": "construct blinded semantic review packet only if advance_to_blind_review=true"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "upload_manifest.json").write_text(json.dumps({"schema_version": "collaborator-exchange.manifest.v1", "target_directory": "five_ideas/reader_unseen_transport_gate_87/" + run_id, "exchange_files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in out.iterdir()]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / ".cache/reader-unseen-validation87-v1")
    parser.add_argument("--sequential-values", type=Path, default=ROOT / "exchange/five_ideas/reader_sequential_protected_updates/20261018T045707538341Z/sequential_values.npz")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "exchange/five_ideas/reader_unseen_transport_gate_87")
    run(parser.parse_args())
