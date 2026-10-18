#!/usr/bin/env python3
"""Independently replay every native trial in a sequential protected-update run."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from applications.rag.protected_boundary_inputs import load_bundle
from applications.rag.protected_update_feasibility import sequential_membership_counts
from applications.rag.quantum_reader_rotation import RotationConfig, make_rotation_head
from scripts.collab.five_ideas.run_reader_protected_update_feasibility import digest, inputs
from scripts.collab.five_ideas.run_reader_sequential_protected_updates import (
    compact_membership,
    validate_trace,
)
from scripts.collab.lib.source_snapshot import archive_sources


AUDIT_SOURCES = (
    "applications/rag/protected_update_feasibility.py",
    "applications/rag/quantum_reader_rotation.py",
    "scripts/collab/five_ideas/run_reader_protected_update_feasibility.py",
    "scripts/collab/five_ideas/run_reader_sequential_protected_updates.py",
    "scripts/collab/five_ideas/audit_reader_sequential_protected_updates.py",
    "scripts/collab/lib/source_snapshot.py",
)


def _verify_source_archive(run: Path, snapshot: dict) -> None:
    path = run / snapshot["file"]
    if digest(path) != snapshot["sha256"]:
        raise ValueError("executed-source archive hash drift")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("source_manifest.json"))
        if manifest["source_hashes"] != snapshot["source_hashes"]:
            raise ValueError("executed-source manifest drift")
        for name, expected in manifest["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("executed-source content drift: " + name)


def audit(args):
    torch.set_num_threads(1)
    trace = json.loads((args.run / "selector_trace.json").read_text(encoding="utf8"))
    if trace["synthetic_only"] or trace["population"] != {"training": 59, "evaluation": 0}:
        raise ValueError("audit requires the actual training59-only run")
    cfg = trace["config"]
    load_args = argparse.Namespace(
        synthetic=False,
        features=args.features,
        source=args.source,
        bundle=args.bundle,
    )
    features, records, targets, weight, bias, _ = inputs(load_args, cfg)
    ranks = [record["candidate_retrieved_ranks"] for record in records]
    if targets != trace["training_targets"] or ranks != trace["ranks"]:
        raise ValueError("qualified population identity drift")
    _verify_source_archive(args.run, trace["provenance"]["source_snapshot"])
    readiness = validate_trace(args.run)
    rows, _ = load_bundle(args.bundle)
    report = {
        "schema_version": "rag.reader_sequential_protected_audit.v1",
        "run_id": trace["run_id"],
        "status": "passed",
        "trace_sha256": digest(args.run / "selector_trace.json"),
        "values_sha256": digest(args.run / "sequential_values.npz"),
        "auditor_sha256": digest(Path(__file__)),
        "source_files_verified": len(trace["provenance"]["source_snapshot"]["source_hashes"]),
        "readiness": readiness,
        "all59_native_baselines": 0,
        "all59_native_trials": 0,
        "accepted_steps": 0,
        "rejected_native_trials": 0,
        "checkpoints": [],
        "data_access": "training59 only; no evaluation100/validation87 judgments or features",
        "claim_ceiling": "L0_train_only_sequential_feasibility",
    }
    with np.load(args.run / "sequential_values.npz", allow_pickle=False) as values:
        with np.load(args.source / "membership_values.npz", allow_pickle=False) as source:
            for point in trace["checkpoints"]:
                family = point["id"].split("_")[0]
                state = {name: torch.from_numpy(values[key].copy()) for name, key in point["head_state"].items()}
                for name, tensor in state.items():
                    np.testing.assert_array_equal(
                        tensor.numpy(), source[f"{family}_membership__initial__{name}"]
                    )
                head = make_rotation_head(
                    weight,
                    bias,
                    family=family,
                    config=RotationConfig(**cfg["source_rotation"]),
                )
                head.load_state_dict(state)

                def forward(theta):
                    with torch.no_grad():
                        head.theta.copy_(torch.as_tensor(theta, dtype=state["theta"].dtype))
                        return np.stack(
                            [head(feature["cls"], feature["pooled"])["scores"].numpy() for feature in features]
                        )

                current_theta = state["theta"].numpy().astype(np.float64).copy()
                baseline = forward(current_theta)
                np.testing.assert_array_equal(baseline, values[point["initial_scores"]])
                initial_health = sequential_membership_counts(baseline, targets, ranks, [set() for _ in targets])
                if compact_membership(initial_health) != point["initial_membership"]:
                    raise ValueError("initial membership drift")
                report["all59_native_baselines"] += 1
                accepted = [set() for _ in targets]
                current_rescue = initial_health["eligible_rescued"]
                checkpoint = {
                    "id": point["id"],
                    "initial_rescued": current_rescue,
                    "steps": [],
                }
                for step in point["steps"]:
                    np.testing.assert_array_equal(current_theta, values[step["theta_before"]])
                    step_record = {
                        "step": step["step"],
                        "accepted": False,
                        "attempts": [],
                    }
                    accepted_native = None
                    for trial in step["trials"]:
                        trial_record = {
                            "attempt": trial["attempt"],
                            "proposal": trial["proposal"],
                            "native": [],
                        }
                        if trial["solver"]["success"]:
                            delta = values[trial["delta"]].reshape(current_theta.shape)
                            for native in trial["native"]:
                                candidate_theta = current_theta + native["fraction"] * delta
                                replayed = forward(candidate_theta)
                                np.testing.assert_array_equal(replayed, values[native["scores"]])
                                health = sequential_membership_counts(replayed, targets, ranks, accepted)
                                safe = (
                                    health["eligible_rescued"] > current_rescue
                                    and health["lost_protected"] == 0
                                    and health["accepted_lost"] == 0
                                    and trial["proposal"]["added"]
                                    in health["selected_indices"][trial["proposal"]["case"]]
                                )
                                if compact_membership(health) != native["membership"] or safe != native["accepted"]:
                                    raise ValueError("native trial replay drift")
                                report["all59_native_trials"] += 1
                                if not safe:
                                    report["rejected_native_trials"] += 1
                                trial_record["native"].append({
                                    "fraction": native["fraction"],
                                    "accepted": safe,
                                    **compact_membership(health),
                                })
                                if safe:
                                    if accepted_native is not None:
                                        raise ValueError("multiple accepted trials in one step")
                                    accepted_native = (candidate_theta.copy(), health, native)
                        step_record["attempts"].append(trial_record)
                    if step["accepted"] != (accepted_native is not None):
                        raise ValueError("accepted-step replay drift")
                    if accepted_native is not None:
                        current_theta, health, native = accepted_native
                        accepted = [set(items) for items in native["accepted_after"]]
                        current_rescue = health["eligible_rescued"]
                        report["accepted_steps"] += 1
                        case = step["accepted_trial"]["proposal"]["case"]
                        step_record.update({
                            "accepted": True,
                            "rescued_after": current_rescue,
                            "lost_protected": health["lost_protected"],
                            "accepted_lost": health["accepted_lost"],
                            "proposed_case": records[case]["prefix"],
                            "question": rows[case]["case"]["question"],
                            "added_index": step["accepted_trial"]["proposal"]["added"],
                            "accepted_total": sum(map(len, native["accepted_after"])),
                        })
                    checkpoint["steps"].append(step_record)
                np.testing.assert_array_equal(current_theta, values[point["final_theta"]])
                final_scores = forward(current_theta)
                final_health = sequential_membership_counts(final_scores, targets, ranks, accepted)
                if compact_membership(final_health) != point["final_membership"]:
                    raise ValueError("final membership replay drift")
                checkpoint["final_membership"] = compact_membership(final_health)
                checkpoint["accepted_final_total"] = sum(map(len, point["accepted_final"]))
                report["checkpoints"].append(checkpoint)
    audit_archive = args.run / "audit_sources.zip"
    audit_archive.unlink(missing_ok=True)
    report["audit_source_snapshot"] = archive_sources(ROOT, AUDIT_SOURCES, audit_archive)
    output = args.run / "independent_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    manifest = {
        "schema_version": "collaborator-exchange.manifest.v1",
        "target_directory": "five_ideas/reader_sequential_protected_updates/" + trace["run_id"],
        "exchange_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(args.run.iterdir())
            if path.is_file() and path.name not in {"upload_manifest.json", "upload_receipts.json"}
        ],
    }
    (args.run / "upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(json.dumps({
        "path": str(output),
        "status": report["status"],
        "all59_native_trials": report["all59_native_trials"],
        "accepted_steps": report["accepted_steps"],
        "rejected_native_trials": report["rejected_native_trials"],
    }), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    audit(parser.parse_args())
