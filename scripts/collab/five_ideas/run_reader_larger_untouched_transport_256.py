#!/usr/bin/env python3
"""Run one immutable, label-blind 256-question Reader transport pass."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from applications.rag.quantum_semantic_reader import reader_semantic_forward
from applications.rag.reader_larger_untouched_manifest import file_sha256
from applications.rag.reader_larger_untouched_transport import (
    VALUE_KEYS,
    build_trace,
    case_record,
    larger_transport_gate,
    restore_rotation_head,
    validate_trace,
    verify_bound_inputs,
)
from applications.rag.reader_unseen_transport import stable_top5, transport_metrics
from scripts.collab.lib.source_snapshot import archive_sources, verify_sources


DEFAULT_MANIFEST = ROOT / "configs/experiments/reader_larger_untouched_scoring_manifest.json"
SOURCES = (
    "applications/rag/quantum_reader_rotation.py",
    "applications/rag/quantum_semantic_reader.py",
    "applications/rag/reader_larger_untouched.py",
    "applications/rag/reader_larger_untouched_manifest.py",
    "applications/rag/reader_larger_untouched_transport.py",
    "applications/rag/reader_unseen_transport.py",
    "configs/experiments/reader_larger_untouched_scoring_manifest.json",
    "scripts/collab/five_ideas/run_reader_larger_untouched_transport_256.py",
    "scripts/collab/five_ideas/audit_reader_larger_untouched_transport_256.py",
    "scripts/collab/lib/source_snapshot.py",
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def input_paths(args: argparse.Namespace) -> dict[str, str]:
    return {
        "manifest": str(args.manifest.resolve()),
        "cohort_directory": str(args.cohort_dir.resolve()),
        "reader_snapshot": str(args.reader_snapshot.resolve()),
        "sequential_values": str(args.sequential_values.resolve()),
    }


def run(args: argparse.Namespace) -> Path:
    import torch
    from transformers import DPRReader, DPRReaderTokenizer

    if args.torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(args.torch_threads)
    manifest, cohort, bindings = verify_bound_inputs(
        root=ROOT,
        manifest_path=args.manifest,
        cohort_dir=args.cohort_dir,
        reader_snapshot=args.reader_snapshot,
        sequential_values=args.sequential_values,
    )
    batch_size = args.batch_size or int(manifest["reader"]["frozen_max_batch_size"])
    if not 1 <= batch_size <= int(manifest["reader"]["frozen_max_batch_size"]):
        raise ValueError("batch size exceeds the frozen resource-only maximum")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    tokenizer = DPRReaderTokenizer.from_pretrained(str(args.reader_snapshot), local_files_only=True)
    reader = DPRReader.from_pretrained(str(args.reader_snapshot), local_files_only=True)
    reader.to(device).eval().requires_grad_(False)
    classifier = reader.span_predictor.qa_classifier
    reader_weight = classifier.weight.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)
    reader_bias = np.asarray(classifier.bias.detach().cpu().numpy().reshape(()), dtype=np.float32)
    with np.load(args.sequential_values, allow_pickle=False) as checkpoint:
        quantum = restore_rotation_head(
            checkpoint, "quantum", manifest, reader_weight, reader_bias, device=device
        )
        classical = restore_rotation_head(
            checkpoint, "classical", manifest, reader_weight, reader_bias, device=device
        )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output / run_id
    out.mkdir(parents=True, exist_ok=False)
    source_snapshot = archive_sources(ROOT, SOURCES, out / "executed_sources.zip")
    records: list[dict] = []
    base_rows: list[np.ndarray] = []
    cls_rows: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []
    quantum_rows: list[np.ndarray] = []
    classical_rows: list[np.ndarray] = []
    try:
        for number, case in enumerate(cohort["cases"], 1):
            prefix = f"unseen_{number:03d}"
            record = case_record(case, prefix=prefix)
            online_texts = []
            for candidate in case["candidates"]:
                title = str(candidate.get("title") or "").strip()
                body = candidate["text"].strip()
                online_texts.append(f"{title}. {body}" if title else body)
            base, pooled, _scalar, _sample = reader_semantic_forward(
                torch,
                tokenizer,
                reader,
                device,
                case["question"],
                online_texts,
                max_length=int(manifest["reader"]["max_length"]),
                batch_size=batch_size,
                capture_first=False,
            )
            base_np = base.numpy().astype(np.float32, copy=True)
            pooled_np = pooled.numpy().astype(np.float32, copy=True)
            cls_np = pooled_np[:, : reader.config.hidden_size].copy()
            with torch.no_grad():
                replayed = (
                    torch.nn.functional.linear(
                        torch.from_numpy(cls_np).to(device),
                        classifier.weight,
                        classifier.bias,
                    )
                    .squeeze(-1).cpu().numpy().astype(np.float32)
                )
            if not np.allclose(base_np, replayed, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Reader classifier parity failed for {prefix}")
            if stable_top5(base_np) != stable_top5(replayed):
                raise ValueError(f"Reader Top-5 parity failed for {prefix}")
            with torch.no_grad():
                cls_tensor = torch.from_numpy(cls_np).to(device)
                pooled_tensor = torch.from_numpy(pooled_np).to(device)
                quantum_np = quantum(cls_tensor, pooled_tensor)["scores"].detach().cpu().numpy().astype(np.float32)
                classical_np = classical(cls_tensor, pooled_tensor)["scores"].detach().cpu().numpy().astype(np.float32)
            records.append(record)
            base_rows.append(base_np)
            cls_rows.append(cls_np)
            pooled_rows.append(pooled_np)
            quantum_rows.append(quantum_np)
            classical_rows.append(classical_np)
            print(f"scored {number}/256", flush=True)

        values = {
            "identity_scores": np.stack(base_rows).astype(np.float32),
            "cls": np.stack(cls_rows).astype(np.float32),
            "pooled": np.stack(pooled_rows).astype(np.float32),
            "quantum_scores": np.stack(quantum_rows).astype(np.float32),
            "classical_scores": np.stack(classical_rows).astype(np.float32),
        }
        if tuple(values) != VALUE_KEYS:
            raise ValueError("transport numeric order drift")
        values_path = out / "transport_values.npz"
        np.savez(values_path, **values)
        metrics = transport_metrics(
            values["identity_scores"], values["quantum_scores"], values["classical_scores"]
        )
        gate = larger_transport_gate(metrics, manifest)
        runtime = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "execution_batch_size": batch_size,
            "torch_threads": args.torch_threads,
        }
        trace = build_trace(
            run_id=run_id,
            records=records,
            values=values,
            values_sha256=file_sha256(values_path),
            manifest=manifest,
            metrics=metrics,
            gate=gate,
            provenance={
                "git_revision": git_revision(),
                "manifest_sha256": file_sha256(args.manifest),
                "cohort_sha256": file_sha256(args.cohort_dir / "cohort.json"),
                "reader_model_sha256": file_sha256(args.reader_snapshot / "pytorch_model.bin"),
                "sequential_values_sha256": file_sha256(args.sequential_values),
                "input_paths": input_paths(args),
                "input_bindings": bindings,
                "source_snapshot": source_snapshot,
                "runtime": runtime,
                "offline_model": True,
            },
        )
        trace_path = out / "selector_trace.json"
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        structural = validate_trace(trace, values, manifest, expected_records=records)
        summary = {
            "schema_version": "rag.reader_larger_untouched_transport_result.v1",
            "run_id": run_id,
            "status": "scoring_completed_replay_pending",
            "population": 256,
            "transport_metrics": metrics,
            "gate": gate,
            "structural_validation": structural,
            "independent_vector_replay": "pending",
            "label_boundary": dict(manifest["label_boundary"]),
            "claim_ceiling": "L0_label_blind_transport",
            "next_step": "run the independent vector audit; semantic review remains blocked until it passes and both transport gates pass",
        }
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        verify_sources(ROOT, source_snapshot)
        files = [path for path in sorted(out.iterdir()) if path.is_file()]
        upload_manifest = {
            "schema_version": "collaborator-exchange.manifest.v1",
            "target_directory": f"five_ideas/reader_larger_untouched_transport_256/{run_id}",
            "exchange_files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in files
            ],
        }
        (out / "upload_manifest.json").write_text(
            json.dumps(upload_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output": str(out), "summary": summary}, ensure_ascii=False, indent=2), flush=True)
        return out
    except Exception:
        failure = {
            "schema_version": "rag.reader_larger_untouched_transport_failure.v1",
            "run_id": run_id,
            "status": "failed_before_complete_trace",
            "label_boundary": dict(manifest["label_boundary"]),
        }
        (out / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--reader-snapshot", type=Path, required=True)
    parser.add_argument("--sequential-values", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    run(parser.parse_args())
