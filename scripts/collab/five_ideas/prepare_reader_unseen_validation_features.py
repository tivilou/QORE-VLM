#!/usr/bin/env python3
"""Build a label-blind DPR Reader feature cache for the frozen validation87 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from applications.rag.quantum_semantic_reader import reader_semantic_forward
from applications.rag.reader_unseen_transport import stable_top5

CONFIG = ROOT / "configs/experiments/reader_unseen_transport_gate_87.json"
SOURCE_SHA = "f723bbb6f9075c3e04fd3d8a7b2d3f8127b6e354422ee2a2f6e167bfe32248ce"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_cases(bundle: Path) -> list[dict]:
    if digest(bundle) != SOURCE_SHA:
        raise ValueError("frozen support bundle hash mismatch")
    with zipfile.ZipFile(bundle) as archive:
        source = json.loads(archive.read("source.json"))
    cases = [case for case in source["cases"] if case.get("future_role") == "validation"]
    if len(cases) != 87 or any(case.get("future_role") != "validation" for case in cases):
        raise ValueError("validation cohort coverage drift")
    # Deliberately copy only model-input fields.  No answer/support field is read.
    return cases


def build(args: argparse.Namespace) -> Path:
    import torch
    from transformers import DPRReader, DPRReaderTokenizer

    if args.torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(args.torch_threads)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    bundle = args.bundle
    cases = source_cases(bundle)
    spec = cfg["reader"]
    options = {"revision": spec["revision"], "local_files_only": True}
    tokenizer = DPRReaderTokenizer.from_pretrained(spec["model_id"], **options)
    reader = DPRReader.from_pretrained(spec["model_id"], **options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reader.to(device).eval().requires_grad_(False)
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {}
    records = []
    for number, case in enumerate(cases, 1):
        candidates = case["candidates"]
        texts = [{key: candidate.get(key) for key in ("title", "text")} for candidate in candidates]
        online_texts = []
        for item in texts:
            title = str(item.get("title") or "").strip()
            body = item.get("text")
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f"candidate text must be non-empty for validation_{number:03d}")
            online_texts.append(f"{title}. {body.strip()}" if title else body.strip())
        batch_size = args.batch_size or spec["batch_size"]
        if batch_size < 1 or batch_size > spec["batch_size"]:
            raise ValueError("batch_size must be between 1 and the frozen maximum")
        base, pooled, _scalar, _hidden = reader_semantic_forward(
            torch, tokenizer, reader, device, case["question"], online_texts,
            max_length=spec["max_length"], batch_size=batch_size, capture_first=False,
        )
        base_np = base.numpy().astype(np.float32, copy=True)
        pooled_np = pooled.numpy().astype(np.float32, copy=True)
        hidden = reader.config.hidden_size
        prefix = f"validation_{number:03d}"
        arrays[prefix + "__base"] = base_np
        arrays[prefix + "__cls"] = pooled_np[:, :hidden]
        arrays[prefix + "__pooled"] = pooled_np
        expected = np.asarray([candidate["reader_score"] for candidate in candidates], dtype=np.float32)
        if not np.allclose(base_np, expected, rtol=1e-4, atol=1e-4):
            raise ValueError(f"Reader score parity failed for {prefix}")
        if stable_top5(base_np) != list(case["reader_top5_indices"]):
            raise ValueError(f"Reader Top5 parity failed for {prefix}")
        records.append({
            "prefix": prefix,
            "number": number,
            "question_sha256": case["question_sha256"],
            "candidate_id_sha256": [hashlib.sha256(str(p["id"]).encode()).hexdigest() for p in candidates],
            "candidate_retrieved_ranks": [int(p["retrieved_rank"]) for p in candidates],
            "baseline_top5_indices": list(case["reader_top5_indices"]),
            "input_sha256": hashlib.sha256((case["question"] + "\n" + "\n".join(online_texts)).encode()).hexdigest(),
            "lane": "validation",
        })
        print(f"cached {number}/87", flush=True)
    np.savez(out / "reader_features.npz", **arrays)
    np.save(out / "relevance_weight.npy", reader.span_predictor.qa_classifier.weight.detach().cpu().numpy().reshape(-1).astype(np.float32))
    np.save(out / "relevance_bias.npy", np.asarray(reader.span_predictor.qa_classifier.bias.detach().cpu().numpy().reshape(()), dtype=np.float32))
    manifest = {
        "schema_version": "rag.reader_unseen_feature_cache.v1",
        "source_bundle_sha256": SOURCE_SHA,
        "model": {"model_id": spec["model_id"], "revision": spec["revision"], "device": str(device),
                  "frozen_batch_size": spec["batch_size"], "execution_batch_size": batch_size},
        "records": records,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for key, value in arrays.items()},
        "label_boundary": {"answers_read": False, "support_annotations_read": False, "silver_oracle_read": False},
        "provenance": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "offline_model": True,
                       "config_sha256": digest(CONFIG), "producer_sha256": digest(Path(__file__))},
    }
    (out / "cache_index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "records": len(records), "device": str(device)}, indent=2), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / ".cache/protected_boundary_reader_inputs/20261011-v1/frozen_support_bundle.zip")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/reader-unseen-validation87-v1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Resource-only microbatch override; must not exceed the frozen batch size")
    build(parser.parse_args())
