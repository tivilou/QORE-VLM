#!/usr/bin/env python3
"""Extract only frozen Reader features from an uploaded, hash-locked run.

No model, GPU, fitting or dataset access. Historical fitted scores/parameters
and evaluation judgments are excluded from the derived input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.collab.lib.source_snapshot import archive_sources

SOURCE_RUN = "20261015T022004545902Z"
SOURCE_TRACE_SHA = "89130f8b349b24a44861c802c13be30aa08fbbc5dc661a18f969ca2c68be4aa6"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def prepare(source, output):
    source, output = Path(source), Path(output)
    if digest(source / "selector_trace.json") != SOURCE_TRACE_SHA:
        raise ValueError("wrong source trace")
    manifest = json.loads((source / "upload_manifest.json").read_text())
    npz_spec = next(x for x in manifest["exchange_files"] if x["name"] == "rotation_values.npz")
    if digest(source / "rotation_values.npz") != npz_spec["sha256"]:
        raise ValueError("source full NPZ hash mismatch")
    trace = json.loads((source / "selector_trace.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    arrays, records, max_error = {}, [], 0.
    with np.load(source / "rotation_values.npz", allow_pickle=False) as raw:
        weight = raw["training_001__relevance_weight"].reshape(-1).copy()
        bias = raw["training_001__relevance_bias"].reshape(()).copy()
        arrays.update(relevance_weight=weight, relevance_bias=bias)
        for lane, expected in (("training", 59), ("evaluation", 100)):
            rows = trace[lane + "_cases"]
            if len(rows) != expected: raise ValueError("source population drift")
            for number, row in enumerate(rows, 1):
                prefix = f"{lane}_{number:03d}"
                for key, shape in (("base", (50,)), ("cls", (50, len(weight))), ("pooled", (50, 5 * len(weight)))):
                    value = raw[prefix + "__" + key].copy()
                    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                        raise ValueError("feature shape/dtype/finite mismatch: " + prefix + key)
                    arrays[prefix + "__" + key] = value
                base = arrays[prefix + "__base"]
                cls = arrays[prefix + "__cls"]
                if not np.array_equal(cls, arrays[prefix + "__pooled"][:, :len(weight)]):
                    raise ValueError("CLS/pooling lineage mismatch")
                error = float(np.max(np.abs(cls @ weight + bias - base)))
                max_error = max(max_error, error)
                if not np.allclose(cls @ weight + bias, base, rtol=1e-5, atol=3e-5):
                    raise ValueError("cached Reader linear score replay failed")
                ranks = row["candidate_retrieved_ranks"]
                if sorted(ranks) != list(range(1, 51)): raise ValueError("nonunique retrieval ranks")
                selected = sorted(range(50), key=lambda i: (-float(base[i]), ranks[i]))[:5]
                frozen = row["methods"]["frozen_reader_topk"]
                np.testing.assert_allclose(base, frozen["score_vector"], rtol=0, atol=0)
                if selected != frozen["selected_indices"]: raise ValueError("source Top5 mismatch")
                records.append({"lane": lane, "number": number, "prefix": prefix,
                    "question_sha256": row["question_sha256"], "candidate_id_sha256": row["candidate_id_sha256"],
                    "candidate_retrieved_ranks": ranks, "baseline_top5_indices": selected})
        for prefix in ("training_002", "evaluation_001", "evaluation_002"):
            np.testing.assert_array_equal(raw[prefix + "__relevance_weight"].reshape(-1), weight)
            np.testing.assert_array_equal(raw[prefix + "__relevance_bias"].reshape(()), bias)
    qids = [r["question_sha256"] for r in records]
    if len(set(qids)) != 159: raise ValueError("duplicate/overlapping question identity")
    np.savez_compressed(output / "reader_features.npz", **arrays)
    identity = trace["provenance"]["reader"]
    index = {"schema_version": "rag.reader_membership_cache.v1", "source_run": SOURCE_RUN,
        "reader": identity, "source_inputs": {"selector_trace.json": SOURCE_TRACE_SHA, "rotation_values.npz": npz_spec["sha256"]},
        "feature_sha256": digest(output / "reader_features.npz"), "records": records,
        "checks": {"cases": 159, "linear_replay_max_abs": max_error, "baseline_top5_replays": 159, "label_fields_copied": False}}
    write(output / "cache_index.json", index)
    snapshot = archive_sources(ROOT, ("scripts/collab/five_ideas/prepare_reader_membership_cache.py", "scripts/collab/lib/source_snapshot.py"), output / "preparation_sources.zip")
    write(output / "input_manifest.json", {"schema_version": "rag.reader_membership_cache.manifest.v1", "source_run": SOURCE_RUN,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(output.iterdir()) if p.is_file()],
        "source_snapshot": snapshot, "checks": index["checks"], "historical_provenance_exception": "Exact prior executed runner is absent; this cache certifies stored Reader coordinates and scores, not historical optimizer/source replay"})
    print(json.dumps({"output": str(output), "checks": index["checks"], "files": json.loads((output / "input_manifest.json").read_text())["files"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prepare(args.source_run, args.output)
