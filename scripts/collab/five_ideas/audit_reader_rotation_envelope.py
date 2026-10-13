#!/usr/bin/env python3
"""Norm-only score envelope for a proposed pre-classifier rotation, no fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EXPECTED_VALUES_SHA = "e1f7be90084113a37842e7ca425e71608a357147a47f7e673944496c6be988c1"
EXPECTED_CASE_SHA = "b24a60bcafdba45fdfa2575214376b7851750e840b19011511ce5c4bfdbd6e72"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if file_hash(args.values) != EXPECTED_VALUES_SHA or file_hash(args.cases) != EXPECTED_CASE_SHA:
        raise ValueError("frozen result hashes differ")
    d = json.loads(args.cases.read_text(encoding="utf-8"))
    with np.load(args.values, allow_pickle=False) as values:
        w = values["training_001__relevance_weight"].astype(np.float64).reshape(-1)
        bias = float(values["training_001__relevance_bias"].reshape(()))
        for prefix in ("training_001", "training_002", "evaluation_001", "evaluation_002"):
            np.testing.assert_array_equal(w, values[prefix + "__relevance_weight"].reshape(-1))
            np.testing.assert_array_equal(bias, values[prefix + "__relevance_bias"].reshape(()))
        observed, radii, baseline_errors = [], [], []
        pair_count, envelope_reachable = 0, 0
        for lane in ("training", "evaluation"):
            rows = [r for r in d["cases"] if r["lane"] == lane]
            for i, row in enumerate(rows, 1):
                prefix = f"{lane}_{i:03d}"
                cls = values[prefix + "__pooled"][:, :len(w)].astype(np.float64)
                base = values[prefix + "__base"].astype(np.float64)
                replay = cls @ w + bias
                error = float(np.max(np.abs(replay - base)))
                np.testing.assert_allclose(replay, base, rtol=1e-4, atol=1e-4)
                baseline_errors.append(error)
                radius = np.linalg.norm(cls, axis=1) * np.linalg.norm(w)
                radii.extend(radius.tolist())
                observed.extend((base - bias).tolist())
                if lane == "training":
                    for p, n in row["training_target"]["draft_pairs"]:
                        pair_count += 1
                        envelope_reachable += int(radius[p] + radius[n] > 0)
    result = {
        "schema_version": "rag.reader_rotation_envelope.v1",
        "input_values_sha256": EXPECTED_VALUES_SHA, "input_case_study_sha256": EXPECTED_CASE_SHA,
        "classifier_dimension": len(w), "padded_dimension": 1 << (len(w) - 1).bit_length(),
        "classifier_weight_norm": float(np.linalg.norm(w)), "classifier_bias": bias,
        "score_replayed_cases": len(baseline_errors), "max_replay_abs_error": max(baseline_errors),
        "centered_reader_score_range": [min(observed), max(observed)],
        "norm_score_radius_range": [min(radii), max(radii)],
        "norm_score_radius_median": float(np.median(radii)),
        "pairs": pair_count, "unrestricted_independent_rotation_pair_envelope_reachable": envelope_reachable,
        "limitations": [
            "Unrestricted independent rotations give an optimistic norm-only envelope, not a trained shared-circuit guarantee.",
            "The proposed shallow circuit, classical controls, optimization, and Top5 utility are not evaluated here.",
            "All arithmetic uses saved tensors; no dataset or model call, labels used only in training pair accounting.",
        ], "claim_ceiling": "L0_diagnostic",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
