#!/usr/bin/env python3
"""Project-owned sample-trace.v2 structural and complete-coordinate replay checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag import reader_membership_training as learning


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def structural(trace, index, cfg):
    require(trace.get("schema_version") == "sample-trace.v2" and trace.get("artifact_type") == "reader_membership_selector_trace", "trace schema/type mismatch")
    require(tuple(trace.get("methods", ())) == learning.METHODS and tuple(trace.get("stages", ())) == learning.STAGES, "stage/method allowlist mismatch")
    records = index["records"]
    require(len(trace["cases"]) == len(records), "trace population coverage")
    expected_samples = [r["prefix"] for r in records if r["number"] <= 2]
    require([s["sample_id"] for s in trace["samples"]] == expected_samples, "selected sample rule mismatch")
    require(trace["sample_selection"]["selected_count"] == len(expected_samples), "sample count drift")
    require(trace["constraints"] == cfg["constraints"], "target boundary/constraint drift")
    for sample in trace["samples"]:
        require([s["stage"] for s in sample["stages"]] == list(learning.STAGES), "sample stage order")
        require(sample["semantic_roles"] == ["input", "reference", "result"], "sample roles missing")
        for stage in sample["stages"]:
            require(stage["status"] in ("observed", "not_applicable"), "unavailable stage")
            if stage["stage"] in ("scoring", "selection"):
                require(not any(ref.startswith(("train_target:", "posthoc:")) for ref in stage["input_refs"]), "target enters online stage")
            for ref in stage["input_refs"] + stage["output_refs"]:
                require(isinstance(ref, str) and ref, "empty lineage reference")
                if ":" not in ref: require(ref in trace["values"], "unresolved numeric lineage reference")
    for record, row in zip(records, trace["cases"]):
        for k in ("prefix", "lane", "number", "question_sha256", "candidate_id_sha256", "candidate_retrieved_ranks", "baseline_top5_indices"):
            require(row[k] == record[k], "case identity/order mismatch")
        require(tuple(row["methods"]) == learning.METHODS, "case method coverage mismatch")
        for item in row["methods"].values():
            require(len(item["score_vector"]) == 50 and np.isfinite(item["score_vector"]).all(), "missing full score vector")
            require(item["selected_indices"] == learning.top5(item["score_vector"], record["candidate_retrieved_ranks"]), "selection does not follow full scores")
            require(item["score_values"] in trace["values"], "score numeric producer missing")
    ntrain = sum(r["lane"] == "training" for r in records)
    if trace["preflight_only"]:
        require(not trace["training_witnesses"], "preflight performed optimization")
    else:
        expected = [(m, e, i) for m in learning.ARMS for e in range(1, cfg["training"]["epochs"] + 1) for i in range(min(2, ntrain))]
        actual = [(w["method"], w["epoch"], w["case_index"]) for w in trace["training_witnesses"]]
        require(actual == expected, "optimization sample/checkpoint coverage mismatch")
    return ntrain


def validate(directory, cfg, index, features, weight, bias, targets):
    directory = Path(directory)
    trace = json.loads((directory / "selector_trace.json").read_text(encoding="utf8"))
    ntrain = structural(trace, index, cfg)
    require(digest(directory / "membership_values.npz") == trace["values_sha256"], "value archive SHA mismatch")
    target_doc = json.loads((directory / "training_targets.private.json").read_text())
    require(target_doc["targets"] == targets and target_doc["question_sha256"] == [r["question_sha256"] for r in index["records"][:ntrain]], "replay target binding mismatch")
    checks = {"complete_arrays": 0, "selected_input_slices": 0, "head_score_replays": 0, "full_output_replays": 0, "optimizer_update_replays": 0}
    with np.load(directory / "membership_values.npz", allow_pickle=False) as arrays:
        require(set(arrays.files) == set(trace["values"]), "coordinate artifact coverage mismatch")
        for key, spec in trace["values"].items():
            value = arrays[key]
            require(list(value.shape) == spec["shape"] and str(value.dtype) == spec["dtype"] and np.isfinite(value).all(), "shape/dtype/finite mismatch: " + key)
            require(learning.array_hash(value) == spec["sha256"] and spec["complete_coordinates"] is True and spec["producer"] == key.split("__", 1)[0], "array hash/producer mismatch: " + key)
            checks["complete_arrays"] += 1
        heads = learning.make_heads(weight, bias, cfg)
        for method, head in heads.items():
            report = trace["training"][method]
            require(set(report["initial_state"]) == set(head.state_dict()) == set(report["final_state"]), "head state coverage mismatch")
            for name in head.state_dict():
                if name != "theta": np.testing.assert_array_equal(arrays[report["initial_state"][name]], arrays[report["final_state"][name]])
            require(not np.any(arrays[report["initial_state"]["theta"]]), "initial head is not exact null")
            head.load_state_dict({k: torch.from_numpy(arrays[ref].copy()) for k, ref in report["initial_state"].items()})
            np.testing.assert_array_equal(head.weight.numpy(), weight.numpy())
            np.testing.assert_array_equal(head.bias.numpy(), bias.numpy())
            if trace["preflight_only"]:
                np.testing.assert_array_equal(arrays[report["initial_state"]["theta"]], arrays[report["final_state"]["theta"]])
            else:
                require(len(report["epochs"]) == cfg["training"]["epochs"] and all(r["updates"] == ntrain for r in report["epochs"]), "training budget drift")
                # Also check final native optimizer state/step survives archival.
                opt = learning.make_optimizer(head, cfg)
                learning.restore_optimizer(opt, report["final_optimizer"], arrays)
                require(len(opt.state) == 1, "final optimizer state absent")
                state = next(iter(opt.state.values()))
                require(float(state["step"]) == ntrain * cfg["training"]["epochs"], "final optimizer update count mismatch")
                require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "unexpected optimizer slots")
            head.load_state_dict({k: torch.from_numpy(arrays[ref].copy()) for k, ref in report["final_state"].items()})
        final_counts = {m: {"rescued_eligible": 0, "lost_protected": 0} for m in learning.ARMS}
        for case_index, (feature, row) in enumerate(zip(features, trace["cases"])):
            if row["number"] <= 2:
                require(set(row["input_values"]) == {"base", "cls", "pooled"}, "missing selected input value")
                for key, ref in row["input_values"].items():
                    np.testing.assert_array_equal(arrays[ref], feature[key].numpy())
                    checks["selected_input_slices"] += 1
            for method, item in row["methods"].items():
                if method == "frozen_reader_topk": scores = feature["base"].numpy()
                else:
                    with torch.no_grad(): result = learning.forward(heads[method], feature)
                    scores = result["scores"].numpy()
                np.testing.assert_allclose(scores, item["score_vector"], rtol=2e-4, atol=3e-5)
                np.testing.assert_array_equal(arrays[item["score_values"]], np.asarray(item["score_vector"], dtype=np.float32))
                require(learning.top5(scores, row["candidate_retrieved_ranks"]) == item["selected_indices"], "reopened head Top5 replay mismatch")
                checks["head_score_replays"] += 1
                if not trace["preflight_only"] and case_index < ntrain:
                    target, selected = targets[case_index], set(item["selected_indices"])
                    expected = {"rescued_eligible": len(selected & set(target["eligible_positive_indices"])),
                        "lost_protected": len(set(target["immutable_selected_indices"]) - selected)}
                    require(item["training_membership"] == expected, "per-case membership diagnostic mismatch")
                    if method in final_counts:
                        for k, value in expected.items(): final_counts[method][k] += value
                if row["number"] <= 2 and method != "frozen_reader_topk":
                    require(set(item["output_values"]) == set(result), "incomplete selected output coordinates")
                    for key, ref in item["output_values"].items():
                        np.testing.assert_allclose(result[key].numpy(), arrays[ref], rtol=2e-4, atol=3e-5)
                    norms = np.linalg.norm(arrays[item["output_values"]["states"]], axis=-1)
                    np.testing.assert_allclose(norms, 1., rtol=1e-4, atol=1e-4)
                    checks["full_output_replays"] += 1
        if not trace["preflight_only"]:
            for method, counts in final_counts.items():
                reported = trace["training"][method]["epochs"][-1]["membership_health"]
                require(all(reported[k] == v for k, v in counts.items()), "final epoch/case membership counts mismatch")
        for w in trace["training_witnesses"]:
            i = w["case_index"]
            require(w["question_sha256"] == index["records"][i]["question_sha256"], "optimizer witness identity mismatch")
            learning.replay_update(w, trace["training"][w["method"]]["initial_state"], features[i], targets[i], cfg, arrays, weight, bias)
            checks["optimizer_update_replays"] += 1
    return {"status": "passed", "structural_validation": "sample-trace.v2 project-owned profile passed",
        "semantic_validation": "complete detached arrays, original cached slices, unmasked scores, checkpoint/native AdamW replay passed",
        "checks": checks, "synthetic_only": trace["synthetic_only"], "preflight_only": trace["preflight_only"],
        "real_training_updates": 0 if trace["synthetic_only"] or trace["preflight_only"] else len(learning.ARMS) * ntrain * cfg["training"]["epochs"],
        "claim_ceiling": "engineering evidence only; utility judged separately"}
