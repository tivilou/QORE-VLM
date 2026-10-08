"""Offline audit only: verify saved screen artifacts and diagnose head collapse.

No data retrieval, model loading, ranking changes, fitting to Silver or experiment
rerun. The decay-only witness is a label-free optimizer arithmetic diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
from applications.rag.qarcg_reader import torch_pairwise_rank_loss


def norm(value):
    # Float32 norm underflows for the collapsed parameters; accumulate in float64.
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def stats(value):
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("empty/nonfinite audited values")
    return {"min": float(value.min()), "median": float(np.median(value)),
            "max": float(value.max()), "nonzero": int(np.count_nonzero(value))}


def check_receipts(directory):
    manifest = screen.old._load_json(directory / "upload_manifest.json")
    receipts = []
    for item in manifest["exchange_files"] + [{"name": "upload_manifest.json"}]:
        path = directory / item["name"]
        receipt = screen.old._load_json(directory / (item["name"] + ".upload.json"))
        digest, size = screen.old._sha256(path), path.stat().st_size
        if receipt["sha256"] != digest or receipt["size_bytes"] != size or receipt["status"] != "stored":
            raise ValueError("receipt mismatch: " + item["name"])
        if receipt["path"] != manifest["target_directory"] + "/" + item["name"]:
            raise ValueError("receipt namespace mismatch")
        if "sha256" in item and (item["sha256"] != digest or item["bytes"] != size):
            raise ValueError("manifest mismatch: " + item["name"])
        receipts.append({"name": item["name"], "bytes": size, "sha256": digest})
    return manifest, receipts


def join_cases(detail, trace):
    if len(detail["cases"]) != 100 or len(trace["cases"]) != 100:
        raise ValueError("population mismatch")
    by_number = {int(c["case_number"]): c for c in detail["cases"]}
    if len(by_number) != 100:
        raise ValueError("duplicate detail identity")
    cases, changes = [], []
    for ct in trace["cases"]:
        case = by_number[ct["case_number"]]
        if hashlib.sha256(case["question"].encode()).hexdigest() != ct["question_sha256"]:
            raise ValueError("question identity mismatch")
        ids = [hashlib.sha256(str(p["id"]).encode()).hexdigest() for p in case["top_50"]]
        ranks = [p["retrieved_rank"] for p in case["top_50"]]
        if ids != ct["candidate_id_sha256"] or ranks != ct["candidate_retrieved_ranks"]:
            raise ValueError("candidate identity/order mismatch")
        baseline = ct["methods"]["frozen_reader_topk"]
        base_top = screen.old._stable_top(baseline["score_vector"], case["top_50"])
        for method, row in ct["methods"].items():
            selected = screen.old._stable_top(row["score_vector"], case["top_50"])
            if screen._metric_values(case, selected) != row["posthoc"]:
                raise ValueError("recorded case metric mismatch")
            added, removed = set(selected) - set(base_top), set(base_top) - set(selected)
            if not added:
                continue
            order = sorted(range(50), key=lambda i: (-float(baseline["score_vector"][i]), ranks[i]))
            score_rank = {i: r + 1 for r, i in enumerate(order)}
            change = {"case_number": ct["case_number"], "method": method}
            for role, selected_members in (("added", added), ("removed", removed)):
                change[role] = [{"retrieved_rank": ranks[i], "reader_score_rank": score_rank[i],
                                 "consensus_label": case["top_50"][i]["evidence"]["consensus_label"],
                                 "positive_consensus": case["top_50"][i]["evidence"]["positive_consensus"],
                                 "direct_consensus": case["top_50"][i]["evidence"]["direct_consensus"],
                                 "base_score": baseline["score_vector"][i],
                                 "score_correction": row["score_vector"][i] - baseline["score_vector"][i]}
                                for i in sorted(selected_members, key=lambda j: ranks[j])]
            change["metric_delta"] = {k: row["posthoc"][k] - baseline["posthoc"][k] for k in baseline["posthoc"]}
            changes.append(change)
        cases.append(case)
    return cases, changes


def audit(directory, detail_path, source_revision):
    import torch
    manifest, receipts = check_receipts(directory)
    summary = screen.old._load_json(directory / "summary.json")
    trace = screen.old._load_json(directory / "selector_trace.json")
    detail = screen.old._load_json(detail_path)
    cfg = summary["provenance"]["effective_config"]
    if screen.old._sha256(detail_path) != cfg["evaluation"]["input_sha256"] or trace["input_sha256"] != cfg["evaluation"]["input_sha256"]:
        raise ValueError("registered detail hash mismatch")
    source_matches = {}
    for path, digest in summary["provenance"]["source_hashes"].items():
        raw = subprocess.check_output(["git", "show", source_revision + ":" + path], cwd=ROOT)
        source_matches[path] = hashlib.sha256(raw).hexdigest() == digest
    if not all(source_matches.values()):
        raise ValueError("executed source not recovered")
    for path, field in ((screen.CONFIG, "config_sha256"), (screen.PLAN, "plugin_plan_sha256")):
        raw = subprocess.check_output(["git", "show", source_revision + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != summary["provenance"][field]:
            raise ValueError("config/plan source mismatch")
    cases, changes = join_cases(detail, trace)
    recomputed = screen.diagnose(cases, trace["cases"], cfg)
    if recomputed != summary["evaluation"]:
        raise ValueError("aggregate/paired/bootstrap report mismatch")
    train = trace["training_identity"]
    train_ids = [c["question_sha256"] for c in train]
    if len(train) != summary["training"]["usable"] or len(set(train_ids)) != len(train_ids):
        raise ValueError("usable training identities mismatch")
    if set(train_ids) & {c["question_sha256"] for c in trace["cases"]}:
        raise ValueError("usable training/evaluation overlap")
    for row in train:
        if len(row["positive_mask"]) != 50 or not 0 < sum(row["positive_mask"]) < 50:
            raise ValueError("invalid weak training mask")
    heads = screen.make_heads(torch, 768, cfg, "cpu")
    validation = screen.validate_trace_file(directory / "selector_trace.json", directory / "semantic_values.npz", cfg)
    trajectories, gradients, decay, head_signals = {}, {}, {}, {}
    with np.load(directory / "semantic_values.npz", allow_pickle=False) as arrays:
        validation.update(screen.validate_values(torch, cfg, heads, arrays, trace["population"], "cpu"))
        for method, head in heads.items():
            trajectories[method] = {}
            for epoch in ("initial", "1", "2", "3"):
                trajectories[method][epoch] = {key: norm(arrays[f"checkpoint__{method}__{epoch}__{key}"])
                                               for key in head.state_dict()}
            gradients[method] = []
            for sample in (1, 2):
                for epoch in ("initial", "witness1_" + str(sample), "3"):
                    screen.restore(torch, head, method, epoch, arrays)
                    case = {k: torch.as_tensor(arrays[f"training_{sample:03d}__{k}"])
                            for k in ("base", "pooled", "scalar", "weak_mask")}
                    head.zero_grad(set_to_none=True)
                    output = screen.head_forward(head, method, case, "cpu")
                    loss = torch_pairwise_rank_loss(output[0], case["weak_mask"]) + cfg["training"]["anchor_penalty"] * (output[0] - case["base"]).square().mean()
                    loss.backward()
                    gradients[method].append({"sample": sample, "checkpoint": epoch, "loss": float(loss.detach()),
                                              "parameters": {k: {"task_gradient_l2": norm(v.grad.detach().numpy()),
                                                                  "coupled_decay_gradient_l2": norm(v.detach().numpy()) * cfg["training"]["weight_decay"]}
                                                             for k, v in head.named_parameters()}})
            rows = [c["methods"][method] for c in trace["cases"]]
            encoding = np.concatenate([row["encoded"] for row in rows])
            correction = np.concatenate([np.asarray(c["methods"][method]["score_vector"]) - np.asarray(c["methods"]["frozen_reader_topk"]["score_vector"]) for c in trace["cases"]])
            head_signals[method] = {"gate": stats(np.concatenate([row["gate"] for row in rows])),
                                    "residual": stats(np.concatenate([row["residual"] for row in rows])),
                                    "score_correction": stats(correction), "encoded_std_by_coordinate": encoding.std(0).tolist(),
                                    "changed_sets": sum(set(c["methods"][method]["selected_retrieved_ranks"]) != set(c["methods"]["frozen_reader_topk"]["selected_retrieved_ranks"]) for c in trace["cases"]),
                                    "exact_baseline_score_vectors": sum(c["methods"][method]["score_vector"] == c["methods"]["frozen_reader_topk"]["score_vector"] for c in trace["cases"])}
            if method == "quantum_scalar_control":
                continue
            # Fresh parameter copy; deliberately no task loss, target or data input.
            screen.restore(torch, head, method, "initial", arrays)
            optimizer = torch.optim.Adam(head.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
            decay[method] = []
            for epoch in range(1, cfg["training"]["epochs"] + 1):
                for _ in range(len(train)):
                    for parameter in head.parameters():
                        parameter.grad = torch.zeros_like(parameter)
                    optimizer.step()
                decay[method].append({"epoch": epoch,
                                      "pure_decay_projection_l2": norm(head.project.weight.detach().numpy()),
                                      "actual_projection_l2": trajectories[method][str(epoch)]["project.weight"],
                                      "pure_decay_gate_bias": float(head.interaction.gate_bias.detach()),
                                      "actual_gate_bias": float(arrays[f"checkpoint__{method}__{epoch}__interaction.gate_bias"])})
    return {"schema_version": "rag.quantum_semantic_reader_audit.v1", "run_id": directory.name,
            "artifact_namespace": manifest["target_directory"], "validity_status": "mixed",
            "evidence_tier": "L0_diagnostic", "candidate_status": "blocked_training_repair",
            "receipts": receipts, "source_recovery": {"reported_revision": summary["provenance"]["code_revision"],
                                                       "matching_committed_file_revision": source_revision, "source_matches": source_matches,
                                                       "note": "Mask fix executed before being committed; hashes match the later fix commit."},
            "validation": validation, "evaluation": recomputed, "training": summary["training"],
            "parameter_trajectories_l2_float64": trajectories, "gradient_probes": gradients,
            "decay_only_probe": decay, "head_signals": head_signals, "changes": changes,
            "case_selection": {"training_probes": "first two usable training cases, as declared by runner; initial/first-epoch witness/final checkpoints",
                               "evaluation": "all 100 cases; every changed scalar set shown separately as exploratory, outcome-selected"},
            "limitations": ["Pure decay reproduces shrinkage; complete per-step task gradients and optimizer state are absent, so it is not a full optimizer replay.",
                            "Only two usable training cases have saved full semantic inputs; skipped training hashes are absent.",
                            "Semantic scores remain exactly baseline: this run does not test a successfully trained semantic intervention.",
                            "No model weights, dataset, labels or online scoring were changed; no real-data experiment was rerun."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--source-revision", default="44c9f7027a386b759ba4dfdfca703e09e87125a8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir, args.detail, args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "validation": report["validation"], "head_signals": report["head_signals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
