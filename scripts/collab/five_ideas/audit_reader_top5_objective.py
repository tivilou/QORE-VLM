#!/usr/bin/env python3
"""CPU-only frozen59 train-cache objective checks. No fitting/model/network."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from applications.rag.protected_boundary_learning import validate_row
from applications.rag.quantum_reader_rotation import top5_retention_objective
from applications.rag.scale_invariant_top5_objective import (
    OBJECTIVE_ID, standardized_scores, standardized_top5_objective, membership_diagnostics,
)
from scripts.collab.lib.source_snapshot import archive_sources, verify_sources

CONFIG = ROOT / "configs/experiments/reader_top5_objective_audit.json"
SOURCES = ("configs/experiments/reader_top5_objective_audit.json",
    "scripts/collab/five_ideas/audit_reader_top5_objective.py",
    "applications/rag/scale_invariant_top5_objective.py", "applications/rag/protected_boundary_learning.py",
    "applications/rag/quantum_reader_rotation.py", "scripts/collab/lib/source_snapshot.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def load_inputs(directory, cfg):
    documents = {}
    for name, expected in cfg["inputs"].items():
        path = Path(directory) / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("frozen input SHA mismatch: " + name)
        documents[name] = json.loads(path.read_text(encoding="utf8"))
    # The raw source also contains evaluation records. Deliberately use only
    # training_cases; neither evaluation selections nor posthoc labels enter.
    records = documents["selector_trace.json"]["training_cases"]
    rows = documents["training_targets.private.json"]["rows"]
    if len(records) != len(rows) or len(rows) != cfg["population"]["training_questions"]:
        raise ValueError("frozen training population mismatch")
    seen = set()
    for record, row in zip(records, rows):
        target, case = row["target"], row["case"]
        validate_row(target)
        qid = hashlib.sha256(case["question"].encode()).hexdigest()
        if qid in seen or qid != record["question_sha256"] or case["future_role"] != "training":
            raise ValueError("training identity/split mismatch")
        seen.add(qid)
        ids = [hashlib.sha256(str(p["id"]).encode()).hexdigest() for p in case["candidates"]]
        if ids != record["candidate_id_sha256"] or len(ids) != 50:
            raise ValueError("training candidate identity mismatch")
        expected_order = target["baseline_top5_indices"]
        base = record["methods"]["frozen_reader_topk"]
        if base["selected_indices"] != expected_order:
            raise ValueError("target baseline mismatch")
        for method in cfg["methods"]:
            item = record["methods"][method]
            values = item["score_vector"]
            if len(values) != 50 or not np.isfinite(values).all():
                raise ValueError("nonfinite or incomplete score vector")
            selected = sorted(range(50), key=lambda i: (-values[i], int(case["candidates"][i]["retrieved_rank"]), str(case["candidates"][i]["id"])))[:5]
            if selected != item["selected_indices"]:
                raise ValueError("saved selection not produced by saved scores")
    counts = {"training_questions": len(rows), "pairs": sum(len(r["target"]["draft_pairs"]) for r in rows),
        "eligible_candidates": sum(len(r["target"]["eligible_positive_indices"]) for r in rows),
        "protected_candidates": sum(len(r["target"]["immutable_selected_indices"]) for r in rows),
        "maximum_slots": sum(r["target"]["maximum_replacement_slots"] for r in rows)}
    if counts != cfg["population"]: raise ValueError("frozen support counts changed")
    return rows, records, counts


def run(directory, output_root):
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    if cfg["objective_id"] != OBJECTIVE_ID: raise ValueError("objective identity changed")
    rows, records, counts = load_inputs(directory, cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    sources = archive_sources(ROOT, SOURCES, output / "executed_sources.zip")
    settings = cfg["objective"]
    arrays, cases, means = {}, [], {m: {"new_total": [], "old_total": [], "rescued_eligible": 0, "lost_protected": 0} for m in cfg["methods"]}
    max_error, collapse_rejections, outlier_checks = 0., 0, 0
    for number, (row, record) in enumerate(zip(rows, records), 1):
        base = torch.tensor(record["methods"]["frozen_reader_topk"]["score_vector"], dtype=torch.float64)
        target = row["target"]
        extreme = base.clone()
        extreme[int(torch.argmin(base))] -= 100 * base.std(unbiased=False)
        original_loss = standardized_top5_objective(base, base, target, **settings)["total"]
        extreme_loss = standardized_top5_objective(extreme, base, target, **settings)["total"]
        if float(extreme_loss) < float(original_loss) - 1e-8:
            raise ValueError("irrelevant extreme score buys a loss decrease")
        outlier_checks += 1
        sample = {"training_case_number": number, "question_sha256": record["question_sha256"], "methods": {}}
        for method in cfg["methods"]:
            scores = torch.tensor(record["methods"][method]["score_vector"], dtype=torch.float64, requires_grad=True)
            parts = standardized_top5_objective(scores, base, target, **settings)
            parts["total"].backward()
            if not bool(scores.grad.isfinite().all()): raise ValueError("nonfinite score gradient")
            errors = []
            for check in cfg["affine_checks"]:
                shifted = standardized_top5_objective(check["scale"] * scores.detach() + check["shift"], base, target, **settings)
                errors.append(max(abs(float(shifted[key] - parts[key].detach())) for key in parts))
            max_error = max(max_error, *errors)
            if max(errors) > 1e-8: raise ValueError("affine compression changes objective")
            try:
                standardized_top5_objective(torch.ones_like(scores), base, target, **settings)
            except ValueError:
                collapse_rejections += 1
            else:
                raise ValueError("constant scores accepted")
            diag = membership_diagnostics(scores.detach(), target)
            loss = {key: float(value.detach()) for key, value in parts.items()}
            old = float(top5_retention_objective(scores, base, target)["total"].detach())
            summary = means[method]
            summary["new_total"].append(loss["total"]); summary["old_total"].append(old)
            summary["rescued_eligible"] += diag["rescued_eligible"]
            summary["lost_protected"] += diag["lost_protected"]
            item = {"objective": loss, "historical_retention_objective": old, "membership": diag,
                    "max_affine_error": max(errors), "score_std": float(scores.detach().std(unbiased=False))}
            if number <= 2:
                prefix = f"training_{number:03d}__{method}"
                for key, value in (("scores", scores.detach()), ("base", base), ("normalized_scores", standardized_scores(scores).detach()), ("total_loss_gradient", scores.grad)):
                    arrays[prefix + "__" + key] = value.numpy().copy()
                item["values_prefix"] = prefix
                item["target"] = {k: target[k] for k in ("baseline_top5_indices", "eligible_positive_indices", "immutable_selected_indices", "maximum_replacement_slots")}
            sample["methods"][method] = item
        cases.append(sample)
    np.savez_compressed(output / "objective_values.npz", **arrays)
    # Reopen emitted bytes and independently repeat every selected score-gradient
    # calculation, without cached autograd graphs or saved gradient substitution.
    replayed = 0
    with np.load(output / "objective_values.npz", allow_pickle=False) as reopened:
        for number in (1, 2):
            for method in cfg["methods"]:
                prefix = f"training_{number:03d}__{method}"
                scores = torch.tensor(reopened[prefix + "__scores"], requires_grad=True)
                base = torch.tensor(reopened[prefix + "__base"])
                parts = standardized_top5_objective(scores, base, rows[number - 1]["target"], **settings)
                parts["total"].backward()
                np.testing.assert_allclose(scores.grad.numpy(), reopened[prefix + "__total_loss_gradient"], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(standardized_scores(scores).detach().numpy(), reopened[prefix + "__normalized_scores"], rtol=1e-10, atol=1e-10)
                replayed += 1
    verify_sources(ROOT, sources)
    summary = {"schema_version": "rag.reader_top5_objective_audit.summary.v1", "run_id": run_id,
        "status": "passed", "claim_ceiling": cfg["claim_ceiling"], "objective_id": OBJECTIVE_ID, "population": counts,
        "methods": {m: {"mean_candidate_objective": float(np.mean(v["new_total"])), "mean_historical_retention_objective": float(np.mean(v["old_total"])), "rescued_eligible": v["rescued_eligible"], "lost_protected": v["lost_protected"]} for m, v in means.items()},
        "checks": {"affine_checks": 59 * len(cfg["methods"]) * len(cfg["affine_checks"]), "max_affine_error": max_error, "constant_rejections": collapse_rejections, "outlier_checks": outlier_checks, "saved_value_arrays": len(arrays), "score_gradient_replays": replayed},
        "constraints": cfg["constraints"], "source_inputs": cfg["inputs"], "source_snapshot": sources,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
        "limitations": cfg["limitations"] + ["Historical checkpoints rescored diagnostically; losses have different definitions and their magnitudes are not utility comparisons.", "No optimizer/head training update was performed; gradient replay is with respect to candidate scores only."]}
    trace = {"schema_version": "rag.reader_top5_objective_audit.trace.v1", "stages": ["frozen_train_inputs", "score_standardization", "objective", "score_gradient", "discrete_membership_audit"],
        "sample_rule": cfg["selected_value_capture"], "training_cases": cases, "values_sha256": digest(output / "objective_values.npz"), "constraints": cfg["constraints"]}
    write(output / "objective_trace.json", trace)
    write(output / "summary.json", summary)
    write(output / "manifest.json", {"schema_version": "rag.reader_top5_objective_audit.manifest.v1", "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(output.iterdir()) if p.is_file()], "formal_training_ready": False})
    print(json.dumps({"output_dir": str(output), "status": "passed", "checks": summary["checks"], "methods": summary["methods"], "formal_training_ready": False}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / ".cache/quantum_reader_rotation_result_audit/20261015T022004545902Z")
    parser.add_argument("--output-root", type=Path, default=ROOT / ".cache/reader_top5_objective_audit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.run_dir, args.output_root)


if __name__ == "__main__":
    main()
