"""Offline repair-result audit, with all-case posthoc and learning diagnostics.

No retrieval/model loading, fitting, new scoring policy or task experiment.
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
from scripts.collab.five_ideas import analyze_quantum_semantic_reader_screen as helper
from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
from scripts.collab.five_ideas.run_semantic_reader_training_repair_100 import CONFIG, PLAN, validate_repair
from applications.rag.semantic_reader_training import gradient_health, norm, require_learning, signal_health
from applications.rag.qarcg_reader import torch_pairwise_rank_loss


def describe(values):
    value = np.asarray(values, dtype=np.float64)
    if not value.size or not np.isfinite(value).all():
        raise ValueError("empty/nonfinite statistic")
    return {"min": float(value.min()), "median": float(np.median(value)), "mean": float(value.mean()),
            "max": float(value.max()), "nonzero": int(np.count_nonzero(value))}


def mobility(base, scores, ranks):
    """Label-free boundary bound: range(delta) < score5-score6 forbids swaps."""
    base, scores = np.asarray(base, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    order = sorted(range(50), key=lambda i: (-base[i], ranks[i]))
    gap = float(base[order[4]] - base[order[5]])
    spread = float(np.ptp(scores - base))
    changed = set(order[:5]) != set(sorted(range(50), key=lambda i: (-scores[i], ranks[i]))[:5])
    if spread < gap and changed:
        raise ValueError("boundary-range invariant violated")
    return {"rank5_rank6_gap": gap, "correction_range": spread, "changed_top5": changed,
            "range_less_than_boundary_gap": spread < gap}


def evidence_rescue_diagnostics(cases, trace, residual_bound):
    """Exploratory Silver-labelled rescue analysis; never an online input."""
    result = {"case_positive_capacity": 0, "reader_positive_total": 0, "methods": {}}
    for method in ("quantum_semantic", "classical_semantic", "quantum_scalar_control"):
        bands = {name: {"missing_positive_candidates": 0, "direction_improved": 0,
                        "overtook_best_removable_negative": 0, "actually_selected": 0,
                        "outside_architectural_residual_bound": 0,
                        "same_rescue_case_nonpositive_candidates": 0, "nonpositive_direction_improved": 0}
                 for name in ("6_10", "11_20", "21_50")}
        target_cases, no_removable, correlations = 0, 0, []
        for case, ct in zip(cases, trace):
            candidates = case["top_50"]
            base = np.asarray(ct["methods"]["frozen_reader_topk"]["score_vector"])
            scores = np.asarray(ct["methods"][method]["score_vector"])
            if base.std() > 0 and (scores - base).std() > 0:
                correlations.append(float(np.corrcoef(base, scores - base)[0, 1]))
            labels = np.asarray([p["evidence"]["positive_consensus"] for p in candidates])
            order = screen.old._stable_top(base, candidates)
            final = set(screen.old._stable_top(scores, candidates))
            all_order = sorted(range(50), key=lambda i: (-base[i], candidates[i]["retrieved_rank"]))
            ranks = {i: n + 1 for n, i in enumerate(all_order)}
            if method == "quantum_semantic":
                result["case_positive_capacity"] += min(5, int(labels.sum()))
                result["reader_positive_total"] += int(labels[order].sum())
            removable = [i for i in order if not labels[i]]
            missing = [i for i in range(50) if labels[i] and i not in order]
            if not removable:
                no_removable += 1
                continue
            if missing:
                target_cases += 1
            weakest = min(removable, key=lambda i: (base[i], -candidates[i]["retrieved_rank"]))
            if missing:
                for i in range(50):
                    if i in order or labels[i]:
                        continue
                    band = "6_10" if ranks[i] <= 10 else "11_20" if ranks[i] <= 20 else "21_50"
                    bands[band]["same_rescue_case_nonpositive_candidates"] += 1
                    bands[band]["nonpositive_direction_improved"] += int((scores[i] - base[i]) > (scores[weakest] - base[weakest]))
            for i in missing:
                band = "6_10" if ranks[i] <= 10 else "11_20" if ranks[i] <= 20 else "21_50"
                row = bands[band]
                row["missing_positive_candidates"] += 1
                row["direction_improved"] += int((scores[i] - base[i]) > (scores[weakest] - base[weakest]))
                row["overtook_best_removable_negative"] += int(scores[i] > scores[weakest])
                row["actually_selected"] += int(i in final)
                # Max possible residual difference before the gate: 2*rho*std(base).
                # Necessary condition only, not a claim of feasible simultaneous rescue.
                max_difference = 2 * residual_bound * max(float(base.std()), 1e-6)
                row["outside_architectural_residual_bound"] += int(base[weakest] - base[i] > max_difference)
        result["methods"][method] = {"cases_with_available_rescue": target_cases,
                                     "cases_with_no_negative_top5_slot": no_removable, "score_rank_bands": bands,
                                     "within_case_base_correction_pearson": describe(correlations)}
    result["reader_missing_positive_capacity"] = result["case_positive_capacity"] - result["reader_positive_total"]
    result["boundary"] = "Posthoc exposed Silver only; missing candidates are not independent units; counts are not a significance test."
    return result


def audit(directory, detail_path, revision, prior_directory=None):
    import torch
    torch.set_num_threads(1)
    manifest, receipts = helper.check_receipts(directory)
    cfg = validate_repair()
    summary = screen.old._load_json(directory / "summary.json")
    trace = screen.old._load_json(directory / "selector_trace.json")
    detail = screen.old._load_json(detail_path)
    provenance = summary["provenance"]
    if provenance["effective_config"] != cfg or provenance != trace["provenance"] or provenance != manifest["provenance"]:
        raise ValueError("configuration/provenance mismatch")
    if screen.old._load_json(directory / "run_metadata.json")["provenance"] != provenance:
        raise ValueError("metadata provenance mismatch")
    if screen.old._sha256(detail_path) != cfg["evaluation"]["input_sha256"] or trace["input_sha256"] != provenance["input_sha256"]:
        raise ValueError("fixed detail identity mismatch")
    sources = {}
    for path, expected in provenance["source_hashes"].items():
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=ROOT)
        sources[path] = hashlib.sha256(raw).hexdigest() == expected
    for path, field in ((CONFIG, "config_sha256"), (PLAN, "plugin_plan_sha256")):
        raw = subprocess.check_output(["git", "show", revision + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest() == provenance[field]
    if not all(sources.values()):
        raise ValueError("executed source bytes mismatch")
    cases, changes = helper.join_cases(detail, trace)
    evaluation = screen.diagnose(cases, trace["cases"], cfg)
    if evaluation != summary["evaluation"]:
        raise ValueError("all-case metrics/bootstrap mismatch")
    train = trace["training_identity"]
    train_ids = [c["question_sha256"] for c in train]
    if len(train_ids) != summary["training"]["usable"] or len(set(train_ids)) != len(train_ids) or set(train_ids) & {c["question_sha256"] for c in trace["cases"]}:
        raise ValueError("usable train identity/heldout split mismatch")
    if any(len(c["positive_mask"]) != 50 or not 0 < sum(c["positive_mask"]) < 50 for c in train):
        raise ValueError("mixed weak-target contract violation")
    archive_path = directory / "semantic_values.npz"
    gradients, witnesses, signals, boundary, trajectories = {}, {}, {}, {}, {}
    with np.load(archive_path, allow_pickle=False) as arrays:
        heads = screen.make_heads(torch, arrays["training_001__pooled"].shape[1] // 5, cfg, "cpu")
        validation = screen.validate_values(torch, cfg, heads, arrays, trace["population"], "cpu")
        validation.update(screen.validate_trace_file(directory / "selector_trace.json", archive_path, cfg))
        for method, head in heads.items():
            rows = summary["training"]["arms"][method]
            history = arrays[f"health__{method}__data_gradient_norms"]
            if history.shape != (len(train) * cfg["training"]["epochs"], 4) or not np.isfinite(history).all() or (history < 0).any():
                raise ValueError("per-step data-gradient coverage invalid")
            columns = rows["training_health"]["gradient_columns"]
            gradients[method] = {name: describe(history[:, i]) for i, name in enumerate(columns)}
            if [x["step"] for x in rows["training_health"]["checks"]] != [64, 434, 868, 1302]:
                raise ValueError("health-check schedule mismatch")
            previous = 0
            for check in rows["training_health"]["checks"]:
                np.testing.assert_allclose(history[previous:check["step"]].sum(0),
                    [check["data_gradient_norm_totals"][k] for k in columns], rtol=1e-12, atol=1e-12)
                require_learning(check, check["data_gradient_norm_totals"], semantic=hasattr(head, "project"))
                previous = check["step"]
            trajectories[method] = {}
            for epoch in ("initial", "1", "2", "3"):
                screen.restore(torch, head, method, epoch, arrays)
                inputs = [{k: torch.from_numpy(arrays[f"training_{i:03d}__{k}"].copy())
                           for k in ("base", "pooled", "scalar", "weak_mask")} for i in (1, 2)]
                outputs = [screen.head_forward(head, method, c, "cpu") for c in inputs]
                initial_norm = (helper.norm(arrays[f"checkpoint__{method}__initial__project.weight"])
                                if hasattr(head, "project") else None)
                measured = signal_health(head, outputs, [c["base"] for c in inputs], initial_norm)
                if epoch != "initial":
                    check = rows["training_health"]["checks"][int(epoch)]
                    for field in ("projection_norm", "projection_norm_ratio", "mean_encoding_std",
                                  "max_abs_effective_correction", "max_normalized_correction_std"):
                        if check[field] is not None:
                            np.testing.assert_allclose(measured[field], check[field], rtol=2e-3, atol=2e-6)
                trajectories[method][epoch] = {**measured,
                    "circuit_norm": norm(head.interaction.circuit if hasattr(head, "interaction") else head.circuit)}
            witnesses[method] = []
            for witness in rows["witnesses"]:
                index, epoch = witness["sample"], witness["epoch"]
                screen.restore(torch, head, method, f"witness{epoch}_{index}", arrays)
                case = {k: torch.from_numpy(arrays[f"training_{index:03d}__{k}"].copy()) for k in ("base", "pooled", "scalar", "weak_mask")}
                head.zero_grad(set_to_none=True)
                out = screen.head_forward(head, method, case, "cpu")
                ranking = torch_pairwise_rank_loss(out[0], case["weak_mask"])
                anchor = (out[0] - case["base"]).square().mean()
                loss = ranking + cfg["training"]["anchor_penalty"] * anchor
                loss.backward()
                np.testing.assert_allclose(float(loss.detach()), witness["loss"], rtol=2e-3, atol=2e-6)
                witnesses[method].append({"sample": index, "epoch": epoch, "weak_positive_count": int(case["weak_mask"].sum()),
                    "rank_loss": float(ranking.detach()), "anchor_mse": float(anchor.detach()), "total_loss": float(loss.detach()),
                    "reader_rank_loss": float(torch_pairwise_rank_loss(case["base"], case["weak_mask"])),
                    "data_gradients": gradient_health(head)})
            collected = {k: [] for k in ("correction", "gate", "residual", "encoded", "encoding_std", "correction_std", "saturation")}
            boundary[method] = []
            for ct in trace["cases"]:
                base = np.asarray(ct["methods"]["frozen_reader_topk"]["score_vector"])
                mt = ct["methods"][method]
                correction = np.asarray(mt["score_vector"]) - base
                for k in ("gate", "residual", "encoded"):
                    collected[k].extend(np.asarray(mt[k]).reshape(-1))
                collected["correction"].extend(correction)
                collected["encoding_std"].append(float(np.asarray(mt["encoded"]).std(0).mean()))
                collected["correction_std"].append(float(correction.std()))
                collected["saturation"].append(float((np.abs(mt["encoded"]) > .99).mean()))
                boundary[method].append({"case_number": ct["case_number"], **mobility(base, mt["score_vector"], ct["candidate_retrieved_ranks"])})
            signals[method] = {k: describe(v) for k, v in collected.items()}
            signals[method]["changed_top5_sets"] = sum(x["changed_top5"] for x in boundary[method])
            signals[method]["range_too_small_to_cross_boundary"] = sum(x["range_less_than_boundary_gap"] for x in boundary[method])
    parity = {"status": "not_checked"}
    if prior_directory:
        prior = screen.old._load_json(prior_directory / "selector_trace.json")
        parity = {"usable_train_identities_exact": train == prior["training_identity"], "reader_score_vectors_exact": 0}
        for a, b in zip(trace["cases"], prior["cases"]):
            if a["question_sha256"] != b["question_sha256"] or a["candidate_id_sha256"] != b["candidate_id_sha256"]:
                raise ValueError("prior candidate parity mismatch")
            parity["reader_score_vectors_exact"] += int(a["methods"]["frozen_reader_topk"]["score_vector"] == b["methods"]["frozen_reader_topk"]["score_vector"])
        with np.load(prior_directory / "semantic_values.npz", allow_pickle=False) as a, np.load(archive_path, allow_pickle=False) as b:
            for i in (1, 2):
                for field in ("base", "pooled", "scalar", "weak_mask"):
                    np.testing.assert_array_equal(a[f"training_{i:03d}__{field}"], b[f"training_{i:03d}__{field}"])
        if parity["reader_score_vectors_exact"] != 100 or not parity["usable_train_identities_exact"]:
            raise ValueError("prior Reader/training cohort not matched")
        parity["first_two_train_inputs_exact"] = True
    by_number = {c["case_number"]: c for c in cases}
    case_details = []
    for change in changes:
        case = by_number[change["case_number"]]
        fields = {p["retrieved_rank"]: p for p in case["top_50"]}
        row = {**change, "question": case["question"]}
        for role in ("added", "removed"):
            row[role] = [{**p, "title": fields[p["retrieved_rank"]].get("title"),
                          "text": fields[p["retrieved_rank"]].get("text")} for p in change[role]]
        case_details.append(row)
    return {"schema_version": "rag.semantic_reader_training_repair_audit.v1", "run_id": directory.name,
            "source_revision": revision, "producer_revision": provenance["code_revision"], "receipts": receipts,
            "source_matches": sources, "validation": validation, "fixed_input_sha256": provenance["input_sha256"],
            "training_count": len(train), "data_gradient_statistics": gradients,
            "parameter_signal_trajectories": trajectories, "training_witnesses": witnesses,
            "evaluation": evaluation, "evaluation_signal_statistics": signals, "boundary_diagnostics": boundary,
            "posthoc_rescue_diagnostics": evidence_rescue_diagnostics(cases, trace["cases"], cfg["head"]["residual_bound"]),
            "all_changed_cases": changes, "prior_parity": parity, "case_details": case_details,
            "task_experiment_run": False, "limitations": ["Exposed Silver panel and short screen: L0 only.",
            "Two saved training cases cannot certify weak supervision quality.",
            "Other 432 cases lack full inputs/optimizer replay; CPU-vs-GPU replay uses tolerances."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--detail", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--prior-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.run_dir, args.detail, args.source_revision, args.prior_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case_details = report.pop("case_details")
    screen.old._write_json(args.output.with_name(args.output.stem + "_case_details.json"), case_details)
    screen.old._write_json(args.output, report)
    if screen.old._load_json(args.output) != report:
        raise ValueError("audit report readback mismatch")
    print(json.dumps({"output": str(args.output), "validation": report["validation"],
                      "signals": report["evaluation_signal_statistics"], "evaluation": report["evaluation"]}))
