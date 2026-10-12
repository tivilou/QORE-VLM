#!/usr/bin/env python3
"""Train matched fresh heads on frozen59/228 targets; label-free 100-case L0 screen."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.collab.five_ideas import run_qarcg_reader_screen_100 as old
from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as metrics
from applications.rag.protected_boundary_inputs import load_bundle
from applications.rag import protected_boundary_training as training
from applications.rag.quantum_semantic_reader import reader_semantic_forward, circuit_resource_contract
from scripts.collab.lib.exchange_upload import download_exchange_file, upload_manifest, _upload_one

CONFIG = ROOT / "configs/experiments/protected_boundary_reader_screen_100.json"
METHODS = ("frozen_reader_topk", *training.ARMS)
STAGES = ("data", "frozen_reader", "semantic_pooling", "residual_head", "optimization", "selection", "posthoc")


def validate_config(path):
    cfg = old._load_json(path)
    if cfg != old._load_json(CONFIG) or cfg["authorization"] != "collaborator_screen" or tuple(cfg["methods"]) != METHODS:
        raise ValueError("unapproved config/method combination")
    if cfg["training"]["cases"] != 59 or cfg["training"]["pairs"] != 228 or cfg["evaluation"]["cases"] != 100:
        raise ValueError("frozen population changed")
    return cfg


def get_bundle(cfg, explicit, args):
    spec = cfg["training"]["input"]
    path = explicit or ROOT / ".cache/protected_boundary_reader_inputs/20261011-v1/frozen_support_bundle.zip"
    if not path.exists():
        download_exchange_file(spec["exchange_path"], path, base_url=args.exchange_url, token_env=args.token_env,
                               expected_size=spec["bytes"], expected_sha256=spec["sha256"])
    if path.is_symlink() or path.stat().st_size != spec["bytes"] or old._sha256(path) != spec["sha256"]:
        raise ValueError("frozen bundle size/hash mismatch")
    return path


def load_reader(cfg, args):
    import torch
    from transformers import DPRReader, DPRReaderTokenizer
    spec = cfg["reader"]
    options = {"revision": spec["revision"], "local_files_only": args.offline_model}
    tokenizer = DPRReaderTokenizer.from_pretrained(spec["model_id"], **options)
    reader = DPRReader.from_pretrained(spec["model_id"], **options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reader.to(device).eval().requires_grad_(False)
    identity = old._reader_identity(reader, tokenizer, spec["model_id"], spec["revision"], device)
    if identity["resolved_revision"] != spec["revision"]:
        raise ValueError("Reader revision drift")
    return torch, tokenizer, reader, device, identity


def feature_case(torch, tokenizer, reader, device, cfg, case, prefix, capture, arrays):
    # Explicit evidence-free projection: answers/annotations never enter Reader/head.
    candidates = case.get("candidates", case.get("top_50"))
    texts = [old._online_text({k: p.get(k) for k in ("title", "text")}) for p in candidates]
    base, pooled, scalar, hidden = reader_semantic_forward(torch, tokenizer, reader, device, case["question"], texts,
        max_length=cfg["reader"]["max_length"], batch_size=cfg["reader"]["batch_size"], capture_first=capture)
    features = {"base": base, "pooled": pooled, "scalar": scalar}
    for key, value in features.items(): arrays[prefix + "__" + key] = value.numpy().copy()
    for key, value in (hidden or {}).items(): arrays[prefix + "__" + key] = np.array(value, copy=True)
    if hidden is not None:
        classifier = reader.span_predictor.qa_classifier
        arrays[prefix + "__relevance_weight"] = classifier.weight.detach().cpu().numpy().copy()
        arrays[prefix + "__relevance_bias"] = classifier.bias.detach().cpu().numpy().copy()
    if "reader_top5_indices" in case:
        expected = np.asarray([p["reader_score"] for p in candidates], dtype=np.float32)
        if not np.allclose(base.numpy(), expected, rtol=1e-4, atol=1e-4):
            raise ValueError("frozen training Reader score parity failed")
        if old._stable_top(base.numpy(), candidates) != case["reader_top5_indices"]:
            raise ValueError("frozen training Top5 parity failed")
    return features


def rank_cases(torch, heads, features, cases, lane, cfg, arrays):
    traces = []
    for number, (f, case) in enumerate(zip(features, cases), 1):
        candidates = case.get("candidates", case.get("top_50"))
        prefix = f"{lane}_{number:03d}"
        record = {"case_number": number, "question_sha256": hashlib.sha256(case["question"].encode()).hexdigest(),
                  "candidate_retrieved_ranks": [p["retrieved_rank"] for p in candidates],
                  "candidate_id_sha256": [hashlib.sha256(str(p["id"]).encode()).hexdigest() for p in candidates], "methods": {}}
        for method in METHODS:
            if method == "frozen_reader_topk":
                scores = f["base"].numpy().copy()
                arrays[prefix + "__" + method + "__scores"] = scores
            else:
                head = heads[method]; device = next(head.parameters()).device
                output = training.capture_output(head, f["base"].to(device), f["pooled"].to(device), arrays, prefix + "__" + method)
                scores = output[0].detach().cpu().numpy()
            indices = old._stable_top(scores, candidates)
            record["methods"][method] = {"score_vector": scores.tolist(), "selected_indices": indices,
                "selected_retrieved_ranks": [candidates[i]["retrieved_rank"] for i in indices],
                "selected_id_sha256": [record["candidate_id_sha256"][i] for i in indices]}
        traces.append(record)
    return traces


def diagnose(cases, trace, cfg):
    fields = ("positive_consensus_count", "direct_consensus_count", "all_models_positive_count", "fixed_silver_set_overlap")
    values = {m: {key: [] for key in fields} for m in METHODS}
    for case, record in zip(cases, trace):
        for method in METHODS:
            row = record["methods"][method]
            measured = metrics._metric_values(case, row["selected_indices"])
            row["posthoc"] = measured
            for key, value in measured.items(): values[method][key].append(value)
    comparisons = {reference: {key: metrics.paired_delta(values["quantum_semantic"][key], values[reference][key],
        cfg["evaluation"]["bootstrap_seed"], cfg["evaluation"]["bootstrap_replicates"]) for key in fields}
        for reference in ("frozen_reader_topk", "classical_semantic")}
    utility = comparisons["frozen_reader_topk"][fields[0]]
    protection = comparisons["frozen_reader_topk"][fields[1]]
    attribution = comparisons["classical_semantic"][fields[0]]
    return {"methods": {m: {k: {"total": sum(v), "mean": float(np.mean(v))} for k, v in fields_.items()} for m, fields_ in values.items()},
        "paired_quantum_minus": comparisons, "gates": {
            "selection_diagnostic": utility["mean"] > 0 and utility["ci95"][0] > 0 and protection["mean"] >= 0,
            "quantum_attribution_diagnostic": attribution["mean"] > 0 and attribution["ci95"][0] > 0},
        "population": "all100 questions, including zero-positive; exposed Silver L0 only"}


def training_diagnosis(rows, trace, cfg):
    result = {}
    for method in METHODS:
        out = {"rescued_qualified_direct": 0, "lost_qualified_direct": 0, "lost_immutable_slots": 0,
               "improved_pair_gaps": 0, "pair_inversions": 0, "loose_reachable_pairs": 0,
               "positive_minus_matched_irrelevant_correction": [], "matched_controls": 0}
        for row, record in zip(rows, trace):
            case, target, annotation = row["case"], row["target"], row["annotation"]
            base = np.asarray([p["reader_score"] for p in case["candidates"]])
            scores = np.asarray(record["methods"][method]["score_vector"])
            delta = scores - base
            selected = set(record["methods"][method]["selected_indices"])
            out["rescued_qualified_direct"] += len(selected & set(target["eligible_positive_indices"]))
            out["lost_qualified_direct"] += len(set(target["protected_indices"]) - selected)
            out["lost_immutable_slots"] += len(set(target["immutable_selected_indices"]) - selected)
            for positive, negative in target["draft_pairs"]:
                out["improved_pair_gaps"] += int(scores[positive] - scores[negative] > base[positive] - base[negative] + 1e-7)
                out["pair_inversions"] += int(scores[positive] > scores[negative])
                out["loose_reachable_pairs"] += int(base[negative] - base[positive] < 2 * cfg["head"]["residual_bound"] * max(base.std(), 1e-6))
            by_id = {p["review_item_id"]: p for p in annotation["items"]}
            # Post-fit diagnostic controls only: original primary irrelevant opinions, not new negatives.
            irrelevant = [i for i, p in enumerate(case["candidates"]) if by_id[p["review_item_id"]]["question_support"] == "irrelevant"]
            for positive in target["eligible_positive_indices"]:
                if irrelevant:
                    control = min(irrelevant, key=lambda i: (abs(base[i] - base[positive]), i))
                    out["positive_minus_matched_irrelevant_correction"].append(float(delta[positive] - delta[control]))
                    out["matched_controls"] += 1
        contrast = out.pop("positive_minus_matched_irrelevant_correction")
        out["mean_positive_minus_score_matched_primary_irrelevant_correction"] = float(np.mean(contrast)) if contrast else None
        out["control_limit"] = "posthoc nearest base score, reused with replacement; primary opinions, not independent gold"
        result[method] = out
    return result


def provenance(cfg, identity, args):
    files = [Path(__file__), ROOT / "applications/rag/protected_boundary_inputs.py",
             ROOT / "applications/rag/protected_boundary_training.py", ROOT / "applications/rag/protected_boundary_learning.py",
             ROOT / "applications/rag/quantum_semantic_reader.py", ROOT / "applications/rag/qarcg_reader.py",
             ROOT / "applications/rag/semantic_reader_training.py", ROOT / "scripts/collab/lib/exchange_upload.py",
             ROOT / "scripts/collab/lib/protected_boundary_trace.py", ROOT / "applications/rag/support_supervision_axes.py",
             ROOT / "applications/rag/support_boundary_bridge.py", Path(old.__file__), Path(metrics.__file__)]
    import torch
    import transformers
    relative = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files]
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *relative], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"code_revision": old._git_revision() or "unavailable", "config_sha256": old._sha256(args.config),
            "plugin_plan_sha256": old._sha256(ROOT / "configs/experiments/protected_boundary_reader_screen_100_plan.json"),
            "code_dirty_paths": [line[3:] for line in dirty.stdout.splitlines()],
            "reader": identity, "effective_config": cfg, "python": sys.version.split()[0],
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "source_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): old._sha256(p) for p in files}}


def fixture(torch, tokenizer, reader, device, cfg, destination, identity, args):
    """Actual Reader values, synthetic targets; never fit real questions in preflight."""
    from scripts.collab.lib.protected_boundary_trace import persist_trace
    arrays, metadata, features, cases, targets = {}, [], [], [], []
    fixture_cfg = json.loads(json.dumps(cfg)); fixture_cfg["training"]["epochs"] = 1
    for lane in ("training", "evaluation"):
        for number in (1, 2):
            case = {"question": f"For fixture {lane} {number}, which city is named as the venue?", "answers": ["London"], "top_50": [
                {"id": str(i), "retrieved_rank": i + 1, "title": "Synthetic venue fixture",
                 "text": f"The event {i} took place in {'London' if i % 2 else 'Paris'}. This is a synthetic software fixture."} for i in range(50)]}
            prefix = f"{lane}_{number:03d}"
            f = feature_case(torch, tokenizer, reader, device, fixture_cfg, case, prefix, True, arrays)
            metadata.append({"lane": lane, "prefix": prefix, "question_sha256": hashlib.sha256(case["question"].encode()).hexdigest()})
            features.append(f); cases.append(case)
            if lane == "training":
                baseline = old._stable_top(f["base"].numpy(), case["top_50"])
                pos = next(i for i in range(50) if i not in baseline); neg = baseline[-1]
                roles = ["held_not_negative"] * 50; roles[pos] = "qualified_missed_direct"; roles[neg] = "qualified_replaceable_irrelevant"
                targets.append({"future_role": "training", "roles": roles, "baseline_top5_indices": baseline,
                    "draft_pairs": [[pos, neg]], "pair_weights": [1.0], "immutable_selected_indices": baseline[:-1],
                    "eligible_positive_indices": [pos], "replaceable_indices": [neg], "maximum_replacement_slots": 1,
                    "scope_qualified": True, "missing_selected_direct_qualification": []})
    heads = training.make_heads(torch, reader.config.hidden_size, fixture_cfg, device)
    reports, witnesses = training.train_heads(torch, fixture_cfg, heads, features[:2], targets, arrays, device, health_checks=False)
    ranked = {lane: rank_cases(torch, heads, features[o:o + 2], cases[o:o + 2], lane, fixture_cfg, arrays)
              for lane, o in (("training", 0), ("evaluation", 2))}
    return persist_trace(destination, fixture_cfg, heads, arrays, metadata, witnesses, targets, cases, ranked, device,
                         provenance(fixture_cfg, identity, args), synthetic=True)


def publish(run_dir, cfg, args, *, failed=False):
    declared = ("summary.json", "report.md", "run_metadata.json", "semantic_values.npz", "failure_values.npz",
                "value_capture.json", "sample_trace.json", "validation.json", "case_study.json", "case_study.md",
                "training_targets.private.json", "preflight.zip")
    allowed = [name for name in declared if (run_dir / name).is_file()]
    unexpected = {p.name for p in run_dir.iterdir() if p.is_file()} - set(declared) - {"upload_manifest.json", "upload_receipts.json"}
    if unexpected: raise ValueError("undeclared run artifact")
    compact = [name for name in ("summary.json", "report.md", "run_metadata.json") if name in allowed]
    for name in compact:
        if (run_dir / name).stat().st_size > old.MAX_GITHUB_BYTES:
            raise ValueError("compact file exceeds1MiB")
        if name.endswith(".json") and old._contains_forbidden(old._load_json(run_dir / name)):
            raise ValueError("compact file contains raw fields")
    manifest = {"target_directory": cfg["output_namespace"] + "/" + run_dir.name,
        "status": "failed" if failed else "validated", "upload_status": "pending_until_receipts", "exchange_files": [
            {"name": n, "bytes": (run_dir / n).stat().st_size, "sha256": old._sha256(run_dir / n)} for n in sorted(allowed)],
        "github_files": compact + ["upload_manifest.json", "upload_receipts.json"],
        "exchange_only": sorted(set(allowed) - set(compact))}
    old._write_json(run_dir / "upload_manifest.json", manifest)
    if args.upload:
        receipts = upload_manifest(run_dir / "upload_manifest.json", base_url=args.exchange_url, token_env=args.token_env)
        expected = {manifest["target_directory"] + "/" + row["name"]: row for row in manifest["exchange_files"]}
        if len(receipts) != len(expected) or any(r["sha256"] != expected[r["path"]]["sha256"] or r["size_bytes"] != expected[r["path"]]["bytes"] for r in receipts):
            raise ValueError("upload receipts differ from frozen manifest")
        old._write_json(run_dir / "upload_receipts.json", {"status": "uploaded", "receipts": receipts})
        _upload_one(args.exchange_url, os.environ[args.token_env], run_dir / "upload_manifest.json", manifest["target_directory"] + "/upload_manifest.json")


def run(args):
    cfg = validate_config(args.config)
    if args.upload and not os.environ.get(args.token_env):
        raise ValueError("missing " + args.token_env + "; configure before starting")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.preflight_only:
        torch, tokenizer, reader, device, identity = load_reader(cfg, args)
        destination = args.output_root / run_id
        destination.mkdir(parents=True, exist_ok=False)
        result = fixture(torch, tokenizer, reader, device, cfg, destination, identity, args)
        print(json.dumps({"output_dir": str(destination), "preflight": result}))
        return
    bundle = get_bundle(cfg, args.bundle, args)
    rows, audit = load_bundle(bundle)
    input_path, temporary = old._prepare_input(args.input, exchange_url=args.exchange_url, token_env=args.token_env)
    arrays, metadata = {}, []
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        cases = old._load_eval_cases(input_path)
        train_ids = {r["case"]["question_sha256"] for r in rows}
        if train_ids & {hashlib.sha256(c["question"].encode()).hexdigest() for c in cases}:
            raise ValueError("training/evaluation question overlap")
        torch, tokenizer, reader, device, identity = load_reader(cfg, args)
        preflight_dir = run_dir / "preflight"
        preflight_dir.mkdir()
        ready = fixture(torch, tokenizer, reader, device, cfg, preflight_dir, identity, args)
        # Bundle all real-value preflight files for a portable exchange-only receipt.
        import zipfile
        with zipfile.ZipFile(run_dir / "preflight.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(preflight_dir.iterdir()): archive.write(p, p.name)
        train_features = []
        for number, row in enumerate(rows, 1):
            prefix = f"training_{number:03d}"
            train_features.append(feature_case(torch, tokenizer, reader, device, cfg, row["case"], prefix, number <= 2, arrays))
            metadata.append({"lane": "training", "prefix": prefix, "question_sha256": row["case"]["question_sha256"]})
            if number % 10 == 0: print(f"Training Reader features {number}/59", flush=True)
        targets = [r["target"] for r in rows]
        heads = training.make_heads(torch, reader.config.hidden_size, cfg, device)
        reports, witnesses = training.train_heads(torch, cfg, heads, train_features, targets, arrays, device)
        ranked_train = rank_cases(torch, heads, train_features, [r["case"] for r in rows], "training", cfg, arrays)
        eval_features = []
        for number, case in enumerate(cases, 1):
            prefix = f"evaluation_{number:03d}"
            eval_features.append(feature_case(torch, tokenizer, reader, device, cfg, case, prefix, number <= 2, arrays))
            metadata.append({"lane": "evaluation", "prefix": prefix, "question_sha256": hashlib.sha256(case["question"].encode()).hexdigest()})
            if number % 10 == 0: print(f"Evaluation Reader features {number}/100", flush=True)
        ranked_eval = rank_cases(torch, heads, eval_features, cases, "evaluation", cfg, arrays)
        evaluation = diagnose(cases, ranked_eval, cfg)  # All rankings frozen before panel judgments opened.
        p = provenance(cfg, identity, args)
        p.update({"training_bundle_sha256": old._sha256(bundle), "evaluation_input_sha256": old._sha256(input_path)})
        from scripts.collab.lib.protected_boundary_trace import persist_trace
        verified = persist_trace(run_dir, cfg, heads, arrays, metadata, witnesses, targets,
            [r["case"] for r in rows] + cases, {"training": ranked_train, "evaluation": ranked_eval}, device, p)
        old._write_json(run_dir / "training_targets.private.json", {"targets": targets, "annotations": [r["annotation"] for r in rows]})
        # Attach complete per-question supervision/loss to the human/agent case-study record.
        detail = old._load_json(run_dir / "case_study.json")
        for i, case in enumerate(detail["cases"][:len(rows)]):
            case["training_target"] = targets[i]
            case["training_loss"] = {method: [{"epoch": e["epoch"], **e["per_question_loss"][i]} for e in report["epochs"]]
                                     for method, report in reports.items()}
        old._write_json(run_dir / "case_study.json", detail)
        summary = {"schema_version": "rag.protected_boundary_reader_screen.summary.v1", "claim_ceiling": "L0_diagnostic",
            "evaluation": evaluation, "training": {"audit": audit, "arms": reports, "postfit_diagnostic": training_diagnosis(rows, ranked_train, cfg)},
            "validation": verified, "preflight": ready, "provenance": p, "circuit": circuit_resource_contract(),
            "generator_called": False, "retrieval_called": False, "silver_used_online": False,
            "limitations": ["Prior-exposed100 panel and same-parent opinions are not independent gold/generalization.",
                            "Exact statevector simulation; quantum utility/resource advantage unestablished.",
                            "59 training questions,228 dependent pairs;87 validation rows unused."]}
        old._write_json(run_dir / "summary.json", summary)
        report = ["# Protected boundary Reader screen", "", "L0诊断：59题/228纠错对，三轮固定预算；100题已暴露面板，标注仅事后使用。", ""]
        for method, v in evaluation["methods"].items():
            report.append(f"- {method}: 有用段{v['positive_consensus_count']['mean']:.4f}/5；直接证据{v['direct_consensus_count']['mean']:.4f}/5；Silver重合{v['fixed_silver_set_overlap']['mean']:.4f}/5")
        report += ["", "门槛：" + json.dumps(evaluation["gates"]), "", "没有调用Generator，故本轮没有新生成的答案。完整文本、打分、Top5和训练轨迹在18083；只提交manifest声明的github_files。"]
        (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        old._write_json(run_dir / "run_metadata.json", {"run_id": run_id, "status": "validated", "provenance": p, "validation": verified})
    except Exception as exc:
        # Preserve detached evidence without leaking arbitrary exception text/tokens.
        np.savez_compressed(run_dir / "failure_values.npz", **arrays)
        old._write_json(run_dir / "run_metadata.json", {"run_id": run_id, "status": "failed", "error_type": type(exc).__name__,
                       "observed_samples": len(metadata), "config_sha256": old._sha256(args.config)})
        publish(run_dir, cfg, args, failed=True)
        raise
    finally:
        if temporary: temporary.cleanup()
    publish(run_dir, cfg, args)
    print(json.dumps({"output_dir": str(run_dir), "uploaded": args.upload, "gates": evaluation["gates"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--output-root", type=Path, default=ROOT / "exchange/five_ideas/protected_boundary_reader_screen_100")
    p.add_argument("--device")
    p.add_argument("--offline-model", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", old.DEFAULT_EXCHANGE_URL))
    p.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    run(p.parse_args())


if __name__ == "__main__":
    main()
