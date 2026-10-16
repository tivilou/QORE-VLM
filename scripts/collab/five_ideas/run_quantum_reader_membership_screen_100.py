#!/usr/bin/env python3
"""Five-arm, fixed-budget Reader membership screen; cached features, no model/data download."""
from __future__ import annotations
import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag import reader_membership_training as learning
from applications.rag.protected_boundary_inputs import load_bundle
from scripts.collab.five_ideas import run_qarcg_reader_screen_100 as io
from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as metrics
from scripts.collab.lib.exchange_upload import download_exchange_file, upload_manifest, _upload_one
from scripts.collab.lib.source_snapshot import archive_sources, verify_sources

CONFIG = ROOT / "configs/experiments/quantum_reader_membership_screen_100.json"
PLAN = ROOT / "configs/experiments/quantum_reader_membership_screen_100_plan.json"
SOURCES = (
    "scripts/collab/five_ideas/run_quantum_reader_membership_screen_100.py",
    "scripts/collab/five_ideas/run_quantum_reader_membership_screen_100.sh",
    "scripts/collab/five_ideas/validate_reader_membership_trace.py",
    "scripts/collab/five_ideas/prepare_reader_membership_cache.py",
    "configs/experiments/quantum_reader_membership_screen_100.json",
    "configs/experiments/quantum_reader_membership_screen_100_plan.json",
    "configs/experiments/reader_membership_recovery.json",
    "configs/experiments/reader_top5_objective_audit.json",
    "applications/rag/reader_membership_training.py", "applications/rag/scale_invariant_top5_objective.py",
    "applications/rag/quantum_reader_rotation.py", "applications/rag/quantum_semantic_reader.py",
    "applications/rag/protected_boundary_learning.py", "applications/rag/protected_boundary_inputs.py",
    "applications/rag/support_supervision_axes.py", "applications/rag/support_boundary_bridge.py",
    "applications/rag/qarcg_reader.py", "scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py",
    "scripts/collab/five_ideas/run_qarcg_reader_screen_100.py", "applications/rag/semantic_reader_training.py",
    "scripts/collab/lib/exchange_upload.py", "scripts/collab/lib/source_snapshot.py",
)


def digest(path):
    return io._sha256(Path(path))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def config():
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    if tuple(cfg["methods"]) != learning.METHODS or cfg["training"]["epochs"] != 3:
        raise ValueError("allowlist/budget drift")
    plan = json.loads(PLAN.read_text(encoding="utf8"))
    if plan["authorization"] not in ("approved", "implemented") or plan["discovery"]["allowlist"] != cfg["methods"]:
        raise ValueError("plugin plan not approved or allowlist mismatch")
    if cfg["training"]["cases"] != 59 or cfg["training"]["pairs"] != 228 or cfg["evaluation"]["cases"] != 100:
        raise ValueError("frozen populations changed")
    audited = json.loads((ROOT / "configs/experiments/reader_top5_objective_audit.json").read_text())
    if cfg["losses"]["membership"] != audited["objective"]: raise ValueError("audited objective changed")
    dossier = ROOT / plan["recovery_contract"]["dossier_path"]
    if digest(dossier) != plan["recovery_contract"]["dossier_sha256"]: raise ValueError("recovery dossier hash drift")
    return cfg


def obtain(path, spec, exchange_path, args):
    path = Path(path)
    if path.is_symlink(): raise ValueError("symlink input forbidden")
    if not path.exists():
        download_exchange_file(exchange_path, path, base_url=args.exchange_url, token_env=args.token_env,
            expected_size=spec["bytes"], expected_sha256=spec["sha256"])
    if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError("input bytes/hash mismatch: " + path.name)
    return path


class FeatureStore(Sequence):
    """Reopen one frozen question at a time; slicing never materializes the cohort."""
    def __init__(self, path, prefixes):
        self.path, self.prefixes = Path(path), tuple(prefixes)

    def __len__(self): return len(self.prefixes)

    def __getitem__(self, index):
        if isinstance(index, slice): return FeatureStore(self.path, self.prefixes[index])
        prefix = self.prefixes[index]
        with np.load(self.path, allow_pickle=False) as z:
            return {k: torch.from_numpy(z[prefix + "__" + k]) for k in ("base", "cls", "pooled")}


def load_features(directory, cfg):
    directory = Path(directory)
    for name, spec in cfg["features"]["files"].items():
        p = directory / name
        if p.is_symlink() or p.stat().st_size != spec["bytes"] or digest(p) != spec["sha256"]:
            raise ValueError("feature cache integrity mismatch: " + name)
    index = json.loads((directory / "cache_index.json").read_text())
    if index["schema_version"] != "rag.reader_membership_cache.v1" or index["source_run"] != cfg["features"]["source_run"]:
        raise ValueError("feature cache source/schema drift")
    if index["feature_sha256"] != cfg["features"]["files"]["reader_features.npz"]["sha256"]:
        raise ValueError("index/archive binding mismatch")
    if index["reader"]["model_id"] != cfg["reader"]["model_id"] or index["reader"]["resolved_revision"] != cfg["reader"]["revision"]:
        raise ValueError("Reader identity mismatch")
    records = index["records"]
    expected = [(lane, i) for lane, count in (("training", 59), ("evaluation", 100)) for i in range(1, count + 1)]
    if [(r["lane"], r["number"]) for r in records] != expected or len({r["question_sha256"] for r in records}) != 159:
        raise ValueError("cache identity/split/order mismatch")
    features = FeatureStore(directory / "reader_features.npz", [r["prefix"] for r in records])
    with np.load(directory / "reader_features.npz", allow_pickle=False) as z:
        expected_keys = {r["prefix"] + "__" + k for r in records for k in ("base", "cls", "pooled")} | {"relevance_weight", "relevance_bias"}
        if set(z.files) != expected_keys: raise ValueError("unexpected/missing cached coordinates")
        weight, bias = torch.from_numpy(z["relevance_weight"]), torch.from_numpy(z["relevance_bias"])
        h = cfg["reader"]["hidden_size"]
        if weight.shape != (h,) or bias.shape != (): raise ValueError("classifier shape mismatch")
        for r, feature in zip(records, features):
            for key, shape in (("base", (50,)), ("cls", (50, h)), ("pooled", (50, 5 * h))):
                value = feature[key]
                if value.shape != shape or value.dtype != torch.float32 or not bool(value.isfinite().all()):
                    raise ValueError("invalid full cached representation")
            if not torch.equal(feature["cls"], feature["pooled"][:, :h]): raise ValueError("pooling lineage drift")
            np.testing.assert_allclose((feature["cls"] @ weight + bias).numpy(), feature["base"].numpy(), rtol=1e-5, atol=3e-5)
            if learning.top5(feature["base"], r["candidate_retrieved_ranks"]) != r["baseline_top5_indices"]:
                raise ValueError("cached baseline Top5 drift")
    return index, features, weight, bias


def bind_cases(records, training_rows, evaluation_cases, cfg):
    counts = {"cases": len(training_rows), "pairs": sum(len(r["target"]["draft_pairs"]) for r in training_rows),
        "eligible_candidates": sum(len(r["target"]["eligible_positive_indices"]) for r in training_rows),
        "protected_candidates": sum(len(r["target"]["immutable_selected_indices"]) for r in training_rows),
        "maximum_slots": sum(r["target"]["maximum_replacement_slots"] for r in training_rows)}
    if any(counts[k] != cfg["training"][k] for k in counts): raise ValueError("qualified training population changed")
    cases = [r["case"] for r in training_rows] + evaluation_cases
    if len(cases) != len(records): raise ValueError("case/cache coverage mismatch")
    for number, (record, case) in enumerate(zip(records, cases)):
        candidates = case.get("candidates", case.get("top_50"))
        ids = [hashlib.sha256(str(p["id"]).encode()).hexdigest() for p in candidates]
        qid = hashlib.sha256(case["question"].encode()).hexdigest()
        ranks = [p["retrieved_rank"] for p in candidates]
        if qid != record["question_sha256"] or ids != record["candidate_id_sha256"] or ranks != record["candidate_retrieved_ranks"]:
            raise ValueError("question/candidate identity/order mismatch")
        if number < len(training_rows) and training_rows[number]["target"]["baseline_top5_indices"] != record["baseline_top5_indices"]:
            raise ValueError("qualified target baseline mismatch")
    return counts


def rank_all(heads, features, records, arrays):
    cases = []
    for feature, record in zip(features, records):
        row = dict(record, methods={})
        capture = record["number"] <= 2
        if capture:
            row["input_values"] = {k: learning.save_value(arrays, record["prefix"] + "__" + k, v) for k, v in feature.items()}
        for method in learning.METHODS:
            if method == "frozen_reader_topk": scores = feature["base"]
            else:
                with torch.no_grad(): values = learning.forward(heads[method], feature)
                scores = values["scores"].cpu()
            key = record["prefix"] + "__" + method
            item = {"score_vector": scores.tolist(), "selected_indices": learning.top5(scores, record["candidate_retrieved_ranks"]),
                "score_values": learning.save_value(arrays, key + "__scores", scores)}
            if capture and method != "frozen_reader_topk":
                item["output_values"] = {k: learning.save_value(arrays, key + "__" + k, v) for k, v in values.items() if k != "scores"}
                item["output_values"]["scores"] = item["score_values"]
            row["methods"][method] = item
        cases.append(row)
    return cases


def diagnose(evaluation_cases, records, cfg):
    fields = ("positive_consensus_count", "direct_consensus_count", "all_models_positive_count", "fixed_silver_set_overlap")
    if len(evaluation_cases) != len(records): raise ValueError("posthoc coverage mismatch")
    values = {m: {k: [] for k in fields} for m in learning.METHODS}
    # This function is invoked only after ALL arms have frozen their Top5.
    for case, record in zip(evaluation_cases, records):
        for method in learning.METHODS:
            item = record["methods"][method]
            item["posthoc"] = metrics._metric_values(case, item["selected_indices"])
            for key in fields: values[method][key].append(item["posthoc"][key])
    refs = ("frozen_reader_topk", "classical_membership", "quantum_retention_control")
    paired = {ref: {k: metrics.paired_delta(values["quantum_membership"][k], values[ref][k],
        cfg["evaluation"]["bootstrap_seed"], cfg["evaluation"]["bootstrap_replicates"]) for k in fields} for ref in refs}
    utility, direct = paired[refs[0]][fields[0]], paired[refs[0]][fields[1]]
    attribution = paired[refs[1]][fields[0]]
    return {"methods": {m: {k: {"total": int(sum(v)), "mean": float(np.mean(v))} for k, v in fields_.items()} for m, fields_ in values.items()},
        "paired_quantum_membership_minus": paired,
        "gates": {"utility": utility["mean"] > 0 and utility["ci95"][0] > 0 and direct["mean"] >= 0,
            "quantum_attribution": attribution["mean"] > 0 and attribution["ci95"][0] > 0},
        "population": f"all{len(records)}, including zero-positive questions; frozen exposed panel remains L0"}


def build_samples(rows, reports, preflight):
    samples = []
    for row in rows:
        if row["number"] > 2: continue
        refs = list(row["input_values"].values())
        score_refs = [item["score_values"] for item in row["methods"].values()]
        score_refs += [ref for item in row["methods"].values() for key, ref in item.get("output_values", {}).items() if key != "scores"]
        train_refs = [v for report in reports.values() for v in report["final_state"].values()]
        training_observed = not preflight and row["lane"] == "training"
        diagnosis_refs = [] if preflight else ["selection:" + row["prefix"],
            ("train_target:" + row["question_sha256"]) if row["lane"] == "training" else ("posthoc:" + row["prefix"])]
        samples.append({"sample_id": row["prefix"], "question_sha256": row["question_sha256"], "lane": row["lane"],
            "semantic_roles": ["input", "reference", "result"], "stages": [
                {"stage": "data", "status": "observed", "input_refs": ["source:cache_index:" + row["prefix"]], "output_refs": ["source:features:" + row["prefix"]]},
                {"stage": "preprocess", "status": "observed", "input_refs": ["source:features:" + row["prefix"]], "output_refs": refs},
                {"stage": "training", "status": "observed" if training_observed else "not_applicable", "input_refs": refs + (["train_target:" + row["question_sha256"]] if training_observed else []), "output_refs": train_refs if training_observed else []},
                {"stage": "scoring", "status": "observed", "input_refs": refs + train_refs, "output_refs": score_refs},
                {"stage": "selection", "status": "observed", "input_refs": score_refs, "output_refs": ["selection:" + row["prefix"]]},
                {"stage": "diagnosis", "status": "not_applicable" if preflight else "observed", "input_refs": diagnosis_refs, "output_refs": [] if preflight else ["diagnosis:" + row["prefix"]]}
            ]})
    return samples


def execute(cfg, index, features, weight, bias, targets, evaluation_cases, output, *, preflight=False, synthetic=False, device="cpu", provenance=None):
    """Shared production path; synthetic inputs are explicit and never promoted to utility."""
    from scripts.collab.five_ideas.validate_reader_membership_trace import validate
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays, reports, witnesses = {}, {}, []
    heads = learning.make_heads(weight, bias, cfg, device)
    ntrain = len(targets)
    records = index["records"]
    null_max = 0.
    for method, head in heads.items():
        print(f"Checking frozen Reader null: {method}", flush=True)
        for feature, record in zip(features, records):
            with torch.no_grad(): scores = learning.forward(head, feature)["scores"].cpu()
            null_max = max(null_max, float((scores - feature["base"]).abs().max()))
            np.testing.assert_allclose(scores.numpy(), feature["base"].numpy(), rtol=1e-5, atol=3e-5)
            if learning.top5(scores, record["candidate_retrieved_ranks"]) != record["baseline_top5_indices"]:
                raise ValueError("initial Reader Top5 null mismatch")
        if preflight:
            initial = {k: learning.save_value(arrays, method + "__initial__" + k, v) for k, v in head.state_dict().items()}
            reports[method] = {"initial_state": initial, "final_state": initial, "epochs": [], "resources": head.resources()}
        else:
            reports[method], w = learning.train_arm(method, head, features[:ntrain], targets, records[:ntrain], cfg, arrays)
            witnesses.extend(w)
    print("Recording final full50 scores and selected complete-coordinate traces", flush=True)
    rows = rank_all(heads, features, records, arrays)
    if not preflight:
        for row, target in zip(rows[:ntrain], targets):
            for item in row["methods"].values():
                selected = set(item["selected_indices"])
                item["training_membership"] = {"rescued_eligible": len(selected & set(target["eligible_positive_indices"])),
                    "lost_protected": len(set(target["immutable_selected_indices"]) - selected)}
    evaluation = None if preflight else diagnose(evaluation_cases, rows[ntrain:], cfg)
    if evaluation is not None:
        h = reports["quantum_membership"]["epochs"][-1]["membership_health"]
        evaluation["gates"]["training_safe_rescue"] = h["rescued_eligible"] > 0 and h["lost_protected"] == 0
    np.savez_compressed(output / "membership_values.npz", **arrays)
    trace = {"schema_version": "sample-trace.v2", "artifact_type": "reader_membership_selector_trace", "run_id": output.name,
        "synthetic_only": synthetic, "preflight_only": preflight, "methods": list(learning.METHODS),
        "stages": list(learning.STAGES), "population": [{k: r[k] for k in ("lane", "number", "prefix", "question_sha256")} for r in records],
        "sample_selection": {"rule": "first2 of each frozen lane; first2 updates of every epoch and fitted arm", "selected_count": min(2, ntrain) + min(2, len(records) - ntrain)},
        "samples": build_samples(rows, reports, preflight), "cases": rows, "training": reports, "training_witnesses": witnesses,
        "values": learning.describe_values(arrays), "values_sha256": digest(output / "membership_values.npz"),
        "target_boundary": "Only train_arm/objective/health/replay receive training targets; forward and ranking accept numeric inputs only; evaluation labels posthoc after all rankings",
        "provenance": provenance or {}, "constraints": cfg["constraints"]}
    write(output / "selector_trace.json", trace)
    write(output / "training_targets.private.json", {"targets": targets, "question_sha256": [r["question_sha256"] for r in records[:ntrain]]})
    # Reopen the serialized values for independent replay without keeping a
    # second full witness archive resident in memory.
    arrays.clear()
    print("Reopening value archive for semantic and native optimizer replay", flush=True)
    readiness = validate(output, cfg, index, features, weight, bias, targets)
    write(output / "readiness.json", readiness)
    summary = {"schema_version": "rag.quantum_reader_membership_screen.summary.v1", "run_id": output.name,
        "claim_ceiling": cfg["claim_ceiling"], "synthetic_only": synthetic, "preflight_only": preflight,
        "methods": list(learning.METHODS), "initial_null_max_abs": null_max,
        "training": {m: {"epochs": r["epochs"], "resources": r["resources"], "objective": learning.ARMS[m][1]} for m, r in reports.items()},
        "evaluation": evaluation, "readiness": readiness, "constraints": cfg["constraints"], "provenance": provenance or {}, "limitations": cfg["limitations"]}
    if io._contains_forbidden(summary): raise ValueError("raw content in compact summary")
    write(output / "summary.json", summary)
    lines = ["# Reader membership screen", "", "L0 diagnostic; cached frozen Reader; no generation; synthetic=" + str(synthetic), ""]
    if evaluation:
        for m, fields in evaluation["methods"].items():
            lines.append(f"- {m}: useful={fields['positive_consensus_count']['total']}; direct={fields['direct_consensus_count']['total']}; Silver={fields['fixed_silver_set_overlap']['total']}")
        lines += ["", "Gates: " + json.dumps(evaluation["gates"])]
    else: lines += ["Frozen-input/null preflight only; zero optimizer updates."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    if not preflight:
        write(output / "case_study.json", {"schema_version": "rag.reader_membership_case_study.v1", "synthetic_only": synthetic,
            "input_source_sha256": cfg["evaluation"]["input_sha256"], "generation_status": "not_applicable; no Generator in selector-only screen",
            "cases": [{"question": c["question"], "gold_answers": c.get("gold_answers", c.get("answers", [])),
                "top_50": c["top_50"], "methods": r["methods"], "question_sha256": r["question_sha256"]} for c, r in zip(evaluation_cases, rows[ntrain:])]})
    return summary


def publish(output, cfg, args):
    compact = ["summary.json", "report.md", "run_metadata.json", "readiness.json"]
    private = ["selector_trace.json", "membership_values.npz", "training_targets.private.json", "executed_sources.zip", "cache_index.json", "case_study.json"]
    files = [p for p in compact + private if (output / p).exists()]
    unexpected = {p.name for p in output.iterdir() if p.is_file()} - set(files) - {"upload_manifest.json", "upload_receipts.json"}
    if unexpected: raise ValueError("undeclared outputs: " + str(unexpected))
    if any((output / p).stat().st_size > 1_048_576 for p in compact): raise ValueError("GitHub output exceeds1MiB")
    suffix = ("preflight/" if args.preflight_only else "") + output.name
    manifest = {"target_directory": cfg["output_namespace"] + "/" + suffix, "status": "validated", "upload_status": "pending_until_receipts",
        "exchange_files": [{"name": n, "bytes": (output / n).stat().st_size, "sha256": digest(output / n)} for n in files],
        "github_files": compact + ["upload_manifest.json", "upload_receipts.json"], "exchange_only": [n for n in files if n not in compact]}
    write(output / "upload_manifest.json", manifest)
    if args.upload:
        receipts = upload_manifest(output / "upload_manifest.json", base_url=args.exchange_url, token_env=args.token_env)
        # Transport checks receipt path, size AND hash for every declared file.
        write(output / "upload_receipts.json", {"status": "uploaded", "receipts": receipts})
        _upload_one(args.exchange_url, os.environ[args.token_env], output / "upload_manifest.json", manifest["target_directory"] + "/upload_manifest.json")
        _upload_one(args.exchange_url, os.environ[args.token_env], output / "upload_receipts.json", manifest["target_directory"] + "/upload_receipts.json")


def run(args):
    cfg = config()
    if args.validate_only:
        print(json.dumps({"status": "valid", "methods": cfg["methods"], "gpu_required": False})); return
    if not args.preflight_only and json.loads(PLAN.read_text(encoding="utf8"))["authorization"] != "implemented":
        raise ValueError("Real training is gated until the frozen-input/exchange preflight passes; --preflight-only performs zero updates")
    if args.upload and not os.environ.get(args.token_env): raise ValueError("configure " + args.token_env + " before upload")
    print("Resolving hash-locked feature cache and qualified support bundle", flush=True)
    feature_dir = args.features or ROOT / ".cache/reader_membership_inputs/20261016-v1"
    for name, spec in cfg["features"]["files"].items(): obtain(feature_dir / name, spec, cfg["features"]["namespace"] + "/" + name, args)
    index, features, weight, bias = load_features(feature_dir, cfg)
    spec = cfg["training"]["input"]
    bundle_path = args.bundle or ROOT / ".cache/protected_boundary_reader_inputs/20261011-v1/frozen_support_bundle.zip"
    obtain(bundle_path, spec, spec["exchange_path"], args)
    rows, audit = load_bundle(bundle_path)
    print("Qualified training bundle verified: 59 questions / 228 pairs", flush=True)
    panel_path = args.input or io.DEFAULT_INPUT
    obtain(panel_path, {"bytes": io.DEFAULT_INPUT_BYTES, "sha256": cfg["evaluation"]["input_sha256"]}, io.DEFAULT_INPUT_EXCHANGE_PATH, args)
    cases = io._load_eval_cases(panel_path)
    bind_cases(index["records"], rows, cases, cfg)
    print("All159 question/candidate identities verified; no Reader or dataset loading", flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_root / ("preflight" if args.preflight_only else "") / run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    # Capture sources before execution in a sibling staging directory, then move
    # the immutable archive into execute's exclusively-created result directory.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="membership-source-", dir=output.parent) as tmp:
        snapshot = archive_sources(ROOT, SOURCES, Path(tmp) / "executed_sources.zip")
        provenance = {"code_revision": io._git_revision() or "unavailable", "config_sha256": digest(CONFIG), "plan_sha256": digest(PLAN),
            "reader": index["reader"], "features": cfg["features"], "bundle_sha256": digest(bundle_path), "panel_sha256": digest(panel_path),
            "source_snapshot": snapshot, "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": args.device},
            "arm_objectives": {m: {"family": f, "loss": loss} for m, (f, loss) in learning.ARMS.items()}, "original_provenance_exception": "prior executed runner source absent; no historical optimizer replay claim"}
        summary = execute(cfg, index, features, weight, bias, [r["target"] for r in rows], cases, output,
            preflight=args.preflight_only, device=args.device, provenance=provenance)
        verify_sources(ROOT, snapshot)
        (Path(tmp) / "executed_sources.zip").rename(output / "executed_sources.zip")
    (output / "cache_index.json").write_bytes((feature_dir / "cache_index.json").read_bytes())
    write(output / "run_metadata.json", {"run_id": run_id, "provenance": provenance, "input_qualification_audit": audit,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(output.iterdir()) if p.is_file()]})
    publish(output, cfg, args)
    print(json.dumps({"output_dir": str(output), "uploaded": args.upload, "preflight_only": args.preflight_only, "evaluation": summary["evaluation"]}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--output-root", type=Path, default=ROOT / "exchange/five_ideas/quantum_reader_membership_screen_100")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", io.DEFAULT_EXCHANGE_URL))
    p.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    args = p.parse_args()
    if args.threads < 1: p.error("positive thread count required")
    torch.set_num_threads(args.threads)
    try: run(args)
    except Exception as exc:
        print(f"Reader membership screen failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__": main()
