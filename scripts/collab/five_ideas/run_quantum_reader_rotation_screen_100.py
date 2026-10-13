#!/usr/bin/env python3
"""Collaborator-only 100-case screen for a quantum Reader feature rotation.

The DPR Reader, retrieval pool, Generator and evaluator remain frozen.  The
only fitted objects are six fresh score heads trained on the immutable 59/228
support bundle.  Evaluation labels are opened only after every arm has ranked
the complete Top50 without masks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from applications.rag import protected_boundary_learning as old_learning
from applications.rag import protected_boundary_inputs
from applications.rag.protected_boundary_training import make_heads as make_old_heads
from applications.rag.quantum_reader_rotation import (
    RotationConfig, make_rotation_head, top5_retention_objective,
)
from applications.rag.quantum_semantic_reader import reader_semantic_forward
from applications.rag.qarcg_reader import QARCGConfig
from scripts.collab.five_ideas import run_protected_boundary_reader_screen_100 as prior
from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as metrics
from scripts.collab.lib.exchange_upload import (
    _upload_one, download_exchange_file, upload_manifest,
)

CONFIG = ROOT / "configs/experiments/quantum_reader_rotation_screen_100.json"
PLAN = ROOT / "configs/experiments/quantum_reader_rotation_screen_100_plan.json"
METHODS = (
    "frozen_reader_topk", "old_head_old_loss", "old_head_new_loss",
    "new_quantum_old_loss", "new_quantum_new_loss", "new_classical_new_loss",
)
STAGES = ("data", "frozen_reader", "head", "optimization", "selection", "posthoc")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value) -> bool:
    return bool(np.isfinite(np.asarray(value)).all())


def validate_config(path: Path):
    cfg = _load_json(path)
    frozen = _load_json(CONFIG)
    if cfg != frozen:
        raise ValueError("config differs from the pinned collaborator contract")
    if tuple(cfg["methods"]) != METHODS:
        raise ValueError("method allowlist drift")
    if cfg["training"]["cases"] != 59 or cfg["training"]["pairs"] != 228 or cfg["evaluation"]["cases"] != 100:
        raise ValueError("frozen population changed")
    plan = _load_json(PLAN)
    if plan.get("authorization") != "approved" or tuple(plan["discovery"]["allowlist"]) != METHODS:
        raise ValueError("plugin plan or allowlist is not approved")
    return cfg


def get_bundle(cfg, explicit: Path | None, args):
    spec = cfg["training"]["input"]
    path = explicit or ROOT / ".cache/protected_boundary_reader_inputs/20261011-v1/frozen_support_bundle.zip"
    if not path.exists():
        download_exchange_file(spec["exchange_path"], path, base_url=args.exchange_url,
                               token_env=args.token_env, expected_size=spec["bytes"],
                               expected_sha256=spec["sha256"])
    if path.is_symlink() or path.stat().st_size != spec["bytes"] or _sha256(path) != spec["sha256"]:
        raise ValueError("frozen support bundle size/hash mismatch")
    return path


def load_reader(cfg, args):
    import torch
    from transformers import DPRReader, DPRReaderTokenizer
    spec = cfg["reader"]
    options = {"revision": spec["revision"], "local_files_only": bool(args.offline_model)}
    tokenizer = DPRReaderTokenizer.from_pretrained(spec["model_id"], **options)
    reader = DPRReader.from_pretrained(spec["model_id"], **options)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reader.to(device).eval().requires_grad_(False)
    identity = prior.old._reader_identity(reader, tokenizer, spec["model_id"], spec["revision"], device)
    if identity["resolved_revision"] != spec["revision"]:
        raise ValueError("pinned Reader revision drift")
    return torch, tokenizer, reader, device, identity


def feature_case(torch, tokenizer, reader, device, cfg, case, prefix, capture, arrays):
    candidates = case.get("candidates", case.get("top_50"))
    texts = [prior.old._online_text({k: p.get(k) for k in ("title", "text")}) for p in candidates]
    base, pooled, scalar, hidden = reader_semantic_forward(
        torch, tokenizer, reader, device, case["question"], texts,
        max_length=cfg["reader"]["max_length"], batch_size=cfg["reader"]["batch_size"],
        capture_first=capture,
    )
    hidden_size = reader.config.hidden_size
    features = {"base": base, "pooled": pooled, "cls": pooled[:, :hidden_size], "scalar": scalar}
    for key, value in features.items():
        arrays[prefix + "__" + key] = value.detach().cpu().numpy().copy()
    for key, value in (hidden or {}).items():
        arrays[prefix + "__" + key] = np.array(value, copy=True)
    if hidden is not None:
        classifier = reader.span_predictor.qa_classifier
        arrays[prefix + "__relevance_weight"] = classifier.weight.detach().cpu().numpy().copy()
        arrays[prefix + "__relevance_bias"] = classifier.bias.detach().cpu().numpy().copy()
    if "reader_top5_indices" in case:
        expected = np.asarray([p["reader_score"] for p in candidates], dtype=np.float32)
        if not np.allclose(base.detach().cpu().numpy(), expected, rtol=1e-4, atol=1e-4):
            raise ValueError("frozen Reader score parity failed")
        if prior.old._stable_top(base.detach().cpu().numpy(), candidates) != case["reader_top5_indices"]:
            raise ValueError("frozen Reader Top5 parity failed")
    return features


def make_rotation_heads(torch, reader, cfg, device):
    classifier = reader.span_predictor.qa_classifier
    rotation_cfg = RotationConfig(**{key: cfg["rotation_head"][key] for key in ("layers", "conditioning", "seed")})
    heads = {
        "new_quantum_old_loss": make_rotation_head(classifier.weight, classifier.bias, family="quantum", config=rotation_cfg).to(device),
        "new_quantum_new_loss": make_rotation_head(classifier.weight, classifier.bias, family="quantum", config=rotation_cfg).to(device),
        "new_classical_new_loss": make_rotation_head(classifier.weight, classifier.bias, family="classical", config=rotation_cfg).to(device),
    }
    return heads


def make_all_heads(torch, reader, cfg, device):
    old_cfg = dict(cfg)
    old_cfg["head"] = cfg["old_head"]
    old_cfg["training"] = dict(cfg["training"])
    old_cfg["training"]["optimizer_policy"] = "adamw_projection_fanin_v1"
    old_heads = make_old_heads(torch, reader.config.hidden_size, old_cfg, device)
    heads = {
        "old_head_old_loss": old_heads["quantum_semantic"],
        "old_head_new_loss": make_old_heads(torch, reader.config.hidden_size, old_cfg, device)["quantum_semantic"],
    }
    heads.update(make_rotation_heads(torch, reader, cfg, device))
    rotation_counts = {sum(p.numel() for p in heads[name].parameters()) for name in
                       ("new_quantum_old_loss", "new_quantum_new_loss", "new_classical_new_loss")}
    if len(rotation_counts) != 1:
        raise ValueError("quantum/classical rotation arms do not have matched trainable parameter count")
    return heads


def arm_kind(method):
    return "old" if method.startswith("old_head") else "new"


def head_output(head, method, feature, device, arrays=None, prefix=None):
    if method.startswith("old_head"):
        output = head(feature["base"].to(device), feature["pooled"].to(device))
        values = {"scores": output[0], "gate": output[1], "residual": output[2],
                  "observables": output[3], "encoded": output[4]}
    else:
        output = head(feature["cls"].to(device), feature["pooled"].to(device))
        values = {"scores": output["scores"], "rotated": output["rotated"],
                  "conditioning": output["conditioning"], "angles": output["angles"],
                  "states": output["states"], "norm": output["norm"]}
    if arrays is not None and prefix is not None:
        for key, value in values.items():
            arrays[prefix + "__" + key] = value.detach().cpu().numpy().copy()
    return values


def objective(method, scores, base, row, cfg):
    settings = cfg["losses"][arm_kind(method)]
    if method.startswith("old_head"):
        return old_learning.boundary_objective(scores, base, row, **settings)
    return top5_retention_objective(scores, base, row, **settings)


def make_optimizer(torch, head, method, cfg):
    if method.startswith("old_head"):
        return torch.optim.AdamW(
            [{"params": [p for n, p in head.named_parameters() if n == "project.weight"], "weight_decay": cfg["training"]["weight_decay"], "lr": cfg["training"]["learning_rate"]},
             {"params": [p for n, p in head.named_parameters() if n != "project.weight"], "weight_decay": 0.0, "lr": cfg["training"]["learning_rate"]}],
            lr=cfg["training"]["learning_rate"],
        )
    return torch.optim.AdamW(head.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])


def train_arm(torch, method, head, features, rows, cfg, device, arrays):
    optimizer = make_optimizer(torch, head, method, cfg)
    reports, witnesses = [], []
    init = {name: p.detach().cpu().numpy().copy() for name, p in head.state_dict().items()}
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        losses = {"total": 0.0, "pair": 0.0, "anchor": 0.0, "protection": 0.0}
        grad_norm = 0.0
        for index, (feature, row) in enumerate(zip(features, rows)):
            optimizer.zero_grad(set_to_none=True)
            witness = index < 2
            prefix = f"{method}__epoch{epoch}__sample{index + 1}"
            if witness:
                for name, value in head.state_dict().items(): arrays[prefix + "__before__" + name] = value.detach().cpu().numpy().copy()
                with torch.no_grad(): head_output(head, method, feature, device, arrays, prefix + "__pre")
            values = head_output(head, method, feature, device)
            parts = objective(method, values["scores"], feature["base"].to(device), row, cfg)
            parts["total"].backward()
            for name, parameter in head.named_parameters():
                if parameter.grad is None or not bool(parameter.grad.isfinite().all()):
                    raise ValueError(f"missing/nonfinite gradient: {method}:{name}")
                grad_norm += float(parameter.grad.detach().norm().cpu())
                if witness: arrays[prefix + "__gradient__" + name] = parameter.grad.detach().cpu().numpy().copy()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["training"]["gradient_clip_norm"])
            optimizer.step()
            if not all(bool(parameter.isfinite().all()) for parameter in head.parameters()):
                raise ValueError(f"nonfinite update: {method}")
            for key in losses: losses[key] += float(parts[key].detach().cpu())
            if witness:
                for name, value in head.state_dict().items(): arrays[prefix + "__after__" + name] = value.detach().cpu().numpy().copy()
                with torch.no_grad(): head_output(head, method, feature, device, arrays, prefix + "__post")
                witnesses.append({"method": method, "epoch": epoch, "sample_index": index, "prefix": prefix})
        reports.append({"epoch": epoch, "updates": len(features), "mean_loss": {k: v / len(features) for k, v in losses.items()}, "gradient_norm_sum": grad_norm})
        print(f"{method}: epoch {epoch}/{cfg['training']['epochs']} complete", flush=True)
    final = {name: p.detach().cpu().numpy().copy() for name, p in head.state_dict().items()}
    return {"parameter_count": sum(p.numel() for p in head.parameters()), "initial": init, "final": final, "epochs": reports}, witnesses


def rank_cases(torch, heads, features, cases, device, arrays, lane):
    traces = []
    for number, (feature, case) in enumerate(zip(features, cases), 1):
        candidates = case.get("candidates", case.get("top_50"))
        prefix = f"{lane}_{number:03d}"
        ids = [hashlib.sha256(str(p["id"]).encode()).hexdigest() for p in candidates]
        record = {"case_number": number, "question_sha256": hashlib.sha256(case["question"].encode()).hexdigest(),
                  "candidate_retrieved_ranks": [p["retrieved_rank"] for p in candidates], "candidate_id_sha256": ids, "methods": {}}
        for method in METHODS:
            if method == "frozen_reader_topk":
                scores = feature["base"].detach().cpu().numpy().copy()
            else:
                values = head_output(heads[method], method, feature, device, arrays, prefix + "__" + method)
                scores = values["scores"].detach().cpu().numpy().copy()
            selected = prior.old._stable_top(scores, candidates)
            record["methods"][method] = {
                "score_vector": scores.tolist(), "selected_indices": selected,
                "selected_retrieved_ranks": [candidates[i]["retrieved_rank"] for i in selected],
                "selected_id_sha256": [ids[i] for i in selected],
            }
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
    comparisons = {}
    for reference in ("frozen_reader_topk", "new_classical_new_loss"):
        comparisons[reference] = {}
        for method in ("old_head_old_loss", "old_head_new_loss", "new_quantum_old_loss", "new_quantum_new_loss", "new_classical_new_loss"):
            if method == reference: continue
            comparisons[reference][method] = {key: metrics.paired_delta(values[method][key], values[reference][key], cfg["evaluation"]["bootstrap_seed"], cfg["evaluation"]["bootstrap_replicates"]) for key in fields}
    primary = comparisons["frozen_reader_topk"]["new_quantum_new_loss"][fields[0]]
    direct = comparisons["frozen_reader_topk"]["new_quantum_new_loss"][fields[1]]
    attribution = comparisons["new_classical_new_loss"]["new_quantum_new_loss"][fields[0]]
    return {
        "methods": {m: {k: {"total": int(sum(v)), "mean": float(np.mean(v))} for k, v in fields_.items()} for m, fields_ in values.items()},
        "paired_deltas": comparisons,
        "gates": {
            "utility": primary["mean"] > 0 and primary["ci95"][0] > 0 and direct["mean"] >= 0,
            "quantum_attribution": attribution["mean"] > 0 and attribution["ci95"][0] > 0,
        },
        "population": "all100 questions including zero-positive; panel/Silver labels opened posthoc only",
    }


def provenance(cfg, identity, args):
    import torch
    import transformers
    sources = [Path(__file__), CONFIG, PLAN, ROOT / "applications/rag/quantum_reader_rotation.py",
               ROOT / "applications/rag/quantum_semantic_reader.py", ROOT / "applications/rag/protected_boundary_learning.py",
               ROOT / "applications/rag/protected_boundary_inputs.py", ROOT / "applications/rag/qarcg_reader.py",
               ROOT / "scripts/collab/lib/exchange_upload.py"]
    return {"code_revision": prior.old._git_revision() or "unavailable", "config_sha256": _sha256(CONFIG), "plan_sha256": _sha256(PLAN),
            "reader": identity, "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "source_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): _sha256(p) for p in sources}}


def publish(run_dir, cfg, args):
    declared = ["summary.json", "report.md", "run_metadata.json", "selector_trace.json", "rotation_values.npz", "training_targets.private.json"]
    files = [name for name in declared if (run_dir / name).is_file()]
    unexpected = {p.name for p in run_dir.iterdir() if p.is_file()} - set(files) - {"upload_manifest.json", "upload_receipts.json"}
    if unexpected: raise ValueError("undeclared output files: " + ",".join(sorted(unexpected)))
    compact = [name for name in ("summary.json", "report.md", "run_metadata.json") if name in files]
    if any((run_dir / name).stat().st_size > prior.old.MAX_GITHUB_BYTES for name in compact):
        raise ValueError("compact output exceeds 1 MiB")
    manifest = {"target_directory": cfg["output_namespace"] + "/" + run_dir.name, "status": "validated", "upload_status": "pending_until_receipts",
                "exchange_files": [{"name": name, "bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in files],
                "github_files": compact + ["upload_manifest.json", "upload_receipts.json"], "exchange_only": [name for name in files if name not in compact]}
    _write_json(run_dir / "upload_manifest.json", manifest)
    if args.upload:
        if not os.environ.get(args.token_env): raise ValueError("missing " + args.token_env)
        receipts = upload_manifest(run_dir / "upload_manifest.json", base_url=args.exchange_url, token_env=args.token_env)
        expected = {manifest["target_directory"] + "/" + row["name"]: row for row in manifest["exchange_files"]}
        if len(receipts) != len(expected) or any(r["sha256"] != expected[r["path"]]["sha256"] or r["size_bytes"] != expected[r["path"]]["bytes"] for r in receipts):
            raise ValueError("exchange receipt mismatch")
        _write_json(run_dir / "upload_receipts.json", {"status": "uploaded", "receipts": receipts})
        _upload_one(args.exchange_url, os.environ[args.token_env], run_dir / "upload_manifest.json", manifest["target_directory"] + "/upload_manifest.json")


def run(args):
    cfg = validate_config(args.config)
    if args.validate_only:
        print(json.dumps({"status": "valid", "methods": METHODS, "config": str(args.config)})); return
    if args.upload and not os.environ.get(args.token_env): raise ValueError("configure " + args.token_env + " before --upload")
    bundle = get_bundle(cfg, args.bundle, args)
    rows, audit = protected_boundary_inputs.load_bundle(bundle)
    input_path, temporary = prior.old._prepare_input(args.input, exchange_url=args.exchange_url, token_env=args.token_env)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    arrays, metadata = {}, []
    try:
        cases = prior.old._load_eval_cases(input_path)
        training_hashes = {r["case"]["question_sha256"] for r in rows}
        eval_hashes = {hashlib.sha256(c["question"].encode()).hexdigest() for c in cases}
        if training_hashes & eval_hashes: raise ValueError("training/evaluation question overlap")
        torch, tokenizer, reader, device, identity = load_reader(cfg, args)
        train_features = []
        for number, row in enumerate(rows, 1):
            prefix = f"training_{number:03d}"
            train_features.append(feature_case(torch, tokenizer, reader, device, cfg, row["case"], prefix, number <= 2, arrays))
            metadata.append({"lane": "training", "prefix": prefix, "question_sha256": row["case"]["question_sha256"]})
            if number % 10 == 0: print(f"Training Reader features {number}/59", flush=True)
        heads = make_all_heads(torch, reader, cfg, device)
        reports, witnesses = {}, []
        for method in METHODS:
            if method == "frozen_reader_topk": continue
            reports[method], local = train_arm(torch, method, heads[method], train_features, rows, cfg, device, arrays)
            witnesses.extend(local)
        train_trace = rank_cases(torch, heads, train_features, [r["case"] for r in rows], device, arrays, "training")
        eval_features = []
        for number, case in enumerate(cases, 1):
            prefix = f"evaluation_{number:03d}"
            eval_features.append(feature_case(torch, tokenizer, reader, device, cfg, case, prefix, number <= 2, arrays))
            metadata.append({"lane": "evaluation", "prefix": prefix, "question_sha256": hashlib.sha256(case["question"].encode()).hexdigest()})
            if number % 10 == 0: print(f"Evaluation Reader features {number}/100", flush=True)
        eval_trace = rank_cases(torch, heads, eval_features, cases, device, arrays, "evaluation")
        evaluation = diagnose(cases, eval_trace, cfg)
        np.savez_compressed(run_dir / "rotation_values.npz", **arrays)
        provenance_data = provenance(cfg, identity, args)
        compact_summary = {"schema_version": "rag.quantum_reader_rotation_screen.summary.v1", "claim_ceiling": "L0_diagnostic",
            "methods": list(METHODS), "training": {"cases": len(rows), "pairs": audit["pairs"], "arms": reports},
            "evaluation": evaluation, "parameter_counts": {m: sum(p.numel() for p in heads[m].parameters()) for m in METHODS if m != "frozen_reader_topk"},
            "audit": {"training_cases": audit["training_cases"], "actual_witness_checks": audit["actual_witness_checks"], "validation_used": False},
            "provenance": provenance_data, "generator_called": False, "evaluator_called": False, "silver_used_online": False,
            "limitations": _load_json(PLAN)["limitations"]}
        _write_json(run_dir / "summary.json", compact_summary)
        trace = {"schema_version": "sample-trace.v2", "artifact_type": "quantum_reader_rotation_selector_trace", "population": metadata,
                 "training_witnesses": witnesses, "training_cases": train_trace, "evaluation_cases": eval_trace,
                 "provenance": provenance_data, "values_sha256": _sha256(run_dir / "rotation_values.npz"),
                 "validation": {"reader_frozen": True, "spans_untouched": True, "full_top50_unmasked": True, "methods": list(METHODS)}}
        _write_json(run_dir / "selector_trace.json", trace)
        _write_json(run_dir / "training_targets.private.json", {"schema_version": "rag.rotation_training_targets.private.v1", "rows": rows, "audit": audit})
        lines = ["# Quantum Reader rotation 100-case screen", "", "L0 diagnostic; frozen DPR Reader; no Generator/evaluator; labels posthoc only.", ""]
        for method, values in evaluation["methods"].items():
            lines.append(f"- {method}: positive {values['positive_consensus_count']['mean']:.4f}/5; direct {values['direct_consensus_count']['mean']:.4f}/5; Silver overlap {values['fixed_silver_set_overlap']['mean']:.4f}/5")
        lines += ["", "Gates: " + json.dumps(evaluation["gates"], ensure_ascii=False), "", "Raw selector trace, targets and complete coordinates are exchange-only."]
        (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        records = [{"name": name, "bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in ("summary.json", "report.md", "selector_trace.json", "rotation_values.npz", "training_targets.private.json")]
        _write_json(run_dir / "run_metadata.json", {"run_id": run_id, "files": records, "provenance": provenance_data, "evaluation": evaluation})
        publish(run_dir, cfg, args)
        print(json.dumps({"output_dir": str(run_dir), "uploaded": args.upload, "gates": evaluation["gates"]}, ensure_ascii=False))
    finally:
        if temporary: temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "exchange/five_ideas/quantum_reader_rotation_screen_100")
    parser.add_argument("--device")
    parser.add_argument("--offline-model", action="store_true")
    parser.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", prior.old.DEFAULT_EXCHANGE_URL))
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Quantum Reader rotation screen failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
