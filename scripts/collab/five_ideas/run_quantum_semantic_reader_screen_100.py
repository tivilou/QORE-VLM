#!/usr/bin/env python3
"""Exploratory semantic-vs-scalar quantum Reader screen; collaborator execution."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.collab.five_ideas import run_qarcg_reader_screen_100 as old
from applications.rag.quantum_semantic_reader import (
    circuit_resource_contract, make_semantic_head, reader_semantic_forward, semantic_pool,
)
from applications.rag.qarcg_reader import QARCGConfig, make_torch_qarcg, torch_pairwise_rank_loss

CONFIG = ROOT / "configs/experiments/quantum_semantic_reader_screen_100.json"
PLAN = ROOT / "configs/experiments/quantum_semantic_reader_screen_100_plan.json"
METHODS = ("frozen_reader_topk", "quantum_semantic", "classical_semantic", "quantum_scalar_control")


def validate_config(path):
    cfg = old._load_json(path)
    frozen = old._load_json(CONFIG)
    if cfg != frozen or cfg["schema_version"] != "rag.quantum_semantic_reader_screen.v1":
        raise ValueError("screen must match the committed preregistered config")
    if tuple(cfg["methods"]) != METHODS or cfg["evaluation"]["input_sha256"] != old.DEFAULT_INPUT_SHA256:
        raise ValueError("method/input contract changed")
    plan = old._load_json(PLAN)
    if tuple(plan["discovery"]["allowlist"]) != METHODS or tuple(plan["composition"]["order"]) != METHODS:
        raise ValueError("plugin allowlist/order mismatch")
    recovery = plan["recovery_contract"]
    if old._sha256(ROOT / recovery["dossier_path"]) != recovery["dossier_sha256"]:
        raise ValueError("mechanism dossier hash mismatch")
    return cfg


def make_heads(torch, hidden_size, cfg, device):
    spec = QARCGConfig(**cfg["head"])
    seed = cfg["training"]["seed"]
    heads = {
        "quantum_semantic": make_semantic_head(hidden_size, config=spec, seed=seed),
        "classical_semantic": make_semantic_head(hidden_size, config=spec, seed=seed, classical=True),
        "quantum_scalar_control": make_torch_qarcg(spec),
    }
    if sum(p.numel() for p in heads["quantum_semantic"].parameters()) != sum(p.numel() for p in heads["classical_semantic"].parameters()):
        raise ValueError("semantic parameter budgets differ")
    for key in heads["quantum_semantic"].project.state_dict():
        torch.testing.assert_close(heads["quantum_semantic"].project.state_dict()[key], heads["classical_semantic"].project.state_dict()[key], rtol=0, atol=0)
    return {m: h.to(device) for m, h in heads.items()}


def head_forward(head, method, case, device):
    feature = case["scalar"] if method == "quantum_scalar_control" else case["pooled"]
    output = head(case["base"].to(device), feature.to(device))
    return (*output, feature.to(device)) if method == "quantum_scalar_control" else output


def add_values(arrays, prefix, case, capture=None):
    for key in ("base", "pooled", "scalar"):
        arrays[prefix + "__" + key] = case[key].detach().cpu().numpy()
    if "weak_mask" in case:
        arrays[prefix + "__weak_mask"] = case["weak_mask"].cpu().numpy()
    if capture:
        for key, value in capture.items():
            arrays[prefix + "__" + key] = value


def checkpoint(head, method, epoch, arrays):
    for key, value in head.state_dict().items():
        arrays[f"checkpoint__{method}__{epoch}__{key}"] = value.detach().cpu().numpy().copy()


def restore(torch, head, method, epoch, arrays):
    state = {key: torch.as_tensor(arrays[f"checkpoint__{method}__{epoch}__{key}"], device=value.device).clone()
             for key, value in head.state_dict().items()}
    head.load_state_dict(state, strict=True)


def train_head(torch, method, head, cases, cfg, device, arrays):
    t = cfg["training"]
    optimizer = torch.optim.Adam(head.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    checkpoint(head, method, "initial", arrays)
    losses, witnesses = [], []
    head.train()
    for epoch in range(t["epochs"]):
        # Every arm receives the exact same seeded case permutation.
        order = np.random.default_rng(t["seed"] + epoch).permutation(len(cases))
        epoch_losses = []
        for index in order:
            case = cases[int(index)]
            optimizer.zero_grad(set_to_none=True)
            output = head_forward(head, method, case, device)
            scores = output[0]
            ranking = torch_pairwise_rank_loss(scores, case["weak_mask"].to(device))
            anchor = (scores - case["base"].to(device)).square().mean()
            loss = ranking + t["anchor_penalty"] * anchor
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), t["gradient_clip_norm"])
            if not bool(torch.isfinite(gradient)):
                raise ValueError("non-finite training gradient")
            if index < 2:
                prefix = f"train_{index + 1:03d}__{method}__epoch{epoch + 1}_preupdate"
                checkpoint(head, method, f"witness{epoch + 1}_{index + 1}", arrays)
                for key, value in zip(("scores", "gate", "residual", "observables", "encoded"), output):
                    arrays[prefix + "__" + key] = value.detach().cpu().numpy()
                witnesses.append({"sample": int(index) + 1, "epoch": epoch + 1,
                                  "loss": float(loss.detach()), "gradient_norm": float(gradient.detach()),
                                  "value_prefix": prefix})
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
        checkpoint(head, method, str(epoch + 1), arrays)
    head.eval()
    return {"case_count": len(cases), "mean_loss_by_epoch": losses, "witnesses": witnesses}


def _metric_values(case, selected):
    candidates = case["top_50"]
    silver = set(old._posthoc_silver_ids(case))
    result = {"fixed_silver_set_overlap": sum(str(candidates[i]["id"]) in silver for i in selected)}
    for key, field in (("positive_consensus_count", "positive_consensus"),
                       ("direct_consensus_count", "direct_consensus"),
                       ("all_models_positive_count", "positive_all_models")):
        values = [candidates[i]["evidence"][field] for i in selected]
        if not all(type(x) is bool for x in values):
            raise ValueError("invalid posthoc panel label")
        result[key] = sum(values)
    return result


def paired_delta(target, reference, seed, reps=2000):
    delta = np.asarray(target) - np.asarray(reference)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(reps, len(delta)))].mean(1)
    return {"mean": float(delta.mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())}


def diagnose(cases, trace, cfg):
    if len(cases) != cfg["evaluation"]["cases"] or len(trace) != len(cases):
        raise ValueError("diagnostic population mismatch")
    metrics = ("positive_consensus_count", "direct_consensus_count", "all_models_positive_count", "fixed_silver_set_overlap")
    values = {m: {k: [] for k in metrics} for m in METHODS}
    for case, ct in zip(cases, trace):
        for method in METHODS:
            mt = ct["methods"][method]
            scores = mt["score_vector"]
            selected = old._stable_top(scores, case["top_50"])
            measured = _metric_values(case, selected)
            mt["posthoc"] = measured
            for key, value in measured.items():
                values[method][key].append(value)
    comparisons = {}
    for offset, reference in enumerate(("frozen_reader_topk", "classical_semantic", "quantum_scalar_control")):
        comparisons[reference] = {key: paired_delta(values["quantum_semantic"][key], values[reference][key],
                                                   cfg["evaluation"]["bootstrap_seed"] + offset,
                                                   cfg["evaluation"]["bootstrap_replicates"])
                                  for key in metrics}
    utility = comparisons["frozen_reader_topk"]["positive_consensus_count"]
    protection = comparisons["frozen_reader_topk"]["direct_consensus_count"]
    attribution = comparisons["classical_semantic"]["positive_consensus_count"]
    return {"methods": {m: {k: {"total": sum(v), "mean": float(np.mean(v))} for k, v in fields.items()}
                        for m, fields in values.items()},
            "paired_quantum_semantic_minus": comparisons,
            "gates": {"selection_diagnostic": utility["mean"] > 0 and utility["ci95"][0] > 0 and protection["mean"] >= 0,
                      "quantum_attribution_diagnostic": attribution["mean"] > 0 and attribution["ci95"][0] > 0},
            "metric_population": "all 100 cases; per-case count; no zero-positive cases discarded"}


def validate_values(torch, cfg, heads, arrays, metadata, device):
    """Reopen detached value archive, check pooling and replay every evaluation arm."""
    current = {m: {k: v.detach().clone() for k, v in h.state_dict().items()} for m, h in heads.items()}
    try:
        for method, head in heads.items():
            restore(torch, head, method, str(cfg["training"]["epochs"]), arrays)
        eval_count = sum(row["lane"] == "evaluation" for row in metadata)
        expected_eval = int(cfg["evaluation"]["cases"])
        if eval_count != expected_eval:
            raise ValueError("evaluation trace population mismatch")
        prefixes = [row["prefix"] for row in metadata]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("duplicate detached sample identity")
        for lane in ("training", "evaluation"):
            rows = [row for row in metadata if row["lane"] == lane]
            if not rows or any((row["prefix"] + "__hidden" in arrays) != (i < 2) for i, row in enumerate(rows)):
                raise ValueError("full hidden sample selection contract violated")
        for row in metadata:
            prefix = row["prefix"]
            base = torch.as_tensor(arrays[prefix + "__base"])
            pooled = torch.as_tensor(arrays[prefix + "__pooled"])
            scalar = torch.as_tensor(arrays[prefix + "__scalar"])
            if base.shape != (50,) or pooled.shape != (50, 5 * heads["quantum_semantic"].hidden_size) or scalar.shape != (50, 4):
                raise ValueError("detached representation shape mismatch")
            if not all(bool(torch.isfinite(x).all()) for x in (base, pooled, scalar)):
                raise ValueError("non-finite detached values")
            if prefix + "__hidden" in arrays:
                hidden = torch.as_tensor(arrays[prefix + "__hidden"])[None]
                qm = torch.as_tensor(arrays[prefix + "__question_mask"])[None]
                pm = torch.as_tensor(arrays[prefix + "__passage_mask"])[None]
                actual, weights = semantic_pool(hidden, qm, pm)
                torch.testing.assert_close(actual[0], pooled[0], rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(weights[0], torch.as_tensor(arrays[prefix + "__pooling_weights"]), rtol=1e-4, atol=1e-4)
            if row["lane"] != "evaluation":
                usable_index = row["usable_case_index"]
                for method, head in heads.items():
                    for epoch in range(1, cfg["training"]["epochs"] + 1):
                        restore(torch, head, method, f"witness{epoch}_{usable_index}", arrays)
                        witness = f"train_{usable_index:03d}__{method}__epoch{epoch}_preupdate"
                        with torch.no_grad():
                            output = head_forward(head, method, {"base": base, "pooled": pooled, "scalar": scalar}, device)
                        for key, value in zip(("scores", "gate", "residual", "observables", "encoded"), output):
                            torch.testing.assert_close(value.cpu(), torch.as_tensor(arrays[witness + "__" + key]), rtol=1e-4, atol=1e-4)
                    restore(torch, head, method, str(cfg["training"]["epochs"]), arrays)
                continue
            for method, head in heads.items():
                with torch.no_grad():
                    output = head_forward(head, method, {"base": base, "pooled": pooled, "scalar": scalar}, device)
                for key, value in zip(("scores", "gate", "residual", "observables", "encoded"), output):
                    torch.testing.assert_close(value.cpu(), torch.as_tensor(arrays[prefix + "__" + method + "__" + key]), rtol=1e-4, atol=1e-4)
                limit = cfg["head"]["residual_bound"] * max(float(base.std(unbiased=False)), 1e-6)
                if float(output[2].abs().max()) > limit + 1e-5:
                    raise ValueError("residual bound violation")
        for method, head in heads.items():
            restore(torch, head, method, "initial", arrays)
            row = metadata[0]; prefix = row["prefix"]
            case = {key: torch.as_tensor(arrays[prefix + "__" + key]) for key in ("base", "pooled", "scalar")}
            with torch.no_grad():
                scores = head_forward(head, method, case, device)[0]
            torch.testing.assert_close(scores.cpu(), case["base"], rtol=0, atol=0)
    finally:
        for method, head in heads.items():
            head.load_state_dict(current[method])
    return {"detached_values": "pass", "pooling_slices": "pass", "training_witness_replay": "pass", "all_evaluation_replay": "pass", "initial_null": "pass"}


def validate_trace_file(path, archive_path, cfg):
    """Project v2 schema: check saved JSON against detached scores and selection."""
    trace = old._load_json(path)
    if trace.get("schema_version") != "sample-trace.v2" or trace.get("values_sha256") != old._sha256(archive_path):
        raise ValueError("trace schema/archive hash mismatch")
    cases = trace.get("cases", [])
    if len(cases) != cfg["evaluation"]["cases"]:
        raise ValueError("JSON trace population mismatch")
    with np.load(archive_path, allow_pickle=False) as arrays:
        for number, case in enumerate(cases, 1):
            ranks = case["candidate_retrieved_ranks"]
            ids = case["candidate_id_sha256"]
            if case["case_number"] != number or sorted(ranks) != list(range(1, 51)) or len(ids) != 50 or len(set(ids)) != 50:
                raise ValueError("JSON candidate identity/order mismatch")
            if tuple(case["methods"]) != METHODS:
                raise ValueError("JSON method allowlist mismatch")
            for method, row in case["methods"].items():
                prefix = f"evaluation_{number:03d}__{method}__"
                scores = np.asarray(row["score_vector"])
                np.testing.assert_array_equal(scores, arrays[prefix + "scores"])
                for field in ("gate", "residual", "observables", "encoded"):
                    if method != "frozen_reader_topk":
                        np.testing.assert_array_equal(row[field], arrays[prefix + field])
                selected = sorted(range(50), key=lambda i: (-float(scores[i]), ranks[i]))[:5]
                if row["selected_retrieved_ranks"] != [ranks[i] for i in selected] or row["selected_id_sha256"] != [ids[i] for i in selected]:
                    raise ValueError("JSON Top-5 does not match archived scores")
    return {"project_trace_structure": "pass", "project_trace_selection": "pass", "json_archive_agreement": "pass"}


def produce_fixture(directory, cfg):
    import torch
    torch.manual_seed(20261008)
    hidden = torch.randn(50, 12, 8)
    qm = torch.zeros(50, 12, dtype=torch.bool); qm[:, 1:4] = True
    pm = torch.zeros_like(qm); pm[:, 4:11] = True
    pooled, weights = semantic_pool(hidden, qm, pm)
    case = {"base": torch.linspace(3, -3, 50), "pooled": pooled.detach(),
            "scalar": torch.randn(50, 4).tanh(), "weak_mask": torch.arange(50) < 5}
    heads = make_heads(torch, 8, cfg, torch.device("cpu"))
    arrays, meta = {}, []
    training, trace = {}, []
    for method, head in heads.items():
        training[method] = train_head(torch, method, head, [case], cfg, "cpu", arrays)
    for lane in ("training", "evaluation"):
        prefix = lane + "_001"
        add_values(arrays, prefix, case, {"hidden": hidden[0].numpy(), "input_ids": np.arange(12),
                                        "question_mask": qm[0].numpy(), "passage_mask": pm[0].numpy(),
                                        "pooling_weights": weights[0].numpy()})
        meta.append({"lane": lane, "prefix": prefix, "usable_case_index": 1})
        if lane == "evaluation":
            ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(50)]
            ct = {"case_number": 1, "candidate_retrieved_ranks": list(range(1, 51)), "candidate_id_sha256": ids, "methods": {}}
            outputs = {"frozen_reader_topk": (case["base"],)}
            for method, head in heads.items():
                with torch.no_grad():
                    outputs[method] = head_forward(head, method, case, "cpu")
            for method, output in outputs.items():
                scores = output[0].numpy()
                selected = sorted(range(50), key=lambda i: (-float(scores[i]), i))[:5]
                row = {"score_vector": scores.tolist(), "selected_retrieved_ranks": [i + 1 for i in selected], "selected_id_sha256": [ids[i] for i in selected]}
                for key, value in zip(("scores", "gate", "residual", "observables", "encoded"), output):
                    arrays[prefix + "__" + method + "__" + key] = value.numpy()
                    if key != "scores":
                        row[key] = value.numpy().tolist()
                ct["methods"][method] = row
            trace.append(ct)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "semantic_values.npz", **arrays)
    # The synthetic cohort has one evaluation case; production remains locked to 100.
    fixture_cfg = {**cfg, "evaluation": {**cfg["evaluation"], "cases": 1}}
    with np.load(directory / "semantic_values.npz", allow_pickle=False) as archive:
        verified = validate_values(torch, fixture_cfg, heads, archive, meta, "cpu")
    old._write_json(directory / "fixture_trace.json", {"schema_version": "sample-trace.v2", "synthetic_only": True,
                    "population": meta, "training": training, "cases": trace, "values_sha256": old._sha256(directory / "semantic_values.npz"),
                    "validation": verified, "circuit": circuit_resource_contract()})
    verified.update(validate_trace_file(directory / "fixture_trace.json", directory / "semantic_values.npz", fixture_cfg))
    return verified


def run(args):
    cfg = validate_config(args.config)
    if args.validate_only:
        print(json.dumps({"status": "valid", "stage": cfg["stage"], "methods": METHODS})); return
    if args.preflight:
        print(json.dumps(produce_fixture(args.preflight, cfg))); return
    if args.upload and not os.environ.get(args.token_env):
        raise ValueError("configure the exchange token environment variable before running")
    input_path, temporary = old._prepare_input(args.input, exchange_url=args.exchange_url, token_env=args.token_env)
    try:
        cases = old._load_eval_cases(input_path)
        torch, tokenizer, reader, device, reader_identity = old._load_reader({"reader": cfg["reader"]}, args.device)
        if reader_identity["resolved_revision"] != cfg["reader"]["revision"]:
            raise ValueError("Reader revision mismatch")
        for parameter in reader.parameters():
            parameter.requires_grad_(False)
        from applications.rag.data import load_dataset_for_rag, make_corpus_manager
        from applications.rag.retrieval import make_encoder
        questions = list(load_dataset_for_rag("nq_open", "train", cfg["training"]["max_questions"]))[:cfg["training"]["max_questions"]]
        if len(questions) != cfg["training"]["max_questions"]:
            raise ValueError("training population incomplete")
        eval_hashes = {hashlib.sha256(c["question"].encode()).hexdigest() for c in cases}
        train_hashes = [hashlib.sha256(q["question"].encode()).hexdigest() for q in questions]
        if set(train_hashes) & eval_hashes or len(set(train_hashes)) != len(train_hashes):
            raise ValueError("training/evaluation overlap or duplicate training questions")
        manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64})
        manager.build(questions)
        encoder = make_encoder("dpr")
        arrays, metadata, train_cases, training_identity = {}, [], [], []
        skipped = Counter()
        for question in questions:
            records, _ = old._retrieve(manager, encoder.encode_queries([question["question"]])[0], 50)
            texts = [old._online_text(p) for p in records]
            positive, reason = old._weak_positive_mask(texts, question["answers"])
            if reason != "ok":
                skipped[reason] += 1; continue
            number = len(train_cases) + 1
            base, pooled, scalar, capture = reader_semantic_forward(torch, tokenizer, reader, device, question["question"], texts,
                                         max_length=cfg["reader"]["max_length"], batch_size=cfg["reader"]["batch_size"], capture_first=number <= 2)
            case = {"base": base, "pooled": pooled, "scalar": scalar, "weak_mask": torch.tensor(positive, dtype=torch.bool)}
            train_cases.append(case)
            training_identity.append({"question_sha256": hashlib.sha256(question["question"].encode()).hexdigest(),
                                      "candidate_id_sha256": [hashlib.sha256(str(p["id"]).encode()).hexdigest() for p in records],
                                      "positive_mask": positive})
            if number <= 2:
                prefix = f"training_{number:03d}"; add_values(arrays, prefix, case, capture)
                metadata.append({"lane": "training", "prefix": prefix, "usable_case_index": number})
            if number % 32 == 0:
                print(f"Training extraction: {number} usable cases", flush=True)
        coverage = len(train_cases) / len(questions)
        if coverage < cfg["training"]["minimum_coverage"]:
            raise ValueError("weak training coverage below preregistered gate")
        heads = make_heads(torch, reader.config.hidden_size, cfg, device)
        training = {method: train_head(torch, method, head, train_cases, cfg, device, arrays) for method, head in heads.items()}
        trace = []
        for number, case in enumerate(cases, 1):
            candidates = case["top_50"]; texts = [old._online_text(p) for p in candidates]
            base, pooled, scalar, capture = reader_semantic_forward(torch, tokenizer, reader, device, case["question"], texts,
                                       max_length=350, batch_size=cfg["reader"]["batch_size"], capture_first=number <= 2)
            values = {"base": base, "pooled": pooled, "scalar": scalar}
            prefix = f"evaluation_{number:03d}"; add_values(arrays, prefix, values, capture)
            metadata.append({"lane": "evaluation", "prefix": prefix, "case_number": number})
            ct = {"case_number": number, "question_sha256": hashlib.sha256(case["question"].encode()).hexdigest(),
                  "candidate_retrieved_ranks": [p["retrieved_rank"] for p in candidates],
                  "candidate_id_sha256": [hashlib.sha256(str(p["id"]).encode()).hexdigest() for p in candidates], "methods": {}}
            outputs = {"frozen_reader_topk": (base,)}
            for method, head in heads.items():
                with torch.no_grad():
                    outputs[method] = head_forward(head, method, values, device)
            for method, output in outputs.items():
                scores = output[0].detach().cpu().numpy()
                selected = old._stable_top(scores, candidates)
                mt = {"score_vector": scores.tolist(), "selected_retrieved_ranks": [candidates[i]["retrieved_rank"] for i in selected],
                      "selected_id_sha256": [ct["candidate_id_sha256"][i] for i in selected]}
                for key, value in zip(("scores", "gate", "residual", "observables", "encoded"), output):
                    array = value.detach().cpu().numpy(); arrays[prefix + "__" + method + "__" + key] = array
                    if key in ("gate", "residual", "observables", "encoded"):
                        mt[key] = array.tolist()
                ct["methods"][method] = mt
            trace.append(ct)
            if number % 10 == 0:
                print(f"Evaluation ranking: {number}/100", flush=True)
        # Silver fields are opened only after all arms have ranked all evaluation cases.
        evaluation = diagnose(cases, trace, cfg)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(run_dir / "semantic_values.npz", **arrays)
        with np.load(run_dir / "semantic_values.npz", allow_pickle=False) as archive:
            verified = validate_values(torch, cfg, heads, archive, metadata, device)
        provenance = {"code_revision": old._git_revision(), "config_sha256": old._sha256(args.config), "plugin_plan_sha256": old._sha256(PLAN),
                      "effective_config": cfg, "input_sha256": old._sha256(input_path), "reader": reader_identity,
                      "source_hashes": {str(Path(p).relative_to(ROOT)): old._sha256(Path(p)) for p in (__file__, ROOT / "applications/rag/quantum_semantic_reader.py", ROOT / "applications/rag/qarcg_reader.py")},
                      "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__}
        summary = {"schema_version": "rag.quantum_semantic_reader_screen.summary.v1", "stage": "exploratory_screen",
                   "claim_ceiling": "L0_diagnostic", "evaluation": evaluation,
                   "training": {"requested": len(questions), "usable": len(train_cases), "coverage": coverage, "skipped": dict(skipped), "arms": training},
                   "parameter_counts": {m: sum(p.numel() for p in h.parameters()) for m, h in heads.items()},
                   "validation": verified, "provenance": provenance, "circuit": circuit_resource_contract(),
                   "generator_called": False, "evaluator_called": False, "silver_used_online": False,
                   "limitations": cfg["limitations"]}
        if old._contains_forbidden(summary):
            raise ValueError("compact summary contains forbidden raw fields")
        old._write_json(run_dir / "summary.json", summary)
        old._write_json(run_dir / "selector_trace.json", {"schema_version": "sample-trace.v2", "artifact_type": "quantum_semantic_reader_selector_trace",
                        "input_sha256": old._sha256(input_path), "population": metadata, "training_identity": training_identity,
                        "cases": trace, "provenance": provenance, "values_sha256": old._sha256(run_dir / "semantic_values.npz"), "validation": verified})
        verified.update(validate_trace_file(run_dir / "selector_trace.json", run_dir / "semantic_values.npz", cfg))
        summary["validation"] = verified
        old._write_json(run_dir / "summary.json", summary)
        report = ["# Quantum semantic Reader screen", "", "Exploratory L0 diagnostic; weak training labels; no Generator/evaluator.", ""]
        for method, metrics in evaluation["methods"].items():
            report.append(f"- {method}: positive {metrics['positive_consensus_count']['mean']:.4f}/5; direct {metrics['direct_consensus_count']['mean']:.4f}/5; overlap {metrics['fixed_silver_set_overlap']['mean']:.4f}/5")
        report.extend(["", "Gates: " + json.dumps(evaluation["gates"]), "", "Compact outputs are mirrored to exchange as well as GitHub; raw trace/values are exchange-only."])
        (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        files = ["summary.json", "report.md", "selector_trace.json", "semantic_values.npz"]
        records = [{"name": name, "bytes": (run_dir / name).stat().st_size, "sha256": old._sha256(run_dir / name)} for name in files]
        old._write_json(run_dir / "run_metadata.json", {"run_id": run_id, "files": records, "provenance": provenance, "validation": verified})
        files.append("run_metadata.json")
        for name in ("summary.json", "report.md", "run_metadata.json"):
            if (run_dir / name).stat().st_size > old.MAX_GITHUB_BYTES:
                raise ValueError("compact file exceeds GitHub limit")
        manifest = {"target_directory": cfg["output_namespace"] + "/" + run_id,
                    "exchange_files": [{"name": name, "bytes": (run_dir / name).stat().st_size, "sha256": old._sha256(run_dir / name)} for name in files],
                    "github_files": ["summary.json", "report.md", "run_metadata.json", "upload_manifest.json"],
                    "exchange_only": ["selector_trace.json", "semantic_values.npz"], "provenance": provenance}
        old._write_json(run_dir / "upload_manifest.json", manifest)
        if args.upload:
            from scripts.collab.lib.exchange_upload import upload_manifest
            receipts = upload_manifest(run_dir / "upload_manifest.json", base_url=args.exchange_url, token_env=args.token_env)
            old._write_json(run_dir / "upload_receipts.json", {"receipts": receipts})
            # Mirror the manifest too, without a self-referential hash entry.
            from scripts.collab.lib.exchange_upload import _upload_one
            _upload_one(args.exchange_url, os.environ[args.token_env], run_dir / "upload_manifest.json",
                        manifest["target_directory"] + "/upload_manifest.json")
        print(json.dumps({"output_dir": str(run_dir), "uploaded": args.upload, "gates": evaluation["gates"]}))
    finally:
        if temporary:
            temporary.cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--input", type=Path)
    p.add_argument("--output-root", type=Path, default=ROOT / "exchange/five_ideas/quantum_semantic_reader_screen_100")
    p.add_argument("--device")
    p.add_argument("--exchange-url", default=old.DEFAULT_EXCHANGE_URL)
    p.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--preflight", type=Path)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Semantic screen failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
