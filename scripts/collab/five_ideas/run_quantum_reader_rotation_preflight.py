#!/usr/bin/env python3
"""Offline actual-Reader engineering fixture, never a real-data training run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from applications.rag.quantum_reader_rotation import (RotationConfig, make_rotation_head,
    reader_rotation_hook, stable_top5, top5_retention_objective)

CONFIG = ROOT / "configs/experiments/quantum_reader_rotation_preflight.json"
ARMS = ("quantum", "classical")
OUTPUTS = ("scores", "rotated", "conditioning", "angles", "states", "norm", "padded")
COMMON = ("hidden", "input_ids", "attention_mask", "question_mask", "passage_mask", "pooling_weights", "pooled", "cls", "base", "weight", "bias")
STAGES = ("data", "tokenization", "frozen_encoder", "pooling", "optimization", "feature_rotation", "readout", "selection", "diagnosis")
CANONICAL = ("data", "preprocess", "training", "retrieval", "scoring", "selection", "context", "generation", "evaluation", "diagnosis")
SOURCES = ("applications/rag/quantum_reader_rotation.py", "applications/rag/quantum_semantic_reader.py",
           "applications/rag/protected_boundary_learning.py", "applications/rag/reader_checkpoint_cache.py",
           "scripts/collab/five_ideas/run_quantum_reader_rotation_preflight.py",
           "configs/experiments/quantum_reader_rotation_preflight.json")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def code_revision():
    """Return an explicit checkout revision, or a deterministic source bundle id."""
    explicit = os.environ.get("QORE_CODE_REVISION")
    if explicit:
        return explicit, "explicit_environment"
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "git"
    except (OSError, subprocess.CalledProcessError):
        source_hash = hashlib.sha256("".join(sha(ROOT / path) for path in SOURCES).encode()).hexdigest()
        return "source-bundle-" + source_hash, "source_hash"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_coordinate(path, expected_sha256, key, coordinates):
    if sha(path) != expected_sha256: raise ValueError("value archive hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        return archive_coordinate(archive, key, coordinates)


def archive_coordinate(archive, key, coordinates):
    value = archive[key]
    if len(coordinates) != value.ndim or any(type(i) is not int or not 0 <= i < n for i, n in zip(coordinates, value.shape)):
        raise ValueError("coordinate outside shape")
    return value[tuple(coordinates)].item()


def put(arrays, key, value):
    arrays[key] = value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.array(value, copy=True)


def validate_loading_info(reader, info):
    unused_pooler = {"span_predictor.encoder.bert_model.pooler.dense.weight",
                     "span_predictor.encoder.bert_model.pooler.dense.bias"}
    unexpected = set(info.get("unexpected_keys", []))
    pooler = reader.span_predictor.encoder.bert_model.pooler
    if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")) or unexpected-unused_pooler or (unexpected and pooler is not None):
        raise ValueError("Reader weights do not exactly cover active checkpoint architecture")
    return sorted(unexpected)


def constrained_cpu_runtime():
    """Read runtime limits, not host-level free RAM which hides container quotas."""
    limit = Path("/sys/fs/cgroup/memory.max")
    try:
        value = limit.read_text().strip()
        return value != "max" and int(value) < 2*1024**3
    except (OSError, ValueError):
        return False


def checkpoint(head, arrays, prefix):
    for name, value in head.state_dict().items(): put(arrays, prefix + name, value)


def output_values(head, cls, pooled, arrays, prefix):
    output = head(cls, pooled)
    for name in OUTPUTS: put(arrays, prefix + name, output[name])
    return output


def target(base):
    baseline = stable_top5(base)
    negative = baseline[-1]
    positive = next(i for i in range(50) if i not in baseline)
    roles = ["held"] * 50
    roles[positive], roles[negative] = "qualified_missed_direct", "qualified_replaceable_irrelevant"
    # Explicit interface fixture, deliberately unrelated to semantic truth.
    return {"future_role": "training", "roles": roles, "baseline_top5_indices": baseline,
        "immutable_selected_indices": baseline[:-1], "replaceable_indices": [negative],
        "eligible_positive_indices": [positive], "draft_pairs": [[positive, negative]], "pair_weights": [1.0],
        "maximum_replacement_slots": 1, "missing_selected_direct_qualification": False, "scope_qualified": True}


def fixture_texts(lane):
    question = "Which material is the fictional Luma bridge made of?" if lane == "training" else "Which city hosts the fictional Neri archive?"
    texts = [f"Engineering fixture passage {i}. " + (
        f"The fictional Luma bridge uses material number {i}. It was described in a construction note." if lane == "training" else
        f"The fictional Neri archive is in city number {i}. Its catalogue describes local records.") for i in range(50)]
    return question, texts


def _actual_batch(torch, tokenizer, reader, encoded, head=None):
    from applications.rag.quantum_semantic_reader import reader_token_masks, semantic_pool
    qm, pm = reader_token_masks(encoded, tokenizer)
    capture = {}
    if head is not None:
        with reader_rotation_hook(reader, head, qm, pm, capture): output = reader(**encoded, return_dict=True)
    else:
        handles = []
        def hook(_module, _inputs, result): capture["hidden"] = result[0]
        try:
            handles.append(reader.span_predictor.encoder.register_forward_hook(hook))
            with torch.no_grad(): output = reader(**encoded, return_dict=True)
        finally:
            for handle in handles: handle.remove()
        capture["pooled"], capture["pooling_weights"] = semantic_pool(capture["hidden"], qm, pm)
        capture["cls"] = capture["hidden"][:, 0]
    capture.update(question_mask=qm, passage_mask=pm)
    return output, capture


def actual_forward(torch, tokenizer, reader, encoded, head=None, batch_size=4):
    """Bound frozen-encoder peak memory; retain only final hidden states."""
    outputs, captures = [], []
    for offset in range(0, len(encoded["input_ids"]), batch_size):
        batch = {k: v[offset:offset+batch_size] for k, v in encoded.items()}
        output, capture = _actual_batch(torch, tokenizer, reader, batch, head)
        outputs.append(output); captures.append(capture)
    result = SimpleNamespace(**{key: torch.cat([getattr(out, key) for out in outputs]) for key in ("relevance_logits", "start_logits", "end_logits")})
    return result, {key: torch.cat([capture[key] for capture in captures]) for key in captures[0]}


def optimizer_values(optimizer, head, arrays, prefix):
    state = optimizer.state.get(head.theta, {})
    for key in ("step", "exp_avg", "exp_avg_sq"):
        if key in state: put(arrays, prefix + key, state[key])


def load_checkpoint(torch, cfg, archive, prefix, family, device):
    head = make_rotation_head(torch.as_tensor(archive["training__weight"].copy(), device=device),
        torch.as_tensor(archive["training__bias"].copy(), device=device), family=family, config=RotationConfig(**{k: cfg["head"][k] for k in ("layers", "conditioning", "seed")}))
    head.load_state_dict({k: torch.as_tensor(archive[prefix+k].copy(), device=device) for k in head.state_dict()})
    return head


def verify(directory, device="cpu"):
    """Reopen full values, independent semantic pooling, readout/update/Top5 replay."""
    import torch
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    archive_path = directory / "rotation_values.npz"
    if sha(archive_path) != manifest["values_sha256"]: raise ValueError("archive drift")
    for path, digest in manifest["source_sha256"].items():
        if sha(ROOT / path) != digest: raise ValueError("source drift: " + path)
    with np.load(archive_path, allow_pickle=False) as values:
        if set(values.files) != set(manifest["inventory"]): raise ValueError("inventory mismatch")
        for key, meta in manifest["inventory"].items():
            value = values[key]
            if list(value.shape) != meta["shape"] or str(value.dtype) != meta["dtype"] or value.size != meta["elements"] or not np.isfinite(value).all() or hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != meta["sha256"]:
                raise ValueError("numeric integrity mismatch: " + key)
        replay_count = 0
        for lane in ("training", "evaluation"):
            hidden, qm, pm = [values[lane+"__"+key] for key in ("hidden", "question_mask", "passage_mask")]
            query = (hidden * qm[:, :, None]).sum(1) / qm.sum(1)[:, None]
            query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
            tokens = hidden / np.maximum(np.linalg.norm(hidden, axis=2, keepdims=True), 1e-12)
            affinity = (tokens * query_norm[:, None]).sum(2)
            affinity[~pm] = -np.inf
            attention = np.exp(affinity - affinity.max(1, keepdims=True)); attention /= attention.sum(1, keepdims=True)
            passage = (hidden * attention[:, :, None]).sum(1)
            pooled = np.concatenate((hidden[:, 0], query, passage, query * passage, abs(query - passage)), 1)
            np.testing.assert_allclose(pooled, values[lane+"__pooled"], rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(attention, values[lane+"__pooling_weights"], rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(hidden[:, 0], values[lane+"__cls"])
            weight, bias = values[lane+"__weight"], values[lane+"__bias"]
            np.testing.assert_allclose(hidden[:, 0] @ weight.reshape(-1) + bias.reshape(()), values[lane+"__base"], atol=5e-5, rtol=2e-5)
            cls = torch.as_tensor(values[lane+"__cls"].copy(), device=device)
            pool = torch.as_tensor(values[lane+"__pooled"].copy(), device=device)
            for family in ARMS:
                for stage in ("initial", "final"):
                    head = load_checkpoint(torch, cfg, values, family+"__"+stage+"__", family, device)
                    output = head(cls, pool)
                    for field in OUTPUTS:
                        np.testing.assert_allclose(output[field].detach().cpu().numpy(), values[f"{lane}__{family}__{stage}__{field}"], atol=5e-5, rtol=2e-5)
                    states = output["states"].detach().cpu().numpy()
                    np.testing.assert_allclose(np.linalg.norm(states, axis=-1), 1., atol=4e-6)
                    scores = output["scores"].detach().cpu().numpy()
                    if stage == "initial": np.testing.assert_allclose(scores, values[lane+"__base"], atol=5e-5, rtol=2e-5)
                    else:
                        if stable_top5(scores) != manifest["selected"][lane][family]: raise ValueError("Top5 replay mismatch")
                    replay_count += 1
        cls = torch.as_tensor(values["training__cls"].copy(), device=device)
        pool = torch.as_tensor(values["training__pooled"].copy(), device=device)
        base = torch.as_tensor(values["training__base"].copy(), device=device)
        for family in ARMS:
            head = load_checkpoint(torch, cfg, values, family+"__initial__", family, device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=0)
            for step in range(cfg["training"]["steps"]):
                prefix = f"{family}__update{step}__"
                np.testing.assert_allclose(head.theta.detach().cpu().numpy(), values[prefix+"before"], atol=3e-6, rtol=2e-5)
                optimizer.zero_grad(set_to_none=True)
                parts = top5_retention_objective(head(cls, pool)["scores"], base, manifest["synthetic_target"], **cfg["loss"])
                parts["total"].backward()
                np.testing.assert_allclose(head.theta.grad.cpu().numpy(), values[prefix+"gradient"], atol=3e-5, rtol=2e-4)
                np.testing.assert_allclose(np.array([float(parts[k].detach()) for k in ("total", "pair", "anchor", "protection")]), values[prefix+"loss"], atol=3e-5, rtol=2e-4)
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["training"]["gradient_clip"])
                optimizer.step()
                np.testing.assert_allclose(head.theta.detach().cpu().numpy(), values[prefix+"after"], atol=3e-6, rtol=2e-5)
                for name, value in optimizer.state[head.theta].items():
                    np.testing.assert_allclose(value.cpu().numpy(), values[prefix+"optimizer_after__"+name], atol=3e-5, rtol=2e-4)
            np.testing.assert_allclose(head.theta.detach().cpu().numpy(), values[family+"__final__theta"], atol=3e-6, rtol=2e-5)
        document = json.loads((directory / "sample_trace.json").read_text(encoding="utf-8"))
        expected_required = set(COMMON) | {f"{a}__{c}__{k}" for a in ARMS for c in ("initial", "final") for k in OUTPUTS}
        if set(document["task_contract"]["required_value_representations"]) != expected_required: raise ValueError("trace omitted mechanism inventory")
        if len(document["samples"]) != 2: raise ValueError("trace population mismatch")
        for sample in document["samples"]:
            if [r["stage"] for r in sample["stages"]] != list(STAGES): raise ValueError("trace stage order drift")
            produced = set()
            for stage in sample["stages"]:
                if any(not ref.startswith(("source:", "artifact:")) and ref not in produced for ref in stage["input_refs"]): raise ValueError("broken lineage")
                produced.update(stage["output_refs"])
            reps = {r["id"]: r for r in sample["representations"]}
            if not expected_required <= set(reps): raise ValueError("missing representation")
            for rep in reps.values():
                key = rep["artifact_refs"][0]["path"].split("#", 1)[1]
                value, capture = values[key], rep["value_capture"]
                if rep["shape"] != list(value.shape) or rep["dtype"] != str(value.dtype) or capture["stored_element_count"] != value.size or capture["logical_element_count"] != value.size or capture["status"] != "complete": raise ValueError("incomplete values")
                if any(ref["sha256"] != manifest["values_sha256"] for ref in rep["artifact_refs"]+capture["artifact_refs"]) or capture["reader_projection"]["source_sha256"] != [manifest["values_sha256"]]: raise ValueError("representation source mismatch")
                for coordinates in ([0]*value.ndim, [n-1 for n in value.shape]):
                    if value[tuple(coordinates)].item() != archive_coordinate(values, key, coordinates): raise ValueError("coordinate resolver mismatch")
        key = "evaluation__quantum__final__states"
        if values[key][-1,-1,-1].item() != read_coordinate(archive_path, manifest["values_sha256"], key, [49,40,1023]): raise ValueError("lazy reader mismatch")
    return {"status": "passed", "full_arrays": len(manifest["inventory"]), "head_replays": replay_count,
        "optimizer_update_replays": len(ARMS)*cfg["training"]["steps"], "lineage": "passed", "semantic_readiness": "passed",
        "selection_replay": "passed", "values_sha256": manifest["values_sha256"], "synthetic_only": True,
        "task_utility_tested": False, "generator_called": False, "dataset_downloaded": False}


def trace_document(manifest, arrays, archive):
    required = list(COMMON) + [f"{a}__{c}__{k}" for a in ARMS for c in ("initial", "final") for k in OUTPUTS]
    artifact = {"path": archive.name, "sha256": sha(archive), "bytes": archive.stat().st_size, "visibility": "private"}
    def display(text):
        return {"summary": text, "default_view": "table", "visible_fields": [{"id": "meaning", "label": "阶段含义", "help": text, "value": text}], "visualizations": []}
    def axes(rid, value):
        suffix = rid.split("__")[-1]
        if suffix == "weight" and value.ndim == 1: return ["encoder_feature"]
        if suffix == "bias" and value.ndim == 0: return []
        known = {"hidden": ["retrieval_order_candidate", "input_token", "encoder_feature"],
            "input_ids": ["retrieval_order_candidate", "input_token"], "attention_mask": ["retrieval_order_candidate", "input_token"],
            "question_mask": ["retrieval_order_candidate", "input_token"], "passage_mask": ["retrieval_order_candidate", "input_token"],
            "pooling_weights": ["retrieval_order_candidate", "input_token"], "cls": ["retrieval_order_candidate", "encoder_feature"],
            "pooled": ["retrieval_order_candidate", "CLS_query_passage_product_difference_feature"],
            "conditioning": ["retrieval_order_candidate", "condition_feature"], "angles": ["retrieval_order_candidate", "ordered_gate"],
            "states": ["retrieval_order_candidate", "initial_then_each_gate_checkpoint", "little_endian_amplitude_coordinate"],
            "rotated": ["retrieval_order_candidate", "padded_feature_coordinate"], "padded": ["retrieval_order_candidate", "padded_feature_coordinate"],
            "norm": ["retrieval_order_candidate", "norm_scalar"], "scores": ["retrieval_order_candidate"], "base": ["retrieval_order_candidate"],
            "weight": ["classifier_output", "encoder_feature"], "bias": ["classifier_output"],
            "theta": ["ordered_gate", "bias_then_four_condition_coefficients"], "projection": ["condition_feature", "pooled_feature"],
            "gradient": ["ordered_gate", "bias_then_four_condition_coefficients"], "before": ["ordered_gate", "bias_then_four_condition_coefficients"],
            "after": ["ordered_gate", "bias_then_four_condition_coefficients"], "exp_avg": ["ordered_gate", "bias_then_four_condition_coefficients"],
            "exp_avg_sq": ["ordered_gate", "bias_then_four_condition_coefficients"], "loss": ["total_pair_anchor_protection"], "step": []}
        return known.get(suffix, ["fixed_coordinate_pair_index"]*value.ndim)
    purposes = {"data": "明确的合成题和固定50个候选；不是真实研究数据", "tokenization": "Reader输入token与问题/段落掩码", "frozen_encoder": "冻结Reader末层隐状态和原相关分",
        "pooling": "问题语义汇聚与固定投影输入", "optimization": "仅合成训练目标：参数、损失、梯度和AdamW状态", "feature_rotation": "零初始化与训练后：条件角度和每一门后的完整振幅",
        "readout": "旋转特征输入原分类头；有符号线性读出", "selection": "不使用标签的分数Top5", "diagnosis": "工程回放；不评价真实任务效果"}
    samples = []
    for lane in ("training", "evaluation"):
        specs = [(k, lane+"__"+k) for k in required]
        if lane == "training": specs += [(k, k) for k in arrays if k.startswith(ARMS)]
        reps, stages = [], []
        def owner(key):
            if key.startswith(ARMS) and not "__scores" in key and any(v in key for v in ("__update", "__theta", "__projection", "__low_", "__high_")): return "optimization"
            if key in ("input_ids", "attention_mask", "question_mask", "passage_mask"): return "tokenization"
            if key in ("pooling_weights", "pooled"): return "pooling"
            if key in COMMON: return "frozen_encoder"
            if key.endswith("__scores"): return "readout"
            return "feature_rotation"
        previous = None
        for number, stage in enumerate(STAGES):
            sid = lane+":"+stage
            status = "not_applicable" if stage == "optimization" and lane == "evaluation" else "observed"
            purpose = purposes[stage]
            inputs = [previous] if previous else ["source:"+lane+"_synthetic_texts"]
            if stage == "optimization": inputs += ["source:synthetic_training_target" if lane == "training" else "artifact:final_checkpoint"]
            stages.append({"stage_id": sid, "stage": stage, "canonical_stage": "preprocess" if stage == "tokenization" else "training" if stage == "optimization" else "scoring" if stage in ("frozen_encoder", "pooling", "feature_rotation", "readout") else stage,
                "step": number, "lane": lane, "status": status, "purpose": purpose, "input_refs": inputs, "output_refs": [sid+":result"],
                "outputs": {"representation_ids": [k for k, _ in specs if owner(k) == stage], "selected": manifest["selected"][lane] if stage == "selection" else {}},
                "display": display(purpose), "target_access": "training_only" if stage == "optimization" and lane == "training" else "outside_generation_input",
                **({"reason": "evaluation never updates parameters"} if status == "not_applicable" else {})})
            previous = sid+":result"
        for rid, key in specs:
            value = arrays[key]; ref = {**artifact, "path": archive.name+"#"+key}
            meaning = "完整实际数值："+rid
            reps.append({"id": rid, "stage_id": lane+":"+owner(rid), "status": "observed", "meaning": meaning,
                "shape": list(value.shape), "dtype": str(value.dtype), "axis_semantics": axes(rid, value),
                "artifact_refs": [ref], "display": display(meaning), "value_capture": {"status": "complete", "shape": list(value.shape),
                    "logical_element_count": int(value.size), "stored_element_count": int(value.size), "artifact_refs": [ref],
                    "reader_projection": {"status": "available", "kind": "coordinate_inspector", "coordinate_access": "arbitrary",
                        "reader_ref": "scripts.collab.five_ideas.run_quantum_reader_rotation_preflight.read_coordinate:"+key,
                        "source_sha256": [artifact["sha256"]]}}})
        question, texts = fixture_texts(lane)
        samples.append({"sample_id": lane+"_synthetic", "lane": lane, "split": lane+"_synthetic", "checkpoint": "final_step2",
            "roles": {"input": {"question": question, "candidates": texts}, "reference": {"status": "synthetic_engineering_only", "semantic_gold": False},
                "result": manifest["selected"][lane]}, "stages": stages, "representations": reps})
    return {"schema_version": "sample-trace.v2", "trace_id": manifest["run_id"], "trace_completeness": "complete",
        "experiment": {"run_id": manifest["run_id"], "dataset": "explicit synthetic two-question fixture; cached real Reader",
            "code_revision": manifest["code_revision"], "config_sha256": manifest["config_sha256"], "model_identity": manifest["model_identity"], "seed": manifest["config"]["head"]["seed"]},
        "sample_selection": {"rule": "all two predeclared synthetic questions and all50 candidates", "population_scope": "engineering only", "selected_count": 2, "selected_sample_ids": [s["sample_id"] for s in samples]},
        "coverage": {k: "observed" if k in ("data", "preprocess", "training", "scoring", "selection", "diagnosis") else "not_applicable" for k in CANONICAL},
        "task_contract": {"task_type": "reader_feature_rotation_engineering", "required_roles": ["input", "reference", "result"],
            "required_representations": required, "required_value_representations": required, "required_splits": ["training_synthetic", "evaluation_synthetic"],
            "required_lanes": ["training", "evaluation"], "required_checkpoints": ["final_step2"], "checkpoint_policy": "zero and fixed two-update weights; no score-based checkpoint choice"}, "samples": samples}


def run(args):
    import torch
    import transformers
    from transformers import DPRConfig, DPRReader, DPRReaderTokenizer
    from huggingface_hub import snapshot_download
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["authorization"] != "engineering_fixture_only" or cfg["methods"] != ["frozen_reader", *ARMS] or cfg["training"]["steps"] != 2 or not cfg["reader"]["offline_only"]:
        raise ValueError("unapproved engineering configuration")
    torch.set_num_threads(2)
    if args.verify_only:
        print(json.dumps(verify(args.verify_only, args.device or "cpu"))); return
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    spec = cfg["reader"]
    snapshot = Path(snapshot_download(spec["model_id"], revision=spec["revision"], local_files_only=True))
    if snapshot.name != spec["revision"] or sha(snapshot/"config.json") != spec["config_file_sha256"]:
        raise ValueError("cached model revision/config mismatch")
    tokenizer = DPRReaderTokenizer.from_pretrained(snapshot, local_files_only=True)
    # Legacy non-ZIP Torch files cannot be memory-mapped. Convert bytes before
    # constructing the model, then use meta initialization + mapped safetensors.
    # Keep the original snapshot immutable and preserve every FP32 tensor byte.
    import zipfile
    source_weights = snapshot/"pytorch_model.bin"
    serialization_cache = None
    model_path, loading = snapshot, {}
    if source_weights.exists() and not zipfile.is_zipfile(source_weights):
        from applications.rag.reader_checkpoint_cache import convert_legacy
        original_sha = sha(source_weights)
        model_path = ROOT/".cache/reader_checkpoint_cache"/original_sha
        serialization_cache = convert_legacy(source_weights, model_path)
        loading = {"config": DPRConfig.from_pretrained(snapshot, local_files_only=True), "use_safetensors": True}
        print("Legacy Reader mapped cache ready; all FP32 tensor bytes verified", flush=True)
    disk_offload = device.type == "cpu" and constrained_cpu_runtime()
    if disk_offload:
        from applications.rag.reader_checkpoint_cache import load_streaming_reader
        mapped_archive = model_path/"model.safetensors"
        if not mapped_archive.exists(): raise ValueError("constrained CPU requires mapped safetensors cache")
        print("Limited-memory CPU: per-module mapped frozen encoder; FP32 retained", flush=True)
        reader, unused_checkpoint_keys = load_streaming_reader(DPRConfig.from_pretrained(snapshot, local_files_only=True), mapped_archive)
    else:
        use_low_memory = importlib.util.find_spec("accelerate") is not None
        reader, loading_info = DPRReader.from_pretrained(model_path, local_files_only=True,
            low_cpu_mem_usage=use_low_memory, output_loading_info=True, **loading)
        unused_checkpoint_keys = validate_loading_info(reader, loading_info)
        reader = reader.to(device)
    reader = reader.eval().requires_grad_(False)
    print("Frozen Reader loaded", flush=True)
    identity = {"model_id": spec["model_id"], "revision": snapshot.name, "resolved_path": str(snapshot.resolve()),
        "config_file_sha256": sha(snapshot/"config.json"),
        "runtime_config_sha256": hashlib.sha256(json.dumps(reader.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest(),
        "files_sha256": {p.name: sha(p) for p in snapshot.iterdir() if p.is_file() and p.suffix in (".bin", ".safetensors", ".json", ".txt")},
        "loading": "per-module mapped CPU" if disk_offload else "low_cpu_mem_usage" if use_low_memory else "ordinary resident Reader",
        "serialization_cache": serialization_cache, "unused_checkpoint_keys": unused_checkpoint_keys,
        "disk_offload": disk_offload, "device_map": getattr(reader, "hf_device_map", None)}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    arrays, features, encodings, base_outputs, health = {}, {}, {}, {}, {}
    for lane in ("training", "evaluation"):
        question, texts = fixture_texts(lane)
        encoded = tokenizer(questions=[question]*50, texts=texts, return_tensors="pt", padding=True, truncation=True, max_length=spec["max_length"])
        # DPRReader does not accept token_type_ids; tokenizer masks recover them.
        encoded = {k: v.to(device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
        base, capture = actual_forward(torch, tokenizer, reader, encoded, batch_size=spec["batch_size"])
        encodings[lane], base_outputs[lane] = encoded, base
        features[lane] = {k: v.detach() for k, v in capture.items()}
        features[lane]["base"] = base.relevance_logits.detach()
        print("Frozen Reader fixture " + lane + " complete", flush=True)
        for key in COMMON:
            value = encoded[key] if key in encoded else reader.span_predictor.qa_classifier.weight if key == "weight" else reader.span_predictor.qa_classifier.bias if key == "bias" else features[lane][key]
            put(arrays, lane+"__"+key, value)
    row = target(features["training"]["base"])
    selected = {lane: {"frozen_reader": stable_top5(features[lane]["base"])} for lane in features}
    resource = {}
    for family in ARMS:
        classifier = reader.span_predictor.qa_classifier
        head = make_rotation_head(classifier.weight, classifier.bias, family=family,
            config=RotationConfig(**{k: cfg["head"][k] for k in ("layers", "conditioning", "seed")}))
        if head.theta.numel() != cfg["head"]["parameter_count"]: raise ValueError("parameter budget drift")
        resource[family] = head.resources()
        checkpoint(head, arrays, family+"__initial__")
        for lane, f in features.items():
            initial = output_values(head, f["cls"], f["pooled"], arrays, f"{lane}__{family}__initial__")
            torch.testing.assert_close(initial["scores"], f["base"], rtol=2e-5, atol=5e-5)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=0)
        for step in range(cfg["training"]["steps"]):
            prefix = f"{family}__update{step}__"
            put(arrays, prefix+"before", head.theta)
            optimizer_values(optimizer, head, arrays, prefix+"optimizer_before__")
            optimizer.zero_grad(set_to_none=True)
            f = features["training"]
            if step == 0:
                actual, capture = actual_forward(torch, tokenizer, reader, encodings["training"], head, spec["batch_size"])
                scores = actual.relevance_logits
                torch.testing.assert_close(scores, f["base"], rtol=0, atol=0)
            else: scores = head(f["cls"], f["pooled"])["scores"]
            parts = top5_retention_objective(scores, f["base"], row, **cfg["loss"])
            parts["total"].backward()
            gradient = head.theta.grad
            if gradient is None or not bool(gradient.isfinite().all()) or float(gradient.norm()) == 0: raise ValueError("dead training gradient")
            put(arrays, prefix+"gradient", gradient)
            put(arrays, prefix+"loss", np.array([float(parts[k].detach()) for k in ("total", "pair", "anchor", "protection")]))
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["training"]["gradient_clip"])
            optimizer.step()
            put(arrays, prefix+"after", head.theta)
            optimizer_values(optimizer, head, arrays, prefix+"optimizer_after__")
        checkpoint(head, arrays, family+"__final__")
        for lane, f in features.items():
            final = output_values(head, f["cls"], f["pooled"], arrays, f"{lane}__{family}__final__")
            with torch.no_grad(): actual, _ = actual_forward(torch, tokenizer, reader, encodings[lane], head, spec["batch_size"])
            torch.testing.assert_close(actual.relevance_logits, final["scores"], rtol=2e-5, atol=5e-5)
            torch.testing.assert_close(actual.start_logits, base_outputs[lane].start_logits, rtol=0, atol=0)
            torch.testing.assert_close(actual.end_logits, base_outputs[lane].end_logits, rtol=0, atol=0)
            selected[lane][family] = stable_top5(final["scores"])
            correction = final["scores"].detach()-f["base"]
            if float(correction.std()) <= 1e-7: raise ValueError("uniform score correction")
            health[lane+"__"+family] = {"correction_std": float(correction.std()), "max_abs_correction": float(correction.abs().max()),
                "conditioning_std": final["conditioning"].detach().std(dim=0).cpu().tolist(), "gradient_norm": float(head.theta.grad.norm())}
        print("Adapter/update fixture " + family + " complete", flush=True)
    # Removing the scoped adapter restores the frozen Reader, including all spans.
    restored, _ = actual_forward(torch, tokenizer, reader, encodings["training"], batch_size=spec["batch_size"])
    torch.testing.assert_close(restored.relevance_logits, features["training"]["base"], rtol=0, atol=0)
    if any(p.grad is not None for p in reader.parameters()): raise ValueError("frozen Reader received gradients")
    archive = directory/"rotation_values.npz"
    np.savez_compressed(archive, **arrays)
    revision, revision_source = code_revision()
    manifest = {"schema_version": "rag.reader_rotation_fixture.v1", "run_id": run_id, "config": cfg, "config_sha256": sha(CONFIG),
        "code_revision": revision, "code_revision_source": revision_source,
        "source_sha256": {p: sha(ROOT/p) for p in SOURCES}, "model_identity": identity,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__, "device": str(device)},
        "inventory": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "elements": int(v.size), "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in arrays.items()},
        "values_sha256": sha(archive), "selected": selected, "synthetic_target": row, "resources": resource, "health": health,
        "real_cohort_fitting": False, "model_downloaded": False, "dataset_downloaded": False, "generator_called": False}
    write_json(directory/"manifest.json", manifest)
    write_json(directory/"sample_trace.json", trace_document(manifest, arrays, archive))
    result = verify(directory, str(device))
    result.update({"actual_reader_hook": "passed", "span_parity": "passed", "disabled_restoration": "passed", "reader_frozen": "passed", "health": health})
    write_json(directory/"validation.json", result)
    print(json.dumps({"output_dir": str(directory), "validation": result}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT/".cache/quantum_reader_rotation_preflight")
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--device")
    run(parser.parse_args())


if __name__ == "__main__": main()
