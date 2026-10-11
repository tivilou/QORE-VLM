"""Project-owned v2 exporter, full numeric readback and label-free selection validator."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from applications.rag import protected_boundary_training as training

STAGES = ("data", "frozen_reader", "semantic_pooling", "residual_head", "optimization", "selection", "posthoc")
CANONICAL = ("data", "preprocess", "training", "retrieval", "scoring", "selection", "context", "generation", "evaluation", "diagnosis")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_coordinate(path, expected_sha256, key, coordinates):
    """Private/lazy exact value reader; no invented preview or reconstruction."""
    if file_hash(path) != expected_sha256: raise ValueError("value-reader archive hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        return archive_coordinate(archive, key, coordinates)


def archive_coordinate(archive, key, coordinates):
    value = archive[key]
    if len(coordinates) != value.ndim or any(type(i) is not int or i < 0 or i >= n for i, n in zip(coordinates, value.shape)):
        raise ValueError("coordinate outside declared shape")
    return value[tuple(coordinates)].item()


def display(summary):
    return {"summary": summary, "default_view": "table", "visible_fields": [
        {"id": "detail", "label": "阶段含义", "help": summary, "value": summary}], "visualizations": []}


def trace_document(cfg, metadata, ranked, arrays, archive, cases, provenance, synthetic):
    selected = [r for lane in ("training", "evaluation") for r in [x for x in metadata if x["lane"] == lane][:2]]
    ref = {"path": archive.name, "sha256": file_hash(archive), "bytes": archive.stat().st_size, "visibility": "private"}
    samples = []
    common_reps = ["base", "hidden", "input_ids", "question_mask", "passage_mask", "relevance_weight", "relevance_bias", "pooling_weights", "pooled", "scalar"]
    for method in training.ARMS: common_reps += [method + "__" + k for k in training.FIELDS]
    for row in selected:
        prefix, lane = row["prefix"], row["lane"]
        position = metadata.index(row); case = cases[position]
        number = [r for r in metadata if r["lane"] == lane].index(row)
        record = ranked[lane][number]
        specs = [(key, prefix + "__" + key, "frozen_reader" if key in ("base", "hidden", "input_ids", "question_mask", "passage_mask", "relevance_weight", "relevance_bias") else
                  "semantic_pooling" if key in ("pooling_weights", "pooled", "scalar") else "residual_head") for key in common_reps]
        # Attach every checkpoint and full gradient/moment to selected training questions.
        if lane == "training":
            for key in arrays:
                if key.startswith(tuple(m + "__" for m in training.ARMS)) and ("__sample" + str(number + 1) + "__" in key or "__initial__" in key or "__epoch" in key and "__sample" not in key):
                    specs.append((key, key, "optimization"))
        stages, reps = [], []
        produced = None
        for step, stage in enumerate(STAGES):
            stage_id = prefix + ":" + stage
            representation_ids = [rid for rid, _, owner in specs if owner == stage]
            output_ref = stage_id + ":result"
            inputs = [produced] if produced else ["source:" + row["question_sha256"]]
            if stage == "optimization": inputs += ["source:frozen_training_pairs" if lane == "training" else "artifact:final_epoch_checkpoint"]
            summary = {"data": "固定候选段和题目；答案只作独立参考", "frozen_reader": "冻结DPR Reader：末层隐状态与原始相关分",
                "semantic_pooling": "按问题语义汇聚段落表征", "residual_head": "同预算量子/经典头：编码、交互、门和有界分数纠正",
                "optimization": "固定纠错对训练；完整参数、梯度和AdamW状态" if lane == "training" else "载入训练末轮权重，不在评估题上更新",
                "selection": "所有方法只按分数取Top5，不使用标注掩码", "posthoc": "排名冻结后对照证据标签，不能反馈到模型"}[stage]
            payload = {"representation_ids": representation_ids}
            if stage == "selection": payload["methods"] = {m: {k: v for k, v in mt.items() if k != "posthoc"} for m, mt in record["methods"].items()}
            if stage == "posthoc": payload["diagnostics"] = {m: mt.get("posthoc", "not available in synthetic fixture") for m, mt in record["methods"].items()}
            stages.append({"stage_id": stage_id, "stage": stage, "canonical_stage": {"data": "data", "frozen_reader": "scoring", "semantic_pooling": "preprocess",
                "residual_head": "scoring", "optimization": "training" if lane == "training" else "scoring", "selection": "selection", "posthoc": "diagnosis"}[stage],
                "lane": lane, "step": step, "status": "observed", "purpose": summary,
                "input_refs": inputs, "output_refs": [output_ref], "outputs": payload, "display": display(summary),
                "target_access": "training_only" if stage == "optimization" and lane == "training" else "outside_generation_input",
                "artifact_refs": [ref]})
            produced = output_ref
        for rid, key, owner in specs:
            value = arrays[key]; shape = list(value.shape)
            refs = [{**ref, "path": archive.name + "#" + key}]
            meaning = "完整数值：" + rid
            reps.append({"id": rid, "stage_id": prefix + ":" + owner, "status": "observed", "meaning": meaning,
                "shape": shape, "dtype": str(value.dtype), "axis_semantics": ["axis" + str(i) for i in range(value.ndim)],
                "artifact_refs": refs, "display": display(meaning), "value_capture": {
                    "status": "complete", "shape": shape, "logical_element_count": int(value.size), "stored_element_count": int(value.size),
                    "artifact_refs": refs, "reader_projection": {"status": "available", "kind": "coordinate_inspector", "coordinate_access": "arbitrary",
                    "reader_ref": "scripts.collab.lib.protected_boundary_trace.read_coordinate:" + key, "source_sha256": [ref["sha256"]]}}})
        samples.append({"sample_id": prefix, "lane": lane, "split": "fixture_" + lane if synthetic else lane,
            "checkpoint": "final_epoch", "roles": {"input": {"question": case["question"], "candidate_count": 50},
                "reference": case.get("answers", case.get("gold_answers", ["reference unavailable"])), "result": record["methods"]},
            "representations": reps, "stages": stages})
    return {"schema_version": "sample-trace.v2", "trace_id": "protected-boundary:" + archive.parent.name, "trace_completeness": "complete",
        "experiment": {"run_id": archive.parent.name, "dataset": "synthetic_texts_actual_DPR_Reader" if synthetic else "frozen59_training_and_exposed100_panel",
            "code_revision": provenance["code_revision"], "config_sha256": provenance["config_sha256"], "model_identity": provenance["reader"],
            "seed": cfg["training"]["seed"], "synthetic_only": synthetic},
        "sample_selection": {"rule": "first two questions each lane; full hidden first candidate only; no result-conditioned selection",
            "population_scope": "fixture two+two" if synthetic else "59 training and100 evaluation", "selected_count": len(samples),
            "selected_sample_ids": [s["sample_id"] for s in samples]},
        "coverage": {k: "observed" if k in ("data", "preprocess", "training", "scoring", "selection", "diagnosis") else "not_applicable" for k in CANONICAL},
        "task_contract": {"task_type": "frozen_top50_to5_protected_boundary", "required_roles": ["input", "reference", "result"],
            "required_representations": common_reps, "required_value_representations": common_reps,
            "required_splits": sorted({s["split"] for s in samples}), "required_lanes": ["training", "evaluation"],
            "required_checkpoints": ["final_epoch"], "checkpoint_policy": "initial and every epoch archived; final epoch fixed, no label-based choice"}, "samples": samples}


def verify_trace(document, arrays, metadata, ranked):
    if document["schema_version"] != "sample-trace.v2" or len(document["samples"]) != 4:
        raise ValueError("v2 sample population mismatch")
    required = set(document["task_contract"]["required_value_representations"])
    for sample in document["samples"]:
        if [s["stage"] for s in sample["stages"]] != list(STAGES): raise ValueError("stage order mismatch")
        produced = set()
        for stage in sample["stages"]:
            if any(not r.startswith(("source:", "artifact:", "external:")) and r not in produced for r in stage["input_refs"]):
                raise ValueError("broken producer-owned lineage")
            produced.update(stage["output_refs"])
        reps = {r["id"]: r for r in sample["representations"]}
        if not required <= set(reps): raise ValueError("missing required representations")
        for rep in reps.values():
            key = rep["artifact_refs"][0]["path"].split("#", 1)[1]
            value = arrays[key]; capture = rep["value_capture"]
            if capture["status"] != "complete" or list(value.shape) != rep["shape"] or capture["stored_element_count"] != value.size or str(value.dtype) != rep["dtype"]:
                raise ValueError("representation coordinate coverage mismatch")
    for row in metadata:
        records = [r for r in metadata if r["lane"] == row["lane"]]
        record = ranked[row["lane"]][records.index(row)]
        ranks = record["candidate_retrieved_ranks"]; ids = record["candidate_id_sha256"]
        if sorted(ranks) != list(range(1, 51)) or len(set(ids)) != 50:
            raise ValueError("candidate identity/order mismatch")
        for method, mt in record["methods"].items():
            scores = arrays[row["prefix"] + "__" + method + "__scores"]
            np.testing.assert_array_equal(scores, mt["score_vector"])
            chosen = sorted(range(50), key=lambda i: (-float(scores[i]), ranks[i]))[:5]
            if chosen != mt["selected_indices"] or [ranks[i] for i in chosen] != mt["selected_retrieved_ranks"] or [ids[i] for i in chosen] != mt["selected_id_sha256"]:
                raise ValueError("label-free selection replay failed")
    return {"v2_structure": "passed", "lineage": "passed", "selection_replay": "passed"}


def persist_trace(destination, cfg, heads, arrays, metadata, witnesses, targets, cases, ranked, device, provenance, synthetic=False):
    import torch
    destination = Path(destination)
    archive = destination / "semantic_values.npz"
    np.savez_compressed(archive, **arrays)
    inventory = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "elements": int(v.size),
                      "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k, v in arrays.items()}
    write_json(destination / "value_capture.json", {"values_sha256": file_hash(archive), "arrays": inventory,
                "population": metadata, "optimizer_witnesses": witnesses, "targets": targets, "effective_config": cfg})
    reopened = json.loads((destination / "value_capture.json").read_text(encoding="utf-8"))
    with np.load(archive, allow_pickle=False) as values:
        if set(values.files) != set(inventory): raise ValueError("archive inventory mismatch")
        for key, meta in reopened["arrays"].items():
            value = values[key]
            if list(value.shape) != meta["shape"] or str(value.dtype) != meta["dtype"] or value.size != meta["elements"] or not np.isfinite(value).all() or hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != meta["sha256"]:
                raise ValueError("archive numeric integrity mismatch")
        result = training.verify_values(torch, cfg, heads, values, metadata, witnesses, targets, device)
        document = trace_document(cfg, metadata, ranked, values, archive, cases, provenance, synthetic)
        write_json(destination / "sample_trace.json", document)
        document = json.loads((destination / "sample_trace.json").read_text(encoding="utf-8"))
        result.update(verify_trace(document, values, metadata, ranked))
        # Test the actual reader beyond first-N previews at both ends of each declared tensor.
        for sample in document["samples"]:
            for rep in sample["representations"]:
                key = rep["artifact_refs"][0]["path"].split("#", 1)[1]
                if not values[key].size: continue
                for coordinates in ([0] * len(rep["shape"]), [n - 1 for n in rep["shape"]]):
                    expected = arrays[key][tuple(coordinates)].item()
                    # Opened and hashed archive, actual resolver, original producer coordinates.
                    if expected != archive_coordinate(values, key, coordinates): raise ValueError("coordinate reader mismatch")
        first = document["samples"][0]["representations"][0]
        key = first["artifact_refs"][0]["path"].split("#", 1)[1]
        if read_coordinate(archive, file_hash(archive), key, [49]) != values[key][49].item(): raise ValueError("lazy reader mismatch")
    result.update({"complete_arrays": len(inventory), "values_sha256": file_hash(archive), "synthetic_targets_only": synthetic})
    write_json(destination / "validation.json", result)
    private_cases, md = [], ["# Protected boundary case study", "", "训练与评估分开；Top5只按模型分数；本轮不运行Generator。", ""]
    for row, case in zip(metadata, cases):
        number = [r for r in metadata if r["lane"] == row["lane"]].index(row)
        record = ranked[row["lane"]][number]
        candidates = case.get("candidates", case.get("top_50"))
        private_cases.append({"lane": row["lane"], "question": case["question"], "answers": case.get("answers", case.get("gold_answers", [])),
            "top_50": candidates, "methods": record["methods"], "generation": {"status": "not_applicable", "reason": "selector-only screen; Generator not called"}})
        md += [f"## {row['lane']} {number + 1}: {case['question']}", "", "参考答案：" + json.dumps(private_cases[-1]["answers"], ensure_ascii=False), ""]
        for method, mt in record["methods"].items():
            md += [f"### {method}", "", "Top5召回排名：" + str(mt["selected_retrieved_ranks"]), ""]
            for i in mt["selected_indices"]: md += [f"- rank{candidates[i]['retrieved_rank']}, score{mt['score_vector'][i]:.6f}: {candidates[i]['title']}", candidates[i]["text"], ""]
        md += ["### 固定Top50及各方法分数", ""]
        for i, passage in enumerate(candidates):
            md += [f"rank{passage['retrieved_rank']} {passage['title']} | " + " / ".join(f"{m}: {record['methods'][m]['score_vector'][i]:.6f}" for m in record["methods"]), passage["text"], ""]
    write_json(destination / "case_study.json", {"schema_version": "rag.protected_boundary_case_study.v1", "provenance": provenance, "cases": private_cases})
    (destination / "case_study.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read an exact private NPZ coordinate")
    parser.add_argument("archive", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("key")
    parser.add_argument("coordinates", nargs="*", type=int)
    args = parser.parse_args()
    print(read_coordinate(args.archive, args.sha256, args.key, args.coordinates))
