#!/usr/bin/env python3
"""Synthetic detached step/optimizer replay, not a Reader/data experiment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def synthetic_trace():
    from applications.rag import support_supervision_axes as axes
    qid = axes.text_hash("explicit synthetic fixture question")
    candidates = []
    for i in range(50):
        text = "Synthetic passage " + str(i) + ": fixture relation."
        title = "Synthetic " + str(i)
        candidates.append({"id": str(i), "review_item_id": axes.text_hash(qid + ":" + str(i)),
            "retrieved_rank": i + 1, "retrieval_score": float(50 - i), "reader_score": float(50 - i),
            "reader_rank": i + 1, "text": text, "title": title, "text_sha256": axes.text_hash(text),
            "title_sha256": axes.text_hash(title)})
    return {"schema_version": axes.TRACE_SCHEMA, "case_count": 1, "training_authorized": False,
        "evaluation_question_sha256": [axes.text_hash("excluded fixture " + str(i)) for i in range(100)],
        "stages": ["data", "retrieval", "frozen_reader_scoring", "baseline_selection", "pending_review"],
        "hidden_tensor_capture": False, "cases": [{"usable_case_index": 1, "question_sha256": qid,
        "question": "explicit synthetic fixture question", "answers": ["fixture answer"],
        "lane": "previous_training_diagnostic_not_heldout", "future_role": "training",
        "candidates": candidates, "reader_top5_indices": [0, 1, 2, 3, 4]}]}


def create_fixture(directory):
    import numpy as np
    import torch
    from applications.rag import support_supervision_axes as axes
    from applications.rag import support_boundary_bridge as bridge
    from applications.rag import protected_boundary_learning as learning
    from applications.rag.quantum_semantic_reader import make_semantic_head
    from applications.rag.qarcg_reader import QARCGConfig
    from applications.rag.semantic_reader_training import make_optimizer, save_optimizer, SCALED
    cfg = json.loads((ROOT / "configs/experiments/support_boundary_bridge.json").read_text(encoding="utf-8"))
    trace = synthetic_trace()
    candidates = trace["cases"][0]["candidates"]
    annotations = axes.pending_template(trace); row = annotations["cases"][0]
    row["scope"] = {k: "explicit synthetic scope, not a real question" for k in axes.SCOPE_FIELDS}
    def qual(record, scope=False):
        binding = axes.qualification_binding(record, scope=scope)
        return {"status": "qualified", "confidence": "independently_adjudicated", "evidence": [
            {"reviewer": name, "role": role, "artifact_ref": "synthetic-not-real:" + name,
             "sha256": axes.text_hash(name), "annotation_sha256": binding, "prior_review_exposure": False}
            for name, role in (("fixture-primary", "primary_review"), ("fixture-independent", "independent_review"))]}
    for index, label in ((0, "direct"), (1, "irrelevant"), (2, "partial"), (7, "direct")):
        item = next(p for p in row["items"] if p["review_item_id"] == candidates[index]["review_item_id"])
        item.update(question_support=label, reference_coverage="exact" if label == "direct" else "not_supported",
                    inference_type="none" if label == "irrelevant" else "explicit", source_grounding="single_passage",
                    rationale="Hand-declared synthetic interface target, not semantic certification",
                    support_quotes=[item["text"]] if label != "irrelevant" else [])
        if index != 2:
            item["qualification"] = qual(item)
    row["scope_qualification"] = qual(row, True)
    target = bridge.protected_draft(trace, annotations)[0]["cases"][0]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261011)
        pooled = torch.randn(50, 40)
    base = torch.arange(50, 0, -1, dtype=torch.float32)
    arrays = {"input__base": base.numpy().copy(), "input__pooled": pooled.numpy().copy()}
    records = []
    training = {"learning_rate": .003, "weight_decay": .0001}
    loss_settings = {"margin": cfg["loss"]["pair_margin"], "anchor_weight": cfg["loss"]["anchor_weight"],
                     "protection_weight": cfg["loss"]["protection_weight"]}
    for classical in (False, True):
        method = "classical" if classical else "quantum"
        head = make_semantic_head(8, classical=classical, config=QARCGConfig(residual_bound=cfg["residual_bound"]))
        optimizer = make_optimizer(torch, head, training, SCALED)
        for step in (1, 2):
            prefix = method + "__step" + str(step)
            for name, value in head.state_dict().items(): arrays[prefix + "__before__" + name] = value.detach().numpy().copy()
            save_optimizer(optimizer, head, arrays, prefix + "__optimizer_before")
            result = learning.boundary_training_step(head, optimizer, base, pooled, target, **loss_settings)
            arrays[prefix + "__scores_before"] = result["scores_before"].numpy().copy()
            for name, grad in result["data_gradients_before_clip"].items(): arrays[prefix + "__gradient__" + name] = grad.numpy().copy()
            for name, value in head.state_dict().items(): arrays[prefix + "__after__" + name] = value.detach().numpy().copy()
            save_optimizer(optimizer, head, arrays, prefix + "__optimizer_after")
            with torch.no_grad():
                values = head(base, pooled)
            for key, value in zip(("scores", "gate", "residual", "observables", "encoding"), values):
                arrays[prefix + "__" + key] = value.detach().numpy().copy()
            records.append({"method": method, "step": step, "loss": result["loss"],
                            "selected_indices": learning.protected_selection(values[0], target)})
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory / "values.npz", **arrays)
    manifest = {"schema_version": "rag.protected_boundary_engineering_fixture.v1", "synthetic_only": True,
        "sample_selection_rule": "one predeclared synthetic case, seed20261011, both arms at steps1/2",
        "target": target, "training": training, "loss_settings": loss_settings, "residual_bound": cfg["residual_bound"],
        "records": records, "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "values_sha256": hashlib.sha256((directory / "values.npz").read_bytes()).hexdigest(),
        "reader_hidden_states": "unavailable: no Reader invoked; synthetic pooled input",
        "training_authorized": False, "training_consumable": False, "quantum_advantage": False}
    (directory / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = verify_fixture(directory)
    (directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_fixture(directory):
    import numpy as np
    import torch
    from applications.rag import protected_boundary_learning as learning
    from applications.rag.quantum_semantic_reader import make_semantic_head
    from applications.rag.qarcg_reader import QARCGConfig
    from applications.rag.semantic_reader_training import make_optimizer, load_optimizer, save_optimizer, SCALED
    directory = Path(directory)
    manifest = json.loads((directory / "fixture.json").read_text(encoding="utf-8"))
    if hashlib.sha256((directory / "values.npz").read_bytes()).hexdigest() != manifest["values_sha256"]:
        raise ValueError("fixture values hash mismatch")
    with np.load(directory / "values.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    if set(arrays) != set(manifest["arrays"]): raise ValueError("numeric coverage mismatch")
    for key, value in arrays.items():
        meta = manifest["arrays"][key]
        if list(value.shape) != meta["shape"] or str(value.dtype) != meta["dtype"] or not np.isfinite(value).all():
            raise ValueError("numeric shape/dtype/finiteness mismatch")
    base = torch.as_tensor(arrays["input__base"]); pooled = torch.as_tensor(arrays["input__pooled"])
    for record in manifest["records"]:
        method, step = record["method"], record["step"]
        prefix = method + "__step" + str(step)
        head = make_semantic_head(8, classical=method == "classical", config=QARCGConfig(residual_bound=manifest["residual_bound"]))
        head.load_state_dict({k: torch.as_tensor(arrays[prefix + "__before__" + k]) for k in head.state_dict()})
        optimizer = make_optimizer(torch, head, manifest["training"], SCALED)
        load_optimizer(torch, optimizer, head, arrays, prefix + "__optimizer_before")
        result = learning.boundary_training_step(head, optimizer, base, pooled, manifest["target"], **manifest["loss_settings"])
        np.testing.assert_allclose(result["scores_before"].numpy(), arrays[prefix + "__scores_before"], rtol=1e-5, atol=1e-7)
        for name, value in head.state_dict().items(): np.testing.assert_allclose(value.detach().numpy(), arrays[prefix + "__after__" + name], rtol=1e-5, atol=1e-7)
        for name, value in result["data_gradients_before_clip"].items(): np.testing.assert_allclose(value.numpy(), arrays[prefix + "__gradient__" + name], rtol=1e-5, atol=1e-7)
        moments = {}; save_optimizer(optimizer, head, moments, prefix + "__optimizer_after")
        for key, value in moments.items(): np.testing.assert_allclose(value, arrays[key], rtol=1e-5, atol=1e-7)
        values = head(base, pooled)
        for key, value in zip(("scores", "gate", "residual", "observables", "encoding"), values):
            np.testing.assert_allclose(value.detach().numpy(), arrays[prefix + "__" + key], rtol=1e-5, atol=1e-7)
        if learning.protected_selection(values[0], manifest["target"]) != record["selected_indices"]:
            raise ValueError("selection replay mismatch")
        for name, loss in result["loss"].items():
            if abs(loss - record["loss"][name]) > 1e-6: raise ValueError("loss replay mismatch")
    return {"status": "passed", "update_replays": len(manifest["records"]), "complete_arrays": len(arrays),
            "synthetic_only": True, "training_authorized": False, "training_consumable": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_fixture(args.output) if args.verify_only else create_fixture(args.output)))
