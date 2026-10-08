"""Bounded engineering qualification on two saved TRAIN cases, never Silver.

64 alternating updates per arm, fixed before execution. This checks optimizer
health and checkpoint/moment replay, not generalization or task improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
from applications.rag.semantic_reader_training import (
    LEGACY, REPAIRED, SCALED, gradient_health, load_optimizer, make_optimizer, norm,
    require_learning, save_optimizer, signal_health,
)
from applications.rag.qarcg_reader import torch_pairwise_rank_loss

INPUT_SHA256 = "4d18cca65cfc8db290020ccc1fea1c5d64773b95be488c9a68d78b9e807c581a"
STEPS = 64


def update(torch, head, method, case, cfg, optimizer):
    optimizer.zero_grad(set_to_none=True)
    output = screen.head_forward(head, method, case, "cpu")
    scores = output[0]
    loss = (torch_pairwise_rank_loss(scores, case["weak_mask"]) +
            cfg["training"]["anchor_penalty"] * (scores - case["base"]).square().mean())
    if not bool(loss.isfinite()):
        raise ValueError("non-finite qualification loss")
    loss.backward()
    gradients = gradient_health(head)
    torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["training"]["gradient_clip_norm"])
    optimizer.step()
    return float(loss.detach()), gradients


def qualify(input_path, output_dir):
    import torch
    torch.set_num_threads(1)
    if screen.old._sha256(input_path) != INPUT_SHA256:
        raise ValueError("saved training archive hash mismatch")
    cfg = screen.validate_config(screen.CONFIG)
    # Only these eight train-role arrays are opened. No evaluation values, scores
    # or panel labels are loaded or used to choose an optimizer/threshold.
    with np.load(input_path, allow_pickle=False) as archive:
        cases = [{k: torch.from_numpy(archive[f"training_{i:03d}__{k}"].copy())
                  for k in ("base", "pooled", "scalar", "weak_mask")} for i in (1, 2)]
    for case in cases:
        if case["weak_mask"].dtype != torch.bool or not 0 < int(case["weak_mask"].sum()) < 50:
            raise ValueError("training target must be mixed weak containment")
        if case["base"].shape != (50,) or case["scalar"].shape != (50, 4):
            raise ValueError("training shape mismatch")
        if not all(bool(v.isfinite().all()) for v in case.values()):
            raise ValueError("non-finite saved train case")
    hidden_size = cases[0]["pooled"].shape[1] // 5
    arrays, results = {}, {}
    for i, case in enumerate(cases, 1):
        screen.add_values(arrays, f"training_{i:03d}", case)
    for method in ("quantum_semantic", "classical_semantic"):
        for policy in (LEGACY, REPAIRED, SCALED):
            key = method + "__" + policy
            head = screen.make_heads(torch, hidden_size, cfg, "cpu")[method]
            optimizer = make_optimizer(torch, head, cfg["training"], policy)
            initial_norm = norm(head.project.weight)
            initial_output = [screen.head_forward(head, method, c, "cpu") for c in cases]
            for output, case in zip(initial_output, cases):
                torch.testing.assert_close(output[0], case["base"], rtol=0, atol=0)
            screen.checkpoint(head, key, "initial", arrays)
            history = []
            for step in range(STEPS):
                if step == STEPS - 1:
                    screen.checkpoint(head, key, "replay_pre", arrays)
                    save_optimizer(optimizer, head, arrays, key + "__optimizer_pre")
                loss, grads = update(torch, head, method, cases[step % 2], cfg, optimizer)
                history.append({"step": step + 1, "case": step % 2 + 1, "loss": loss, "data_gradient_norms": grads})
            screen.checkpoint(head, key, "final", arrays)
            save_optimizer(optimizer, head, arrays, key + "__optimizer_final")
            outputs = [screen.head_forward(head, method, c, "cpu") for c in cases]
            health = signal_health(head, outputs, [c["base"] for c in cases], initial_norm)
            gradient_totals = {k: sum(row["data_gradient_norms"][k] for row in history[1:])
                               for k in history[0]["data_gradient_norms"]}
            try:
                require_learning(health, gradient_totals)
                passed, failure = True, None
            except ValueError as exc:
                passed, failure = False, str(exc)
            for i, output in enumerate(outputs, 1):
                for name, value in zip(("scores", "gate", "residual", "observables", "encoded"), output):
                    arrays[f"{key}__training_{i:03d}__{name}"] = value.detach().numpy().copy()
            results[key] = {"policy": policy, "exact_initial_null": True, "health": health,
                            "passed": passed, "failure": failure, "history": history,
                            "optimizer_groups": [{k: g[k] for k in ("parameter_names", "weight_decay", "lr") if k in g}
                                                 for g in optimizer.param_groups]}
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "qualification_values.npz", **arrays)
    # Reopen complete inputs, parameters and moments; replay the final update.
    with np.load(output_dir / "qualification_values.npz", allow_pickle=False) as archive:
        reopened = [{k: torch.from_numpy(archive[f"training_{i:03d}__{k}"].copy())
                     for k in ("base", "pooled", "scalar", "weak_mask")} for i in (1, 2)]
        for key, result in results.items():
            method = "classical_semantic" if key.startswith("classical") else "quantum_semantic"
            head = screen.make_heads(torch, hidden_size, cfg, "cpu")[method]
            screen.restore(torch, head, key, "replay_pre", archive)
            optimizer = make_optimizer(torch, head, cfg["training"], result["policy"])
            load_optimizer(torch, optimizer, head, archive, key + "__optimizer_pre")
            update(torch, head, method, reopened[(STEPS - 1) % 2], cfg, optimizer)
            for name, value in head.state_dict().items():
                np.testing.assert_array_equal(value.detach().numpy(), archive[f"checkpoint__{key}__final__{name}"])
            moments = {}
            save_optimizer(optimizer, head, moments, key + "__optimizer_final")
            for name, value in moments.items():
                np.testing.assert_array_equal(value, archive[name])
            for i, case in enumerate(reopened, 1):
                outputs = screen.head_forward(head, method, case, "cpu")
                for name, value in zip(("scores", "gate", "residual", "observables", "encoded"), outputs):
                    np.testing.assert_array_equal(value.detach().numpy(), archive[f"{key}__training_{i:03d}__{name}"])
            result["reopened_parameter_moment_signal_replay"] = "exact_pass"
    report = {"schema_version": "rag.semantic_reader_training_qualification.v1", "input_sha256": INPUT_SHA256,
              "sample_rule": "first two saved usable training cases only; alternate, 64 updates per arm",
              "config_sha256": screen.old._sha256(screen.CONFIG), "seed": cfg["training"]["seed"],
              "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
              "source_hashes": {p: screen.old._sha256(ROOT / p) for p in (
                  "applications/rag/semantic_reader_training.py", "applications/rag/quantum_semantic_reader.py",
                  "applications/rag/qarcg_reader.py", "scripts/collab/five_ideas/qualify_semantic_reader_training.py")},
              "values_sha256": screen.old._sha256(output_dir / "qualification_values.npz"), "arms": results,
              "passed": all(v["passed"] for k, v in results.items() if k.endswith(SCALED)),
              "evaluation_values_opened": False, "silver_used": False, "task_improvement_established": False,
              "limitation": "Two-case engineering fit; not all-step replay of the old 434-case training, not a task rerun."}
    report["source_hash_policy"] = "Raw hashes retained; deployment identity uses source bytes with CRLF normalized to LF."
    report["canonical_source_hashes"] = {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                         for p in report["source_hashes"]}
    screen.old._write_json(output_dir / "report.json", report)
    reopened_report = screen.old._load_json(output_dir / "report.json")
    if reopened_report != report:
        raise ValueError("qualification report readback mismatch")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = qualify(args.input, args.output)
    print(json.dumps({"passed": result["passed"], "arms": {k: {"passed": v["passed"], **v["health"]}
                                                               for k, v in result["arms"].items()}}))
    raise SystemExit(0 if result["passed"] else 1)
