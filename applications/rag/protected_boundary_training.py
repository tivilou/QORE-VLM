"""Matched head training and complete detached update replay for frozen support pairs."""
from __future__ import annotations
import numpy as np

from applications.rag.protected_boundary_learning import boundary_training_step
from applications.rag.quantum_semantic_reader import make_semantic_head, semantic_pool
from applications.rag.qarcg_reader import QARCGConfig
from applications.rag.semantic_reader_training import (
    make_optimizer, save_optimizer, load_optimizer, norm, signal_health, require_learning,
)

ARMS = ("quantum_semantic", "classical_semantic")
FIELDS = ("scores", "gate", "residual", "observables", "encoded")


def make_heads(torch, hidden_size, cfg, device):
    heads = {m: make_semantic_head(hidden_size, classical=m == "classical_semantic",
                                 seed=cfg["training"]["seed"], config=QARCGConfig(**cfg["head"])).to(device) for m in ARMS}
    a, b = (heads[m] for m in ARMS)
    torch.testing.assert_close(a.project.weight, b.project.weight, rtol=0, atol=0)
    if sum(p.numel() for p in a.parameters()) != sum(p.numel() for p in b.parameters()):
        raise ValueError("matched head parameter budget differs")
    return heads


def checkpoint(head, arrays, prefix):
    for name, value in head.state_dict().items():
        arrays[prefix + "__" + name] = value.detach().cpu().numpy().copy()


def restore(torch, head, arrays, prefix):
    head.load_state_dict({name: torch.as_tensor(np.array(arrays[prefix + "__" + name], copy=True), device=value.device)
                         for name, value in head.state_dict().items()})


def capture_output(head, base, pooled, arrays, prefix):
    import torch
    with torch.no_grad():
        output = head(base, pooled)
    for key, value in zip(FIELDS, output):
        arrays[prefix + "__" + key] = value.detach().cpu().numpy().copy()
    return output


def train_heads(torch, cfg, heads, features, targets, arrays, device, *, health_checks=True):
    reports, witnesses = {}, []
    for method, head in heads.items():
        optimizer = make_optimizer(torch, head, cfg["training"], cfg["training"]["optimizer_policy"])
        initial_norm = norm(head.project.weight)
        checkpoint(head, arrays, method + "__initial")
        epoch_reports = []
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            total = {k: 0.0 for k in ("total", "pair", "anchor", "protection")}
            per_question_loss = []
            gradients = {k: 0.0 for k in ("projection", "readout", "circuit", "scale")}
            for i, (feature, target) in enumerate(zip(features, targets)):
                base = feature["base"].to(device); pooled = feature["pooled"].to(device)
                label = f"{method}__epoch{epoch}__sample{i + 1}"
                if i < 2:
                    checkpoint(head, arrays, label + "__before")
                    save_optimizer(optimizer, head, arrays, label + "__optimizer_before")
                    capture_output(head, base, pooled, arrays, label + "__pre")
                result = boundary_training_step(head, optimizer, base, pooled, target,
                    gradient_clip=cfg["training"]["gradient_clip_norm"], **cfg["training"]["loss"])
                if not result["updated"]:
                    raise ValueError("frozen training row unexpectedly has no pair")
                for key, loss in result["loss"].items(): total[key] += loss
                per_question_loss.append({"sample_index": i, "loss": result["loss"]})
                for name, grad in result["data_gradients_before_clip"].items():
                    role = "projection" if name.startswith("project.") else "circuit" if name.endswith("circuit") else "scale" if name.endswith("residual_scale") else "readout"
                    gradients[role] += norm(grad)
                    if i < 2: arrays[label + "__gradient__" + name] = grad.numpy().copy()
                if i < 2:
                    checkpoint(head, arrays, label + "__after")
                    save_optimizer(optimizer, head, arrays, label + "__optimizer_after")
                    capture_output(head, base, pooled, arrays, label + "__post")
                    witnesses.append({"method": method, "epoch": epoch, "sample_index": i, "prefix": label, "loss": result["loss"]})
            checkpoint(head, arrays, method + "__epoch" + str(epoch))
            probes = features[:2]
            outputs = [head(f["base"].to(device), f["pooled"].to(device)) for f in probes]
            health = signal_health(head, outputs, [f["base"] for f in probes], initial_norm)
            if health_checks: require_learning(health, gradients)
            epoch_reports.append({"epoch": epoch, "updates": len(features), "mean_loss": {k: v / len(features) for k, v in total.items()},
                                  "gradient_norm_sums": gradients, "health": health, "per_question_loss": per_question_loss})
            print(f"{method}: epoch {epoch}/{cfg['training']['epochs']} complete", flush=True)
        reports[method] = {"parameter_count": sum(p.numel() for p in head.parameters()), "epochs": epoch_reports}
    return reports, witnesses


def verify_values(torch, cfg, heads, arrays, metadata, witnesses, targets, device):
    """Recompute real hidden pooling, every online arm and all captured AdamW updates."""
    state = {m: {k: v.detach().clone() for k, v in h.state_dict().items()} for m, h in heads.items()}
    try:
        for lane in ("training", "evaluation"):
            rows = [r for r in metadata if r["lane"] == lane]
            for i, row in enumerate(rows):
                prefix = row["prefix"]
                base, pooled = (torch.as_tensor(np.array(arrays[prefix + "__" + k], copy=True)) for k in ("base", "pooled"))
                if base.shape != (50,) or pooled.shape != (50, 5 * heads[ARMS[0]].hidden_size):
                    raise ValueError("full coordinate coverage mismatch")
                if (prefix + "__hidden" in arrays) != (i < 2):
                    raise ValueError("predeclared full hidden sample coverage mismatch")
                if i < 2:
                    hidden = torch.as_tensor(np.array(arrays[prefix + "__hidden"], copy=True))[None]
                    qm, pm = (torch.as_tensor(np.array(arrays[prefix + "__" + k], copy=True))[None] for k in ("question_mask", "passage_mask"))
                    actual, weights = semantic_pool(hidden, qm, pm)
                    torch.testing.assert_close(actual[0], pooled[0], rtol=1e-4, atol=1e-4)
                    torch.testing.assert_close(weights[0], torch.as_tensor(np.array(arrays[prefix + "__pooling_weights"], copy=True)), rtol=1e-4, atol=1e-4)
                    weight, bias = (torch.as_tensor(np.array(arrays[prefix + "__" + k], copy=True)) for k in ("relevance_weight", "relevance_bias"))
                    relevance = torch.nn.functional.linear(hidden[0, 0], weight, bias).reshape(())
                    torch.testing.assert_close(relevance, base[0], rtol=1e-4, atol=1e-4)
                for m, head in heads.items():
                    restore(torch, head, arrays, m + "__initial")
                    torch.testing.assert_close(head(base.to(device), pooled.to(device))[0].cpu(), base, rtol=0, atol=0)
                    restore(torch, head, arrays, m + "__epoch" + str(cfg["training"]["epochs"]))
                    output = head(base.to(device), pooled.to(device))
                    for key, value in zip(FIELDS, output):
                        np.testing.assert_allclose(value.detach().cpu().numpy(), arrays[prefix + "__" + m + "__" + key], rtol=1e-4, atol=1e-4)
                    limit = cfg["head"]["residual_bound"] * max(float(base.std(unbiased=False)), 1e-6)
                    if float(output[2].abs().max()) > limit + 1e-5:
                        raise ValueError("residual amplitude violation")
        training = [r for r in metadata if r["lane"] == "training"]
        expected = len(heads) * cfg["training"]["epochs"] * min(2, len(training))
        if len(witnesses) != expected:
            raise ValueError("optimizer witness coverage mismatch")
        for w in witnesses:
            head = heads[w["method"]]; prefix = w["prefix"]
            restore(torch, head, arrays, prefix + "__before")
            optimizer = make_optimizer(torch, head, cfg["training"], cfg["training"]["optimizer_policy"])
            load_optimizer(torch, optimizer, head, arrays, prefix + "__optimizer_before")
            fprefix = training[w["sample_index"]]["prefix"]
            base, pooled = (torch.as_tensor(np.array(arrays[fprefix + "__" + k], copy=True), device=device) for k in ("base", "pooled"))
            pre = head(base, pooled)
            for key, value in zip(FIELDS, pre):
                np.testing.assert_allclose(value.detach().cpu().numpy(), arrays[prefix + "__pre__" + key], rtol=1e-4, atol=1e-4)
            result = boundary_training_step(head, optimizer, base, pooled, targets[w["sample_index"]],
                gradient_clip=cfg["training"]["gradient_clip_norm"], **cfg["training"]["loss"])
            for name, grad in result["data_gradients_before_clip"].items():
                np.testing.assert_allclose(grad.numpy(), arrays[prefix + "__gradient__" + name], rtol=2e-4, atol=1e-6)
            for name, value in head.state_dict().items():
                np.testing.assert_allclose(value.detach().cpu().numpy(), arrays[prefix + "__after__" + name], rtol=2e-4, atol=1e-6)
            moments = {}; save_optimizer(optimizer, head, moments, prefix + "__optimizer_after")
            for key, value in moments.items(): np.testing.assert_allclose(value, arrays[key], rtol=2e-4, atol=1e-6)
            for name, loss in result["loss"].items():
                np.testing.assert_allclose(loss, w["loss"][name], rtol=1e-4, atol=1e-5)
            post = head(base, pooled)
            for key, value in zip(FIELDS, post):
                np.testing.assert_allclose(value.detach().cpu().numpy(), arrays[prefix + "__post__" + key], rtol=2e-4, atol=1e-5)
    finally:
        for m, head in heads.items(): head.load_state_dict(state[m])
    return {"status": "passed", "hidden_pooling_replay": "passed", "reader_cls_relevance_replay": "passed", "initial_null": "passed",
            "all_sample_head_replays": len(metadata) * len(heads), "optimizer_update_replays": len(witnesses)}
