"""Isolated optimizer repair and train-only learning-health checks.

The head/initialization/loss are unchanged. Historical coupled Adam and unscaled
AdamW remain explicit diagnostic controls. AdamW decays only the projection
matrix; its LR can be fan-in scaled. Null scale, readouts, circuit and biases
receive no decay. No evaluation labels are accepted.
"""
from __future__ import annotations

import numpy as np

LEGACY = "adam_coupled_all_v1"
REPAIRED = "adamw_projection_only_v1"
SCALED = "adamw_projection_fanin_v1"


class TrainingHealthError(ValueError):
    def __init__(self, method, step, health, gradients, reason):
        super().__init__(reason)
        self.record = {"method": method, "step": step, "health": health,
                       "data_gradient_norm_totals": gradients, "reason": reason}


def make_optimizer(torch, head, training, policy=REPAIRED):
    if policy == LEGACY:
        return torch.optim.Adam(head.parameters(), lr=training["learning_rate"],
                                weight_decay=training["weight_decay"])
    if policy not in (REPAIRED, SCALED):
        raise ValueError("unknown optimizer policy")
    named = list(head.named_parameters())
    decay = [(n, p) for n, p in named if n == "project.weight"]
    excluded = [(n, p) for n, p in named if n != "project.weight"]
    # Adam is coordinate-normalized: a coherent update of a wide projection
    # need not scale down with fan-in. A dimension-only correction keeps the
    # 3840->4 tanh projection from saturating; no panel-driven LR search.
    projection_lr = training["learning_rate"]
    if policy == SCALED and decay:
        projection_lr /= float(head.project.in_features) ** 0.5
    groups = [{"params": [p for _, p in group], "parameter_names": [n for n, _ in group],
               "weight_decay": wd, "lr": lr} for group, wd, lr in
              ((decay, training["weight_decay"], projection_lr), (excluded, 0.0, training["learning_rate"])) if group]
    return torch.optim.AdamW(groups, lr=training["learning_rate"])


def norm(tensor):
    return float(tensor.detach().double().norm().cpu())


def gradient_health(head):
    """Data/loss gradients before clipping or optimizer regularization."""
    groups = {"projection": [], "readout": [], "circuit": [], "scale": []}
    for name, p in head.named_parameters():
        if p.grad is None:
            raise ValueError("missing data gradient: " + name)
        if not bool(p.grad.isfinite().all()):
            raise ValueError("non-finite data gradient: " + name)
        role = ("projection" if name.startswith("project.") else "circuit" if name.endswith("circuit")
                else "scale" if name.endswith("residual_scale") else "readout")
        groups[role].append(norm(p.grad) ** 2)
    return {key: float(np.sqrt(sum(values))) for key, values in groups.items()}


def signal_health(head, outputs, bases, initial_projection_norm=None):
    if not all(bool(p.isfinite().all()) for p in head.parameters()):
        raise ValueError("non-finite trained parameter")
    corrections, spreads, saturation, base_spreads = [], [], [], []
    for output, base in zip(outputs, bases):
        if not all(bool(v.isfinite().all()) for v in output):
            raise ValueError("non-finite head signal")
        # Measure the actual float32 addition, not a sub-ULP residual alone.
        delta = (output[0] - base.to(output[0].device)).detach().double().cpu().numpy()
        corrections.append(delta)
        spreads.append(float(output[4].detach().double().std(0, unbiased=False).mean().cpu()))
        saturation.append(float((output[4].detach().abs() > 0.99).float().mean().cpu()))
        base_spreads.append(max(float(base.double().std(unbiased=False)), 1e-12))
    projection_norm = norm(head.project.weight) if hasattr(head, "project") else None
    return {"projection_norm": projection_norm,
            "projection_norm_ratio": projection_norm / initial_projection_norm if initial_projection_norm else None,
            "mean_encoding_std": float(np.mean(spreads)),
            "encoding_saturation_fraction": float(np.mean(saturation)),
            "effective_correction_count": int(sum(np.count_nonzero(x) for x in corrections)),
            "max_within_case_correction_std": max(float(x.std()) for x in corrections),
            "max_normalized_correction_std": max(float(x.std()) / s for x, s in zip(corrections, base_spreads)),
            "max_abs_effective_correction": max(float(np.abs(x).max()) for x in corrections)}


def require_learning(health, gradients, *, semantic=True):
    """Collapse guard, not a utility/selection-improvement gate."""
    if health["mean_encoding_std"] <= 1e-6:
        raise ValueError("learning-health gate: collapsed encoding")
    if not health["effective_correction_count"] or health["max_normalized_correction_std"] <= 1e-6:
        raise ValueError("learning-health gate: no input-dependent float32 corrections")
    if gradients["readout"] <= 0 or gradients["scale"] <= 0:
        raise ValueError("learning-health gate: blocked readout/scale data gradients")
    if semantic and (health["projection_norm_ratio"] <= 1e-6 or gradients["projection"] <= 0):
        raise ValueError("learning-health gate: collapsed/blocked projection")


def save_optimizer(optimizer, head, arrays, prefix):
    """Detached moments/steps by parameter name; no pickle or training graph."""
    for name, parameter in head.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if key not in ("step", "exp_avg", "exp_avg_sq"):
                raise ValueError("unexpected optimizer state")
            arrays[f"{prefix}__{name}__{key}"] = value.detach().cpu().numpy().copy()


def load_optimizer(torch, optimizer, head, arrays, prefix):
    optimizer.state.clear()
    for name, parameter in head.named_parameters():
        keys = [f"{prefix}__{name}__{k}" for k in ("step", "exp_avg", "exp_avg_sq")]
        present = [k in arrays for k in keys]
        if not any(present):
            continue  # first update: empty Adam state is intentional
        if not all(present):
            raise ValueError("incomplete optimizer state: " + name)
        state = {}
        for key, path in zip(("step", "exp_avg", "exp_avg_sq"), keys):
            value = np.array(arrays[path], copy=True)
            if not np.isfinite(value).all() or (key != "step" and value.shape != tuple(parameter.shape)):
                raise ValueError("invalid optimizer state: " + name)
            state[key] = torch.as_tensor(value, device="cpu" if key == "step" else parameter.device).clone()
        optimizer.state[parameter] = state


def validate_optimizer_witnesses(torch, cfg, heads, arrays, device):
    from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
    from applications.rag.qarcg_reader import torch_pairwise_rank_loss
    saved = {m: {k: v.detach().clone() for k, v in h.state_dict().items()} for m, h in heads.items()}
    count = 0
    try:
        for method, head in heads.items():
            for epoch in range(1, cfg["training"]["epochs"] + 1):
                for sample in (1, 2):
                    label = f"witness{epoch}_{sample}"
                    if f"checkpoint__{method}__{label}__" + next(iter(head.state_dict())) not in arrays:
                        continue  # synthetic fixture declares a single training case
                    screen.restore(torch, head, method, label, arrays)
                    optimizer = make_optimizer(torch, head, cfg["training"], cfg["training"]["optimizer_policy"])
                    load_optimizer(torch, optimizer, head, arrays, f"optimizer__{method}__{label}")
                    case = {key: torch.as_tensor(np.array(arrays[f"training_{sample:03d}__{key}"], copy=True))
                            for key in ("base", "pooled", "scalar", "weak_mask")}
                    optimizer.zero_grad(set_to_none=True)
                    scores = screen.head_forward(head, method, case, device)[0]
                    loss = (torch_pairwise_rank_loss(scores, case["weak_mask"].to(device)) +
                            cfg["training"]["anchor_penalty"] * (scores - case["base"].to(device)).square().mean())
                    loss.backward()
                    gradient_health(head)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["training"]["gradient_clip_norm"])
                    optimizer.step()
                    for name, value in head.state_dict().items():
                        torch.testing.assert_close(value, torch.as_tensor(
                            np.array(arrays[f"checkpoint__{method}__{label}_post__{name}"], copy=True), device=value.device),
                            rtol=1e-5, atol=1e-7)
                    moments = {}
                    save_optimizer(optimizer, head, moments, f"optimizer__{method}__{label}_post")
                    for name, value in moments.items():
                        np.testing.assert_allclose(value, arrays[name], rtol=1e-5, atol=1e-7)
                    count += 1
    finally:
        for method, head in heads.items():
            head.load_state_dict(saved[method])
    if not count:
        raise ValueError("no optimizer witnesses replayed")
    return {"optimizer_witness_replay": "pass"}
