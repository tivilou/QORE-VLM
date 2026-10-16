"""Isolated cached-Reader training with native AdamW update witnesses.

Only objective/health functions receive training targets. Forward and stable
selection receive fixed numeric features and ranks, never reference labels.
"""
from __future__ import annotations
import copy
import hashlib
import numpy as np
import torch
from applications.rag.quantum_reader_rotation import RotationConfig, make_rotation_head, top5_retention_objective
from applications.rag.scale_invariant_top5_objective import standardized_top5_objective
from applications.rag.protected_boundary_learning import validate_row

ARMS = {
    "quantum_membership": ("quantum", "membership"),
    "classical_membership": ("classical", "membership"),
    "quantum_retention_control": ("quantum", "retention"),
    "classical_retention_control": ("classical", "retention"),
}
METHODS = ("frozen_reader_topk", *ARMS)
STAGES = ("data", "preprocess", "training", "scoring", "selection", "diagnosis")


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save_value(arrays, key, value):
    if key in arrays: raise ValueError("duplicate numeric producer: " + key)
    a = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if a.dtype.kind not in "fiu" or not np.isfinite(a).all(): raise ValueError("invalid numeric value")
    arrays[key] = a.copy()
    return key


def describe_values(arrays):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": array_hash(v),
        "producer": k.split("__", 1)[0], "complete_coordinates": True} for k, v in arrays.items()}


def top5(scores, ranks):
    values = scores.detach().cpu().tolist() if isinstance(scores, torch.Tensor) else list(scores)
    if len(values) != 50 or not np.isfinite(values).all() or sorted(ranks) != list(range(1, 51)):
        raise ValueError("finite full50 scores and unique frozen ranks required")
    return sorted(range(50), key=lambda i: (-float(values[i]), ranks[i]))[:5]


def make_heads(weight, bias, cfg, device="cpu"):
    rc = RotationConfig(**cfg["rotation_head"])
    return {m: make_rotation_head(weight, bias, family=family, config=rc).to(device)
            for m, (family, _) in ARMS.items()}


def forward(head, feature):
    device = head.theta.device
    return head(feature["cls"].to(device), feature["pooled"].to(device))


def objective(method, scores, base, target, cfg):
    loss = ARMS[method][1]
    fn = standardized_top5_objective if loss == "membership" else top5_retention_objective
    return fn(scores, base.to(scores), target, **cfg["losses"][loss])


def make_optimizer(head, cfg):
    c = cfg["training"]
    return torch.optim.AdamW(head.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"],
        betas=tuple(c["betas"]), eps=c["eps"], foreach=False)


def capture_optimizer(opt, prefix, arrays):
    state = opt.state_dict()
    result = {"param_groups": copy.deepcopy(state["param_groups"]), "state": {}}
    for pid, fields in state["state"].items():
        result["state"][str(pid)] = {k: save_value(arrays, prefix + f"__p{pid}__{k}", v) for k, v in fields.items()}
    return result


def restore_optimizer(opt, record, arrays):
    state = {int(pid): {k: torch.from_numpy(np.asarray(arrays[ref]).copy()) for k, ref in fields.items()}
             for pid, fields in record["state"].items()}
    opt.load_state_dict({"param_groups": copy.deepcopy(record["param_groups"]), "state": state})


def capture_output(head, feature, prefix, arrays):
    with torch.no_grad(): values = forward(head, feature)
    return {k: save_value(arrays, prefix + "__" + k, v) for k, v in values.items()}


def health(head, features, targets, records):
    rescued, lost, correction_spread, minimum_iqr = 0, 0, [], float("inf")
    with torch.no_grad():
        for feature, target, record in zip(features, targets, records):
            scores = forward(head, feature)["scores"].cpu()
            selected = set(top5(scores, record["candidate_retrieved_ranks"]))
            rescued += len(selected & set(target["eligible_positive_indices"]))
            lost += len(set(target["immutable_selected_indices"]) - selected)
            correction_spread.append(float((scores - feature["base"]).std(unbiased=False)))
            minimum_iqr = min(minimum_iqr, float(torch.quantile(scores, .75) - torch.quantile(scores, .25)))
    return {"rescued_eligible": rescued, "lost_protected": lost,
            "mean_correction_std": float(np.mean(correction_spread)), "minimum_score_iqr": minimum_iqr}


def train_arm(method, head, features, targets, records, cfg, arrays):
    if not (len(features) == len(targets) == len(records)) or not features:
        raise ValueError("training coverage mismatch")
    for target in targets: validate_row(target)
    opt = make_optimizer(head, cfg)
    initial = {k: save_value(arrays, method + "__initial__" + k, v) for k, v in head.state_dict().items()}
    reports, witnesses = [], []
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        losses, norms = [], []
        for index, (feature, target, record) in enumerate(zip(features, targets, records)):
            prefix = f"{method}__e{epoch}__s{index+1}"
            witness = index < 2
            opt.zero_grad(set_to_none=True)
            if witness:
                before = save_value(arrays, prefix + "__theta_before", head.theta)
                optimizer_before = capture_optimizer(opt, prefix + "__optimizer_before", arrays)
                pre = capture_output(head, feature, prefix + "__pre", arrays)
            values = forward(head, feature)
            parts = objective(method, values["scores"], feature["base"], target, cfg)
            parts["total"].backward()
            if head.theta.grad is None or not bool(head.theta.grad.isfinite().all()):
                raise ValueError("missing/nonfinite data gradient")
            if witness: raw_grad = save_value(arrays, prefix + "__raw_gradient", head.theta.grad)
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["training"]["gradient_clip_norm"])
            if not bool(grad_norm.isfinite()): raise ValueError("nonfinite clipping norm")
            if witness: clipped = save_value(arrays, prefix + "__clipped_gradient", head.theta.grad)
            opt.step()
            if not bool(head.theta.isfinite().all()): raise ValueError("nonfinite update")
            losses.append({k: float(v.detach()) for k, v in parts.items()}); norms.append(float(grad_norm))
            if witness:
                witnesses.append({"method": method, "epoch": epoch, "case_index": index,
                    "question_sha256": record["question_sha256"], "prefix": prefix,
                    "theta_before": before, "theta_after": save_value(arrays, prefix + "__theta_after", head.theta),
                    "raw_gradient": raw_grad, "clipped_gradient": clipped,
                    "optimizer_before": optimizer_before, "optimizer_after": capture_optimizer(opt, prefix + "__optimizer_after", arrays),
                    "before": pre, "after": capture_output(head, feature, prefix + "__post", arrays),
                    "loss": losses[-1], "preclip_gradient_norm": float(grad_norm)})
        h = health(head, features, targets, records)
        reports.append({"epoch": epoch, "updates": len(features),
            "mean_loss": {k: float(np.mean([r[k] for r in losses])) for k in losses[0]},
            "mean_preclip_gradient_norm": float(np.mean(norms)), "membership_health": h})
        print(f"{method}: epoch {epoch}/{cfg['training']['epochs']}; rescue={h['rescued_eligible']}; lost_protected={h['lost_protected']}", flush=True)
    final = {k: save_value(arrays, method + "__final__" + k, v) for k, v in head.state_dict().items()}
    final_opt = capture_optimizer(opt, method + "__final_optimizer", arrays)
    return {"epochs": reports, "initial_state": initial, "final_state": final,
            "final_optimizer": final_opt, "resources": head.resources()}, witnesses


def replay_update(witness, initial_state, feature, target, cfg, arrays, weight, bias):
    """Recompute data gradient AND native AdamW step; never substitute saved gradients."""
    method = witness["method"]
    head = make_heads(weight, bias, cfg)[method]
    head.load_state_dict({k: torch.from_numpy(np.asarray(arrays[v]).copy()) for k, v in initial_state.items()})
    with torch.no_grad(): head.theta.copy_(torch.from_numpy(np.asarray(arrays[witness["theta_before"]]).copy()))
    opt = make_optimizer(head, cfg)
    restore_optimizer(opt, witness["optimizer_before"], arrays)
    for stage in ("before", "after"):
        if stage == "after":
            opt.zero_grad(set_to_none=True)
            parts = objective(method, forward(head, feature)["scores"], feature["base"], target, cfg)
            for k, value in parts.items(): np.testing.assert_allclose(float(value.detach()), witness["loss"][k], rtol=2e-5, atol=3e-5)
            parts["total"].backward()
            np.testing.assert_allclose(head.theta.grad.numpy(), arrays[witness["raw_gradient"]], rtol=2e-4, atol=3e-5)
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["training"]["gradient_clip_norm"])
            np.testing.assert_allclose(float(norm), witness["preclip_gradient_norm"], rtol=2e-4, atol=3e-5)
            np.testing.assert_allclose(head.theta.grad.numpy(), arrays[witness["clipped_gradient"]], rtol=2e-4, atol=3e-5)
            opt.step()
            np.testing.assert_allclose(head.theta.detach().numpy(), arrays[witness["theta_after"]], rtol=2e-4, atol=3e-5)
        with torch.no_grad(): output = forward(head, feature)
        for k, ref in witness[stage].items():
            np.testing.assert_allclose(output[k].numpy(), arrays[ref], rtol=2e-4, atol=3e-5)
    state = opt.state_dict()
    if state["param_groups"] != witness["optimizer_after"]["param_groups"]:
        raise ValueError("optimizer config replay mismatch")
    if set(map(str, state["state"])) != set(witness["optimizer_after"]["state"]):
        raise ValueError("optimizer state coverage mismatch")
    for pid, fields in witness["optimizer_after"]["state"].items():
        if set(fields) != set(state["state"][int(pid)]): raise ValueError("optimizer slot mismatch")
        for k, ref in fields.items():
            np.testing.assert_allclose(state["state"][int(pid)][k].cpu().numpy(), arrays[ref], rtol=2e-4, atol=3e-5)
