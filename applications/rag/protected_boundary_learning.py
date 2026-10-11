"""Opt-in engineering loss/step and protected selection; no production registry.

Consumes explicit per-question pairs, not weak masks. This interface grants no
experiment permission and does not accept oracle/generation features.
"""
from __future__ import annotations
import math


def validate_row(row):
    if row["future_role"] != "training":
        raise ValueError("training objective cannot consume validation")
    baseline = row["baseline_top5_indices"]
    pairs = row["draft_pairs"]; weights = row["pair_weights"]
    immutable = row["immutable_selected_indices"]
    positives = row["eligible_positive_indices"]; negatives = row["replaceable_indices"]
    groups = (baseline, immutable, positives, negatives)
    if len(row["roles"]) != 50:
        raise ValueError("Top50 role coverage required")
    if any(len(g) != len(set(g)) or any(type(i) is not int or not 0 <= i < 50 for i in g) for g in groups):
        raise ValueError("invalid candidate indices")
    if len(baseline) != 5 or set(immutable) | set(negatives) != set(baseline) or set(immutable) & set(negatives):
        raise ValueError("baseline protection partition mismatch")
    if set(positives) & set(baseline) or row["maximum_replacement_slots"] != min(len(positives), len(negatives)):
        raise ValueError("invalid positive/replacement budget")
    if any(len(p) != 2 or any(type(i) is not int for i in p) for p in pairs):
        raise ValueError("invalid pair indices")
    if len(pairs) != len({tuple(p) for p in pairs}) or len(weights) != len(pairs):
        raise ValueError("pair duplicates/weights mismatch")
    if any(len(p) != 2 or p[0] not in positives or p[1] not in negatives or
           row["roles"][p[0]] != "qualified_missed_direct" or row["roles"][p[1]] != "qualified_replaceable_irrelevant" for p in pairs):
        raise ValueError("pair not a qualified direct-to-irrelevant target")
    if pairs and (row["missing_selected_direct_qualification"] or not row["scope_qualified"] or
                  any(not math.isfinite(w) or w <= 0 for w in weights) or abs(sum(weights) - 1) > 1e-8):
        raise ValueError("pending protection/scope or non-normalized weights")
    return row


def boundary_objective(scores, base, row, *, margin=0.0, anchor_weight=0.01, protection_weight=1.0):
    import torch
    validate_row(row)
    if scores.shape != (50,) or base.shape != (50,) or not bool(scores.isfinite().all() & base.isfinite().all()):
        raise ValueError("finite Top50 scores required")
    if any(not math.isfinite(x) or x < 0 for x in (margin, anchor_weight, protection_weight)):
        raise ValueError("nonnegative finite loss settings required")
    if not row["draft_pairs"]:
        zero = scores.sum() * 0
        return {"total": zero, "pair": zero, "anchor": zero, "protection": zero}
    pair = torch.as_tensor(row["draft_pairs"], dtype=torch.long, device=scores.device)
    weights = torch.as_tensor(row["pair_weights"], dtype=scores.dtype, device=scores.device)
    rank = (torch.nn.functional.softplus(scores[pair[:, 1]] - scores[pair[:, 0]] + margin) * weights).sum()
    anchor = (scores - base.detach().to(scores.device)).square().mean()
    kept = row["immutable_selected_indices"]
    outsiders = row["eligible_positive_indices"]
    # Keep all baseline slots not explicitly qualified replaceable, including partial/held.
    protection = torch.relu(scores[outsiders][:, None] - scores[kept][None, :]).square().mean() if kept else scores.sum() * 0
    return {"total": rank + anchor_weight * anchor + protection_weight * protection,
            "pair": rank, "anchor": anchor, "protection": protection}


def protected_selection(scores, row):
    validate_row(row)
    values = scores.detach().cpu().tolist() if hasattr(scores, "detach") else list(scores)
    if len(values) != 50 or any(not math.isfinite(v) for v in values):
        raise ValueError("finite Top50 scores required")
    if not row["draft_pairs"]:
        return list(row["baseline_top5_indices"])
    kept = set(row["immutable_selected_indices"])
    pool = set(row["replaceable_indices"]) | set(row["eligible_positive_indices"])
    chosen = kept | set(sorted(pool, key=lambda i: (-values[i], i))[:5 - len(kept)])
    return sorted(chosen, key=lambda i: (-values[i], i))


def boundary_training_step(head, optimizer, base, pooled, row, *, gradient_clip=1.0, **loss_settings):
    """One engineering update for either matched semantic head; skip no-pair decay."""
    import torch
    validate_row(row)
    if not row["draft_pairs"]:
        return {"updated": False, "reason": "no_qualified_pairs"}
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("invalid gradient clip")
    optimizer.zero_grad(set_to_none=True)
    output = head(base, pooled)
    parts = boundary_objective(output[0], base, row, **loss_settings)
    parts["total"].backward()
    for name, p in head.named_parameters():
        if p.grad is None or not bool(p.grad.isfinite().all()):
            raise ValueError("missing/nonfinite task gradient: " + name)
    gradients = {name: p.grad.detach().cpu().clone() for name, p in head.named_parameters()}
    torch.nn.utils.clip_grad_norm_(head.parameters(), gradient_clip)
    optimizer.step()
    if not all(bool(p.isfinite().all()) for p in head.parameters()):
        raise ValueError("nonfinite updated head")
    return {"updated": True, "loss": {key: float(value.detach()) for key, value in parts.items()},
            "scores_before": output[0].detach().cpu().clone(), "data_gradients_before_clip": gradients}
