"""Train-only, scale-invariant Top5 rescue and retention objective.

This does not score/select passages online. It is an objective candidate, not a
certified support-preserving optimizer or a quantum mechanism. No Silver or
evaluation fields are accepted by this API. Exact/near collapse fails closed.
"""
from __future__ import annotations

import math

from applications.rag.protected_boundary_learning import validate_row

OBJECTIVE_ID = "robust_standardized_top5_membership.v1"


def standardized_scores(scores, *, min_std=1e-8):
    """Differentiable median/IQR scaling; never detach the candidate scale.

No additive epsilon: above the explicit numerical floor, every positive
affine transformation has the same normalized scores and objective. Constant
or numerically collapsed scores are invalid, not a cheap low-loss state.
"""
    import torch
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("positive finite min_std required")
    if scores.shape != (50,) or not scores.is_floating_point() or not bool(scores.isfinite().all()):
        raise ValueError("50 finite floating scores required")
    q25, median, q75 = torch.quantile(scores, scores.new_tensor([.25, .5, .75])).unbind()
    # A single extreme score must not inflate the denominator and make all
    # genuine boundary gaps look small. IQR is robust to that counterexample.
    # 1.349 expresses IQR in approximate standard-deviation units for a normal
    # distribution; this fixed conversion is not estimated from labels.
    scale = (q75 - q25) / 1.349
    if not bool(scale > min_std):
        raise ValueError("collapsed score scale: objective is undefined")
    return (scores - median) / scale


def _membership_terms(z, indices, margin):
    import torch
    terms = []
    for index in indices:
        others = torch.cat((z[:index], z[index + 1:]))
        # Fifth of the OTHER49 is the true entry/retention boundary. Excluding
        # the candidate avoids a self-threshold that can hide membership loss.
        boundary = torch.topk(others, 5).values[-1]
        terms.append(torch.nn.functional.softplus(margin + boundary - z[index]))
    return torch.stack(terms) if terms else z[:0]


def standardized_top5_objective(scores, base, row, *, margin=0.1,
                                anchor_weight=0.01, protection_weight=1.0,
                                min_std=1e-8):
    """Encourage feasible rescue slots and protect actual baseline membership.

Only the best R eligible positives receive rescue pressure, where R is the
frozen maximum replacement capacity. Requiring all eligible positives to enter
five slots would be infeasible. All other49 candidates, including unlabelled
ones, define the entry boundary; no label mask is used for online selection.
The smooth surrogate cannot guarantee retention: actual counts remain a gate.
"""
    import torch
    validate_row(row)
    if base.shape != (50,) or scores.device != base.device or scores.dtype != base.dtype:
        raise ValueError("matching score/base shape, dtype and device required")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("strict positive dimensionless margin required")
    if any(not math.isfinite(v) or v < 0 for v in (anchor_weight, protection_weight)):
        raise ValueError("nonnegative finite weights required")
    z = standardized_scores(scores, min_std=min_std)
    reference = standardized_scores(base.detach(), min_std=min_std)
    zero = z.sum() * 0
    if not row["draft_pairs"]:
        return {"total": zero, "rescue": zero, "anchor": zero, "protection": zero}
    positives = list(row["eligible_positive_indices"])
    capacity = row["maximum_replacement_slots"]
    # Deterministic index tie break; use detach only for choosing a feasible
    # subset, never for score standardization or the membership gradients.
    chosen = sorted(positives, key=lambda i: (-float(z[i].detach()), i))[:capacity]
    rescue = _membership_terms(z, chosen, margin).mean()
    kept = _membership_terms(z, row["immutable_selected_indices"], margin)
    protection = kept.mean() if len(kept) else zero
    anchor = (z - reference).square().mean()
    total = rescue + anchor_weight * anchor + protection_weight * protection
    return {"total": total, "rescue": rescue, "anchor": anchor, "protection": protection}


def membership_diagnostics(scores, row):
    """Train-only counts; protected items are not automatically direct support."""
    validate_row(row)
    standardized_scores(scores)
    selected = sorted(range(50), key=lambda i: (-float(scores[i].detach()), i))[:5]
    eligible = set(row["eligible_positive_indices"])
    kept = set(row["immutable_selected_indices"])
    return {"selected_indices": selected,
            "rescue_capacity": row["maximum_replacement_slots"],
            "rescued_eligible": len(set(selected) & eligible),
            "protected_count": len(kept), "lost_protected": len(kept - set(selected))}
